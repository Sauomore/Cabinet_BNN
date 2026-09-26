# -*- coding: utf-8 -*-
"""
Cabinet-BNN MVP 实验一：HSH-64 码质量 + 桶分布验证。

目的（对应路线文档 M1 + M3.5）：
  M1   在 3109 词上复现论文的纯 Hamming Recall@10（论文基准：h256=0.7382，h512=0.7404）
  M3.5 测量 (feat, sim) 桶大小分布，验证 abs 的准完美哈希是否可行（最大桶须 ≤ 256）

实验设计说明（重要）：
    本机 bge-small-zh-v1.5 权重缺失，只有 bge-large-zh-v1.5。
    因此"教师"用 bge-large（与论文一致），"学生输入"用 bge-large 的
    PCA 截断到 512 维来模拟低维学生输入。
    —— 这不是论文的严格配置，会在结果里明确标注。仅用于验证【码的几何与桶结构】。
    等 bge-small 权重到位后，把 --student-dim 参数改回真实 512 维学生嵌入即可。

用法：
    python scripts/02_mvp_hsh64.py --hidden-dim 512 --epochs 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

# Windows 控制台默认 GBK，强制 UTF-8 以免中文/符号输出报错
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cabinet_bnn.paths import results_dir

from cabinet_bnn.data.embedding_cache import read_cache, normalize_rows
from cabinet_bnn.hsh import codec
from cabinet_bnn.hsh.deep_hash import (
    bits_to_u64,
    export_deep_hash,
    project,
    quantize_bits,
    train_deep_hash,
)
from cabinet_bnn.hsh.metrics import evaluate_codes, recall_curve_paper
from cabinet_bnn.hsh.pos_map import assign_feat, feat_histogram_text
from cabinet_bnn.hsh.post_optimize import (
    export_override_cache,
    greedy_flip_optimize,
    hamming_matrix,
    recall_at_k_from_ranking,
)


def make_student_input(x: np.ndarray, dim: int, seed: int = 0) -> np.ndarray:
    """把教师嵌入降维成学生输入（模拟低维学生编码器）。

    用 PCA 截断，保留最多的语义信息；再重新归一化。
    dim <= 0 或 >= 原维度时不做降维。
    """
    if dim <= 0 or dim >= x.shape[1]:
        return x
    xc = x - x.mean(axis=0, keepdims=True)
    # 用 SVD 求主成分（N=3109 时很快）
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    proj = xc @ vt[:dim].T
    return normalize_rows(proj)


def main() -> int:
    ap = argparse.ArgumentParser(description="Cabinet-BNN MVP：HSH-64 码质量与桶分布验证")
    ap.add_argument("--teacher-cache", type=Path,
                    default=results_dir() / "bge_large_3109.cache")
    ap.add_argument("--student-dim", type=int, default=512,
                    help="学生输入维度（PCA 截断；0 表示与教师同维）")
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--n-bits", type=int, default=52)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feat-mode", choices=["noun", "jieba", "hash"], default="jieba",
                    help="feat 段分配方式（noun=复现论文 benchmark 的硬编码行为）")
    ap.add_argument("--post-iters", type=int, default=10,
                    help="贪心后处理迭代数（0 表示跳过）")
    ap.add_argument("--neg-weight", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", type=Path, default=results_dir())
    ap.add_argument("--tag", default="mvp1")
    args = ap.parse_args()

    t_start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"config": {k: str(v) for k, v in vars(args).items()}}

    # ---------------- 1. 载入教师嵌入 ----------------
    print("=" * 74)
    print("Cabinet-BNN MVP · HSH-64 码质量与桶分布验证")
    print("=" * 74)
    dim_t, words, teacher = read_cache(args.teacher_cache)
    n = len(words)
    print(f"[数据] {n} 个词，教师维度 {dim_t}  来源: {args.teacher_cache.name}")

    # ---------------- 2. 构造学生输入 ----------------
    student = make_student_input(teacher, args.student_dim, args.seed)
    dim_s = student.shape[1]
    print(f"[数据] 学生输入维度 {dim_s} "
          f"({'PCA 截断模拟' if dim_s != dim_t else '与教师同维'})")
    report["dims"] = {"teacher": dim_t, "student": dim_s, "n_items": n}

    # 教师余弦相似度矩阵（评测真值）
    tn = normalize_rows(teacher)
    sim_teacher = tn @ tn.T

    # ---------------- 3. feat 分配 ----------------
    feats, feat_stats = assign_feat(words, args.feat_mode)
    print(f"\n[feat] 模式 = {args.feat_mode}")
    if args.feat_mode != "noun":
        print(feat_histogram_text(feat_stats, n))
    else:
        print(f"    全部为 NOUN(0x0) —— 复现论文 benchmark 的硬编码行为")
    report["feat"] = {"mode": args.feat_mode, "stats": {str(k): v for k, v in feat_stats.items()}}

    # ---------------- 4. 训练 Deep Hash ----------------
    print(f"\n[训练] Deep Hash MLP: {dim_s} -> {args.hidden_dim} -> {args.n_bits}"
          f"  epochs={args.epochs}  device={args.device}")
    t0 = time.time()
    mean, model, hist = train_deep_hash(
        student, n_bits=args.n_bits, hidden_dim=args.hidden_dim,
        epochs=args.epochs, lr=args.lr, pos_k=args.pos_k,
        teacher_x=teacher, seed=args.seed, device=args.device, log_every=100,
    )
    train_sec = time.time() - t0
    print(f"[训练] 完成，用时 {train_sec:.1f}s")
    report["train_seconds"] = round(train_sec, 2)
    report["train_history_tail"] = {k: v[-1] for k, v in asdict(hist).items()}

    # ---------------- 5. 导出模型 ----------------
    model_path = args.out_dir / f"deep_hash_{args.tag}_h{args.hidden_dim}_s{args.seed}.bin"
    export_deep_hash(model_path, mean, model)
    print(f"[导出] {model_path.name} ({model_path.stat().st_size:,} bytes)")
    report["model_path"] = str(model_path)
    report["model_bytes"] = model_path.stat().st_size

    # ---------------- 6. 量化 + 打包 64 位码 ----------------
    u = project(model, mean, student, device=args.device)
    sim_bits = quantize_bits(u)                     # (N, 52)
    sim_vals = bits_to_u64(sim_bits)                # (N,) u64
    feat_arr = np.array(feats, dtype=np.uint8)

    # ---------------- 7. 评测（后处理前）----------------
    print("\n" + "-" * 74)
    print("【后处理前】")
    print("-" * 74)

    # 先按 (feat, sim) 分桶，再对每桶搜 abs 种子
    abs_search = assign_abs(words, feat_arr, sim_vals)
    codes64 = codec.pack_codes(feat_arr, sim_vals, np.array(abs_search["abs"], dtype=np.uint8))

    rep_pre = evaluate_codes(codes64, feat_arr, sim_vals, sim_bits, sim_teacher)
    print(rep_pre.pretty())
    report["pre_post"] = rep_pre.to_dict()
    print_abs_report(abs_search)

    # ---------------- 8. 贪心后处理 ----------------
    report["post"] = None
    bits_opt = sim_bits
    if args.post_iters > 0:
        print("\n" + "-" * 74)
        print(f"【贪心比特翻转后处理】max_iters={args.post_iters} neg_weight={args.neg_weight}")
        print("-" * 74)
        pos_idx = np.argsort(-sim_teacher, axis=1)[:, 1 : args.pos_k + 1]
        rec0 = recall_at_k_from_ranking(pos_idx, hamming_matrix(sim_bits), 10)
        print(f"  [init] Recall@10 = {rec0:.4f}")

        t0 = time.time()
        bits_opt = greedy_flip_optimize(
            sim_bits, sim_teacher, pos_k=args.pos_k, k=10,
            neg_weight=args.neg_weight, max_iters=args.post_iters, verbose=True,
        )
        post_sec = time.time() - t0
        print(f"  [后处理] 用时 {post_sec:.1f}s")
        report["post_seconds"] = round(post_sec, 2)

        opt_path = args.out_dir / f"sim_override_{args.tag}_h{args.hidden_dim}_s{args.seed}.bin"
        export_override_cache(opt_path, words, bits_opt)
        print(f"  [导出] {opt_path.name} ({opt_path.stat().st_size:,} bytes)")
        report["override_path"] = str(opt_path)

        sim_vals_opt = bits_to_u64(bits_opt)
        codes64_opt = codec.pack_codes(feat_arr, sim_vals_opt,
                                       np.array(abs_search["abs"], dtype=np.uint8))
        rep_post = evaluate_codes(codes64_opt, feat_arr, sim_vals_opt, bits_opt, sim_teacher)
        print("\n【后处理后】")
        print(rep_post.pretty())
        report["post"] = rep_post.to_dict()
        report["post_delta_recall10"] = rep_post.recall[10] - rep_pre.recall[10]

    # ---------------- 9. 汇总 ----------------
    report["total_seconds"] = round(time.time() - t_start, 2)

    # 两种 Recall 口径都记录，避免误读
    rc_pre = recall_curve_paper(sim_bits, sim_teacher)
    rc_post = recall_curve_paper(bits_opt, sim_teacher)
    report["recall_denominator_Pi"] = {
        "note": "分母为 |P_i|（参考实现口径），随 K 单调不减",
        "pre_post": {str(k): v for k, v in rc_pre.items()},
        "post": {str(k): v for k, v in rc_post.items()},
    }
    report["recall_denominator_K"] = {
        "note": "分母为 K（论文式 35 口径）；多召回的无关项会拉低比值，故不保证随 K 单调",
        "pre_post": {str(k): v for k, v in rep_pre.recall.items()},
        "post": {str(k): v for k, v in (report["post"]["recall"].items() if report["post"] else [])},
    }
    if report["post"]:
        report["post_delta_recall10_Pi"] = rc_post[10] - rc_pre[10]

    out_json = args.out_dir / f"mvp_report_{args.tag}.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print("结论")
    print("=" * 74)
    print("  Recall@K（分母 |P_i|，参考实现口径）:")
    print(f"    后处理前   K=1 {rc_pre[1]:.4f}   K=5 {rc_pre[5]:.4f}   "
          f"K=10 {rc_pre[10]:.4f}   K=20 {rc_pre[20]:.4f}")
    print(f"    后处理后   K=1 {rc_post[1]:.4f}   K=5 {rc_post[5]:.4f}   "
          f"K=10 {rc_post[10]:.4f}   K=20 {rc_post[20]:.4f}")
    print("  Recall@K（分母 K，论文式 35 口径）:")
    print(f"    后处理前   K=10 {rep_pre.recall[10]:.4f}")
    if report["post"]:
        print(f"    后处理后   K=10 {report['post']['recall']['10']:.4f}")
    print("\n  论文基准 Recall@10 = 0.7382 (h256) / 0.7404 (h512)，含后处理")
    print(f"  本模型 Recall@10   = {rc_post[10]:.4f}")
    print(f"  准完美哈希可行 = {rep_pre.buckets['perfect_hash_possible']}  "
          f"(最大桶 {rep_pre.buckets['max_bucket']} / 上限 256)")
    print(f"  报告已写入 {out_json}")
    return 0


def assign_abs(words: list[str], feats: np.ndarray, sim_vals: np.ndarray) -> dict:
    """按 (feat, sim) 分桶，对每桶搜索准完美哈希种子，返回每词的 abs。"""
    n = len(words)
    # 用 (feat, sim) 组合键分桶 —— 注意桶地址是 (feat, sim)，不是 (feat, abs)
    keys = [f"{int(f)}:{int(s)}" for f, s in zip(feats, sim_vals)]

    buckets: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        buckets.setdefault(k, []).append(i)

    abs_all = np.zeros(n, dtype=np.uint8)
    seeds: dict[str, int] = {}
    perfect = 0
    imperfect = 0
    max_collision = 0

    for k, idxs in buckets.items():
        ws = [words[i] for i in idxs]
        res = codec.search_seed(ws)
        seeds[k] = res.seed
        for pos, i in enumerate(idxs):
            abs_all[i] = res.abs_values[pos]
        if res.perfect:
            perfect += 1
        else:
            imperfect += 1
            max_collision = max(max_collision, res.collisions)

    return {
        "abs": abs_all.tolist(),
        "n_buckets": len(buckets),
        "perfect_buckets": perfect,
        "imperfect_buckets": imperfect,
        "max_collisions_in_a_bucket": max_collision,
        "max_bucket_size": max(len(v) for v in buckets.values()) if buckets else 0,
        "bucket_size_hist": _hist([len(v) for v in buckets.values()]),
    }


def _hist(sizes: list[int]) -> dict[str, int]:
    h: dict[str, int] = {}
    for s in sizes:
        h[str(s)] = h.get(str(s), 0) + 1
    return dict(sorted(h.items(), key=lambda kv: int(kv[0])))


def print_abs_report(a: dict) -> None:
    print(f"\n  准完美哈希（abs 段）:")
    print(f"    桶总数                : {a['n_buckets']}")
    print(f"    找到无冲突种子的桶    : {a['perfect_buckets']}")
    print(f"    无法无冲突的桶        : {a['imperfect_buckets']}")
    print(f"    最大桶内 token 数     : {a['max_bucket_size']}  (abs 槽位上限 256)")
    if a["imperfect_buckets"]:
        print(f"    单桶最大冲突数        : {a['max_collisions_in_a_bucket']}")


if __name__ == "__main__":
    raise SystemExit(main())
