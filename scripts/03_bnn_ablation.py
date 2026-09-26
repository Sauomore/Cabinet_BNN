# -*- coding: utf-8 -*-
"""
Cabinet-BNN MVP 实验二：BNN 二值激活网络 + per-token 权重码（路线文档 M3.6 核心消融）。

要回答的问题（这是判断整套新架构能否成立的唯一关键实验）：
    「128 位权重码 + 共享基矩阵生成 per-token 权重」
    与
    「普通全局共享权重」
    在【同参数量、同数据、同训练步数】下谁更好？

为什么必须做这个消融：
    per-token 可训练参数在数学上就是 embedding 层 —— Transformer 的 embedding
    表一直是 per-token 参数、一直由梯度更新、且只更新 batch 内出现的行。
    必须证明「码 + 共享基」不只是 embedding 的低配版，否则架构失去理由。

三种模式：
    code          完整方案：W_t = Σ_j s_j(hash_t, p_t)·B_j，激活二值
    shared        基线：普通全局权重矩阵（per-token 机制被消除）
    code_no_param 消融：param 段置零并冻结，只用 hash 段 → 检验 param 段的贡献

任务：字符级 next-char prediction。词索引决定用哪套 per-token 权重。

用法：
    python scripts/03_bnn_ablation.py --epochs 60 --d-model 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cabinet_bnn.paths import results_dir, vocab_3109

from cabinet_bnn.bnn.corpus import build_char_corpus, make_batches, train_val_split
from cabinet_bnn.bnn.model import build_model, count_params
from cabinet_bnn.data.embedding_cache import read_cache, normalize_rows
from cabinet_bnn.hsh.deep_hash import bits_to_u64, load_deep_hash, project, quantize_bits


# ---------------------------------------------------------------- 评测

@torch.no_grad()
def evaluate(model, seqs, idx, batch_size, device) -> tuple[float, float, float]:
    """返回 (loss, token 准确率, 首字符准确率)。

    注意：必须手动分批。make_batches 会打乱顺序，而 word_ids 必须与批次内
    样本一一对应，用打乱后的全局 idx 去索引会造成错位（静默的错误标签）。
    """
    model.eval()
    crit = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    total_loss = 0.0
    total_tok = 0
    correct = 0
    first_ok = 0
    first_n = 0

    for s in range(0, len(idx), batch_size):
        chunk = idx[s : s + batch_size]
        full = seqs[chunk]
        x = torch.from_numpy(full[:, :-1].astype(np.int64)).to(device)
        y = torch.from_numpy(
            np.where(full[:, 1:] == 0, -100, full[:, 1:]).astype(np.int64)
        ).to(device)
        wid = torch.from_numpy(chunk.astype(np.int64)).to(device)

        logits = model(x, wid)
        total_loss += float(crit(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())

        flat_y = y.reshape(-1)
        mask = flat_y != -100
        total_tok += int(mask.sum().item())
        pred = logits.reshape(-1, logits.shape[-1]).argmax(dim=-1)
        correct += int(((pred == flat_y) & mask).sum().item())

        fm = y[:, 0] != -100
        if fm.any():
            p0 = logits[:, 0, :].argmax(dim=-1)
            first_ok += int(((p0 == y[:, 0]) & fm).sum().item())
            first_n += int(fm.sum().item())

    return (total_loss / max(total_tok, 1),
            correct / max(total_tok, 1),
            first_ok / max(first_n, 1))


# ---------------------------------------------------------------- 单模式训练

def run_mode(mode: str, args, corpus, tr, va, hash_codes, device) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = build_model(
        mode, char_vocab=corpus.vocab_size, n_words=len(corpus.words),
        d_model=args.d_model, k_basis=args.k_basis, rank=args.rank,
        param_bits=args.param_bits, use_scale=args.use_scale,
        hidden=args.hidden, token_dim=args.token_dim,
    ).to(device)

    if mode in ("code", "code_no_param"):
        model.code_table.set_hash_from_u64(hash_codes)
        if mode == "code_no_param":
            # 消融：把 param 段真正排除（退出计算图），只保留 hash 段
            with torch.no_grad():
                model.code_table.param_shadow.zero_()
            model.code_table.param_shadow.requires_grad_(False)
        else:
            # 完整模式：param 初值策略很关键。
            # 早期版本用 N(0, 0.5) 随机初值 → 相当于在语义 mask 上叠加 64 位随机
            # 符号翻转，把码学到的几何彻底打乱，val_loss 反而比 code_no_param 差一倍。
            # 改为「中性初值」：param 全置同一常数 → sign 后为常量 +1，
            # 等价于从 code_no_param 的几何出发，后续再由梯度逐步引入 per-token 差异。
            with torch.no_grad():
                if args.param_init == "neutral":
                    model.code_table.param_shadow.fill_(args.param_neutral_value)
                elif args.param_init == "random":
                    model.code_table.param_shadow.normal_(0.0, 0.5)
                else:
                    raise ValueError(f"未知 param_init: {args.param_init}")
            model.code_table.refresh_codes()

    pstats = count_params(model)
    n_params = pstats["total"]

    # 基线：参数量对齐到 code 模式（粗对齐到同一量级）
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    crit = nn.CrossEntropyLoss(ignore_index=-100)

    hist = {"loss": [], "acc": []}
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    for ep in range(args.epochs):
        order = rng.permutation(len(tr))
        tr_sh = tr[order]
        model.train()
        for s in range(0, len(tr_sh), args.batch_size):
            chunk = tr_sh[s : s + args.batch_size]
            full = corpus.sequences[chunk]
            x = torch.from_numpy(full[:, :-1].astype(np.int64)).to(device)
            y = torch.from_numpy(
                np.where(full[:, 1:] == 0, -100, full[:, 1:]).astype(np.int64)
            ).to(device)
            wid = torch.from_numpy(chunk.astype(np.int64)).to(device)

            opt.zero_grad()
            logits = model(x, wid)
            loss = crit(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()

        # 滞回翻转：每若干 epoch 刷新一次码（路线文档 §4.2）
        if mode == "code" and (ep + 1) % args.flip_every == 0:
            model.code_table.refresh_codes()

        vl, vacc, wacc = evaluate(model, corpus.sequences, va, args.batch_size, device)
        hist["loss"].append(vl)
        hist["acc"].append(vacc)
        if (ep + 1) % max(1, args.epochs // 10) == 0 or ep == 0:
            print(f"    [{mode:14s}] epoch {ep + 1:3d}/{args.epochs}  "
                  f"val_loss={vl:.4f}  tok_acc={vacc:.4f}  first_acc={wacc:.4f}", flush=True)
    train_sec = time.time() - t0

    vl, vacc, wacc = evaluate(model, corpus.sequences, va, args.batch_size, device)
    tl, tacc, twacc = evaluate(model, corpus.sequences, tr, args.batch_size, device)

    res = {
        "mode": mode,
        "params": pstats,
        "val_loss": vl, "val_tok_acc": vacc, "val_first_acc": wacc,
        "train_loss": tl, "train_tok_acc": tacc, "train_first_acc": twacc,
        "train_seconds": round(train_sec, 1),
        "history_loss": hist["loss"],
    }
    if mode in ("code", "code_no_param"):
        res["code_stats"] = model.code_stats()
        if mode == "code":
            res["weight_diversity"] = model.layer1.weight_diversity()
    return res


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(description="Cabinet-BNN BNN 消融（M3.6）")
    ap.add_argument("--vocab", type=Path,
                    default=vocab_3109())
    ap.add_argument("--teacher-cache", type=Path,
                    default=results_dir() / "bge_large_3109.cache")
    ap.add_argument("--deep-hash", type=Path,
                    default=results_dir() / "deep_hash_mvp1_h512_s42.bin")
    ap.add_argument("--override", type=Path,
                    default=results_dir() / "sim_override_mvp1_h512_s42.bin")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--k-basis", type=int, default=64)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--param-bits", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--flip-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=0,
                    help="shared 模式隐层宽度（用于把基线参数量对齐到 code 模式；0=等于 d_model）")
    ap.add_argument("--token-dim", type=int, default=0,
                    help="token_embed 模式的 per-token 向量维度（0=等于 d_model）")
    ap.add_argument("--param-init", choices=["neutral", "random"], default="neutral",
                    help="param 段初值策略：neutral=全常数（从几何出发）；random=随机 ±1")
    ap.add_argument("--param-neutral-value", type=float, default=0.05,
                    help="neutral 初值的常数（需为非零，否则 sign 为 0）")
    ap.add_argument("--use-scale", action="store_true", default=True)
    ap.add_argument("--no-scale", dest="use_scale", action="store_false")
    ap.add_argument("--modes", nargs="+", default=["shared", "token_embed", "code_no_param", "code"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=results_dir() / "bnn_ablation.json")
    args = ap.parse_args()

    print("=" * 78)
    print("Cabinet-BNN · BNN 二值激活 + per-token 权重码（M3.6 核心消融）")
    print("=" * 78)
    device = torch.device(args.device)

    # ---- 语料 ----
    corpus = build_char_corpus(args.vocab)
    print("[语料]")
    print(corpus.stats())
    tr, va = train_val_split(corpus, 0.1, args.seed)
    print(f"  训练词 {len(tr)} / 验证词 {len(va)}")

    # ---- 载入 HSH-64 码（hash 段）----
    dim_t, words, teacher = read_cache(args.teacher_cache)
    if words != corpus.words:
        print("[警告] 嵌入缓存的词序与词表不一致，按词表顺序重排")
        pos = {w: i for i, w in enumerate(words)}
        teacher = teacher[[pos[w] for w in corpus.words]]
    S = normalize_rows(teacher) @ normalize_rows(teacher).T

    loaded = load_deep_hash(args.deep_hash)
    xc = teacher - teacher.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    student_dim = min(loaded.dim, xc.shape[1])
    student = normalize_rows(xc @ vt[:student_dim].T)
    u = project(loaded.model, loaded.mean, student)
    bits = quantize_bits(u)

    # 用后处理后的 sim 码（若存在）替换
    if args.override.exists():
        from cabinet_bnn.hsh.post_optimize import load_override_cache
        ow, obits = load_override_cache(args.override)
        if ow == corpus.words:
            bits = obits
            print(f"[码] 使用后处理后的 sim 码: {args.override.name}")
        else:
            print("[码] override 词序不一致，忽略")

    sim_u64 = bits_to_u64(bits)
    # 组 64 位 hash 码：feat(4)=0 + sim(52) + abs(8)=0
    from cabinet_bnn.hsh.codec import pack_codes
    hash_codes = pack_codes(
        np.zeros(len(corpus.words), dtype=np.uint8),
        sim_u64,
        np.zeros(len(corpus.words), dtype=np.uint8),
    )
    print(f"[码] hash 段唯一码数 = {len(np.unique(hash_codes))} / {len(corpus.words)}")

    # ---- 逐模式训练 ----
    results = []
    for mode in args.modes:
        print(f"\n{'─' * 78}\n[模式] {mode}\n{'─' * 78}")
        r = run_mode(mode, args, corpus, tr, va, hash_codes, device)
        ps = r["params"]
        print(f"    参数量: 总计 {ps['total']:,}  (per-token {ps['per_token']:,} + "
              f"共享 {ps['shared']:,})")
        print(f"    验证: loss={r['val_loss']:.4f}  tok_acc={r['val_tok_acc']:.4f}  "
              f"first_acc={r['val_first_acc']:.4f}   ({r['train_seconds']}s)")
        results.append(r)

    # ---- 对比 ----
    print("\n" + "=" * 78)
    print("对比结论")
    print("=" * 78)
    print(f"  {'模式':<16}{'参数量':>12}{'val_loss':>11}{'val_tok_acc':>13}{'val_1st_acc':>13}")
    print("  " + "-" * 66)
    for r in results:
        print(f"  {r['mode']:<16}{r['params']['total']:>12,}{r['val_loss']:>11.4f}"
              f"{r['val_tok_acc']:>13.4f}{r['val_first_acc']:>13.4f}")

    base = next((r for r in results if r["mode"] == "shared"), None)
    code = next((r for r in results if r["mode"] == "code"), None)
    if base and code:
        dl = code["val_loss"] - base["val_loss"]
        da = code["val_tok_acc"] - base["val_tok_acc"]
        print(f"\n  code vs shared:")
        print(f"    val_loss    {base['val_loss']:.4f} → {code['val_loss']:.4f}  ({dl:+.4f})")
        print(f"    val_tok_acc {base['val_tok_acc']:.4f} → {code['val_tok_acc']:.4f}  ({da:+.4f})")
        print(f"    → {'[OK] 码权重方案成立（不劣于共享权重）' if dl <= 0.02 else '[FAIL] 码权重方案劣于共享权重，架构需重新定位'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {k: str(v) for k, v in vars(args).items()},
        "corpus": {"n_words": len(corpus.words), "vocab_size": corpus.vocab_size,
                   "n_train": len(tr), "n_val": len(va)},
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
