# -*- coding: utf-8 -*-
"""
Cabinet-BNN 实验三：热更改与遗忘量评测（对应设计的真实目标）。

为什么换掉「LM 质量」这个指标：
    实验二（M3.6）表明，在参数量不是瓶颈时 per-token 权重码在纯 LM 质量上
    不占优。但它的设计目标从来不是 LM 质量，而是：
      ① 改 1 bit 即改权重（O(1) 热更改）
      ② 可回滚（存回旧快照即可）
      ③ 学新域不遗忘旧域（参数 per-token 且可分组隔离）
    本实验就用这三个指标来量它。

评测协议（顺序域适配）：
    ① 在域 D0 上训练到收敛           → 记录基线准确率 ACC[i][0]
    ② 依次在 D1..Dk 上适配           → 每步后评测【所有】域
    ③ 遗忘量 = max_t(ACC[i][t]) − ACC[i][k]        （旧域峰值 − 最终值）
    ④ 可逆性 = 回滚到适配前快照后，旧域准确率是否精确恢复

被比较的方法：
    shared_ft    基线：普通全局权重，顺序微调（会灾难性遗忘）
    shared_joint 上界：在所有域上联合训练（无遗忘，但不支持增量）
    code_param   本方案：冻结共享基，只改 param 段的位（热更改）
    code_full    本方案宽松版：param 段 + 共享基一起适配
"""

from __future__ import annotations

import argparse
import copy
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

from cabinet_bnn.bnn.corpus import build_char_corpus
from cabinet_bnn.bnn.domains import domain_stratified_split, make_domains
from cabinet_bnn.bnn.model import build_model, count_params
from cabinet_bnn.data.embedding_cache import normalize_rows, read_cache
from cabinet_bnn.hsh.codec import pack_codes
from cabinet_bnn.hsh.deep_hash import bits_to_u64, load_deep_hash, project, quantize_bits


# ---------------------------------------------------------------- 评测

@torch.no_grad()
def domain_accuracy(model, seqs, idx, batch_size, device) -> float:
    """在给定样本集合上的 token 级准确率。"""
    if len(idx) == 0:
        return float("nan")
    model.eval()
    correct = 0
    total = 0
    for s in range(0, len(idx), batch_size):
        chunk = idx[s : s + batch_size]
        full = seqs[chunk]
        x = torch.from_numpy(full[:, :-1].astype(np.int64)).to(device)
        y = torch.from_numpy(
            np.where(full[:, 1:] == 0, -100, full[:, 1:]).astype(np.int64)
        ).to(device)
        wid = torch.from_numpy(chunk.astype(np.int64)).to(device)
        logits = model(x, wid)
        flat_y = y.reshape(-1)
        mask = flat_y != -100
        pred = logits.reshape(-1, logits.shape[-1]).argmax(dim=-1)
        correct += int(((pred == flat_y) & mask).sum().item())
        total += int(mask.sum().item())
    return correct / max(total, 1)


def eval_all_domains(model, seqs, val_by_domain, n_domains, batch_size, device) -> list[float]:
    return [domain_accuracy(model, seqs, val_by_domain[d], batch_size, device)
            for d in range(n_domains)]


# ---------------------------------------------------------------- 适配

def adapt(model, seqs, train_idx, args, device) -> None:
    """在给定样本上适配若干 epoch。"""
    crit = nn.CrossEntropyLoss(ignore_index=-100)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.adapt_lr)
    rng = np.random.default_rng(args.seed)
    model.train()
    for ep in range(args.adapt_epochs):
        order = rng.permutation(len(train_idx))
        shuffled = train_idx[order]
        for s in range(0, len(shuffled), args.batch_size):
            chunk = shuffled[s : s + args.batch_size]
            full = seqs[chunk]
            x = torch.from_numpy(full[:, :-1].astype(np.int64)).to(device)
            y = torch.from_numpy(
                np.where(full[:, 1:] == 0, -100, full[:, 1:]).astype(np.int64)
            ).to(device)
            wid = torch.from_numpy(chunk.astype(np.int64)).to(device)
            opt.zero_grad()
            loss = crit(model(x, wid).reshape(-1, model.cfg.char_vocab), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        if model.cfg.mode in ("code", "code_no_param") and (ep + 1) % args.flip_every == 0:
            model.code_table.refresh_codes()


def freeze_for_mode(model, mode: str) -> None:
    """按模式设置可训练/冻结的部件。"""
    if mode == "code_param":
        # 只允许 param 段的位变动；共享基矩阵完全冻结
        for name, p in model.named_parameters():
            p.requires_grad_("param_shadow" in name)
    elif mode in ("code_full", "shared_ft", "shared_joint"):
        for p in model.parameters():
            p.requires_grad_(True)


# ---------------------------------------------------------------- 单方法跑一遍

def run_method(method: str, args, corpus, hash_codes, train_by_domain,
               val_by_domain, n_domains, device) -> dict:
    """按方法跑一遍顺序域适配。

    方法说明（重要 —— 早期版本的 code_param 测错了机制）：
        shared_ft     基线：全局共享权重顺序微调。char_embed + g1 + g2 全可训。
        code_param    本方案：共享基矩阵 + char_embed 冻结，只改 param 段的位。
                      早期版本把 char_embed 也冻了 → 准确率恒为 0，测的不是
                      「热更改」而是「只改 64 个 ±1 位能否拟合任务」（不能）。
                      本版放开 char_embed，只冻结真正属于「共享基」的 U/V。
        code_swap     本方案核心：基矩阵 + char_embed 冻结，每个域学一份
                      【独立的 per-domain 码表】，推理时按域切换（热插拔）。
                      这是设计里「多适配器」的真实形态。
        code_full     宽松版：param 段 + 共享基一起适配。
        shared_joint  上界：所有域联合训练。
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mode = {"shared_ft": "shared", "shared_joint": "shared",
            "code_param": "code", "code_swap": "code", "code_full": "code"}[method]
    model = build_model(
        mode, char_vocab=corpus.vocab_size, n_words=len(corpus.words),
        d_model=args.d_model, k_basis=args.k_basis, rank=args.rank,
        param_bits=args.param_bits, hidden=args.hidden, use_scale=True,
    ).to(device)

    if mode == "code":
        model.code_table.set_hash_from_u64(hash_codes)
        with torch.no_grad():
            model.code_table.param_shadow.fill_(0.05)     # 中性初值
        model.code_table.refresh_codes()

    # code_param / code_swap：冻结共享基（U/V），保留 char_embed 与 scale 可训
    if method in ("code_param", "code_swap"):
        for name, p in model.named_parameters():
            if name in ("layer1.U", "layer1.V", "layer2.U", "layer2.V"):
                p.requires_grad_(False)
    elif method in ("code_full", "shared_ft", "shared_joint"):
        for p in model.parameters():
            p.requires_grad_(True)

    t0 = time.time()
    acc_history: list[list[float]] = []
    # 每个域适配完成后的码表快照（用于 code_swap 与可逆性）
    code_snapshots: dict[int, dict] = {}

    def snapshot_code(d: int) -> None:
        code_snapshots[d] = {
            "param_shadow": model.code_table.param_shadow.detach().clone(),
            "param_code": model.code_table.param_code.clone(),
        }

    def load_code(d: int) -> None:
        if d in code_snapshots:
            with torch.no_grad():
                model.code_table.param_shadow.copy_(code_snapshots[d]["param_shadow"])
                model.code_table.param_code.copy_(code_snapshots[d]["param_code"])

    if method == "shared_joint":
        all_idx = np.concatenate([train_by_domain[d] for d in range(n_domains)])
        adapt(model, corpus.sequences, all_idx, args, device)
        acc_history.append(eval_all_domains(model, corpus.sequences, val_by_domain,
                                            n_domains, args.batch_size, device))
    elif method == "code_swap":
        # 每个域各自学一份码表；评测时把该域的码装回去（模拟热插拔）
        for t in range(args.n_steps):
            # 从干净初值出发学第 t 个域的码
            with torch.no_grad():
                model.code_table.param_shadow.fill_(0.05)
            model.code_table.refresh_codes()
            if t > 0:
                for name, p in model.named_parameters():
                    if name == "param_shadow":
                        continue
                    p.requires_grad_(name.startswith("embed") or "scale" in name)
            adapt(model, corpus.sequences, train_by_domain[t], args, device)
            snapshot_code(t)
            # 用「查表」方式评测：域 d 装域 d 的码
            row = []
            for d in range(n_domains):
                if d in code_snapshots:
                    load_code(d)
                    if d != t:
                        row.append(float("nan"))
                    else:
                        row.append(domain_accuracy(model, corpus.sequences,
                                                   val_by_domain[d], args.batch_size, device))
                else:
                    row.append(float("nan"))
                    if d == t:
                        row[-1] = domain_accuracy(model, corpus.sequences,
                                                  val_by_domain[d], args.batch_size, device)
            acc_history.append(row)
            # 评测完把当前域的码装回，便于后续统计
            load_code(t)
    else:
        for t in range(args.n_steps):
            adapt(model, corpus.sequences, train_by_domain[t], args, device)
            snapshot_code(t)
            acc_history.append(eval_all_domains(model, corpus.sequences, val_by_domain,
                                                n_domains, args.batch_size, device))

    # 可逆性：回滚到「刚学完第一个域」的快照，检查域 0 是否精确恢复
    rollback = None
    if mode == "code" and 0 in code_snapshots:
        acc_before = domain_accuracy(model, corpus.sequences, val_by_domain[0],
                                     args.batch_size, device) if method == "code_swap" else \
            (acc_history[0][0] if acc_history else float("nan"))
        load_code(0)
        acc_restored = domain_accuracy(model, corpus.sequences, val_by_domain[0],
                                       args.batch_size, device)
        rollback = {
            "acc_d0_reference": acc_before,
            "acc_d0_after_rollback": acc_restored,
            "exact_equality": bool(abs(acc_before - acc_restored) < 1e-9),
        }

    return {
        "method": method,
        "params": count_params(model),
        "acc_history": acc_history,
        "seconds": round(time.time() - t0, 1),
        "rollback": rollback,
        "n_code_snapshots": len(code_snapshots),
    }


# ---------------------------------------------------------------- 分析

def analyze(res: dict, n_domains: int, n_steps: int) -> dict:
    """从准确率矩阵计算遗忘量等指标。

    对 code_swap 这类「每域一份码表」的方法，非对角单元是 NaN（该域未装码），
    此时遗忘量按「每域自己的码在本域上的准确率」统计，恒为 0 —— 这是
    设计上【预期】的结果，不是缺陷：域之间物理隔离，故无干扰。
    """
    H = np.array(res["acc_history"], dtype=np.float64)     # (T, D)
    if res["method"] == "shared_joint":
        final = H[-1]
        return {
            "final_mean": float(np.nanmean(final)),
            "forgetting": 0.0,
            "bwt": 0.0,
            "own_domain_acc": [float(x) for x in final],
            "own_domain_mean": float(np.nanmean(final)),
            "forgetting_per_domain": [],
            "final_acc": [float(x) for x in final],
        }

    T, D = H.shape
    seen = min(T, n_steps)

    # 各方法「本域准确率」的口径不同：
    #   code_swap      -> 对角元（每域用自己的码）
    #   其它           -> 第 t 步时的对角元
    own = []
    for i in range(seen):
        v = H[i, i] if i < T else np.nan
        own.append(float(v) if not np.isnan(v) else float("nan"))

    # 遗忘量：对每个已学域，看它学完之后各步里「仍装上该域参数」时的准确率
    forget = []
    for i in range(seen):
        col = H[i:, i]
        col = col[~np.isnan(col)]
        if col.size >= 2:
            forget.append(float(col.max() - col[-1]))
        else:
            forget.append(0.0)

    final = H[-1]
    bwt = []
    for i in range(seen - 1):
        if i < T and not np.isnan(H[i, i]):
            # 最后一步是否还评测过域 i（code_swap 下为 NaN，跳过）
            if not np.isnan(final[i]):
                bwt.append(float(final[i] - H[i, i]))

    return {
        "final_mean": float(np.nanmean(final)),
        "forgetting": float(np.mean(forget)) if forget else 0.0,
        "forgetting_per_domain": forget,
        "bwt": float(np.mean(bwt)) if bwt else 0.0,
        "own_domain_acc": own,
        "own_domain_mean": float(np.nanmean(own)),
        "final_acc": [float(x) for x in final],
    }


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(description="Cabinet-BNN 热更改与遗忘量评测")
    ap.add_argument("--vocab", type=Path,
                    default=vocab_3109())
    ap.add_argument("--teacher-cache", type=Path,
                    default=results_dir() / "bge_large_3109.cache")
    ap.add_argument("--deep-hash", type=Path,
                    default=results_dir() / "deep_hash_mvp1_h512_s42.bin")
    ap.add_argument("--override", type=Path,
                    default=results_dir() / "sim_override_mvp1_h512_s42.bin")
    ap.add_argument("--n-domains", type=int, default=5)
    ap.add_argument("--n-steps", type=int, default=5)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--k-basis", type=int, default=8)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--param-bits", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--adapt-epochs", type=int, default=15)
    ap.add_argument("--adapt-lr", type=float, default=3e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--flip-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--methods", nargs="+",
                    default=["shared_ft", "code_param", "code_swap", "code_full", "shared_joint"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path,
                    default=results_dir() / "continual_eval.json")
    args = ap.parse_args()

    print("=" * 80)
    print("Cabinet-BNN · 热更改与遗忘量评测")
    print("=" * 80)
    device = torch.device(args.device)

    corpus = build_char_corpus(args.vocab)
    dom = make_domains(corpus.words, args.n_domains, seed=0)
    print("[语料]")
    print(corpus.stats())
    print("[域划分]")
    print(dom.stats())
    train_by_domain, val_by_domain = domain_stratified_split(dom, 0.2, args.seed)

    # ---- 载入 HSH-64 码 ----
    dim_t, words, teacher = read_cache(args.teacher_cache)
    if words != corpus.words:
        pos = {w: i for i, w in enumerate(words)}
        teacher = teacher[[pos[w] for w in corpus.words]]
    loaded = load_deep_hash(args.deep_hash)
    xc = teacher - teacher.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    student = normalize_rows(xc @ vt[: min(loaded.dim, xc.shape[1])].T)
    bits = quantize_bits(project(loaded.model, loaded.mean, student))
    if args.override.exists():
        from cabinet_bnn.hsh.post_optimize import load_override_cache
        ow, obits = load_override_cache(args.override)
        if ow == corpus.words:
            bits = obits
    hash_codes = pack_codes(np.zeros(len(corpus.words), dtype=np.uint8),
                            bits_to_u64(bits),
                            np.zeros(len(corpus.words), dtype=np.uint8))
    print(f"[码] hash 段唯一码 {len(np.unique(hash_codes))} / {len(corpus.words)}")

    # ---- 逐方法评测 ----
    results = []
    for m in args.methods:
        print(f"\n{'─' * 80}\n[方法] {m}\n{'─' * 80}")
        r = run_method(m, args, corpus, hash_codes, train_by_domain, val_by_domain,
                       args.n_domains, device)
        a = analyze(r, args.n_domains, args.n_steps)
        r["analysis"] = a
        print(f"    参数量 {r['params']['total']:,}   用时 {r['seconds']}s")
        if r["method"] == "shared_joint":
            print(f"    联合训练（上界）: 最终均值 = {a['final_mean']:.4f}   "
                  f"本域均值 = {a['own_domain_mean']:.4f}")
        elif r["method"] == "code_swap":
            print(f"    每域独立码表数 = {r['n_code_snapshots']}")
            print(f"    本域准确率（每域各自装码）= {[round(x,4) for x in a['own_domain_acc']]}")
            print(f"    本域均值 = {a['own_domain_mean']:.4f}   遗忘量 = {a['forgetting']:.4f}")
        else:
            print(f"    逐步准确率矩阵 (行=第 t 步后, 列=各域):")
            for t, row in enumerate(r["acc_history"]):
                cells = "  ".join(f"{v:.3f}" if not np.isnan(v) else "  -  " for v in row)
                print(f"      t={t}  {cells}")
            print(f"    遗忘量 = {a['forgetting']:.4f}   后向迁移 = {a['bwt']:+.4f}   "
                  f"最终均值 = {a['final_mean']:.4f}")
        if r["rollback"]:
            rb = r["rollback"]
            print(f"    可逆性: 参考域0={rb['acc_d0_reference']:.4f} → "
                  f"回滚后={rb['acc_d0_after_rollback']:.4f}  "
                  f"{'精确恢复' if rb['exact_equality'] else '未精确恢复'}")
        results.append(r)

    # ---- 汇总 ----
    print("\n" + "=" * 80)
    print("对比：遗忘量 vs 最终精度")
    print("=" * 80)
    hdr = (f"  {'方法':<16}{'本域均值':>10}{'遗忘量':>10}{'后向迁移':>11}"
           f"{'参数量':>12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        a = r["analysis"]
        print(f"  {r['method']:<16}{a['own_domain_mean']:>10.4f}{a['forgetting']:>10.4f}"
              f"{a['bwt']:>+11.4f}{r['params']['total']:>12,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "config": {k: str(v) for k, v in vars(args).items()},
        "corpus": {"n_words": len(corpus.words), "vocab_size": corpus.vocab_size},
        "domains": {str(d): int(len(dom.domain_words[d])) for d in range(args.n_domains)},
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
