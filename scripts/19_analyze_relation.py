# -*- coding: utf-8 -*-
"""
分析训练后的关系门学到了什么。

三个问题：
  ① γ 在各层的分布 —— 哪些层更需要关系信息？
  ② 关系分数高的 token 对有什么共性？（是否捕捉到语法/搭配结构）
  ③ 关系矩阵与【共现统计】是否相关 —— 这是「外网存注意力模式」的直接检验

用法：
    python scripts/19_analyze_relation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from cabinet_bnn.bnn.attention import build_transformer


def main() -> int:
    ap = argparse.ArgumentParser(description="分析训练后的关系门")
    ap.add_argument("--ckpt", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "results/lm_relgate_ft/final.pt")
    ap.add_argument("--data-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data/corpus_all")
    ap.add_argument("--n-pairs", type=int, default=200_000)
    args = ap.parse_args()

    device = torch.device("cpu")
    stats = json.loads((args.data_dir / "data_stats.json").read_text(encoding="utf-8"))
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = ck.get("args", {})
    model = build_transformer(
        vocab_size=stats["vocab_size"], n_words=stats["vocab_size"],
        preset=a.get("preset", "base"), max_len=a.get("ctx", 256),
        relation_gate=True, relation_dim=a.get("relation_dim", 16),
    )
    model.load_state_dict(ck["model"])
    model.eval()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.data_dir / "tokenizer.json"))

    print("=" * 76)
    print("训练后的关系门分析")
    print("=" * 76)
    print(f"  checkpoint step {ck.get('step')}  val_loss {ck.get('val_loss'):.4f}")

    # ---------- ① γ 分布 ----------
    print("\n[1] γ 分布（门强度）")
    gammas = {}
    for i, blk in enumerate(model.blocks):
        g = blk.attn.rel_gamma.detach()
        gammas[i] = float(g[0])
    for i, g in gammas.items():
        bar = "#" * int(abs(g) * 300)
        print(f"    layer {i}: γ = {g:+.5f}  {bar}")
    vals = list(gammas.values())
    print(f"    |γ| 均值 {np.mean(np.abs(vals)):.5f}   "
          f"最大 {max(vals, key=abs):+.5f} (layer {max(gammas, key=lambda k: abs(gammas[k]))})")

    # ---------- ② 关系分数最高的 token 对 ----------
    print("\n[2] 关系分数最高 / 最低的 token 对（取第 2 层，γ 最大）")
    layer = max(gammas, key=lambda k: abs(gammas[k]))
    att = model.blocks[layer].attn
    V = stats["vocab_size"]
    # 只用高频 token（否则大量是未充分训练的）
    with torch.no_grad():
        u = att.rel_embed.weight.detach()             # (V, r)
        un = u / (u.norm(dim=1, keepdim=True) + 1e-9)
        # 取前 4000 个高频 token（BPE 词表按频率排序，id 越小越常用）
        K = 4000
        unk = un[:K]
        sim = unk @ unk.T
        sim.fill_diagonal_(-2.0)

        flat = sim.flatten()
        top = torch.topk(flat, 60).indices
        seen, shown = set(), 0
        print(f"    （层 {layer}，前 {K} 个高频 token）")
        print(f"\n    关系最【强】的 token 对:")
        for idx in top:
            i, j = int(idx // K), int(idx % K)
            key = tuple(sorted((i, j)))
            if key in seen:
                continue
            seen.add(key)
            wi = tok.decode([i]); wj = tok.decode([j])
            print(f"      {wi!r:>14} <-> {wj!r:<14}  分数 {sim[i,j]:+.4f}")
            shown += 1
            if shown >= 12:
                break

        bot = torch.topk(flat, 60, largest=False).indices
        seen2, shown = set(), 0
        print(f"\n    关系最【弱】的 token 对:")
        for idx in bot:
            i, j = int(idx // K), int(idx % K)
            key = tuple(sorted((i, j)))
            if key in seen2:
                continue
            seen2.add(key)
            wi = tok.decode([i]); wj = tok.decode([j])
            print(f"      {wi!r:>14} <-> {wj!r:<14}  分数 {sim[i,j]:+.4f}")
            shown += 1
            if shown >= 8:
                break

    # ---------- ③ 与共现统计的相关性 ----------
    print("\n[3] 关系分数 vs 【实际共现频率】（「外网存注意力模式」的检验）")
    print("    若相关，说明关系向量确实编码了 token 之间的统计关联")
    arr = np.memmap(args.data_dir / "train.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    ctx = 512
    n_win = max(1, args.n_pairs // ctx)
    counts = {}
    pair_cnt = {}
    total = 0
    for _ in range(n_win):
        s = int(rng.integers(0, len(arr) - ctx - 1))
        seq = np.asarray(arr[s:s + ctx], dtype=np.int64)
        seq = seq[seq < 4000]                          # 只统计高频区
        uniq = np.unique(seq)
        for w in uniq:
            counts[int(w)] = counts.get(int(w), 0) + 1
        for x in range(len(uniq)):
            for y in range(x + 1, len(uniq)):
                k = (int(uniq[x]), int(uniq[y]))
                pair_cnt[k] = pair_cnt.get(k, 0) + 1
        total += 1

    with torch.no_grad():
        u = att.rel_embed.weight.detach()[:4000]
        un2 = u / (u.norm(dim=1, keepdim=True) + 1e-9)
        rel = un2 @ un2.T

    # PMI 近似：log( p(x,y) / (p(x)p(y)) )
    N = total
    xs, ys, pmi_vals, rel_vals = [], [], [], []
    for (i, j), c in pair_cnt.items():
        if counts.get(i, 0) < 5 or counts.get(j, 0) < 5:
            continue
        pxy = c / N
        px = counts[i] / N
        py = counts[j] / N
        pmi = np.log(pxy / (px * py) + 1e-12)
        xs.append(i); ys.append(j)
        pmi_vals.append(pmi)
        rel_vals.append(float(rel[i, j]))

    if len(pmi_vals) > 100:
        pmi_vals = np.array(pmi_vals)
        rel_vals = np.array(rel_vals)
        corr = float(np.corrcoef(pmi_vals, rel_vals)[0, 1])
        print(f"      统计了对数: {len(pmi_vals):,}")
        print(f"      corr(PMI, 关系分数) = {corr:+.4f}")
        # 分位数对比
        q = np.quantile(pmi_vals, [0.1, 0.5, 0.9])
        print(f"      PMI 低 10% 的平均关系分数: "
              f"{rel_vals[pmi_vals <= q[0]].mean():+.5f}")
        print(f"      PMI 中位的平均关系分数  : "
              f"{rel_vals[(pmi_vals > q[0]) & (pmi_vals <= q[2])].mean():+.5f}")
        print(f"      PMI 高 10% 的平均关系分数: "
              f"{rel_vals[pmi_vals > q[2]].mean():+.5f}")
        if corr > 0.1:
            print("      >> 正相关：关系向量确实编码了 token 间的统计关联 ✅")
        elif corr < -0.1:
            print("      >> 负相关：需进一步分析")
        else:
            print("      >> 接近 0：关系向量未必学到共现结构（也可能是任务不需要）")

    # 保存
    out = Path(args.ckpt).parent / "relation_analysis.json"
    out.write_text(json.dumps({
        "step": ck.get("step"), "val_loss": ck.get("val_loss"),
        "gammas": gammas, "gamma_absmean": float(np.mean(np.abs(vals))),
        "corr_pmi_vs_relation": corr if len(pmi_vals) > 100 else None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
