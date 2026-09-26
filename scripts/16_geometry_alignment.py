# -*- coding: utf-8 -*-
"""
验证「几何对齐」分析：独立 param 段 vs 从语义码导出的 param 段。

预测（来自 _analyze_geometry.py 的推导）：
    ① 独立 param（当前实现）
       E[s_j(a)s_j(b)] = 0  ->  权重相似度与 Hamming 距离【无关】
    ② param = G(hash)（共享函数导出）
       语义近 -> hash 近 -> param 近 -> 权重相似度应随距离【单调下降】

不需要训练：只构造码、算权重、测相关性。

用法：
    python scripts/16_geometry_alignment.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from cabinet_bnn.bnn.model import WeightCodeConfig, WeightCodeTable


def build_sign_patterns(mode: str, n: int, k: int, seed: int = 0):
    """构造符号模式，并【同时返回语义码的位矩阵】。

    ⚠️ 关键设计（踩过两次坑）：
        必须区分两个不同的 Hamming 距离：
          · d_H(语义码)     —— 输入侧的几何（检索用的那个）
          · d_H(符号模式)   —— 权重空间的几何
        权重由符号模式生成，所以「d_H(符号模式) → 权重相似度」是【恒等式】，
        用它测相关性没有意义（第一次做就踩了这个坑）。

        有意义的命题是：**语义码的几何能否预测权重的几何**。
        那才是「per-token 权重设计」要解决的问题。

    Returns:
        (sign_patterns, code_bits, k)
        sign_patterns: (n, k)  ∈ {−1,+1}
        code_bits:     (n, k)  ∈ {0,1}   语义码的位
    """
    torch.manual_seed(seed)
    cfg = WeightCodeConfig(k_basis=k, rank=16, param_bits=64)
    ct = WeightCodeTable(n, cfg)

    # 构造有语义结构的 hash 码：让一部分 token 聚成簇
    # 同簇内 bit 差异小（模拟"语义近"），跨簇差异大
    n_clusters = 8
    base = torch.randint(0, 2, (n_clusters, k), dtype=torch.int64)
    codes = []
    for i in range(n):
        b = base[i % n_clusters].clone()
        # 每 n/n_clusters 个 token 增加翻转数，形成梯度
        step = i // n_clusters
        if step > 0:
            idx = torch.randperm(k)[:step]
            b[idx] ^= 1
        codes.append(b)
    hb = torch.stack(codes)                # (n, k)
    weights = (torch.ones(k, dtype=torch.int64) << torch.arange(k))
    h64 = (hb * weights).sum(-1)
    ct.set_hash_from_u64(h64)

    with torch.no_grad():
        if mode == "random_param":
            ct.param_shadow.normal_(0, 0.5)
        elif mode == "learned_fn":
            G = torch.randn(k, ct.param_code.shape[1]) * 0.35
            ct.param_shadow.copy_(hb.float() @ G)
        elif mode == "mask_only":
            ct.param_shadow.fill_(0.05)      # sign -> 全 +1，即 param 不起作用
    ct.refresh_codes()
    return ct.sign_patterns(), hb


def weight_similarity(s: torch.Tensor, k: int, seed: int = 1) -> torch.Tensor:
    """给定符号模式，计算两两权重矩阵的余弦相似度。"""
    g = torch.Generator().manual_seed(seed)
    d = 32
    U = torch.randn(k, 4, d, generator=g)
    V = torch.randn(k, 4, d, generator=g)
    B = torch.einsum("jro,jri->joi", U, V)          # (k, d, d)
    W = torch.einsum("nj,joi->noi", s, B)           # (n, d, d)
    Wf = W.reshape(W.shape[0], -1)
    Wn = Wf / (Wf.norm(dim=1, keepdim=True) + 1e-9)
    return Wn @ Wn.T


def code_hamming(code_bits: torch.Tensor) -> torch.Tensor:
    """【语义码】的 Hamming 距离 —— 输入侧几何（检索用的那个）。"""
    return (code_bits[:, None, :] != code_bits[None, :, :]).sum(-1).float()


def analyze(mode: str, n: int, k: int, seed: int = 0) -> dict:
    s, code_bits = build_sign_patterns(mode, n, k, seed)
    cos = weight_similarity(s, k)
    hd = code_hamming(code_bits)                     # 注意：用语义码，不是符号模式

    iu = torch.triu_indices(n, n, offset=1)
    h, c = hd[iu[0], iu[1]], cos[iu[0], iu[1]]

    # 若 h 无变化则相关系数无定义
    if float(h.std()) < 1e-6:
        corr = float("nan")
    else:
        corr = float(torch.corrcoef(torch.stack([h, c]))[0, 1])

    nb = max(1, k // 4)
    bins = {}
    for lo in range(0, k + 1, nb):
        m = (h >= lo) & (h < lo + nb)
        if m.any():
            bins[f"{lo}-{lo + nb - 1}"] = float(c[m].mean())
    return {
        "mode": mode, "n": n, "k": k,
        "corr_codeHamming_vs_cos": corr,
        "cos_overall_absmean": float(c.abs().mean()),
        "cos_bins": bins,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="几何对齐验证")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results/geometry_alignment.json")
    args = ap.parse_args()

    print("=" * 76)
    print("几何对齐验证：权重相似度 vs 符号模式 Hamming 距离")
    print("=" * 76)
    print(f"  n={args.n} tokens, k={args.k} 基矩阵")
    print("  命题：语义码的 Hamming 距离能否预测权重的相似度？")
    print("        （注意：用符号模式的 Hamming 测是无意义的恒等式 —— 踩过这个坑）")

    results = []
    for mode in ("random_param", "learned_fn", "mask_only"):
        r = analyze(mode, args.n, args.k)
        results.append(r)
        print(f"\n  【{mode}】")
        print(f"    相关系数 corr(语义码Hamming, 权重余弦) = {r['corr_codeHamming_vs_cos']:+.4f}")
        print(f"    权重余弦平均 |cos|               = {r['cos_overall_absmean']:.4f}")
        print(f"    按距离分桶的平均相似度：")
        for k_, v in r["cos_bins"].items():
            print(f"      距离 {k_:>6}: {v:+.4f}")

    print("\n" + "=" * 76)
    print("结论")
    print("=" * 76)
    d = {r["mode"]: r for r in results}
    rp = d["random_param"]["corr_codeHamming_vs_cos"]
    lf = d["learned_fn"]["corr_codeHamming_vs_cos"]
    mo = d["mask_only"]["corr_codeHamming_vs_cos"]

    print(f"  ① 独立 param（当前实现）      : {rp:+.4f}   "
          f"{'符合预测（无关）' if abs(rp) < 0.25 else '与预测不符'}")
    print(f"  ② param = G(hash)（共享函数）  : {lf:+.4f}   "
          f"{'符合预测（负相关）' if lf < -0.25 else '与预测不符'}")
    print(f"  ③ 只有 mask（纯语义码）        : {mo:+.4f}")

    if abs(rp) < 0.25 and lf < -0.25:
        print("\n  >> 分析得到验证：")
        print("     · 独立 param 段会【破坏】权重几何（与语义无关）")
        print("     · 让 param 由共享函数从语义码导出，可恢复单调关系")
        print("     · 这就是「per-token 权重该怎么设计」的可操作答案")
    else:
        print("\n  >> 预测未完全成立，需要进一步分析")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
