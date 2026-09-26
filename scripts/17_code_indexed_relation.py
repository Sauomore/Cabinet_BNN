# -*- coding: utf-8 -*-
"""
码几何 → 关系几何 的传导验证（不需要训练）。

命题（来自设计）：
    「码是索引，信息码是内容」—— 如果关系向量 u_t 由【码】导出，那么
    「码的 Hamming 距离近」应当结构性导致「关系分数高」。
    若 u_t 随机初始化，则两者无关。

这个验证把两件事分开：
    · 机制对不对（本脚本）    —— 用【受控的】码结构，不依赖真实语义
    · 码好不好（实验一已做）  —— Recall@10 = 0.7426

为什么要分开：若直接用真实码测出无关，无法判断是机制问题还是码的问题。

设计要点：
    用【簇结构】的码做受控实验 —— 已知哪些 token 语义近（同簇），
    然后检验：
      ① 码导出的 u 是否让关系分数恢复簇结构
      ② 随机 u 是否做不到
      ③ 训练能否让随机 u 也学到（若 ① 成立，训练应更快/更好）

用法：
    python scripts/17_code_indexed_relation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from cabinet_bnn.bnn.attention import CausalSelfAttention


# ---------------------------------------------------------------- 受控码

def make_clustered_codes(n: int, bits: int, n_clusters: int,
                         seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """构造带簇结构的码。

    返回 (codes, labels)：
        codes  : (n, bits) ∈ {0,1}
        labels : (n,)      每个 token 属于哪个语义簇
    同簇内 Hamming 距离小，跨簇大 —— 模拟「HSH 码里语义近的 token 码也近」。
    """
    g = torch.Generator().manual_seed(seed)
    centers = torch.randint(0, 2, (n_clusters, bits), generator=g)
    labels = torch.arange(n) % n_clusters
    codes = centers[labels].clone()
    # 同簇内加少量噪声（模拟码不是完全相同的）
    noise = (torch.rand(n, bits, generator=g) < 0.06).long()
    codes = codes ^ noise
    return codes, labels


def codes_to_u(codes: torch.Tensor, dim: int, scale: float = 1.0,
               seed: int = 0) -> torch.Tensor:
    """把 0/1 码映射为关系向量（这是「码作为索引」的落地方式之一）。

    做法：把码的位用固定的随机投影打到 r 维，使
        <u_a, u_b> ∝ (码 a 与码 b 的一致位数 - 不一致位数)
    即码越近 -> 内积越大。 用 ±1 表示位，投影用随机矩阵 R (bits, r)：
        u_t = (2·code_t - 1) @ R / sqrt(bits)
    因为 R 固定，<u_a,u_b> = (s_a·s_b) 的线性函数，而 s_a·s_b = bits - 2·d_H。
    所以 Hamming 距离与关系分数是【严格单调】关系 —— 这是结构性保证。
    """
    g = torch.Generator().manual_seed(seed)
    bits = codes.shape[1]
    R = torch.randn(bits, dim, generator=g) / (bits ** 0.5)
    s = (2.0 * codes.float() - 1.0)                      # (n, bits) ∈ ±1
    u = s @ R * scale
    return u


# ---------------------------------------------------------------- 度量

def metric_correlation(u: torch.Tensor, codes: torch.Tensor,
                       labels: torch.Tensor) -> dict:
    """计算三组量并给出相关性/可分性。

    ① corr(码 Hamming 距离, 关系分数)      <- 期望显著为负
    ② 簇内平均关系 vs 簇间平均关系          <- 期望簇内显著更高
    ③ 用关系分数做簇分类的准确率（最近邻）  <- 期望远高于随机
    """
    n = u.shape[0]
    rel = (u @ u.T)                                      # (n, n) 关系分数
    hd = (codes[:, None, :] != codes[None, :, :]).sum(-1).float()

    iu = torch.triu_indices(n, n, offset=1)
    h, r = hd[iu[0], iu[1]], rel[iu[0], iu[1]]
    corr = float(torch.corrcoef(torch.stack([h, r]))[0, 1]) if h.std() > 1e-9 else float("nan")

    same = labels[:, None] == labels[None, :]
    off = ~torch.eye(n, dtype=torch.bool)
    within = float(rel[same & off].mean()) if (same & off).any() else float("nan")
    between = float(rel[~same].mean()) if (~same).any() else float("nan")

    # 最近邻簇一致率（排除自身）
    r2 = rel.clone()
    r2.fill_diagonal_(-float("inf"))
    nn = r2.argmax(dim=1)
    nn_acc = float((labels[nn] == labels).float().mean())

    return {"corr_hamming_vs_relation": corr,
            "within_cluster": within, "between_cluster": between,
            "separation": within - between,
            "nn_cluster_acc": nn_acc}


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(description="码几何 -> 关系几何 传导验证")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--bits", type=int, default=64, help="码位数")
    ap.add_argument("--clusters", type=int, default=16)
    ap.add_argument("--dim", type=int, default=16, help="关系向量维度 r")
    ap.add_argument("--steps", type=int, default=400, help="训练步骤（验证可学性）")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "results/code_indexed_relation.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    codes, labels = make_clustered_codes(args.n, args.bits, args.clusters)
    print("=" * 76)
    print("码几何 -> 关系几何 传导验证")
    print("=" * 76)
    print(f"  n={args.n} tokens, 码 {args.bits} 位, {args.clusters} 个语义簇, "
          f"关系维度 r={args.dim}")

    hd = (codes[:, None, :] != codes[None, :, :]).sum(-1).float()
    iu = torch.triu_indices(args.n, args.n, offset=1)
    same = (labels[:, None] == labels[None, :])[iu[0], iu[1]]
    print(f"  簇内平均 Hamming 距离 = {hd[iu[0], iu[1]][same].mean():.2f}")
    print(f"  簇间平均 Hamming 距离 = {hd[iu[0], iu[1]][~same].mean():.2f}")

    results = {}

    # ---------- ① 码导出 ----------
    u_code = codes_to_u(codes, args.dim, scale=1.0)
    m1 = metric_correlation(u_code, codes, labels)
    results["code_derived"] = m1
    print(f"\n  【① 码导出 u_t = 码 @ R】")
    print(f"    corr(码Hamming, 关系分数) = {m1['corr_hamming_vs_relation']:+.4f}")
    print(f"    簇内 {m1['within_cluster']:+.4f}  vs  簇间 {m1['between_cluster']:+.4f}"
          f"   分离度 {m1['separation']:+.4f}")
    print(f"    最近邻簇一致率 = {m1['nn_cluster_acc']*100:.1f}%  "
          f"（随机基线 {100/args.clusters:.1f}%）")

    # ---------- ② 随机初始化 ----------
    g = torch.Generator().manual_seed(42)
    u_rand = torch.randn(args.n, args.dim, generator=g) / (args.dim ** 0.5)
    m2 = metric_correlation(u_rand, codes, labels)
    results["random_init"] = m2
    print(f"\n  【② 随机初始化（当前默认）】")
    print(f"    corr(码Hamming, 关系分数) = {m2['corr_hamming_vs_relation']:+.4f}")
    print(f"    簇内 {m2['within_cluster']:+.4f}  vs  簇间 {m2['between_cluster']:+.4f}"
          f"   分离度 {m2['separation']:+.4f}")
    print(f"    最近邻簇一致率 = {m2['nn_cluster_acc']*100:.1f}%")

    # ---------- ③ 训练能否从随机学到簇结构 ----------
    print(f"\n  【③ 从随机初始化出发，训练 {args.steps} 步能否学到簇结构】")
    print("    目标：让关系分数复现码的几何（等价于让 u 学成码的线性像）")
    u = u_rand.clone().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=0.05)
    target = (codes.float() * 2 - 1) @ (codes.float() * 2 - 1).T      # (n,n) 码内积
    tgt_n = target / target.abs().max()
    for step in range(args.steps):
        opt.zero_grad()
        pred = (u @ u.T)
        pred_n = pred / (pred.abs().max() + 1e-9)
        loss = F.mse_loss(pred_n, tgt_n)
        loss.backward()
        opt.step()
        if (step + 1) % max(args.steps // 4, 1) == 0:
            mm = metric_correlation(u.detach(), codes, labels)
            print(f"      step {step+1:>4}: loss={loss.item():.5f}  "
                  f"corr={mm['corr_hamming_vs_relation']:+.4f}  "
                  f"簇一致率={mm['nn_cluster_acc']*100:.1f}%")
    m3 = metric_correlation(u.detach(), codes, labels)
    results["trained_from_random"] = m3

    # ---------- ④ 码导出 + 训练（对照：起点好是否收敛更快）----------
    print(f"\n  【④ 码导出再训练 {args.steps} 步（对照起点的影响）】")
    u2 = u_code.clone().requires_grad_(True)
    opt2 = torch.optim.Adam([u2], lr=0.05)
    hist = []
    for step in range(args.steps):
        opt2.zero_grad()
        pred = (u2 @ u2.T)
        pred_n = pred / (pred.abs().max() + 1e-9)
        loss = F.mse_loss(pred_n, tgt_n)
        loss.backward()
        opt2.step()
        if (step + 1) % max(args.steps // 4, 1) == 0:
            mm = metric_correlation(u2.detach(), codes, labels)
            hist.append((step + 1, loss.item(),
                         mm["corr_hamming_vs_relation"], mm["nn_cluster_acc"]))
            print(f"      step {step+1:>4}: loss={loss.item():.5f}  "
                  f"corr={mm['corr_hamming_vs_relation']:+.4f}  "
                  f"簇一致率={mm['nn_cluster_acc']*100:.1f}%")
    m4 = metric_correlation(u2.detach(), codes, labels)
    results["trained_from_code"] = m4

    # ---------- 结论 ----------
    print("\n" + "=" * 76)
    print("结论")
    print("=" * 76)
    print(f"  {'配置':<24}{'corr':>10}{'簇内-簇间':>12}{'最近邻簇一致率':>16}")
    for name, m in [("① 码导出(未训练)", m1), ("② 随机(未训练)", m2),
                    ("③ 随机+训练", m3), ("④ 码导出+训练", m4)]:
        print(f"  {name:<24}{m['corr_hamming_vs_relation']:>+10.4f}"
              f"{m['separation']:>+12.4f}{m['nn_cluster_acc']*100:>15.1f}%")
    print()
    ok1 = m1["corr_hamming_vs_relation"] < -0.5
    ok2 = abs(m2["corr_hamming_vs_relation"]) < 0.3
    print(f"  码导出确实恢复几何: {'✅ 成立' if ok1 else '❌ 不成立'}"
          f"  (corr {m1['corr_hamming_vs_relation']:+.4f})")
    print(f"  随机初始化确实无关: {'✅ 成立' if ok2 else '❌ 不成立'}"
          f"  (corr {m2['corr_hamming_vs_relation']:+.4f})")
    if m3["nn_cluster_acc"] > m2["nn_cluster_acc"] + 0.2:
        print(f"  训练能从随机学到簇结构: ✅ "
              f"({m2['nn_cluster_acc']*100:.1f}% -> {m3['nn_cluster_acc']*100:.1f}%)")
    else:
        print(f"  训练从随机学到簇结构: ⚠️ 效果有限 "
              f"({m2['nn_cluster_acc']*100:.1f}% -> {m3['nn_cluster_acc']*100:.1f}%)")
    print()
    print("  >> 若①成立而②不成立，说明「码作为索引」是可行的设计：")
    print("     关系几何可以【结构性】地由码几何保证，不必靠训练去学。")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
