# -*- coding: utf-8 -*-
"""
Cabinet-BNN 架构图与实验图表生成。

产出（全部 200 dpi PNG，可直接嵌入报告）：
    fig01_architecture.png   完整架构图（离线训练 / 在线推理 / 热更改三部分）
    fig02_bitlayout.png      128 位权重码位布局
    fig03_weight_gen.png     权重生成机制：s = mask ⊙ p，W_t = Σ s_j B_j
    fig04_hamming.png        Hamming 距离分布（后处理前/后 + 随机基线）
    fig05_continual.png      持续学习：准确率矩阵热图 + 遗忘量对比
    fig06_ablation.png       M3.6 等参数消融对比

用法：
    python scripts/05_make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "docs" / "figures"

# 中文字体
for fam in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DengXian"]:
    try:
        matplotlib.rcParams["font.sans-serif"] = [fam]
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

# 配色
C_HASH = "#2E6FB7"
C_PARAM = "#D9722B"
C_FREEZE = "#8C8C92"
C_OK = "#2E8B57"
C_BAD = "#B03A2E"
C_BG = "#F7F8FA"


# =====================================================================
# 图 1：完整架构图
# =====================================================================

def box(ax, x, y, w, h, text, fc, ec="none", fs=9, tc="black", lw=1.4, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.45)


def arrow(ax, p0, p1, color="#444444", lw=1.6, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                                 color=color, linewidth=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def fig01_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(15.2, 9.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="white", zorder=0))

    # ---------- 三个分区 ----------
    ax.add_patch(FancyBboxPatch((0.015, 0.545), 0.97, 0.435,
                                boxstyle="round,pad=0.008", facecolor="#EAF2FB",
                                edgecolor=C_HASH, linewidth=1.6, zorder=1))
    ax.text(0.030, 0.958, "① 离线：编码与训练（Python / PyTorch）",
            fontsize=12, fontweight="bold", color=C_HASH, va="top")

    ax.add_patch(FancyBboxPatch((0.015, 0.055), 0.97, 0.455,
                                boxstyle="round,pad=0.008", facecolor="#FDF1E7",
                                edgecolor=C_PARAM, linewidth=1.6, zorder=1))
    ax.text(0.030, 0.485, "② 在线：推理（Rust 引擎）",
            fontsize=12, fontweight="bold", color=C_PARAM, va="top")

    # ---------- ① 离线 ----------
    box(ax, 0.045, 0.815, 0.145, 0.095,
        "中文语料\n(vocab_3109)", "#FFFFFF", C_FREEZE, 9)
    box(ax, 0.225, 0.815, 0.155, 0.095,
        "bge-large-zh\n(教师, 冻结)", "#FFFFFF", C_FREEZE, 9)
    box(ax, 0.415, 0.815, 0.175, 0.095,
        "Deep Hash 投影头\n512 → H → 52\n(STE + 多目标损失)",
        "#DCE9F7", C_HASH, 8.5)
    box(ax, 0.625, 0.815, 0.155, 0.095,
        "召回导向\n贪心比特翻转\n(后处理)", "#DCE9F7", C_HASH, 8.5)
    box(ax, 0.815, 0.815, 0.155, 0.095,
        "sim 码\n3109 × 52 bit\nRecall@10 = 0.7426", "#CFE3F5", C_HASH, 8.5, bold=True)

    arrow(ax, (0.190, 0.862), (0.225, 0.862))
    arrow(ax, (0.380, 0.862), (0.415, 0.862))
    arrow(ax, (0.590, 0.862), (0.625, 0.862))
    arrow(ax, (0.780, 0.862), (0.815, 0.862))

    box(ax, 0.045, 0.615, 0.300, 0.105,
        "共享基矩阵  B_j = U_j V_j^T   (k=8, rank=32)\n"
        "U_j ∈ R^{r×d_out},  V_j ∈ R^{r×d_in}",
        "#FFFFFF", C_HASH, 8.5)
    box(ax, 0.390, 0.615, 0.265, 0.105,
        "param 段（可学）\nper-token 影子参数 θ ∈ R^{V×64}",
        "#FBE3D0", C_PARAM, 9)
    box(ax, 0.700, 0.615, 0.270, 0.105,
        "码表 · 每个域一份（热插拔）\nCodeTable[domain][token]",
        "#FBE3D0", C_PARAM, 9, bold=True)

    arrow(ax, (0.860, 0.815), (0.860, 0.720), color=C_HASH, ls="--")
    arrow(ax, (0.522, 0.720), (0.522, 0.815), color=C_PARAM, ls="--", rad=-0.25)

    # ---------- ② 在线 ----------
    box(ax, 0.045, 0.330, 0.120, 0.080, "token id", "#FFFFFF", C_FREEZE, 9.5)
    box(ax, 0.200, 0.330, 0.150, 0.080, "CodeTable\n→ 128 位码", "#FBE3D0", C_PARAM, 9)

    box(ax, 0.045, 0.195, 0.150, 0.105,
        "hash 段 (64 bit)\nfeat4+sim52+abs8", "#DCE9F7", C_HASH, 8.5)
    box(ax, 0.220, 0.195, 0.150, 0.105,
        "param 段 (64 bit)\nper-token 参数", "#FBE3D0", C_PARAM, 8.5)

    box(ax, 0.415, 0.195, 0.215, 0.105,
        "符号合成\ns = mask(hash) ⊙ p\n∈ {−1,+1}^k", "#E8E4F5", "#6A5ACD", 8.5, bold=True)
    box(ax, 0.665, 0.195, 0.155, 0.105,
        "权重生成\nW_t = Σ_j s_j·B_j", "#E8E4F5", "#6A5ACD", 8.5, bold=True)
    box(ax, 0.855, 0.195, 0.125, 0.105,
        "二值激活\nsign(W_t x)\n× scale", "#D6EFDC", C_OK, 8.5)
    arrow(ax, (0.760, 0.2475), (0.855, 0.2475), color="#6A5ACD", lw=2.0)

    arrow(ax, (0.165, 0.370), (0.200, 0.370))
    arrow(ax, (0.275, 0.330), (0.275, 0.300))
    arrow(ax, (0.120, 0.330), (0.120, 0.300))
    arrow(ax, (0.120, 0.300), (0.120, 0.2475), ls="--", color=C_FREEZE)
    arrow(ax, (0.120, 0.2475), (0.195, 0.2475), ls="--", color=C_FREEZE)
    arrow(ax, (0.295, 0.300), (0.295, 0.2475), color=C_PARAM, lw=1.8)
    arrow(ax, (0.370, 0.2475), (0.415, 0.2475), color=C_PARAM, lw=1.8)

    # 后端检索
    box(ax, 0.415, 0.095, 0.215, 0.075,
        "Hamming 检索 / MIH\npopcount(a XOR b)", "#DCE9F7", C_HASH, 8.5)
    box(ax, 0.665, 0.095, 0.315, 0.075,
        "二值联想记忆后端（16 B/条：key u64 + value u64）",
        "#DCE9F7", C_HASH, 8.5)
    arrow(ax, (0.120, 0.195), (0.120, 0.1325), color=C_HASH, ls="--")
    arrow(ax, (0.120, 0.1325), (0.415, 0.1325), color=C_HASH, ls="--")
    arrow(ax, (0.630, 0.1325), (0.665, 0.1325), color=C_HASH)
    ax.text(0.240, 0.150, "仅 hash 段参与检索", fontsize=8, color=C_HASH, style="italic")

    # ---------- ③ 热更改（独立带，避免压住数据流框）----------
    ax.add_patch(FancyBboxPatch((0.045, 0.048), 0.935, 0.038,
                                boxstyle="round,pad=0.006", facecolor="#FFF7D6",
                                edgecolor="#C9A227", linewidth=1.8,
                                linestyle="--", zorder=2))
    ax.text(0.512, 0.067,
            "③ 热更改（核心卖点）：翻转 1 bit → W_t 立即改变   ·   "
            "无锁 COW + 原子指针交换   ·   回滚 = 存回旧 Arc，O(1)   ·   "
            "每域一份码表 → 遗忘量 = 0",
            ha="center", va="center", fontsize=8.8, color="#5C4A08",
            fontweight="bold", zorder=3)
    arrow(ax, (0.742, 0.086), (0.742, 0.195), color="#C9A227", lw=1.8, ls="--")

    ax.text(0.5, 0.012,
            "Cabinet-BNN 架构：64 位 HSH-64 检索码 + 64 位 per-token 参数段 = 128 位权重码；"
            "权重由共享基矩阵按码的符号模式组合而成",
            ha="center", fontsize=9.5, color="#333333", style="italic")

    fig.savefig(OUT / "fig01_architecture.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return OUT / "fig01_architecture.png"


# =====================================================================
# 图 2：位布局
# =====================================================================

def fig02_bitlayout() -> Path:
    fig, ax = plt.subplots(figsize=(14.5, 4.6))
    ax.set_xlim(0, 128); ax.set_ylim(0, 1); ax.axis("off")

    segs = [(0, 8, "abs\n8 bit", "#A8C8E8"),
            (8, 60, "sim\n52 bit", C_HASH),
            (60, 64, "feat\n4 bit", "#7FA8D4"),
            (64, 128, "param\n64 bit（不参与检索）", C_PARAM)]
    for x0, x1, label, color in segs:
        w = x1 - x0
        ax.add_patch(Rectangle((x0, 0.42), w, 0.34, facecolor=color,
                               edgecolor="white", linewidth=2))
        tc = "black" if color in ("#A8C8E8",) else "white"
        ax.text((x0 + x1) / 2, 0.59, label, ha="center", va="center",
                fontsize=11 if w > 12 else 8.5, color=tc, fontweight="bold",
                linespacing=1.35)

    ax.annotate("", xy=(0, 0.36), xytext=(64, 0.36),
                arrowprops=dict(arrowstyle="<->", color=C_HASH, lw=2))
    ax.text(32, 0.29, "hash 段 = 64 位：参与 popcount 检索，语义几何完整",
            ha="center", fontsize=10.5, color=C_HASH, fontweight="bold")

    ax.annotate("", xy=(64, 0.80), xytext=(128, 0.80),
                arrowprops=dict(arrowstyle="<->", color=C_PARAM, lw=2))
    ax.text(96, 0.87, "param 段：只用于权重生成",
            ha="center", fontsize=10.5, color=C_PARAM, fontweight="bold")

    ax.text(0, 0.12, "bit 63 … 56      55 … 4        3 … 0       63 … 0",
            fontsize=8, color="#666666")
    ax.text(0, 0.02,
            "u64 #1  [原始 HSH-64]                                              "
            "u64 #2  [本设计的扩展段]",
            fontsize=8.5, color="#666666")

    fig.savefig(OUT / "fig02_bitlayout.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return OUT / "fig02_bitlayout.png"


# =====================================================================
# 图 3：权重生成机制
# =====================================================================

def fig03_weight_gen() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2),
                             gridspec_kw={"width_ratios": [1.05, 1]})

    # --- 左：三路合成 ---
    ax = axes[0]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.02, 0.72, 0.30, 0.15, "hash 段\n(sim 位)", "#DCE9F7", C_HASH, 9.5)
    box(ax, 0.02, 0.44, 0.30, 0.15, "param 段\np ∈ {−1,+1}^P", "#FBE3D0", C_PARAM, 9.5)
    box(ax, 0.40, 0.58, 0.24, 0.15, "⊙\n逐元素乘", "#E8E4F5", "#6A5ACD", 10.5, bold=True)
    box(ax, 0.70, 0.58, 0.28, 0.15, "s ∈ {−1,+1}^k\n符号向量",
        "#E8E4F5", "#6A5ACD", 10, bold=True)
    arrow(ax, (0.32, 0.795), (0.40, 0.685), color=C_HASH)
    arrow(ax, (0.32, 0.515), (0.40, 0.625), color=C_PARAM)
    arrow(ax, (0.64, 0.655), (0.70, 0.655), color="#6A5ACD", lw=2)

    box(ax, 0.02, 0.20, 0.30, 0.16, "共享基矩阵\nB_j = U_j V_j^T  (j=1..k)",
        "#FFFFFF", C_HASH, 9.5)
    box(ax, 0.40, 0.19, 0.24, 0.18, "Σ_j s_j · B_j", "#E8E4F5", "#6A5ACD", 11, bold=True)
    box(ax, 0.70, 0.20, 0.28, 0.16, "W_t\n(d_out × d_in)", "#D6EFDC", C_OK, 11, bold=True)
    arrow(ax, (0.32, 0.28), (0.40, 0.28), color=C_HASH)
    arrow(ax, (0.82, 0.58), (0.82, 0.36), color="#6A5ACD", lw=2)
    arrow(ax, (0.64, 0.28), (0.70, 0.28), color="#6A5ACD", lw=2)

    ax.text(0.5, 0.075,
            "几何隔离（定理 1）：同桶 token 的权重差异\n"
            "$s(a)\\odot s(b) = p_a\\odot p_b$ —— 与共享 mask 无关",
            ha="center", fontsize=9.5, color="#333333", linespacing=1.5)

    # --- 右：Hamming 几何 → 权重几何 ---
    ax = axes[1]
    rng = np.random.default_rng(3)
    n, m = 26, 52
    base = rng.integers(0, 2, size=(n, m))
    # 制造一个"语义近邻"结构：后半数从第一个派生
    for i in range(n // 2, n):
        b = base[0].copy()
        flip = rng.choice(m, size=4, replace=False)
        b[flip] ^= 1
        base[i] = b
    D = (base[:, None, :] != base[None, :, :]).sum(-1)
    im = ax.imshow(D, cmap="viridis", interpolation="nearest")
    ax.set_title("Hamming 距离矩阵（模拟：后半为前者的近邻）",
                 fontsize=10.5, color="#333333")
    ax.set_xlabel("token 索引", fontsize=9.5)
    ax.set_ylabel("token 索引", fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("d_H", fontsize=9.5)
    ax.text(0.5, -0.22,
            "距离 b 位 => 权重恰好差 b 个符号翻转\n"
            "（码空间的 Hamming 几何 = 权重空间的 L1 几何）",
            transform=ax.transAxes, ha="center", fontsize=9.3,
            color="#333333", linespacing=1.5)

    fig.savefig(OUT / "fig03_weight_gen.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return OUT / "fig03_weight_gen.png"


# =====================================================================
# 图 4：Hamming 距离分布
# =====================================================================

def _recompute_distance(which: str) -> dict:
    """旧版报告缺少直方图时，从码文件现算（保证图始终可生成）。"""
    sys.path.insert(0, str(ROOT))
    from cabinet_bnn.data.embedding_cache import normalize_rows, read_cache
    from cabinet_bnn.hsh.deep_hash import bits_to_u64, load_deep_hash, project, quantize_bits
    from cabinet_bnn.hsh.metrics import distance_stats

    _, words, teacher = read_cache(RES / "bge_large_3109.cache")
    S = normalize_rows(teacher) @ normalize_rows(teacher).T

    if which == "post":
        from cabinet_bnn.hsh.post_optimize import load_override_cache
        _, bits = load_override_cache(RES / "sim_override_mvp1_h512_s42.bin")
    else:
        loaded = load_deep_hash(RES / "deep_hash_mvp1_h512_s42.bin")
        xc = teacher - teacher.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
        student = normalize_rows(xc @ vt[: min(loaded.dim, xc.shape[1])].T)
        bits = quantize_bits(project(loaded.model, loaded.mean, student))

    return distance_stats(bits_to_u64(bits), S)


def fig04_hamming() -> Path:
    p = RES / "mvp_report_mvp1.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    pre, post = d["pre_post"], d["post"]

    # 直方图可能未写入旧版报告 —— 缺失时现算
    dpre = pre["distance"] if "pos_hist" in pre["distance"] else _recompute_distance("pre")
    dpost = post["distance"] if "pos_hist" in post["distance"] else _recompute_distance("post")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9))

    # 左：分阶段 Recall
    ax = axes[0]
    ks = [1, 5, 10, 20]
    x = np.arange(len(ks)); w = 0.36
    rp = [pre["recall"][str(k)] for k in ks]
    ro = [post["recall"][str(k)] for k in ks]
    ax.bar(x - w / 2, rp, w, label="后处理前", color="#9CC0E0", edgecolor="#2E6FB7")
    ax.bar(x + w / 2, ro, w, label="后处理后", color=C_HASH, edgecolor="#1B4B80")
    for xi, v in zip(x - w / 2, rp):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5)
    for xi, v in zip(x + w / 2, ro):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.axhline(0.7404, color=C_BAD, ls="--", lw=1.4, label="论文 h512 基准 0.7404")
    ax.set_xticks(x); ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_ylabel("Recall@K（分母 K）", fontsize=10)
    ax.set_title("贪心比特翻转后处理的增益", fontsize=11.5, color="#333333")
    ax.legend(fontsize=8.8); ax.grid(axis="y", alpha=0.28)
    ax.set_ylim(0, 0.85)

    # 右：距离分布
    ax = axes[1]

    def norm(h):
        h = np.asarray(h, dtype=float)
        s = h.sum()
        return h / s if s > 0 else h

    ph, nh = norm(dpre["pos_hist"]), norm(dpre["neg_hist"])
    ph2, nh2 = norm(dpost["pos_hist"]), norm(dpost["neg_hist"])

    ax.plot(np.arange(len(ph)), ph, color="#9CC0E0", lw=1.8, ls="--", label="正样本对（前）")
    ax.plot(np.arange(len(ph2)), ph2, color=C_OK, lw=2.4, label="正样本对（后）")
    ax.plot(np.arange(len(nh)), nh, color="#E0A090", lw=1.8, ls="--", label="负样本对（前）")
    ax.plot(np.arange(len(nh2)), nh2, color=C_BAD, lw=2.4, label="负样本对（后）")
    ax.axvline(dpost["random_baseline_mean"], color="#888888", ls=":", lw=1.5,
               label=f"随机基线 μ={dpost['random_baseline_mean']:.0f}")

    p0, p1 = dpost["pos_mean"], dpost["neg_mean"]
    ymax = max(ph.max(), ph2.max(), nh.max(), nh2.max())
    ax.annotate("", xy=(p0, ymax * 0.84), xytext=(p1, ymax * 0.84),
                arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.5))
    ax.text((p0 + p1) / 2, ymax * 0.88, f"分离度 {dpost['separation']:.1f} bit",
            ha="center", fontsize=9.5, fontweight="bold", color="#333333")
    ax.set_xlabel("Hamming 距离（bit）", fontsize=10)
    ax.set_ylabel("归一化密度", fontsize=10)
    ax.set_title("距离分布：正样本对左移，负样本对基本不动", fontsize=11.5, color="#333333")
    ax.legend(fontsize=8.4); ax.grid(alpha=0.28)
    ax.set_xlim(0, 55)

    fig.savefig(OUT / "fig04_hamming.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return OUT / "fig04_hamming.png"


# =====================================================================
# 图 5：持续学习
# =====================================================================

def fig05_continual() -> Path:
    p = RES / "continual_eval.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    by = {r["method"]: r for r in d["results"]}
    order = ["shared_ft", "code_param", "code_swap", "code_full", "shared_joint"]
    order = [m for m in order if m in by]

    fig = plt.figure(figsize=(15.5, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.34)

    # --- 左：准确率矩阵热图（shared_ft） ---
    ax = fig.add_subplot(gs[0, 0])
    H = np.array(by["shared_ft"]["acc_history"], dtype=float)
    im = ax.imshow(H, cmap="YlGnBu", aspect="auto", vmin=0, vmax=0.4)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            ax.text(j, i, f"{H[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if H[i, j] > 0.22 else "#333333")
    ax.set_xlabel("域", fontsize=10); ax.set_ylabel("适配步 t", fontsize=10)
    ax.set_title("shared_ft 准确率矩阵\n（对角=本域，左下=旧域退化）",
                 fontsize=10.5, color="#333333")
    ax.set_xticks(range(H.shape[1])); ax.set_yticks(range(H.shape[0]))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    # --- 中：遗忘量 vs 本域均值 ---
    ax = fig.add_subplot(gs[0, 1])
    names, fg, own = [], [], []
    for m in order:
        a = by[m]["analysis"]
        names.append(m); fg.append(a["forgetting"]); own.append(a["own_domain_mean"])
    x = np.arange(len(names)); w = 0.38
    b1 = ax.bar(x - w / 2, own, w, label="本域均值", color=C_OK, edgecolor="#1E5C38")
    b2 = ax.bar(x + w / 2, fg, w, label="遗忘量", color=C_BAD, edgecolor="#7B2318")
    for xi, v in zip(x - w / 2, own):
        ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, fg):
        ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8.5)
    ax.set_ylabel("准确率 / 遗忘量", fontsize=10)
    ax.set_title("遗忘量 vs 本域精度", fontsize=10.5, color="#333333")
    ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.28)

    # --- 右：可逆性 ---
    ax = fig.add_subplot(gs[0, 2])
    rb = [(m, by[m].get("rollback")) for m in order if by[m].get("rollback")]
    labels, before, after = [], [], []
    for m, r in rb:
        labels.append(m); before.append(r["acc_d0_reference"])
        after.append(r["acc_d0_after_rollback"])
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w / 2, before, w, label="回滚前", color="#9CC0E0", edgecolor="#2E6FB7")
    ax.bar(x + w / 2, after, w, label="回滚后", color="#6A5ACD", edgecolor="#4A3A9D")
    for xi, v in zip(x - w / 2, before):
        ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, after):
        ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=8)
    for xi, (m, r) in zip(x, rb):
        tag = "精确" if r["exact_equality"] else "不精确"
        col = C_OK if r["exact_equality"] else C_BAD
        ax.text(xi, max(before + after) * 1.06, tag, ha="center",
                fontsize=9, color=col, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=8.5)
    ax.set_ylabel("域 0 准确率", fontsize=10)
    ax.set_title("可逆性：回滚到旧码后的域 0 精度", fontsize=10.5, color="#333333")
    ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.28)

    fig.savefig(OUT / "fig05_continual.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return OUT / "fig05_continual.png"


# =====================================================================
# 图 6：M3.6 等参数消融
# =====================================================================

def fig06_ablation() -> Path:
    d = json.loads((RES / "bnn_fair_small.json").read_text(encoding="utf-8"))
    by = {r["mode"]: r for r in d["results"]}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))

    # 左：val_loss
    ax = axes[0]
    names = ["shared", "code", "token_embed"]
    names = [n for n in names if n in by]
    loss = [by[n]["val_loss"] for n in names]
    cols = [C_OK if n == "shared" else (C_HASH if n == "code" else C_PARAM) for n in names]
    bars = ax.bar(names, loss, color=cols, edgecolor="#333333", width=0.58)
    for b, v in zip(bars, loss):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.3f}",
                ha="center", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("验证集 loss", fontsize=10)
    ax.set_title("等参数预算下的 LM 质量\n（越低越好）", fontsize=10.5, color="#333333")
    ax.grid(axis="y", alpha=0.28)

    # 中：参数量
    ax = axes[1]
    tot = [by[n]["params"]["total"] for n in names]
    per = [by[n]["params"]["per_token"] for n in names]
    shared_n = [t - p for t, p in zip(tot, per)]
    x = np.arange(len(names))
    ax.bar(x, shared_n, 0.58, label="共享参数", color="#9CC0E0", edgecolor="#2E6FB7")
    ax.bar(x, per, 0.58, bottom=shared_n, label="per-token 参数",
           color=C_PARAM, edgecolor="#8A4517")
    for xi, t in zip(x, tot):
        ax.text(xi, t * 1.02, f"{t/1000:.0f}k", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("参数量", fontsize=10)
    ax.set_title("参数量构成（总预算已对齐）", fontsize=10.5, color="#333333")
    ax.legend(fontsize=8.8); ax.grid(axis="y", alpha=0.28)

    # 右：训练稳定性
    ax = axes[2]
    for n, c in zip(names, cols):
        h = by[n]["history_loss"]
        ax.plot(range(1, len(h) + 1), h, color=c, lw=1.9, label=n)
    ax.set_xlabel("epoch", fontsize=10); ax.set_ylabel("验证 loss", fontsize=10)
    ax.set_title("训练稳定性\n（code 的震荡是 per-token 权重的固有问题）",
                 fontsize=10.5, color="#333333")
    ax.legend(fontsize=9); ax.grid(alpha=0.28)

    fig.savefig(OUT / "fig06_ablation.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return OUT / "fig06_ablation.png"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in (fig01_architecture, fig02_bitlayout, fig03_weight_gen,
               fig04_hamming, fig05_continual, fig06_ablation):
        try:
            p = fn()
            made.append(p)
            print(f"  [OK] {p.name}  ({p.stat().st_size:,} bytes)")
        except Exception as e:
            print(f"  [FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n共 {len(made)} 张图 -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
