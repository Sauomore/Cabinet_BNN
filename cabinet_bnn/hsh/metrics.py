# -*- coding: utf-8 -*-
"""
评测指标：Hamming 几何、Recall@K、桶分布、码本健康度。

对应 HSH-64 论文 §7 的指标定义，以及路线文档 M1/M3.5 的验收项：
  · Recall@K          —— 论文式(35)，分母为 K
  · 正/负样本对距离分布 —— 论文图 2
  · (feat, sim) 桶分布  —— 路线文档 M3.5（决定准完美哈希是否成立）
  · 码本健康度          —— 唯一码数、每 bit 翻转率（检测码塌缩）
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .codec import (
    ABS_SLOTS,
    bucket_stats,
    hamming_matrix_blocked,
    hamming_popcount,
)


# ---------------------------------------------------------------- Recall@K

def truth_topk(sim: np.ndarray, k: int) -> np.ndarray:
    """教师相似度矩阵 → 真实 top-K 邻居索引 (N, k)，排除自身。"""
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    return np.argpartition(-s, k - 1, axis=1)[:, :k]


def recall_at_k(codes: np.ndarray, sim: np.ndarray, k: int) -> float:
    """论文式(35)：Recall@K = (1/N) Σ |R_i ∩ G_i| / K。"""
    n = codes.shape[0]
    k = min(k, n - 1)
    truth = truth_topk(sim, k)

    d = hamming_matrix_blocked(codes)
    np.fill_diagonal(d, 10**9)
    ham = np.argpartition(d, k - 1, axis=1)[:, :k]

    hits = 0
    for i in range(n):
        hits += len(set(truth[i].tolist()) & set(ham[i].tolist()))
    return hits / (n * k)


def recall_curve(codes: np.ndarray, sim: np.ndarray, ks: tuple[int, ...] = (1, 5, 10, 20)) -> dict[int, float]:
    """一次算多个 K（复用距离矩阵）。

    实现要点：必须先完整排序再取前 K，否则 Recall@K 不满足单调性。
    np.argpartition 返回的组内顺序是任意的，直接对每个 K 单独调用会导致
    选中的集合互不包含（出现 Recall@20 < Recall@10 的假象）。
    """
    n = codes.shape[0]
    d = hamming_matrix_blocked(codes)
    np.fill_diagonal(d, 10**9)

    # 完整排序后的候选顺序（一次，供所有 K 复用）—— 保证嵌套性
    ham_order = np.argsort(d, axis=1, kind="stable")

    # 真值也完整排序，同样保证嵌套性
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    truth_order = np.argsort(-s, axis=1, kind="stable")

    out = {}
    for k in ks:
        kk = min(k, n - 1)
        hits = 0
        for i in range(n):
            truth = set(truth_order[i, :kk].tolist())
            ham = set(ham_order[i, :kk].tolist())
            hits += len(truth & ham)
        out[k] = hits / (n * kk)
    return out


def recall_curve_paper(bits: np.ndarray, sim: np.ndarray,
                       ks: tuple[int, ...] = (1, 5, 10, 20)) -> dict[int, float]:
    """参考实现口径的 Recall@K：分母为 |P_i|（每个查询的真实正样本个数）。

    与 recall_curve（分母 K，论文式 35）的差别：
        · 分母 K     —— K 增大时，多召回的无关项会拉低比值，故【不保证随 K 单调】
        · 分母 |P_i| —— 等价于「真实邻居被找回的比例」，随 K 单调不减

    两个口径在 K = |P_i| 时数值相同。报告时建议同时给出，避免误读。
    """
    from .post_optimize import hamming_matrix, recall_at_k_from_ranking
    n = bits.shape[0]
    d = hamming_matrix(bits)
    out = {}
    for k in ks:
        kk = min(k, n - 1)
        pos_idx = np.argsort(-sim, axis=1)[:, 1 : kk + 1]
        out[k] = recall_at_k_from_ranking(pos_idx, d, kk)
    return out


# ---------------------------------------------------------------- 距离几何

def distance_stats(codes: np.ndarray, sim: np.ndarray, pos_k: int = 10) -> dict:
    """正/负样本对的 Hamming 距离分布（论文图 2 的数值版）。

    随机基线应为 M/2（M=52 时为 26，σ=3.6056）—— 这是判定码是否真的学到东西的标尺。
    """
    n = codes.shape[0]
    pos_k = min(pos_k, n - 1)
    pos_idx = truth_topk(sim, pos_k)

    d = hamming_matrix_blocked(codes)
    np.fill_diagonal(d, 10**9)

    pos_vals = np.concatenate([d[i, pos_idx[i]] for i in range(n)]).astype(np.float32)

    # 负样本：随机采样以控制内存
    rng = np.random.default_rng(0)
    sample = min(200, n)
    neg_vals = []
    for i in rng.choice(n, size=sample, replace=False):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[pos_idx[i]] = False
        idx = np.flatnonzero(mask)
        if idx.size > 200:
            idx = rng.choice(idx, size=200, replace=False)
        neg_vals.append(d[i, idx])
    neg_vals = np.concatenate(neg_vals).astype(np.float32)

    m = codes.dtype.itemsize * 8 if codes.dtype.kind == "u" else 52
    n_bins = int(m) + 2
    return {
        "pos_mean": float(pos_vals.mean()),
        "pos_std": float(pos_vals.std()),
        "neg_mean": float(neg_vals.mean()),
        "neg_std": float(neg_vals.std()),
        "separation": float(neg_vals.mean() - pos_vals.mean()),
        "random_baseline_mean": m / 2.0,
        "random_baseline_std": float(np.sqrt(m / 4.0)),
        # 直方图用于复现论文图 2
        "pos_hist": np.histogram(pos_vals, bins=range(0, n_bins))[0].tolist(),
        "neg_hist": np.histogram(neg_vals, bins=range(0, n_bins))[0].tolist(),
        "n_pos_samples": int(pos_vals.size),
        "n_neg_samples": int(neg_vals.size),
    }


# ---------------------------------------------------------------- 码本健康度

def codebook_health(sim_bits: np.ndarray) -> dict:
    """码本健康度：检测码塌缩与 bit 偏置。

    sim_bits: (N, 52) 的 0/1 矩阵。
    """
    n, m = sim_bits.shape
    bit_mean = sim_bits.mean(axis=0)                       # 每个 bit 的 1 比例
    packed = np.packbits(sim_bits.astype(np.uint8), axis=1, bitorder="little")
    if m <= 52:
        uniq = len({bytes(row[:7]) for row in packed})
    else:
        uniq = len({bytes(row) for row in packed})
    # 位平衡损失（论文式 14）：Σ (b̄_j − 0.5)² / M
    balance_loss = float(((bit_mean - 0.5) ** 2).mean())
    return {
        "n_bits": int(m),
        "unique_codes": int(uniq),
        "unique_ratio": float(uniq / n),
        "bit_mean_min": float(bit_mean.min()),
        "bit_mean_max": float(bit_mean.max()),
        "bit_mean_std": float(bit_mean.std()),
        "balance_loss": balance_loss,
        "collapsed": bool(uniq < max(2, n * 0.5)),
    }


# ---------------------------------------------------------------- 汇总

@dataclass
class EvalReport:
    """一次完整评测的结构化结果。"""
    n_items: int
    n_bits: int
    recall: dict[int, float]
    distance: dict
    health: dict
    buckets: dict

    def to_dict(self) -> dict:
        d = asdict(self)
        d["recall"] = {str(k): v for k, v in self.recall.items()}
        return d

    def pretty(self) -> str:
        lines = []
        lines.append(f"  词表规模      : {self.n_items}")
        lines.append(f"  码长          : {self.n_bits} bit")
        lines.append(f"  Recall@1/5/10/20: " + " / ".join(
            f"{self.recall.get(k, float('nan')):.4f}" for k in (1, 5, 10, 20)))
        d = self.distance
        lines.append(f"  正样本对距离  : {d['pos_mean']:.2f} ± {d['pos_std']:.2f}")
        lines.append(f"  负样本对距离  : {d['neg_mean']:.2f} ± {d['neg_std']:.2f}")
        lines.append(f"  分离度        : {d['separation']:.2f} bit")
        lines.append(f"  随机基线      : {d['random_baseline_mean']:.2f} ± {d['random_baseline_std']:.2f}")
        h = self.health
        lines.append(f"  唯一码        : {h['unique_codes']} / {self.n_items} "
                     f"({h['unique_ratio']*100:.1f}%){'  [WARN] 码塌缩' if h['collapsed'] else ''}")
        lines.append(f"  balance loss  : {h['balance_loss']:.5f}")
        b = self.buckets
        lines.append(f"  语义桶数      : {b['n_buckets']}   最大桶: {b['max_bucket']}   "
                     f"溢出桶(>256): {b['buckets_over_abs_slots']}")
        lines.append(f"  准完美哈希    : {'[OK] 可行' if b['perfect_hash_possible'] else '[FAIL] 不可行'}")
        return "\n".join(lines)


def evaluate_codes(
    codes64: np.ndarray,
    feat: np.ndarray,
    sim: np.ndarray,
    sim_bits: np.ndarray,
    teacher_sim: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10, 20),
) -> EvalReport:
    """一站式评测：Recall + 距离几何 + 码本健康度 + 桶分布。"""
    bs = bucket_stats(feat, sim)
    return EvalReport(
        n_items=int(codes64.shape[0]),
        n_bits=int(sim_bits.shape[1]),
        recall=recall_curve(codes64, teacher_sim, ks),
        distance=distance_stats(codes64, teacher_sim),
        health=codebook_health(sim_bits),
        buckets={
            "n_buckets": bs.n_buckets,
            "max_bucket": bs.max_bucket,
            "mean_bucket": bs.mean_bucket,
            "buckets_over_abs_slots": bs.buckets_over_abs_slots,
            "overflow_items": bs.overflow_items,
            "perfect_hash_possible": bs.perfect_hash_possible,
            "histogram": bs.histogram,
        },
    )
