# -*- coding: utf-8 -*-
"""
HSH-64 编码：位布局 / popcount / 桶划分 / 准完美哈希种子搜索。

与 Rust 端逐位兼容（src/hsh64.rs、src/perfect_hash.rs）：

    [63:60] feat (4 bit)  ｜  [59:8] sim (52 bit)  ｜  [7:0] abs (8 bit)

    raw = (feat << 60) | ((sim & MASK52) << 8) | abs
    d_H(a, b) = popcount(a ^ b)

关键架构事实（见路线文档 §3.3）：
    语义桶的地址是 (feat, sim)，不是 (feat, abs)。
    桶数 = 2^(4+52) = 2^56；每桶可用 abs 槽位 = 256。
    abs 的准完美哈希【仅在桶内 token 数 <= 256 时】才成立。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------- 位布局常量（与 Rust 一致）----------------
FEAT_BITS = 4
SIM_BITS = 52
ABS_BITS = 8

SIM_SHIFT = ABS_BITS              # 8
FEAT_SHIFT = SIM_SHIFT + SIM_BITS  # 60

MAX_FEAT = (1 << FEAT_BITS) - 1   # 15
MAX_SIM = (1 << SIM_BITS) - 1     # 2^52 - 1
MAX_ABS = (1 << ABS_BITS) - 1     # 255

#: abs 槽位数。桶内 token 数超过它，准完美哈希必然冲突。
ABS_SLOTS = 1 << ABS_BITS         # 256

# ---------------- popcount 查找表 ----------------
_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
_VIEW = "<u8" if np.little_endian else ">u8"


def pack_codes(feat: np.ndarray, sim: np.ndarray, abs_: np.ndarray) -> np.ndarray:
    """把三个字段打包成 u64 数组。"""
    feat = np.asarray(feat, dtype=np.uint64) & np.uint64(MAX_FEAT)
    sim = np.asarray(sim, dtype=np.uint64) & np.uint64(MAX_SIM)
    abs_ = np.asarray(abs_, dtype=np.uint64) & np.uint64(MAX_ABS)
    return (feat << np.uint64(FEAT_SHIFT)) | (sim << np.uint64(SIM_SHIFT)) | abs_


def unpack_codes(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """拆出 (feat, sim, abs)，dtype 分别为 uint8 / uint64 / uint8。"""
    codes = np.asarray(codes, dtype=np.uint64)
    feat = ((codes >> np.uint64(FEAT_SHIFT)) & np.uint64(MAX_FEAT)).astype(np.uint8)
    sim = (codes >> np.uint64(SIM_SHIFT)) & np.uint64(MAX_SIM)
    abs_ = (codes & np.uint64(MAX_ABS)).astype(np.uint8)
    return feat, sim, abs_


def hamming_popcount(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐元素 popcount(a XOR b)（u64 数组）。"""
    x = np.bitwise_xor(np.asarray(a, dtype=np.uint64), np.asarray(b, dtype=np.uint64))
    return _POPCOUNT8[x.view(np.uint8).reshape(-1, 8)].sum(axis=1).astype(np.int32)


def hamming_matrix(codes: np.ndarray) -> np.ndarray:
    """N×N 两两 Hamming 距离矩阵。

    对 N=3109 会分配约 38 MB(int32)，可接受。更大规模请用分块版本。
    """
    codes = np.asarray(codes, dtype=np.uint64)
    n = codes.shape[0]
    out = np.empty((n, n), dtype=np.int32)
    for i in range(n):
        out[i] = hamming_popcount(np.full(n, codes[i], dtype=np.uint64), codes)
    return out


def hamming_matrix_blocked(codes: np.ndarray, block: int = 512) -> np.ndarray:
    """分块计算 N×N Hamming 距离矩阵（内存友好）。"""
    codes = np.asarray(codes, dtype=np.uint64)
    n = codes.shape[0]
    out = np.empty((n, n), dtype=np.int32)
    bytes_view = codes.view(np.uint8).reshape(n, 8)
    for s in range(0, n, block):
        e = min(s + block, n)
        xor = bytes_view[s:e, None, :] ^ bytes_view[None, :, :]   # (b, n, 8)
        out[s:e] = _POPCOUNT8[xor].sum(axis=2).astype(np.int32)
    return out


# ---------------- 准完美哈希（abs）----------------

def bkdr_hash(word: str) -> int:
    """BKDR 字符串哈希（与 Rust 实现一致：seed=131，u64 回绕）。"""
    h = 0
    for byte in word.encode("utf-8"):
        h = (h * 131 + byte) & 0xFFFFFFFFFFFFFFFF
    return h


def compute_abs(word: str, seed: int) -> int:
    """abs = (BKDR(word) XOR seed) mod 256。"""
    return (bkdr_hash(word) ^ (seed & 0xFF)) & 0xFF


@dataclass
class SeedSearchResult:
    """单桶的种子搜索结果。"""
    seed: int
    abs_values: list[int]
    collisions: int
    perfect: bool


def search_seed(words: list[str]) -> SeedSearchResult:
    """在 s ∈ [0,255] 中搜索使桶内 abs 互不相同的种子。

    这是参考实现 src/perfect_hash.rs::search_seed 的等价 Python 版本。
    若所有 256 个种子都无法做到无冲突（即桶内 token 数 > 256），
    则退化为「冲突最少」的种子——此时准完美哈希失效。
    """
    n = len(words)
    if n == 0:
        return SeedSearchResult(0, [], 0, True)

    hashes = np.array([bkdr_hash(w) for w in words], dtype=np.uint64)

    # 先用 numpy 快速筛：对全部 256 个种子一次性算冲突数
    seeds = np.arange(256, dtype=np.uint64)
    # abs[seed, i] = (h_i XOR seed) & 0xFF
    abs_all = (hashes[None, :] ^ seeds[:, None]) & np.uint64(0xFF)   # (256, n)

    best_seed, best_collisions = 0, None
    for s in range(256):
        vals = abs_all[s]
        uniq = np.unique(vals).size
        coll = n - uniq
        if best_collisions is None or coll < best_collisions:
            best_seed, best_collisions = s, coll
            if coll == 0:
                break

    abs_values = [int(v) for v in abs_all[best_seed]]
    return SeedSearchResult(best_seed, abs_values, int(best_collisions), best_collisions == 0)


# ---------------- 桶分布分析（MVP 核心验收项）----------------

@dataclass
class BucketStats:
    """(feat, sim) 桶大小分布统计。"""
    n_items: int
    n_buckets: int
    max_bucket: int
    mean_bucket: float
    buckets_over_abs_slots: int      # > 256 的桶数 → 准完美哈希失效
    overflow_items: int              # 落在溢出桶里的 token 总数
    histogram: dict[int, int]        # 桶大小 → 桶数量

    @property
    def perfect_hash_possible(self) -> bool:
        return self.buckets_over_abs_slots == 0

    def summary(self) -> str:
        lines = [
            f"  词表规模 N          = {self.n_items}",
            f"  语义桶数            = {self.n_buckets}",
            f"  最大桶              = {self.max_bucket}",
            f"  平均桶              = {self.mean_bucket:.3f}",
            f"  每桶 abs 槽位上限   = {ABS_SLOTS}",
            f"  超过上限的桶数      = {self.buckets_over_abs_slots}",
            f"  落在溢出桶的 token  = {self.overflow_items} ({self.overflow_items / max(self.n_items,1) * 100:.2f}%)",
            f"  准完美哈希可行      = {'✅ 是' if self.perfect_hash_possible else '❌ 否'}",
        ]
        return "\n".join(lines)


def bucket_stats(feat: np.ndarray, sim: np.ndarray) -> BucketStats:
    """统计 (feat, sim) 桶大小分布。

    这是路线文档 M3.5 的直接测量：若最大桶 > 256，
    则 abs 的准完美哈希失效，架构需要调整。
    """
    feat = np.asarray(feat)
    sim = np.asarray(sim)
    # 把 (feat, sim) 组合成单一键
    keys = feat.astype(np.uint64) * np.uint64(1 << SIM_BITS) + sim.astype(np.uint64)
    uniq, counts = np.unique(keys, return_counts=True)

    over = counts > ABS_SLOTS
    hist: dict[int, int] = {}
    for c in counts:
        hist[int(c)] = hist.get(int(c), 0) + 1

    return BucketStats(
        n_items=int(feat.shape[0]),
        n_buckets=int(uniq.size),
        max_bucket=int(counts.max()) if counts.size else 0,
        mean_bucket=float(counts.mean()) if counts.size else 0.0,
        buckets_over_abs_slots=int(over.sum()),
        overflow_items=int(counts[over].sum()) if over.any() else 0,
        histogram=dict(sorted(hist.items())),
    )


# ---------------- 评测指标 ----------------

def recall_at_k(
    codes: np.ndarray,
    truth_sim: np.ndarray,
    k: int,
    exclude_self: bool = True,
) -> float:
    """纯 Hamming 空间的 Recall@K。

    Args:
        codes: (N,) u64 码
        truth_sim: (N, N) 真实相似度矩阵（教师余弦）
        k: top-K

    Returns:
        Recall@K 平均值
    """
    codes = np.asarray(codes, dtype=np.uint64)
    n = codes.shape[0]
    k = min(k, n - 1 if exclude_self else n)

    # 真实 top-K 邻居（排除自身）
    t = truth_sim.copy()
    if exclude_self:
        np.fill_diagonal(t, -np.inf)
    truth_topk = np.argpartition(-t, k - 1, axis=1)[:, :k]

    # Hamming top-K
    d = hamming_matrix_blocked(codes)
    if exclude_self:
        np.fill_diagonal(d, 10**9)
    ham_topk = np.argpartition(d, k - 1, axis=1)[:, :k]

    hits = 0
    for i in range(n):
        hits += len(set(truth_topk[i].tolist()) & set(ham_topk[i].tolist()))
    return hits / (n * k)


def distance_distribution(codes: np.ndarray, truth_sim: np.ndarray, pos_k: int = 10) -> dict:
    """正样本对 / 负样本对的 Hamming 距离分布（对应论文图 2）。"""
    codes = np.asarray(codes, dtype=np.uint64)
    n = codes.shape[0]
    t = truth_sim.copy()
    np.fill_diagonal(t, -np.inf)
    pos_idx = np.argpartition(-t, pos_k - 1, axis=1)[:, :pos_k]

    d = hamming_matrix_blocked(codes)
    pos_vals, neg_vals = [], []
    for i in range(n):
        pos_set = set(pos_idx[i].tolist())
        pos_vals.extend(d[i, list(pos_set)].tolist())
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[list(pos_set)] = False
        neg_vals.extend(d[i, mask].tolist())

    pos_vals = np.array(pos_vals, dtype=np.float32)
    neg_vals = np.array(neg_vals, dtype=np.float32)
    return {
        "pos_mean": float(pos_vals.mean()),
        "pos_std": float(pos_vals.std()),
        "neg_mean": float(neg_vals.mean()),
        "neg_std": float(neg_vals.std()),
        "separation": float(neg_vals.mean() - pos_vals.mean()),
        "pos_hist": np.histogram(pos_vals, bins=range(0, 54))[0].tolist(),
        "neg_hist": np.histogram(neg_vals, bins=range(0, 54))[0].tolist(),
    }
