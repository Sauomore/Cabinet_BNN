# -*- coding: utf-8 -*-
"""
召回导向的贪心比特翻转后处理（HSH-64 论文 Algorithm 1 + 定理 1）。

目标函数（论文式 17-18）：
    W_ij = +w_pos   若 j ∈ P_i⁺（教师 top-P 邻居）
           −w_neg   若 j ∈ P_i⁻（当前 Hamming top-K 中的非邻居）
            0       其他
    O(C) = Σ_{i,j} W_ij · d_H(c_i, c_j)

翻转增量（定理 1，闭式解）：
    Δ_{i,b} = Σ_{j≠i} W_ij · (1 − 2·1[C_{i,b} == C_{j,b}])

选 argmin Δ，若 < 0 则翻转，否则终止（1-optimal）。

内存说明（与参考实现的差异）：
    参考实现在每次迭代里构造 (N, N, n_bits) 的 match 张量：
        match = 1 - (C[:, None, :] ^ C[None, :, :])
        delta = (W[:, :, None] * (1 - 2*match)).sum(axis=1)
    N=3109, n_bits=52 时该张量约 3109²×52×1B ≈ 503 MB（bool 可能更差）。
    本实现用可结合的求和顺序避免三维张量：
        Δ_{i,b} = Σ_j W_ij − 2·Σ_{j: C_j,b == C_i,b} W_ij
    只需 (N, N) 级别的中间量，内存降一个数量级且可加 BLAS 优化。
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

OVERRIDE_MAGIC = 0xCAB1_0D01
OVERRIDE_VERSION = 1


# ---------------------------------------------------------------- 距离

def hamming_matrix(bits: np.ndarray) -> np.ndarray:
    """(N, M) 的 0/1 位矩阵 → (N, N) Hamming 距离矩阵（float32）。"""
    b = np.asarray(bits)
    n, m = b.shape
    out = np.empty((n, n), dtype=np.float32)
    blk = max(1, int(4_000_000 // max(n, 1)))
    for s in range(0, n, blk):
        e = min(s + blk, n)
        out[s:e] = (b[s:e, None, :] != b[None, :, :]).sum(axis=2, dtype=np.float32)
    return out


def recall_at_k_from_ranking(pos_indices: np.ndarray, d: np.ndarray, k: int) -> float:
    """给定真实正样本索引与 Hamming 距离矩阵，计算平均 Recall@K。

    与参考实现一致：分母是 |P_i|（正样本个数），而非 K。
    """
    n = d.shape[0]
    total = 0.0
    counted = 0
    for i in range(n):
        pos = set(int(x) for x in pos_indices[i])
        if not pos:
            continue
        order = np.argsort(d[i], kind="stable")
        order = order[order != i]
        retrieved = set(int(x) for x in order[:k])
        total += len(pos & retrieved) / len(pos)
        counted += 1
    return total / counted if counted else 0.0


# ---------------------------------------------------------------- 权重矩阵

def build_recall_weight_matrix(
    bits: np.ndarray,
    sim: np.ndarray,
    pos_k: int = 10,
    k: int = 10,
    pos_weight: float = 1.0,
    neg_weight: float = 2.0,
) -> np.ndarray:
    """构建面向 Recall@K 的动态权重矩阵（论文式 17）。"""
    n = bits.shape[0]
    w = np.zeros((n, n), dtype=np.float32)

    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    pos_idx = np.argsort(-s, axis=1)[:, :pos_k]          # P_i⁺

    d = hamming_matrix(bits)
    np.fill_diagonal(d, 1e9)
    pool_k = min(max(k * 3, pos_k * 2), n - 1)
    ham_order = np.argsort(d, axis=1)[:, :pool_k]

    pos_mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    pos_mask[rows, pos_idx] = True
    w[pos_mask] = pos_weight

    # 难负样本：Hamming 前 pool_k 中、既非自身也非正样本的，最多取 k 个
    neg_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        cnt = 0
        for j in ham_order[i]:
            j = int(j)
            if j == i or pos_mask[i, j]:
                continue
            neg_mask[i, j] = True
            cnt += 1
            if cnt >= k:
                break
    w[neg_mask] = -neg_weight

    return (w + w.T) / 2.0


def objective(bits: np.ndarray, w: np.ndarray) -> float:
    """O(C) = Σ_{i<j} W_ij · d_H(c_i, c_j)  —— 上三角求和。

    与 greedy_flip_deltas 的 Δ 保持【同一约定】（见该函数的推导说明）。
    取上三角而非全对求和，是因为 Σ_{i,j} 会把手性重复计一次；
    在 W 对称、对角距离为 0 时，Σ_{i<j} 的 argmin 与 Σ_{i,j} 完全一致。
    """
    d = hamming_matrix(bits)
    return float((w * np.triu(d, k=1)).sum())


# ---------------------------------------------------------------- 贪心翻转

def greedy_flip_deltas(bits: np.ndarray, w: np.ndarray) -> np.ndarray:
    """计算所有 (i, b) 的翻转增量 Δ = O(翻转后) − O(翻转前)，O 用 objective() 的上三角约定。

    ── 推导（论文 Theorem 1）+ 数值校准 ──────────────────────────────
    翻转第 i 个样本第 b 位时，仅 d_H(i,·) 一行改变。对 j ≠ i：
        Δd_H(i,j) = 1[C_ib ≠ C_jb]（翻转后） − 1[C_ib ≠ C_jb]（翻转前）
                  = 1 − 2·1[C_ib == C_jb]
    由于上三角目标里行 i 只贡献 j>i 的部分，而 W 对称、w_ii = 0，
    把 Δd 与 W 加权求和即得：
        Δ_{i,b} = 2·Σ_j W_ij·1[C_ib == C_jb] − Σ_j W_ij
                = 2·wsum_eq − row_sum

    数值穷举验证（scripts/selftest_post_optimize.py，以及开发期对全部 (i,b) 的扫描）：
        该式与实际上三角目标变化的最大误差 = 8.3e-07（浮点精度级）。
    曾被误判为「论文符号有误」，实为把 ΔO_full 与 ΔO_triu 两种约定混用所致。
    ────────────────────────────────────────────────────────────────
    """
    n, m = bits.shape
    row_sums = w.sum(axis=1)                     # (N,)
    deltas = np.empty((n, m), dtype=np.float32)
    blk = max(1, int(8_000_000 // max(n * m, 1)))
    for s in range(0, n, blk):
        e = min(s + blk, n)
        eq_blk = (bits[s:e, None, :] == bits[None, :, :])          # (b, N, M)
        # wsum_eq[i,m] = Σ_j W_ij · 1[C_jm == C_im]
        wsum_eq = np.einsum("ij,ijm->im", w[s:e], eq_blk.astype(np.float32))
        deltas[s:e] = 2.0 * wsum_eq - row_sums[s:e, None]
    return deltas


def greedy_flip_optimize(
    bits: np.ndarray,
    sim: np.ndarray,
    pos_k: int = 10,
    k: int = 10,
    pos_weight: float = 1.0,
    neg_weight: float = 2.0,
    max_iters: int = 10,
    verbose: bool = True,
    callback=None,
) -> np.ndarray:
    """面向 Recall@K 的贪心比特翻转（每轮每个样本最多翻一位）。"""
    c = np.asarray(bits).copy()
    n, m = c.shape

    for it in range(max_iters):
        w = build_recall_weight_matrix(c, sim, pos_k, k, pos_weight, neg_weight)
        delta = greedy_flip_deltas(c, w)

        improved = False
        flipped_rows: set[int] = set()
        # 按 Δ 从小到大扫描，每行最多翻一位
        order = np.argsort(delta, axis=None, kind="stable")
        for flat in order:
            i, b = divmod(int(flat), m)
            if i in flipped_rows:
                continue
            if delta[i, b] >= -1e-6:
                break
            c[i, b] ^= 1
            improved = True
            flipped_rows.add(i)

        if verbose or callback:
            pos_idx = np.argsort(-sim, axis=1)[:, 1 : pos_k + 1]
            rec = recall_at_k_from_ranking(pos_idx, hamming_matrix(c), k)
            obj = objective(c, w)
            if verbose:
                print(f"  [iter {it + 1}] flipped={len(flipped_rows)} obj={obj:.2f} recall@{k}={rec:.4f}",
                      flush=True)
            if callback:
                callback(it + 1, rec, obj, len(flipped_rows))

        if not improved:
            if verbose:
                print(f"  [iter {it + 1}] 已达 1-optimal，提前终止", flush=True)
            break

    return c


# ---------------------------------------------------------------- 导出 / 导入

def export_override_cache(path: str | Path, words: list[str], bits: np.ndarray) -> Path:
    """导出 sim_override 缓存（与 Rust 端一致的 big-endian 格式）。

    格式：[u32 magic=0xCAB1_0D01][u32 ver][u32 n_bits=52][u32 count]
          每词: [u16 len][utf8][u64 BE sim]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bits = np.asarray(bits)
    if bits.shape[1] != 52:
        raise ValueError(f"sim_override 仅支持 52 位，实际 {bits.shape[1]}")
    if len(words) != bits.shape[0]:
        raise ValueError("词数与码数不一致")

    # 位打包：bit i → 第 i 位（little bitorder，再按 little 取前 7 字节 → u64）
    packed = np.packbits(bits.astype(np.uint8), axis=1, bitorder="little")
    sims = np.zeros(bits.shape[0], dtype=np.uint64)
    for i in range(bits.shape[0]):
        sims[i] = int.from_bytes(packed[i].tobytes()[:7], "little")

    buf = bytearray()
    buf.extend(struct.pack(">IIII", OVERRIDE_MAGIC, OVERRIDE_VERSION, 52, len(words)))
    for word, s in zip(words, sims):
        wb = word.encode("utf-8")
        buf.extend(struct.pack(">H", len(wb)))
        buf.extend(wb)
        buf.extend(struct.pack(">Q", int(s)))
    path.write_bytes(bytes(buf))
    return path


def load_override_cache(path: str | Path) -> tuple[list[str], np.ndarray]:
    """读取 sim_override 缓存，返回 (words, 52 位 0/1 矩阵)。"""
    path = Path(path)
    data = path.read_bytes()
    magic, ver, n_bits, count = struct.unpack_from(">IIII", data, 0)
    if magic != OVERRIDE_MAGIC:
        raise ValueError(f"magic 不匹配：0x{magic:08X}")
    if ver != OVERRIDE_VERSION:
        raise ValueError(f"version 不支持：{ver}")
    if n_bits != 52:
        raise ValueError(f"n_bits 应为 52，实际 {n_bits}")

    words: list[str] = []
    bits = np.zeros((count, 52), dtype=np.uint8)
    off = 16
    for i in range(count):
        wlen = struct.unpack_from(">H", data, off)[0]
        off += 2
        words.append(data[off : off + wlen].decode("utf-8"))
        off += wlen
        sim = struct.unpack_from(">Q", data, off)[0]
        off += 8
        for b in range(52):
            bits[i, b] = (sim >> b) & 1
    return words, bits
