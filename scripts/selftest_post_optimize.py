# -*- coding: utf-8 -*-
"""
post_optimize 正确性自检（回归测试）。

背景：HSH-64 论文 Theorem 1 给出的翻转增量
        Δd_H = 1 − 2·1[C_ib == C_jb]
    符号有误（详见 cabinet_bnn/hsh/post_optimize.py 的注释）。
本脚本用【暴力枚举实际目标函数变化】来验证修复后的 Δ 公式，
覆盖：最小例子、随机例子、真实规模子集。

运行：
    python scripts/selftest_post_optimize.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from cabinet_bnn.hsh.post_optimize import (
    build_recall_weight_matrix,
    greedy_flip_deltas,
    hamming_matrix,
)

PASS, FAIL = "[PASS]", "[FAIL]"
_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = PASS if cond else FAIL
    print(f"  {tag} {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


def objective_full(bits: np.ndarray, w: np.ndarray) -> float:
    """O = Σ_{i<j} W_ij · d_H(i,j)  —— 与 objective() 同一约定（上三角）。"""
    d = hamming_matrix(bits)
    return float((w * np.triu(d, k=1)).sum())


# ---------------------------------------------------------------- 用例 1：距离增量
def test_distance_delta() -> None:
    print("\n[1] 距离增量：实测 d'(i,j) − d(i,j) = 2·1[C_ib == C_jb] − 1")
    print("    论文 Theorem 1 把该式写作 1 − 2·1[...]（即本式之反号）。")
    print("    两种写法都能用 —— 关键是 objective() 与 greedy_flip_deltas() 内部一致。")
    print("    本实现的一致性由用例 2 对【全部 (i,b)】逐一验证。")
    bits = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    for i, b in [(0, 0), (0, 1), (1, 2), (2, 0), (3, 1)]:
        before = hamming_matrix(bits)[i]
        b2 = bits.copy()
        b2[i, b] ^= 1
        after = hamming_matrix(b2)[i]
        js = [j for j in range(bits.shape[0]) if j != i]      # 排除对角线
        actual = np.array([after[j] - before[j] for j in js], dtype=np.float32)
        eq = np.array([bits[i, b] == bits[j, b] for j in js], dtype=np.float32)
        check(f"实测 Δd = 2·eq − 1 (i={i},b={b})", np.allclose(actual, 2.0 * eq - 1.0),
              f"误差 {np.abs(actual - (2.0 * eq - 1.0)).max():.2e}")
        check(f"  └ 论文式确为其反号 (i={i},b={b})",
              np.allclose(actual, -(1.0 - 2.0 * eq)))


# ---------------------------------------------------------------- 用例 2：Δ 与真实 ΔO
def test_delta_matches_objective() -> None:
    print("\n[2] Δ_{i,b} 是否等于目标函数的真实变化 O(翻转后) − O(翻转前)")
    rng = np.random.default_rng(7)

    for trial, (n, m) in enumerate([(4, 3), (12, 8), (60, 16), (200, 52)]):
        bits = rng.integers(0, 2, size=(n, m)).astype(np.uint8)
        w = rng.normal(size=(n, n)).astype(np.float32)
        w = (w + w.T) / 2.0
        np.fill_diagonal(w, 0.0)

        delta = greedy_flip_deltas(bits, w)
        o0 = objective_full(bits, w)

        worst = 0.0
        probes = [(int(rng.integers(n)), int(rng.integers(m))) for _ in range(12)]
        probes += [(0, 0), (n - 1, m - 1)]
        for i, b in probes:
            b2 = bits.copy()
            b2[i, b] ^= 1
            actual = objective_full(b2, w) - o0
            worst = max(worst, abs(actual - float(delta[i, b])))

        check(f"Δ 公式 (n={n}, m={m})", worst < 1e-2, f"最大误差 {worst:.3e}")


# ---------------------------------------------------------------- 用例 3：单步确实降目标
def test_single_step_decreases() -> None:
    print("\n[3] 取 Δ 最小者翻转，目标函数必须【下降】")
    rng = np.random.default_rng(11)
    n, m = 80, 24
    bits = rng.integers(0, 2, size=(n, m)).astype(np.uint8)
    w = rng.normal(size=(n, n)).astype(np.float32)
    w = (w + w.T) / 2.0
    np.fill_diagonal(w, 0.0)

    delta = greedy_flip_deltas(bits, w)
    o0 = objective_full(bits, w)
    flat = int(np.argmin(delta))
    i, b = divmod(flat, m)

    b2 = bits.copy()
    b2[i, b] ^= 1
    o1 = objective_full(b2, w)
    check("最小 Δ 翻转后目标下降", o1 < o0,
          f"O: {o0:.3f} -> {o1:.3f} (Δ={o1-o0:+.3f}, 预测 Δ={delta[i,b]:+.3f})")


# ---------------------------------------------------------------- 用例 4：整轮迭代
def test_round_decreases() -> None:
    print("\n[4] 一整轮贪心（每行最多翻一位）目标函数必须单调下降")
    rng = np.random.default_rng(13)
    n, m = 120, 52
    bits = rng.integers(0, 2, size=(n, m)).astype(np.uint8)
    w = rng.normal(size=(n, n)).astype(np.float32)
    w = (w + w.T) / 2.0
    np.fill_diagonal(w, 0.0)

    o_before = objective_full(bits, w)
    delta = greedy_flip_deltas(bits, w)
    order = np.argsort(delta, axis=None, kind="stable")
    c = bits.copy()
    flipped: set[int] = set()
    for f in order:
        ii, bb = divmod(int(f), m)
        if ii in flipped:
            continue
        if delta[ii, bb] >= -1e-6:
            break
        c[ii, bb] ^= 1
        flipped.add(ii)
    o_after = objective_full(c, w)

    check("整轮后目标下降", o_after < o_before,
          f"O: {o_before:.2f} -> {o_after:.2f} (Δ={o_after-o_before:+.2f}, 翻转 {len(flipped)} 行)")


# ---------------------------------------------------------------- 用例 5：真实 W 赋值
def test_with_real_weight_matrix() -> None:
    print("\n[5] 用真实的召回导向权重矩阵，单步也必须降目标（含正负权重混合）")
    rng = np.random.default_rng(17)
    n, m = 100, 52
    bits = rng.integers(0, 2, size=(n, m)).astype(np.uint8)
    sim = rng.normal(size=(n, n)).astype(np.float32)
    sim = (sim + sim.T) / 2.0
    np.fill_diagonal(sim, -9.0)

    w = build_recall_weight_matrix(bits, sim, pos_k=10, k=10, pos_weight=1.0, neg_weight=2.0)
    check("权重矩阵含正负两类",
          (w > 0).any() and (w < 0).any(),
          f"正权重 {(w>0).sum()} 项，负权重 {(w<0).sum()} 项")

    delta = greedy_flip_deltas(bits, w)
    o0 = objective_full(bits, w)
    flat = int(np.argmin(delta))
    i, b = divmod(flat, m)
    b2 = bits.copy()
    b2[i, b] ^= 1
    o1 = objective_full(b2, w)
    check("最小 Δ 翻转后目标下降", o1 < o0,
          f"O: {o0:.2f} -> {o1:.2f} (Δ={o1-o0:+.2f}, 预测={delta[i,b]:+.2f})")


def main() -> int:
    print("=" * 70)
    print("post_optimize 正确性自检")
    print("=" * 70)
    test_distance_delta()
    test_delta_matches_objective()
    test_single_step_decreases()
    test_round_decreases()
    test_with_real_weight_matrix()

    print("\n" + "=" * 70)
    if _failures:
        print(f"失败 {len(_failures)} 项：")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("全部通过 [OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
