# -*- coding: utf-8 -*-
"""
MoE 稀疏 FFN 自检。

必须验证的六件事：
  ① 路由确实只激活 k' 个专家（稀疏性）
  ② 路由由码决定，且【确定性】—— 同一 token 任何时候都路由到同一组
  ③ 专家使用率不坍缩（负载均衡）
  ④ 折叠式两级 einsum 与「显式构造权重矩阵」数值等价
  ⑤ 稀疏【写】：改一个 token 的码只影响它激活的专家对应的参数
  ⑥ 参数量确实远小于标准 FFN，且显存可控

运行：
    python scripts/selftest_moe_ffn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from cabinet_bnn.bnn.model import WeightCodeConfig, WeightCodeTable
from cabinet_bnn.bnn.moe_ffn import MoEConfig, MoECodeFFN

PASS, FAIL = "[PASS]", "[FAIL]"
_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {PASS if cond else FAIL} {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


def make(vocab=4096, d=64, d_ff=32, k=16, ka=4, r=8, seed=0):
    torch.manual_seed(seed)
    cfg = WeightCodeConfig(k_basis=k, rank=r, param_bits=64)
    ct = WeightCodeTable(vocab, cfg)
    ct.set_hash_from_u64(torch.randint(0, 2 ** 62, (vocab,), dtype=torch.int64))
    m = MoECodeFFN(MoEConfig(d_model=d, d_ff=d_ff, k_experts=k,
                             k_active=ka, rank=r), ct)
    return m, ct


def test_sparsity() -> None:
    print("\n[1] 稀疏性：每个 token 只激活 k' 个专家")
    m, _ = make()
    ids = torch.randint(0, 4096, (64,))
    idx, w = m.route(ids)
    check("路由索引形状 == (n, k_active)", idx.shape == (64, 4),
          f"{tuple(idx.shape)}")
    check("每行权重和为 1", torch.allclose(w.sum(1), torch.ones(64), atol=1e-5))
    uniq_per_row = [len(set(row.tolist())) for row in idx]
    check("每行专家不重复", all(u == 4 for u in uniq_per_row),
          f"最小不重复数 {min(uniq_per_row)}")


def test_determinism() -> None:
    print("\n[2] 确定性：同一 token 始终路由到同一组专家")
    m, _ = make()
    ids = torch.randint(0, 4096, (32,))
    i1, w1 = m.route(ids)
    i2, w2 = m.route(ids)
    check("两次路由结果一致", bool((i1 == i2).all() and torch.allclose(w1, w2)))

    # 打乱顺序后，逐 token 的结果仍应相同
    perm = torch.randperm(32)
    i3, _ = m.route(ids[perm])
    check("顺序无关", bool((i3 == i1[perm]).all()))


def test_load_balance() -> None:
    print("\n[3] 负载均衡：专家使用率不应坍缩")
    m, _ = make(vocab=32000, k=16, ka=4)
    st = m.usage_stats(n_sample=32000)
    print(f"    使用计数: {[int(x) for x in st['counts']]}")
    print(f"    最大占比 {st['max_share']:.3f}  最小占比 {st['min_share']:.3f}  "
          f"归一化熵 {st['entropy_norm']:.4f}")
    print(f"    （完全均衡时每专家占比 {4/16:.3f}，熵 1.0）")
    check("专家未坍缩（熵 > 0.8）", st["entropy_norm"] > 0.8,
          f"熵 {st['entropy_norm']:.4f}")
    check("最大占比 < 2 倍理想值", st["max_share"] < 2 * 4 / 16,
          f"{st['max_share']:.3f} vs 理想 {4/16:.3f}")


def test_einsum_equivalence() -> None:
    print("\n[4] 折叠式两级 einsum 与显式权重矩阵等价")
    m, _ = make(vocab=256, d=32, d_ff=16, k=8, ka=3, r=6)
    x = torch.randn(4, 5, 32)
    tid = torch.tensor([3, 17, 17, 200])

    with torch.no_grad():
        got = m(x, tid)

        # 参考实现：显式构造等价权重并走标准 FFN 路径。
        #
        # 数学：
        #   前向 h = A_sᵀ (B_s · x)，  A_s = Σ w_j U_j : (r, d_ff)
        #                              B_s = Σ w_j V_j : (r, d)
        #   => 等价权重 W = A_sᵀ B_s，形状 (d_ff, d)
        #      即 h = W · x = (d_ff, d) @ (d, L) -> (d_ff, L)   ✓
        #   ⚠️ 注意 einsum 的输入顺序必须是 (A, B, x)，写成 (B, A, x) 会得到
        #      (d, d_ff) 的转置权重并静默跑通。
        uniq, inv = torch.unique(tid, return_inverse=True)
        idx, w = m.route(uniq)
        outs = []
        for b in range(4):
            A = torch.zeros(6, 16)          # (r, d_ff)
            Bm = torch.zeros(6, 32)         # (r, d)
            for kk in range(3):
                j = int(idx[inv[b], kk])
                wj = w[inv[b], kk]
                A = A + wj * m.U[j]
                Bm = Bm + wj * m.V[j]
            h = torch.einsum("ro,ri,li->lo", A, Bm, x[b])          # (L, d_ff)
            outs.append(m.down(torch.nn.functional.silu(h)))
        ref = torch.stack(outs)
        err = (got - ref).abs().max().item()
    check("einsum 等价", err < 1e-5, f"最大误差 {err:.2e}")


def test_sparse_write() -> None:
    print("\n[5] 稀疏写：改一个 token 的码，只影响它激活的专家")
    m, ct = make(vocab=512, k=16, ka=4)

    # 只改一个 token 的 hash（模拟「信息码热更改」）。
    # ⚠️ 必须改在【路由真正使用】的位上：路由用 hash 的 0..59 位，
    #    高 4 位（60-63，即 feat 段）不参与路由。改错位置会得到
    #    「路由不变」的假失败。
    target = 42
    before = m.route(torch.tensor([target]))[0][0].tolist()
    ct.hash_code[target] = int(ct.hash_code[target]) ^ (0x3F << 20)
    after = m.route(torch.tensor([target]))[0][0].tolist()
    changed = set(before) != set(after)
    print(f"    改前激活: {before}")
    print(f"    改后激活: {after}")

    # 关键：改一个 token【不应】影响其它 token 的路由
    others = [i for i in range(512) if i != target]
    oids = torch.tensor(others)
    idx_o = m.route(oids)[0].clone()
    ct.hash_code[target] = int(ct.hash_code[target]) ^ (0x3FFF << 24)
    idx_o2 = m.route(oids)[0]
    check("改一个 token 不影响其它 token 的路由",
          bool((idx_o == idx_o2).all()),
          f"受影响 token 数 {int((idx_o != idx_o2).any(1).sum())}")
    check("被改的 token 自身路由确实变了", changed)
    print(f"    >> 爆炸半径 = 1 个 token 的 k'={4} 个专家，而非全部 {16} 个")


def test_param_and_memory() -> None:
    print("\n[6] 参数量与显存")
    d, V = 512, 32000
    for k, ka, r_, dff in [(16, 4, 16, 256), (32, 4, 16, 256), (16, 4, 32, 512)]:
        m, _ = make(vocab=V, d=d, d_ff=dff, k=k, ka=ka, r=r_)
        rep = m.param_report()
        std = 2 * d * 2048
        print(f"    k={k:>2} k'={ka} r={r_:>2} d_ff={dff:>3}: "
              f"基 {rep['basis']:>8,} + down {rep['down']:>8,} = {rep['total']:>9,}"
              f"  (标准 FFN {std:,} 的 {rep['total']/std*100:.1f}%)")
        if k == 16 and dff == 256:
            x = torch.randn(16, 256, d)
            tid = torch.randint(0, V, (16,))
            with torch.no_grad():
                y = m(x, tid)
            check("大维度前向成功", y.shape == (16, 256, d),
                  f"输出 {tuple(y.shape)}")
            nbytes = sum(p.numel() * p.element_size() for p in m.parameters())
            print(f"    权重显存 {nbytes/1024**2:.2f} MB")

    # 梯度
    m, _ = make()
    m2, _ = make()
    x = torch.randn(4, 6, 64, requires_grad=True)
    tid = torch.tensor([1, 2, 3, 4])
    out = m2(x, tid)
    out.pow(2).mean().backward()
    check("U 有梯度", m2.U.grad is not None and m2.U.grad.abs().max().item() > 0)
    check("V 有梯度", m2.V.grad is not None and m2.V.grad.abs().max().item() > 0)
    check("down 有梯度", m2.down.weight.grad is not None
          and m2.down.weight.grad.abs().max().item() > 0)
    check("输入有梯度", x.grad is not None and x.grad.abs().max().item() > 0)


def main() -> int:
    print("=" * 76)
    print("MoE 稀疏 FFN 自检")
    print("=" * 76)
    test_sparsity()
    test_determinism()
    test_load_balance()
    test_einsum_equivalence()
    test_sparse_write()
    test_param_and_memory()

    print("\n" + "=" * 76)
    if _failures:
        print(f"失败 {len(_failures)} 项：")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("全部通过 [OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
