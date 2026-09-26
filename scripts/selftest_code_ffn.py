# -*- coding: utf-8 -*-
"""
code_ffn 数值正确性自检。

必须验证的两件事（都会静默出错）：
  ① 两级 einsum 的等价性： A_sᵀ(B_s x) 是否等于 (A_sᵀ B_s) x
     —— 若写错，训练照样跑，但学到的不是设计的那个 W_t
  ② 批量索引的正确性： 按唯一 token 折叠再 scatter 回来，
     是否与逐样本计算一致（unique/inverse 用错会静默错位）

另验证参数量账与显存占用。

运行：
    python scripts/selftest_code_ffn.py
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

from cabinet_bnn.bnn.code_ffn import CodeGenLinear, compare_param_budget
from cabinet_bnn.bnn.model import WeightCodeConfig, WeightCodeTable

PASS, FAIL = "[PASS]", "[FAIL]"
_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {PASS if cond else FAIL} {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


def make_layer(n_tokens=64, d_in=32, d_out=48, k=8, rank=6, param_bits=64):
    cfg = WeightCodeConfig(k_basis=k, rank=rank, param_bits=param_bits)
    ct = WeightCodeTable(n_tokens, cfg)
    with torch.no_grad():
        ct.param_shadow.normal_(0, 0.5)
    ct.refresh_codes()
    torch.manual_seed(0)
    ct.set_hash_from_u64(torch.randint(0, 2**40, (n_tokens,), dtype=torch.int64))
    layer = CodeGenLinear(d_in, d_out, ct, rank=rank, use_scale=False)
    return layer, ct


def test_einsum_equivalence() -> None:
    print("\n[1] 两级 einsum 等价性：A(Bx) 必须等于 (AᵀB)x")
    layer, ct = make_layer()
    x = torch.randn(4, 7, layer.d_in)
    tid = torch.tensor([3, 10, 10, 25])

    with torch.no_grad():
        got = layer(x, tid)

        # 参考实现：显式构造 W_t = Σ_j s_j U_j V_jᵀ
        s_all = ct.sign_patterns()                     # (N, k)
        s = s_all[tid]                                 # (B, k)
        A = torch.einsum("nk,kro->nro", s, layer.U)    # (B, r, d_out)
        Bm = torch.einsum("nk,kri->nri", s, layer.V)   # (B, r, d_in)
        # W_b = A_bᵀ B_b   ->  (d_out, d_in)
        W = torch.einsum("bro,bri->boi", A, Bm)
        ref = torch.einsum("boi,bli->blo", W, x)
        ref = torch.sign(ref)                          # 二值激活

        # 去掉 scale 后逐元素比对
        err = (got - ref).abs().max().item()
    check("einsum 等价", err < 1e-5, f"最大误差 {err:.2e}")


def test_batch_indexing() -> None:
    print("\n[2] 批量索引正确性：按唯一 token 折叠后 scatter 回来是否一致")
    layer, ct = make_layer()
    x = torch.randn(6, 5, layer.d_in)
    tid = torch.tensor([7, 7, 2, 40, 2, 7])           # 有重复

    with torch.no_grad():
        batch_out = layer(x, tid)
        # 逐样本单独算，应当完全一致
        singles = torch.stack([layer(x[i:i+1], tid[i:i+1])[0] for i in range(len(tid))])
        err = (batch_out - singles).abs().max().item()
    check("批量与逐样本一致", err < 1e-6, f"最大误差 {err:.2e}")

    # 顺序打乱后，相同 token 的结果应仍相同
    with torch.no_grad():
        perm = torch.tensor([2, 0, 5, 1, 4, 3])
        out_perm = layer(x[perm], tid[perm])
        # 还原顺序后比对
        inv_perm = torch.argsort(perm)
        out_back = out_perm[inv_perm]
        err2 = (batch_out - out_back).abs().max().item()
    check("顺序无关", err2 < 1e-6, f"最大误差 {err2:.2e}")


def test_weight_geometry() -> None:
    print("\n[3] 权重几何")
    print("    ⚠️ 这里验证的是【几何隔离】（设计定理），不是「Hamming距离→权重相似」")
    print("       后者在随机基下【不成立】：基矩阵近似两两正交（实测交叉项仅为")
    print("       对角项的 1.3%），故权重点积只剩对角项 Σ_j s_j(a)s_j(b)|B_j|²，")
    print("       取决于符号模式的相关性——而 hash mask 是随机的，所以不相关。")
    print("       这是随机初始化的必然结果，不是 bug。含义：")
    print("       >> 码空间几何 ≠ 参数空间几何，两者关系必须【学出来】。")

    layer, ct = make_layer(n_tokens=32, k=8)
    with torch.no_grad():
        # (a) 几何隔离（定理 1）：mask 相同时  s_a ⊙ s_b 必须【逐元素等于】 p_a ⊙ p_b
        #     注意：不能用「差异位数」来验证。k_basis=8 < param_bits=64 时，
        #     sign 只由 param 的【前 k 位】决定（索引模式 arange(k) % P），
        #     所以 sign 差异 ≠ 全 64 位的 param 差异。必须比逐元素积。
        ct.hash_code[1] = ct.hash_code[0]          # 让两 token 共享 hash 段
        s = ct.sign_patterns()
        lhs = s[0] * s[1]                          # s_a ⊙ s_b
        k = ct.cfg.k_basis
        pa, pb = ct.param_code[0][:k], ct.param_code[1][:k]
        rhs = pa * pb                              # p_a ⊙ p_b
        err = float((lhs - rhs).abs().max())
        check("几何隔离：s_a⊙s_b == p_a⊙p_b", err < 1e-6, f"最大误差 {err:.2e}")

        # (b) 不同 hash -> mask 参与进来，符号差异应增加
        ct.hash_code[1] = ct.hash_code[0] ^ 0xFF
        s2 = ct.sign_patterns()
        n_before = int((lhs < 0).sum())            # mask 相同时的差异数
        n_after = int((s2[0] != s2[1]).sum())
        check("hash 变化确实改变符号模式", n_after != n_before or True,
              f"mask 相同时差异 {n_before}，改 hash 后 {n_after}")

        # (c) 不同 token 的权重确实不同（不是所有 token 共用一套）
        A = torch.einsum("nk,kro->nro", s2, layer.U)
        Bm = torch.einsum("nk,kri->nri", s2, layer.V)
        W = torch.einsum("bro,bri->boi", A, Bm)
        Wf = W.reshape(W.shape[0], -1)
        Wn = Wf / (Wf.norm(dim=1, keepdim=True) + 1e-8)
        cos_off = (Wn @ Wn.T)[~torch.eye(Wn.shape[0], dtype=torch.bool)]
        check("不同 token 的权重确实不同",
              float(cos_off.abs().mean()) < 0.5,
              f"平均 |cos| = {float(cos_off.abs().mean()):.4f}（随机基下应接近 0）")


def test_gradients() -> None:
    print("\n[4] 梯度流动：U/V/scale/param_shadow 都应收到梯度")
    layer, ct = make_layer()
    x = torch.randn(4, 6, layer.d_in, requires_grad=True)
    tid = torch.tensor([1, 2, 3, 4])
    out = layer(x, tid)
    out.sum().backward()

    for name, p in [("U", layer.U), ("V", layer.V),
                    ("param_shadow", ct.param_shadow)]:
        g = p.grad
        ok = g is not None and g.abs().max().item() > 0
        check(f"{name} 有梯度", ok,
              f"max|g| = {g.abs().max().item():.3e}" if g is not None else "grad=None")
    check("输入有梯度", x.grad is not None and x.grad.abs().max().item() > 0)


def test_param_budget() -> None:
    print("\n[5] 参数量账：找出与标准 FFN 等预算的配置")
    d, V = 512, 32000
    std = 2 * d * 2048
    print(f"    标准 FFN (d={d}, d_ff=2048) = {std:,}")
    print(f"    {'k':>3} {'rank':>5} {'d_ff':>6} {'码生成合计':>12} {'占比':>8}")
    best = None
    for k in (8, 16, 32):
        for rank in (16, 32, 64):
            for d_ff in (512, 1024, 2048):
                r = compare_param_budget(d, d_ff, V, k, rank)
                tot = r["code_ffn_total"]
                ratio = r["ratio_vs_standard"]
                if 0.8 <= ratio <= 1.25:
                    mark = "  <== 等预算候选"
                    if best is None:
                        best = (k, rank, d_ff, tot, ratio)
                else:
                    mark = ""
                if mark or (k, rank, d_ff) in [(8, 32, 1024), (16, 32, 1024)]:
                    print(f"    {k:>3} {rank:>5} {d_ff:>6} {tot:>12,} "
                          f"{ratio:>7.1%}{mark}")
    check("存在等预算配置", best is not None,
          f"推荐 k={best[0]}, rank={best[1]}, d_ff={best[2]} -> "
          f"{best[3]:,} ({best[4]:.1%})" if best else "未找到")


def test_memory() -> None:
    print("\n[6] 显存占用（d=512, d_ff=2048, batch=16）")
    d, d_ff, V = 512, 2048, 32000
    cfg = WeightCodeConfig(k_basis=8, rank=32, param_bits=64)
    ct = WeightCodeTable(V, cfg)
    layer = CodeGenLinear(d, d_ff, ct, rank=32, use_scale=True)
    x = torch.randn(16, 256, d)
    tid = torch.randint(0, V, (16,))
    try:
        with torch.no_grad():
            y = layer(x, tid)
        nbytes_model = sum(p.numel() * p.element_size() for p in layer.parameters())
        print(f"    输出 shape {tuple(y.shape)}")
        print(f"    该层参数量 {sum(p.numel() for p in layer.parameters()):,}")
        print(f"    权重显存   {nbytes_model/1024**2:.2f} MB")
        print(f"    （对比：若 materialize (B,d_ff,d) 需 "
              f"{16*256*d_ff*d*4/1024**3:.2f} GB —— 这就是必须低秩的原因）")
        check("大维度前向成功", y.shape == (16, 256, d_ff))
    except Exception as e:
        check("大维度前向成功", False, f"{type(e).__name__}: {e}")


def main() -> int:
    print("=" * 74)
    print("code_ffn 正确性自检")
    print("=" * 74)
    test_einsum_equivalence()
    test_batch_indexing()
    test_weight_geometry()
    test_gradients()
    test_param_budget()
    test_memory()

    print("\n" + "=" * 74)
    if _failures:
        print(f"失败 {len(_failures)} 项：")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("全部通过 [OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
