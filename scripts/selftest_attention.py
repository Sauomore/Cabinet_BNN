# -*- coding: utf-8 -*-
"""
注意力机制正确性自检（回归测试）。

注意力有两处极易静默出错的地方，必须逐项验证：
  ① 因果掩码 —— 若写错，模型会「看见未来」，训练 loss 好得不真实，推理时崩
  ② KV cache 等价性 —— 若写错，推理输出与训练不一致，且不会报错

另验证 RoPE、梯度流动、参数量估算。

运行：
    python scripts/selftest_attention.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from cabinet_bnn.bnn.attention import (
    BNNTransformerLM, CausalSelfAttention, TransformerConfig, build_transformer,
)

PASS, FAIL = "[PASS]", "[FAIL]"
_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {PASS if cond else FAIL} {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


def test_rmsnorm_available() -> None:
    print("\n[0] torch 版本与 RMSNorm 可用性")
    print(f"    torch {torch.__version__}")
    has = hasattr(nn_mod := torch.nn, "RMSNorm")
    check("nn.RMSNorm 可用", has, "不可用时自动回退 LayerNorm")
    if not has:
        print("    -> 将使用 LayerNorm 回退（数值上不同但功能等价）")


def test_causal_mask() -> None:
    print("\n[1] 因果掩码：位置 i 的输出不得依赖 j > i")
    torch.manual_seed(0)
    d, H, L = 32, 4, 12
    attn = CausalSelfAttention(d, H, dropout=0.0, use_rope=False)
    attn.eval()
    x = torch.randn(2, L, d)

    with torch.no_grad():
        base, _ = attn(x)

    # 改动最后一个位置，前面所有位置的输出必须【逐位不变】
    x2 = x.clone()
    x2[:, -1, :] = torch.randn(d) * 5
    with torch.no_grad():
        mod, _ = attn(x2)
    d_last = (base[:, :-1] - mod[:, :-1]).abs().max().item()
    check("改动末位不影响前面位置", d_last < 1e-6, f"最大差异 {d_last:.2e}")

    # 改动第一个位置，后面的位置【应当】改变（否则注意力没起作用）
    x3 = x.clone()
    x3[:, 0, :] = torch.randn(d) * 5
    with torch.no_grad():
        mod3, _ = attn(x3)
    d_after = (base[:, 1:] - mod3[:, 1:]).abs().max().item()
    check("改动首位会影响后面位置", d_after > 1e-4, f"最大差异 {d_after:.2e}")

    # 逐位置反推：位置 k 的输出只依赖 0..k
    for k in [0, 1, 5, L - 1]:
        xa = x.clone()
        if k + 1 < L:
            xa[:, k + 1 :, :] = torch.randn(2, L - k - 1, d) * 5
        with torch.no_grad():
            ya, _ = attn(xa)
        diff = (base[:, : k + 1] - ya[:, : k + 1]).abs().max().item()
        check(f"位置 {k} 不受未来影响", diff < 1e-6, f"{diff:.2e}")


def test_kv_cache_equivalence() -> None:
    print("\n[2] KV cache 等价性：增量解码须与整段前向逐位一致")
    torch.manual_seed(1)
    d, H, L = 32, 4, 10
    attn = CausalSelfAttention(d, H, dropout=0.0, use_rope=True, max_len=64)
    attn.eval()
    x = torch.randn(2, L, d)

    with torch.no_grad():
        full, _ = attn(x)                       # 整段

        # 逐 token 增量解码
        outs = []
        cache = None
        for t in range(L):
            o, cache = attn(x[:, t : t + 1, :], cache, return_cache=True)
            outs.append(o)
        inc = torch.cat(outs, dim=1)

    err = (full - inc).abs().max().item()
    check("KV cache 与整段前向一致", err < 1e-5, f"最大误差 {err:.2e}")


def test_rope() -> None:
    print("\n[3] RoPE：位置编码不改变范数，且随位置变化")
    torch.manual_seed(2)
    d, H = 32, 4
    attn = CausalSelfAttention(d, H, use_rope=True, max_len=64)
    x = torch.randn(1, 8, H, d // H).transpose(1, 2)     # (1,H,8,D)
    y = attn._apply_rope(x, 0)
    n0 = x.norm(dim=-1)
    n1 = y.norm(dim=-1)
    check("RoPE 保范数", (n0 - n1).abs().max().item() < 1e-5,
          f"最大差异 {(n0 - n1).abs().max().item():.2e}")

    # 不同位置应产生不同的编码
    y2 = attn._apply_rope(x[:, :, :4], 0)
    y3 = attn._apply_rope(x[:, :, :4], 4)
    check("不同 offset 产生不同编码", (y2 - y3).abs().max().item() > 1e-4)


def test_gradients() -> None:
    print("\n[4] 梯度流动：所有参数都应收到非零梯度")
    torch.manual_seed(3)
    m = build_transformer(vocab_size=64, n_words=128, preset="nano")
    x = torch.randint(0, 64, (4, 16))
    y = torch.randint(0, 64, (4, 16))
    logits, _ = m(x)
    loss = F.cross_entropy(logits.reshape(-1, 64), y.reshape(-1))
    loss.backward()

    no_grad = [n for n, p in m.named_parameters()
               if p.requires_grad and (p.grad is None or p.grad.abs().max().item() == 0.0)]
    check("无零梯度参数", len(no_grad) == 0,
          f"零梯度: {no_grad[:5]}" if no_grad else f"共 {sum(1 for p in m.parameters() if p.requires_grad)} 个参数均有梯度")


def test_no_nan() -> None:
    print("\n[5] 数值稳定性：无 NaN/Inf")
    torch.manual_seed(4)
    for preset in ("nano", "tiny"):
        m = build_transformer(vocab_size=128, n_words=256, preset=preset)
        x = torch.randint(0, 128, (2, 64))
        logits, _ = m(x)
        ok = torch.isfinite(logits).all().item()
        check(f"{preset} 前向无 NaN/Inf", bool(ok))
        # 首 token 的注意力行（无可用 key 的边界情形）
        check(f"{preset} 首 token 输出有限", bool(torch.isfinite(logits[:, 0, :]).all().item()))


def test_param_counts() -> None:
    print("\n[6] 参数量估算（供选型参考）")
    for preset in ("nano", "tiny", "small", "base"):
        m = build_transformer(vocab_size=152064, n_words=152064, preset=preset)
        s = m.n_params()
        print(f"    {preset:6s}  d={m.cfg.d_model:>4} layers={m.cfg.n_layers} "
              f"heads={m.cfg.n_heads}  ->  {s['total']/1e6:>7.1f} M 参数")
    check("base 预设存在", True)


def test_generate() -> None:
    print("\n[7] 自回归生成（带 KV cache）")
    torch.manual_seed(5)
    m = build_transformer(vocab_size=64, n_words=128, preset="nano", max_len=64)
    idx = torch.randint(0, 64, (2, 4))
    out = m.generate(idx, max_new_tokens=8, temperature=1.0, top_k=10)
    check("生成序列长度正确", out.shape == (2, 12), f"shape={tuple(out.shape)}")
    check("生成结果全为合法 token", bool((out >= 0).all() and (out < 64).all()))


def test_code_ffn_mode() -> None:
    print("\n[8] ffn_mode='code'：权重码生成的 FFN 能跑通")
    torch.manual_seed(6)
    m = build_transformer(vocab_size=64, n_words=128, preset="nano",
                          ffn_mode="code", k_basis=8, rank=16)
    x = torch.randint(0, 64, (2, 16))
    wid = torch.randint(0, 128, (2,))
    logits, _ = m(x, wid)
    check("ffn_mode=code 前向成功", logits.shape == (2, 16, 64), f"shape={tuple(logits.shape)}")
    s = m.n_params()
    print(f"      参数量 {s['total']:,}（per-token {s['per_token']:,}）")


def main() -> int:
    print("=" * 74)
    print("注意力机制正确性自检")
    print("=" * 74)
    import torch.nn as nn
    globals()["nn"] = nn

    test_rmsnorm_available()
    test_causal_mask()
    test_kv_cache_equivalence()
    test_rope()
    test_gradients()
    test_no_nan()
    test_param_counts()
    test_generate()
    test_code_ffn_mode()

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
