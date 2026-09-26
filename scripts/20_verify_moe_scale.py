# -*- coding: utf-8 -*-
"""
MoE FFN 的初始化量级校验（数值，非公式推导）。

为什么需要单独校验：
    这个模块的初始化量级错了两次，两次都表现为「训练 loss 不收敛」，
    但根因不同（一次放大 16 倍，一次缩小到 1.3%）。
    凭公式推导容易错，必须直接测输出分布。

判据：在相同输入下，MoE 的输出 std 应与标准 FFN 同量级（比值 0.4~2.5）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from cabinet_bnn.bnn.model import WeightCodeConfig, WeightCodeTable
from cabinet_bnn.bnn.moe_ffn import MoEConfig, MoECodeFFN


def main() -> int:
    d, d_ff, V = 512, 2048, 32000
    B, L = 8, 128

    torch.manual_seed(0)
    # 标准 FFN（与 CodeTransformerBlock 的 global 模式一致）
    w1 = nn.Linear(d, d_ff, bias=False)
    w2 = nn.Linear(d_ff, d, bias=False)
    nn.init.normal_(w1.weight, std=0.02)
    nn.init.normal_(w2.weight, std=0.02)

    cfg = WeightCodeConfig(k_basis=16, rank=16, param_bits=64)
    ct = WeightCodeTable(V, cfg)
    gen = torch.Generator().manual_seed(1337)
    ct.set_hash_from_u64(torch.randint(0, 2 ** 62, (V,), generator=gen,
                                       dtype=torch.int64))
    ct.refresh_codes()

    moe = MoECodeFFN(MoEConfig(d_model=d, d_ff=256, k_experts=16,
                               k_active=4, rank=16), ct)
    # MoE 的 down 输出维度是 d，与标准 FFN 的 w2 对应

    x = torch.randn(B, L, d)
    ids = torch.randint(0, V, (B,))

    with torch.no_grad():
        a1 = w1(x)
        s1 = F.silu(a1)
        out_std = w2(s1)

        # MoE：按 token 索引（每个样本一个 token id）
        wid = ids.unsqueeze(1).expand(B, L).reshape(-1)
        out_moe = moe(x.reshape(-1, d), wid).reshape(B, L, -1)

    print("=" * 72)
    print("MoE FFN 初始化量级校验")
    print("=" * 72)
    print(f"  输入        std = {x.std().item():.4f}")
    print(f"  标准 w1(x)  std = {a1.std().item():.4f}")
    print(f"  标准 silu   std = {s1.std().item():.4f}")
    print(f"  标准 输出   std = {out_std.std().item():.4f}   "
          f"absmax {out_std.abs().max().item():.3f}")
    print(f"  MoE  输出   std = {out_moe.std().item():.4f}   "
          f"absmax {out_moe.abs().max().item():.3f}")
    ratio = out_moe.std().item() / max(out_std.std().item(), 1e-12)
    print(f"\n  比值 MoE/标准 = {ratio:.3f}")
    ok = 0.4 < ratio < 2.5
    print(f"  [{'PASS' if ok else 'FAIL'}] 量级匹配（判据 0.4~2.5）")

    print("\n  各层权重 std（对照：标准 FFN 用 0.02，即 nn.Linear 默认值）:")
    print(f"    U    std = {moe.U.std().item():.5f}   shape {tuple(moe.U.shape)}")
    print(f"    V    std = {moe.V.std().item():.5f}   shape {tuple(moe.V.shape)}")
    print(f"    down std = {moe.down.weight.std().item():.5f}")
    print(f"    w1   std = {w1.weight.std().item():.5f}   (标准)")
    print(f"    w2   std = {w2.weight.std().item():.5f}   (标准)")

    print("\n  参数量:")
    rep = moe.param_report()
    std_n = w1.weight.numel() + w2.weight.numel()
    for k_, v in rep.items():
        print(f"    {k_:<8} {v:>10,}")
    print(f"    标准FFN  {std_n:>10,}   -> MoE 占 {rep['total']/std_n*100:.1f}%")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
