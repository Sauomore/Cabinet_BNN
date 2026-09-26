# -*- coding: utf-8 -*-
"""
高效的码生成 FFN（低秩基矩阵 + 批量索引）。

为什么需要新实现：
    已有的 `CodeWeightLinear`（model.py）会 materialize 完整的 (B, d_out, d_in)
    权重张量。在 d=512, d_ff=2048 时，单层单批次就是 512×2048×4B ≈ 4MB/样本，
    直接爆显存。本模块改为：

    数学上仍然是  W_t = Σ_j s_j(t) · (U_j V_jᵀ)      （与设计完全一致）
    但计算路径为：
        ① 按【唯一 token】求符号模式 s        (U_uniq, k)
        ② 把 s 折叠进基矩阵：A_s = Σ_j s_j U_j   →  (U_uniq, r, d_ff)
                             B_s = Σ_j s_j V_j   →  (U_uniq, r, d)
        ③ 输出用两级 einsum： W_t x = A_sᵀ(B_s x)
           或 W_t x = Σ_r A_s[r] · (B_s[r] · x)

    这样峰值显存只与【批次内唯一 token 数 U】成正比，而不是与 d_ff×d 的完整矩阵
    成正比。这是让码生成方式在真实规模上可行的关键。

参数量（关键：可与标准 FFN 对齐）：
    标准 FFN        : 2 · d · d_ff
    码生成 FFN      : k · r · (d_ff + d)  +  V · P/8（码表）
    通过调小 d_ff 或 r，可以让两者参数预算相同 —— 这才是公平对比的前提。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .model import WeightCodeTable, binary_activation


class CodeGenLinear(nn.Module):
    """用权重码生成权重的线性层（低秩、批量索引高效实现）。

    与 `CodeWeightLinear` 数学等价，但不会 materialize 完整权重矩阵。

    Args:
        d_in, d_out: 输入/输出维度
        code_table:  权重码表（提供 sign_patterns）
        rank:        低秩分解的秩 r
        use_scale:   是否使用共享逐通道 scale（二值网络标配）
    """

    def __init__(self, d_in: int, d_out: int, code_table: WeightCodeTable,
                 rank: int | None = None, use_scale: bool = True):
        super().__init__()
        cfg = code_table.cfg
        self.d_in, self.d_out = d_in, d_out
        self.code_table = code_table
        self.k = cfg.k_basis
        self.rank = rank if rank is not None else cfg.rank

        bound = (1.0 / max(d_in, 1)) ** 0.5
        # U_j: (k, r, d_out),  V_j: (k, r, d_in)
        self.U = nn.Parameter(torch.randn(self.k, self.rank, d_out) * bound)
        self.V = nn.Parameter(torch.randn(self.k, self.rank, d_in) * bound)

        self.use_scale = use_scale
        self.scale = nn.Parameter(torch.ones(d_out)) if use_scale else None

    # ---------------- 折叠：把符号模式折进基矩阵 ----------------

    def folded_basis(self, token_ids: torch.Tensor | None = None):
        """返回 (A_s, B_s)，形状均为 (n, r, ·)。

        A_s[n] = Σ_j s_j(t_n) · U_j
        B_s[n] = Σ_j s_j(t_n) · V_j

        token_ids 为 None 时对全词表折叠（代价 V·k·r，慎用）。
        """
        s_all = self.code_table.sign_patterns()          # (N, k)
        if token_ids is not None:
            s = s_all[token_ids]                          # (n, k)
        else:
            s = s_all
        # (n,k) × (k,r,d) -> (n,r,d)
        A = torch.einsum("nk,kro->nro", s, self.U)
        B = torch.einsum("nk,kri->nri", s, self.V)
        return A, B

    # ---------------- 前向 ----------------

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """
        x:         (B, L, d_in)  （也支持 (B, d_in)）
        token_ids: (B,)  每个样本属于哪个 token
        返回:      同 x 的前两维 + d_out，且已过二值激活
        """
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)
        B, L, d_in = x.shape

        # 按唯一 token 折叠基矩阵，避免为每个样本重复计算
        uniq, inv = torch.unique(token_ids, return_inverse=True)   # (U,), (B,)
        A_u, B_u = self.folded_basis(uniq)                          # (U,r,out), (U,r,in)
        A = A_u[inv]                                                # (B,r,out)
        Bm = B_u[inv]                                               # (B,r,in)

        # W_t x = A_sᵀ (B_s x)   —— 两级 einsum，不 materialize (d_out,d_in)
        h = torch.einsum("brd,bld->brl", Bm, x)                     # (B, r, L)
        y = torch.einsum("bro,brl->blo", A, h)                      # (B, L, d_out)

        if self.use_scale:
            y = y * self.scale
        y = binary_activation(y)
        return y.squeeze(1) if squeeze else y

    def effective_weight_norm(self) -> float:
        """|W_t| 的平均 Frobenius 范数（全词表，诊断用，代价较高）。"""
        with torch.no_grad():
            A, B = self.folded_basis(None)
            # |W_t|² = |A_s B_sᵀ|² 不易直接算，这里用 |A||B| 作为上界近似
            return float((A.norm(dim=(1, 2)) * B.norm(dim=(1, 2))).mean())


@dataclass
class CodeFFNConfig:
    d_model: int
    d_ff: int
    k_basis: int = 8
    rank: int = 32
    param_bits: int = 64
    use_scale: bool = True


class CodeGenFFN(nn.Module):
    """码生成的前馈层：w2(silu(w1(x)))。

    w1 用码生成（CodeGenLinear），w2 保持标准全局权重 ——
    这样「per-token 机制」的贡献可以单独看，而不是和所有层混在一起。
    """

    def __init__(self, cfg: CodeFFNConfig, code_table: WeightCodeTable):
        super().__init__()
        self.w1 = CodeGenLinear(cfg.d_model, cfg.d_ff, code_table,
                                rank=cfg.rank, use_scale=cfg.use_scale)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        nn.init.normal_(self.w2.weight, std=0.02)

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x, token_ids)))

    def param_report(self) -> dict:
        code_params = self.w1.U.numel() + self.w1.V.numel()
        if self.w1.scale is not None:
            code_params += self.w1.scale.numel()
        shared = self.w2.weight.numel()
        return {"code_basis": code_params, "shared_w2": shared,
                "total": code_params + shared}


# ---------------------------------------------------------------- 参数账

def compare_param_budget(d_model: int, d_ff: int, vocab: int,
                         k: int, rank: int, param_bits: int = 64) -> dict:
    """对比标准 FFN 与码生成 FFN 的参数量，用于选定等预算配置。"""
    std_ffn = 2 * d_model * d_ff

    # 码生成 FFN：w1 用码生成（k·r·(d_ff+d_model)），w2 保持标准（d_ff·d_model）
    code_w1 = k * rank * (d_ff + d_model)
    std_w2 = d_ff * d_model
    code_table = vocab * param_bits // 8

    return {
        "standard_ffn": std_ffn,
        "code_ffn_basis": code_w1 + std_w2,
        "code_table_bytes": code_table,
        "code_ffn_total": code_w1 + std_w2,
        "ratio_vs_standard": (code_w1 + std_w2) / std_ffn,
        "extra_per_token_bytes": param_bits // 8,
    }
