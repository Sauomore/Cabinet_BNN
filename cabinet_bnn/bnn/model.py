# -*- coding: utf-8 -*-
"""
Cabinet-BNN：以 HSH-64 权重码为参数载体的二值激活网络。

设计约束（路线文档 v2 §1.3 / §3）：
  ① 激活二值 ±1             —— 前向里每个激活只承载 1 bit
  ② 权重浮点可训            —— 数值范围只能靠权重提供
  ③ 无全局权重，per-token  —— W_t 由该 token 的权重码生成

权重生成（路线文档式 1 + 定义 2）：
    s_j(t) = mask_j(hash_t) · p_t[j mod P]
    W_t    = Σ_{j=1..k} s_j(t) · B_j          其中 B_j = U_j V_jᵀ（低秩）

几何隔离（路线文档定理 1）：同 (feat,sim) 桶内两 token 的权重差异
    只由 param 段 p 决定，与共享的 mask 无关。

为什么默认用 128 位码：
    64 位码 = feat(4)+sim(52)+abs(8)，abs 只有 256 个取值。
    3109 词表下必然有约 12 个 token 共享同一套 per-token 参数，
    无法做严格的 per-token 权重消融。故默认 hash(64)+param(64)=128 位，
    并把 param_bits 做成可调参数以支持退化配置的消融。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


# =====================================================================
# 基础算子
# =====================================================================

def binary_activation(x: torch.Tensor) -> torch.Tensor:
    """二值激活 sign(x)，STE 直通。

    前向：b = sign(x) ∈ {−1, +1}
    反向：∂b/∂x ≈ 1（直通估计器）
    """
    return x + (torch.sign(x) - x).detach()


ste_sign = binary_activation      # 别名，与 HSH-64 代码风格统一


# =====================================================================
# 权重码
# =====================================================================

@dataclass
class WeightCodeConfig:
    """权重码配置。"""
    sim_bits: int = 52              # 语义位（参与检索）
    param_bits: int = 64            # per-token 参数位（不参与检索）
    k_basis: int = 64               # 基矩阵数量 = 符号向量维度
    rank: int = 32                  # 低秩分解的秩

    @property
    def total_bits(self) -> int:
        return 64 + self.param_bits  # hash 段固定 64 位 + param 段


class WeightCodeTable(nn.Module):
    """可写的权重码表。

    hash  段（64 位）：语义码，默认外部注入且冻结（来自 HSH-64 投影头）
    param 段（P  位）：per-token 参数，本模型的可学部分

    码本身是离散 buffer（不参与梯度）；梯度先累积在影子参数 shadow 上，
    越过滞回阈值后才真正翻位（路线文档 §4.2）。
    """

    def __init__(self, n_tokens: int, cfg: WeightCodeConfig, freeze_hash: bool = True):
        super().__init__()
        self.cfg = cfg
        self.n_tokens = n_tokens
        self.freeze_hash = freeze_hash

        self.register_buffer("hash_code", torch.zeros(n_tokens, dtype=torch.int64))
        # 影子参数：小随机初始化 → sign 后为随机的 ±1 模式（块常数见 §3.5）
        self.param_shadow = nn.Parameter(torch.randn(n_tokens, cfg.param_bits) * 0.02)
        self.register_buffer("param_code", torch.ones(n_tokens, cfg.param_bits))
        self.register_buffer("flip_count", torch.zeros((), dtype=torch.long))

    # ---------------- 注入与刷新 ----------------

    @torch.no_grad()
    def set_hash_from_u64(self, codes) -> None:
        """注入 64 位 hash 码。"""
        c = torch.as_tensor(np.asarray(codes), dtype=torch.int64)
        if c.shape != (self.n_tokens,):
            raise ValueError(f"期望形状 ({self.n_tokens},)，实际 {tuple(c.shape)}")
        self.hash_code.copy_(c)

    @torch.no_grad()
    def refresh_codes(self) -> int:
        """按 shadow 的符号刷新 param 码，返回本次翻转位数（滞回机制）。"""
        new = torch.sign(self.param_shadow)
        new[new == 0] = 1.0
        flipped = int((new != self.param_code).sum().item())
        self.param_code.copy_(new)
        self.flip_count += flipped
        return flipped

    # ---------------- 符号模式 ----------------

    def sign_patterns(self) -> torch.Tensor:
        """每个 token 的符号向量 s ∈ {−1,+1}^{N×k}。

        s_j = mask_j(hash) · p[j mod P]
        mask 由 hash 段低 k 位给出（bit=1 → +1，bit=0 → −1）。

        关键：param 段必须走 STE。若直接用 torch.sign(param_shadow)，
        该函数导数几乎处处为 0，梯度永远无法到达 param_shadow，
        per-token 参数就学不到任何东西（这正是首版冒烟测试暴露的 bug）。
        这里用 ste_sign 让梯度直通，离散码仍由 refresh_codes() 的滞回机制更新。
        """
        k = self.cfg.k_basis
        shifts = torch.arange(k, device=self.hash_code.device)
        hash_bits = (self.hash_code.unsqueeze(1) >> shifts.unsqueeze(0)) & 1   # (N, k)
        mask = hash_bits.to(self.param_shadow.dtype) * 2.0 - 1.0              # {−1,+1}

        if not self.param_shadow.requires_grad:
            # code_no_param 等消融：param 段不参与，直接返回 mask
            return mask

        p = ste_sign(self.param_shadow)                                       # (N, P)
        if p.shape[1] == k:
            pf = p
        else:
            idx = torch.arange(k, device=p.device) % p.shape[1]
            pf = p[:, idx]
        return mask * pf

    def extra_repr(self) -> str:
        return (f"n_tokens={self.n_tokens}, total_bits={self.cfg.total_bits}, "
                f"k={self.cfg.k_basis}, rank={self.cfg.rank}")


# =====================================================================
# 由码生成权重的二值激活层
# =====================================================================

class CodeWeightLinear(nn.Module):
    """无全局权重层：权重由 token 的码即时组合共享基矩阵得到。

        W_t = Σ_j s_j(t) · (U_j V_jᵀ)
        y   = scale ⊙ sign(W_t x)

    scale 是共享的逐通道浮点因子（二值网络标配，见路线文档 D7b），
    它不构成"全局权重"，只是给二值激活找回幅度信息。
    """

    def __init__(self, d_in: int, d_out: int, code_table: WeightCodeTable,
                 use_scale: bool = True):
        super().__init__()
        cfg = code_table.cfg
        self.d_in, self.d_out = d_in, d_out
        self.code_table = code_table
        k, r = cfg.k_basis, cfg.rank

        bound = (1.0 / max(d_in, 1)) ** 0.5
        self.U = nn.Parameter(torch.randn(k, r, d_out) * bound)
        self.V = nn.Parameter(torch.randn(k, r, d_in) * bound)
        self.use_scale = use_scale
        self.scale = nn.Parameter(torch.ones(d_out)) if use_scale else None

    def effective_weights(self) -> torch.Tensor:
        """W ∈ (N, d_out, d_in)。"""
        s = self.code_table.sign_patterns()                  # (N, k)
        B = torch.einsum("jro,jri->joi", self.U, self.V)      # (k, d_out, d_in)
        return torch.einsum("nj,joi->noi", s, B)             # (N, d_out, d_in)

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """x: (B, d_in)；token_ids: (B,) 决定每个样本用哪套权重。"""
        W = self.effective_weights()
        Wb = W[token_ids]                                    # (B, d_out, d_in)
        y = torch.einsum("bod,bd->bo", Wb, x)
        if self.scale is not None:
            y = y * self.scale
        return binary_activation(y)

    @torch.no_grad()
    def weight_diversity(self, sample: int = 256) -> float:
        """不同 token 权重矩阵的平均相对差异 —— 检验 per-token 权重是否真的不同。"""
        W = self.effective_weights()
        n = W.shape[0]
        idx = torch.randperm(n, device=W.device)[: min(sample, n)]
        Ws = W[idx]
        Wf = Ws.reshape(Ws.shape[0], -1)
        Wn = Wf / (Wf.norm(dim=1, keepdim=True) + 1e-8)
        sim = Wn @ Wn.T
        off = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)]
        return float(1.0 - off.mean().item())


# =====================================================================
# 模型
# =====================================================================

@dataclass
class BNNConfig:
    char_vocab: int
    n_words: int
    d_model: int = 128
    k_basis: int = 64
    rank: int = 32
    param_bits: int = 64
    use_scale: bool = True
    mode: str = "code"          # code | shared | code_no_param | token_embed
    hidden: int = 0             # shared 模式的隐层宽度；0 表示等于 d_model
                                # 用于把基线参数量对齐到 code 模式，消除混淆
    token_dim: int = 0          # token_embed 模式：每个词一个 d 维向量；0 表示等于 d_model


class BNNWordLM(nn.Module):
    """字符级二值激活语言模型：每个词拥有由权重码生成的唯一权重矩阵。

    前向（逐位置，标准 char-LM）：
        h       = Embed(x)                        # (B, L, d)
        W_t     = Σ_j s_j(t)·(U_j V_jᵀ)           # 该词的专属权重
        h1      = sign(W_t h)                     # 二值激活
        logits  = W'_t h1                         # (B, L, V_char)

    关键：线性层没有全局权重矩阵 —— 权重由该 token 的权重码即时生成。
    同一个词在序列的所有位置共用它自己的那套权重。

    三种模式（M3.6 核心消融）：
        code           完整方案：per-token 权重由码生成
        shared         基线：普通全局权重（消除 per-token 机制）
        code_no_param  消融：param 段置零冻结，只用 hash 段
    """

    def __init__(self, cfg: BNNConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        wcfg = WeightCodeConfig(k_basis=cfg.k_basis, rank=cfg.rank,
                                param_bits=cfg.param_bits)
        self.code_cfg = wcfg
        self.code_table = WeightCodeTable(cfg.n_words, wcfg)

        self.embed = nn.Embedding(cfg.char_vocab, d)
        nn.init.normal_(self.embed.weight, std=0.02)

        if cfg.mode == "shared":
            # hidden 用于把参数量对齐到 code 模式（消除"参数量不同"这一混淆）
            h = cfg.hidden if cfg.hidden > 0 else d
            self.g1 = nn.Linear(d, h)
            self.g2 = nn.Linear(h, cfg.char_vocab)
        elif cfg.mode == "token_embed":
            # 对照基线：经典 per-token 参数 —— 每个词一个可学向量（无共享基、无码）
            td = cfg.token_dim if cfg.token_dim > 0 else d
            self.token_vec = nn.Embedding(cfg.n_words, td)
            nn.init.normal_(self.token_vec.weight, std=0.02)
            self.t1 = nn.Linear(td, d)
            self.t2 = nn.Linear(d, cfg.char_vocab)
        else:
            self.layer1 = CodeWeightLinear(d, d, self.code_table, use_scale=cfg.use_scale)
            self.layer2 = CodeWeightLinear(d, cfg.char_vocab, self.code_table, use_scale=False)

    def forward(self, x: torch.Tensor, word_ids: torch.Tensor) -> torch.Tensor:
        """x: (B, L) 字符索引；word_ids: (B,) 词索引。返回 (B, L, char_vocab)。"""
        h = self.embed(x)                                   # (B, L, d)
        B_, L, d = h.shape

        if self.cfg.mode == "shared":
            return self.g2(torch.tanh(self.g1(h)))          # (B, L, V)

        if self.cfg.mode == "token_embed":
            # 经典做法：per-token 参数直接当作向量用，不加任何结构约束
            tv = self.token_vec(word_ids)                   # (B, td)
            t = torch.tanh(self.t1(tv))                     # (B, d)
            return self.t2(h + t.unsqueeze(1))              # (B, L, V)

        # 权重只依赖 word_ids；按唯一词去重可避免 B 个样本重复算同一套权重
        uniq, inv = torch.unique(word_ids, return_inverse=True)      # (U,), (B,)

        W1 = self.layer1.effective_weights()[uniq]                   # (U, d, d)
        W1b = W1[inv]                                               # (B, d, d)
        y1 = torch.einsum("bld,bde->ble", h, W1b)                   # (B, L, d)
        if self.layer1.use_scale:
            y1 = y1 * self.layer1.scale
        h1 = binary_activation(y1)

        W2 = self.layer2.effective_weights()[uniq]                   # (U, V, d)
        W2b = W2[inv]                                               # (B, V, d)
        return torch.einsum("bld,bvd->blv", h1, W2b)                # (B, L, V)

    @torch.no_grad()
    def param_norm(self) -> float:
        return float(self.code_table.param_shadow.abs().mean().item())

    @torch.no_grad()
    def code_stats(self) -> dict:
        pc = self.code_table.param_code
        shifts = torch.arange(pc.shape[1], dtype=torch.int64, device=pc.device)
        bits = (pc > 0).to(torch.int64)
        vals = (bits * (1 << shifts)).sum(dim=1)
        s = self.code_table.sign_patterns()
        return {
            "n_tokens": int(pc.shape[0]),
            "param_bits": int(pc.shape[1]),
            "unique_param_codes": int(vals.unique().numel()),
            "unique_hash_codes": int(self.code_table.hash_code.unique().numel()),
            "sign_pattern_std": float(s.std().item()),
            "flip_count": int(self.code_table.flip_count.item()),
            "param_shadow_absmean": float(self.code_table.param_shadow.abs().mean().item()),
        }


def build_model(mode: str, char_vocab: int, n_words: int, **kw) -> BNNWordLM:
    """便捷构造。"""
    cfg = BNNConfig(char_vocab=char_vocab, n_words=n_words, mode=mode, **kw)
    return BNNWordLM(cfg)


def count_params(model: nn.Module) -> dict:
    """参数统计：per-token 部分 vs 共享部分。"""
    out = {"per_token": 0, "shared": 0, "total": 0}
    for name, p in model.named_parameters():
        n = p.numel()
        out["total"] += n
        if "param_shadow" in name or "token_vec" in name:
            out["per_token"] += n
        else:
            out["shared"] += n
    return out


def solve_hidden_for_budget(target_total: int, char_vocab: int, d_model: int) -> int:
    """求 shared 模式的隐层宽度 h，使其总参数量接近 target_total。

    shared 模式参数量 = embed(char_vocab×d)
                       + d×h + h + h×char_vocab + char_vocab
    解出 h。用于把基线对齐到 code 模式的参数预算，消除参数量混淆。
    """
    embed = char_vocab * d_model
    fixed = embed + char_vocab
    # 解 (d_model + char_vocab) * h = target_total - fixed
    denom = d_model + char_vocab
    h = int(round((target_total - fixed) / max(denom, 1)))
    return max(h, 8)
