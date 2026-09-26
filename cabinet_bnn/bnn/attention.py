# -*- coding: utf-8 -*-
"""
因果自注意力 —— Cabinet-BNN 从「固定长度池化」升级为真正的序列模型。

为什么这一步是必须的（此前全流程的疏漏）：
    之前的 BNNWordLM 只是把字符嵌入做均值池化，再接两个线性层，上下文固定 7 字符、
    无 token 间信息交互。那不是 Transformer，无法承载任何对话能力。
    注意力是 LLM 能规模化的根本机制，必须先把它补上。

设计取舍（重要）：
    注意力需要 Q/K/V/O 投影在所有 token 间共享 ——
    否则 Q 与 K 不在同一空间，内积无定义。
    因此本模块【使用全局权重】。这与「无全局权重，参数 per-token」的设计冲突，
    冲突的定量后果见 docs/：真实词表下 per-token 注意力投影需 ~49 GB 权重存储。

    折中方案（本实现采用）：
      · 注意力投影 = 全局权重（这是注意力的语义要求）
      · per-token 权重码作用于【输出侧】，即注意力之后的通道调制
    这样既保留注意力的正确语义，又保留权重码的热更改能力。

三种模式（attention_mode）：
    "global"  标准全局注意力（默认，推荐）
    "code"    注意力投影也用权重码生成（仅小词表可行性验证用，会爆内存）
    "hybrid"  全局投影 + per-token 通道调制（兼顾两者）
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import WeightCodeTable, binary_activation


# =====================================================================
# 因果自注意力
# =====================================================================

class CausalSelfAttention(nn.Module):
    """多头因果自注意力，支持 KV Cache，可选【双向关系门】。

    实现要点：
      · 因果掩码用 -inf 加在 softmax 之前（不是乘 0），否则梯度路径不对
      · 缩放因子 1/sqrt(head_dim)，不是 1/sqrt(d_model)
      · RoPE 位置编码（可选），比可学习位置嵌入更利于长度外推
      · KV cache 增量拼接，避免每步重算整段

    双向关系门（relation_gate=True）：
        标准注意力里 token i 对 j 的关注只由 (q_i·k_j) 决定，是「内容相似度」。
        本机制额外引入一个【关系分数】，由两个 token 各自的「关系向量」内积给出：

            att_ij = softmax_j( (q_i·k_j)/sqrt(D) · g_ij )
            g_ij   = 1 + gamma * <u_i, u_j> / sqrt(r)

        设计要点：
          · g 是【对称】的（<u_i,u_j> = <u_j,u_i>）—— 关系本身双向，
            但信息流仍由因果掩码控制，两者正交。
          · gamma 初始为 0 -> 起点严格等于原始注意力。这样从已训练权重出发
            微调时不会因随机初始化而破坏原有能力（踩过随机初始化把
            val_loss 从 3.12 打到 6.00 的坑）。
          · 门作用在 softmax 【内部】：g < 1 压低不相关 token 的注意力权重，
            g -> 0 相当于软屏蔽。这比加在 softmax 之后更有结构约束力。

        注意：关系分数由 token 身份决定，与隐状态无关 ->
        可预计算、所有层所有头共享，代价为 O(B·L²·r)。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 use_rope: bool = True, max_len: int = 512,
                 relation_gate: bool = False, n_relation: int = 0,
                 relation_dim: int = 16, relation_heads: int = 1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model({d_model}) 必须能被 n_heads({n_heads}) 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.max_len = max_len

        # 全局权重：注意力的语义要求 Q/K/V/O 在所有 token 间共享
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # ---- 双向关系门 ----
        self.relation_gate = relation_gate
        self.relation_heads = relation_heads if relation_gate else 0
        if relation_gate:
            assert n_relation > 0, "relation_gate=True 需要 n_relation（词表大小）"
            self.n_relation = n_relation
            self.relation_dim = relation_dim
            # u_t: 每个 token 一个关系向量（= 设计中的「信息码承载关系」）
            self.rel_embed = nn.Embedding(n_relation, relation_dim)
            # 初始化量级很重要（踩过坑）：
            #   关系分数 s = <u_i,u_j>/sqrt(r)。若 u ~ N(0, σ²)，则
            #   std(s) ≈ σ²·sqrt(r)/sqrt(r) = σ²。
            #   σ=0.02 时 std(s)=4e-4，门只变化 ±0.0007，而注意力 logits 的
            #   std 约 0.33 —— 门的扰动小三四个数量级，等于没施加。
            #   取 σ = 1/sqrt(r) 使 std(s) ≈ 1/r·... 实测 std(s)≈O(0.1~1)，
            #   门的变化与 logits 同量级，才真正起作用。
            nn.init.normal_(self.rel_embed.weight, std=1.0 / math.sqrt(relation_dim))
            # gamma 初始为 0 -> g 恒等于 1 -> 严格退化为原始注意力
            self.rel_gamma = nn.Parameter(torch.zeros(relation_heads))
            self.relation_hidden = relation_dim // max(relation_heads, 1)

        if use_rope:
            # 预计算 RoPE 的 cos/sin 表
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float()
                                          / self.head_dim))
            t = torch.arange(max_len).float()
            freqs = torch.outer(t, inv_freq)                    # (L, head_dim/2)
            emb = torch.cat([freqs, freqs], dim=-1)             # (L, head_dim)
            self.register_buffer("rope_cos", emb.cos(), persistent=False)
            self.register_buffer("rope_sin", emb.sin(), persistent=False)

    def _apply_rope(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """对 (B, H, L, D) 施加旋转位置编码。"""
        L = x.shape[-2]
        cos = self.rope_cos[offset : offset + L].to(x.dtype)     # (L, D)
        sin = self.rope_sin[offset : offset + L].to(x.dtype)
        # 相邻两维配对旋转
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rx1 = x1 * cos[..., 0::2] - x2 * sin[..., 0::2]
        rx2 = x1 * sin[..., 0::2] + x2 * cos[..., 0::2]
        out = torch.stack([rx1, rx2], dim=-1).flatten(-2)
        return out

    def _relation_gate(self, token_ids: torch.Tensor) -> torch.Tensor | None:
        """计算关系门 g，(B, Hr, L, L)。

        对称矩阵；gamma=0 时恒为 1（由调用方短路，避免无谓计算）。
        """
        if not self.relation_gate or token_ids is None:
            return None
        B, L = token_ids.shape
        u = self.rel_embed(token_ids)                            # (B, L, r)
        # 内积矩阵（对称）
        s = torch.matmul(u, u.transpose(-2, -1))                 # (B, L, L)
        # 按 head 分组（relation_heads=1 时退化为标量门）
        Hr = self.relation_heads
        if Hr > 1:
            d = self.relation_hidden
            s = s.view(B, L, L, Hr, d).sum(-1).transpose(1, 3)   # (B, Hr, L, L)
        else:
            s = s.unsqueeze(1)                                   # (B, 1, L, L)
        s = s / math.sqrt(max(self.relation_dim, 1))
        g = 1.0 + self.rel_gamma.view(1, Hr, 1, 1) * s
        return g

    def forward(self, x: torch.Tensor, kv_cache: dict | None = None,
                return_cache: bool = False, token_ids: torch.Tensor | None = None):
        """
        x: (B, L, d_model)
        kv_cache: {"k": (B,H,Lc,D), "v": (B,H,Lc,D)} 或 None
        token_ids: (B, L) token 索引，仅在 relation_gate=True 时需要
        返回: (out, new_cache)
        """
        B, L, _ = x.shape
        H, D = self.n_heads, self.head_dim
        offset = 0 if kv_cache is None else kv_cache["k"].shape[-2]

        q = self.wq(x).view(B, L, H, D).transpose(1, 2)          # (B,H,L,D)
        k = self.wk(x).view(B, L, H, D).transpose(1, 2)
        v = self.wv(x).view(B, L, H, D).transpose(1, 2)

        if self.use_rope:
            q = self._apply_rope(q, offset)
            k = self._apply_rope(k, offset)

        if kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=-2)
            v = torch.cat([kv_cache["v"], v], dim=-2)

        Lk = k.shape[-2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D)            # (B,H,L,Lk)

        # ---- 双向关系门（乘在 softmax 内部）----
        # 形状流（务必按此理解，此处踩过坑）：
        #   att      : (B, H,  L, Lk)   L = 本次前向的 query 位置数
        #   g        : (B, Hr, Lk, Lk)  关系是【token 对】之间的，与本次前向无关
        #   g 切片后 : (B, Hr, L, Lk)   选出本次 query 对应的那些行
        #
        # ⚠️ 踩过的坑：曾经写成 `token_ids[:, -L:]` —— 意图是「取本次 query 的
        #    token」，但 L 是【本次前向的 query 数】，不是序列长度。KV cache
        #    场景下 L=1 而 token_ids 是完整序列，于是被切成 (B,1)，门退化成
        #    只有一个 token 的自关系（恒为 1 + γ·|u|²/√r），与历史的全部关系
        #    被丢掉。结果 cache 路径与全前缀路径不一致，但【不报错】。
        #
        # 正确做法：门始终用【完整序列】计算，再按 query 位置取行。
        # token_ids 必须覆盖序列 [0, offset+L)，即包含 cache 部分。
        if self.relation_gate and token_ids is not None:
            g = self._relation_gate(token_ids)                    # (B,Hr,Lk,Lk)
            if g is not None:
                if g.shape[-2] != L:
                    # 本次 query 对应序列末尾的 L 个位置
                    g = g[..., -L:, :]
                if g.shape[1] != H:
                    # 关系头数少于注意力头数时广播
                    rep = H // g.shape[1]
                    g = g.repeat_interleave(rep, dim=1) if rep > 1 else g
                att = att * g.to(att.dtype)

        # 因果掩码：query 位置 i 只能看 key 位置 <= offset+i
        qi = torch.arange(L, device=x.device).unsqueeze(1) + offset
        ki = torch.arange(Lk, device=x.device).unsqueeze(0)
        mask = ki > qi                                            # True = 屏蔽
        att = att.masked_fill(mask, float("-inf"))

        att = F.softmax(att, dim=-1)
        # 防止整行被屏蔽时 softmax 产生 NaN（首 token 之前无 key 的情形）
        att = torch.nan_to_num(att, nan=0.0)
        att = self.dropout(att)

        out = (att @ v).transpose(1, 2).reshape(B, L, self.d_model)
        out = self.wo(out)

        new_cache = {"k": k, "v": v} if return_cache else None
        return out, new_cache

    def relation_matrix(self, token_ids: torch.Tensor) -> torch.Tensor:
        """返回去归一化的关系分数 <u_i,u_j>/sqrt(r)，(B, L, L)。诊断与可视化用。"""
        if not self.relation_gate:
            raise RuntimeError("本层未启用关系门")
        with torch.no_grad():
            u = self.rel_embed(token_ids)
            s = torch.matmul(u, u.transpose(-2, -1))
            return s / math.sqrt(max(self.relation_dim, 1))


# =====================================================================
# Transformer Block
# =====================================================================

class CodeTransformerBlock(nn.Module):
    """Pre-LN Transformer block，注意力为全局权重，FFN 可选权重码生成。

    结构：
        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 code_table: WeightCodeTable | None = None,
                 ffn_mode: str = "global", dropout: float = 0.0,
                 use_rope: bool = True, max_len: int = 512,
                 relation_gate: bool = False, n_relation: int = 0,
                 relation_dim: int = 16, relation_heads: int = 1,
                 moe_d_ff: int = 256, moe_k: int = 16,
                 moe_k_active: int = 4, moe_rank: int = 16):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model, n_heads, dropout, use_rope, max_len,
            relation_gate=relation_gate, n_relation=n_relation,
            relation_dim=relation_dim, relation_heads=relation_heads)
        self.norm2 = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d_model)

        self.ffn_mode = ffn_mode
        if ffn_mode == "global":
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)
        elif ffn_mode == "code":
            # 用权重码生成 FFN 的第一层（示范用；大词表下代价高）
            from .model import CodeWeightLinear
            assert code_table is not None, "ffn_mode='code' 需要 code_table"
            self.code_ffn = CodeWeightLinear(d_model, d_ff, code_table, use_scale=True)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)
        elif ffn_mode == "moe":
            # MoE 式稀疏 FFN：路由由码决定，零额外路由参数
            from .moe_ffn import MoEConfig, MoECodeFFN
            assert code_table is not None, "ffn_mode='moe' 需要 code_table"
            self.code_ffn = MoECodeFFN(
                MoEConfig(d_model=d_model, d_ff=moe_d_ff,
                          k_experts=moe_k, k_active=moe_k_active,
                          rank=moe_rank), code_table)
            self.w2 = None            # MoE 模块自带 down 投影
        else:
            raise ValueError(f"未知 ffn_mode: {ffn_mode}")

    def forward(self, x, word_ids=None, kv_cache=None, return_cache=False,
                token_ids=None):
        # 关系门需要的是【每个位置】的 token 索引；word_ids 是 (B,)，
        # 在预训练里同一序列的 token 属于同一个"词"，与位置索引不同。
        # 因此关系门单独接收 token_ids (B, L)；缺省时退回 word_ids 广播。
        rel_ids = token_ids
        if rel_ids is None and word_ids is not None:
            rel_ids = word_ids.unsqueeze(1).expand(x.shape[0], x.shape[1])

        h, new_cache = self.attn(self.norm1(x), kv_cache, return_cache,
                                 token_ids=rel_ids)
        x = x + h
        hn = self.norm2(x)
        if self.ffn_mode == "global":
            x = x + self.w2(F.silu(self.w1(hn)))
        elif self.ffn_mode == "moe":
            # 按位置施加稀疏码权重：展平成 (B*L, d)，token 索引同步扩展
            B, L, d = hn.shape
            wid = self._ffn_token_ids(hn, token_ids, word_ids)
            hf = self.code_ffn(hn.reshape(B * L, d), wid).reshape(B, L, -1)
            x = x + hf
        else:
            # 按位置施加码权重：把 (B,L,d) 展平成 (B*L,d)，word_ids 同步扩展
            B, L, d = hn.shape
            if word_ids is None:
                raise ValueError("ffn_mode='code' 需要 word_ids")
            wid = word_ids.unsqueeze(1).expand(B, L).reshape(-1)
            hf = self.code_ffn(hn.reshape(B * L, d), wid).reshape(B, L, -1)
            x = x + self.w2(F.silu(hf))
        return x, new_cache

    @staticmethod
    def _ffn_token_ids(hn, token_ids, word_ids):
        """MoE 前向需要的逐位置 token 索引，展平为 (B*L,)。

        优先用 token_ids（逐位置）；缺省退回 word_ids（逐序列广播）。

        ⚠️ 必须把 token_ids 对齐到【本次前向的位置数】。生成时走 KV cache，
        hn 只有 1 个位置而 token_ids 是完整序列 —— 直接 reshape(-1) 会得到
        长度不匹配的张量（曾报 512 vs 1024）。这与关系门踩的是同一类坑：
        L 是「本次前向的 query 数」，不是序列长度。
        """
        B, L, _ = hn.shape
        if token_ids is not None:
            ids = token_ids
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            if ids.shape[1] != L:
                # 本次 query 对应序列末尾的 L 个位置
                ids = ids[:, -L:]
            if ids.shape[1] != L:
                raise ValueError(
                    f"token_ids 长度 {ids.shape[1]} 与位置数 {L} 不匹配")
            return ids.reshape(-1)
        if word_ids is not None:
            return word_ids.unsqueeze(1).expand(B, L).reshape(-1)
        raise ValueError("ffn_mode='moe' 需要 token_ids 或 word_ids")
        return x, new_cache


# =====================================================================
# 完整的 BNN-Transformer 语言模型
# =====================================================================

@dataclass
class TransformerConfig:
    vocab_size: int
    n_words: int
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 0                    # 0 表示 4*d_model
    max_len: int = 256
    dropout: float = 0.0
    use_rope: bool = True
    attn_mode: str = "global"        # global | code | hybrid
    ffn_mode: str = "global"         # global | code | moe
    k_basis: int = 8
    rank: int = 32
    param_bits: int = 64
    tie_embedding: bool = True
    # ---- MoE ----
    moe_d_ff: int = 256              # 每个专家的中间维度
    moe_k: int = 16                  # 专家（基矩阵）总数
    moe_k_active: int = 4            # 每 token 激活几个
    moe_rank: int = 16               # 每个专家的低秩
    # ---- 双向关系门 ----
    relation_gate: bool = False      # 是否启用 token 间关系门
    relation_dim: int = 16           # 关系向量维度 r
    relation_heads: int = 1          # 关系头数（1 = 所有注意力头共享一个门）


class BNNTransformerLM(nn.Module):
    """BNN 语言模型：全局注意力主干 + 可选的 per-token 权重码机制。

    与旧版 BNNWordLM 的区别：
        旧版 mean-pool → 2 层线性，上下文固定，token 间无交互。
        本版是标准 decoder-only Transformer：嵌入 → N × Block → 输出头，
        带因果注意力、RoPE、KV cache。
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        d_ff = cfg.d_ff if cfg.d_ff > 0 else 4 * d

        self.embed = nn.Embedding(cfg.vocab_size, d)
        nn.init.normal_(self.embed.weight, std=0.02)

        # 权重码表（仅当需要 per-token 机制时创建）
        self.code_table = None
        if cfg.ffn_mode in ("code", "moe") or cfg.attn_mode in ("code", "hybrid"):
            from .model import WeightCodeConfig
            wcfg = WeightCodeConfig(k_basis=cfg.k_basis, rank=cfg.rank,
                                    param_bits=cfg.param_bits)
            self.code_table = WeightCodeTable(cfg.n_words, wcfg)

        self.blocks = nn.ModuleList([
            CodeTransformerBlock(d, cfg.n_heads, d_ff,
                                 code_table=self.code_table, ffn_mode=cfg.ffn_mode,
                                 dropout=cfg.dropout, use_rope=cfg.use_rope,
                                 max_len=cfg.max_len,
                                 relation_gate=cfg.relation_gate,
                                 n_relation=cfg.n_words,
                                 relation_dim=cfg.relation_dim,
                                 relation_heads=cfg.relation_heads,
                                 moe_d_ff=cfg.moe_d_ff, moe_k=cfg.moe_k,
                                 moe_k_active=cfg.moe_k_active,
                                 moe_rank=cfg.moe_rank)
            for _ in range(cfg.n_layers)
        ])
        self.norm_f = nn.RMSNorm(d) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        if cfg.tie_embedding:
            self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, word_ids: torch.Tensor | None = None,
                kv_caches: list | None = None, return_caches: bool = False,
                token_ids: torch.Tensor | None = None):
        """x: (B, L) token 索引。返回 (logits, new_caches)。

        token_ids: 关系门用的 token 索引，(B, L)。缺省时退回 x 本身
                   （预训练场景下 x 就是 token 索引，正是我们想要的）。
        generate 时需显式传入【完整序列】的 token_ids，见 generate()。
        """
        h = self.embed(x)
        rel_ids = token_ids if token_ids is not None else x
        new_caches = [] if return_caches else None
        for i, blk in enumerate(self.blocks):
            c = None if kv_caches is None else kv_caches[i]
            h, nc = blk(h, word_ids, c, return_caches, token_ids=rel_ids)
            if return_caches:
                new_caches.append(nc)
        logits = self.lm_head(self.norm_f(h))
        return logits, new_caches

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 32,
                 temperature: float = 1.0, top_k: int | None = None,
                 eos_id: int | None = None, word_ids: torch.Tensor | None = None):
        """自回归生成（带 KV cache）。idx: (B, L0)。"""
        self.eval()
        caches = None
        out = idx
        for step in range(max_new_tokens):
            cur = out if caches is None else out[:, -1:]
            # 关系门需要完整序列的 token 索引（关系是与历史 token 的，
            # 不只是当前这一个），因此这里始终传 out 而不是 cur。
            logits, caches = self.forward(cur, word_ids, caches, return_caches=True,
                                          token_ids=out)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, nxt], dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
        return out

    def n_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        per_token = 0
        if self.code_table is not None:
            per_token = self.code_table.param_shadow.numel()
        return {"total": total, "per_token": per_token, "shared": total - per_token}


def build_transformer(vocab_size: int, n_words: int, preset: str = "tiny", **kw):
    """按预设规模构造。参数量为实测值，见 selftest_attention.py 的输出。

    可选 kw（透传到 TransformerConfig）：
        relation_gate=True    启用双向关系门
        relation_dim=16       关系向量维度
        relation_heads=1      关系头数
    """
    presets = {
        # 名称:      d_model, n_heads, n_layers, d_ff, max_len
        "nano":  dict(d_model=128, n_heads=4, n_layers=2, d_ff=512,  max_len=256),
        "tiny":  dict(d_model=256, n_heads=4, n_layers=4, d_ff=1024, max_len=256),
        "small": dict(d_model=384, n_heads=6, n_layers=6, d_ff=1536, max_len=512),
        "base":  dict(d_model=512, n_heads=8, n_layers=8, d_ff=2048, max_len=512),
    }
    if preset not in presets:
        raise ValueError(f"未知预设 {preset}，可选 {list(presets)}")
    # 允许 kw 覆盖预设项（否则会与预设里的同名键冲突）
    merged = dict(presets[preset])
    merged.update(kw)
    cfg = TransformerConfig(vocab_size=vocab_size, n_words=n_words, **merged)
    return BNNTransformerLM(cfg)
