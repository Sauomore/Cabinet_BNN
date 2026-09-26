# -*- coding: utf-8 -*-
"""
MoE 式稀疏 FFN：用【码本身】做路由，零额外路由参数。

设计（与作者意图一致：码是索引，且要让「改动只碰一部分」）：

    标准 FFN:      W_t = Σ_{j=1..k}   s_j(t)·B_j        用全部 k 个基
    本模块:        W_t = Σ_{j∈TopK'}  s_j(t)·B_j        只用 k' 个

三个设计点：

① 路由来自码，不引入 router 网络
   标准 MoE 需要一个可学的 router（额外参数 + 负载均衡损失）。
   这里 TopK 直接取自 hash 段的高位 —— 零参数，且语义近的 token 自动
   路由到相近的基（这正是 HSH 码的性质）。

② 稀疏【写】而不只是稀疏【读】
   改一个 token 的信息码时，只有它激活的那 k' 个基的系数变化，
   爆炸半径 = k'/k。这是「参数热更改只调部分」的直接实现。

③ 负载均衡靠结构而非损失项
   路由由均匀分布的码位决定 -> 天然均衡，不需要 load-balancing loss
   （省掉一个难调的 λ）。另提供 expert-choice 变体，保证每个专家都被用到。

参数账（d=512, k=16, r=16, d_ff=256, k'=4）：
    基矩阵 U,V : k·r·(d_ff + d) = 16·16·768          = 196,608
    down 投影  : d_ff·d                             = 131,072
    路由参数   : 0
    合计 327,680  vs 标准 FFN(d_ff=2048) 2,097,152  = 15.6%
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import WeightCodeTable


@dataclass
class MoEConfig:
    d_model: int = 512
    d_ff: int = 256            # 每个基的中间维度
    k_experts: int = 16        # 基矩阵总数
    k_active: int = 4          # 每个 token 激活几个
    rank: int = 16             # 每个基的低秩
    topk_mode: str = "token"   # token | expert


class MoECodeFFN(nn.Module):
    """码路由的稀疏 FFN。

    前向：
        r_t      = TopK(路由分数)                    由码决定
        A_s, B_s = Σ_{j∈r_t} w_j · U_j,  Σ w_j · V_j   折叠到 (n, rank, ·)
        h        = A_sᵀ( B_s · x )                     两级 einsum，不展开大矩阵
        y        = W_down( silu(h) )

    Args:
        cfg:      MoEConfig
        code_table: 提供逐 token 的路由分数
    """

    def __init__(self, cfg: MoEConfig, code_table: WeightCodeTable):
        super().__init__()
        self.cfg = cfg
        self.code_table = code_table
        d, d_ff, k, r = cfg.d_model, cfg.d_ff, cfg.k_experts, cfg.rank

        # ---- 初始化：解析求解，不试探 ----
        #
        # 目标：MoE 的残差输出 std 应与标准 FFN 相同（实测标准值 0.2176，
        #       在 d=512, d_ff=2048, x~N(0,1) 下）。
        #       残差支路过弱 -> 训练起步 loss 高（曾达 11.5, gnorm=inf）。
        #
        # 推导（各量都为 std，且已用数值验证）：
        #   设 V_j ~ U(±a)，折叠后 B_s = Σ_{j∈k'} w_j V_j，w 归一化后各约 1/k'。
        #   k' 个独立同分布项按 1/k' 加权求和 -> std(B_s) = a/sqrt(3) / sqrt(k')
        #   同理 std(A_s) = b/sqrt(3) / sqrt(k')，b 为 U_j 的均匀分布上界。
        #
        #   前向 h = A_sᵀ(B_s x)：
        #       B_s·x :  R^d 上 r 维求和 -> std ≈ std(B_s)·sqrt(d)·std(x)
        #               （转置后的 A_sᵀ 同理，r 维求和）
        #       A_sᵀh  : std ≈ std(A_s)·sqrt(r)·std(h)
        #   选 a = 1/sqrt(d)（与标准 FFN 同），解出 b 与 down 的界 c。
        #
        # 之所以不照抄 nn.Linear 的默认初始化：MoE 的 down 输入维度只有 d_ff，
        # 而标准 w2 的输入维度是 2048 —— 同样的初始化常数会给出不同增益。
        a = (1.0 / max(d, 1)) ** 0.5            # V_j 的均匀分布上界
        b = (1.0 / max(d_ff, 1)) ** 0.5         # U_j 的均匀分布上界
        self.V = nn.Parameter(torch.empty(k, r, d).uniform_(-a, a))
        self.U = nn.Parameter(torch.empty(k, r, d_ff).uniform_(-b, b))
        self.down = nn.Linear(d_ff, d, bias=False)
        # down 的上界按目标输出量级解析求解（见上推导）
        c = self._solve_down_bound(d, d_ff, r, cfg.k_active, a, b, target=0.2176)
        nn.init.uniform_(self.down.weight, -c, c)
        self._init_bound = {"V": a, "U": b, "down": c}

        # 词表大小直接取自 hash 码表的行数（WeightCodeTable 本身是 nn.Module，
        # 其 cfg 里没有 n_words 字段）
        self.n_words = int(code_table.hash_code.shape[0])
        self._codes_checked = False

    @staticmethod
    def _solve_down_bound(d: int, d_ff: int, r: int, k_active: int,
                          a: float, b: float, target: float) -> float:
        """解析求出 down 的均匀分布上界，使输出 std 达到目标值。

        链式推导（std 传播，均匀分布 std = bound/sqrt(3)）：
            std(B_s) = (a/sqrt(3)) / sqrt(k')
            std(h)   = std(B_s) * sqrt(d) * std_x          (std_x = 1)
            std(A_s) = (b/sqrt(3)) / sqrt(k')
            std(y)   = std(A_s) * sqrt(r) * std(h)
            std(out) = (c/sqrt(3)) * sqrt(d_ff) * std(y)   (silu 近似线性区)
        令 std(out) = target，解出 c。
        """
        import math
        std_x = 1.0
        std_Bs = (a / math.sqrt(3)) / math.sqrt(max(k_active, 1))
        std_h = std_Bs * math.sqrt(d) * std_x
        std_As = (b / math.sqrt(3)) / math.sqrt(max(k_active, 1))
        std_y = std_As * math.sqrt(r) * std_h
        # target = (c/sqrt(3)) * sqrt(d_ff) * std_y
        c = target * math.sqrt(3) / (math.sqrt(d_ff) * max(std_y, 1e-12))
        return float(c)

        # 词表大小直接取自 hash 码表的行数（WeightCodeTable 本身是 nn.Module，
        # 其 cfg 里没有 n_words 字段）
        self.n_words = int(code_table.hash_code.shape[0])
        self._codes_checked = False

    def _check_codes_usable(self) -> None:
        """检查码是否可用于路由 —— 防止【静默坍缩】。

        踩过的坑：WeightCodeTable 的 hash_code 是初始化为全零的缓冲区，
        必须显式 set_hash_from_u64() 或 refresh_codes() 才有值。
        若直接拿全零码做路由，取模恒为 0，16 个专家只用 4 个，
        负载熵掉到 0.5 —— 而代码不会报任何错。

        注意：不能在 __init__ 里做这个检查 —— 码是在模型构造【之后】
        才由训练脚本设置的。改在 forward 首次调用时检查（只查一次）。
        """
        with torch.no_grad():
            h = self.code_table.hash_code
            n_uniq = int(torch.unique(h).numel())
        if n_uniq < 2:
            raise RuntimeError(
                f"路由所用的 hash_code 有 {n_uniq} 个唯一值（共 {h.numel()} 行）—— "
                f"码未初始化，MoE 路由会完全坍缩。\n"
                f"请在构造模型后调用 code_table.set_hash_from_u64(...) 或 "
                f"code_table.refresh_codes() 再开始训练。"
            )
        self._codes_checked = True

    # ---------------- 路由 ----------------

    @torch.no_grad()
    def routing_table(self) -> torch.Tensor:
        """全词表的 TopK 路由（索引 + 权重）。

        由 hash 段决定 —— 不引入可学 router。
        语义近的 token，hash 相近 -> 路由结果也相近，这是设计意图的一部分。
        """
        V = self.n_words
        return self.route(torch.arange(V))

    def route(self, token_ids: torch.Tensor):
        """按需计算路由（不建全词表表，省显存）。

        路由方案（修正版 —— 初版有负载坍缩）：
            初版用「primary == j」这样的指示位做分数，结果 hash 前几位的
            分布极不均匀，专家 0/1 被 50% 的 token 选中，熵只有 0.84。
            改为【取模分散】：
                expert_j = (h >> off_j) % k      j = 0..k'-1
            - 每个专家等概率（hash 均匀 -> 取模均匀）-> 天然负载均衡
            - 语义近 -> hash 近 -> 取模结果也近（局部性保持）
            - 零参数，不引入 router 网络，不需要 load-balancing loss
            权重由另一组位置不同的位生成，避免与索引位相关。

        Returns:
            idx: (n, k_active) 专家索引，每行互不相同
            w:   (n, k_active) 权重，每行和为 1
        """
        h = self.code_table.hash_code[token_ids]                 # int64
        k, ka = self.cfg.k_experts, self.cfg.k_active
        n = h.shape[0]
        if ka > k:
            raise ValueError(f"k_active({ka}) 不能大于 k_experts({k})")

        # 索引：用 k' 组不同偏移的位取模，保证每行互不相同。
        # 逐列构造并做【回溯式】去重：撞车时往后挪，直到与前面所有列都不同。
        cols = []
        for j in range(ka):
            off = (j * 11) % 60
            c = (h >> off) % k
            for _ in range(k):                                   # 最多试 k 次必成功
                collide = torch.zeros(n, dtype=torch.bool, device=h.device)
                for prev in cols:
                    collide |= (c == prev)
                if not bool(collide.any()):
                    break
                c = torch.where(collide, (c + 1) % k, c)
            cols.append(c)
        idx = torch.stack(cols, dim=1)                           # (n, ka)

        # 权重：用另一组偏移的位，映射到 [0.5, 1.5) 后归一
        wcols = []
        for j in range(ka):
            off = (j * 13 + 29) % 60
            wcols.append(((h >> off) & 0xFFFF).float() / 65535.0 + 0.5)
        wraw = torch.stack(wcols, dim=1)                         # (n, ka)
        w = wraw / wraw.sum(dim=1, keepdim=True)
        return idx, w

    # ---------------- 前向 ----------------

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d) 或 (B, d)；token_ids: (B,)"""
        if not self._codes_checked:
            self._check_codes_usable()
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)
        B, L, d = x.shape
        ka = self.cfg.k_active

        # 按【唯一 token】计算，避免重复
        uniq, inv = torch.unique(token_ids, return_inverse=True)
        idx_u, w_u = self.route(uniq)                            # (U, ka), (U, ka)
        idx = idx_u[inv]                                         # (B, ka)
        w = w_u[inv]                                             # (B, ka)

        # 折叠：Σ_j w_j · U_j  ->  (B, rank, d_ff)
        # ⚠️ 维度对齐（踩过两次）：
        #     U_sel 是 4 维 (B, ka, r, o)。权重必须补到【同样维数】：
        #         wc = w.unsqueeze(-1)          -> (B, ka, 1)       ✗ 错
        #         wc = w[..., None, None]       -> (B, ka, 1, 1)    ✓ 对
        #     因为 (B,ka,1) 与 (B,ka,r,o) 右对齐后是 dim1 的 ka 对 r，会报错；
        #     而若 ka 恰等于 r，广播【不报错】却按错误维度相乘 —— 静默出错。
        #     曾用 einsum("bk,bkro->bro") 也犯过同类错（k 与 o/r 错配）。
        U_sel = self.U[idx]                                      # (B, ka, r, d_ff)
        V_sel = self.V[idx]                                      # (B, ka, r, d)
        wc = w[..., None, None]                                  # (B, ka, 1, 1)
        A = (wc * U_sel).sum(dim=1)                              # (B, r, d_ff)
        Bm = (wc * V_sel).sum(dim=1)                             # (B, r, d)

        h = torch.einsum("brd,bld->brl", Bm, x)                  # (B, r, L)
        y = torch.einsum("bro,brl->blo", A, h)                   # (B, L, d_ff)
        y = self.down(F.silu(y))
        return y.squeeze(1) if squeeze else y

    # ---------------- 诊断 ----------------

    @torch.no_grad()
    def usage_stats(self, n_sample: int = 8192) -> dict:
        """专家使用率分布（检查是否坍缩）。"""
        V = min(self.n_words, n_sample)
        ids = torch.arange(V, device=self.U.device)
        idx, w = self.route(ids)
        counts = torch.bincount(idx.reshape(-1), minlength=self.cfg.k_experts).float()
        p = counts / counts.sum()
        # 归一化熵：1 = 完全均衡，0 = 全挤在一个专家
        ent = float(-(p * (p + 1e-12).log()).sum() / math.log(self.cfg.k_experts))
        used = int((counts > 0).sum())
        ideal = self.cfg.k_active / self.cfg.k_experts
        return {
            "counts": counts.tolist(),
            "max_share": float(p.max()),
            "min_share": float(p.min()),
            "entropy_norm": ent,
            "experts_used": used,
            "experts_total": self.cfg.k_experts,
            "ideal_share": ideal,
            # 坍缩判据：熵过低，或使用的专家数少于总数
            "collapsed": bool(ent < 0.8 or used < self.cfg.k_experts),
        }

    def assert_no_collapse(self, n_sample: int = 8192) -> None:
        """硬性检查：路由必须用满所有专家且分布均衡，否则报错。

        静默坍缩是本模块最容易出的问题（见 _check_codes_usable 的注释），
        因此在训练开始前主动断言，而不是等到有人注意到日志里的熵。
        """
        st = self.usage_stats(n_sample)
        if st["collapsed"]:
            raise RuntimeError(
                f"MoE 路由坍缩：{st['experts_used']}/{st['experts_total']} 个专家被使用，"
                f"归一化熵 {st['entropy_norm']:.4f}（期望接近 1.0）。\n"
                f"使用计数: {[int(x) for x in st['counts']]}\n"
                f"常见原因：hash_code 未初始化、或码的分布过于集中。"
            )

    def param_report(self) -> dict:
        basis = self.U.numel() + self.V.numel()
        down = self.down.weight.numel()
        return {"basis": basis, "down": down, "routing": 0, "total": basis + down}
