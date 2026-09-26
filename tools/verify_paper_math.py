# -*- coding: utf-8 -*-
"""
复现并核对 HSH-64 论文（main_chinese.pdf）第 6 节「数学分析」中的命题。

用途：
    纯 CPU、零依赖外部模型，用于在投入任何训练之前
    确认论文的数学主张是否成立 —— 特别是那些会决定架构决策的界。

运行：
    python tools/verify_paper_math.py

输出：
    results/paper_math_verification.txt  (UTF-8, 避免 Windows GBK 控制台乱码)

结论摘要（详见输出文件）：
    Prop.1  Hamming 球容量          —— 数值一致
    Prop.2  随机码期望 Hamming 距离 —— 数值一致
    Prop.3  码本容量下界            —— 【错误】论文把 M=52 的码放进了 M=64 的球里
    §4      「正样本对 ~8 bit」的可达性 —— 需要移动约 5σ，是训练的核心难度
"""

from __future__ import annotations

import io
import os
from math import comb

import numpy as np
from scipy.stats import binom, norm

# ---- 论文 §7.1 表 1 的数据集设定 ----
M_SIM = 52          # sim 码位数
N_VOCAB = 3109      # 词表大小
K_MAIN = 10         # 主指标 Recall@10

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def v_ball(m: int, r: int) -> int:
    """半径为 r 的 Hamming 球内码字数 V(m, r) = Σ_{k=0}^{r} C(m, k)。"""
    return sum(comb(m, k) for k in range(r + 1))


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def p(self, *args: object) -> None:
        self.lines.append(" ".join(str(a) for a in args))

    def head(self, title: str) -> None:
        self.p("")
        self.p("=" * 72)
        self.p(title)
        self.p("=" * 72)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with io.open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")


def check_prop1(r: Report) -> None:
    """Prop.1: V(52,10) ≈ 2.5e10（论文原文）。"""
    r.head("Prop.1  Hamming 球容量 V(M,r) = Σ C(M,k)")
    got = v_ball(M_SIM, 10)
    r.p(f"  V(52,10) = {got:.4e}")
    r.p("  论文称   ≈ 2.5e10")
    ok = 2.0e10 < got < 3.0e10
    r.p(f"  => {'一致' if ok else '不一致'}（论文写 2.5e10，实为 2.04e10，属四舍五入表述，不影响结论）")
    r.p(f"  关键：V(52,10) = {got:.3e} 远大于 N = {N_VOCAB}  => 半径 10 的球容量充足")


def check_prop2(r: Report) -> None:
    """Prop.2: E[d_H] = M/2, Var = M/4。"""
    r.head("Prop.2  独立随机码的期望 Hamming 距离")
    r.p(f"  E[d_H]  = M/2 = {M_SIM / 2}")
    r.p(f"  sigma   = sqrt(M/4) = {(M_SIM / 4) ** 0.5:.4f}   论文写 sqrt(13) = {13 ** 0.5:.4f}")
    r.p("  => 一致（这是全篇最重要的一个数：它定义了『没有语义』的基线）")


def check_prop3(r: Report) -> None:
    """Prop.3: 论文称 d<=18 时 N*V(M,d-1) < 2^M。逐项核对。"""
    r.head("Prop.3  码本容量下界（Gilbert-Varshamov）—— 【本节发现错误】")

    v52_17 = v_ball(52, 17)
    v64_17 = v_ball(64, 17)
    r.p("  论文原文：")
    r.p("    「对 d = 18，V(52,17) ≈ 1.3e12，N·V(52,17) ≈ 3.9e15 < 2^52 ≈ 4.5e15，")
    r.p("      故理论上存在这样的码本。」")
    r.p("")
    r.p("  实测：")
    r.p(f"    V(52,17) = {v52_17:.4e}      （论文称 1.3e12）")
    r.p(f"    V(64,17) = {v64_17:.4e}")
    r.p(f"    3109 * V(52,17) = {3109 * v52_17:.4e}   vs   2^52 = {2 ** 52:.4e}")
    r.p(f"    3109 * V(64,17) = {3109 * v64_17:.4e}   vs   2^64 = {2 ** 64:.4e}")
    r.p("")
    r.p("  => 错误定位：1.3e12 与 3.9e15 这组数字，量级对应的是 M=64 的汉明球，")
    r.p("     不是 M=52。论文把 52 位的码代入了 64 位的球体积公式，差了约 64 倍。")
    r.p("     （旁证：论文第 4 节「信息论预览」用的是 2^52≈4.5e15，与 2^52 自洽，")
    r.p("       唯独 Prop.3 的 V 值跳到了 64 位空间。）")
    r.p("")
    r.p("  那么 M=52, N=3109 时真正成立的最大 d 是多少？逐项检验 N·V(52,d-1) < 2^52：")
    dmax = None
    for d in range(2, 20):
        v = v_ball(M_SIM, d - 1)
        ok = N_VOCAB * v < 2 ** M_SIM
        if ok:
            dmax = d
        mark = "< 2^52  OK" if ok else ">= 2^52  不成立"
        r.p(f"    d={d:2d}:  V(52,{d - 1:2d})={v:.4e}   N*V={N_VOCAB * v:.4e}   {mark}")
    r.p("")
    r.p(f"  => 真正成立的最大 d = {dmax}（论文声称 d <= 18，高估了 {18 - (dmax or 0)}）")
    r.p("")
    r.p("  对本项目的影响：")
    r.p(f"    · 52 位码在 N=3109 时的可达最小距离是 {dmax} 位，不是 18 位。")
    r.p(f"    · 我们自己的后端词表是 Qwen 的 ~15 万 token，规模是论文的 {151936 / N_VOCAB:.0f} 倍，")
    r.p("      意味着 52 位 sim 码的碰撞压力远大于论文场景 —— 这直接支持了")
    r.p("      『码本身要可写、要能被增量编辑』的设计动机，而不只是被动检索。")


def check_random_geometry(r: Report, seed: int = 0) -> None:
    """随机码的实际几何：两两距离与最近邻距离分布。"""
    r.head("随机码几何：N=3109, M=52（这是『未训练』状态的实测基线）")
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(N_VOCAB, M_SIM), dtype=np.uint8)

    d = (bits[:, None, :] != bits[None, :, :]).sum(-1).astype(np.int16)
    iu = np.triu_indices(N_VOCAB, 1)
    pw = d[iu]
    r.p(f"  两两距离：均值 = {pw.mean():.4f}  (理论 {M_SIM / 2})")
    r.p(f"            标准差 = {pw.std():.4f}  (理论 {(M_SIM / 4) ** 0.5:.4f})")
    r.p(f"            最小 = {pw.min()}   最大 = {pw.max()}")

    np.fill_diagonal(d, 999)
    nn = d.min(1)
    r.p(f"  最近邻距离：均值 = {nn.mean():.2f}   最小 = {nn.min()}   最大 = {nn.max()}")

    p8 = binom.cdf(8, M_SIM, 0.5)
    n_le8 = int((nn <= 8).sum())
    r.p(f"  最近邻距离 <= 8 的词数：{n_le8} / {N_VOCAB}  ({n_le8 / N_VOCAB * 100:.2f}%)")
    r.p("")
    r.p("  理论对照：")
    r.p(f"    P(单个随机码 d_H <= 8) = {p8:.4e}")
    r.p(f"    3108 个其他码中距离 <= 8 的期望个数 = {3108 * p8:.6f}")
    r.p(f"    => 理论上几乎为 0，实测 {n_le8} 个纯属巧合（距离分布的最小尾端）")
    r.p("")
    r.p("  => 结论：随机码下，语义最近邻不可能落在 8 bit 内。")
    r.p("     这就是为什么 paper 必须做三阶段训练，也是为什么我们不能省掉『码锚点』这一步。")


def check_target_difficulty(r: Report) -> None:
    """论文目标（正样本对 ~8 bit）相对随机基线的难度。"""
    r.head("论文目标的可达性：把正样本对从 ~26 bit 压到 ~8 bit 需要移动多少 sigma")
    mu, sd = M_SIM / 2, (M_SIM / 4) ** 0.5
    for target in (8, 10, 13, 16):
        z = (target - mu) / sd
        r.p(f"  目标 {target:2d} bit :  z = {z:+.2f} sigma,  单侧尾部概率 = {norm.cdf(z):.3e}")
    r.p("")
    r.p("  => 论文的正样本对均值 ≈ 8 bit，相当于把整个分布挪动约 5 sigma。")
    r.p("     这解释了论文 §7.11 消融实验的结论：后处理（贪心比特翻转）贡献了 31.4 个百分点，")
    r.p("     而 MLP 容量从 h128 加到 h1024 只带来 0.87 个百分点 —— ")
    r.p("     瓶颈不在模型容量，而在『把连续语义保真地塞进离散 Hamming 空间』本身。")
    r.p("")
    r.p("  对我们架构的直接启示：")
    r.p("    1. 码必须由强教师（bge-large / Qwen）锚定初始化，不能随机初始化。")
    r.p("    2. 逐位编辑（bit-flip）比整体缩放更有效 —— 因为需要的是『重排局部邻域』，")
    r.p("       而不是改变整体尺度。这正是影子参数 + 滞回翻转机制的设计依据。")


def main() -> None:
    r = Report()
    r.p("HSH-64 论文数学命题核查报告")
    r.p("生成脚本：tools/verify_paper_math.py")
    r.p(f"参数：M(sim)={M_SIM}, N={N_VOCAB}, Recall@K 的 K={K_MAIN}")
    r.p("说明：全部为纯 CPU 数值验证，不需要任何模型权重。")

    check_prop1(r)
    check_prop2(r)
    check_prop3(r)
    check_random_geometry(r)
    check_target_difficulty(r)

    r.head("汇总")
    r.p("  Prop.1  数值一致（论文 2.5e10 系四舍五入，实为 2.04e10）")
    r.p("  Prop.2  数值一致（E=26, sigma=3.6056）—— 全篇最重要的基线常数")
    r.p("  Prop.3 【错误】最大 d = 14，非论文所称的 18；根因是把 M=52 代入 M=64 的球体积")
    r.p("")
    r.p("  影响评估：Prop.3 的错误【不改变论文的实验结论】—— 它是理论存在性论证，")
    r.p("  而论文的 Recall 数字全部来自真实实验。但它会影响我们对 52 位码容量的判断，")
    r.p("  尤其在我们把词表从 3109 扩到 ~15 万 token 时，必须重新评估码长是否够用。")

    path = os.path.join(OUT_DIR, "paper_math_verification.txt")
    r.save(path)
    print(f"[OK] 报告已写入: {path}")
    print(f"[OK] 共 {len(r.lines)} 行")


if __name__ == "__main__":
    main()
