# -*- coding: utf-8 -*-
"""
把单个词表切分成若干「域」，用于持续学习 / 遗忘量评测。

为什么要合成域：
    本机没有多领域标注语料，而「不遗忘」这件事必须在**顺序学习多个域**的
    设定下才能测。用确定性规则把词表切分，可以精确控制域的大小与重叠度，
    且完全可复现 —— 比引入不可控的真实语料更适合机制验证。

划分规则（确定性）：
    以词的首字符哈希对 n_domains 取模。相同首字符的词必然落入同一域，
    因此域内词在字符分布上高度相似 —— 这正是我们希望「域适配」能捕获的结构。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass
class DomainSplit:
    """域划分结果。"""
    n_domains: int
    word_domain: np.ndarray          # (N,) 每个词属于哪个域
    domain_words: list[np.ndarray]   # 每域的词索引
    domain_names: list[str]

    def stats(self) -> str:
        lines = [f"  域数 {self.n_domains}"]
        for d in range(self.n_domains):
            lines.append(f"    域 {d}: {len(self.domain_words[d]):>5} 词")
        sizes = np.array([len(w) for w in self.domain_words])
        lines.append(f"  域大小: 最小 {sizes.min()} / 最大 {sizes.max()} / "
                     f"标准差 {sizes.std():.1f}")
        return "\n".join(lines)


def make_domains(words: list[str], n_domains: int = 10, seed: int = 0) -> DomainSplit:
    """按首字符哈希切分域。"""
    if not words:
        raise ValueError("词表为空")

    assign = np.zeros(len(words), dtype=np.int64)
    for i, w in enumerate(words):
        key = (w[0] if w else "") + f"#{seed}"
        h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
        assign[i] = h % n_domains

    # 处理空域：把最大的域拆出一些给空域
    for d in range(n_domains):
        if (assign == d).sum() == 0:
            biggest = int(np.argmax([(assign == k).sum() for k in range(n_domains)]))
            victims = np.flatnonzero(assign == biggest)
            if len(victims) > 1:
                assign[victims[-1]] = d

    domain_words = [np.flatnonzero(assign == d) for d in range(n_domains)]
    names = [f"D{d}" for d in range(n_domains)]
    return DomainSplit(n_domains, assign, domain_words, names)


def domain_stratified_split(
    dom: DomainSplit,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """每个域内部再切训练/验证集（保证每域都有验证样本）。"""
    rng = np.random.default_rng(seed)
    train: dict[int, np.ndarray] = {}
    val: dict[int, np.ndarray] = {}
    for d in range(dom.n_domains):
        idx = dom.domain_words[d].copy()
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_ratio))
        val[d] = idx[:n_val]
        train[d] = idx[n_val:]
    return train, val
