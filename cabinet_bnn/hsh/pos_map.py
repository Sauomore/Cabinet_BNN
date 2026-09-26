# -*- coding: utf-8 -*-
"""
词性 → 特征码 feat（4 bit）映射，与 Rust src/pos_map.rs 一致。

注意（实测发现）：参考实现的 examples/benchmark_pure_hsh64.rs 里
【所有词都硬编码传 "n"】，即 feat 恒为 0x0。因此论文报告的数字是在
feat 不携带任何信息的前提下得到的 —— sim 段独立承担全部区分能力。

本模块提供两种模式供消融对比：
  · mode="noun"  : 全部映射到 NOUN（复现论文 benchmark 的行为）
  · mode="jieba" : 用 jieba 词性标注真实分配（检验 feat 段是否有增益）
  · mode="hash"  : 用词本身确定性哈希分配（作为 feat 信息的对照基线）
"""

from __future__ import annotations

import hashlib

# feat 值常量（与 Rust FeatureCode 一致）
NOUN, VERB, ADJ, ADV = 0x0, 0x1, 0x2, 0x3
PRONOUN, PREP, CONJ, AUX = 0x4, 0x5, 0x6, 0x7
NUM, MEASURE, TIME, LOC = 0x8, 0x9, 0xA, 0xB
PUNCT, STRING, COMMON, FALLBACK = 0xC, 0xD, 0xE, 0xF

FEAT_NAMES = {
    NOUN: "名词", VERB: "动词", ADJ: "形容词", ADV: "副词",
    PRONOUN: "代词", PREP: "介词", CONJ: "连词", AUX: "助词",
    NUM: "数词", MEASURE: "量词", TIME: "时间词", LOC: "方位词",
    PUNCT: "标点", STRING: "字符串", COMMON: "常用词", FALLBACK: "兜底",
}

_POS_TO_FEAT = {
    **{p: NOUN for p in ["n", "nr", "nr1", "nr2", "nrj", "nrf", "ns", "nsf",
                         "nt", "nz", "nl", "ng"]},
    **{p: VERB for p in ["v", "vd", "vn", "vf", "vx", "vi", "vl", "vg"]},
    **{p: ADJ for p in ["a", "ad", "an", "ag", "al"]},
    **{p: ADV for p in ["d", "df", "dg"]},
    **{p: PRONOUN for p in ["r", "rr", "rz", "rzt", "rzs", "rzv", "ry",
                            "ryt", "rys", "ryv", "rg", "ryy"]},
    **{p: PREP for p in ["p", "pba", "pbei"]},
    **{p: CONJ for p in ["c", "cc"]},
    **{p: AUX for p in ["u", "ud", "ug", "uj", "ul", "uv", "uz", "y", "z"]},
    **{p: NUM for p in ["m", "mq"]},
    **{p: MEASURE for p in ["q", "qv", "qt"]},
    **{p: TIME for p in ["t", "tg"]},
    **{p: LOC for p in ["f", "fg", "s"]},
    **{p: PUNCT for p in ["w", "wkz", "wky", "wyz", "wyy", "wj", "ww", "wt",
                          "wd", "wf", "wn", "wm", "ws", "wp", "wb", "wh"]},
    **{p: STRING for p in ["x", "xx", "xu", "xi", "wjb", "nx"]},
}


def pos_to_feat(pos: str) -> int:
    """jieba 词性标签 → feat（未知归 FALLBACK）。"""
    return _POS_TO_FEAT.get(pos, FALLBACK)


def assign_feat(words: list[str], mode: str = "noun") -> tuple[list[int], dict]:
    """为词表分配 feat。

    Returns:
        (feat_list, stats)  stats 含每个 feat 的计数
    """
    if mode == "noun":
        feats = [NOUN] * len(words)
    elif mode == "hash":
        feats = [int(hashlib.md5(w.encode("utf-8")).hexdigest()[:2], 16) & 0x0F
                 for w in words]
    elif mode == "jieba":
        import jieba.posseg as pseg
        feats = []
        for w in words:
            pairs = list(pseg.cut(w))
            # 单词多为一整块；取第一个词块的词性
            pos = pairs[0].flag if pairs else "x"
            feats.append(pos_to_feat(pos))
    else:
        raise ValueError(f"不支持的 mode: {mode}")

    stats: dict[int, int] = {}
    for f in feats:
        stats[f] = stats.get(f, 0) + 1
    return feats, stats


def feat_histogram_text(stats: dict[int, int], total: int) -> str:
    """把 feat 分布渲染成可读文本。"""
    lines = []
    for f in sorted(stats):
        name = FEAT_NAMES.get(f, "?")
        c = stats[f]
        lines.append(f"    0x{f:X} {name:<6} {c:>6} ({c / total * 100:5.1f}%)")
    return "\n".join(lines)
