# -*- coding: utf-8 -*-
"""
语料摸底：为「重新做一个基座」提供决策依据。

昨天失败的教训：
    我引入了 65% 的 skypile（网页抓取）数据，但【从未评估过它的质量】。
    结果基座的事实探针从 7/12 掉到 2/12，置信度整体崩塌。

所以这次先量，再决定。本脚本回答四个问题：

  Q1  指令数据里，到底有多少是【模板化格式】？（空表格、重复列表、代码块…）
      —— 这决定「格式过拟合」这个诊断是否成立、以及该过滤多少

  Q2  三个来源的【事实密度】如何？（用探针词的出现率近似）
      —— 事实密度高的来源才该进预训练语料

  Q3  各来源的文本质量（重复率、异常字符率）

  Q4  过滤掉模板样本后，还剩多少可用数据？

用法：
    python scripts/34_corpus_survey.py
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from cabinet_bnn.paths import corpus_dir

# ---- 模板化格式的特征 ----
TEMPLATE_PATTERNS = {
    "空表格行": re.compile(r"^\s*\|[\s|]*\|\s*$"),
    "表格分隔": re.compile(r"^\s*\|?[\s:-]*\|[\s:|-]*$"),
    "代码块": re.compile(r"```"),
    "重复列表项": re.compile(r"(\d+[.、]\s*\S{1,6}\s*){4,}"),
    "连续重复词": re.compile(r"(.{2,6})\1{3,}"),
    "编号堆砌": re.compile(r"[、，]\s*\d+[、，]\s*\d+[、，]\s*\d+"),
}

# ---- 事实探针里用到的关键词（近似衡量知识密度）----
FACT_WORDS = ["北京", "首都", "中国", "太阳", "地球", "水", "化学",
              "曹雪芹", "红楼梦", "巴黎", "法国", "等于", "因为", "答案"]


def read_samples(path: Path, fn, n_obj: int, max_bytes: int) -> list[str]:
    """用生成语料时的同一套读取器抽样，保证和训练时看到的一致。"""
    out = []
    used = 0
    for line in fn(path, max_bytes):
        out.append(line.rstrip("\n"))
        used += len(line.encode("utf-8"))
        if len(out) >= n_obj:
            break
    return out


def analyze(name: str, texts: list[str]) -> dict:
    n = len(texts)
    if n == 0:
        return {"name": name, "n": 0}
    total_chars = sum(len(t) for t in texts)
    tmpl_hits = Counter()
    n_tmpl = 0
    for t in texts:
        hit = False
        for k, pat in TEMPLATE_PATTERNS.items():
            if pat.search(t):
                tmpl_hits[k] += 1
                hit = True
        if hit:
            n_tmpl += 1
    fact_cnt = sum(1 for t in texts if any(w in t for w in FACT_WORDS))
    # 字符异常率
    weird = sum(t.count("\ufffd") + t.count("\x00") for t in texts)
    # 长度
    lens = sorted(len(t) for t in texts)
    return {
        "name": name, "n": n,
        "avg_len": total_chars / n,
        "p50_len": lens[n // 2], "p90_len": lens[int(n * 0.9)],
        "template_rate": n_tmpl / n,
        "template_breakdown": {k: v / n for k, v in tmpl_hits.items()},
        "fact_rate": fact_cnt / n,
        "weird_char_rate": weird / max(total_chars, 1),
        "total_chars": total_chars,
    }


def main() -> int:
    C = corpus_dir()
    # 复用 22 脚本的读取器（与生成语料时完全一致）
    src = io.open(ROOT / "scripts/22_rebuild_corpus.py", encoding="utf-8").read()
    ns = {"__file__": str(ROOT / "scripts/22_rebuild_corpus.py"), "__name__": "x"}
    exec(compile(src.split("def probe(")[0], "x", "exec"), ns)
    iter_wiki, iter_jsonl, iter_belle = ns["iter_wiki"], ns["iter_jsonl"], ns["iter_belle"]

    N = 20000              # 每个来源抽 2 万条
    CAP = 400 * 1024 ** 2  # 最多读 400 MB

    print("=" * 80)
    print("语料摸底（为重新做基座提供依据）")
    print("=" * 80)
    print(f"  每来源抽样 {N:,} 条，最多读 {CAP//1024**2} MB\n")

    jobs = [
        ("wiki", C / "wikipedia-cn-20230720-filtered.json", iter_wiki),
        ("skypile", C / "data/2020-40_zh_head_0000.jsonl", iter_jsonl),
        ("instruct", C / "train_3.5M_CN.json", iter_belle),
    ]

    results = {}
    for name, path, fn in jobs:
        if not path.exists():
            print(f"  {name}: 文件不存在，跳过")
            continue
        t0 = time.time()
        texts = read_samples(path, fn, N, CAP)
        r = analyze(name, texts)
        results[name] = r
        print(f"  [{name}] 抽样 {r['n']:,} 条，用时 {time.time()-t0:.0f}s")

    print("\n" + "=" * 80)
    print("Q1/Q3  模板化程度与文本质量")
    print("=" * 80)
    print(f"  {'来源':<10}{'平均长度':>9}{'p50':>7}{'p90':>8}"
          f"{'模板率':>9}{'事实词率':>10}{'异常字符':>10}")
    print("  " + "-" * 72)
    for name, r in results.items():
        print(f"  {name:<10}{r['avg_len']:>9.0f}{r['p50_len']:>7}"
              f"{r['p90_len']:>8}{r['template_rate']*100:>8.1f}%"
              f"{r['fact_rate']*100:>9.1f}%{r['weird_char_rate']*1e4:>9.2f}‱")

    print("\n  模板类型细分（占该来源的比例）:")
    for name, r in results.items():
        parts = "  ".join(f"{k}:{v*100:.1f}%"
                          for k, v in sorted(r["template_breakdown"].items(),
                                             key=lambda x: -x[1]) if v > 0.001)
        print(f"    {name:<10} {parts if parts else '（无）'}")

    print("\n" + "=" * 80)
    print("Q4  过滤模板样本后还剩多少")
    print("=" * 80)
    total_keep = 0.0
    for name, r in results.items():
        keep = 1 - r["template_rate"]
        # 用抽样字符数推算全量（粗略）
        print(f"  {name:<10} 保留 {keep*100:>5.1f}%  "
              f"（丢弃 {r['template_rate']*100:.1f}%）")
        total_keep += keep

    print("\n" + "=" * 80)
    print("结论与建议")
    print("=" * 80)
    inst = results.get("instruct", {})
    sky = results.get("skypile", {})
    wiki = results.get("wiki", {})

    if inst:
        print(f"  指令数据模板率 {inst['template_rate']*100:.1f}% —— "
              f"{'诊断成立，值得过滤' if inst['template_rate'] > 0.05 else '比例不高，过滤收益有限'}")
    if sky and inst:
        print(f"  事实词出现率: skypile {sky['fact_rate']*100:.1f}%  vs  "
              f"instruct {inst['fact_rate']*100:.1f}%  "
              f"-> {'网页数据知识密度更低，支持昨天的结论' if sky['fact_rate'] < inst['fact_rate'] else '未发现明显差异'}")
    if wiki and inst:
        print(f"  wiki 事实词率 {wiki['fact_rate']*100:.1f}%  vs  "
              f"instruct {inst['fact_rate']*100:.1f}%")

    out = ROOT / "results/corpus_survey.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  结果 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
