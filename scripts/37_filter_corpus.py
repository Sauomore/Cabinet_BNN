# -*- coding: utf-8 -*-
"""
过滤旧语料里的模板化样本，生成 corpus_clean。

依据（来自 scripts/34_corpus_survey.py 的实测）：
    corpus_all 的模板类型细分：
      连续重复词  5.6%   <- 最大问题（如「通海通海通海」）
      代码块      4.8%   <- ``` 包裹，对中文语料是噪音
      重复列表项  1.0%   <- 「1.药 2.药 3.药 4.药」
      编号堆砌    0.3%
    合计约 9%（有重叠）

⚠️ 设计原则：保守过滤
    宁可少删，不可多删。删掉太多会损失指令数据的多样性，
    而指令数据正是旧语料里质量最高的部分。

用法：
    python scripts/37_filter_corpus.py --sample 200000     # 先小样看效果
    python scripts/37_filter_corpus.py                     # 全量
"""

from __future__ import annotations

import argparse
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

# ---- 过滤规则 ----
# 每条规则返回 True 表示「这条样本该丢」
RULES = {
    "连续重复词": lambda t: re.search(r"(.{2,6})\1{3,}", t) is not None,
    "代码块": lambda t: "```" in t,
    "重复列表项": lambda t: re.search(r"(\d+[.、]\s*\S{1,6}\s*){4,}", t) is not None,
    "编号堆砌": lambda t: re.search(r"[、，]\s*\d+[、，]\s*\d+[、，]\s*\d+", t) is not None,
    "空表格": lambda t: re.search(r"\|\s*\|\s*\|\s*\|", t) is not None,
    "异常字符": lambda t: "\ufffd" in t or "\x00" in t,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="过滤模板化样本")
    ap.add_argument("--src", type=Path,
                    default=ROOT / "data/corpus_all/text_all.txt")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/corpus_clean/text_clean.txt")
    ap.add_argument("--sample", type=int, default=0,
                    help=">0 时只处理这么多行（小样验证）")
    ap.add_argument("--report", type=Path,
                    default=ROOT / "data/corpus_clean/filter_report.json")
    args = ap.parse_args()

    if not args.src.exists():
        print(f"找不到 {args.src}")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("过滤模板化样本")
    print("=" * 74)
    print(f"  源   {args.src}  ({args.src.stat().st_size/1024**3:.2f} GB)")
    print(f"  输出 {args.out}")
    print(f"  规则 {list(RULES)}")
    if args.sample:
        print(f"  ⚠️ 小样模式：只处理前 {args.sample:,} 行")
    print()

    hits = Counter()
    n_in = n_out = 0
    bytes_in = bytes_out = 0
    samples_dropped: dict[str, list[str]] = {k: [] for k in RULES}
    t0 = time.time()

    with io.open(args.src, encoding="utf-8", errors="ignore") as f, \
         io.open(args.out, "w", encoding="utf-8") as w:
        for line in f:
            n_in += 1
            bytes_in += len(line)
            body = line.rstrip("\n")
            if not body.strip():
                continue
            dropped_by = None
            for name, rule in RULES.items():
                try:
                    if rule(body):
                        dropped_by = name
                        break
                except Exception:
                    pass
            if dropped_by:
                hits[dropped_by] += 1
                if len(samples_dropped[dropped_by]) < 3:
                    samples_dropped[dropped_by].append(body[:120])
            else:
                w.write(line)
                n_out += 1
                bytes_out += len(line)

            if n_in % 500000 == 0:
                print(f"  {n_in:>10,} 行  保留 {n_out:>10,}  "
                      f"({n_out/max(n_in,1)*100:.1f}%)  "
                      f"{(time.time()-t0)/60:.1f}min", flush=True)
            if args.sample and n_in >= args.sample:
                break

    dt = (time.time() - t0) / 60
    print("\n" + "=" * 74)
    print("结果")
    print("=" * 74)
    print(f"  输入 {n_in:,} 行  {bytes_in/1024**3:.2f} GB")
    print(f"  输出 {n_out:,} 行  {bytes_out/1024**3:.2f} GB")
    print(f"  丢弃 {n_in-n_out:,} 行 ({(n_in-n_out)/max(n_in,1)*100:.2f}%)")
    print(f"  用时 {dt:.1f} 分钟")
    print()
    print("  各规则命中:")
    for name in RULES:
        c = hits.get(name, 0)
        print(f"    {name:<12} {c:>9,}  ({c/max(n_in,1)*100:>5.2f}%)")

    print("\n  被丢弃的样本示例:")
    for name, exs in samples_dropped.items():
        if exs:
            print(f"\n    [{name}]")
            for e in exs:
                print(f"      {e}")

    report = {
        "src": str(args.src), "out": str(args.out),
        "lines_in": n_in, "lines_out": n_out,
        "bytes_in": bytes_in, "bytes_out": bytes_out,
        "drop_rate": (n_in - n_out) / max(n_in, 1),
        "by_rule": dict(hits), "minutes": dt,
        "sample_mode": bool(args.sample),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\n  报告 -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
