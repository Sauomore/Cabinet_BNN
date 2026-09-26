# -*- coding: utf-8 -*-
"""
精确测量新语料的来源构成（按字节与 token 双口径）。

为什么需要：
    scripts/22_rebuild_corpus.py 里的 counts 用的是 len(line) —— 那是
    Unicode【字符】数，中文 1 字 = 3 字节，因此该字段低估约 2.6 倍。
    写进 recipe.json 的 actual_gb 不可信。

本脚本给出两个可信口径：
    ① 按【UTF-8 字节】—— 与文件大小一致
    ② 按【token 数】—— 这才是训练时真正关心的量

用法：
    python scripts/24_measure_recipe.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from cabinet_bnn.paths import corpus_dir

ROOT = Path(__file__).resolve().parent.parent

# 复用 22 的读取器
_src = io.open(ROOT / "scripts/22_rebuild_corpus.py", encoding="utf-8").read()
_ns = {"__file__": str(ROOT / "scripts/22_rebuild_corpus.py"), "__name__": "rb"}
exec(compile(_src.split("def probe(")[0], "rb", "exec"), _ns)
iter_wiki = _ns["iter_wiki"]
iter_jsonl = _ns["iter_jsonl"]
iter_belle = _ns["iter_belle"]

BIG = 99 * 1024 ** 3


def measure(name: str, path: Path, fn, tok, budget_bytes: int | None = None):
    """按字节与 token 双口径测量一个来源。

    budget_bytes 为 None 时读尽全量；否则只读够该字节数（按 UTF-8 计）。
    """
    nb = 0
    nt = 0
    cnt = 0
    buf: list[str] = []
    for line in fn(path, BIG):
        n = len(line.encode("utf-8"))          # ← 按字节，不是字符
        nb += n
        cnt += 1
        buf.append(line)
        # 每 8 MB 批量编码一次，避免逐行调 tokenizer
        if sum(len(x.encode("utf-8")) for x in buf) > 8 * 1024 * 1024:
            nt += len(tok.encode("".join(buf)).ids)
            buf = []
        if budget_bytes and nb >= budget_bytes:
            break
    if buf:
        nt += len(tok.encode("".join(buf)).ids)
    return {"bytes": nb, "tokens": nt, "objects": cnt}


def main() -> int:
    from tokenizers import Tokenizer
    C = corpus_dir()
    tok = Tokenizer.from_file(str(ROOT / "data/corpus_all/tokenizer.json"))

    # 与 22 脚本的配额一致
    WIKI_Q = int(0.47 * 1024 ** 3)
    SKY_Q = int(3.46 * 1024 ** 3)
    INS_Q = int(1.17 * 1024 ** 3)

    jobs = [
        ("wiki", C / "wikipedia-cn-20230720-filtered.json", iter_wiki, WIKI_Q),
        ("skypile", None, None, SKY_Q),          # 3 个分片合计
        ("instruct", C / "train_3.5M_CN.json", iter_belle, INS_Q),
    ]

    print("=" * 76)
    print("新语料来源构成（按配额实测，双口径）")
    print("=" * 76)

    res = {}
    # wiki
    r = measure("wiki", jobs[0][1], jobs[0][2], tok, jobs[0][3])
    res["wiki"] = r
    print(f"  wiki      字节 {r['bytes']/1024**3:>5.2f} GB  "
          f"token {r['tokens']:>13,}  {r['objects']:>9,} 条")

    # skypile：三个分片平分配额
    per = SKY_Q // 3
    tot = {"bytes": 0, "tokens": 0, "objects": 0}
    for i in range(3):
        p = C / f"data/2020-40_zh_head_{i:04d}.jsonl"
        if not p.exists():
            continue
        r = measure(f"skypile{i}", p, iter_jsonl, tok, per)
        for k in tot:
            tot[k] += r[k]
    res["skypile"] = tot
    print(f"  skypile   字节 {tot['bytes']/1024**3:>5.2f} GB  "
          f"token {tot['tokens']:>13,}  {tot['objects']:>9,} 条")

    # instruct
    r = measure("instruct", jobs[2][1], jobs[2][2], tok, jobs[2][3])
    res["instruct"] = r
    print(f"  instruct  字节 {r['bytes']/1024**3:>5.2f} GB  "
          f"token {r['tokens']:>13,}  {r['objects']:>9,} 条")

    tb = sum(v["bytes"] for v in res.values())
    tt = sum(v["tokens"] for v in res.values())
    print()
    print(f"  合计      字节 {tb/1024**3:>5.2f} GB  token {tt:>13,}")
    print()
    print("  按【字节】占比:")
    for k, v in res.items():
        print(f"    {k:<10} {v['bytes']/tb*100:>5.1f}%")
    print()
    print("  按【token】占比（训练时真正关心的）:")
    for k, v in res.items():
        print(f"    {k:<10} {v['tokens']/tt*100:>5.1f}%")
    gen_b = (res["wiki"]["bytes"] + res["skypile"]["bytes"]) / tb * 100
    gen_t = (res["wiki"]["tokens"] + res["skypile"]["tokens"]) / tt * 100
    print()
    print(f"  >> 通用文本: 字节口径 {gen_b:.1f}%   token 口径 {gen_t:.1f}%")
    print(f"  >> 指令数据: 字节口径 {res['instruct']['bytes']/tb*100:.1f}%   "
          f"token 口径 {res['instruct']['tokens']/tt*100:.1f}%")

    out = ROOT / "data/corpus_mixed"
    (out / "composition_measured.json").write_text(
        json.dumps({"by_source": res, "total_bytes": tb, "total_tokens": tt},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 -> {out/'composition_measured.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
