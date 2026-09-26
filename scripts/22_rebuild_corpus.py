# -*- coding: utf-8 -*-
"""
按【指定配比】重建语料，而不是"测量后修正"。

为什么换方案：
    之前的做法是先测 text_all.txt 的来源构成，再决定怎么调。但抽样估算
    在这个数据上不可靠（wiki 的 JSON 对象跨多行，正则只匹配到一部分；
    实测估算总量 0.25 GB vs 实际 3.91 GB，差 15 倍）。
    直接指定配比可以完全绕开估算 —— 每个来源写多少字节是确定的。

配方（依据）：
    预训练阶段应以通用文本为主，指令数据只占少量，否则模型会提前过拟合
    到「问答模板」的格式上。已实测的症状：空 Markdown 表格、重复列表、
    偶发乱码（见 docs/RESULTS.md 的语料配比问题）。

    通用文本（wiki + skypile）  85%   —— 学语言本身
    指令数据（BelleGroup）      15%   —— 保留少量指令格式感

    指令数据留给 SFT 阶段，不在预训练里大量出现。

用法：
    python scripts/22_rebuild_corpus.py --total-gb 4.0 --dry-run
    python scripts/22_rebuild_corpus.py --total-gb 4.0
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from cabinet_bnn.paths import corpus_dir

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- 读取器

def _stream_objects(path: Path, budget: int, field: str):
    """通用流式 JSON 对象读取：按花括号配对切分，取指定字段。

    ⚠️ 为什么不能按行解析（踩过的坑）：
        wiki 文件是 **pretty-printed 的 JSON 数组**，一个对象跨多行：
            [
              {
                "completion": "……",
                "source": "wikipedia.zh2307"
              },
              ...
        「一行一对象」的假设会让 json.loads 全部失败 —— 实测产率算出 0.000，
        被误判为「该来源没有可用文本」。
        BelleGroup 则是紧凑单行数组。两者格式不同，必须用配对解析统一处理。
    """
    used = 0
    buf: list[str] = []
    depth = 0
    in_str = False
    esc = False
    with io.open(path, encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            for c in chunk:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = not in_str
                elif not in_str:
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                if depth > 0 or (depth == 0 and buf):
                    buf.append(c)
                if depth == 0 and buf:
                    raw = "".join(buf).strip()
                    buf = []
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    txt = obj.get(field)
                    if not isinstance(txt, str) or len(txt.strip()) < 30:
                        continue
                    line_out = txt.strip().replace("\n", " ") + "\n"
                    yield line_out
                    used += len(line_out.encode("utf-8"))
                    if used >= budget:
                        return


def iter_wiki(path: Path, budget: int):
    """wiki: JSON 数组，对象含 completion 字段。"""
    yield from _stream_objects(path, budget, "completion")


def iter_belle(path: Path, budget: int):
    """BelleGroup: JSON 数组，对象含 conversations[{from,value}]。"""
    used = 0
    buf: list[str] = []
    depth = 0
    in_str = False
    esc = False
    with io.open(path, encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            for c in chunk:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = not in_str
                elif not in_str:
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                if depth > 0 or (depth == 0 and buf):
                    buf.append(c)
                if depth == 0 and buf:
                    raw = "".join(buf).strip()
                    buf = []
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    parts = [t.get("value", "").strip()
                             for t in obj.get("conversations", [])]
                    txt = " ".join(p for p in parts if p)
                    if len(txt) < 20:
                        continue
                    line_out = txt.replace("\n", " ") + "\n"
                    yield line_out
                    used += len(line_out.encode("utf-8"))
                    if used >= budget:
                        return


def iter_jsonl(path: Path, budget: int, fields=("text", "content", "completion")):
    """SkyPile 等 jsonl，按候选字段名取文本。"""
    used = 0
    with io.open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            txt = ""
            for fld in fields:
                v = obj.get(fld)
                if isinstance(v, str) and v.strip():
                    txt = v.strip()
                    break
            if len(txt) < 50:
                continue
            line_out = txt.replace("\n", " ") + "\n"
            yield line_out
            used += len(line_out.encode("utf-8"))
            if used >= budget:
                return


# ---------------------------------------------------------------- 主流程

def probe(path: Path, fn, probe_bytes: int = 60 * 1024 * 1024) -> dict:
    """用【真实读取器】测产率：读够 probe_bytes 文本就停。

    为什么必须用真实读取器：
        之前用「按行 json.loads」的简化测量，对 pretty-printed 的 wiki 文件
        全部解析失败，算出产率 0.000 —— 差点把 wiki 整个排除掉。
        测产率必须走和正式生成【完全相同】的代码路径。
    """
    t0 = time.time()
    n = 0
    cnt = 0
    for line in fn(path, probe_bytes):
        n += len(line.encode("utf-8"))
        cnt += 1
    return {"text_bytes": n, "objects": cnt, "seconds": time.time() - t0,
            "avg_len": n / max(cnt, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description="按指定配比重建语料")
    ap.add_argument("--total-gb", type=float, default=0.0,
                    help="目标文本总大小（GB）。0 = 用尽所有来源（受各配额约束）")
    ap.add_argument("--wiki-pct", type=float, default=0.08,
                    help="wiki 目标占比（注意：wiki 全量只有 ~0.47 GB，"
                         "占比上限约 6-8%%，给多了会拿不满）")
    ap.add_argument("--skypile-pct", type=float, default=0.60,
                    help="skypile 目标占比（通用网页文本，是通用语料的主力）")
    ap.add_argument("--instruct-pct", type=float, default=0.20,
                    help="指令数据目标占比（预训练阶段应压低，留给 SFT）")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/corpus_mixed",
                    help="输出目录（不覆盖 corpus_all，便于对比）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只计算配额，不实际写出")
    ap.add_argument("--probe", action="store_true",
                    help="用真实读取器测各来源产率后退出")
    args = ap.parse_args()

    src = corpus_dir()
    wiki_p = src / "wikipedia-cn-20230720-filtered.json"
    belle_p = src / "train_3.5M_CN.json"
    sky_ps = [src / "data/2020-40_zh_head_0000.jsonl",
              src / "data/2020-40_zh_head_0001.jsonl",
              src / "data/2020-40_zh_head_0002.jsonl"]

    # ---------------- probe 模式 ----------------
    if args.probe:
        print("=" * 76)
        print("用真实读取器测各来源产率")
        print("=" * 76)
        jobs = [("wiki", wiki_p, iter_wiki)]
        for i, p in enumerate(sky_ps):
            jobs.append((f"skypile{i:04d}", p, iter_jsonl))
        jobs.append(("instruct", belle_p, iter_belle))

        res = {}
        tot = 0.0
        for name, p, fn in jobs:
            if not p.exists():
                print(f"  {name:<14} 不存在")
                continue
            r = probe(p, fn)
            sz = p.stat().st_size
            res[name] = {"text_bytes": r["text_bytes"], "objects": r["objects"],
                         "avg_len": r["avg_len"], "size_gb": sz / 1024 ** 3}
            print(f"  {name:<14} 源 {sz/1024**3:>5.2f} GB  取到文本 "
                  f"{r['text_bytes']/1024**2:>7.1f} MB  {r['objects']:>7,} 条  "
                  f"平均 {r['avg_len']:>6.0f} 字符  ({r['seconds']:.0f}s)")
            tot += r["text_bytes"]
        print(f"\n  合计取到 {tot/1024**3:.2f} GB 文本（各取 60 MB 为限）")
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "yield_probe.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  结果 -> {args.out/'yield_probe.json'}")
        return 0

    # ---------------- 配额计算 ----------------
    # 实测各来源全量可产文本（见 --probe 与 data/corpus_mixed/yield_full.json）：
    #     wiki      0.47 GB   (254,546 条)
    #     skypile   3.46 GB   (1,313,929 条, 3 个分片)
    #     instruct  4.04 GB   (3,604,405 条)
    # 因此 wiki 的占比上限约 6-8%，给多了根本拿不满。
    WIKI_MAX = 0.47 * 1024 ** 3
    SKY_MAX = 3.46 * 1024 ** 3
    INS_MAX = 4.04 * 1024 ** 3

    if args.total_gb > 0:
        total = int(args.total_gb * 1024 ** 3)
    else:
        # 无指定总量时，以【最稀缺的来源】反推：wiki 只有 0.47 GB，
        # 若它要占 wiki_pct，则总量上限 = 0.47 / wiki_pct。
        # （早先用 instruct 反推得 20 GB，远超实际可产出量，是错的。）
        total = int(WIKI_MAX / max(args.wiki_pct, 1e-6))

    q = {
        "wiki": min(int(total * args.wiki_pct), int(WIKI_MAX)),
        "skypile": min(int(total * args.skypile_pct), int(SKY_MAX)),
        "instruct": min(int(total * args.instruct_pct), int(INS_MAX)),
    }

    print("=" * 76)
    print("按指定配比重建语料")
    print("=" * 76)
    print(f"  目标总量 {total/1024**3:.2f} GB")
    print(f"  配方: wiki {args.wiki_pct:.0%} / "
          f"skypile {args.skypile_pct:.0%} / instruct {args.instruct_pct:.0%}")
    print(f"  字节配额（已按来源上限裁剪）:")
    for k, v in q.items():
        cap = {"wiki": WIKI_MAX, "skypile": SKY_MAX, "instruct": INS_MAX}[k]
        note = "  ← 受来源上限限制" if v >= cap else ""
        print(f"    {k:<10} {v/1024**3:>5.2f} GB  (来源上限 {cap/1024**3:.2f} GB){note}")

    print("\n  来源检查:")
    for label, ps in [("wiki", [wiki_p]), ("instruct", [belle_p]),
                      ("skypile", sky_ps)]:
        for p in ps:
            mark = "✅" if p.exists() else "❌"
            sz = f"{p.stat().st_size/1024**3:.2f} GB" if p.exists() else "不存在"
            print(f"    {mark} {label:<9} {p.name:<45} {sz}")

    if args.dry_run:
        print("\n  [dry-run] 未写出任何文件")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    txt = args.out / "text_mixed.txt"
    counts = {"wiki": 0, "skypile": 0, "instruct": 0}
    t0 = time.time()

    with io.open(txt, "w", encoding="utf-8") as w:
        # ---- wiki ----
        if wiki_p.exists() and q["wiki"] > 0:
            print(f"\n  [1/3] wiki (配额 {q['wiki']/1024**3:.2f} GB) …", flush=True)
            n = 0
            for line in iter_wiki(wiki_p, q["wiki"]):
                w.write(line)
                n += len(line)
            counts["wiki"] = n
            print(f"        写入 {n/1024**3:.2f} GB  ({time.time()-t0:.0f}s)",
                  flush=True)

        # ---- skypile（多分片平分配额）----
        avail = [p for p in sky_ps if p.exists()]
        if avail and q["skypile"] > 0:
            print(f"  [2/3] skypile (配额 {q['skypile']/1024**3:.2f} GB, "
                  f"{len(avail)} 个分片) …", flush=True)
            per = q["skypile"] // len(avail)
            n = 0
            for p in avail:
                for line in iter_jsonl(p, per):
                    w.write(line)
                    n += len(line)
            counts["skypile"] = n
            print(f"        写入 {n/1024**3:.2f} GB  ({time.time()-t0:.0f}s)",
                  flush=True)

        # ---- instruct ----
        if belle_p.exists() and q["instruct"] > 0:
            print(f"  [3/3] instruct (配额 {q['instruct']/1024**3:.2f} GB) …",
                  flush=True)
            n = 0
            for line in iter_belle(belle_p, q["instruct"]):
                w.write(line)
                n += len(line)
            counts["instruct"] = n
            print(f"        写入 {n/1024**3:.2f} GB  ({time.time()-t0:.0f}s)",
                  flush=True)

    total_written = sum(counts.values())
    print("\n" + "=" * 76)
    print("完成")
    print("=" * 76)
    print(f"  输出 {txt}")
    print(f"  实际大小 {txt.stat().st_size/1024**3:.2f} GB  用时 "
          f"{(time.time()-t0)/60:.1f} 分钟")
    print("\n  实际占比:")
    for k, v in counts.items():
        print(f"    {k:<10} {v/1024**3:>6.2f} GB  {v/max(total_written,1)*100:>5.1f}%")

    stats = {
        "recipe": {"wiki": args.wiki_pct, "skypile": args.skypile_pct,
                   "instruct": args.instruct_pct},
        "target_gb": args.total_gb,
        "actual_gb": {k: v / 1024**3 for k, v in counts.items()},
        "total_gb": txt.stat().st_size / 1024**3,
        "minutes": (time.time() - t0) / 60,
    }
    (args.out / "recipe.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  配方 -> {args.out/'recipe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
