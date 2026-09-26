# -*- coding: utf-8 -*-
"""
扩展并重建语料：把 BelleGroup 指令数据与 SkyPile 并入，重新训练 tokenizer 并编码。

为什么要重建 tokenizer：
    现有 tokenizer 只在维基文本上训过。加入对话/指令数据后，
    新领域的子词（客服话术、口语、代码片段）覆盖不足，
    会退化成大量碎子词。重训一次成本约 12 分钟，收益是整条流水线的表示质量。

输出（写入 --out-dir）：
    text_all.txt          合并后的纯文本（每行一段）
    tokenizer.json        重训的 BPE
    train.bin / val.bin   uint16 token 流

用法：
    python scripts/09_expand_corpus.py --vocab-size 32000
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent

_WS = re.compile(r"[ \t\u3000]+")
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def clean_text(text: str, min_len: int = 30, max_len: int = 4000,
               min_cjk: float = 0.25) -> str | None:
    """通用清洗：去控制符、压空白、长度与中文占比过滤。"""
    if not text:
        return None
    t = _CTRL.sub("", str(text))
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _WS.sub(" ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if len(t) < min_len:
        return None
    if len(t) > max_len:
        t = t[:max_len]
    cjk = sum(1 for c in t if "\u4e00" <= c <= "\u9fff")
    if cjk / max(len(t), 1) < min_cjk:
        return None
    return t


def iter_belle(path: Path, limit: int | None = None):
    """流式解析 BelleGroup 的 [{"conversations": [{"from","value"},...]}, ...]。

    预训练阶段把人类与助手的轮次拼成连续文本：
    这样模型能学到「问 → 答」的接续模式，为后续 SFT 打基础。
    """
    n = 0
    for obj in stream_json_objects(path, limit):
        conv = obj.get("conversations")
        if not isinstance(conv, list):
            continue
        parts = []
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            v = turn.get("value")
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
        if parts:
            yield "\n".join(parts)
            n += 1
            if limit and n >= limit:
                return


def stream_json_objects(path: Path, limit: int | None = None):
    """通用：流式读取一个巨型 JSON 数组，逐项 yield dict。

    直接 json.load 一个 4.6 GB 文件会吃掉 ~20 GB 内存，必须流式。
    实现：按顶层花括号配对切分，逐块 json.loads。
    注意必须处理字符串内的花括号（用 in_string / escape 状态机）。
    """
    buf: list[str] = []
    depth = 0
    in_str = False
    esc = False
    started = False
    n = 0
    CHUNK = 1 << 20

    with io.open(path, encoding="utf-8") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            for ch in chunk:
                if not started:
                    if ch == "{":
                        started = True
                        depth = 1
                        buf = ["{"]
                        in_str = esc = False
                    continue
                buf.append(ch)
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            yield json.loads("".join(buf))
                            n += 1
                            if limit and n >= limit:
                                return
                        except json.JSONDecodeError:
                            pass
                        started = False
                        buf = []


def stream_json_array(path: Path, field: str, limit: int | None = None):
    """流式读取 JSON 数组，逐项 yield 指定字段的字符串值。"""
    for obj in stream_json_objects(path, limit):
        v = obj.get(field)
        if isinstance(v, str) and v:
            yield v


def stream_jsonl(path: Path, fields: list[str], limit: int | None = None):
    """流式读取 jsonl，按候选字段名取文本。"""
    n = 0
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            for k in fields:
                v = obj.get(k)
                if isinstance(v, str) and v:
                    yield v
                    n += 1
                    break
            if limit and n >= limit:
                return


def main() -> int:
    ap = argparse.ArgumentParser(description="扩展语料并重建 tokenizer")
    ap.add_argument("--corpus-dir", type=Path, default=ROOT / "data/corpus")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/corpus_all")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--wiki-limit", type=int, default=None)
    ap.add_argument("--instruct-limit", type=int, default=None)
    ap.add_argument("--skypile-limit", type=int, default=None)
    ap.add_argument("--max-total-mb", type=int, default=2000,
                    help="合并文本上限（MB），防止磁盘爆掉")
    args = ap.parse_args()

    src = args.corpus_dir
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    txt = out / "text_all.txt"

    sources = [
        ("wiki", src / "wikipedia-cn-20230720-filtered.json", "completion", args.wiki_limit),
        ("instruct", src / "train_3.5M_CN.json", None, args.instruct_limit),
        ("skypile0", src / "data/2020-40_zh_head_0000.jsonl", None, args.skypile_limit),
        ("skypile1", src / "data/2020-40_zh_head_0001.jsonl", None, args.skypile_limit),
    ]

    limit_bytes = args.max_total_mb * 1024 * 1024
    written = 0
    counts: dict[str, int] = {}
    t0 = time.time()

    with io.open(txt, "w", encoding="utf-8") as w:
        for name, path, field, lim in sources:
            if not path.exists():
                print(f"[跳过] {name}: 文件不存在 {path.name}", flush=True)
                continue
            print(f"[读取] {name}  {path.name} ({path.stat().st_size/1024**2:.0f} MB)",
                  flush=True)
            n_kept = 0
            if name == "instruct":
                gen = iter_belle(path, lim)
            elif path.suffix == ".json":
                gen = stream_json_array(path, field, lim)
            else:
                gen = stream_jsonl(path, ["text", "content", "completion"], lim)

            for raw in gen:
                c = clean_text(raw)
                if c is None:
                    continue
                w.write(c.replace("\n", " ") + "\n")
                n_kept += 1
                written += len(c.encode("utf-8"))
                if n_kept % 100000 == 0:
                    print(f"    {n_kept:,} 条 / {written/1024**2:.0f} MB  "
                          f"({time.time()-t0:.0f}s)", flush=True)
                if written >= limit_bytes:
                    print(f"    达到上限 {args.max_total_mb} MB，停止", flush=True)
                    break
            counts[name] = n_kept
            print(f"  -> {name}: {n_kept:,} 条", flush=True)
            if written >= limit_bytes:
                break

    print(f"\n[合并] 共 {sum(counts.values()):,} 条，{written/1024**2:.0f} MB", flush=True)
    print(f"  {json.dumps(counts, ensure_ascii=False)}", flush=True)

    # ---- 重训 tokenizer ----
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    print(f"\n[tokenizer] 重训 BPE，词表 {args.vocab_size:,}", flush=True)
    t1 = time.time()
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train([str(txt)], trainer)
    tok.save(str(out / "tokenizer.json"))
    print(f"[tokenizer] 完成 {time.time()-t1:.0f}s", flush=True)

    # ---- 编码 ----
    print(f"\n[编码] uint16 token 流", flush=True)
    tok = Tokenizer.from_file(str(out / "tokenizer.json"))
    V = tok.get_vocab_size()
    if V > 65536:
        raise ValueError(f"词表 {V} 超 uint16 上限")

    total = txt.stat().st_size
    VAL_BYTES = max(int(total * 0.01), 8 * 1024 * 1024)
    split_at = total - VAL_BYTES
    f_tr = io.open(out / "train.bin", "wb")
    f_va = io.open(out / "val.bin", "wb")
    n_tr = n_va = 0
    consumed = 0
    t2 = time.time()
    f = io.open(txt, encoding="utf-8")
    batch: list[str] = []
    bbytes = 0
    BATCH_BYTES = 64 * 1024 * 1024

    def flush(lines: list[str], consumed_bytes: int) -> tuple[int, int]:
        if not lines:
            return 0, 0
        encs = tok.encode_batch(lines)
        ids = np.fromiter((i for e in encs for i in e.ids), dtype=np.uint16)
        if consumed_bytes < split_at:
            f_tr.write(ids.tobytes()); return ids.size, 0
        f_va.write(ids.tobytes()); return 0, ids.size

    for line in f:
        batch.append(line.rstrip("\n"))
        bbytes += len(line.encode("utf-8"))
        if bbytes >= BATCH_BYTES:
            a, b = flush(batch, consumed)
            n_tr += a; n_va += b
            consumed += bbytes
            batch, bbytes = [], 0
            print(f"    {consumed/1024**2:.0f} MB  train {n_tr/1e6:.1f} M  "
                  f"val {n_va/1e6:.2f} M  ({time.time()-t2:.0f}s)", flush=True)
    a, b = flush(batch, consumed)
    n_tr += a; n_va += b
    f.close(); f_tr.close(); f_va.close()

    stats = {
        "vocab_size": V,
        "train_tokens": int(n_tr),
        "val_tokens": int(n_va),
        "text_bytes": int(total),
        "sources": counts,
        "encode_seconds": round(time.time() - t2, 1),
        "total_minutes": round((time.time() - t0) / 60, 1),
    }
    (out / "data_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[完成] train {n_tr/1e6:.1f} M / val {n_va/1e6:.2f} M tokens，"
          f"总用时 {stats['total_minutes']:.1f} 分钟")
    print(f"  统计 -> {out/'data_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
