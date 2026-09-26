# -*- coding: utf-8 -*-
"""
语言模型生成质量评测：把「重复塌缩」这类问题量化出来。

为什么要专门做这个：
    训练脚本里打印的样例是随手抽的，看不出退化程度。
    小模型最常见的失败模式是【重复塌缩】—— 输出语法正确但无限重复同一片段。
    这必须用指标量化，否则会被单次看起来通顺的样本骗过去。

指标：
    rep_2/3/4      n-gram 重复率（不重复 token 数 / 总 token 数），越低越重复
    distinct_k     相邻 k 个 token 的唯一比例
    max_run        最长连续重复子串长度
    entropy        输出的 unigram 熵（越高越多样）
    ppl            在验证集上的困惑度（与生成质量独立）

用法：
    # 评测一批 checkpoint
    python scripts/10_eval_lm.py --ckpt results/lm_tiny_2ep/final.pt \
        --ckpt results/lm_nano_1ep/final.pt

    # 只用命令行指定的模型做多种采样的对比
    python scripts/10_eval_lm.py --ckpt results/lm_tiny_2ep/final.pt --sweep
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cabinet_bnn.bnn.attention import build_transformer

PROMPTS = ["", "中国", "北京是", "他说道：", "2020年", "这项研究"]


# ---------------------------------------------------------------- 指标

def max_repeated_substring_len(ids: list[int], max_check: int = 30) -> int:
    """最长连续重复子串的长度（如 '职业职业职业' -> 6 表示 3-gram 重复两次）。"""
    n = len(ids)
    best = 0
    for L in range(1, min(max_check, n // 2) + 1):
        for i in range(n - 2 * L + 1):
            if ids[i : i + L] == ids[i + L : i + 2 * L]:
                # 向后延伸
                k = 2
                while i + (k + 1) * L <= n and ids[i : i + L] == ids[i + k * L : i + (k + 1) * L]:
                    k += 1
                best = max(best, k * L)
    return best


def text_metrics(ids: list[int]) -> dict:
    """一组多样性 / 重复度指标。"""
    if not ids:
        return {}
    n = len(ids)
    out = {}
    for k in (2, 3, 4):
        grams = [tuple(ids[i : i + k]) for i in range(n - k + 1)]
        out[f"distinct_{k}"] = len(set(grams)) / max(len(grams), 1)
    cnt = Counter(ids)
    probs = np.array(list(cnt.values()), dtype=float)
    probs /= probs.sum()
    out["entropy"] = float(-(probs * np.log(probs + 1e-12)).sum())
    out["unique_token_ratio"] = len(cnt) / n
    out["max_run"] = max_repeated_substring_len(ids)
    out["length"] = n
    return out


# ---------------------------------------------------------------- 生成

@torch.no_grad()
def gen_ids(model, tok, prompt: str, n: int, temp: float, top_k: int, device) -> list[int]:
    model.eval()
    if prompt:
        ids = tok.encode(prompt).ids
    else:
        ids = [tok.token_to_id("<bos>")]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=n, temperature=temp, top_k=top_k)
    return out[0].tolist()[len(ids):]     # 只保留新生成部分


@torch.no_grad()
def val_ppl(model, path: Path, ctx: int, device, n_batches: int = 20, bs: int = 16) -> float:
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    losses = []
    model.eval()
    for _ in range(n_batches):
        ix = rng.integers(0, len(arr) - ctx - 1, size=bs)
        x = np.stack([np.asarray(arr[i : i + ctx], dtype=np.int64) for i in ix])
        y = np.stack([np.asarray(arr[i + 1 : i + ctx + 1], dtype=np.int64) for i in ix])
        xt = torch.from_numpy(x).to(device)
        yt = torch.from_numpy(y).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            logits, _ = model(xt)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yt.reshape(-1))
        losses.append(float(loss.item()))
    return float(np.mean(losses))


# ---------------------------------------------------------------- 主流程

def load_model(ckpt_path: Path, data_dir: Path, ctx: int, device):
    from tokenizers import Tokenizer
    stats = json.loads((data_dir / "data_stats.json").read_text(encoding="utf-8"))
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck.get("args", {})
    model = build_transformer(
        vocab_size=stats["vocab_size"], n_words=stats["vocab_size"],
        preset=a.get("preset", "nano"), max_len=ctx,
        ffn_mode=a.get("ffn_mode", "global"),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
    return model, tok, ck, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="语言模型生成质量评测")
    ap.add_argument("--ckpt", type=Path, action="append", required=True)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/corpus")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--n-tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--sweep", action="store_true", help="对采样参数做扫描")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("语言模型生成质量评测")
    print("=" * 78)
    print(f"  设备 {device}  ctx {args.ctx}  生成长度 {args.n_tokens}")
    print(f"  采样 temperature={args.temperature} top_k={args.top_k}")
    print(f"  评测提示词 {len(PROMPTS)} 个: {PROMPTS}")

    all_results = []
    for ckpt in args.ckpt:
        if not ckpt.exists():
            print(f"\n[跳过] {ckpt} 不存在")
            continue
        print(f"\n{'='*78}\n[模型] {ckpt}\n{'='*78}")
        model, tok, ck, stats = load_model(ckpt, args.data_dir, args.ctx, device)
        ps = model.n_params()
        ppl = val_ppl(model, args.data_dir / "val.bin", args.ctx, device)
        print(f"  参数量 {ps['total']:,}   step {ck.get('step','?')}   "
              f"val_loss {ck.get('val_loss', float('nan')):.4f}  "
              f"实测 ppl {math.exp(min(ppl,20)):.1f}")

        record = {"ckpt": str(ckpt), "params": ps, "step": ck.get("step"),
                  "val_loss_ckpt": ck.get("val_loss"), "ppl": math.exp(min(ppl, 20)),
                  "samples": [], "sweep": []}

        temps = [(args.temperature, args.top_k)]
        if args.sweep:
            temps = [(0.6, 20), (0.7, 40), (0.8, 40), (0.9, 60), (1.0, 100), (0.8, 0)]

        for temp, tk in temps:
            tag = f"T={temp} k={tk}" if tk else f"T={temp} k=全部"
            print(f"\n  --- {tag} ---")
            agg = []
            for p in PROMPTS:
                ids = gen_ids(model, tok, p, args.n_tokens, temp, tk or None, device)
                m = text_metrics(ids)
                txt = tok.decode(ids)
                agg.append(m)
                print(f"    {p!r:12s} rep2={m['distinct_2']:.3f} rep3={m['distinct_3']:.3f} "
                      f"maxrun={m['max_run']:>3} ent={m['entropy']:.2f}")
                print(f"      {txt[:110]!r}")
            mean = {k: float(np.mean([a[k] for a in agg if k in a])) for k in agg[0]}
            print(f"    >> 均值 distinct_2={mean['distinct_2']:.3f} "
                  f"distinct_3={mean['distinct_3']:.3f} max_run={mean['max_run']:.1f} "
                  f"entropy={mean['entropy']:.2f}")
            record["sweep" if args.sweep else "samples"].append(
                {"temp": temp, "top_k": tk, "mean": mean,
                 "samples": [tok.decode(gen_ids(model, tok, p, args.n_tokens, temp,
                                                tk or None, device)) for p in PROMPTS]})
        all_results.append(record)

    # ---- 汇总 ----
    if len(all_results) > 1:
        print("\n" + "=" * 78)
        print("跨模型对比")
        print("=" * 78)
        print(f"  {'模型':<34}{'参数量':>12}{'ppl':>9}{'distinct_3':>12}{'max_run':>9}")
        print("  " + "-" * 74)
        for r in all_results:
            src = r["sweep"] or r["samples"]
            m = src[0]["mean"] if src else {}
            name = Path(r["ckpt"]).parent.name + "/" + Path(r["ckpt"]).name
            print(f"  {name:<34}{r['params']['total']:>12,}{r['ppl']:>9.1f}"
                  f"{m.get('distinct_3', float('nan')):>12.3f}{m.get('max_run', 0):>9.1f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"\n  报告 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
