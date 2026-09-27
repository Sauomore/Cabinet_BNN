# -*- coding: utf-8 -*-
"""
用【同一把尺子】测两个基座，回答「ppl 差异有多少来自语料」。

问题：
    新基座 step 16000  ppl 79.2
    旧基座 step 15000  ppl 30.0
    看起来差 2.6 倍，但两者不在同一测试集上，不能直接比。

做法：交叉评测 —— 每个模型都在【两套验证集】上测一遍。
    旧模型 (lm_base_1ep/ckpt_10000)  x  旧 val (corpus_all)   <- 已知 ≈31.5
    旧模型                            x  新 val (corpus_mixed)
    新模型 (lm_base_mixed/best.pt)    x  新 val
    新模型                            x  旧 val

关键：tokenizer 是同一个（32000 词表，新语料复用了旧 tokenizer），
      所以两个模型可以直接吃对方的 val.bin，无需换分词。

解读：
    · 若旧模型在新 val 上也大幅变差 -> 差异来自【语料难度】
    · 若旧模型在新 val 上差不多     -> 差异来自【模型/训练】
"""

from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from cabinet_bnn.bnn.attention import build_transformer


class Stream:
    def __init__(self, path: Path):
        self.arr = np.memmap(path, dtype=np.uint16, mode="r")
        self.n = len(self.arr)

    def batch(self, bs, ctx, rng, device):
        ix = rng.integers(0, self.n - ctx - 1, size=bs)
        x = np.stack([np.asarray(self.arr[i:i + ctx], dtype=np.int64) for i in ix])
        y = np.stack([np.asarray(self.arr[i + 1:i + ctx + 1], dtype=np.int64)
                      for i in ix])
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def evaluate(model, stream, ctx, bs, device, n_batches=20):
    model.eval()
    rng = np.random.default_rng(999)
    tot, n = 0.0, 0
    for _ in range(n_batches):
        x, y = stream.batch(bs, ctx, rng, device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y.reshape(-1), reduction="sum")
        tot += float(loss)
        n += y.numel()
    model.train()
    return tot / n, math.exp(min(tot / n, 20))


def load_model(ckpt: Path, preset: str, ctx: int, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = ck.get("args", {})
    vocab = ck["model"]["embed.weight"].shape[0]
    m = build_transformer(vocab_size=vocab, n_words=vocab,
                          preset=cfg.get("preset", preset),
                          max_len=ctx, ffn_mode="global").to(device)
    m.load_state_dict(ck["model"], strict=False)
    return m, ck


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="交叉评测两个基座")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--bs", type=int, default=4)
    args = ap.parse_args()
    device = torch.device("cuda")

    old_ck = ROOT / "results/lm_base_1ep/ckpt_10000.pt"
    new_ck = ROOT / "results/lm_base_mixed/best.pt"
    old_val = ROOT / "data/corpus_all/val.bin"
    new_val = ROOT / "data/corpus_mixed/val.bin"

    for p in (old_ck, new_ck, old_val, new_val):
        if not p.exists():
            print(f"  缺失: {p}")
            return 1

    print("=" * 74)
    print("交叉评测：ppl 差异有多少来自语料？")
    print("=" * 74)
    print(f"  旧模型 {old_ck.relative_to(ROOT)}")
    print(f"  新模型 {new_ck.relative_to(ROOT)}")
    print("  两套验证集各测 20 批 x bs4 x ctx256（低干扰配置）")

    results = {}
    for tag, ck, preset in [("旧", old_ck, "base"), ("新", new_ck, "base")]:
        model, ckinfo = load_model(ck, preset, args.ctx, device)
        step = ckinfo.get("step", "?")
        vl = ckinfo.get("val_loss", float("nan"))
        print(f"\n  [{tag}模型] step={step}  记录 val_loss={vl:.4f}")
        for vtag, vp in [("旧val", old_val), ("新val", new_val)]:
            st = Stream(vp)
            loss, ppl = evaluate(model, st, args.ctx, args.bs, device)
            results[f"{tag}_{vtag}"] = {"loss": loss, "ppl": ppl}
            print(f"      {vtag}: loss {loss:.4f}  ppl {ppl:>7.1f}")
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 74)
    print("交叉矩阵（行=模型，列=验证集）")
    print("=" * 74)
    print(f"  {'':<10}{'旧val':>12}{'新val':>12}{'差异':>12}")
    for tag in ("旧", "新"):
        a = results[f"{tag}_旧val"]["ppl"]
        b = results[f"{tag}_新val"]["ppl"]
        print(f"  {tag}模型{'':<6}{a:>12.1f}{b:>12.1f}{b-a:>+12.1f}")

    print("\n  解读:")
    da = results["新_新val"]["ppl"] - results["旧_新val"]["ppl"]
    db = results["新_旧val"]["ppl"] - results["旧_旧val"]["ppl"]
    ca = results["新_旧val"]["ppl"] - results["旧_旧val"]["ppl"]
    print(f"    同一验证集上的模型差异:")
    print(f"      在旧 val 上, 新模型比旧模型 ppl {db:+.1f}")
    print(f"      在新 val 上, 新模型比旧模型 ppl {da:+.1f}")
    print(f"    同一模型上的验证集差异:")
    print(f"      旧模型: 旧val -> 新val ppl "
          f"{results['旧_新val']['ppl']-results['旧_旧val']['ppl']:+.1f}")
    print(f"      新模型: 旧val -> 新val ppl "
          f"{results['新_新val']['ppl']-results['新_旧val']['ppl']:+.1f}")

    out = ROOT / "results/cross_eval.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  结果 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
