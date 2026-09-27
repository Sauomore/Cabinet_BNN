# -*- coding: utf-8 -*-
"""
训练配置吞吐基准：找出在当前 4060 8GB 上最快的配置。

为什么先做基准而不是直接改：
    「GPU 利用率 98%」只说明没在等数据，不代表配置已最优。
    梯度累积次数、batch、ctx 都会影响每步的实际开销，
    必须实测。基准跑 30-40 步，约 1 分钟，代价远低于改错配置重训 6 小时。

注意：本基准会与正在进行的训练争用 GPU（显存充足，可共存）。
      因此绝对数值会偏低，但【相对比较】仍有效。
      如果正在训练，建议先看相对排序，再决定是否重启训练。

用法：
    python scripts/25_bench_configs.py
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from cabinet_bnn.bnn.attention import build_transformer


def bench(preset: str, ctx: int, bs: int, accum: int, steps: int,
          vocab: int, device, dtype=torch.float16) -> dict:
    """测一个配置的吞吐。返回 tokens/s 与峰值显存。"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    model = build_transformer(vocab_size=vocab, n_words=vocab, preset=preset,
                              max_len=ctx, ffn_mode="global").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    # 合成数据：只测算力，不碰磁盘
    gen = torch.Generator(device="cpu").manual_seed(0)
    xb = torch.randint(0, vocab, (bs, ctx), generator=gen)
    yb = torch.randint(0, vocab, (bs, ctx), generator=gen)
    xb, yb = xb.to(device), yb.to(device)

    model.train()
    # 预热（含 cuDNN autotune）
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            logits, _ = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   yb.reshape(-1)) / accum
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            with torch.autocast("cuda", dtype=dtype):
                logits, _ = model(xb)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       yb.reshape(-1)) / accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    dt = time.time() - t0

    toks = steps * bs * accum * ctx
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    del model, opt
    torch.cuda.empty_cache()
    gc.collect()
    return {"tokens": toks, "seconds": dt, "tok_per_s": toks / dt,
            "ms_per_step": dt / steps * 1000, "peak_gb": peak}


def main() -> int:
    ap = argparse.ArgumentParser(description="训练配置吞吐基准")
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--steps", type=int, default=12, help="每个配置测几步")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "results" / "bench_configs.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    print("=" * 78)
    print("训练配置吞吐基准")
    print("=" * 78)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"  每个配置预热 3 步 + 测 {args.steps} 步")
    print()
    print(f"  {'preset':<7}{'ctx':>5}{'bs':>5}{'accum':>7}{'等效批':>8}"
          f"{'tok/s':>10}{'步/时':>8}{'峰值GB':>8}")
    print("  " + "-" * 66)

    # 当前配置 + 各种变体
    configs = [
        # (preset, ctx, bs, accum)  —— 等效批 = bs*accum*ctx
        ("base", 256, 16, 4),      # 当前配置（基准）
        ("base", 256, 32, 2),      # 大 batch，少累积
        ("base", 256, 64, 1),      # 无累积
        ("base", 256, 8, 8),       # 小 batch，多累积（对照）
        ("base", 128, 32, 4),      # 短上下文
        ("base", 512, 8, 4),       # 长上下文
    ]

    results = []
    for preset, ctx, bs, accum in configs:
        try:
            r = bench(preset, ctx, bs, accum, args.steps, args.vocab, device)
        except torch.cuda.OutOfMemoryError:
            print(f"  {preset:<7}{ctx:>5}{bs:>5}{accum:>7}"
                  f"{bs*accum*ctx:>8}  OOM（显存不足）")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"  {preset:<7}{ctx:>5}{bs:>5}{accum:>7}  失败: "
                  f"{type(e).__name__}: {str(e)[:40]}")
            torch.cuda.empty_cache()
            continue
        steps_per_hour = 3600 / (r["ms_per_step"] / 1000 * accum)
        results.append({"preset": preset, "ctx": ctx, "bs": bs, "accum": accum,
                        **r})
        print(f"  {preset:<7}{ctx:>5}{bs:>5}{accum:>7}{bs*accum*ctx:>8}"
              f"{r['tok_per_s']:>10,.0f}{steps_per_hour:>8,.0f}{r['peak_gb']:>8.2f}")

    if results:
        best = max(results, key=lambda r: r["tok_per_s"])
        cur = next((r for r in results
                    if (r["ctx"], r["bs"], r["accum"]) == (256, 16, 4)), None)
        print()
        print("=" * 78)
        print("结论")
        print("=" * 78)
        print(f"  最快: {best['preset']} ctx={best['ctx']} bs={best['bs']} "
              f"accum={best['accum']}  ->  {best['tok_per_s']:,.0f} tok/s")
        if cur:
            gain = best["tok_per_s"] / cur["tok_per_s"]
            print(f"  当前: base ctx=256 bs=16 accum=4  ->  "
                  f"{cur['tok_per_s']:,.0f} tok/s")
            print(f"  理论提速 {gain:.2f}x")
            if gain > 1.1:
                # 换算成 50000 步的耗时
                t_now = 50000 * 16384 / cur["tok_per_s"] / 3600
                t_best = 50000 * (best["bs"] * best["accum"] * best["ctx"]) \
                    / best["tok_per_s"] / 3600
                print(f"\n  以 50000 步固定【等效批大小】计：")
                print(f"    当前配置 {t_now:.1f} 小时")
                print(f"    最快配置 {t_best:.1f} 小时")
            else:
                print("  提速不足 10%，不值得重启训练。")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n  结果 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
