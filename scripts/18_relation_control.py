# -*- coding: utf-8 -*-
"""
关系门微调的【对照实验】。

必须做对照的原因：
    从 base 出发再训 6000 步，val_loss 从 3.0111 降到 2.9863。
    但这 0.025 的改善**可能完全来自「多训了 6000 步」**，与关系门无关。
    没有对照就无法判断关系门是否真的有贡献。

对照设计：
    受控变量：优化器、lr 调度、数据顺序（同 seed）、步数、batch
    唯一差异：关系门是否生效
        A. 关系门开启（已完成）       results/lm_relgate_ft
        B. 关系门关闭（γ 冻结为 0）   本脚本

    实现「关闭」的方式：把 rel_gamma 的 requires_grad 设为 False 并固定为 0。
    这样模型结构与参数量与 A 完全相同（排除参数量差异的干扰），
    且前向计算图完全等价于原始注意力（γ=0 时 g≡1，已逐位验证）。

用法：
    python scripts/18_relation_control.py
    python scripts/18_relation_control.py --steps 6000 --lr 2e-4
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
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


class TokenStream:
    def __init__(self, path: Path):
        self.arr = np.memmap(path, dtype=np.uint16, mode="r")
        self.n = len(self.arr)

    def batch(self, bs, ctx, rng, device):
        ix = rng.integers(0, self.n - ctx - 1, size=bs)
        x = np.stack([np.asarray(self.arr[i:i + ctx], dtype=np.int64) for i in ix])
        y = np.stack([np.asarray(self.arr[i + 1:i + ctx + 1], dtype=np.int64) for i in ix])
        return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))


@torch.no_grad()
def evaluate(model, val, ctx, bs, device, n_batches=40, amp_dtype=torch.float16):
    model.eval()
    rng = np.random.default_rng(12345)
    tot, n = 0.0, 0
    for _ in range(n_batches):
        x, y = val.batch(bs, ctx, rng, device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y.reshape(-1), reduction="sum")
        tot += float(loss)
        n += y.numel()
    model.train()
    return tot / n


def main() -> int:
    ap = argparse.ArgumentParser(description="关系门对照实验（γ 冻结为 0）")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/corpus_all")
    ap.add_argument("--init-from", type=Path,
                    default=ROOT / "results/lm_base_1ep/final.pt")
    ap.add_argument("--preset", default="base")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--relation-dim", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1337, help="必须与实验组一致")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "results/lm_relgate_ctrl")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = json.loads((args.data_dir / "data_stats.json").read_text(encoding="utf-8"))
    train = TokenStream(args.data_dir / "train.bin")
    val = TokenStream(args.data_dir / "val.bin")

    print("=" * 76)
    print("关系门对照实验（γ 冻结为 0，其余与实验组完全相同）")
    print("=" * 76)

    model = build_transformer(
        vocab_size=stats["vocab_size"], n_words=stats["vocab_size"],
        preset=args.preset, max_len=args.ctx,
        relation_gate=True, relation_dim=args.relation_dim,
    ).to(device)

    ck = torch.load(args.init_from, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    other_missing = [k for k in missing if "rel_" not in k]
    if other_missing:
        raise SystemExit(f"配置不匹配，缺失非关系门权重: {other_missing[:5]}")
    print(f"[初始化] 从 {args.init_from.name} 加载（缺失 {len(missing)} 项，"
          f"均为关系门）")
    print(f"         原 val_loss = {ck.get('val_loss', float('nan')):.4f}")

    # ---- 冻结关系门：γ 固定为 0，u 因 γ=0 而梯度为 0（数学必然）----
    n_frozen = 0
    for name, p in model.named_parameters():
        if "rel_" in name:
            p.requires_grad_(False)
            p.zero_()
            n_frozen += p.numel()
    print(f"[冻结] 关系门参数 {n_frozen:,} 个全部冻结为 0")
    print(f"       -> 前向等价于原始注意力（γ=0 时 g≡1，已逐位验证）")

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[可训练参数] {sum(p.numel() for p in trainable):,}")

    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * prog))

    rng = np.random.default_rng(args.seed)          # 与实验组同 seed
    model.train()
    t0 = time.time()
    hist = []
    best = float("inf")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for step in range(args.steps):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = train.batch(args.batch_size, args.ctx, rng, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       y.reshape(-1)) / args.grad_accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        scaler.step(opt)
        scaler.update()

        if (step + 1) % args.log_every == 0:
            print(f"  step {step+1:>5}/{args.steps}  loss {float(loss)*args.grad_accum:.4f}  "
                  f"lr {lr:.2e}  gnorm {float(gn):.2f}  "
                  f"{(time.time()-t0)/60:.1f}min", flush=True)
        if (step + 1) % args.eval_every == 0:
            vl = evaluate(model, val, args.ctx, args.batch_size, device)
            print(f"    [eval] step {step+1}  val_loss {vl:.4f}  ppl {math.exp(min(vl,20)):.1f}",
                  flush=True)
            hist.append({"step": step + 1, "val_loss": vl})
            if vl < best:
                best = vl
                torch.save({"model": model.state_dict(), "step": step + 1,
                            "val_loss": vl, "args": vars(args)},
                           args.out_dir / "best.pt")

    final = evaluate(model, val, args.ctx, args.batch_size, device)
    print(f"\n训练结束")
    print(f"  最终 val_loss {final:.4f}  ppl {math.exp(min(final,20)):.1f}")
    print(f"  最优 val_loss {best:.4f}")

    # ---- 与实验组对比 ----
    exp_path = ROOT / "results/lm_relgate_ft/train_history.json"
    print("\n" + "=" * 76)
    print("对比")
    print("=" * 76)
    base_vl = float(ck.get("val_loss", float("nan")))
    print(f"  {'配置':<34}{'val_loss':>10}{'ppl':>8}{'vs base':>12}")
    print(f"  {'base（起点）':<34}{base_vl:>10.4f}"
          f"{math.exp(min(base_vl,20)):>8.1f}{'—':>12}")
    print(f"  {'B. 关系门冻结（对照）':<34}{best:>10.4f}"
          f"{math.exp(min(best,20)):>8.1f}{best-base_vl:>+12.4f}")
    exp_best = None
    if exp_path.exists():
        h = json.loads(exp_path.read_text(encoding="utf-8"))
        vl = h.get("val_loss") or []
        if vl:
            exp_best = min(v["loss"] for v in vl)
            print(f"  {'A. 关系门开启（实验组）':<34}{exp_best:>10.4f}"
                  f"{math.exp(min(exp_best,20)):>8.1f}{exp_best-base_vl:>+12.4f}")
    if exp_best is not None:
        d = best - exp_best
        print(f"\n  实验组相对对照组: {d:+.4f}  "
              f"({'实验组更好' if d > 0 else '实验组更差' if d < 0 else '无差异'})")
        if abs(d) < 0.005:
            print("  >> 差异在噪声量级内，【不能】断定关系门有贡献")
        else:
            print(f"  >> 差异 {abs(d):.4f}，需结合多次重复实验判断显著性")

    (args.out_dir / "train_history.json").write_text(
        json.dumps({"final_val_loss": final, "best_val_loss": best,
                    "minutes": (time.time() - t0) / 60,
                    "frozen_relation_gate": True, "hist": hist},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
