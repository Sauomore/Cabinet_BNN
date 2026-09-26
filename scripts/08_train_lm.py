# -*- coding: utf-8 -*-
"""
Cabinet-BNN 语言模型训练（带因果注意力的 Transformer）。

这是「从零预训练」路径的第一次真实尝试：
    语料：中文维基 110.8 M tokens（词表 32k BPE）
    模型：BNNTransformerLM（因果注意力 + RoPE + KV cache）
    目标：验证架构能否训出「语法通顺的中文」

用法：
    # 快速验证（约 10 分钟）
    python scripts/08_train_lm.py --preset nano --max-steps 800

    # 完整训练
    python scripts/08_train_lm.py --preset tiny --max-steps 20000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cabinet_bnn.bnn.attention import build_transformer


# ---------------------------------------------------------------- 数据

class TokenStream:
    """uint16 token 流的内存映射读取器。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"找不到 {self.path}")
        self.arr = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.n = len(self.arr)

    def batch(self, bs: int, ctx: int, rng: np.random.Generator, device):
        """随机采样一批 (x, y)，y 为 x 右移一位。"""
        ix = rng.integers(0, self.n - ctx - 1, size=bs)
        x = np.stack([np.asarray(self.arr[i : i + ctx], dtype=np.int64) for i in ix])
        y = np.stack([np.asarray(self.arr[i + 1 : i + ctx + 1], dtype=np.int64) for i in ix])
        return (torch.from_numpy(x).to(device, non_blocking=True),
                torch.from_numpy(y).to(device, non_blocking=True))


# ---------------------------------------------------------------- 学习率

def lr_at(step: int, args) -> float:
    """线性 warmup + 余弦退火。"""
    if step < args.warmup:
        return args.lr * (step + 1) / max(args.warmup, 1)
    if step >= args.max_steps:
        return args.min_lr
    prog = (step - args.warmup) / max(args.max_steps - args.warmup, 1)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * prog))


# ---------------------------------------------------------------- 评测

@torch.no_grad()
def evaluate(model, stream: TokenStream, args, device, n_batches: int = 20) -> float:
    model.eval()
    rng = np.random.default_rng(1234)
    losses = []
    for _ in range(n_batches):
        x, y = stream.batch(args.batch_size, args.ctx, rng, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y.reshape(-1), reduction="mean")
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def sample_text(model, tok, args, device, prompt: str = "", n: int = 60,
                temperature: float = 0.9, top_k: int = 40) -> str:
    model.eval()
    if prompt:
        ids = tok.encode(prompt).ids
        idx = torch.tensor([ids], dtype=torch.long, device=device)
    else:
        idx = torch.tensor([[tok.token_to_id("<bos>")]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=n, temperature=temperature, top_k=top_k)
    model.train()
    return tok.decode(out[0].tolist())


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(description="Cabinet-BNN 语言模型训练")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/corpus")
    ap.add_argument("--preset", default="nano",
                    choices=["nano", "tiny", "small", "base"])
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--min-lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/lm")
    ap.add_argument("--ffn-mode", default="global", choices=["global", "code"])
    ap.add_argument("--k-basis", type=int, default=8)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--resume", type=Path, default=None)
    # ---- 关系门 ----
    ap.add_argument("--relation-gate", action="store_true",
                    help="启用双向注意力关系门")
    ap.add_argument("--relation-dim", type=int, default=16,
                    help="关系向量维度 r")
    ap.add_argument("--relation-heads", type=int, default=1,
                    help="关系头数（1 = 所有注意力头共享一个门）")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="从该 checkpoint 【只加载模型权重】后开始新训练"
                         "（优化器与步数重置）。与 --resume 的区别："
                         "--resume 是继续同一轮，--init-from 是换配置重新开始")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Cabinet-BNN 语言模型训练")
    print("=" * 78)
    print(f"  设备 {device}  精度 {args.dtype}")
    if device.type == "cuda":
        print(f"  GPU {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # ---- 数据 ----
    stats = json.loads((args.data_dir / "data_stats.json").read_text(encoding="utf-8"))
    train = TokenStream(args.data_dir / "train.bin")
    val = TokenStream(args.data_dir / "val.bin")
    print(f"[数据] 词表 {stats['vocab_size']:,}  "
          f"train {train.n/1e6:.1f} M tokens  val {val.n/1e6:.2f} M tokens")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.data_dir / "tokenizer.json"))

    # ---- 模型 ----
    model = build_transformer(
        vocab_size=stats["vocab_size"], n_words=stats["vocab_size"],
        preset=args.preset, max_len=args.ctx,
        ffn_mode=args.ffn_mode, k_basis=args.k_basis, rank=args.rank,
        relation_gate=args.relation_gate,
        relation_dim=args.relation_dim,
        relation_heads=args.relation_heads,
    ).to(device)
    ps = model.n_params()
    print(f"[模型] {args.preset}  d={model.cfg.d_model} layers={model.cfg.n_layers} "
          f"heads={model.cfg.n_heads}  ctx={args.ctx}")
    print(f"       参数量 {ps['total']:,}  (per-token {ps['per_token']:,} + "
          f"共享 {ps['shared']:,})")
    if args.relation_gate:
        n_rel = sum(p.numel() for k, p in model.named_parameters() if "rel_" in k)
        print(f"       双向关系门: 开启  r={args.relation_dim}  "
              f"heads={args.relation_heads}  参数 {n_rel:,}")

    # 参数量与数据量的配比诊断
    tokens_per_param = train.n / max(ps["total"], 1)
    chinchilla = ps["total"] * 20
    print(f"       tokens/param = {tokens_per_param:.1f}  "
          f"(Chinchilla 最优约 20，可用数据 {train.n/1e6:.0f} M vs 最优需求 "
          f"{chinchilla/1e6:.0f} M)")
    if tokens_per_param < 2:
        print("       [警告] 数据量远低于参数量需求，会严重过拟合。"
              "建议换更小的 preset 或加数据。")

    # ---- 优化器 ----
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.dtype]
    use_amp = device.type == "cuda" and amp_dtype != torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    start_step = 0
    if args.init_from and args.init_from.exists():
        # 只加载模型权重，优化器与步数重置 —— 用于「换配置重新训练」
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        sd = ck["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        rel_missing = [k for k in missing if "rel_" in k]
        other_missing = [k for k in missing if "rel_" not in k]
        print(f"[初始化] 从 {args.init_from.name} 加载模型权重")
        print(f"         缺失 {len(missing)} 项（其中关系门 {len(rel_missing)} 项）"
              f"，多余 {len(unexpected)} 项")
        if other_missing:
            print(f"         [警告] 非关系门的权重要缺失 {len(other_missing)} 项: "
                  f"{other_missing[:5]}")
            raise SystemExit("配置不匹配，拒绝继续（避免训练出无意义的结果）")
        print(f"         原 checkpoint step={ck.get('step', '?')}  "
              f"val_loss={ck.get('val_loss', float('nan')):.4f}")
    if args.resume and args.resume.exists():
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        print(f"[恢复] 从 step {start_step} 继续")

    # ---- 训练 ----
    rng = np.random.default_rng(args.seed)
    hist = {"step": [], "train_loss": [], "val_loss": [], "lr": [], "gnorm": []}
    t0 = time.time()
    best_val = float("inf")
    model.train()

    print(f"\n[训练] max_steps={args.max_steps}  batch={args.batch_size}"
          f"×{args.grad_accum}={args.batch_size*args.grad_accum}  "
          f"lr={args.lr}  warmup={args.warmup}")
    print("-" * 78)

    for step in range(start_step, args.max_steps):
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = train.batch(args.batch_size, args.ctx, rng, device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       y.reshape(-1)) / args.grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += float(loss.item())

        if scaler.is_enabled():
            scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], args.grad_clip)
        if scaler.is_enabled():
            scaler.step(opt); scaler.update()
        else:
            opt.step()

        if (step + 1) % args.log_every == 0 or step == start_step:
            dt = time.time() - t0
            done = step + 1 - start_step
            tps = done * args.batch_size * args.grad_accum * args.ctx / max(dt, 1e-6)
            print(f"  step {step+1:>6}/{args.max_steps}  loss {total_loss:.4f}  "
                  f"lr {lr:.2e}  gnorm {float(gnorm):.2f}  "
                  f"{tps/1e3:.1f}k tok/s  {dt/60:.1f}min", flush=True)
            hist["step"].append(step + 1)
            hist["train_loss"].append(total_loss)
            hist["lr"].append(lr)
            hist["gnorm"].append(float(gnorm))

        if (step + 1) % args.eval_every == 0:
            vl = evaluate(model, val, args, device)
            ppl = math.exp(min(vl, 20))
            print(f"    [eval] step {step+1}  val_loss {vl:.4f}  ppl {ppl:.1f}", flush=True)
            hist["val_loss"].append({"step": step + 1, "loss": vl, "ppl": ppl})
            if vl < best_val:
                best_val = vl
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step + 1, "val_loss": vl, "args": vars(args)},
                           args.out_dir / "best.pt")
            print(f"    [sample] {sample_text(model, tok, args, device, n=50)!r}", flush=True)

        if (step + 1) % args.save_every == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1, "args": vars(args)},
                       args.out_dir / f"ckpt_{step+1}.pt")

    # ---- 收尾 ----
    vl = evaluate(model, val, args, device)
    print("\n" + "=" * 78)
    print("训练结束")
    print("=" * 78)
    print(f"  最终 val_loss {vl:.4f}  ppl {math.exp(min(vl,20)):.1f}  "
          f"最优 val_loss {best_val:.4f}  ppl {math.exp(min(best_val,20)):.1f}")
    print(f"  总用时 {(time.time()-t0)/60:.1f} 分钟")
    print("\n  生成样例：")
    for p in ["", "中国", "北京是", "他说道："]:
        print(f"    提示 {p!r:12s} -> {sample_text(model, tok, args, device, p, n=60)!r}")

    torch.save({"model": model.state_dict(), "step": args.max_steps,
                "val_loss": vl, "args": vars(args)}, args.out_dir / "final.pt")
    hist["final_val_loss"] = vl
    hist["params"] = ps
    hist["minutes"] = (time.time() - t0) / 60
    (args.out_dir / "train_history.json").write_text(
        json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  产物 -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
