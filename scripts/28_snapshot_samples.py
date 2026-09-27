# -*- coding: utf-8 -*-
"""
到指定 step 时自动抓取训练样本与指标，单独存档。

为什么需要：
    训练日志里的 [sample] 是每 2500 步打一次的一大段，事后从里面挑特定
    step 的样本要翻很久。本脚本盯着日志，到点就把那一段切片存成独立文件，
    便于比对不同 step 的样本质量。

用法：
    python scripts/28_snapshot_samples.py --steps 15000 20000 --hours 4
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> int:
    ap = argparse.ArgumentParser(description="到指定 step 抓取样本快照")
    ap.add_argument("--log", type=Path,
                    default=ROOT / "results/lm_base_mixed/train.log",
                    help="训练日志路径（27 编排写的 train.log，或 08 的 stdout 转存）")
    ap.add_argument("--steps", type=int, nargs="+", default=[15000, 20000])
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/snapshots")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = sorted(set(args.steps))
    done: set[int] = set()
    deadline = time.time() + args.hours * 3600
    shown = 0

    print("=" * 72)
    print("样本快照监视")
    print("=" * 72)
    print(f"  日志   {args.log}")
    print(f"  目标   step {targets}")
    print(f"  时长   {args.hours} 小时")

    while time.time() < deadline and len(done) < len(targets):
        if args.log.exists():
            txt = io.open(args.log, encoding="utf-8", errors="ignore").read()
            # 找出所有 [eval] 出现的位置
            evals = [(int(m.group(1)), m.start())
                     for m in re.finditer(r"\[eval\] step (\d+)", txt)]
            for target in targets:
                if target in done:
                    continue
                # 找最接近且不超过 target+2000 的 eval
                cand = [e for e in evals if e[0] >= target - 2500]
                if not cand:
                    continue
                step_seen, pos = cand[0]
                # 取该 eval 之后的一段（含 sample 行）
                seg = txt[pos:pos + 3000]
                snap = args.out_dir / f"step_{step_seen:06d}.txt"
                snap.write_text(seg, encoding="utf-8")
                done.add(target)
                print(f"\n[{time.strftime('%H:%M:%S')}] 捕获 step {step_seen} "
                      f"-> {snap.name}")
                # 打印出关键行，便于直接看
                for line in seg.splitlines()[:6]:
                    print(f"    {line.strip()[:150]}")
                shown += 1
        if len(done) < len(targets):
            time.sleep(60)

    print(f"\n完成，共捕获 {shown} 个快照 -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
