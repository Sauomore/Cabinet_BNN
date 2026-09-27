# -*- coding: utf-8 -*-
"""
编排：等当前基座训练结束后，自动接着跑 MoE 对照组，最后出对比报告。

为什么需要编排而不是直接串行跑：
    08_train_lm.py 正在占用 GPU（约 3.5 小时后结束）。直接启动第二组会显存
    争用，两个都变慢。本脚本先等到 GPU 空闲，再依次跑两组并出报告。

两组配置**完全相同**，唯一差异是 --ffn-mode：
    对照组  --ffn-mode global
    实验组  --ffn-mode moe --moe-k 16 --moe-k-active 4 --moe-d-ff 256 --moe-rank 16

这样「FFN 类型」是唯一变量，结论才站得住。
（之前那次对比作废，就是因为对照组保留了已训练的 FFN 而实验组是随机的。）

用法：
    python scripts/27_run_comparison.py            # 等 GPU 空闲后自动串跑
    python scripts/27_run_comparison.py --now      # 不等，立刻开始（会争显存）
"""

from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DATA_DIR = ROOT / "data/corpus_mixed"
OUT_A = ROOT / "results/lm_cmp_global"
OUT_B = ROOT / "results/lm_cmp_moe"
LOG = ROOT / "logs/comparison.log"

sys.path.insert(0, str(ROOT))
from cabinet_bnn.paths import logs_dir  # noqa: E402

COMMON = [
    "--data-dir", str(DATA_DIR), "--preset", "base", "--ctx", "256",
    "--batch-size", "16", "--grad-accum", "4",
    "--max-steps", "30000", "--warmup", "1000",
    "--lr", "6e-4", "--min-lr", "6e-5", "--weight-decay", "0.1",
    "--grad-clip", "1.0", "--seed", "2024",
    "--eval-every", "2000", "--log-every", "1000", "--save-every", "10000",
]
MOE_ARGS = ["--moe-k", "16", "--moe-k-active", "4",
            "--moe-d-ff", "256", "--moe-rank", "16"]


def log(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    logs_dir().mkdir(parents=True, exist_ok=True)
    with io.open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def gpu_free() -> bool:
    """GPU 上是否已无 python 训练进程。

    ⚠️ 踩过的坑（导致编排空等 3h45m 而没启动训练）：
        PowerShell 命令是通过 subprocess 传的【参数列表】运行的，
        不是经过外层 shell，因此 **`$_` 前不能加反斜杠转义**。
        写成 `\\$_.CommandLine` 会让 PowerShell 收到字面的 `\.CommandLine`
        并报 CommandNotFound，输出为空 —— 于是这里判定「无训练进程」，
        但真正的后果是编排在等待循环里空转到超时。

    正确写法：PowerShell 部分用单引号包住，内部正常写 `$_`。
    另外把「检测到训练进程」与「检测失败」区分开：检测失败时不应
    误判为空闲，而应保守地当作「忙」继续等。
    """
    ps = ("Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
          "ForEach-Object { $_.CommandLine }")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, timeout=30)
        out = r.stdout or ""
        if r.returncode != 0:
            return False                      # 检测失败 -> 保守当作忙
        return "08_train_lm" not in out
    except Exception:
        return False


def wait_gpu(max_hours: float = 12.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < max_hours * 3600:
        if gpu_free():
            return True
        time.sleep(120)
    return False


def run(tag: str, extra: list[str], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(ROOT / "scripts/08_train_lm.py"),
           *COMMON, *extra, "--out-dir", str(out_dir)]
    log(f"[{tag}] 启动: ffn-mode={' '.join(extra[:2])}")
    t0 = time.time()
    with io.open(out_dir / "train.log", "w", encoding="utf-8",
                 errors="replace") as lf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", bufsize=1)
        for line in p.stdout:
            lf.write(line)
            if any(k in line for k in ("[eval]", "训练结束", "最终", "Traceback")):
                log(f"  [{tag}] {line.rstrip()}")
        p.wait()
    log(f"[{tag}] 结束，退出码 {p.returncode}，用时 {(time.time()-t0)/3600:.2f} 小时")
    return p.returncode


def report() -> int:
    def h(d: Path):
        f = d / "train_history.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    a, b = h(OUT_A), h(OUT_B)
    log("=" * 70)
    log("FFN 机制对比：标准 FFN vs MoE 稀疏 FFN")
    log("=" * 70)
    if not a or not b:
        log(f"  ⚠️ 缺少数据（对照={bool(a)} MoE={bool(b)}）")
        return 1

    va, vb = a["final_val_loss"], b["final_val_loss"]
    pa = a["params"]["total"] if isinstance(a["params"], dict) else a["params"]
    pb = b["params"]["total"] if isinstance(b["params"], dict) else b["params"]

    log(f"  {'配置':<20}{'参数量':>12}{'val_loss':>11}{'ppl':>9}")
    log(f"  {'标准 FFN':<20}{pa:>12,}{va:>11.4f}{math.exp(min(va,20)):>9.1f}")
    log(f"  {'MoE 稀疏 FFN':<20}{pb:>12,}{vb:>11.4f}{math.exp(min(vb,20)):>9.1f}")
    log(f"  {'差异':<20}{pb-pa:>+12,}{vb-va:>+11.4f}")
    log("")
    if abs(vb - va) < 0.01:
        log("  >> 结论：两者【相当】（差异 < 0.01）")
    elif vb < va:
        log(f"  >> 结论：MoE 更好（{-vb+va:.4f}）")
    else:
        log(f"  >> 结论：MoE 更差（{vb-va:.4f}）")
    log(f"  >> 参数量比 {pb/pa:.3f}（MoE 每 token 只激活 k'=4/16 个专家）")
    log("  >> 真正的优势是【稀疏写】：改一个 token 只影响它的 4 个专家")

    (ROOT / "results/ffn_comparison.json").write_text(json.dumps({
        "control": {"final_val_loss": va, "params": pa, "minutes": a.get("minutes")},
        "moe": {"final_val_loss": vb, "params": pb, "minutes": b.get("minutes")},
        "delta": vb - va, "param_ratio": pb / pa,
        "config": {"steps": 30000, "ctx": 256, "seed": 2024,
                   "corpus": str(DATA_DIR)},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  结果 -> results/ffn_comparison.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="FFN 机制对比编排")
    ap.add_argument("--now", action="store_true", help="不等 GPU 空闲，立刻开始")
    ap.add_argument("--max-wait-hours", type=float, default=12.0)
    args = ap.parse_args()

    log("=" * 70)
    log("FFN 机制对比编排启动")
    log("=" * 70)

    if not args.now:
        if not gpu_free():
            log("GPU 上有训练在跑，等待其结束 …（每 2 分钟检查一次）")
        if not wait_gpu(args.max_wait_hours):
            log(f"等待超过 {args.max_wait_hours} 小时仍未空闲，放弃")
            return 1
        log("GPU 已空闲，开始跑对照组")

    rc = run("对照组", ["--ffn-mode", "global"], OUT_A)
    if rc != 0:
        log(f"对照组失败（{rc}），中止")
        return rc

    rc = run("实验组", ["--ffn-mode", "moe", *MOE_ARGS], OUT_B)
    if rc != 0:
        log(f"实验组失败（{rc}），中止")
        return rc

    return report()


if __name__ == "__main__":
    raise SystemExit(main())
