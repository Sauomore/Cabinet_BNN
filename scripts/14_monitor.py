# -*- coding: utf-8 -*-
"""
长时间训练监控：定期采样系统 + GPU + 训练进度，写入日志。

为什么需要：
    训练要跑 6~8 小时，期间人不在。出问题（进程死掉、显存爆、温度过高、
    磁盘满）如果没人发现，损失的是整晚算力。

设计：
    · 每 INTERVAL 秒采样一次，追加到 logs/monitor_YYYYMMDD.log
    · 同时写一份最新状态到 logs/monitor_latest.txt（便于快速查看）
    · 异常情况写 ALERT 行，便于事后 grep
    · 检测到训练进程消失时，额外记录一条醒目告警（不自动重启，交给人决定）

用法：
    python scripts/14_monitor.py                       # 默认每 5 分钟，跑 8 小时
    python scripts/14_monitor.py --interval 300 --hours 8
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"


def sh(cmd: list[str], timeout: int = 20) -> str:
    """执行命令并返回 stdout（失败返回空串）。"""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def gpu_stats() -> dict:
    out = sh(["nvidia-smi",
              "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
              "--format=csv,noheader,nounits"])
    if not out:
        return {}
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    try:
        return {"util": float(parts[0]), "mem_used": float(parts[1]),
                "mem_total": float(parts[2]), "temp": float(parts[3]),
                "power": float(parts[4]) if len(parts) > 4 and parts[4] != "[N/A]" else -1}
    except Exception:
        return {}


def mem_stats() -> dict:
    """物理内存（不依赖 psutil）。"""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {"avail_gb": vm.available / 1024**3, "pct": vm.percent}
    except Exception:
        return {}


def py_procs() -> list[dict]:
    """枚举 python.exe 进程（PID + 常驻内存）。

    注意：不能用 wmic —— 新版 Windows 已将其移除（本机实测 "not recognized"）。
    这里用 tasklist，CSV 输出形如：
        "python.exe","21956","Console","1","2,916,636 K"
    内存字段带千位逗号与 K 后缀，需清洗。
    """
    procs = []
    out = sh(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"])
    for line in out.splitlines():
        line = line.strip()
        if not line or "No tasks" in line or not line.startswith('"'):
            continue
        try:
            # 简单 CSV 解析（字段内无逗号转义，因为数字用引号包裹）
            fields = [f.strip().strip('"') for f in line.split('","')]
            pid = int(fields[1])
            mem_raw = fields[-1].replace(",", "").replace("K", "").strip()
            rss_gb = float(mem_raw) / 1024**2
            procs.append({"pid": pid, "rss_gb": rss_gb})
        except Exception:
            continue
    return procs


def disk_free(drive: str) -> float:
    out = sh(["powershell", "-NoProfile", "-Command",
              f"(Get-PSDrive {drive}).Free"])
    try:
        return float(out) / 1024**3
    except Exception:
        return -1.0


def train_progress(out_dir: Path) -> dict:
    """从 checkpoint 和 history 推断训练进度。"""
    info = {"ckpts": [], "best": None, "history": None}
    if not out_dir.exists():
        return info
    for f in sorted(out_dir.glob("ckpt_*.pt")):
        try:
            info["ckpts"].append((f.name, f.stat().st_mtime))
        except Exception:
            pass
    b = out_dir / "best.pt"
    if b.exists():
        info["best"] = b.stat().st_mtime
    h = out_dir / "train_history.json"
    if h.exists():
        info["history"] = h.stat().st_mtime
    return info


def fmt_ts(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%H:%M:%S")


def main() -> int:
    ap = argparse.ArgumentParser(description="训练监控")
    ap.add_argument("--interval", type=int, default=300, help="采样间隔（秒）")
    ap.add_argument("--hours", type=float, default=8.0, help="监控时长（小时）")
    ap.add_argument("--train-dir", type=Path,
                    default=ROOT / "results/lm_base_1ep")
    ap.add_argument("--temp-warn", type=float, default=88.0)
    ap.add_argument("--temp-crit", type=float, default=93.0)
    ap.add_argument("--mem-crit-gb", type=float, default=3.0)
    ap.add_argument("--disk-crit-gb", type=float, default=5.0)
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y%m%d")
    logf = LOG_DIR / f"monitor_{day}.log"
    latest = LOG_DIR / "monitor_latest.txt"

    def emit(line: str, to_console: bool = True) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with io.open(logf, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {line}\n")
        if to_console:
            print(line, flush=True)

    emit(f"=== 监控启动  间隔 {args.interval}s  时长 {args.hours}h ===")
    emit(f"    训练目录 {args.train_dir}")
    emit(f"    日志 {logf}")

    end = time.time() + args.hours * 3600
    prev_ckpt = None
    seen_python = False
    dead_reported = False
    n = 0

    while time.time() < end:
        n += 1
        g = gpu_stats()
        m = mem_stats()
        procs = py_procs()
        d_c = disk_free("C")
        d_f = disk_free("F")
        tp = train_progress(args.train_dir)

        # --- 汇总一行 ---
        parts = [f"#{n:03d}"]
        if g:
            parts.append(f"GPU {g['util']:.0f}% {g['mem_used']:.0f}/{g['mem_total']:.0f}MB "
                         f"{g['temp']:.0f}C {g['power']:.0f}W")
        else:
            parts.append("GPU n/a")
        if m:
            parts.append(f"RAM {m['avail_gb']:.1f}G avail ({m['pct']:.0f}%)")
        parts.append(f"py={len(procs)}")
        if procs:
            parts.append(f"maxRSS {max(p['rss_gb'] for p in procs):.2f}G")
        parts.append(f"C:{d_c:.1f}G F:{d_f:.1f}G")
        if tp["ckpts"]:
            name, mt = tp["ckpts"][-1]
            parts.append(f"last {name}@{fmt_ts(mt)}")
        emit("  ".join(parts))

        # --- 告警 ---
        if g:
            if g["temp"] >= args.temp_crit:
                emit(f"ALERT  GPU 温度过高 {g['temp']:.0f}C >= {args.temp_crit}")
            elif g["temp"] >= args.temp_warn:
                emit(f"WARN   GPU 温度偏高 {g['temp']:.0f}C")
        if m and m["avail_gb"] < args.mem_crit_gb:
            emit(f"ALERT  内存不足 仅剩 {m['avail_gb']:.1f} GB")
        if 0 < d_c < args.disk_crit_gb:
            emit(f"ALERT  C 盘空间不足 {d_c:.1f} GB")
        if 0 < d_f < args.disk_crit_gb:
            emit(f"ALERT  F 盘空间不足 {d_f:.1f} GB")

        # --- 训练进程存活 ---
        if procs:
            seen_python = True
            dead_reported = False
        elif seen_python and not dead_reported:
            emit("ALERT  !! Python 进程消失 —— 训练可能已结束或崩溃，请检查 !!")
            emit(f"       查看训练日志与 {args.train_dir}")
            dead_reported = True

        # --- checkpoint 更新 ---
        if tp["ckpts"]:
            cur = tp["ckpts"][-1][0]
            if prev_ckpt is not None and cur != prev_ckpt:
                emit(f"OK     新 checkpoint: {cur}")
            prev_ckpt = cur

        # --- latest ---
        try:
            latest.write_text("\n".join([
                f"更新时间: {datetime.now():%Y-%m-%d %H:%M:%S}",
                f"采样序号: {n}",
                f"GPU: {g}",
                f"内存: {m}",
                f"Python 进程: {procs}",
                f"磁盘: C={d_c:.1f}G F={d_f:.1f}G",
                f"训练 checkpoint: {[c[0] for c in tp['ckpts']][-3:]}",
                f"日志: {logf}",
            ]), encoding="utf-8")
        except Exception:
            pass

        # 分段睡眠，便于中断
        slept = 0
        while slept < args.interval and time.time() < end:
            time.sleep(min(10, args.interval - slept))
            slept += 10

    emit(f"=== 监控结束，共采样 {n} 次 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
