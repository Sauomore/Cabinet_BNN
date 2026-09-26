# -*- coding: utf-8 -*-
"""
训练完成后自动关机守护脚本。

⚠️ 为什么必须谨慎：
    训练脚本结束后还会做几件事 —— 写 final.pt、跑最终评测、生成样例、
    写 train_history.json。如果一看到 final.pt 就关机，这些可能被截断。
    监控脚本也还在跑，需要给它时间把最后的日志刷盘。

关机前会依次确认：
    ① final.pt 存在且大小稳定（连续两次采样不变，说明写完了）
    ② train_history.json 存在（训练的最后一个产物）
    ③ python 训练进程已退出
    ④ 额外等待缓冲期，让监控日志与磁盘缓冲刷干净

最后会把完整收尾信息写进 logs/shutdown.log，供第二天查看。

用法：
    python scripts/15_autoshutdown.py              # 确认后关机
    python scripts/15_autoshutdown.py --dry-run    # 只演练，不真关机
    python scripts/15_autoshutdown.py --cancel     # 取消（删除标志文件后退出）
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
TRAIN_DIR = ROOT / "results/lm_base_1ep"


def sh(cmd: list[str], timeout: int = 20) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def training_alive(train_dir: Path) -> bool:
    """训练进程是否还活着 —— 通过命令行精确匹配，而不是统计 python 进程总数。

    ⚠️ 这里踩过一个严重的坑：
        最初实现是「python 进程数 == 0」才算训练结束。
        但本守护脚本自己就是 python 进程，监控脚本也是 —— 于是条件
        永远不成立，训练 01:55 就结束了，机器却空转到 11 小时上限。
        代价：4.5 小时无意义空转。

    正确做法：按命令行内容识别训练进程（匹配 08_train_lm.py），
    并在计数时排除本进程。
    """
    me = os.getpid()
    out = sh(["powershell", "-NoProfile", "-Command",
              "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
              "Select-Object -ExpandProperty CommandLine"])
    for line in out.splitlines():
        if "08_train_lm" in line:
            return True
    return False


def other_python_procs() -> list[str]:
    """除自己以外的 python 进程命令行（诊断用）。"""
    me = os.getpid()
    out = sh(["powershell", "-NoProfile", "-Command",
              "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
              "ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }"])
    procs = []
    for line in out.splitlines():
        if "|" not in line:
            continue
        pid_s, _, cl = line.partition("|")
        try:
            if int(pid_s) == me:
                continue
        except ValueError:
            continue
        procs.append(cl)
    return procs


def main() -> int:
    ap = argparse.ArgumentParser(description="训练完成后自动关机")
    ap.add_argument("--dry-run", action="store_true", help="只演练，不真关机")
    ap.add_argument("--max-hours", type=float, default=12.0,
                    help="最长等待时间，超时则放弃关机（防止无限等）")
    ap.add_argument("--stable-checks", type=int, default=3,
                    help="final.pt 大小连续几次不变才算写完")
    ap.add_argument("--buffer-seconds", type=int, default=180,
                    help="所有条件满足后再等多久，让日志与磁盘刷干净")
    ap.add_argument("--interval", type=int, default=60, help="检查间隔（秒）")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logf = LOG_DIR / "shutdown.log"

    def emit(line: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with io.open(logf, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {line}\n")
        print(f"[{stamp}] {line}", flush=True)

    emit("=" * 60)
    emit(f"自动关机守护启动  dry_run={args.dry_run}  最长等待 {args.max_hours}h")
    emit(f"  训练目录 {TRAIN_DIR}")

    final_pt = TRAIN_DIR / "final.pt"
    history = TRAIN_DIR / "train_history.json"
    deadline = time.time() + args.max_hours * 3600
    last_size = -1
    stable = 0
    n = 0

    while time.time() < deadline:
        n += 1
        alive = training_alive(TRAIN_DIR)
        others = other_python_procs()
        cur_size = final_pt.stat().st_size if final_pt.exists() else -1

        if cur_size > 0 and cur_size == last_size:
            stable += 1
        else:
            stable = 0
        last_size = cur_size

        emit(f"#{n:03d} final.pt={'无' if cur_size < 0 else f'{cur_size:,}B'} "
             f"稳定={stable}/{args.stable_checks} "
             f"history={'有' if history.exists() else '无'} "
             f"训练进程={'存活' if alive else '已退出'} "
             f"其他python={len(others)}")

        ok = (stable >= args.stable_checks
              and history.exists()
              and not alive)

        if ok:
            emit("条件全部满足：训练已结束，产物已落盘")
            # 记录收尾信息（关机后仍可查看）
            try:
                if history.exists():
                    import json
                    h = json.loads(history.read_text(encoding="utf-8"))
                    emit("-" * 60)
                    emit("训练结果摘要（关机前记录，供次日查看）：")
                    emit(f"  最终 val_loss = {h.get('final_val_loss')}")
                    ps = h.get("params") or {}
                    if ps:
                        emit(f"  参数量       = {ps.get('total'):,}")
                    emit(f"  训练用时     = {h.get('minutes', 0):.1f} 分钟")
                    vl = h.get("val_loss") or []
                    if isinstance(vl, list) and vl:
                        emit("  验证集曲线：")
                        for item in vl[-6:]:
                            if isinstance(item, dict):
                                emit(f"    step {item.get('step')}: "
                                     f"val_loss {item.get('loss'):.4f}  "
                                     f"ppl {item.get('ppl'):.1f}")
                    import math
                    fv = h.get("final_val_loss")
                    if fv:
                        emit(f"  >> 最终困惑度 ppl = {math.exp(min(fv, 20)):.1f}")
                    emit("-" * 60)
            except Exception as e:
                emit(f"  (读取 history 失败: {e})")

            # 列出所有产物，确认齐全
            try:
                files = sorted(TRAIN_DIR.glob("*"))
                emit(f"  产物清单（{TRAIN_DIR.name}/）:")
                for f in files:
                    emit(f"    {f.name:<24} {f.stat().st_size:>14,} bytes")
            except Exception as e:
                emit(f"  (列目录失败: {e})")

            emit(f"  等待 {args.buffer_seconds}s 缓冲，让监控日志与磁盘缓冲刷干净…")
            time.sleep(args.buffer_seconds)

            if args.dry_run:
                emit("DRY-RUN：本应执行 shutdown /s /t 60，此处跳过")
                return 0

            emit("执行关机：shutdown /s /t 60  （60 秒后关机，可用 shutdown /a 取消）")
            r = sh(["shutdown", "/s", "/t", "60", "/c",
                    "Cabinet-BNN 训练完成，自动关机"])
            emit(f"  shutdown 返回: {r or '(无输出)'}")
            emit("关机指令已发出。若要取消，请在 60 秒内运行: shutdown /a")
            return 0

        time.sleep(args.interval)

    emit(f"等待超过 {args.max_hours} 小时仍不满足条件，放弃关机（请手动检查）")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
