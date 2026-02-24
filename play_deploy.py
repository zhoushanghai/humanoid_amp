#!/usr/bin/env python3
"""
play_deploy.py — 自动寻找最新 Checkpoint 并启动推理，或打开 TensorBoard

用法：
    python play_deploy.py                    # 自动找最新 checkpoint 并推理
    python play_deploy.py <checkpoint_path>  # 手动指定 checkpoint 并推理
    python play_deploy.py --tb               # 自动找最新 run 目录，打开 TensorBoard

项目专属配置集中在 "── 项目配置 ──" 区域，移植到其他项目时只需修改那一块。
"""

import os
import re
import sys
import subprocess
from pathlib import Path


# ─── 项目配置（仅需修改此处）────────────────────────────────────────────────
TASK = "Isaac-G1-AMP-Deploy-Direct-v0"
NUM_ENVS = 32
# log 根目录（相对于项目根目录）
LOG_BASE = "logs/skrl/g1_amp_dance"
# checkpoint 所在子目录名
CHECKPOINT_SUBDIR = "checkpoints"
# 要匹配的 checkpoint 文件 glob
CHECKPOINT_PATTERN = "agent_*.pt"
# 从文件名中提取 step 数字的正则
STEP_REGEX = re.compile(r"agent_(\d+)\.pt$")
# ──────────────────────────────────────────────────────────────────────────────


def extract_step(filename: str) -> int:
    """从 checkpoint 文件名中提取 step 号，找不到返回 -1。"""
    match = STEP_REGEX.search(filename)
    return int(match.group(1)) if match else -1


def find_latest_checkpoint(log_base: Path) -> Path:
    """
    在 log_base 下按文件名（时间戳）排序找最新的运行目录，
    再从其 checkpoints/ 中找 step 号最大的文件。
    """
    if not log_base.is_dir():
        raise FileNotFoundError(f"日志目录不存在: {log_base}")

    run_dirs = sorted(
        [d for d in log_base.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,  # 最新的排在前面
    )

    for run_dir in run_dirs:
        ckpt_dir = run_dir / CHECKPOINT_SUBDIR
        if not ckpt_dir.is_dir():
            continue
        candidates = list(ckpt_dir.glob(CHECKPOINT_PATTERN))
        if not candidates:
            continue
        return max(candidates, key=lambda p: extract_step(p.name))

    raise FileNotFoundError(f"在 {log_base} 下未找到任何 checkpoint")


def find_latest_run_dir(log_base: Path) -> Path:
    """
    在 log_base 下按文件名（时间戳）排序，返回最新的运行目录。
    """
    if not log_base.is_dir():
        raise FileNotFoundError(f"日志目录不存在: {log_base}")

    run_dirs = sorted(
        [d for d in log_base.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )

    if not run_dirs:
        raise FileNotFoundError(f"在 {log_base} 下未找到任何训练目录")

    return run_dirs[0]


def main():
    # ── --tb 模式：打开 TensorBoard ──────────────────────────────────────────
    if len(sys.argv) > 1 and sys.argv[1] == "--tb":
        log_base = Path(LOG_BASE)
        run_dir = find_latest_run_dir(log_base)
        print(f"[INFO] 打开 TensorBoard，logdir: {run_dir}")
        cmd = ["tensorboard", "--logdir", str(run_dir)]
        print(f"[INFO] 执行命令: {' '.join(cmd)}")
        os.execvp(cmd[0], cmd)

    # ── play 模式：确定 checkpoint 路径 ─────────────────────────────────────
    if len(sys.argv) > 1:
        checkpoint = sys.argv[1]
        print(f"[INFO] 使用指定 checkpoint: {checkpoint}")
    else:
        log_base = Path(LOG_BASE)
        print(f"[INFO] 自动搜索 checkpoint，日志目录: {log_base}")
        checkpoint = str(find_latest_checkpoint(log_base))
        print(f"[INFO] 找到最新 checkpoint: {checkpoint}")

    # 构建并执行 play 命令（继承当前进程的环境，输出直接打印）
    cmd = [
        sys.executable,
        "-m",
        "humanoid_amp.play",
        "--task",
        TASK,
        "--num_envs",
        str(NUM_ENVS),
        "--checkpoint",
        checkpoint,
    ]

    print(f"[INFO] 执行命令: {' '.join(cmd)}")
    os.execvp(cmd[0], cmd)  # 替换当前进程，信号传递更干净


if __name__ == "__main__":
    main()
