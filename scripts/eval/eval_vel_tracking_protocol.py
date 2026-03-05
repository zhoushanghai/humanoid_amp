"""
文件用途: 按 Velocity Tracking 协议执行全量评测并导出汇总/明细结果。
主要内容: 配置读取、命令阶段调度(ramp/settle/record)、视频分段录制(含重置起始对齐段)、指标计算、CSV/JSON 导出与结果图表生成。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


parser = argparse.ArgumentParser(description="Evaluate velocity tracking protocol (full run).")
parser.add_argument("--config", type=str, required=True, help="Path to eval YAML config.")
parser.add_argument("--video", action="store_true", default=False, help="Record one evaluation video.")
parser.add_argument("--video_length", type=int, default=1200, help="Recorded video length in steps.")
parser.add_argument("--task", type=str, default=None, help="Task override.")
parser.add_argument("--num_envs", type=int, default=None, help="Override number of envs.")
parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path.")
parser.add_argument("--seed", type=int, default=None, help="Override seed.")
parser.add_argument("--algorithm", type=str, default=None, choices=["AMP", "PPO", "IPPO", "MAPPO"])
parser.add_argument("--ml_framework", type=str, default=None, choices=["torch", "jax", "jax-numpy"])

AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import skrl
import torch
from packaging import version

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import humanoid_amp  # noqa: F401

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    raise RuntimeError(f"Unsupported skrl version: {skrl.__version__}. need >= {SKRL_VERSION}")

if (args_cli.ml_framework or "torch").startswith("torch"):
    from skrl.utils.runner.torch import Runner
else:
    from skrl.utils.runner.jax import Runner


@dataclass(frozen=True)
class Combo:
    group: str
    combo_id: str
    vx: float
    vy: float
    wz: float


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    if yaml is not None:
        data = yaml.safe_load(raw) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Invalid config root type: {type(data).__name__}")
        return data

    # Minimal fallback parser for simple "key: value" config files.
    data: dict[str, object] = {}
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if ":" not in s:
            continue
        key, val = s.split(":", maxsplit=1)
        key = key.strip()
        val = val.strip()
        if val == "":
            continue
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            data[key] = val[1:-1]
            continue
        low = val.lower()
        if low in ("true", "false"):
            data[key] = low == "true"
            continue
        try:
            if "." in val:
                data[key] = float(val)
            else:
                data[key] = int(val)
            continue
        except ValueError:
            data[key] = val
    return data


cfg_file = _load_cfg(args_cli.config)
task_name = args_cli.task or cfg_file.get("task", "Isaac-G1-AMP-Deploy-Direct-v0")
algorithm_name = (args_cli.algorithm or cfg_file.get("algorithm", "AMP")).upper()
ml_framework = args_cli.ml_framework or cfg_file.get("ml_framework", "torch")
if args_cli.num_envs is None:
    num_envs = int(cfg_file.get("num_envs", 64))
else:
    num_envs = int(args_cli.num_envs)
checkpoint_path = args_cli.checkpoint or cfg_file.get("active_checkpoint")
if not checkpoint_path:
    raise ValueError("Missing checkpoint. Provide --checkpoint or config.active_checkpoint")
device_name = args_cli.device or cfg_file.get("device")
seed_value = args_cli.seed if args_cli.seed is not None else int(cfg_file.get("seed", 42))
output_root = Path(cfg_file.get("output_dir", "outputs/vel_tracking"))

ramp_inc = float(cfg_file.get("ramp_inc", 0.2))
ramp_dur = float(cfg_file.get("ramp_dur", 0.5))
settle_s = float(cfg_file.get("settle_s", 2.0))
record_s = float(cfg_file.get("record_s", 10.0))
max_vx_pass_err = float(cfg_file.get("max_vx_pass_err", 0.5))
max_vy_pass_err = float(cfg_file.get("max_vy_pass_err", 0.3))
fixed_speed_from_start = bool(cfg_file.get("fixed_speed_from_start", True))
reset_between_combos = bool(cfg_file.get("reset_between_combos", True))
eval_reset_strategy = str(cfg_file.get("eval_reset_strategy", "default"))
reset_sync_steps = int(cfg_file.get("reset_sync_steps", 2))
video_camera_zoom_out = float(cfg_file.get("video_camera_zoom_out", 1.25))
video_camera_lift_z = float(cfg_file.get("video_camera_lift_z", 0.2))
video_render_flush_frames = int(cfg_file.get("video_render_flush_frames", 8))
video_reset_lead_in_s = float(cfg_file.get("video_reset_lead_in_s", 0.5))

agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm_name.lower() == "ppo" else f"skrl_{algorithm_name.lower()}_cfg_entry_point"


def _steps_from_seconds(seconds: float, step_dt: float) -> int:
    return max(1, int(round(seconds / step_dt)))


def _get_policy_actions(runner: Runner, obs: dict, env) -> torch.Tensor | dict:
    outputs = runner.agent.act(obs, timestep=0, timesteps=0)
    if hasattr(env, "possible_agents"):
        return {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
    return outputs[-1].get("mean_actions", outputs[0])


def _extract_done_mask(terminated, truncated) -> torch.Tensor:
    if not torch.is_tensor(terminated):
        terminated = torch.as_tensor(terminated)
    if not torch.is_tensor(truncated):
        truncated = torch.as_tensor(truncated)
    done = torch.logical_or(terminated.bool(), truncated.bool())
    return done.reshape(-1)


def _set_command(unwrapped_env, target: torch.Tensor) -> None:
    unwrapped_env.set_fixed_command_targets(target)


def _current_cmd_tensor(unwrapped_env) -> torch.Tensor:
    return unwrapped_env.command_target_speed


def _current_actual_speeds(unwrapped_env) -> tuple[torch.Tensor, torch.Tensor]:
    current_speed_w = unwrapped_env.robot.data.body_lin_vel_w[:, unwrapped_env.ref_body_index]
    current_quat_w = unwrapped_env.robot.data.body_quat_w[:, unwrapped_env.ref_body_index]
    current_speed_b = quat_apply_inverse(current_quat_w, current_speed_w)
    lin_xy = current_speed_b[:, :2]

    current_ang_vel_w = unwrapped_env.robot.data.body_ang_vel_w[:, unwrapped_env.ref_body_index]
    current_ang_vel_b = quat_apply_inverse(current_quat_w, current_ang_vel_w)
    yaw_z = current_ang_vel_b[:, 2]
    return lin_xy, yaw_z


def _run_steps(env, runner: Runner, obs: dict, num_steps: int, on_step=None, on_frame=None):
    for _ in range(num_steps):
        # Do not run env.step under inference_mode: it can create inference tensors
        # inside simulator buffers and break subsequent reset() inplace updates.
        with torch.no_grad():
            actions = _get_policy_actions(runner, obs, env)
        obs, _, terminated, truncated, _ = env.step(actions)
        done_mask = _extract_done_mask(terminated, truncated)
        if on_step is not None:
            on_step(done_mask)
        if on_frame is not None:
            on_frame()
    return obs


def _build_combos() -> list[Combo]:
    combos: list[Combo] = []

    for vx in [0.5, 1.0, 1.5, 2.0, 2.5]:
        combos.append(Combo("low_lin", f"low_lin_vx_{vx:.2f}", vx, 0.0, 0.0))
    for vy in [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]:
        combos.append(Combo("low_lin", f"low_lin_vy_{vy:.2f}", 0.0, vy, 0.0))

    for vx in [3.0, 3.25, 3.5, 3.75, 4.0]:
        combos.append(Combo("high_lin", f"high_lin_vx_{vx:.2f}", vx, 0.0, 0.0))
    for vy in [-2.5, -2.0, -1.5, 1.5, 2.0, 2.5]:
        combos.append(Combo("high_lin", f"high_lin_vy_{vy:.2f}", 0.0, vy, 0.0))

    for wz in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        combos.append(Combo("yaw_low", f"yaw_low_wz_{wz:.2f}", 1.0, 0.0, wz))
    for wz in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        combos.append(Combo("yaw_high", f"yaw_high_wz_{wz:.2f}", 3.0, 0.0, wz))

    return combos


def _ramp_targets(goal: torch.Tensor, inc: float) -> list[torch.Tensor]:
    # component-wise step ramp from 0 toward goal
    ramps: list[torch.Tensor] = []
    cur = torch.zeros_like(goal)
    while True:
        delta = goal - cur
        step = torch.sign(delta) * torch.minimum(torch.abs(delta), torch.full_like(delta, inc))
        nxt = cur + step
        ramps.append(nxt.clone())
        cur = nxt
        if torch.allclose(cur, goal):
            break
    return ramps


def _acc_from_errors_lin(err_lin_bar: torch.Tensor, cmd_xy: torch.Tensor) -> torch.Tensor:
    denom = torch.maximum(torch.norm(cmd_xy, dim=-1), torch.full_like(err_lin_bar, 0.1))
    raw = 1.0 - err_lin_bar / denom
    return torch.clamp(raw, 0.0, 1.0) * 100.0


def _acc_from_errors_yaw(err_yaw_bar: torch.Tensor, cmd_wz: torch.Tensor) -> torch.Tensor:
    denom = torch.maximum(torch.abs(cmd_wz), torch.full_like(err_yaw_bar, 0.1))
    raw = 1.0 - err_yaw_bar / denom
    return torch.clamp(raw, 0.0, 1.0) * 100.0


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _safe_std(values: list[float]) -> float:
    if not values:
        return float("nan")
    m = _safe_mean(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return float(math.sqrt(var))


class PerComboVideoRecorder:
    def __init__(self, enabled: bool, render_env, video_dir: Path, render_flush_frames: int = 0):
        self.enabled = enabled
        self.render_env = render_env
        self.video_dir = video_dir
        self.render_flush_frames = max(0, int(render_flush_frames))
        self.writer = None
        if self.enabled:
            self.video_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_name(self, name: str) -> str:
        safe = []
        for c in name:
            if c.isalnum() or c in ("-", "_", "."):
                safe.append(c)
            else:
                safe.append("_")
        return "".join(safe)

    def start(self, item_name: str):
        if not self.enabled:
            return
        self.close()
        # Isaac rendering can be one/few frames behind after reset; drop stale frames before opening a new file.
        for _ in range(self.render_flush_frames):
            try:
                self.render_env.render()
            except Exception:
                break
        out = self.video_dir / f"{self._sanitize_name(item_name)}.mp4"
        self.writer = imageio.get_writer(str(out), fps=60)

    def capture(self):
        if not self.enabled or self.writer is None:
            return
        frame = self.render_env.render()
        if frame is not None:
            self.writer.append_data(frame)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def _record_video_reset_lead_in(
    env,
    runner: Runner,
    obs: dict,
    unwrapped_env,
    step_dt: float,
    video_recorder: PerComboVideoRecorder | None,
    lead_in_s: float,
):
    """Record a short zero-command pre-roll so each video starts from reset formation."""
    if video_recorder is None:
        return obs
    lead_in_s = max(0.0, float(lead_in_s))
    lead_steps = _steps_from_seconds(lead_in_s, step_dt) if lead_in_s > 0.0 else 0
    _set_command(unwrapped_env, torch.zeros((num_envs, int(unwrapped_env.command_dim)), device=unwrapped_env.device))
    video_recorder.capture()
    if lead_steps > 0:
        obs = _run_steps(env, runner, obs, lead_steps, on_frame=video_recorder.capture)
    return obs


def _evaluate_combo(
    env,
    render_env,
    unwrapped_env,
    runner: Runner,
    obs: dict,
    combo: Combo,
    step_dt: float,
    use_fixed_speed_from_start: bool,
    video_recorder: PerComboVideoRecorder | None,
    reset_before_start: bool,
    reset_sync_steps_local: int,
    video_reset_lead_in_s_local: float,
) -> tuple[dict, dict]:
    if reset_before_start:
        # Reset first and run at least one sync step so the first captured frame is from the new test.
        obs = _reset_env_for_new_test(env, unwrapped_env, runner, max(1, int(reset_sync_steps_local)))

    n_env = int(unwrapped_env.num_envs)
    command_dim = int(unwrapped_env.command_dim)
    goal = torch.zeros((n_env, command_dim), device=unwrapped_env.device, dtype=torch.float32)
    goal[:, 0] = combo.vx
    goal[:, 1] = combo.vy
    if command_dim == 3:
        goal[:, 2] = combo.wz

    survived = torch.ones((n_env,), device=unwrapped_env.device, dtype=torch.bool)

    def update_survival(done_mask: torch.Tensor):
        nonlocal survived
        survived = torch.logical_and(survived, ~done_mask.to(unwrapped_env.device))

    if video_recorder is not None:
        video_recorder.start(combo.combo_id)
        obs = _record_video_reset_lead_in(
            env,
            runner,
            obs,
            unwrapped_env,
            step_dt,
            video_recorder,
            video_reset_lead_in_s_local,
        )

    if use_fixed_speed_from_start:
        _set_command(unwrapped_env, goal)
    else:
        ramp_steps = _steps_from_seconds(ramp_dur, step_dt)
        for ramp_target in _ramp_targets(goal, ramp_inc):
            _set_command(unwrapped_env, ramp_target)
            obs = _run_steps(
                env,
                runner,
                obs,
                ramp_steps,
                on_step=update_survival,
                on_frame=(video_recorder.capture if video_recorder is not None else None),
            )

    _set_command(unwrapped_env, goal)
    obs = _run_steps(
        env,
        runner,
        obs,
        _steps_from_seconds(settle_s, step_dt),
        on_step=update_survival,
        on_frame=(video_recorder.capture if video_recorder is not None else None),
    )

    record_steps = _steps_from_seconds(record_s, step_dt)
    err_lin_sum = torch.zeros((n_env,), device=unwrapped_env.device, dtype=torch.float32)
    err_yaw_sum = torch.zeros((n_env,), device=unwrapped_env.device, dtype=torch.float32)
    # record-window survival time (seconds), only counting record window
    record_survival_s = torch.zeros((n_env,), device=unwrapped_env.device, dtype=torch.float32)
    alive_in_record = survived.clone()
    record_started_alive = survived.clone()

    def collect_errors(done_mask: torch.Tensor):
        nonlocal survived, err_lin_sum, err_yaw_sum, record_survival_s, alive_in_record
        alive_before_step = alive_in_record.clone()
        record_survival_s += alive_before_step.float() * float(step_dt)
        alive_in_record = torch.logical_and(alive_in_record, ~done_mask.to(unwrapped_env.device))
        survived = torch.logical_and(survived, ~done_mask.to(unwrapped_env.device))
        cmd = _current_cmd_tensor(unwrapped_env)
        act_lin_xy, act_yaw = _current_actual_speeds(unwrapped_env)
        err_lin = torch.norm(cmd[:, :2] - act_lin_xy, dim=-1)
        err_yaw = torch.abs(cmd[:, 2] - act_yaw) if command_dim == 3 else torch.zeros_like(err_lin)
        err_lin_sum += err_lin
        err_yaw_sum += err_yaw

    obs = _run_steps(
        env,
        runner,
        obs,
        record_steps,
        on_step=collect_errors,
        on_frame=(video_recorder.capture if video_recorder is not None else None),
    )

    err_lin_bar = err_lin_sum / float(record_steps)
    err_yaw_bar = err_yaw_sum / float(record_steps)
    cmd = _current_cmd_tensor(unwrapped_env)
    acc_lin = _acc_from_errors_lin(err_lin_bar, cmd[:, :2])
    acc_yaw = _acc_from_errors_yaw(err_yaw_bar, cmd[:, 2]) if command_dim == 3 else torch.zeros_like(err_lin_bar)
    survived_full_record = torch.logical_and(record_started_alive, alive_in_record)

    # Per-env tracking accuracy for this combo:
    # - yaw groups use yaw accuracy
    # - all other groups use linear accuracy
    combo_is_yaw = combo.group.startswith("yaw")
    tracking_acc = acc_yaw.clone() if combo_is_yaw else acc_lin.clone()
    tracking_acc[~survived_full_record] = torch.nan

    survived_cpu = survived.detach().cpu()
    valid = int(survived_cpu.sum().item()) > 0

    detail = {
        "group": combo.group,
        "combo_id": combo.combo_id,
        "cmd_vx": combo.vx,
        "cmd_vy": combo.vy,
        "cmd_wz": combo.wz,
        "n_env": n_env,
        "n_survived": int(survived_cpu.sum().item()),
        "valid_combo": int(valid),
        "lin_err_mean": float(err_lin_bar[survived].mean().item()) if valid else float("nan"),
        "yaw_err_mean": float(err_yaw_bar[survived].mean().item()) if valid else float("nan"),
        "lin_acc_mean": float(acc_lin[survived].mean().item()) if valid else float("nan"),
        "yaw_acc_mean": float(acc_yaw[survived].mean().item()) if valid else float("nan"),
    }

    per_env = {
        "survived": survived_cpu,
        "err_lin_bar": err_lin_bar.detach().cpu(),
        "err_yaw_bar": err_yaw_bar.detach().cpu(),
        "acc_lin": acc_lin.detach().cpu(),
        "acc_yaw": acc_yaw.detach().cpu(),
        "record_survival_s": record_survival_s.detach().cpu(),
        "tracking_acc": tracking_acc.detach().cpu(),
        "tracking_metric": ("yaw_acc" if combo_is_yaw else "lin_acc"),
        "survived_full_record": survived_full_record.detach().cpu(),
    }
    if video_recorder is not None:
        video_recorder.close()
    return detail, per_env, obs


def _evaluate_step_survival(
    env,
    render_env,
    unwrapped_env,
    runner: Runner,
    obs: dict,
    step_dt: float,
    video_recorder: PerComboVideoRecorder | None,
    reset_before_start: bool,
    reset_sync_steps_local: int,
    video_reset_lead_in_s_local: float,
) -> tuple[float, dict]:
    if reset_before_start:
        # Keep step-survival as one sequence, but ensure it starts from a clean reset state.
        obs = _reset_env_for_new_test(env, unwrapped_env, runner, max(1, int(reset_sync_steps_local)))

    n_env = int(unwrapped_env.num_envs)
    command_dim = int(unwrapped_env.command_dim)
    survived = torch.ones((n_env,), device=unwrapped_env.device, dtype=torch.bool)
    phase_logs: list[dict] = []
    sequence = [
        ((0.0, 0.0, 0.0), 3.0),
        ((4.0, 0.0, 0.0), 10.0),
        ((0.0, 2.0, 0.0), 10.0),
        ((4.0, 0.0, 0.0), 10.0),
    ]

    for phase_idx, (cmd_tuple, dur_s) in enumerate(sequence, start=1):
        if video_recorder is not None:
            video_recorder.start(f"step_phase_{phase_idx}_{cmd_tuple[0]}_{cmd_tuple[1]}_{cmd_tuple[2]}")
            obs = _record_video_reset_lead_in(
                env,
                runner,
                obs,
                unwrapped_env,
                step_dt,
                video_recorder,
                video_reset_lead_in_s_local,
            )
        target = torch.zeros((n_env, command_dim), device=unwrapped_env.device, dtype=torch.float32)
        target[:, 0] = cmd_tuple[0]
        target[:, 1] = cmd_tuple[1]
        if command_dim == 3:
            target[:, 2] = cmd_tuple[2]
        _set_command(unwrapped_env, target)

        steps = _steps_from_seconds(dur_s, step_dt)
        last5_count = max(1, _steps_from_seconds(5.0, step_dt))
        err_lin_tail = []
        err_yaw_tail = []

        def on_step(done_mask: torch.Tensor):
            nonlocal survived
            survived = torch.logical_and(survived, ~done_mask.to(unwrapped_env.device))
            cmd = _current_cmd_tensor(unwrapped_env)
            act_lin_xy, act_yaw = _current_actual_speeds(unwrapped_env)
            err_lin = torch.norm(cmd[:, :2] - act_lin_xy, dim=-1)
            err_yaw = torch.abs(cmd[:, 2] - act_yaw) if command_dim == 3 else torch.zeros_like(err_lin)
            err_lin_tail.append(err_lin.detach().cpu())
            err_yaw_tail.append(err_yaw.detach().cpu())
            if len(err_lin_tail) > last5_count:
                err_lin_tail.pop(0)
                err_yaw_tail.pop(0)

        obs = _run_steps(
            env,
            runner,
            obs,
            steps,
            on_step=on_step,
            on_frame=(video_recorder.capture if video_recorder is not None else None),
        )
        alive_rate = float(survived.float().mean().item())
        lin_tail = torch.stack(err_lin_tail).mean(dim=0) if err_lin_tail else torch.zeros((n_env,))
        yaw_tail = torch.stack(err_yaw_tail).mean(dim=0) if err_yaw_tail else torch.zeros((n_env,))
        phase_logs.append(
            {
                "phase": phase_idx,
                "cmd_vx": cmd_tuple[0],
                "cmd_vy": cmd_tuple[1],
                "cmd_wz": cmd_tuple[2],
                "duration_s": dur_s,
                "alive_rate": alive_rate,
                "tail5_lin_err_mean": float(lin_tail.mean().item()),
                "tail5_yaw_err_mean": float(yaw_tail.mean().item()),
            }
        )
        if video_recorder is not None:
            video_recorder.close()

    return float(survived.float().mean().item()), {"step_phase_logs": phase_logs}, obs


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _plot_combo_survival(details: list[dict], out_dir: Path) -> None:
    labels = [d["combo_id"] for d in details]
    survived_counts = [int(d["n_survived"]) for d in details]
    valid_flags = [int(d["valid_combo"]) for d in details]

    colors = ["#2ca02c" if v == 1 else "#d62728" for v in valid_flags]
    fig_h = max(8, 0.22 * len(labels))
    fig, ax = plt.subplots(figsize=(16, fig_h))
    y_pos = list(range(len(labels)))
    ax.barh(y_pos, survived_counts, color=colors, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Survived Envs")
    ax.set_title("Velocity Tracking: Survived Count per Combo")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "combo_survival.png", dpi=180)
    plt.close(fig)


def _plot_combo_lin_acc(details: list[dict], out_dir: Path) -> None:
    labels = []
    lin_acc = []
    for d in details:
        if int(d["valid_combo"]) != 1:
            continue
        val = d["lin_acc_mean"]
        if isinstance(val, float) and math.isnan(val):
            continue
        labels.append(d["combo_id"])
        lin_acc.append(float(val))

    if not labels:
        return

    fig_h = max(6, 0.20 * len(labels))
    fig, ax = plt.subplots(figsize=(16, fig_h))
    y_pos = list(range(len(labels)))
    ax.barh(y_pos, lin_acc, color="#1f77b4")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Linear Tracking Accuracy (%)")
    ax.set_title("Velocity Tracking: Linear Accuracy per Valid Combo")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "combo_lin_acc.png", dpi=180)
    plt.close(fig)


def _plot_summary_metrics(summary: dict, out_dir: Path) -> None:
    keys = [
        "low_lin_lin_acc",
        "high_lin_lin_acc",
        "yaw_low_yaw_acc",
        "yaw_high_yaw_acc",
        "max_vx",
        "max_vy",
        "step_survival",
    ]
    vals = []
    labels = []
    for k in keys:
        v = summary.get(k, float("nan"))
        if isinstance(v, float) and math.isnan(v):
            continue
        labels.append(k)
        vals.append(float(v))

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, vals, color="#ff7f0e")
    ax.set_title("Velocity Tracking: Summary Metrics")
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_metrics.png", dpi=180)
    plt.close(fig)


def _plot_step_alive(step_phase_logs: list[dict], out_dir: Path) -> None:
    if not step_phase_logs:
        return
    phases = [int(p["phase"]) for p in step_phase_logs]
    alive = [float(p["alive_rate"]) for p in step_phase_logs]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(phases, alive, marker="o", linewidth=2, color="#9467bd")
    ax.set_xticks(phases)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Step Phase")
    ax.set_ylabel("Alive Rate")
    ax.set_title("Step Response: Alive Rate by Phase")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "step_alive_rate.png", dpi=180)
    plt.close(fig)


def _build_group_colors(details: list[dict]) -> dict[str, str]:
    # Use only two shades as requested: dark blue / light blue (alternating by group order)
    shades = ["#0b3c6d", "#5fa8ff"]
    groups = []
    for d in details:
        g = d["group"]
        if g not in groups:
            groups.append(g)
    return {g: shades[i % 2] for i, g in enumerate(groups)}


def _plot_global_record_survival(per_env_rows: list[dict], details: list[dict], out_dir: Path) -> None:
    combo_order = [d["combo_id"] for d in details]
    combo_to_group = {d["combo_id"]: d["group"] for d in details}
    group_colors = _build_group_colors(details)

    data = []
    labels = []
    colors = []
    for combo_id in combo_order:
        vals = [float(r["record_survival_s"]) for r in per_env_rows if r["combo_id"] == combo_id]
        if not vals:
            continue
        data.append(vals)
        labels.append(combo_id)
        colors.append(group_colors[combo_to_group[combo_id]])

    if not data:
        return

    fig, ax = plt.subplots(figsize=(18, 7))
    bplot = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Record-window Survival Time (s)")
    ax.set_title("Global Summary: Record-window Survival Time per Combo (Per-env)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "global_record_survival_boxplot.png", dpi=180)
    plt.close(fig)


def _plot_global_tracking_acc(per_env_rows: list[dict], details: list[dict], out_dir: Path) -> None:
    combo_order = [d["combo_id"] for d in details]
    combo_to_group = {d["combo_id"]: d["group"] for d in details}
    group_colors = _build_group_colors(details)

    data = []
    labels = []
    colors = []
    for combo_id in combo_order:
        vals = []
        for r in per_env_rows:
            if r["combo_id"] != combo_id:
                continue
            v = r["tracking_acc"]
            if isinstance(v, float) and math.isnan(v):
                continue
            vals.append(float(v))
        if not vals:
            continue
        data.append(vals)
        labels.append(combo_id)
        colors.append(group_colors[combo_to_group[combo_id]])

    if not data:
        return

    fig, ax = plt.subplots(figsize=(18, 7))
    bplot = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Tracking Accuracy (%)")
    ax.set_title("Global Summary: Tracking Accuracy per Combo (Per-env)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "global_tracking_acc_boxplot.png", dpi=180)
    plt.close(fig)


def _reset_env_for_new_test(env, unwrapped_env, runner: Runner, sync_steps: int):
    # skrl IsaacLabWrapper caches reset and only performs a real reset once.
    # Force it to execute a full underlying env reset for every test item.
    if hasattr(env, "_reset_once"):
        try:
            env._reset_once = True
        except Exception:
            pass
    obs, _ = env.reset()
    _set_command(unwrapped_env, torch.zeros((num_envs, int(unwrapped_env.command_dim)), device=unwrapped_env.device))
    if sync_steps > 0:
        obs = _run_steps(env, runner, obs, sync_steps)
    return obs


@hydra_task_config(task_name, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    env_cfg.scene.num_envs = num_envs
    if device_name is not None:
        env_cfg.sim.device = device_name
    # For evaluation consistency: start each combo from a canonical formation.
    if hasattr(env_cfg, "reset_strategy"):
        env_cfg.reset_strategy = eval_reset_strategy

    if seed_value == -1:
        random.seed(int(time.time()))
        resolved_seed = random.randint(0, 10000)
    else:
        resolved_seed = seed_value
    experiment_cfg["seed"] = resolved_seed
    env_cfg.seed = resolved_seed

    ckpt_abs = os.path.abspath(checkpoint_path)
    if not os.path.isfile(ckpt_abs):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_abs}")
    run_dir = os.path.dirname(os.path.dirname(ckpt_abs))
    env_cfg.log_dir = run_dir

    # Video-only camera adjustment: slightly zoom out and lift camera to enlarge visible area.
    if args_cli.video and hasattr(env_cfg, "viewer"):
        try:
            eye = list(env_cfg.viewer.eye)
            lookat = list(env_cfg.viewer.lookat)
            if len(eye) == 3 and len(lookat) == 3:
                for i in range(3):
                    eye[i] = lookat[i] + (eye[i] - lookat[i]) * video_camera_zoom_out
                eye[2] += video_camera_lift_z
                env_cfg.viewer.eye = tuple(eye)
                env_cfg.viewer.lookat = tuple(lookat)
        except Exception:
            pass

    raw_env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = raw_env
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm_name.lower() in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    step_dt = float(getattr(env, "step_dt", env.unwrapped.step_dt))
    unwrapped_env = env.unwrapped
    if not hasattr(unwrapped_env, "set_fixed_command_targets"):
        raise RuntimeError("Env does not support fixed command targets.")
    if int(getattr(unwrapped_env, "command_dim", 0)) != 3:
        raise RuntimeError(f"This evaluator expects command_dim=3, got {getattr(unwrapped_env, 'command_dim', None)}")

    env = SkrlVecEnvWrapper(env, ml_framework=ml_framework)
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    if str(experiment_cfg.get("agent", {}).get("class", "")).upper() == "AMP":
        experiment_cfg["agent"]["amp_batch_size"] = min(int(experiment_cfg["agent"].get("amp_batch_size", 512)), 64)
        experiment_cfg["agent"]["discriminator_batch_size"] = min(
            int(experiment_cfg["agent"].get("discriminator_batch_size", 4096)), 512
        )
        if "motion_dataset" in experiment_cfg:
            experiment_cfg["motion_dataset"]["memory_size"] = min(
                int(experiment_cfg["motion_dataset"].get("memory_size", 200000)), 20000
            )
        if "reply_buffer" in experiment_cfg:
            experiment_cfg["reply_buffer"]["memory_size"] = min(
                int(experiment_cfg["reply_buffer"].get("memory_size", 1000000)), 50000
            )

    runner = Runner(env, experiment_cfg)
    try:
        env.unwrapped._skrl_agent = runner.agent
    except Exception:
        pass
    runner.agent.load(ckpt_abs)
    runner.agent.set_running_mode("eval")

    obs, _ = env.reset()

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_name = os.path.splitext(os.path.basename(ckpt_abs))[0]
    out_dir = output_root / f"{ckpt_name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_recorder = PerComboVideoRecorder(
        args_cli.video,
        raw_env,
        out_dir / "videos",
        render_flush_frames=video_render_flush_frames,
    )

    details: list[dict] = []
    per_env_rows: list[dict] = []
    group_lin_scores: dict[str, list[float]] = {"low_lin": [], "high_lin": []}
    group_yaw_scores: dict[str, list[float]] = {"yaw_low": [], "yaw_high": []}

    combos = _build_combos()
    for combo in combos:
        detail, per_env, obs = _evaluate_combo(
            env,
            raw_env,
            unwrapped_env,
            runner,
            obs,
            combo,
            step_dt,
            use_fixed_speed_from_start=fixed_speed_from_start,
            video_recorder=video_recorder,
            reset_before_start=reset_between_combos,
            reset_sync_steps_local=reset_sync_steps,
            video_reset_lead_in_s_local=video_reset_lead_in_s,
        )
        details.append(detail)
        for env_id in range(int(num_envs)):
            per_env_rows.append(
                {
                    "group": combo.group,
                    "combo_id": combo.combo_id,
                    "env_id": env_id,
                    "record_survival_s": float(per_env["record_survival_s"][env_id].item()),
                    "tracking_acc": float(per_env["tracking_acc"][env_id].item()),
                    "tracking_metric": per_env["tracking_metric"],
                    "survived_full_record": int(bool(per_env["survived_full_record"][env_id].item())),
                }
            )
        if detail["valid_combo"]:
            if combo.group in group_lin_scores:
                group_lin_scores[combo.group].append(detail["lin_acc_mean"])
            if combo.group in group_yaw_scores:
                group_yaw_scores[combo.group].append(detail["yaw_acc_mean"])

    vx_scan = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    vy_scan = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    max_vx_per_env = torch.zeros((num_envs,), dtype=torch.float32)
    max_vy_per_env = torch.zeros((num_envs,), dtype=torch.float32)

    for spd in vx_scan:
        combo = Combo("max_vx_scan", f"max_vx_{spd:.2f}", spd, 0.0, 0.0)
        detail, per_env, obs = _evaluate_combo(
            env,
            raw_env,
            unwrapped_env,
            runner,
            obs,
            combo,
            step_dt,
            use_fixed_speed_from_start=fixed_speed_from_start,
            video_recorder=video_recorder,
            reset_before_start=reset_between_combos,
            reset_sync_steps_local=reset_sync_steps,
            video_reset_lead_in_s_local=video_reset_lead_in_s,
        )
        details.append(detail)
        for env_id in range(int(num_envs)):
            per_env_rows.append(
                {
                    "group": combo.group,
                    "combo_id": combo.combo_id,
                    "env_id": env_id,
                    "record_survival_s": float(per_env["record_survival_s"][env_id].item()),
                    "tracking_acc": float(per_env["tracking_acc"][env_id].item()),
                    "tracking_metric": per_env["tracking_metric"],
                    "survived_full_record": int(bool(per_env["survived_full_record"][env_id].item())),
                }
            )
        pass_mask = per_env["survived"] & (per_env["err_lin_bar"] < max_vx_pass_err)
        max_vx_per_env[pass_mask] = float(spd)

    for spd in vy_scan:
        combo = Combo("max_vy_scan", f"max_vy_{spd:.2f}", 0.0, spd, 0.0)
        detail, per_env, obs = _evaluate_combo(
            env,
            raw_env,
            unwrapped_env,
            runner,
            obs,
            combo,
            step_dt,
            use_fixed_speed_from_start=fixed_speed_from_start,
            video_recorder=video_recorder,
            reset_before_start=reset_between_combos,
            reset_sync_steps_local=reset_sync_steps,
            video_reset_lead_in_s_local=video_reset_lead_in_s,
        )
        details.append(detail)
        for env_id in range(int(num_envs)):
            per_env_rows.append(
                {
                    "group": combo.group,
                    "combo_id": combo.combo_id,
                    "env_id": env_id,
                    "record_survival_s": float(per_env["record_survival_s"][env_id].item()),
                    "tracking_acc": float(per_env["tracking_acc"][env_id].item()),
                    "tracking_metric": per_env["tracking_metric"],
                    "survived_full_record": int(bool(per_env["survived_full_record"][env_id].item())),
                }
            )
        pass_mask = per_env["survived"] & (per_env["err_lin_bar"] < max_vy_pass_err)
        max_vy_per_env[pass_mask] = float(spd)

    step_survival, step_diag, obs = _evaluate_step_survival(
        env,
        raw_env,
        unwrapped_env,
        runner,
        obs,
        step_dt,
        video_recorder=video_recorder,
        reset_before_start=reset_between_combos,
        reset_sync_steps_local=reset_sync_steps,
        video_reset_lead_in_s_local=video_reset_lead_in_s,
    )

    summary = {
        "low_lin_lin_acc": _safe_mean(group_lin_scores["low_lin"]),
        "low_lin_lin_acc_std": _safe_std(group_lin_scores["low_lin"]),
        "high_lin_lin_acc": _safe_mean(group_lin_scores["high_lin"]),
        "high_lin_lin_acc_std": _safe_std(group_lin_scores["high_lin"]),
        "yaw_low_yaw_acc": _safe_mean(group_yaw_scores["yaw_low"]),
        "yaw_low_yaw_acc_std": _safe_std(group_yaw_scores["yaw_low"]),
        "yaw_high_yaw_acc": _safe_mean(group_yaw_scores["yaw_high"]),
        "yaw_high_yaw_acc_std": _safe_std(group_yaw_scores["yaw_high"]),
        "max_vx": float(max_vx_per_env.mean().item()),
        "max_vx_std": float(max_vx_per_env.std(unbiased=False).item()),
        "max_vy": float(max_vy_per_env.mean().item()),
        "max_vy_std": float(max_vy_per_env.std(unbiased=False).item()),
        "step_survival": float(step_survival),
    }

    summary_fields = [
        "low_lin_lin_acc",
        "low_lin_lin_acc_std",
        "high_lin_lin_acc",
        "high_lin_lin_acc_std",
        "yaw_low_yaw_acc",
        "yaw_low_yaw_acc_std",
        "yaw_high_yaw_acc",
        "yaw_high_yaw_acc_std",
        "max_vx",
        "max_vx_std",
        "max_vy",
        "max_vy_std",
        "step_survival",
    ]
    _write_csv(out_dir / "metrics_summary.csv", [summary], summary_fields)

    detail_fields = [
        "group",
        "combo_id",
        "cmd_vx",
        "cmd_vy",
        "cmd_wz",
        "n_env",
        "n_survived",
        "valid_combo",
        "lin_err_mean",
        "yaw_err_mean",
        "lin_acc_mean",
        "yaw_acc_mean",
    ]
    _write_csv(out_dir / "metrics_combo_details.csv", details, detail_fields)
    _write_csv(
        out_dir / "metrics_per_env_details.csv",
        per_env_rows,
        ["group", "combo_id", "env_id", "record_survival_s", "tracking_acc", "tracking_metric", "survived_full_record"],
    )

    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config_path": args_cli.config,
                "task": task_name,
                "algorithm": algorithm_name,
                "ml_framework": ml_framework,
                "num_envs": num_envs,
                "seed": resolved_seed,
                "device": device_name,
                "checkpoint": ckpt_abs,
                "step_dt": step_dt,
                "ramp_inc": ramp_inc,
                "ramp_dur": ramp_dur,
                "settle_s": settle_s,
                "record_s": record_s,
                "max_vx_pass_err": max_vx_pass_err,
                "max_vy_pass_err": max_vy_pass_err,
                "fixed_speed_from_start": fixed_speed_from_start,
                "reset_between_combos": reset_between_combos,
                "eval_reset_strategy": eval_reset_strategy,
                "reset_sync_steps": reset_sync_steps,
                "video_reset_lead_in_s": video_reset_lead_in_s,
                "summary": summary,
                "video_dir": str((out_dir / "videos").resolve()) if args_cli.video else None,
                **step_diag,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    _plot_combo_survival(details, out_dir)
    _plot_combo_lin_acc(details, out_dir)
    _plot_summary_metrics(summary, out_dir)
    _plot_step_alive(step_diag.get("step_phase_logs", []), out_dir)
    _plot_global_record_survival(per_env_rows, details, out_dir)
    _plot_global_tracking_acc(per_env_rows, details, out_dir)

    print(f"[DONE] Velocity tracking evaluation complete.\n[OUT] {out_dir}")
    video_recorder.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
