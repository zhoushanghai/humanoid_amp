# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl with velocity logging and plotting.

Usage:
    python play_with_velocity.py --task humanoid_amp:G1-AMP-Walk-v0 --checkpoint path/to/checkpoint.pt --num_envs 1 --max_steps 500

This will:
1. Run the simulation for the specified number of steps
2. Save velocity data to a CSV file
3. Generate velocity plots
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Play a checkpoint of an RL agent from skrl with velocity logging."
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Path to model checkpoint."
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)
# New arguments for velocity logging
parser.add_argument(
    "--max_steps",
    type=int,
    default=500,
    help="Maximum number of steps to run (for velocity logging).",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="velocity_logs",
    help="Directory to save velocity data and plots.",
)
parser.add_argument(
    "--env_id",
    type=int,
    default=0,
    help="Environment ID to log velocity for (when num_envs > 1).",
)
parser.add_argument(
    "--target_velocity",
    type=float,
    default=None,
    help="Target velocity to set for the environment (m/s).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import time
from datetime import datetime

import gymnasium as gym
import skrl
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import humanoid_amp  # noqa: F401  # Register G1 AMP environments
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = (
        "skrl_cfg_entry_point"
        if algorithm in ["ppo"]
        else f"skrl_{algorithm}_cfg_entry_point"
    )
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


class VelocityLogger:
    """Class to log and plot robot velocities."""

    def __init__(
        self, save_dir: str, env_id: int = 0, target_velocity: float | None = None
    ):
        self.save_dir = save_dir
        self.env_id = env_id
        self.target_velocity = target_velocity
        os.makedirs(save_dir, exist_ok=True)

        # Data storage
        self.timestamps = []
        self.linear_velocities_x = []
        self.linear_velocities_y = []
        self.linear_velocities_z = []
        self.angular_velocities_x = []
        self.angular_velocities_y = []
        self.angular_velocities_z = []
        self.speeds = []  # Total speed magnitude

    def log(
        self, step: int, dt: float, linear_vel: torch.Tensor, angular_vel: torch.Tensor
    ):
        """Log velocity data for one timestep."""
        self.timestamps.append(step * dt)

        # Extract velocities for the specified environment
        lin_vel = linear_vel[self.env_id].cpu().numpy()
        ang_vel = angular_vel[self.env_id].cpu().numpy()

        self.linear_velocities_x.append(lin_vel[0])
        self.linear_velocities_y.append(lin_vel[1])
        self.linear_velocities_z.append(lin_vel[2])
        self.angular_velocities_x.append(ang_vel[0])
        self.angular_velocities_y.append(ang_vel[1])
        self.angular_velocities_z.append(ang_vel[2])

        # Calculate total horizontal speed
        speed = np.sqrt(lin_vel[0] ** 2 + lin_vel[1] ** 2)
        self.speeds.append(speed)

    def save_csv(self):
        """Save velocity data to CSV file."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.save_dir, f"velocity_data_{timestamp_str}.csv")

        df = pd.DataFrame(
            {
                "time_s": self.timestamps,
                "linear_vel_x": self.linear_velocities_x,
                "linear_vel_y": self.linear_velocities_y,
                "linear_vel_z": self.linear_velocities_z,
                "angular_vel_x": self.angular_velocities_x,
                "angular_vel_y": self.angular_velocities_y,
                "angular_vel_z": self.angular_velocities_z,
                "horizontal_speed": self.speeds,
            }
        )
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Velocity data saved to: {csv_path}")
        return csv_path

    def plot(self):
        """Generate and save velocity plots."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle("Robot Velocity During Walking", fontsize=14, fontweight="bold")

        # Plot 1: Linear velocities
        ax1 = axes[0]
        ax1.plot(
            self.timestamps,
            self.linear_velocities_x,
            "r-",
            label="Vx (forward)",
            linewidth=1.5,
        )
        ax1.plot(
            self.timestamps,
            self.linear_velocities_y,
            "g-",
            label="Vy (lateral)",
            linewidth=1.5,
        )
        ax1.plot(
            self.timestamps,
            self.linear_velocities_z,
            "b-",
            label="Vz (vertical)",
            linewidth=1.5,
        )
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Linear Velocity (m/s)")
        ax1.set_title("Linear Velocities")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

        # Plot 2: Angular velocities
        ax2 = axes[1]
        ax2.plot(
            self.timestamps,
            self.angular_velocities_x,
            "r-",
            label="ωx (roll)",
            linewidth=1.5,
        )
        ax2.plot(
            self.timestamps,
            self.angular_velocities_y,
            "g-",
            label="ωy (pitch)",
            linewidth=1.5,
        )
        ax2.plot(
            self.timestamps,
            self.angular_velocities_z,
            "b-",
            label="ωz (yaw)",
            linewidth=1.5,
        )
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Angular Velocity (rad/s)")
        ax2.set_title("Angular Velocities")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

        # Plot 3: Horizontal speed
        ax3 = axes[2]
        ax3.plot(
            self.timestamps,
            self.speeds,
            "purple",
            linewidth=2,
            label="Horizontal Speed",
        )
        ax3.fill_between(self.timestamps, self.speeds, alpha=0.3, color="purple")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Speed (m/s)")
        ax3.set_title("Horizontal Speed (√(Vx² + Vy²))")
        ax3.grid(True, alpha=0.3)

        # Add statistics
        avg_speed = np.mean(self.speeds)
        max_speed = np.max(self.speeds)
        ax3.axhline(
            y=avg_speed,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label=f"Average: {avg_speed:.2f} m/s",
        )
        if self.target_velocity is not None:
            ax3.axhline(
                y=self.target_velocity,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Target: {self.target_velocity:.2f} m/s",
            )
        ax3.legend(loc="upper right")

        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(self.save_dir, f"velocity_plot_{timestamp_str}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Velocity plot saved to: {plot_path}")

        # Also save individual component plots
        self._save_component_plots(timestamp_str)

        plt.close(fig)
        return plot_path

    def _save_component_plots(self, timestamp_str: str):
        """Save individual component plots for detailed analysis."""
        # Forward velocity plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.timestamps, self.linear_velocities_x, "r-", linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Forward Velocity Vx (m/s)")
        ax.set_title("Forward Velocity Over Time")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        avg_vx = np.mean(self.linear_velocities_x)
        ax.axhline(
            y=avg_vx, color="orange", linestyle="--", label=f"Average: {avg_vx:.2f} m/s"
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, f"forward_velocity_{timestamp_str}.png"),
            dpi=150,
        )
        plt.close(fig)

    def print_statistics(self):
        """Print velocity statistics."""
        print("\n" + "=" * 50)
        print("VELOCITY STATISTICS")
        print("=" * 50)
        print(f"Total simulation time: {self.timestamps[-1]:.2f} s")
        print(f"Number of samples: {len(self.timestamps)}")
        print("-" * 50)
        print("Linear Velocities (m/s):")
        print(
            f"  Vx (forward):  mean={np.mean(self.linear_velocities_x):.3f}, "
            f"std={np.std(self.linear_velocities_x):.3f}, "
            f"min={np.min(self.linear_velocities_x):.3f}, "
            f"max={np.max(self.linear_velocities_x):.3f}"
        )
        print(
            f"  Vy (lateral):  mean={np.mean(self.linear_velocities_y):.3f}, "
            f"std={np.std(self.linear_velocities_y):.3f}, "
            f"min={np.min(self.linear_velocities_y):.3f}, "
            f"max={np.max(self.linear_velocities_y):.3f}"
        )
        print(
            f"  Vz (vertical): mean={np.mean(self.linear_velocities_z):.3f}, "
            f"std={np.std(self.linear_velocities_z):.3f}, "
            f"min={np.min(self.linear_velocities_z):.3f}, "
            f"max={np.max(self.linear_velocities_z):.3f}"
        )
        print("-" * 50)
        print("Horizontal Speed (m/s):")
        print(
            f"  mean={np.mean(self.speeds):.3f}, "
            f"std={np.std(self.speeds):.3f}, "
            f"min={np.min(self.speeds):.3f}, "
            f"max={np.max(self.speeds):.3f}"
        )
        print("=" * 50 + "\n")


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    experiment_cfg: dict,
):
    """Play with skrl agent and log velocities."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    experiment_cfg["seed"] = (
        args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    )
    env_cfg.seed = experiment_cfg["seed"]

    # Set target velocity if specified
    if args_cli.target_velocity is not None:
        print(f"[INFO] Setting target velocity to {args_cli.target_velocity} m/s")
        # Initialize min and max to the same value to force a specific velocity
        env_cfg.min_velocity = args_cli.target_velocity
        env_cfg.max_velocity = args_cli.target_velocity
        env_cfg.target_velocity = args_cli.target_velocity

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join(
        "logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"]
    )
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print(
                "[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task."
            )
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path,
            run_dir=f".*_{algorithm}_{args_cli.ml_framework}",
            other_dirs=["checkpoints"],
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # configure and instantiate the skrl runner
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    # Get the unwrapped environment to access robot data
    unwrapped_env = env.unwrapped
    while hasattr(unwrapped_env, "env"):
        unwrapped_env = unwrapped_env.env
    if hasattr(unwrapped_env, "unwrapped"):
        unwrapped_env = unwrapped_env.unwrapped

    # Initialize velocity logger
    save_dir = (
        os.path.join(log_dir, args_cli.save_dir) if log_dir else args_cli.save_dir
    )
    velocity_logger = VelocityLogger(
        save_dir=save_dir,
        env_id=args_cli.env_id,
        target_velocity=args_cli.target_velocity,
    )

    print(f"[INFO] Will log velocity data for environment {args_cli.env_id}")
    print(f"[INFO] Will run for {args_cli.max_steps} steps")
    print(f"[INFO] Velocity data will be saved to: {save_dir}")

    # reset environment
    obs, _ = env.reset()
    timestep = 0

    # simulate environment
    print("[INFO] Starting simulation with velocity logging...")
    while simulation_app.is_running() and timestep < args_cli.max_steps:
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            if hasattr(env, "possible_agents"):
                actions = {
                    a: outputs[-1][a].get("mean_actions", outputs[0][a])
                    for a in env.possible_agents
                }
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)

            # Log velocity data
            try:
                # Access robot velocity data
                robot = unwrapped_env.robot
                ref_body_index = unwrapped_env.ref_body_index

                linear_vel = robot.data.body_lin_vel_w[:, ref_body_index]
                angular_vel = robot.data.body_ang_vel_w[:, ref_body_index]

                velocity_logger.log(timestep, dt, linear_vel, angular_vel)
            except Exception as e:
                if timestep == 0:
                    print(f"[WARNING] Could not access robot velocity data: {e}")
                    print("[INFO] Trying alternative method...")
                    # Try alternative access method
                    try:
                        robot = unwrapped_env.scene.articulations["robot"]
                        ref_body_index = 0  # pelvis is usually index 0
                        linear_vel = robot.data.body_lin_vel_w[:, ref_body_index]
                        angular_vel = robot.data.body_ang_vel_w[:, ref_body_index]
                        velocity_logger.log(timestep, dt, linear_vel, angular_vel)
                        print("[INFO] Alternative method successful!")
                    except Exception as e2:
                        print(f"[ERROR] Alternative method also failed: {e2}")
                        break

        timestep += 1

        # Print progress
        if timestep % 100 == 0:
            print(f"[INFO] Step {timestep}/{args_cli.max_steps}")

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    print(f"[INFO] Simulation completed. Total steps: {timestep}")

    # Save and plot velocity data
    if len(velocity_logger.timestamps) > 0:
        velocity_logger.print_statistics()
        csv_path = velocity_logger.save_csv()
        plot_path = velocity_logger.plot()
        print(f"\n[INFO] Velocity logging complete!")
        print(f"[INFO] CSV data: {csv_path}")
        print(f"[INFO] Plot: {plot_path}")
    else:
        print("[WARNING] No velocity data was logged!")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
