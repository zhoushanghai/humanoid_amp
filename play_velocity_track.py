# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl with velocity tracking and logging.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl with velocity tracking.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
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
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
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
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--target_vel", type=float, default=1.0, help="Target velocity to track (m/s).")
parser.add_argument("--duration", type=int, default=1000, help="Duration of the playback in steps.")

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

import random
import time
from datetime import datetime
import gymnasium as gym
import skrl
import torch
import numpy as np
import matplotlib.pyplot as plt
from packaging import version
import humanoid_amp.agents as agents

# Explicitly register the environment to ensure we use the local version with the fix
from humanoid_amp.g1_amp_env import G1AmpEnv
from humanoid_amp.g1_amp_env_cfg import G1AmpSpeedEnvCfg

gym.register(
    id="Isaac-G1-AMP-Speed-Direct-v0",
    entry_point="humanoid_amp.g1_amp_env:G1AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "humanoid_amp.g1_amp_env_cfg:G1AmpSpeedEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_g1_custom_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_g1_custom_amp_cfg.yaml",
    },
)

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
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent and track velocity."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder_name = f"{timestamp}_{args_cli.task}"
    output_dir = os.path.join(log_dir, "play_results", output_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Saving results to: {output_dir}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

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
            "video_folder": output_dir,
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
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    
    # Data logging
    target_speeds = []
    actual_speeds = []
    steps = []

    print(f"[INFO] Starting playback with target velocity: {args_cli.target_vel} m/s")

    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # OVERRIDE TARGET VELOCITY
        # We need to access the underlying environment to set the command
        # SkrlVecEnvWrapper -> env -> DirectRLEnv (G1AmpEnv)
        # However, due to wrappers, we might need to unwrap or access correctly.
        # Direct access via `env.unwrapped` should give us the G1AmpEnv instance.
        
        # Check if we can access the command_target_speed
        if hasattr(env.unwrapped, "command_target_speed"):
             env.unwrapped.command_target_speed[:] = args_cli.target_vel
        
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            
            # env stepping
            obs, _, _, _, _ = env.step(actions)
            
            # Record data
            # Assuming env.unwrapped is G1AmpEnv
            if hasattr(env.unwrapped, "robot") and hasattr(env.unwrapped, "ref_body_index"):
                # Get current linear velocity of the reference body (likely torso/base)
                # Shape: [num_envs, num_bodies, 3] -> [:, ref_body_index, :2] for planar velocity
                lin_vel = env.unwrapped.robot.data.body_lin_vel_w[:, env.unwrapped.ref_body_index, :2]
                speed = torch.norm(lin_vel, dim=-1).mean().item()
                actual_speeds.append(speed)
                target_speeds.append(args_cli.target_vel)
                steps.append(timestep)

        if args_cli.video:
             if timestep == args_cli.video_length:
                break
        
        timestep += 1
        if timestep >= args_cli.duration:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()

    # Plotting
    print("[INFO] Generating velocity tracking plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(steps, target_speeds, label='Target Speed (m/s)', linestyle='--')
    plt.plot(steps, actual_speeds, label='Actual Speed (m/s)')
    plt.xlabel('Step')
    plt.ylabel('Speed (m/s)')
    plt.title(f'Velocity Tracking Performance (Target: {args_cli.target_vel} m/s)')
    plt.legend()
    plt.grid(True)
    
    output_plot_path = os.path.join(output_dir, "velocity_tracking.png")
    plt.savefig(output_plot_path)
    print(f"[INFO] Plot saved to: {output_plot_path}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
