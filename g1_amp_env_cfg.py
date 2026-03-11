# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Purpose: Define the shared configuration base for the remaining G1 AMP direct RL task.
Main contents: shared reward, command, observation-history, simulation, and robot defaults used by the
proprioception environment.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .g1_cfg import G1_CFG


@configclass
class G1AmpEnvCfg(DirectRLEnvCfg):
    """Shared configuration base for the G1 AMP direct RL environment."""

    # reward
    rew_termination = -1.0
    rew_action_l2 = -0.1
    rew_joint_pos_limits = -10.0
    rew_joint_acc_l2 = -1.0e-06
    rew_joint_vel_l2 = -0.001
    rew_track_vel = 1.0
    rew_track_ang_vel_z = 0.0

    # env
    episode_length_s = 10.0
    decimation = 1
    env_log_interval_steps = 1
    track_vel_range = (-1.0, 1.0)
    command_lin_vel_x_range = (-1.0, 1.0)
    command_lin_vel_y_range = (0.0, 0.0)
    command_ang_vel_z_range = (0.0, 0.0)
    include_ang_vel_command = False
    command_resampling_time_range = (4.0, 7.0)
    enable_track_vel_curriculum = False
    track_vel_curriculum_delta = 0.1
    track_vel_curriculum_threshold_ratio = 0.8
    track_ang_vel_curriculum_delta = 0.1
    track_ang_vel_curriculum_threshold_ratio = 0.8
    track_vel_curriculum_limit_range = (-1.0, 1.0)
    command_lin_vel_x_curriculum_limit_range = (-1.0, 1.0)
    command_lin_vel_y_curriculum_limit_range = (0.0, 0.0)
    command_ang_vel_z_curriculum_limit_range = (0.0, 0.0)

    # spaces
    observation_space = 102
    action_space = 29
    state_space = 0
    num_amp_observations = 32
    amp_observation_space = 83
    num_actor_observations = 1
    policy_base_obs_size = 64
    history_include_last_actions = True
    history_include_command = True

    early_termination = True
    termination_height = 0.5

    motion_file: str = MISSING
    reference_body = "pelvis"
    reset_strategy = "random-start"

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # robot
    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
