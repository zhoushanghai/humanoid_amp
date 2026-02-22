# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING
from .g1_cfg import G1_CFG


from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class G1AmpEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # reward
    rew_termination = -0
    rew_action_l2 = -0.00
    rew_joint_pos_limits = -0
    rew_joint_acc_l2 = -0.00
    rew_joint_vel_l2 = -0.00
    rew_track_vel = 0.0

    # env
    episode_length_s = 10.0
    decimation = 2
    track_vel_range = (0.0, 0.0)

    # spaces
    observation_space = 71 + 3 * (8 + 5) - 6 + 1  # add progress feature
    action_space = 29
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 71 + 3 * 10

    early_termination = True
    termination_height = 0.5

    motion_file: str = MISSING
    reference_body = "pelvis"
    reset_strategy = "random"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

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
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    # robot
    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class G1AmpEnvCfg_CUSTOM(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # reward
    rew_termination = -1.0
    rew_action_l2 = -0.1
    rew_joint_pos_limits = -10
    rew_joint_acc_l2 = -1.0e-06
    rew_joint_vel_l2 = -0.001
    rew_track_vel = 1.0

    # env
    episode_length_s = 10.0
    decimation = 1
    track_vel_range = (-1.0, 1.0)  # (-1 to 1 for generic random tracking velocities)
    command_resampling_time_range = (
        4.0,
        7.0,
    )  # how often to resample target velocities in seconds

    # spaces (Observation now includes history and different features, AMP observation does not)
    # dof_pos(29) + dof_vel(29) + projected_gravity(3) + root_rot_tangent_and_normal(6) + root_lin_vel(3) + root_angular_vel(3) + key_body_pos(12) + last_action(29) = 114
    # If velocity tracking is enabled, adds +2 per frame
    # Single frame observation: 114 (or 116 if track_vel_range is set)
    single_obs_dim = 114 + 2
    history_steps = 5
    observation_space = single_obs_dim * history_steps
    action_space = 29
    state_space = 0
    num_amp_observations = 10
    amp_observation_space = 83

    early_termination = True
    termination_height = 0.5

    motion_file: str = MISSING
    reference_body = "pelvis"
    reset_strategy = "random-start"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

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
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    # robot
    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class G1AmpWalkEnvCfg(G1AmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "G1_walk.npz")


@configclass
class G1AmpDanceEnvCfg(G1AmpEnvCfg_CUSTOM):
    motion_file = os.path.join(MOTIONS_DIR, "G1_dance.npz")


@configclass
class G1AmpCustomEnvCfg(G1AmpEnvCfg_CUSTOM):
    episode_length_s = 5.0
    motion_file = os.path.join(MOTIONS_DIR, "custom_motion.npz")


@configclass
class G1AmpDeployEnvCfg(G1AmpEnvCfg_CUSTOM):
    episode_length_s = 20.0
    motion_file = "/home/hz/g1/humanoid_amp/motions/motion_config.yaml"
    reset_strategy = "random"
    track_vel_range = (1.0, 1.0)

    # only velocity tracking reward, disable all penalties
    rew_termination = 0.0
    rew_action_l2 = 0.0
    rew_joint_pos_limits = 0.0
    rew_joint_acc_l2 = 0.0
    rew_joint_vel_l2 = 0.0
    rew_track_vel = 1.0
