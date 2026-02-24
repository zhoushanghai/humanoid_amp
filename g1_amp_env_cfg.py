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
    observation_space = (
        71  # originally 83, now 71 after moving key_body_positions to amp_obs
    )
    action_space = 29
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 71 + 3 * 10
    num_actor_observations = 1  # number of stacked frames for actor input

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

    # spaces
    # per-frame actor obs: 71 (base) + 29 (last_actions) + 2 (cmd) = 102
    # single frame (no history stacking)
    observation_space = 102
    action_space = 29
    state_space = 0
    num_amp_observations = 10
    amp_observation_space = 83
    num_actor_observations = 1  # single frame, no history stacking

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
    episode_length_s = 10.0
    motion_file = "/home/hz/g1/humanoid_amp/motions/motion_config.yaml"
    reset_strategy = "random"
    track_vel_range = (0.5, 1.5)

    # only velocity tracking reward, disable all penalties
    rew_termination = 0.0
    rew_action_l2 = 0.0
    rew_joint_pos_limits = 0.0
    rew_joint_acc_l2 = 0.0
    rew_joint_vel_l2 = 0.0
    rew_track_vel = 1.0

    # Policy 基础观测维度（Sim2Real 版本：64 维，不再包含 key_body）
    # 包含: dof_positions(29) + dof_velocities(29) + projected_gravity(3) + root_angular_vel(3)
    policy_base_obs_size: int = 64

    # ✅ 只需修改这一个値来控制历史帧数，observation_space 会自动换算
    num_actor_observations = 5

    # ——— Ablation 开关：控制「历史帧」（非当前帧）包含哪些内容 ———
    # 当前帧始终包含完整内容： base_obs(A) + last_actions(B) + command(C)
    # 实验②：两个均为 False（历史帧只含 A）
    # 实验③： history_include_last_actions=True（历史帧含 A+B）
    # 实验④：两个均为 True（历史帧完整 A+B+C）
    history_include_last_actions: bool = True
    history_include_command: bool = True

    def __post_init__(self):
        # 使用新的 policy_base_obs_size (64) 而不是旧的 amp_obs - key_body (71)
        base_obs_size = self.policy_base_obs_size  # 64
        command_size = 2 if self.rew_track_vel > 0.0 else 0

        # 当前帧始终完整： A + B + C
        current_frame_size = (
            base_obs_size + self.action_space + command_size
        )  # 64 + 29 + 2 = 95

        if self.num_actor_observations <= 1:
            self.observation_space = current_frame_size
        else:
            # 历史帧大小由开关决定
            hist_frame_size = base_obs_size  # A 始终包含
            if self.history_include_last_actions:
                hist_frame_size += self.action_space  # +B
            if self.history_include_command:
                hist_frame_size += command_size  # +C
            # 总维度 = 当前帧 + (n-1) × 历史帧
            # 当前帧: 95, 历史帧: 95, num_actor_observations=5
            # Total = 95 + 4 * 95 = 475
            self.observation_space = (
                current_frame_size + (self.num_actor_observations - 1) * hist_frame_size
            )
