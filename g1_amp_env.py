# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_rotate_inverse

from .g1_amp_env_cfg import G1AmpEnvCfg
from .motions import MotionLoader


class G1AmpEnv(DirectRLEnv):
    cfg: G1AmpEnvCfg

    def __init__(self, cfg: G1AmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # action offset and scale
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits

        # load motion
        self._motion_loader = MotionLoader(
            motion_file=self.cfg.motion_file, device=self.device
        )

        # DOF and key body indexes
        key_body_names = [
            "right_rubber_hand",
            "left_rubber_hand",
            "right_ankle_roll_link",
            "left_ankle_roll_link",
        ]

        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [
            self.robot.data.body_names.index(name) for name in key_body_names
        ]
        # Used to for reset strategy
        self.motion_dof_indexes = self._motion_loader.get_dof_index(
            self.robot.data.joint_names
        )
        self.motion_ref_body_index = self._motion_loader.get_body_index(
            [self.cfg.reference_body]
        )[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(
            key_body_names
        )
        self.amp_observation_size = (
            self.cfg.num_amp_observations * self.cfg.amp_observation_space
        )
        self.amp_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.amp_observation_size,)
        )
        self.amp_observation_buffer = torch.zeros(
            (
                self.num_envs,
                self.cfg.num_amp_observations,
                self.cfg.amp_observation_space,
            ),
            device=self.device,
        )
        self.include_ang_vel_command = bool(
            getattr(self.cfg, "include_ang_vel_command", False)
        )
        self.command_dim = 2 + (1 if self.include_ang_vel_command else 0)
        self.command_lin_vel_x_range = tuple(
            getattr(self.cfg, "command_lin_vel_x_range", self.cfg.track_vel_range)
        )
        self.command_lin_vel_y_range = tuple(
            getattr(self.cfg, "command_lin_vel_y_range", (0.0, 0.0))
        )
        self.command_ang_vel_z_range = tuple(
            getattr(self.cfg, "command_ang_vel_z_range", (0.0, 0.0))
        )
        self.command_target_speed = torch.zeros(
            (self.num_envs, self.command_dim), device=self.device, dtype=torch.float32
        )
        self.command_time_left = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self._episode_track_vel_sum = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.motion_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.motion_start_times = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        # last actions buffer (always maintained)
        self.last_actions = torch.zeros(
            (self.num_envs, self.cfg.action_space), device=self.device
        )

        # actor observation history buffer (used when num_actor_observations > 1)
        self.key_body_obs_size = len(key_body_names) * 3  # 4 bodies * 3 dims = 12
        # Policy base obs size: 64 (from compute_policy_obs) vs old 71 (amp_obs - key_body)
        # Use config if available, otherwise default to 64
        self.policy_base_obs_size = getattr(self.cfg, "policy_base_obs_size", 64)
        if self.cfg.num_actor_observations > 1:
            base_obs_size = self.policy_base_obs_size  # 64 for new policy obs
            command_size = self.command_dim if self.cfg.rew_track_vel > 0.0 else 0
            # ablation 开关：历史帧包含哪些内容（默认 True 以兼容旧配置）
            _inc_act = getattr(self.cfg, "history_include_last_actions", True)
            _inc_cmd = getattr(self.cfg, "history_include_command", True)
            self.actor_obs_hist_per_frame = base_obs_size  # A 始终包含
            if _inc_act:
                self.actor_obs_hist_per_frame += self.cfg.action_space
            if _inc_cmd:
                self.actor_obs_hist_per_frame += command_size
            # buffer 只存储 (n-1) 个历史帧，当前帧单独处理
            self.actor_obs_history_buffer = torch.zeros(
                (
                    self.num_envs,
                    self.cfg.num_actor_observations - 1,
                    self.actor_obs_hist_per_frame,
                ),
                device=self.device,
            )
            self._just_reset_mask = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()
        # self.pre_actions = actions.clone()

        # update command timers
        self.command_time_left -= self.step_dt

        # resample target velocity for environments where timer has expired
        expired_envs = (self.command_time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if len(expired_envs) > 0:
            x_min, x_max = self.command_lin_vel_x_range
            y_min, y_max = self.command_lin_vel_y_range
            if x_max > x_min or y_max > y_min:
                vx = (
                    torch.rand(len(expired_envs), device=self.device) * (x_max - x_min)
                    + x_min
                )
                vy = (
                    torch.rand(len(expired_envs), device=self.device) * (y_max - y_min)
                    + y_min
                )
                self.command_target_speed[expired_envs, 0] = vx
                self.command_target_speed[expired_envs, 1] = vy
                if self.include_ang_vel_command:
                    z_min, z_max = self.command_ang_vel_z_range
                    wz = (
                        torch.rand(len(expired_envs), device=self.device)
                        * (z_max - z_min)
                        + z_min
                    )
                    self.command_target_speed[expired_envs, 2] = wz
            else:
                self.command_target_speed[expired_envs, 0] = x_min
                self.command_target_speed[expired_envs, 1] = y_min
                if self.include_ang_vel_command:
                    self.command_target_speed[expired_envs, 2] = (
                        self.command_ang_vel_z_range[0]
                    )
            self.command_time_left[expired_envs] = (
                torch.rand(len(expired_envs), device=self.device)
                * (
                    self.cfg.command_resampling_time_range[1]
                    - self.cfg.command_resampling_time_range[0]
                )
                + self.cfg.command_resampling_time_range[0]
            )

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)
        # record the applied actions for use in the next observation step
        self.last_actions = self.actions.clone()

    def _get_observations(self) -> dict:
        # Compute AMP observation (full 83-dim, for discriminator)
        amp_obs = compute_amp_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )

        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # build AMP observation (full obs including key_body_positions)
        self.amp_observation_buffer[:, 0] = amp_obs.clone()
        self.extras = {
            "amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)
        }

        # Compute Policy observation (64-dim, Sim2Real friendly)
        # gravity: (num_envs, 3) - broadcast to all envs
        gravity = torch.tensor(
            [0.0, 0.0, -9.81],
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.num_envs, 1)
        base_actor_obs = compute_policy_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            gravity,
        )

        if self.cfg.num_actor_observations > 1:
            _inc_act = getattr(self.cfg, "history_include_last_actions", True)
            _inc_cmd = getattr(self.cfg, "history_include_command", True)

            # 当前帧（始终完整： A + B + C）
            current_parts = [base_actor_obs, self.last_actions]
            if self.cfg.rew_track_vel > 0.0:
                current_parts.append(self.command_target_speed)
            current_frame = torch.cat(current_parts, dim=-1)

            # 历史帧（由 ablation 开关决定）
            hist_parts = [base_actor_obs]  # A 始终包含
            if _inc_act:
                hist_parts.append(self.last_actions)
            if _inc_cmd and self.cfg.rew_track_vel > 0.0:
                hist_parts.append(self.command_target_speed)
            hist_frame = torch.cat(hist_parts, dim=-1)

            # warm-start：刚重置的环境用当前历史帧填满所有历史槽
            if self._just_reset_mask.any():
                for i in range(self.cfg.num_actor_observations - 1):
                    self.actor_obs_history_buffer[self._just_reset_mask, i] = (
                        hist_frame[self._just_reset_mask]
                    )
                self._just_reset_mask[:] = False

            # shift 历史 buffer 并写入最新历史帧
            for i in reversed(range(self.cfg.num_actor_observations - 2)):
                self.actor_obs_history_buffer[:, i + 1] = self.actor_obs_history_buffer[
                    :, i
                ]
            self.actor_obs_history_buffer[:, 0] = hist_frame

            # 最终 obs = 当前帧 | 历史帧（展平）
            actor_obs = torch.cat(
                [current_frame, self.actor_obs_history_buffer.view(self.num_envs, -1)],
                dim=-1,
            )
        else:
            # single-frame with last_actions
            actor_obs = torch.cat([base_actor_obs, self.last_actions], dim=-1)
            if self.cfg.rew_track_vel > 0.0:
                actor_obs = torch.cat([actor_obs, self.command_target_speed], dim=-1)

        return {"policy": actor_obs}

    # def _get_rewards(self) -> torch.Tensor:
    #     return torch.ones((self.num_envs,), dtype=torch.float32, device=self.sim.device)
    def _get_rewards(self) -> torch.Tensor:

        # ================= speed tracking reward ==========================
        if self.cfg.rew_track_vel > 0.0:
            # calculate planar speed (2D: vx, vy) in local frame
            current_speed_w = self.robot.data.body_lin_vel_w[:, self.ref_body_index]
            current_quat_w = self.robot.data.body_quat_w[:, self.ref_body_index]
            current_speed_b = quat_rotate_inverse(current_quat_w, current_speed_w)
            current_speed = current_speed_b[:, :2]
            cmd = self.command_target_speed

            if self.include_ang_vel_command:
                current_ang_vel_w = self.robot.data.body_ang_vel_w[:, self.ref_body_index]
                current_ang_vel_b = quat_rotate_inverse(current_quat_w, current_ang_vel_w)
                track_target = torch.cat([current_speed, current_ang_vel_b[:, 2:3]], dim=-1)
                track_vel_error = torch.norm(track_target - cmd, dim=-1)
            else:
                track_vel_error = torch.norm(current_speed - cmd[:, :2], dim=-1)
            rew_track_vel = exp_reward_with_floor(
                torch.square(track_vel_error),
                self.cfg.rew_track_vel,
                0.5,  # sigma for velocity tracking
                floor=4.0,
            )
        else:
            rew_track_vel = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )
        self._episode_track_vel_sum += rew_track_vel

        # ================= basic reward (call the original compute_rewards function) ==========================

        basic_reward, basic_reward_log = compute_rewards(
            self.cfg.rew_termination,
            self.cfg.rew_action_l2,
            self.cfg.rew_joint_pos_limits,
            self.cfg.rew_joint_acc_l2,
            self.cfg.rew_joint_vel_l2,
            self.reset_terminated,
            self.actions,
            self.robot.data.joint_pos,
            self.robot.data.soft_joint_pos_limits,
            self.robot.data.joint_acc,
            self.robot.data.joint_vel,
        )

        # ================= total reward ==========================
        total_reward = basic_reward + rew_track_vel

        # ============== log ================================
        log_dict = {
            "total_reward": total_reward.mean().item(),
        }
        if self.cfg.rew_track_vel > 0.0:
            log_dict["rew_track_vel"] = rew_track_vel.mean().item()
            log_dict["error_track_vel"] = track_vel_error.mean().item()
            log_dict["cmd_lin_vel_x_min"] = float(self.command_lin_vel_x_range[0])
            log_dict["cmd_lin_vel_x_max"] = float(self.command_lin_vel_x_range[1])
            log_dict["cmd_lin_vel_y_min"] = float(self.command_lin_vel_y_range[0])
            log_dict["cmd_lin_vel_y_max"] = float(self.command_lin_vel_y_range[1])
            if self.include_ang_vel_command:
                log_dict["cmd_ang_vel_z_min"] = float(self.command_ang_vel_z_range[0])
                log_dict["cmd_ang_vel_z_max"] = float(self.command_ang_vel_z_range[1])

        # add basic reward log
        for key, value in basic_reward_log.items():
            if isinstance(value, torch.Tensor):
                log_dict[key] = value.mean().item()
            else:
                log_dict[key] = float(value)

        self.extras["log"] = log_dict

        # directly record to TensorBoard (if agent is available)
        if (
            hasattr(self, "_skrl_agent")
            and getattr(self, "_skrl_agent", None) is not None
        ):
            try:
                agent = getattr(self, "_skrl_agent")
                for k, v in log_dict.items():
                    agent.track_data(f"Reward / {k}", v)
            except Exception:
                pass

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = (
                self.robot.data.body_pos_w[:, self.ref_body_index, 2]
                < self.cfg.termination_height
            )
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        pre_reset_episode_length = self.episode_length_buf[env_ids].clone()
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if (
            getattr(self.cfg, "enable_track_vel_curriculum", False)
            and self.cfg.rew_track_vel > 0.0
            and len(env_ids) > 0
        ):
            time_out_mask = pre_reset_episode_length >= self.max_episode_length - 1
            timed_out_env_ids = env_ids[time_out_mask]
            if len(timed_out_env_ids) > 0:
                completed_steps = (pre_reset_episode_length[time_out_mask] + 1).to(
                    dtype=torch.float32
                )
                avg_track_rew = torch.mean(
                    self._episode_track_vel_sum[timed_out_env_ids] / completed_steps
                )
                threshold = (
                    self.cfg.rew_track_vel
                    * self.cfg.track_vel_curriculum_threshold_ratio
                )
                if avg_track_rew > threshold:
                    delta = self.cfg.track_vel_curriculum_delta
                    x_lim = getattr(
                        self.cfg,
                        "command_lin_vel_x_curriculum_limit_range",
                        self.cfg.track_vel_curriculum_limit_range,
                    )
                    y_lim = getattr(
                        self.cfg,
                        "command_lin_vel_y_curriculum_limit_range",
                        self.command_lin_vel_y_range,
                    )
                    z_lim = getattr(
                        self.cfg,
                        "command_ang_vel_z_curriculum_limit_range",
                        self.command_ang_vel_z_range,
                    )
                    x_min, x_max = self.command_lin_vel_x_range
                    y_min, y_max = self.command_lin_vel_y_range
                    self.command_lin_vel_x_range = (
                        max(x_min - delta, x_lim[0]),
                        min(x_max + delta, x_lim[1]),
                    )
                    self.command_lin_vel_y_range = (
                        max(y_min - delta, y_lim[0]),
                        min(y_max + delta, y_lim[1]),
                    )
                    if self.include_ang_vel_command:
                        z_min, z_max = self.command_ang_vel_z_range
                        self.command_ang_vel_z_range = (
                            max(z_min - delta, z_lim[0]),
                            min(z_max + delta, z_lim[1]),
                        )

        self._episode_track_vel_sum[env_ids] = 0.0

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(
                env_ids, start
            )
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # reset last_actions for the reset envs
        self.last_actions[env_ids] = 0.0
        if self.cfg.num_actor_observations > 1:
            # do NOT zero the buffer here; instead mark these envs so
            # _get_observations will warm-start the buffer with the real
            # first observation on the next step.
            self._just_reset_mask[env_ids] = True

    # reset strategies

    def _reset_strategy_default(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample random motion times (or zeros if start is True)
        num_samples = env_ids.shape[0]
        motion_ids, times = self._motion_loader.sample_times(num_samples, start=start)

        self.motion_ids[env_ids] = torch.tensor(
            motion_ids, dtype=torch.long, device=self.device
        )
        self.motion_start_times[env_ids] = torch.tensor(
            times, dtype=torch.float32, device=self.device
        )

        # sample random motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(
            num_samples=num_samples, times=times, motion_ids=motion_ids
        )

        # get root transforms (the humanoid torso)
        motion_torso_index = self._motion_loader.get_body_index(["pelvis"])[0]
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = (
            body_positions[:, motion_torso_index] + self.scene.env_origins[env_ids]
        )
        root_state[
            :, 2
        ] += 0.05  # lift the humanoid slightly to avoid collisions with the ground
        root_state[:, 3:7] = body_rotations[:, motion_torso_index]
        root_state[:, 7:10] = body_linear_velocities[:, motion_torso_index]
        root_state[:, 10:13] = body_angular_velocities[:, motion_torso_index]
        # get DOFs state
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        # update AMP observation
        amp_observations = self.collect_reference_motions(
            num_samples, times, motion_ids
        )
        self.amp_observation_buffer[env_ids] = amp_observations.view(
            num_samples, self.cfg.num_amp_observations, -1
        )

        # sample random target speed command
        x_min, x_max = self.command_lin_vel_x_range
        y_min, y_max = self.command_lin_vel_y_range
        if x_max > x_min or y_max > y_min:
            vx = torch.rand(len(env_ids), device=self.device) * (x_max - x_min) + x_min
            vy = torch.rand(len(env_ids), device=self.device) * (y_max - y_min) + y_min
            self.command_target_speed[env_ids, 0] = vx
            self.command_target_speed[env_ids, 1] = vy
            if self.include_ang_vel_command:
                z_min, z_max = self.command_ang_vel_z_range
                wz = (
                    torch.rand(len(env_ids), device=self.device) * (z_max - z_min) + z_min
                )
                self.command_target_speed[env_ids, 2] = wz
            self.command_time_left[env_ids] = (
                torch.rand(len(env_ids), device=self.device)
                * (
                    self.cfg.command_resampling_time_range[1]
                    - self.cfg.command_resampling_time_range[0]
                )
                + self.cfg.command_resampling_time_range[0]
            )
        else:
            self.command_target_speed[env_ids, 0] = x_min
            self.command_target_speed[env_ids, 1] = y_min
            if self.include_ang_vel_command:
                self.command_target_speed[env_ids, 2] = self.command_ang_vel_z_range[0]
            self.command_time_left[env_ids] = float("inf")

        return root_state, dof_pos, dof_vel

    # env methods

    def collect_reference_motions(
        self,
        num_samples: int,
        current_times: np.ndarray | None = None,
        motion_ids: np.ndarray | None = None,
    ) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        if current_times is None:
            motion_ids, current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()

        if motion_ids is not None:
            motion_ids_expanded = np.repeat(motion_ids, self.cfg.num_amp_observations)
        else:
            motion_ids_expanded = np.zeros_like(times, dtype=np.int32)

        # get motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(
            num_samples=num_samples, times=times, motion_ids=motion_ids_expanded
        )
        # compute AMP observation

        amp_observation = compute_amp_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )
        return amp_observation.view(-1, self.amp_observation_size)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def exp_reward_with_floor(
    error: torch.Tensor, weight: float, sigma: float, floor: float = 3.0
) -> torch.Tensor:
    """
    piecewise exponential reward function: large error region use linear, small error region use exponential

    Args:
        error: error value (already squared error)
        weight: reward weight
        sigma: standard deviation parameter of exponential function
        floor: threshold, unit is sigma² multiple

    Returns:
        piecewise exponential reward value
    """
    sigma_sq = sigma * sigma
    threshold = floor * sigma_sq

    # exponential part at threshold and gradient
    exp_val_at_threshold = weight * torch.exp(-floor)
    linear_slope = (
        weight / sigma_sq * torch.exp(-floor)
    )  # ensure first-order continuous

    # large error region: use linear penalty (keep negative slope)
    linear_reward = exp_val_at_threshold - linear_slope * (error - threshold)

    # small error region: use exponential reward
    exp_reward = weight * torch.exp(-error / sigma_sq)

    # choose the corresponding reward function based on the error size
    return torch.where(error > threshold, linear_reward, exp_reward)


@torch.jit.script
def quat_rotate_inverse_jit(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Quaternion inverse rotation (JIT-compatible version).

    Rotates vector v by the inverse of quaternion q.
    q: (N, 4) quaternion (w, x, y, z)
    v: (N, 3) vector
    """
    # Conjugate of q (inverse for unit quaternion)
    q_conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)  # (N, 4)
    # Use quat_apply from isaaclab (handles the rotation)
    return quat_apply(q_conj, v)


@torch.jit.script
def compute_policy_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_rotations: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    gravity: torch.Tensor,
) -> torch.Tensor:
    """Compute policy observation (64-dim for Sim2Real compatibility).

    Includes: dof_positions(29) + dof_velocities(29) + projected_gravity(3) + root_angular_velocities(3)
    Excludes: root height, tangent/normal, root linear velocity, key_body positions
    """
    projected_gravity = quat_rotate_inverse_jit(root_rotations, gravity)
    obs = torch.cat(
        [
            dof_positions,           # 29
            dof_velocities,          # 29
            projected_gravity,       # 3
            root_angular_velocities, # 3
        ],
        dim=-1,
    )
    return obs  # 64-dim


@torch.jit.script
def compute_amp_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    obs_list = [
        dof_positions,
        dof_velocities,
        root_positions[:, 2:3],  # root body height
        quaternion_to_tangent_and_normal(root_rotations),
        root_linear_velocities,
        root_angular_velocities,
        (key_body_positions - root_positions.unsqueeze(-2)).view(
            key_body_positions.shape[0], -1
        ),
    ]

    obs = torch.cat(
        obs_list,
        dim=-1,
    )
    return obs


@torch.jit.script
def compute_rewards(
    rew_scale_termination: float,
    rew_scale_action_l2: float,
    rew_scale_joint_pos_limits: float,
    rew_scale_joint_acc_l2: float,
    rew_scale_joint_vel_l2: float,
    reset_terminated: torch.Tensor,
    actions: torch.Tensor,
    joint_pos: torch.Tensor,
    soft_joint_pos_limits: torch.Tensor,
    joint_acc: torch.Tensor,
    joint_vel: torch.Tensor,
):
    rew_termination = rew_scale_termination * reset_terminated.float()
    rew_action_l2 = rew_scale_action_l2 * torch.sum(torch.square(actions), dim=1)

    out_of_limits = -(joint_pos - soft_joint_pos_limits[:, :, 0]).clip(max=0.0)
    out_of_limits += (joint_pos - soft_joint_pos_limits[:, :, 1]).clip(min=0.0)
    rew_joint_pos_limits = rew_scale_joint_pos_limits * torch.sum(out_of_limits, dim=1)

    rew_joint_acc_l2 = rew_scale_joint_acc_l2 * torch.sum(
        torch.square(joint_acc), dim=1
    )
    rew_joint_vel_l2 = rew_scale_joint_vel_l2 * torch.sum(
        torch.square(joint_vel), dim=1
    )
    total_reward = (
        rew_termination
        + rew_action_l2
        + rew_joint_pos_limits
        + rew_joint_acc_l2
        + rew_joint_vel_l2
    )

    log = {
        "pub_termination": (rew_termination).mean(),
        "pub_action_l2": (rew_action_l2).mean(),
        "pub_joint_pos_limits": (rew_joint_pos_limits).mean(),
        "pub_joint_acc_l2": (rew_joint_acc_l2).mean(),
        "pub_joint_vel_l2": (rew_joint_vel_l2).mean(),
    }
    return total_reward, log
