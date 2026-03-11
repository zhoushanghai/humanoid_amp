"""
Purpose: Define the standalone configuration for the G1 AMP proprioception exploration task.
Main contents: single-task training defaults, observation-history sizing, heterogeneous scene settings,
and exploration-specific reward and scene-bank knobs.
"""

from __future__ import annotations

import os

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .g1_amp_env_cfg import G1AmpEnvCfg
from .g1_amp_poprioception_constants import (
    CONTACT_COUNT_PER_STEP_CAP,
    CONTACT_COUNT_REWARD_SCALE,
    CONTACT_FORCE_THRESHOLD_N,
    GEOMETRY_REWARD_SCALE,
    NUM_OBSTACLE_SLOTS,
    OBSTACLE_SURFACE_SPACING_M,
    ROOM_SIZE_M,
    ROOM_TERMINATION_MARGIN_M,
    ROOM_WALL_HEIGHT_M,
    ROOM_WALL_MARGIN_M,
    ROOM_WALL_THICKNESS_M,
    ROBOT_SPAWN_CLEARANCE_M,
    SURFACE_CORNER_BAND_M,
    SURFACE_EDGE_BAND_M,
    SURFACE_GRID_CELL_SIZE_M,
    SURFACE_GRID_REWARD_SCALE,
)


@configclass
class G1AmpPoprioceptionEnvCfg(G1AmpEnvCfg):
    """Configuration for the remaining single-task G1 AMP proprioception environment."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=5.0,
        replicate_physics=False,
    )

    motion_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "motions",
        "motion_poprioception.yaml",
    )

    episode_length_s = 20.0
    reset_strategy = "default"
    env_log_interval_steps = 50

    # proprioception keeps AMP/style prior but disables velocity-tracking commands.
    rew_track_vel = 0.0
    rew_track_ang_vel_z = 0.0
    rew_termination = 0.0
    rew_action_l2 = 0.0
    rew_joint_pos_limits = 0.0
    rew_joint_acc_l2 = 0.0
    rew_joint_vel_l2 = 0.0
    include_ang_vel_command = False
    enable_track_vel_curriculum = False
    command_lin_vel_x_range = (0.0, 0.0)
    command_lin_vel_y_range = (0.0, 0.0)
    command_ang_vel_z_range = (0.0, 0.0)

    num_amp_observations = 32
    num_actor_observations = 5
    history_include_last_actions = True
    history_include_command = False

    num_obstacle_slots = NUM_OBSTACLE_SLOTS
    room_size_m = ROOM_SIZE_M
    room_wall_height_m = ROOM_WALL_HEIGHT_M
    room_wall_thickness_m = ROOM_WALL_THICKNESS_M
    room_wall_margin_m = ROOM_WALL_MARGIN_M
    robot_spawn_clearance_m = ROBOT_SPAWN_CLEARANCE_M
    obstacle_surface_spacing_m = OBSTACLE_SURFACE_SPACING_M
    room_termination_margin_m = ROOM_TERMINATION_MARGIN_M

    contact_force_threshold = CONTACT_FORCE_THRESHOLD_N
    contact_count_mode = "body_object_pairs"
    contact_count_per_step_cap = CONTACT_COUNT_PER_STEP_CAP

    surface_grid_cell_size_m = SURFACE_GRID_CELL_SIZE_M
    surface_edge_band_m = SURFACE_EDGE_BAND_M
    surface_corner_band_m = SURFACE_CORNER_BAND_M

    rew_contact_count = CONTACT_COUNT_REWARD_SCALE
    rew_surface_grid = SURFACE_GRID_REWARD_SCALE
    rew_geometry = GEOMETRY_REWARD_SCALE

    scene_bank_size = 16
    scene_bank_total_layout_budget = 4096 * scene_bank_size

    def __post_init__(self) -> None:
        """Derive actor observation size from the configured history contents."""
        command_size = 0
        if self.rew_track_vel > 0.0 or self.rew_track_ang_vel_z > 0.0:
            command_size = 2 + (1 if self.include_ang_vel_command else 0)

        current_frame_size = self.policy_base_obs_size + self.action_space + command_size
        if self.num_actor_observations <= 1:
            self.observation_space = current_frame_size
            return

        hist_frame_size = self.policy_base_obs_size
        if self.history_include_last_actions:
            hist_frame_size += self.action_space
        if self.history_include_command:
            hist_frame_size += command_size
        self.observation_space = current_frame_size + (
            self.num_actor_observations - 1
        ) * hist_frame_size
