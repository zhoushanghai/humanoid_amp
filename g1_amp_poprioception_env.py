"""
Purpose: Implement the standalone G1 AMP proprioception exploration environment.
Main contents: scene wiring, startup obstacle library initialization, reset-time obstacle layout sampling, and exploration rewards.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .g1_amp_env import G1AmpEnv
from .g1_amp_poprioception_constants import (
    ENV_REGEX_NS,
    GROUND_PRIM_PATH,
    HIDDEN_OBSTACLE_Z_M,
    LIGHT_PRIM_PATH,
    MAX_SURFACE_CELLS_PER_CANDIDATE,
    OBSTACLE_KIND_TO_ID,
    ROOM_HALF_EXTENT_M,
    VALID_CONTACT_BODIES,
)
from .g1_amp_poprioception_env_cfg import G1AmpPoprioceptionEnvCfg
from .g1_amp_poprioception_rewards import (
    compute_candidate_cell_counts,
    compute_contact_pair_count_reward,
    compute_surface_grid_rewards,
)
from .g1_amp_poprioception_scene import (
    apply_startup_obstacle_scales,
    build_obstacle_candidate_cfgs,
    build_obstacle_candidate_specs,
    build_room_asset_cfgs,
    build_upper_body_contact_sensor_cfgs,
    create_scene_parent_prims,
    sample_episode_obstacle_layout,
    sample_startup_obstacle_shape_params,
    yaw_to_quat_wxyz,
)


class G1AmpPoprioceptionEnv(G1AmpEnv):
    """AMP exploration environment with static rooms and upper-limb tactile rewards."""

    cfg: G1AmpPoprioceptionEnvCfg

    def __init__(self, cfg: G1AmpPoprioceptionEnvCfg, render_mode: str | None = None, **kwargs):
        self._candidate_specs = build_obstacle_candidate_specs()
        super().__init__(cfg, render_mode, **kwargs)

        self.contact_sensor_names = tuple(f"{body_name}_contact" for body_name in VALID_CONTACT_BODIES)
        self.contact_sensors = [self.scene.sensors[name] for name in self.contact_sensor_names]
        self.rewarded_body_indices = [
            self.robot.data.body_names.index(body_name) for body_name in VALID_CONTACT_BODIES
        ]
        self.candidate_names = tuple(spec.asset_name for spec in self._candidate_specs)
        self.candidate_kind_ids = torch.tensor(
            [OBSTACLE_KIND_TO_ID[spec.kind] for spec in self._candidate_specs],
            dtype=torch.long,
            device=self.device,
        )
        self.num_candidates = len(self._candidate_specs)
        self.candidate_shape_params = self._startup_candidate_shape_params.to(device=self.device)
        self.candidate_cell_counts = compute_candidate_cell_counts(
            self.candidate_kind_ids,
            self.candidate_shape_params,
            self.cfg.surface_grid_cell_size_m,
        )
        self.surface_grid_visited = torch.zeros(
            (self.num_envs, self.num_candidates, MAX_SURFACE_CELLS_PER_CANDIDATE),
            dtype=torch.bool,
            device=self.device,
        )
        self.candidate_active_mask = torch.zeros(
            (self.num_envs, self.num_candidates),
            dtype=torch.bool,
            device=self.device,
        )
        self.candidate_positions_w = torch.zeros(
            (self.num_envs, self.num_candidates, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.candidate_positions_w[:, :, 2] = HIDDEN_OBSTACLE_Z_M
        self.candidate_yaws = torch.zeros(
            (self.num_envs, self.num_candidates),
            dtype=torch.float32,
            device=self.device,
        )

    def _setup_scene(self):
        spawn_ground_plane(
            prim_path=GROUND_PRIM_PATH,
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )

        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        create_scene_parent_prims(self.scene.env_prim_paths, self._candidate_specs)

        self.room_assets = {}
        for asset_name, asset_cfg in build_room_asset_cfgs().items():
            self.room_assets[asset_name] = asset_cfg.class_type(asset_cfg)
            self.scene.rigid_objects[asset_name] = self.room_assets[asset_name]

        self.obstacle_assets = {}
        obstacle_cfgs = build_obstacle_candidate_cfgs(self._candidate_specs)
        for asset_name, asset_cfg in obstacle_cfgs.items():
            self.obstacle_assets[asset_name] = asset_cfg.class_type(asset_cfg)
            self.scene.rigid_objects[asset_name] = self.obstacle_assets[asset_name]

        sensor_cfgs = build_upper_body_contact_sensor_cfgs(self._candidate_specs)
        for sensor_name, sensor_cfg in sensor_cfgs.items():
            self.scene.sensors[sensor_name] = ContactSensor(sensor_cfg)

        self._startup_candidate_shape_params, startup_scales = sample_startup_obstacle_shape_params(
            self.scene.num_envs,
            self._candidate_specs,
            device=self.device,
        )
        apply_startup_obstacle_scales(self.obstacle_assets, self._candidate_specs, startup_scales)

        self.scene.filter_collisions(global_prim_paths=[GROUND_PRIM_PATH])

        light_cfg = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.78, 0.80, 0.84))
        light_cfg.func(LIGHT_PRIM_PATH, light_cfg)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self.surface_grid_visited[env_ids] = False
        self._reset_obstacle_layout(env_ids)

    def _reset_obstacle_layout(self, env_ids: torch.Tensor) -> None:
        active_mask, positions, yaws = sample_episode_obstacle_layout(
            env_origins=self.scene.env_origins,
            candidate_shape_params=self.candidate_shape_params,
            env_ids=env_ids,
            device=self.device,
        )
        orientations = yaw_to_quat_wxyz(yaws.view(-1)).view(len(env_ids), self.num_candidates, 4)

        self.candidate_active_mask[env_ids] = active_mask
        self.candidate_positions_w[env_ids] = positions
        self.candidate_yaws[env_ids] = yaws

        for candidate_index, candidate_name in enumerate(self.candidate_names):
            root_state = torch.zeros((len(env_ids), 13), dtype=torch.float32, device=self.device)
            root_state[:, :3] = positions[:, candidate_index]
            root_state[:, 3:7] = orientations[:, candidate_index]
            self.obstacle_assets[candidate_name].write_root_state_to_sim(root_state, env_ids=env_ids)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        died, time_out = super()._get_dones()
        root_positions = self.robot.data.body_pos_w[:, self.ref_body_index]
        env_origins = self.scene.env_origins
        local_root_xy = root_positions[:, :2] - env_origins[:, :2]
        boundary_limit = ROOM_HALF_EXTENT_M - self.cfg.room_termination_margin_m
        boundary_violation = (local_root_xy.abs() > boundary_limit).any(dim=-1)
        return died | boundary_violation, time_out

    def _get_rewards(self) -> torch.Tensor:
        total_reward = super()._get_rewards()

        contact_force_list = []
        for sensor in self.contact_sensors:
            force_matrix = sensor.data.force_matrix_w
            if force_matrix is None:
                raise RuntimeError("Upper-body contact sensor was created without filter_prim_paths_expr.")
            contact_force_list.append(force_matrix[:, 0])
        stacked_contact_forces = torch.stack(contact_force_list, dim=1)
        stacked_contact_forces = stacked_contact_forces * self.candidate_active_mask.unsqueeze(1).unsqueeze(-1)

        contact_reward_count, raw_pair_count, contact_mask = compute_contact_pair_count_reward(
            stacked_contact_forces,
            threshold=self.cfg.contact_force_threshold,
            per_step_cap=self.cfg.contact_count_per_step_cap,
        )

        body_positions_w = self.robot.data.body_pos_w[:, self.rewarded_body_indices]
        new_surface_cells, geometry_weight_sum = compute_surface_grid_rewards(
            visited_surface_cells=self.surface_grid_visited,
            contact_mask=contact_mask,
            body_positions_w=body_positions_w,
            candidate_positions_w=self.candidate_positions_w,
            candidate_yaws=self.candidate_yaws,
            candidate_shape_params=self.candidate_shape_params,
            candidate_kind_ids=self.candidate_kind_ids,
            candidate_cell_counts=self.candidate_cell_counts,
            cell_size=self.cfg.surface_grid_cell_size_m,
            edge_band=self.cfg.surface_edge_band_m,
            corner_band=self.cfg.surface_corner_band_m,
        )

        contact_reward = self.cfg.rew_contact_count * contact_reward_count
        surface_grid_reward = self.cfg.rew_surface_grid * new_surface_cells
        geometry_reward = self.cfg.rew_geometry * geometry_weight_sum
        exploration_reward = contact_reward + surface_grid_reward + geometry_reward
        total_reward = total_reward + exploration_reward

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["rew_contact_count"] = contact_reward.mean().item()
        self.extras["log"]["rew_surface_grid"] = surface_grid_reward.mean().item()
        self.extras["log"]["rew_geometry"] = geometry_reward.mean().item()
        self.extras["log"]["rew_exploration_total"] = exploration_reward.mean().item()
        self.extras["log"]["contact_pair_count_raw"] = raw_pair_count.mean().item()
        self.extras["log"]["contact_pair_count_capped"] = contact_reward_count.mean().item()
        self.extras["log"]["new_surface_cells"] = new_surface_cells.mean().item()
        self.extras["log"]["geometry_weight_sum"] = geometry_weight_sum.mean().item()
        self.extras["log"]["active_obstacles"] = self.candidate_active_mask.sum(dim=-1).float().mean().item()
        self.extras["log"]["total_reward"] = total_reward.mean().item()

        if hasattr(self, "_skrl_agent") and getattr(self, "_skrl_agent", None) is not None:
            try:
                agent = getattr(self, "_skrl_agent")
                agent.track_data("Reward / rew_contact_count", self.extras["log"]["rew_contact_count"])
                agent.track_data("Reward / rew_surface_grid", self.extras["log"]["rew_surface_grid"])
                agent.track_data("Reward / rew_geometry", self.extras["log"]["rew_geometry"])
                agent.track_data("Reward / rew_exploration_total", self.extras["log"]["rew_exploration_total"])
                agent.track_data("Reward / contact_pair_count_raw", self.extras["log"]["contact_pair_count_raw"])
                agent.track_data("Reward / new_surface_cells", self.extras["log"]["new_surface_cells"])
                agent.track_data("Reward / geometry_weight_sum", self.extras["log"]["geometry_weight_sum"])
            except Exception:
                pass

        return total_reward
