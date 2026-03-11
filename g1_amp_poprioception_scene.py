"""
Purpose: Build the static room scene and obstacle candidate library for the G1 AMP proprioception task.
Main contents: room asset configs, obstacle candidate configs, startup size randomization, and per-reset layout sampling.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, UsdGeom, Vt

from .g1_amp_poprioception_constants import (
    CONTACT_FORCE_THRESHOLD_N,
    CUBE_BASE_EDGE_RANGE_M,
    CYLINDER_RADIUS_RANGE_M,
    ENV_REGEX_NS,
    HIDDEN_OBSTACLE_Z_M,
    NUM_OBSTACLE_SLOTS,
    OBSTACLE_HEIGHT_RANGE_M,
    OBSTACLE_KINDS,
    OBSTACLE_KIND_TO_ID,
    OBSTACLE_SURFACE_SPACING_M,
    ObstacleCandidateSpec,
    ROOM_HALF_EXTENT_M,
    ROOM_NAMESPACE,
    ROOM_WALL_HEIGHT_M,
    ROOM_WALL_MARGIN_M,
    ROOM_WALL_THICKNESS_M,
    ROBOT_SPAWN_CLEARANCE_M,
    UNIT_CUBE_SIZE_M,
    UNIT_CYLINDER_HEIGHT_M,
    UNIT_CYLINDER_RADIUS_M,
    VALID_CONTACT_BODIES,
    make_obstacle_candidate_name,
    make_obstacle_candidate_prim_path,
)


@dataclass(frozen=True)
class RoomAssetSpec:
    """Metadata describing one static room asset."""

    asset_name: str
    size: tuple[float, float, float]
    position: tuple[float, float, float]
    color: tuple[float, float, float]
    prim_path: str


ROOM_ASSET_SPECS: tuple[RoomAssetSpec, ...] = (
    RoomAssetSpec(
        asset_name="left_wall",
        size=(ROOM_WALL_THICKNESS_M, 2.0 * ROOM_HALF_EXTENT_M, ROOM_WALL_HEIGHT_M),
        position=(
            -(ROOM_HALF_EXTENT_M + ROOM_WALL_THICKNESS_M * 0.5),
            0.0,
            ROOM_WALL_HEIGHT_M * 0.5,
        ),
        color=(0.82, 0.80, 0.77),
        prim_path=f"{ROOM_NAMESPACE}/left_wall",
    ),
    RoomAssetSpec(
        asset_name="right_wall",
        size=(ROOM_WALL_THICKNESS_M, 2.0 * ROOM_HALF_EXTENT_M, ROOM_WALL_HEIGHT_M),
        position=(
            ROOM_HALF_EXTENT_M + ROOM_WALL_THICKNESS_M * 0.5,
            0.0,
            ROOM_WALL_HEIGHT_M * 0.5,
        ),
        color=(0.82, 0.80, 0.77),
        prim_path=f"{ROOM_NAMESPACE}/right_wall",
    ),
    RoomAssetSpec(
        asset_name="front_wall",
        size=(2.0 * ROOM_HALF_EXTENT_M, ROOM_WALL_THICKNESS_M, ROOM_WALL_HEIGHT_M),
        position=(
            0.0,
            ROOM_HALF_EXTENT_M + ROOM_WALL_THICKNESS_M * 0.5,
            ROOM_WALL_HEIGHT_M * 0.5,
        ),
        color=(0.82, 0.80, 0.77),
        prim_path=f"{ROOM_NAMESPACE}/front_wall",
    ),
    RoomAssetSpec(
        asset_name="back_wall",
        size=(2.0 * ROOM_HALF_EXTENT_M, ROOM_WALL_THICKNESS_M, ROOM_WALL_HEIGHT_M),
        position=(
            0.0,
            -(ROOM_HALF_EXTENT_M + ROOM_WALL_THICKNESS_M * 0.5),
            ROOM_WALL_HEIGHT_M * 0.5,
        ),
        color=(0.82, 0.80, 0.77),
        prim_path=f"{ROOM_NAMESPACE}/back_wall",
    ),
)


def _make_static_rigid_object_cfg(
    prim_path: str,
    spawn_cfg: sim_utils.CuboidCfg | sim_utils.CylinderCfg,
    position: tuple[float, float, float],
) -> RigidObjectCfg:
    """Create a kinematic rigid-object config used by the room and obstacle assets."""
    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        spawn=spawn_cfg,
    )


def build_room_asset_cfgs() -> dict[str, RigidObjectCfg]:
    """Return configs for the four static room walls."""
    wall_cfgs: dict[str, RigidObjectCfg] = {}
    for spec in ROOM_ASSET_SPECS:
        wall_cfgs[spec.asset_name] = _make_static_rigid_object_cfg(
            prim_path=spec.prim_path,
            position=spec.position,
            spawn_cfg=sim_utils.CuboidCfg(
                size=spec.size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=spec.color,
                    roughness=0.8,
                ),
            ),
        )
    return wall_cfgs


def build_obstacle_candidate_specs() -> list[ObstacleCandidateSpec]:
    """Return the ordered obstacle candidates used everywhere else in the task."""
    specs: list[ObstacleCandidateSpec] = []
    for slot_index in range(NUM_OBSTACLE_SLOTS):
        for kind in OBSTACLE_KINDS:
            specs.append(
                ObstacleCandidateSpec(
                    asset_name=make_obstacle_candidate_name(slot_index, kind),
                    kind=kind,
                    slot_index=slot_index,
                    prim_path=make_obstacle_candidate_prim_path(slot_index, kind),
                )
            )
    return specs


def build_obstacle_candidate_cfgs(
    candidate_specs: Sequence[ObstacleCandidateSpec],
) -> dict[str, RigidObjectCfg]:
    """Return configs for all pre-spawned obstacle candidates."""
    cfgs: dict[str, RigidObjectCfg] = {}
    for spec in candidate_specs:
        base_position = (0.0, 0.0, HIDDEN_OBSTACLE_Z_M)
        common_kwargs = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            activate_contact_sensors=True,
        )
        if spec.kind == "cube":
            spawn_cfg = sim_utils.CuboidCfg(
                size=UNIT_CUBE_SIZE_M,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.72, 0.47, 0.26),
                    roughness=0.55,
                ),
                **common_kwargs,
            )
        else:
            spawn_cfg = sim_utils.CylinderCfg(
                radius=UNIT_CYLINDER_RADIUS_M,
                height=UNIT_CYLINDER_HEIGHT_M,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.30, 0.55, 0.70),
                    roughness=0.45,
                ),
                **common_kwargs,
            )
        cfgs[spec.asset_name] = _make_static_rigid_object_cfg(
            prim_path=spec.prim_path,
            position=base_position,
            spawn_cfg=spawn_cfg,
        )
    return cfgs


def build_upper_body_contact_sensor_cfgs(
    candidate_specs: Sequence[ObstacleCandidateSpec],
) -> dict[str, ContactSensorCfg]:
    """Return one filtered contact sensor per rewarded upper-body link."""
    filter_exprs = [spec.prim_path for spec in candidate_specs]
    sensor_cfgs: dict[str, ContactSensorCfg] = {}
    for body_name in VALID_CONTACT_BODIES:
        sensor_cfgs[f"{body_name}_contact"] = ContactSensorCfg(
            prim_path=f"{ENV_REGEX_NS}/Robot/{body_name}",
            history_length=1,
            force_threshold=CONTACT_FORCE_THRESHOLD_N,
            filter_prim_paths_expr=filter_exprs,
        )
    return sensor_cfgs


def create_scene_parent_prims(
    env_prim_paths: Sequence[str],
    candidate_specs: Sequence[ObstacleCandidateSpec],
) -> None:
    """Create per-environment Xform parents needed by regex-based room and obstacle spawns."""
    slot_prim_paths = {
        f"{env_prim_path}/Obstacles/slot_{spec.slot_index}"
        for env_prim_path in env_prim_paths
        for spec in candidate_specs
    }
    for env_prim_path in env_prim_paths:
        sim_utils.create_prim(f"{env_prim_path}/Room", "Xform")
        sim_utils.create_prim(f"{env_prim_path}/Obstacles", "Xform")
    for slot_prim_path in sorted(slot_prim_paths):
        sim_utils.create_prim(slot_prim_path, "Xform")


def sample_startup_obstacle_shape_params(
    num_envs: int,
    candidate_specs: Sequence[ObstacleCandidateSpec],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one size library per environment before physics starts.

    Returns:
        A tuple of:
        - shape_params: tensor of shape (num_envs, num_candidates, 3)
        - scales: tensor of shape (num_envs, num_candidates, 3)
    """
    cpu_device = torch.device("cpu")
    num_candidates = len(candidate_specs)
    shape_params = torch.zeros((num_envs, num_candidates, 3), dtype=torch.float32, device=cpu_device)
    scales = torch.ones((num_envs, num_candidates, 3), dtype=torch.float32, device=cpu_device)

    for candidate_index, spec in enumerate(candidate_specs):
        if spec.kind == "cube":
            edge = torch.empty(num_envs, device=cpu_device).uniform_(*CUBE_BASE_EDGE_RANGE_M)
            height = torch.empty(num_envs, device=cpu_device).uniform_(*OBSTACLE_HEIGHT_RANGE_M)
            shape_params[:, candidate_index, 0] = edge
            shape_params[:, candidate_index, 1] = edge
            shape_params[:, candidate_index, 2] = height
            scales[:, candidate_index, 0] = edge / UNIT_CUBE_SIZE_M[0]
            scales[:, candidate_index, 1] = edge / UNIT_CUBE_SIZE_M[1]
            scales[:, candidate_index, 2] = height / UNIT_CUBE_SIZE_M[2]
        else:
            radius = torch.empty(num_envs, device=cpu_device).uniform_(*CYLINDER_RADIUS_RANGE_M)
            height = torch.empty(num_envs, device=cpu_device).uniform_(*OBSTACLE_HEIGHT_RANGE_M)
            shape_params[:, candidate_index, 0] = radius
            shape_params[:, candidate_index, 1] = radius
            shape_params[:, candidate_index, 2] = height
            scales[:, candidate_index, 0] = radius / UNIT_CYLINDER_RADIUS_M
            scales[:, candidate_index, 1] = radius / UNIT_CYLINDER_RADIUS_M
            scales[:, candidate_index, 2] = height / UNIT_CYLINDER_HEIGHT_M

    return shape_params.to(device=device), scales


def apply_startup_obstacle_scales(
    obstacle_assets: dict[str, RigidObject],
    candidate_specs: Sequence[ObstacleCandidateSpec],
    scales: torch.Tensor,
) -> None:
    """Apply per-environment obstacle scales before physics starts."""
    stage = get_current_stage()
    with Sdf.ChangeBlock():
        for candidate_index, spec in enumerate(candidate_specs):
            prim_paths = sim_utils.find_matching_prim_paths(obstacle_assets[spec.asset_name].cfg.prim_path)
            for env_index, prim_path in enumerate(prim_paths):
                prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
                scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
                if scale_spec is None:
                    scale_spec = Sdf.AttributeSpec(
                        prim_spec,
                        prim_path + ".xformOp:scale",
                        Sdf.ValueTypeNames.Double3,
                    )
                scale_value = scales[env_index, candidate_index].cpu().tolist()
                scale_spec.default = Gf.Vec3f(*scale_value)
                op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if op_order_spec is None:
                    op_order_spec = Sdf.AttributeSpec(
                        prim_spec,
                        UsdGeom.Tokens.xformOpOrder,
                        Sdf.ValueTypeNames.TokenArray,
                    )
                op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])


def compute_layout_radius(kind_id: int, shape_params: torch.Tensor) -> float:
    """Compute the 2D placement radius used by rejection sampling."""
    if kind_id == OBSTACLE_KIND_TO_ID["cube"]:
        edge = float(shape_params[0].item())
        return edge / math.sqrt(2.0)
    radius = float(shape_params[0].item())
    return radius


def yaw_to_quat_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    """Convert yaw angles in radians to wxyz quaternions."""
    half_yaw = yaw * 0.5
    quat = torch.zeros((yaw.shape[0], 4), dtype=torch.float32, device=yaw.device)
    quat[:, 0] = torch.cos(half_yaw)
    quat[:, 3] = torch.sin(half_yaw)
    return quat


def sample_episode_obstacle_layout(
    env_origins: torch.Tensor,
    candidate_shape_params: torch.Tensor,
    env_ids: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample which candidates are active and where they are placed for one reset."""
    cpu_device = torch.device("cpu")
    env_ids_cpu = env_ids.to(device=cpu_device, dtype=torch.long)
    origins_cpu = env_origins[env_ids].to(device=cpu_device)
    shape_params_cpu = candidate_shape_params[env_ids].to(device=cpu_device)

    num_envs = len(env_ids_cpu)
    num_candidates = shape_params_cpu.shape[1]
    active_mask = torch.zeros((num_envs, num_candidates), dtype=torch.bool, device=cpu_device)
    positions = torch.zeros((num_envs, num_candidates, 3), dtype=torch.float32, device=cpu_device)
    yaws = torch.zeros((num_envs, num_candidates), dtype=torch.float32, device=cpu_device)
    positions[:, :, 2] = HIDDEN_OBSTACLE_Z_M

    for row_index in range(num_envs):
        placed_centers: list[tuple[float, float, float]] = []
        num_active_slots = int(torch.randint(1, NUM_OBSTACLE_SLOTS + 1, (1,), device=cpu_device).item())
        slot_order = torch.randperm(NUM_OBSTACLE_SLOTS, device=cpu_device)
        enabled_slots = set(slot_order[:num_active_slots].tolist())
        origin_x = float(origins_cpu[row_index, 0].item())
        origin_y = float(origins_cpu[row_index, 1].item())

        for slot_index in range(NUM_OBSTACLE_SLOTS):
            cube_candidate = slot_index * len(OBSTACLE_KINDS) + OBSTACLE_KIND_TO_ID["cube"]
            cyl_candidate = slot_index * len(OBSTACLE_KINDS) + OBSTACLE_KIND_TO_ID["cylinder"]

            if slot_index not in enabled_slots:
                continue

            chosen_kind = int(torch.randint(0, len(OBSTACLE_KINDS), (1,), device=cpu_device).item())
            candidate_index = slot_index * len(OBSTACLE_KINDS) + chosen_kind
            kind_id = chosen_kind
            params = shape_params_cpu[row_index, candidate_index]
            radius = compute_layout_radius(kind_id, params)

            sample_limit = 64
            chosen_xy: tuple[float, float] | None = None
            for _ in range(sample_limit):
                max_extent = ROOM_HALF_EXTENT_M - ROOM_WALL_MARGIN_M - radius
                if max_extent <= 0.0:
                    break
                x = float(torch.empty(1, device=cpu_device).uniform_(-max_extent, max_extent).item())
                y = float(torch.empty(1, device=cpu_device).uniform_(-max_extent, max_extent).item())
                if math.hypot(x, y) < (ROBOT_SPAWN_CLEARANCE_M + radius):
                    continue

                valid = True
                for prev_x, prev_y, prev_radius in placed_centers:
                    min_center_distance = prev_radius + radius + OBSTACLE_SURFACE_SPACING_M
                    if math.hypot(x - prev_x, y - prev_y) < min_center_distance:
                        valid = False
                        break
                if valid:
                    chosen_xy = (x, y)
                    placed_centers.append((x, y, radius))
                    break

            if chosen_xy is None:
                continue

            active_mask[row_index, candidate_index] = True
            height = float(params[2].item())
            positions[row_index, candidate_index, 0] = origin_x + chosen_xy[0]
            positions[row_index, candidate_index, 1] = origin_y + chosen_xy[1]
            positions[row_index, candidate_index, 2] = height * 0.5
            yaws[row_index, candidate_index] = float(
                torch.empty(1, device=cpu_device).uniform_(-math.pi, math.pi).item()
            )

            inactive_candidate = cyl_candidate if chosen_kind == OBSTACLE_KIND_TO_ID["cube"] else cube_candidate
            positions[row_index, inactive_candidate, 0] = origin_x
            positions[row_index, inactive_candidate, 1] = origin_y
            positions[row_index, inactive_candidate, 2] = HIDDEN_OBSTACLE_Z_M

    return active_mask.to(device=device), positions.to(device=device), yaws.to(device=device)
