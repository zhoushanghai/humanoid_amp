"""
Purpose: Build the static room scene and obstacle candidate library for the G1 AMP proprioception task.
Main contents: room asset configs, obstacle candidate configs, startup size randomization, precomputed scene-bank
generation, progress reporting, and fallback obstacle placement.
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


def _is_layout_position_valid(
    x: float,
    y: float,
    radius: float,
    placed_centers: Sequence[tuple[float, float, float]],
) -> bool:
    """Check whether an obstacle center is valid against robot spawn clearance and other obstacles."""
    if math.hypot(x, y) < (ROBOT_SPAWN_CLEARANCE_M + radius):
        return False

    for prev_x, prev_y, prev_radius in placed_centers:
        min_center_distance = prev_radius + radius + OBSTACLE_SURFACE_SPACING_M
        if math.hypot(x - prev_x, y - prev_y) < min_center_distance:
            return False

    return True


def _place_fallback_obstacle(
    row_index: int,
    origin_x: float,
    origin_y: float,
    shape_params_row: torch.Tensor,
    active_mask: torch.Tensor,
    positions: torch.Tensor,
    yaws: torch.Tensor,
    placed_centers: list[tuple[float, float, float]],
    cpu_device: torch.device,
) -> None:
    """Guarantee that each reset leaves at least one obstacle visible in the room."""
    candidate_order = sorted(
        range(shape_params_row.shape[0]),
        key=lambda candidate_index: compute_layout_radius(
            candidate_index % len(OBSTACLE_KINDS),
            shape_params_row[candidate_index],
        ),
    )
    corner_signs = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))

    for candidate_index in candidate_order:
        kind_id = candidate_index % len(OBSTACLE_KINDS)
        params = shape_params_row[candidate_index]
        radius = compute_layout_radius(kind_id, params)
        max_extent = ROOM_HALF_EXTENT_M - ROOM_WALL_MARGIN_M - radius
        if max_extent <= 0.0:
            continue

        for sign_x, sign_y in corner_signs:
            x = sign_x * max_extent
            y = sign_y * max_extent
            if not _is_layout_position_valid(x, y, radius, placed_centers):
                continue

            active_mask[row_index, candidate_index] = True
            positions[row_index, candidate_index, 0] = origin_x + x
            positions[row_index, candidate_index, 1] = origin_y + y
            positions[row_index, candidate_index, 2] = float(params[2].item()) * 0.5
            yaws[row_index, candidate_index] = float(
                torch.empty(1, device=cpu_device).uniform_(-math.pi, math.pi).item()
            )
            placed_centers.append((x, y, radius))

            slot_index = candidate_index // len(OBSTACLE_KINDS)
            cube_candidate = slot_index * len(OBSTACLE_KINDS) + OBSTACLE_KIND_TO_ID["cube"]
            cyl_candidate = slot_index * len(OBSTACLE_KINDS) + OBSTACLE_KIND_TO_ID["cylinder"]
            inactive_candidate = cyl_candidate if kind_id == OBSTACLE_KIND_TO_ID["cube"] else cube_candidate
            positions[row_index, inactive_candidate, 0] = origin_x
            positions[row_index, inactive_candidate, 1] = origin_y
            positions[row_index, inactive_candidate, 2] = HIDDEN_OBSTACLE_Z_M
            return


def _sample_local_obstacle_layout(
    shape_params_row: torch.Tensor,
    cpu_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one local obstacle layout for a single environment."""
    num_candidates = shape_params_row.shape[0]
    active_mask = torch.zeros((1, num_candidates), dtype=torch.bool, device=cpu_device)
    positions = torch.zeros((1, num_candidates, 3), dtype=torch.float32, device=cpu_device)
    yaws = torch.zeros((1, num_candidates), dtype=torch.float32, device=cpu_device)
    positions[:, :, 2] = HIDDEN_OBSTACLE_Z_M

    placed_centers: list[tuple[float, float, float]] = []
    num_active_slots = int(torch.randint(1, NUM_OBSTACLE_SLOTS + 1, (1,), device=cpu_device).item())
    slot_order = torch.randperm(NUM_OBSTACLE_SLOTS, device=cpu_device)
    enabled_slots = set(slot_order[:num_active_slots].tolist())

    for slot_index in range(NUM_OBSTACLE_SLOTS):
        cube_candidate = slot_index * len(OBSTACLE_KINDS) + OBSTACLE_KIND_TO_ID["cube"]
        cyl_candidate = slot_index * len(OBSTACLE_KINDS) + OBSTACLE_KIND_TO_ID["cylinder"]

        if slot_index not in enabled_slots:
            continue

        chosen_kind = int(torch.randint(0, len(OBSTACLE_KINDS), (1,), device=cpu_device).item())
        candidate_index = slot_index * len(OBSTACLE_KINDS) + chosen_kind
        kind_id = chosen_kind
        params = shape_params_row[candidate_index]
        radius = compute_layout_radius(kind_id, params)

        sample_limit = 64
        chosen_xy: tuple[float, float] | None = None
        for _ in range(sample_limit):
            max_extent = ROOM_HALF_EXTENT_M - ROOM_WALL_MARGIN_M - radius
            if max_extent <= 0.0:
                break
            x = float(torch.empty(1, device=cpu_device).uniform_(-max_extent, max_extent).item())
            y = float(torch.empty(1, device=cpu_device).uniform_(-max_extent, max_extent).item())
            if _is_layout_position_valid(x, y, radius, placed_centers):
                chosen_xy = (x, y)
                placed_centers.append((x, y, radius))
                break

        if chosen_xy is None:
            continue

        active_mask[0, candidate_index] = True
        height = float(params[2].item())
        positions[0, candidate_index, 0] = chosen_xy[0]
        positions[0, candidate_index, 1] = chosen_xy[1]
        positions[0, candidate_index, 2] = height * 0.5
        yaws[0, candidate_index] = float(
            torch.empty(1, device=cpu_device).uniform_(-math.pi, math.pi).item()
        )

        inactive_candidate = cyl_candidate if chosen_kind == OBSTACLE_KIND_TO_ID["cube"] else cube_candidate
        positions[0, inactive_candidate, 0] = 0.0
        positions[0, inactive_candidate, 1] = 0.0
        positions[0, inactive_candidate, 2] = HIDDEN_OBSTACLE_Z_M

    if not active_mask[0].any():
        _place_fallback_obstacle(
            row_index=0,
            origin_x=0.0,
            origin_y=0.0,
            shape_params_row=shape_params_row,
            active_mask=active_mask,
            positions=positions,
            yaws=yaws,
            placed_centers=placed_centers,
            cpu_device=cpu_device,
        )

    return active_mask[0], positions[0], yaws[0]


def build_obstacle_scene_bank(
    candidate_shape_params: torch.Tensor,
    scene_bank_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute a bank of valid obstacle layouts for each environment."""
    cpu_device = torch.device("cpu")
    shape_params_cpu = candidate_shape_params.to(device=cpu_device)
    scene_bank_size = max(int(scene_bank_size), 1)

    num_envs = shape_params_cpu.shape[0]
    num_candidates = shape_params_cpu.shape[1]
    active_mask_bank = torch.zeros(
        (num_envs, scene_bank_size, num_candidates),
        dtype=torch.bool,
        device=cpu_device,
    )
    local_positions_bank = torch.zeros(
        (num_envs, scene_bank_size, num_candidates, 3),
        dtype=torch.float32,
        device=cpu_device,
    )
    local_positions_bank[:, :, :, 2] = HIDDEN_OBSTACLE_Z_M
    yaws_bank = torch.zeros(
        (num_envs, scene_bank_size, num_candidates),
        dtype=torch.float32,
        device=cpu_device,
    )
    progress_interval_envs = max(num_envs // 8, 1)

    for env_index in range(num_envs):
        for layout_index in range(scene_bank_size):
            active_mask, local_positions, yaws = _sample_local_obstacle_layout(
                shape_params_row=shape_params_cpu[env_index],
                cpu_device=cpu_device,
            )
            active_mask_bank[env_index, layout_index] = active_mask
            local_positions_bank[env_index, layout_index] = local_positions
            yaws_bank[env_index, layout_index] = yaws
        if num_envs >= 256 and (
            (env_index + 1) % progress_interval_envs == 0 or (env_index + 1) == num_envs
        ):
            print(f"[INFO]: Obstacle scene bank progress: {env_index + 1}/{num_envs} envs")

    return (
        active_mask_bank.to(device=device),
        local_positions_bank.to(device=device),
        yaws_bank.to(device=device),
    )


def sample_scene_bank_layouts(
    scene_bank_active_mask: torch.Tensor,
    scene_bank_local_positions: torch.Tensor,
    scene_bank_yaws: torch.Tensor,
    env_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one precomputed layout per reset environment."""
    layout_ids = torch.randint(
        scene_bank_active_mask.shape[1],
        (len(env_ids),),
        device=env_ids.device,
    )
    return (
        scene_bank_active_mask[env_ids, layout_ids],
        scene_bank_local_positions[env_ids, layout_ids],
        scene_bank_yaws[env_ids, layout_ids],
    )
