"""
Purpose: Compute exploration rewards for the G1 AMP proprioception task.
Main contents: body-object pair counting, obstacle surface discretization, and first-touch geometry weighting.
"""

from __future__ import annotations

import math

import torch

from .g1_amp_poprioception_constants import (
    GEOMETRY_WEIGHT_CORNER,
    GEOMETRY_WEIGHT_EDGE,
    GEOMETRY_WEIGHT_FACE,
    MAX_SURFACE_CELLS_PER_CANDIDATE,
    OBSTACLE_KIND_TO_ID,
)


def compute_contact_pair_count_reward(
    contact_forces: torch.Tensor,
    threshold: float,
    per_step_cap: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Count active body-object pairs from filtered contact forces."""
    contact_magnitudes = torch.linalg.norm(contact_forces, dim=-1)
    contact_mask = contact_magnitudes > threshold
    pair_count = contact_mask.sum(dim=(1, 2)).to(dtype=torch.float32)
    capped_pair_count = pair_count.clamp(max=float(per_step_cap))
    return capped_pair_count, pair_count, contact_mask


def compute_candidate_cell_counts(
    candidate_kind_ids: torch.Tensor,
    candidate_shape_params: torch.Tensor,
    cell_size: float,
) -> torch.Tensor:
    """Return the number of valid surface cells per candidate."""
    num_envs, num_candidates, _ = candidate_shape_params.shape
    cell_counts = torch.zeros((num_envs, num_candidates), dtype=torch.long, device=candidate_shape_params.device)

    cube_mask = candidate_kind_ids == OBSTACLE_KIND_TO_ID["cube"]
    cyl_mask = candidate_kind_ids == OBSTACLE_KIND_TO_ID["cylinder"]

    if cube_mask.any():
        cube_params = candidate_shape_params[:, cube_mask]
        nx = torch.ceil(cube_params[..., 0] / cell_size).to(torch.long).clamp(min=1)
        ny = torch.ceil(cube_params[..., 1] / cell_size).to(torch.long).clamp(min=1)
        nz = torch.ceil(cube_params[..., 2] / cell_size).to(torch.long).clamp(min=1)
        cube_counts = 2 * (ny * nz + nx * nz + nx * ny)
        cell_counts[:, cube_mask] = cube_counts.clamp(max=MAX_SURFACE_CELLS_PER_CANDIDATE)

    if cyl_mask.any():
        cyl_params = candidate_shape_params[:, cyl_mask]
        radius = cyl_params[..., 0]
        height = cyl_params[..., 2]
        theta = torch.ceil((2.0 * math.pi * radius) / cell_size).to(torch.long).clamp(min=8)
        radial = torch.ceil(radius / cell_size).to(torch.long).clamp(min=1)
        vertical = torch.ceil(height / cell_size).to(torch.long).clamp(min=1)
        cyl_counts = theta * vertical + 2 * theta * radial
        cell_counts[:, cyl_mask] = cyl_counts.clamp(max=MAX_SURFACE_CELLS_PER_CANDIDATE)

    return cell_counts


def compute_surface_grid_rewards(
    visited_surface_cells: torch.Tensor,
    contact_mask: torch.Tensor,
    body_positions_w: torch.Tensor,
    candidate_positions_w: torch.Tensor,
    candidate_yaws: torch.Tensor,
    candidate_shape_params: torch.Tensor,
    candidate_kind_ids: torch.Tensor,
    candidate_cell_counts: torch.Tensor,
    cell_size: float,
    edge_band: float,
    corner_band: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update the per-episode surface coverage buffer and return new-cell statistics."""
    num_envs, _, _ = contact_mask.shape
    new_cell_counts = torch.zeros(num_envs, dtype=torch.float32, device=visited_surface_cells.device)
    new_weight_sums = torch.zeros(num_envs, dtype=torch.float32, device=visited_surface_cells.device)

    candidate_contact_any = contact_mask.any(dim=1)
    if not candidate_contact_any.any():
        return new_cell_counts, new_weight_sums

    first_contact_sensor = torch.argmax(contact_mask.to(torch.int32), dim=1)
    env_candidate_ids = candidate_contact_any.nonzero(as_tuple=False)
    env_ids = env_candidate_ids[:, 0]
    candidate_ids = env_candidate_ids[:, 1]
    sensor_ids = first_contact_sensor[env_ids, candidate_ids]

    contact_points_w = body_positions_w[env_ids, sensor_ids]
    obstacle_positions_w = candidate_positions_w[env_ids, candidate_ids]
    yaws = candidate_yaws[env_ids, candidate_ids]
    shape_params = candidate_shape_params[env_ids, candidate_ids]
    cell_counts = candidate_cell_counts[env_ids, candidate_ids]
    kind_ids = candidate_kind_ids[candidate_ids]

    delta = contact_points_w - obstacle_positions_w
    cos_yaw = torch.cos(yaws)
    sin_yaw = torch.sin(yaws)
    local_x = cos_yaw * delta[:, 0] + sin_yaw * delta[:, 1]
    local_y = -sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1]
    local_z = delta[:, 2]
    local_points = torch.stack((local_x, local_y, local_z), dim=-1)

    cell_ids = torch.full((env_candidate_ids.shape[0],), -1, dtype=torch.long, device=visited_surface_cells.device)
    weights = torch.zeros((env_candidate_ids.shape[0],), dtype=torch.float32, device=visited_surface_cells.device)

    cube_mask = kind_ids == OBSTACLE_KIND_TO_ID["cube"]
    if cube_mask.any():
        cube_cell_ids, cube_weights = _map_cuboid_points_to_surface_cells(
            local_points[cube_mask],
            shape_params[cube_mask],
            cell_size,
            edge_band,
            corner_band,
        )
        cell_ids[cube_mask] = cube_cell_ids
        weights[cube_mask] = cube_weights

    cyl_mask = kind_ids == OBSTACLE_KIND_TO_ID["cylinder"]
    if cyl_mask.any():
        cyl_cell_ids, cyl_weights = _map_cylinder_points_to_surface_cells(
            local_points[cyl_mask],
            shape_params[cyl_mask],
            cell_size,
            edge_band,
        )
        cell_ids[cyl_mask] = cyl_cell_ids
        weights[cyl_mask] = cyl_weights

    valid_mask = (cell_ids >= 0) & (cell_ids < cell_counts)
    if not valid_mask.any():
        return new_cell_counts, new_weight_sums

    env_ids = env_ids[valid_mask]
    candidate_ids = candidate_ids[valid_mask]
    cell_ids = cell_ids[valid_mask]
    weights = weights[valid_mask]

    is_new = ~visited_surface_cells[env_ids, candidate_ids, cell_ids]
    if not is_new.any():
        return new_cell_counts, new_weight_sums

    new_env_ids = env_ids[is_new]
    new_candidate_ids = candidate_ids[is_new]
    new_cell_ids = cell_ids[is_new]
    new_weights = weights[is_new]

    visited_surface_cells[new_env_ids, new_candidate_ids, new_cell_ids] = True
    new_cell_counts.index_add_(
        0,
        new_env_ids,
        torch.ones_like(new_weights, dtype=torch.float32),
    )
    new_weight_sums.index_add_(0, new_env_ids, new_weights)
    return new_cell_counts, new_weight_sums


def _map_cuboid_points_to_surface_cells(
    local_points: torch.Tensor,
    shape_params: torch.Tensor,
    cell_size: float,
    edge_band: float,
    corner_band: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map local cuboid points to a contiguous surface-cell id and geometry weight."""
    num_points = local_points.shape[0]
    cell_ids = torch.full((num_points,), -1, dtype=torch.long, device=local_points.device)
    weights = torch.full((num_points,), GEOMETRY_WEIGHT_FACE, dtype=torch.float32, device=local_points.device)

    half_extents = shape_params * 0.5
    abs_points = local_points.abs()
    face_axis = torch.argmin(torch.abs(half_extents - abs_points), dim=-1)
    nx = torch.ceil(shape_params[:, 0] / cell_size).to(torch.long).clamp(min=1)
    ny = torch.ceil(shape_params[:, 1] / cell_size).to(torch.long).clamp(min=1)
    nz = torch.ceil(shape_params[:, 2] / cell_size).to(torch.long).clamp(min=1)

    face_count_x = ny * nz
    face_count_y = nx * nz

    for axis in range(3):
        axis_mask = face_axis == axis
        if not axis_mask.any():
            continue

        ids = axis_mask.nonzero(as_tuple=False).squeeze(-1)
        points = local_points[ids]
        half = half_extents[ids]
        sign_positive = points[:, axis] >= 0.0

        if axis == 0:
            u = points[:, 1].clamp(min=-half[:, 1], max=half[:, 1])
            v = points[:, 2].clamp(min=-half[:, 2], max=half[:, 2])
            u_count = ny[ids]
            v_count = nz[ids]
            face_count = face_count_x[ids]
            base = torch.where(sign_positive, face_count, torch.zeros_like(face_count))
            half_u = half[:, 1]
            half_v = half[:, 2]
            dim_u = (2.0 * half_u).clamp(min=1.0e-6)
            dim_v = (2.0 * half_v).clamp(min=1.0e-6)
        elif axis == 1:
            u = points[:, 0].clamp(min=-half[:, 0], max=half[:, 0])
            v = points[:, 2].clamp(min=-half[:, 2], max=half[:, 2])
            u_count = nx[ids]
            v_count = nz[ids]
            x_face_count = face_count_x[ids]
            face_count = face_count_y[ids]
            base = 2 * x_face_count + torch.where(sign_positive, face_count, torch.zeros_like(face_count))
            half_u = half[:, 0]
            half_v = half[:, 2]
            dim_u = (2.0 * half_u).clamp(min=1.0e-6)
            dim_v = (2.0 * half_v).clamp(min=1.0e-6)
        else:
            u = points[:, 0].clamp(min=-half[:, 0], max=half[:, 0])
            v = points[:, 1].clamp(min=-half[:, 1], max=half[:, 1])
            u_count = nx[ids]
            v_count = ny[ids]
            x_face_count = face_count_x[ids]
            y_face_count = face_count_y[ids]
            face_count = u_count * v_count
            base = 2 * x_face_count + 2 * y_face_count + torch.where(
                sign_positive, face_count, torch.zeros_like(face_count)
            )
            half_u = half[:, 0]
            half_v = half[:, 1]
            dim_u = (2.0 * half_u).clamp(min=1.0e-6)
            dim_v = (2.0 * half_v).clamp(min=1.0e-6)

        u_idx = torch.floor(((u + dim_u * 0.5) / dim_u) * u_count.to(torch.float32)).to(torch.long)
        v_idx = torch.floor(((v + dim_v * 0.5) / dim_v) * v_count.to(torch.float32)).to(torch.long)
        u_idx = u_idx.clamp(min=0)
        v_idx = v_idx.clamp(min=0)
        u_idx = torch.minimum(u_idx, u_count - 1)
        v_idx = torch.minimum(v_idx, v_count - 1)
        cell_ids[ids] = base + u_idx * v_count + v_idx

        edge_u = (half_u - u.abs()) <= edge_band
        edge_v = (half_v - v.abs()) <= edge_band
        corner_u = (half_u - u.abs()) <= corner_band
        corner_v = (half_v - v.abs()) <= corner_band
        face_weight = torch.full((ids.shape[0],), GEOMETRY_WEIGHT_FACE, dtype=torch.float32, device=local_points.device)
        both_edges = corner_u & corner_v
        one_edge = (edge_u | edge_v) & ~both_edges
        face_weight[one_edge] = GEOMETRY_WEIGHT_EDGE
        face_weight[both_edges] = GEOMETRY_WEIGHT_CORNER
        weights[ids] = face_weight

    return cell_ids, weights


def _map_cylinder_points_to_surface_cells(
    local_points: torch.Tensor,
    shape_params: torch.Tensor,
    cell_size: float,
    edge_band: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map local cylinder points to a contiguous surface-cell id and geometry weight."""
    radius = shape_params[:, 0].clamp(min=1.0e-6)
    height = shape_params[:, 2].clamp(min=1.0e-6)
    radial_distance = torch.linalg.norm(local_points[:, :2], dim=-1).clamp(min=1.0e-6)
    half_height = height * 0.5

    theta = torch.atan2(local_points[:, 1], local_points[:, 0])
    theta = torch.remainder(theta + 2.0 * math.pi, 2.0 * math.pi)
    theta_count = torch.ceil((2.0 * math.pi * radius) / cell_size).to(torch.long).clamp(min=8)
    radial_count = torch.ceil(radius / cell_size).to(torch.long).clamp(min=1)
    height_count = torch.ceil(height / cell_size).to(torch.long).clamp(min=1)
    theta_idx = torch.floor(theta / (2.0 * math.pi) * theta_count.to(torch.float32)).to(torch.long)
    theta_idx = torch.minimum(theta_idx.clamp(min=0), theta_count - 1)

    radial_gap = torch.abs(radial_distance - radius)
    cap_gap = torch.abs(local_points[:, 2].abs() - half_height)
    lateral_mask = radial_gap <= cap_gap

    cell_ids = torch.full((local_points.shape[0],), -1, dtype=torch.long, device=local_points.device)
    weights = torch.full((local_points.shape[0],), GEOMETRY_WEIGHT_FACE, dtype=torch.float32, device=local_points.device)

    if lateral_mask.any():
        ids = lateral_mask.nonzero(as_tuple=False).squeeze(-1)
        z_clamped = local_points[ids, 2].clamp(min=-half_height[ids], max=half_height[ids])
        z_idx = torch.floor(((z_clamped + half_height[ids]) / height[ids]) * height_count[ids].to(torch.float32)).to(
            torch.long
        )
        z_idx = torch.minimum(z_idx.clamp(min=0), height_count[ids] - 1)
        cell_ids[ids] = theta_idx[ids] * height_count[ids] + z_idx
        near_cap = (half_height[ids] - z_clamped.abs()) <= edge_band
        lateral_weights = weights[ids].clone()
        lateral_weights[near_cap] = GEOMETRY_WEIGHT_EDGE
        weights[ids] = lateral_weights

    cap_mask = ~lateral_mask
    if cap_mask.any():
        ids = cap_mask.nonzero(as_tuple=False).squeeze(-1)
        radial_clamped = radial_distance[ids].clamp(max=radius[ids])
        radial_idx = torch.floor((radial_clamped / radius[ids]) * radial_count[ids].to(torch.float32)).to(torch.long)
        radial_idx = torch.minimum(radial_idx.clamp(min=0), radial_count[ids] - 1)
        lateral_count = theta_count[ids] * height_count[ids]
        is_top_cap = local_points[ids, 2] >= 0.0
        cap_offset = torch.where(is_top_cap, theta_count[ids] * radial_count[ids], torch.zeros_like(lateral_count))
        cell_ids[ids] = lateral_count + cap_offset + theta_idx[ids] * radial_count[ids] + radial_idx
        near_rim = (radius[ids] - radial_clamped) <= edge_band
        cap_weights = weights[ids].clone()
        cap_weights[near_rim] = GEOMETRY_WEIGHT_EDGE
        weights[ids] = cap_weights

    return cell_ids, weights
