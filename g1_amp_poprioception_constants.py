"""
Purpose: Define shared constants for the G1 AMP proprioception exploration task.
Main contents: room geometry defaults, obstacle naming templates, valid contact bodies, and reward/grid hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass

ROOM_SIZE_M = 3.0
ROOM_HALF_EXTENT_M = ROOM_SIZE_M * 0.5
ROOM_WALL_HEIGHT_M = 1.5
ROOM_WALL_THICKNESS_M = 0.08
ROOM_WALL_MARGIN_M = 0.25
ROBOT_SPAWN_CLEARANCE_M = 0.75
OBSTACLE_SURFACE_SPACING_M = 0.5
ROOM_TERMINATION_MARGIN_M = 0.35

NUM_OBSTACLE_SLOTS = 3
OBSTACLE_KINDS = ("cube", "cylinder")
OBSTACLE_KIND_TO_ID = {"cube": 0, "cylinder": 1}

CUBE_BASE_EDGE_RANGE_M = (0.40, 0.80)
CYLINDER_RADIUS_RANGE_M = (0.20, 0.35)
OBSTACLE_HEIGHT_RANGE_M = (0.50, 1.00)

SURFACE_GRID_CELL_SIZE_M = 0.10
SURFACE_EDGE_BAND_M = 0.08
SURFACE_CORNER_BAND_M = 0.08
MAX_SURFACE_CELLS_PER_CANDIDATE = 1024

GEOMETRY_WEIGHT_FACE = 1.0
GEOMETRY_WEIGHT_EDGE = 3.0
GEOMETRY_WEIGHT_CORNER = 5.0

CONTACT_FORCE_THRESHOLD_N = 1.0
CONTACT_COUNT_PER_STEP_CAP = 4
CONTACT_COUNT_REWARD_SCALE = 0.10
SURFACE_GRID_REWARD_SCALE = 0.20
GEOMETRY_REWARD_SCALE = 0.05

VALID_CONTACT_BODIES = (
    "left_rubber_hand",
    "right_rubber_hand",
    "left_elbow_link",
    "right_elbow_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
)

ENV_REGEX_NS = "/World/envs/env_.*"
ROOM_NAMESPACE = f"{ENV_REGEX_NS}/Room"
OBSTACLE_NAMESPACE = f"{ENV_REGEX_NS}/Obstacles"

GROUND_PRIM_PATH = "/World/ground"
LIGHT_PRIM_PATH = "/World/Light"

UNIT_CUBE_SIZE_M = (1.0, 1.0, 1.0)
UNIT_CYLINDER_RADIUS_M = 0.5
UNIT_CYLINDER_HEIGHT_M = 1.0
HIDDEN_OBSTACLE_Z_M = -10.0


@dataclass(frozen=True)
class ObstacleCandidateSpec:
    """Metadata for a single pre-spawned obstacle candidate."""

    asset_name: str
    kind: str
    slot_index: int
    prim_path: str


def make_obstacle_candidate_name(slot_index: int, kind: str) -> str:
    """Return the unique asset name used for one obstacle candidate."""
    return f"obstacle_slot_{slot_index}_{kind}"


def make_obstacle_candidate_prim_path(slot_index: int, kind: str) -> str:
    """Return the prim regex for one obstacle candidate."""
    return f"{OBSTACLE_NAMESPACE}/slot_{slot_index}/{kind}"
