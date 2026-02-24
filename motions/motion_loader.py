# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import torch
from typing import Optional
import glob
import yaml


def _resolve_motion_files(motion_file: str) -> list[str]:
    """Resolve motion files from various input formats.

    Supports:
    - YAML config file with motion_files list
    - Glob patterns (*, ?)
    - Directory paths
    - Individual file paths
    - Comma-separated file paths

    Args:
        motion_file: Path to config/file/pattern

    Returns:
        List of resolved file paths
    """
    # Check if it's a YAML config file
    if motion_file.endswith('.yaml') or motion_file.endswith('.yml'):
        config_dir = os.path.dirname(motion_file)
        with open(motion_file, 'r') as f:
            config = yaml.safe_load(f)

        files = []
        if config and 'motion_files' in config:
            for path in config['motion_files']:
                # Resolve relative paths based on config file location
                if not os.path.isabs(path):
                    path = os.path.join(config_dir, path)
                if os.path.exists(path):
                    files.append(path)
                else:
                    print(f"Warning: File not found: {path}")

        # Also support glob_pattern from config
        if not files and 'glob_pattern' in config:
            pattern = config['glob_pattern']
            if not os.path.isabs(pattern):
                pattern = os.path.join(config_dir, pattern)
            files = sorted(glob.glob(pattern))

        if not files:
            raise ValueError(f"No valid motion files found in config: {motion_file}")
        return files

    # Comma-separated files
    if ',' in motion_file:
        files = []
        for f in motion_file.split(','):
            f = f.strip()
            if os.path.exists(f):
                files.append(f)
        if files:
            return files

    # Glob pattern
    if "*" in motion_file or "?" in motion_file:
        files = sorted(glob.glob(motion_file))
        if files:
            return files

    # Directory
    if os.path.isdir(motion_file):
        files = sorted(glob.glob(os.path.join(motion_file, "*.npz")))
        if files:
            return files

    # Single file
    if os.path.exists(motion_file):
        return [motion_file]

    raise ValueError(f"No files found for pattern: {motion_file}")


class MotionLoader:
    """
    Helper class to load and sample motion data from NumPy-file format.
    Supports loading multiple files via:
    - YAML config file with motion_files list
    - Glob wildcard patterns (*, ?)
    - Directory paths
    - Individual file paths
    - Comma-separated file paths
    """

    def __init__(self, motion_file: str, device: torch.device) -> None:
        files = _resolve_motion_files(motion_file)
        print(f"Loading {len(files)} motion file(s) from: {motion_file}")

        self.device = device

        dof_pos_list = []
        dof_vel_list = []
        body_pos_list = []
        body_rot_list = []
        body_lin_vel_list = []
        body_ang_vel_list = []

        self.traj_starts = []
        self.traj_ends = []
        self.durations = []
        self.num_trajectories = len(files)

        current_frame = 0
        for f in files:
            data = np.load(f)
            if current_frame == 0:
                self._dof_names = data["dof_names"].tolist()
                self._body_names = data["body_names"].tolist()
                self.dt = 1.0 / data["fps"]

            dof_pos_list.append(data["dof_positions"])
            dof_vel_list.append(data["dof_velocities"])
            body_pos_list.append(data["body_positions"])
            body_rot_list.append(data["body_rotations"])
            body_lin_vel_list.append(data["body_linear_velocities"])
            body_ang_vel_list.append(data["body_angular_velocities"])

            n_frames = data["dof_positions"].shape[0]
            self.traj_starts.append(current_frame)
            current_frame += n_frames
            self.traj_ends.append(current_frame - 1)
            self.durations.append(self.dt * (n_frames - 1))

        self.traj_starts = np.array(self.traj_starts)
        self.traj_ends = np.array(self.traj_ends)
        self.durations = np.array(self.durations)

        self.dof_positions = torch.tensor(
            np.concatenate(dof_pos_list), dtype=torch.float32, device=self.device
        )
        self.dof_velocities = torch.tensor(
            np.concatenate(dof_vel_list), dtype=torch.float32, device=self.device
        )
        self.body_positions = torch.tensor(
            np.concatenate(body_pos_list), dtype=torch.float32, device=self.device
        )
        self.body_rotations = torch.tensor(
            np.concatenate(body_rot_list), dtype=torch.float32, device=self.device
        )
        self.body_linear_velocities = torch.tensor(
            np.concatenate(body_lin_vel_list), dtype=torch.float32, device=self.device
        )
        self.body_angular_velocities = torch.tensor(
            np.concatenate(body_ang_vel_list), dtype=torch.float32, device=self.device
        )

        self.num_frames = current_frame
        self.duration = float(np.sum(self.durations))
        print(
            f"Motion loaded: {self.num_trajectories} files, total duration: {self.duration} sec, total frames: {self.num_frames}"
        )

    @property
    def dof_names(self) -> list[str]:
        """Skeleton DOF names."""
        return self._dof_names

    @property
    def body_names(self) -> list[str]:
        """Skeleton rigid body names."""
        return self._body_names

    @property
    def num_dofs(self) -> int:
        """Number of skeleton's DOFs."""
        return len(self._dof_names)

    @property
    def num_bodies(self) -> int:
        """Number of skeleton's rigid bodies."""
        return len(self._body_names)

    def _interpolate(
        self,
        a: torch.Tensor,
        *,
        b: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Linear interpolation between consecutive values.

        Args:
            a: The first value. Shape is (N, X) or (N, M, X).
            b: The second value. Shape is (N, X) or (N, M, X).
            blend: Interpolation coefficient between 0 (a) and 1 (b).
            start: Indexes to fetch the first value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).
            end: Indexes to fetch the second value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).

        Returns:
            Interpolated values. Shape is (N, X) or (N, M, X).
        """
        if start is not None and end is not None:
            return self._interpolate(a=a[start], b=a[end], blend=blend)
        if a.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if a.ndim >= 3:
            blend = blend.unsqueeze(-1)
        return (1.0 - blend) * a + blend * b

    def _slerp(
        self,
        q0: torch.Tensor,
        *,
        q1: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Interpolation between consecutive rotations (Spherical Linear Interpolation).

        Args:
            q0: The first quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            q1: The second quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            blend: Interpolation coefficient between 0 (q0) and 1 (q1).
            start: Indexes to fetch the first quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).
            end: Indexes to fetch the second quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).

        Returns:
            Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
        """
        if start is not None and end is not None:
            return self._slerp(q0=q0[start], q1=q0[end], blend=blend)
        if q0.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if q0.ndim >= 3:
            blend = blend.unsqueeze(-1)

        qw, qx, qy, qz = 0, 1, 2, 3  # wxyz
        cos_half_theta = (
            q0[..., qw] * q1[..., qw]
            + q0[..., qx] * q1[..., qx]
            + q0[..., qy] * q1[..., qy]
            + q0[..., qz] * q1[..., qz]
        )

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)
        cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

        ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
        ratio_b = torch.sin(blend * half_theta) / sin_half_theta

        new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
        new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
        new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
        new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

        new_q = torch.cat(
            [new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1
        )
        new_q = torch.where(
            torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q
        )
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
        return new_q

    def _compute_frame_blend(
        self, times: np.ndarray, motion_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the indexes of the first and second values, as well as the blending time
        to interpolate between them and the given times.

        Args:
            times: Times, between 0 and motion duration, to sample motion values.
            motion_ids: Array of sequence IDs corresponding to each time.

        Returns:
            First value global indexes, Second value global indexes, and blending time.
        """
        durations = self.durations[motion_ids]
        starts = self.traj_starts[motion_ids]
        ends = self.traj_ends[motion_ids]

        phase = np.clip(times / durations, 0.0, 1.0)
        local_index_0 = (phase * (ends - starts)).round(decimals=0).astype(int)
        local_index_1 = np.minimum(local_index_0 + 1, ends - starts)

        index_0 = starts + local_index_0
        index_1 = starts + local_index_1

        blend = ((times - local_index_0 * self.dt) / self.dt).round(decimals=5)

        return index_0, index_1, blend

    def sample_times(
        self, num_samples: int, start: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample random motion times uniformly from uniformly random sequences.

        Args:
            num_samples: Number of time samples to generate.
            start: Whether to sample exactly from the start (t=0) of the sequences.

        Returns:
            Tuple of (motion_ids, times)
        """
        motion_ids = np.random.randint(0, self.num_trajectories, size=num_samples)
        if start:
            times = np.zeros(num_samples)
        else:
            times = (
                np.random.uniform(low=0.0, high=1.0, size=num_samples)
                * self.durations[motion_ids]
            )
        return motion_ids, times

    def sample(
        self,
        num_samples: int,
        times: Optional[np.ndarray] = None,
        duration: float | None = None,
        motion_ids: np.ndarray | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Sample motion data.

        Args:
            num_samples: Number of time samples to generate. If `times` is defined, this parameter is ignored.
            times: Motion time used for sampling.
                If not defined, motion data will be random sampled uniformly in time.
            duration: Maximum motion duration to sample.
                If not defined, samples will be within the range of the motion duration.
                If `times` is defined, this parameter is ignored.
            motion_ids: Array of sequence IDs corresponding to each time.

        Returns:
            Sampled motion DOF positions, DOF velocities,
            body positions, body rotations,
            body linear velocities and body angular velocities.
        """
        if times is None:
            motion_ids_new, times = self.sample_times(num_samples)
            if motion_ids is None:
                motion_ids = motion_ids_new
        elif motion_ids is None:
            motion_ids = np.zeros(num_samples, dtype=np.int32)

        index_0, index_1, blend = self._compute_frame_blend(
            times, motion_ids=motion_ids
        )
        blend = torch.tensor(blend, dtype=torch.float32, device=self.device)

        return (
            self._interpolate(
                self.dof_positions, blend=blend, start=index_0, end=index_1
            ),
            self._interpolate(
                self.dof_velocities, blend=blend, start=index_0, end=index_1
            ),
            self._interpolate(
                self.body_positions, blend=blend, start=index_0, end=index_1
            ),
            self._slerp(self.body_rotations, blend=blend, start=index_0, end=index_1),
            self._interpolate(
                self.body_linear_velocities, blend=blend, start=index_0, end=index_1
            ),
            self._interpolate(
                self.body_angular_velocities, blend=blend, start=index_0, end=index_1
            ),
        )

    def get_dof_index(self, dof_names: list[str]) -> list[int]:
        """Get skeleton DOFs indexes by DOFs names.

        Args:
            dof_names: List of DOFs names.

        Raises:
            AssertionError: If the specified DOFs name doesn't exist.

        Returns:
            List of DOFs indexes.
        """
        indexes = []
        for name in dof_names:
            assert (
                name in self._dof_names
            ), f"The specified DOF name ({name}) doesn't exist: {self._dof_names}"
            indexes.append(self._dof_names.index(name))
        return indexes

    def get_body_index(self, body_names: list[str]) -> list[int]:
        """Get skeleton body indexes by body names.

        Args:
            dof_names: List of body names.

        Raises:
            AssertionError: If the specified body name doesn't exist.

        Returns:
            List of body indexes.
        """
        indexes = []
        for name in body_names:
            assert (
                name in self._body_names
            ), f"The specified body name ({name}) doesn't exist: {self._body_names}"
            indexes.append(self._body_names.index(name))
        return indexes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Motion file")
    args, _ = parser.parse_known_args()

    motion = MotionLoader(args.file, "cpu")

    print("- number of frames:", motion.num_frames)
    print("- number of DOFs:", motion.num_dofs)
    print("- dt:", motion.dt)
    print("- fps:", 1.0 / motion.dt)
    print("- number of bodies:", motion.num_bodies)
