# DEV_LOG.md

## 2026-02-11 Initialization

- **Action**: Global code formatting.
- **Details**: Executed `black .` on all project files.
- **Environment**:
    - Conda Environment: g1_amp
    - Python Version: 3.11.14

## 2026-02-11 Feature Update: Data Converter CLI

- **Action**: Refactor `motions/data_convert.py` to support command-line arguments.
- **Details**:
    - Introduced `argparse` to handle dynamic input/output paths and robot model configurations.
    - Supported Arguments:
        - `--input` / `-i`: Path to input CSV file.
        - `--output` / `-o`: Path to output NPZ file.
        - `--urdf`: Path to robot URDF file.
        - `--mesh-dir`: Path to robot mesh directory.
        - `--start-frame` / `--end-frame`: Frame range selection.
    - **Execution Record**:
        ```bash
        conda run -n g1_amp python motions/data_convert.py \
          --input datasets/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv \
          --output motions/walk1_subject1_v2.npz \
          --urdf datasets/LAFAN1_Retargeting_Dataset/robot_description/g1/g1_29dof_rev_1_0.urdf \
          --mesh-dir datasets/LAFAN1_Retargeting_Dataset/robot_description/g1
        ```
    - **Reason**: To allow flexible batch processing and support different robot models without modifying the source code.

## 2026-02-11 Feature Update: Custom AMP Training Task

- **Action**: Enable training with custom motion file `motions/walk1_subject1_custom.npz`.
- **Details**:
    - **Environment Config**: Added `G1AmpWalkCustomEnvCfg` in `g1_amp_env_cfg.py` pointing to the new motion file.
    - **Agent Config**: Created `agents/skrl_g1_walk_custom_amp_cfg.yaml` with unique experiment name `g1_amp_walk_custom`.
    - **Task Registration**: Registered new Gym task `Isaac-G1-AMP-Walk-Custom-v0` in `__init__.py`.
    - **Bug Fix**: Fixed `omni.log.warn` AttributeError in `train.py` (replaced with `pass` as `omni` module behavior changed).
    - **Execution Record**:
        ```bash
        conda run -n g1_amp python -m humanoid_amp.train --task Isaac-G1-AMP-Walk-Custom-v0 --headless
        ```
    - **Reason**: To validate the pipeline with the custom converted motion data without overwriting the baseline `G1_walk.npz` configuration.

## 2026-02-11 Feature Update: Enhanced Play Video Naming

- **Action**: Update `play.py` to support unique video naming.
- **Details**:
    - **Modification**: Modified `play.py` to inject `name_prefix` into `gym.wrappers.RecordVideo`.
    - **Format**: `{CheckpointName}_{Timestamp}` (e.g., `agent_40000_2026-02-11_14-30-00`).
    - **Execution Record** (Example Play Command):
        ```bash
        conda run -n g1_amp python -m humanoid_amp.play \
          --task Isaac-G1-AMP-Walk-Custom-v0 \
          --checkpoint logs/skrl/g1_amp_walk_custom/2026-02-11_14-25-39_ppo_torch/checkpoints/agent_40000.pt \
          --num_envs 1 \
          --video \
          --video_length 300
        ```
    - **Reason**: To prevent video file overwrites and provide clear identification of source checkpoints and recording times.

## 2026-02-11 Bug Fix: Play Script Module Import

- **Action**: Fix `ModuleNotFoundError` in `play.py`.
- **Details**:
    - **Modification**: Commented out `from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint`.
    - **Module Function**: This module is used to retrieve pre-trained checkpoints from the NVIDIA Nucleus server (cloud-based asset storage).
    - **Reason for Removal**:
        1.  The module appears to be deprecated or moved in the current Isaac Lab version.
        2.  **Redundancy**: Our workflow exclusively uses local checkpoints (trained on-disk) provided via the `--checkpoint` argument. We do not need to download assets from Nucleus, rendering this import unnecessary for our current task.
