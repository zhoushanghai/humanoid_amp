# Developer Log

## Initialization

- **Date**: 2026-02-13
- **Action**: Project initialization per AGENT.md protocol.
- **Environment**:
    - **Python**: 3.10 (Target per AGENT.md)
    - **Conda Env**: Unknown (Environment check timed out)
- **Status**: 
    - Formatting check: Attempted
    - Initial commit: Skipped due to environment timeout.

## Configuration Change
- **Date**: 2026-02-13
- **Action**: Switch `g1_cfg.py` to use URDF model.
- **Details**: 
    - Replaced `UsdFileCfg` with `UrdfFileCfg`.
    - Set `asset_path` to `g1_model/urdf/g1_29dof_rev_1_0.urdf`.
    - Added `fix_base=False` and `default_drive_type="position"`.
    - **Correction**: Replaced `default_drive_type` with `joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(...)` to fix TypeError.
    - **Correction (2)**: Added `gains` with `stiffness=0.0` and `damping=0.0` to `JointDriveCfg` to fix `Missing values detected` error.
    - **Correction (3)**: Set `merge_fixed_joints=False` to prevent `right_rubber_hand` from being merged, fixing `ValueError`.

## Tool Enhancement
- **Date**: 2026-02-13
- **Action**: updated `motions/data_convert.py`
- **Details**:
    - Corrected relative paths for `urdf_path` and `mesh_dir`.
    - Added `argparse` support for command-line arguments: `--input`, `--output`, `--start-frame`, `--end-frame`.
    - **Correction**: Used `os.path.abspath(__file__)` to resolve URDF/mesh paths, fixing `FileNotFoundError` when running from different directories.




