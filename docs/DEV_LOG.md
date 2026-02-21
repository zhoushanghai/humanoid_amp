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

转化脚本
python motions/data_convert.py \
  --csv  datasets/walk1_subject1.csv\
  --urdf g1_model/urdf/g1_29dof_rev_1_0.urdf \
  --meshes g1_model/urdf \
  --start 110 \
  --end 265
  
  
python motions/motion_replayer.py \
  --motion motions/custom_motion.npz

python motions/motion_replayer.py \
  --motion /home/hz/datasets/g1_amp_nzp/walk1_subject1.npz



---
 speed
```
python -m humanoid_amp.train --task Isaac-G1-AMP-Speed-Direct-v0 --headless 


python -m humanoid_amp.train --task Isaac-G1-AMP-Speed-Direct-v0 --headless \
 --checkpoint logs/skrl/g1_amp_dance/2026-02-14_04-58-01_ppo_torch/checkpoints/agent_450000.pt
```
paly

python -m humanoid_amp.play \
--task Isaac-G1-AMP-Custom-Direct-v0 \
--num_envs 32 \
--checkpoint logs/skrl/g1_amp_dance/2026-02-14_04-58-01_ppo_torch/checkpoints/agent_450000.pt

---
python -m humanoid_amp.play_velocity_track \
  --task Isaac-G1-AMP-Speed-Direct-v0 \
  --num_envs 1 \
  --video \
  --video_length 400 \
  --target_vel 1.0 \
  --checkpoint logs/skrl/g1_amp_dance/2026-02-14_05-15-31_ppo_torch/checkpoints/agent_450000.pt \
  --headless

python -m humanoid_amp.play_velocity_track \
  --task Isaac-G1-AMP-Speed-Direct-v0 \
  --num_envs 1 \
  --headless \
  --target_vel 1.0 \
  --warmup_steps 50 \
  --video \
  --video_length 400 \
  --checkpoint logs/skrl/g1_amp_dance/2026-02-14_05-15-31_ppo_torch/checkpoints/agent_450000.pt


---
拼接 12 条 walk 数据集
python -m humanoid_amp.train --task Isaac-G1-AMP-Deploy-Direct-v0 --headless

python -m humanoid_amp.play \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --num_envs 32 \
  --checkpoint logs/skrl/g1_amp_dance/2026-02-21_01-30-28_ppo_torch/checkpoints/agent_1240000.pt