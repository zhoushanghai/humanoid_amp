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
  --motion /home/hz/g1/humanoid_amp/motions/custom_motion.npz

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
最终送入 Actor 网络的观测维度为： 单帧特征维度 (95 维) × 历史帧数 (5 帧) = 475 维。

其中，**每一帧（95 维）**的具体内容依次拼接如下：

dof_positions (29 维): 机器人的 29 个关节局部坐标角。
dof_velocities (29 维): 机器人的 29 个关节角速度。
projected_gravity (3 维): 【新增特征】将全局重力方向 $(0, 0, -1)$ 投影到机器人本体局部坐标系中，主要反映机器人的 Roll 和 Pitch 倾斜状态，忽略了全局 Yaw 朝向。
root_angular_velocities (3 维): 根节点（骨盆）的三维角速度。
last_actions (29 维): 【新增特征】网络在上一控制步输出的动作指令。
command_target_speed (2 维): 如果开启了速度奖励追踪，则拼接指令下发的机器人局部坐标系下的目标前向和侧向速度。
在具体环境步进和传输给网络时，环境会在每个 step 更新 self.obs_history_buffer（形状为 [num_envs, 5, 95]），提取最近 5 步的数据将其展平为一维（也就是上述对应的 475），再送给 policy。

这些特征获取全都在我们刚刚写的 g1_amp_env.py 的 compute_policy_obs 方法中。你可以检查或者跑一下看看实际效果如何！




拼接 12 条 walk 数据集
python -m humanoid_amp.train --task Isaac-G1-AMP-Deploy-Direct-v0 --headless

python -m humanoid_amp.play \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --num_envs 32 \
  --checkpoint /home/hz/g1/humanoid_amp/logs/skrl/g1_amp_dance/2026-02-21_23-33-40_ppo_torch/checkpoints/agent_1425000.pt
  walk1_subject1.npz: 15459 frames, 257.65 sec, fps=60
  walk1_subject2.npz: 15459 frames, 257.65 sec, fps=60
  walk1_subject5.npz: 15459 frames, 257.65 sec, fps=60
  walk2_subject1.npz: 14071 frames, 234.52 sec, fps=60
  walk2_subject3.npz: 14071 frames, 234.52 sec, fps=60
  walk2_subject4.npz: 14071 frames, 234.52 sec, fps=60
  walk3_subject1.npz: 14577 frames, 242.95 sec, fps=60
  walk3_subject2.npz: 14577 frames, 242.95 sec, fps=60
  walk3_subject3.npz: 14577 frames, 242.95 sec, fps=60
  walk3_subject4.npz: 14577 frames, 242.95 sec, fps=60
  walk3_subject5.npz: 14577 frames, 242.95 sec, fps=60
  walk4_subject1.npz: 9615 frames, 160.25 sec, fps=60