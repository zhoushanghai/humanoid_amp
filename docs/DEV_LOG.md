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



python -m humanoid_amp.play \
--task Isaac-G1-AMP-Custom-Direct-v0 \
--num_envs 32 \
--checkpoint logs/skrl/g1_amp_dance/2026-02-14_04-58-01_ppo_torch/checkpoints/agent_450000.pt


拼接 12 条 walk 数据集


python -m humanoid_amp.play \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --num_envs 32 \
  --checkpoint logs/skrl/g1_amp_dance/2026-02-22_11-40-31_ppo_torch/checkpoints/agent_50000.pt

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

  --- 
  2026/02/22
  Actor的输入加上last_actions (29)	然后加上了历史
  训不出来

  先不加上lastaction 呢？
  好像也训不出来

  测试只加last action，但不加历史
可以，但是他会跳起来
不过最终还是会像走路
最后的效果让他训一晚上吧
效果好像还可以,他开始交叉腿走了

效果都是差不多的，还是可以走路，但是不能学会交叉腿。


加上历史帧数，但只有两针，他就学不会了
## 关键命令

- **[2026-02-23]** `git commit`: docs(docs): update DEV_LOG with training command and experiment results / 更新开发日志，新增训练命令及实验结果
- **[2026-02-23]** `git commit`: feat(env,cfg): add last_actions to policy history for 2-frame input / 策略历史输入中增加上一步动作感知

## 工具优化

- **Date**: 2026-02-23
- **Action**: 优化 `play_deploy.sh` 与新增辅助脚本 `scripts/find_latest_checkpoint.py`。
- **Details**:
    - `play_deploy.sh`: 添加自动搜索逻辑，无需参数即可运行；配置集中在头部 `LOG_BASE`、`TASK`、`NUM_ENVS`。
    - `scripts/find_latest_checkpoint.py`: 自动搜索最新日志目录（按文件名排序 = 时间排序）和最大 Step的 Checkpoint。
    - 项目专属配置（`CHECKPOINT_SUBDIR` / `CHECKPOINT_PATTERN` / `STEP_REGEX`）集中在脚本顶部，方便移植到其他项目。
- **运行命令**:

```bash
./play_deploy.sh               # 自动搜索最新 checkpoint
./play_deploy.sh <ckpt_path>   # 手动指定
```

## Configuration Change

- **Date**: 2026-02-23
- **Action**: 将 `last_actions` 加入 Actor 的历史观测输入中。
- **Details**: 
    - **文件**: `g1_amp_env_cfg.py`
        - 修改 `G1AmpDeployEnvCfg` 的 `observation_space` 为 `204` (计算: $(71 + 29 + 2) \times 2 = 204$)。
    - **文件**: `g1_amp_env.py`
        - 修改 `__init__` 以在 `self.actor_obs_per_frame` 中包含 `last_action_size`。
        - 修改 `_get_observations`，在堆叠历史帧时将 `self.last_actions` 包含在 `per_frame_parts` 中。
- **Purpose**: 为策略提供过去动作的感知，有助于提高控制的平滑性和动态响应能力。

## Bug Fix

- **Date**: 2026-02-23
- **Action**: 修复历史 Buffer 在 Episode 重置时被全部清零的问题（零值污染 RunningStandardScaler）。
- **Details**:
    - **文件**: `g1_amp_env.py`
        - `__init__`：新增 `_just_reset_mask` 布尔张量，用于标记刚被重置的环境。
        - `_reset_idx`：不再直接将 `actor_obs_history_buffer` 归零，改为设置 `_just_reset_mask`。
        - `_get_observations`：在正常 shift 前，对 `_just_reset_mask` 为 True 的环境，
          将所有历史槽用当前帧的真实观测值预填充（Warm-Start），消除异常零值输入。
- **Root Cause**: Episode 开始时历史帧全为零，与真实观测量级差异极大，
  导致归一化统计被污染，策略无法从历史帧中学到有意义的信息。
