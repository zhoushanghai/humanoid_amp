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

python -m humanoid_amp.train --task Isaac-G1-AMP-Deploy-Direct-v0 --headless

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
- **[2026-02-23]** `git commit`: feat(play,env,cfg): 新增部署推理脚本并更新多历史帧环境配置 / add play_deploy.py, update env & cfg for multi-history obs
- **[2026-02-23]** `git commit`: docs(ablation): 记录实验②成功并准备实验③ / record success of phase ② and prep phase ③
- **[2026-02-23]** `git commit`: docs(ablation): 记录实验③成功并准备实验④ / record success of phase ③ and prep phase ④
- **[2026-02-23]** `git commit`: docs(ablation): 消融实验圆满完成，记录实验④结果 / ablation study completed, record phase ④ results

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
## Bug Fix & Root Cause Analysis

- **Date**: 2026-02-23
- **Action**: 诊断并修复 `num_actor_observations=2` 时训练失败（Policy Std 水平线）的根本原因。
- **Root Cause**: `agents/skrl_g1_deploy_amp_cfg.yaml` 中 `fixed_log_std: True`，Policy 的探索噪声 std 被永久固定在 `exp(-2.9) ≈ 0.055`，完全不随梯度更新。这就是 TensorBoard 中 `Policy / Standard Deviation` 曲线始终水平不变的原因。在 `n=1` 时任务简单尚能收敛；`n=2` 时任务难度加倍，固定的极小 std 使得 Policy 无法有效探索，梯度信号极弱，导致完全训不出来。
- **Fix**: 修改 `agents/skrl_g1_deploy_amp_cfg.yaml`：
    - `fixed_log_std: True` → `fixed_log_std: False`（允许 std 随训练自适应更新）
    - `initial_log_std: -2.9` → `initial_log_std: -1.0`（初始 std 从 0.055 提升至 0.37，提供合理的初始探索幅度）

## Experiment Record: Ablation Study - Phase ②
- **Date**: 2026-02-23 19:10
- **Model**: `logs/skrl/g1_amp_dance/2026-02-23_17-26-04_ppo_torch/checkpoints/agent_185000.pt`
- **Configuration**:
    - `num_actor_observations = 2`
    - `history_include_last_actions = False`
    - `history_include_command = False`
    - `observation_space = 173`
- **Phenomenon**: 策略成功训练 (Standard Deviation 正常波动，Reward 持续上升)。
- **Conclusion**: 机制验证通过，纯运动学历史信息（71维）可以正常训练。

## Experiment Record: Ablation Study - Phase ③
- **Date**: 2026-02-23 19:19
- **Model**: `logs/skrl/g1_amp_dance/2026-02-23_19-13-39_ppo_torch/checkpoints/agent_10000.pt`
- **Configuration**:
    - `num_actor_observations = 2`
    - `history_include_last_actions = True`
    - `history_include_command = False`
    - `observation_space = 202`
- **Phenomenon**: 策略训练正常 (Std 变动，Reward 上升)。
- **Conclusion**: `last_actions` 加入历史帧后依然稳定。

## Experiment Record: Ablation Study - Phase ④
- **Date**: 2026-02-23 19:31
- **Model**: `logs/skrl/g1_amp_dance/2026-02-23_19-24-57_ppo_torch/checkpoints/agent_10000.pt`
- **Configuration**:
    - `num_actor_observations = 2`
    - `history_include_last_actions = True`
    - `history_include_command = True`
    - `observation_space = 204`
- **Phenomenon**: 策略训练非常稳定 (Standard Deviation 动态更新，Reward 持续上升)。
- **Conclusion**: **消融实验圆满完成**。之前 204 维训练失败并非因为维度过高或 command 历史冲突，而是由于 **History Warm-Start Bug** 和 **Fixed Log Std** 配置导致的探索受阻。

## 核心修复总结 (Deep Explanation)
为什么现在 204 维能训出来了？

1. **历史帧预填充 (Warm-Start Fix)**:
    - 之前在 Episode 重置时，历史 buffer 会被清零。高维度的零向量输入会导致 `RunningStandardScaler` 的均值和方差被瞬间拉低，造成观测归一化异常。
    - 现在我们使用当前帧真实观测填充所有历史槽，消除了”零值污染”。
2. **解锁探索噪声 (Exploration Fix)**:
    - 之前的配置 `fixed_log_std: True` 锁死了 Policy 的标准差。
    - 现在设置为 `False` 并调大了初始 `std`，允许策略在 204 维的高维状态空间中进行充分探索。

## 2026-02-24 关键命令

- **2026-02-24** `git commit`: feat(env): 修改 Policy 观测为 Sim2Real 友好版本 / modify policy obs for Sim2Real compatibility
- **2026-02-24** `git commit`: fix(env): 修复侧向速度重采样导致斜走问题，添加 Walk+Run 混合训练 / fix lateral velocity sampling, add Walk+Run training

## Bug Fix

- **Date**: 2026-02-25
- **Action**: 修复自定义速度范围日志未出现在 TensorBoard 的问题。
- **Details**:
    - **根因**: `g1_amp_env.py` 中 `agent.track_data(...)` 依赖环境对象上的 `_skrl_agent` 句柄，但 `train.py`/`play.py` 在创建 `Runner` 后未将 `runner.agent` 回挂到环境，导致 `Reward / cmd_lin_vel_*` 与 `Reward / cmd_ang_vel_z_*` 没有被写入。
    - **修改文件**:
        - `train.py`: 在 `runner = Runner(env, agent_cfg)` 后新增 `env.unwrapped._skrl_agent = runner.agent`（带容错）。
        - `play.py`: 同步新增 `env.unwrapped._skrl_agent = runner.agent`（带容错）。
    - **影响**:
        - 训练时可在 TensorBoard 中看到以下标量：
          `Reward / cmd_lin_vel_x_min`,
          `Reward / cmd_lin_vel_x_max`,
          `Reward / cmd_lin_vel_y_min`,
          `Reward / cmd_lin_vel_y_max`,
          `Reward / cmd_ang_vel_z_min`,
          `Reward / cmd_ang_vel_z_max`。

## 2026-02-25 关键命令

```bash
python -m humanoid_amp.train \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --headless
```

- **[2026-02-25]** `git commit`: fix(log): 修复课程日志写入分组 / fix curriculum logging groups

## Bug Fix

- **Date**: 2026-02-25
- **Action**: 将速度课程范围指标从 Reward 面板拆分到独立面板。
- **Details**:
    - **文件**: `g1_amp_env.py`
    - 将 `cmd_*` 指标的 TensorBoard 前缀由 `Reward /` 改为 `Curriculum /`。
    - 其他奖励相关指标保持在 `Reward /` 下不变。
- **Effect**:
    - 新面板: `Curriculum / cmd_lin_vel_x_min`, `Curriculum / cmd_lin_vel_x_max`, `Curriculum / cmd_lin_vel_y_min`, `Curriculum / cmd_lin_vel_y_max`, `Curriculum / cmd_ang_vel_z_min`, `Curriculum / cmd_ang_vel_z_max`。
