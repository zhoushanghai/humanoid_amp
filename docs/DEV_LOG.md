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
- **[2026-02-28]** `git commit`: feat(play): 增加固定速度配置 / add fixed speed config

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

- **[2026-02-26]** `git commit`: feat(env): z轴课程独立触发 / split z-axis curriculum trigger

## Bug Fix

- **Date**: 2026-02-26
- **Action**: 移除 Curriculum 面板中的冗余通用指标，仅保留分轴版本。
- **Details**:
    - **文件**: `g1_amp_env.py`
    - 删除通用兼容日志键：
      `curriculum_avg_track_rew`、
      `curriculum_margin`、
      `curriculum_threshold`、
      `curriculum_triggered`。

## Documentation Update

- **Date**: 2026-03-05
- **Action**: 新增跨项目可复用的速度跟踪测试执行文档。
- **Details**:
    - **文件**: `docs/vel_tracking_test_protocol.md`
    - 从 `docs/vel_tracking_metrics.md` 提取测试流程、命令集合、指标公式、聚合口径与 CSV 字段。
    - 新增执行检查清单，便于在不同项目中落地同一评测标准。
- **Execution Record**:
    - 无命令执行（本次为文档编写与规则整理）。
    - 保留并继续记录分轴指标：
      `curriculum_avg_track_rew_xy`、`curriculum_avg_track_rew_z`、
      `curriculum_margin_xy`、`curriculum_margin_z`、
      `curriculum_threshold_xy`、`curriculum_threshold_z`、
      `curriculum_triggered_xy`、`curriculum_triggered_z`。
    - 同步清理对应的冗余内部状态变量，避免死代码。
- **Execution Record**:
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

## Bug Fix

- **Date**: 2026-02-25
- **Action**: 新增课程触发判据可视化指标，便于定位“为什么没加难度”。
- **Details**:
    - **文件**: `g1_amp_env.py`
    - 新增 `Curriculum / curriculum_avg_track_rew`、`Curriculum / curriculum_threshold`、`Curriculum / curriculum_margin`、`Curriculum / curriculum_triggered`。
    - 其中 `curriculum_triggered=1` 表示该次 reset 判定触发了难度升级，`0` 表示未触发。
    - `cmd_*` 与 `curriculum_*` 统一放入 TensorBoard 的 `Curriculum /` 分组，和 `Reward /` 分开显示。

## 2026-02-25 关键命令

```bash
python -m humanoid_amp.train \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --headless
```

## Bug Fix

- **Date**: 2026-02-25
- **Action**: 修复 `Curriculum / curriculum_threshold` 显示为 NaN 的问题。
- **Details**:
    - **文件**: `g1_amp_env.py`
    - 将课程统计默认值从 `NaN` 改为数值初始化：
      `curriculum_avg_track_rew=0.0`，`curriculum_threshold=rew_track_vel * threshold_ratio`。
    - 在每步日志中直接按配置实时计算并写入 `curriculum_threshold`，避免因未触发 timeout 判定而保持 NaN。
    - `curriculum_margin` 改为使用实时阈值计算，确保曲线始终可读。
- **[2026-02-25]** `git commit`: fix(env): 修复课程日志NaN / fix NaN curriculum logs
- **[2026-03-03]** `git commit`: feat(plot): 新增课程命令范围绘图 / add curriculum command range plot

## Feature Update

- **Date**: 2026-02-26
- **Action**: 为 Deploy 任务新增 z 轴旋转命令跟踪与独立旋转奖励，并沿用同一 curriculum 升难逻辑。
- **Details**:
    - **文件**: `g1_amp_env_cfg.py`
    - 新增奖励权重配置 `rew_track_ang_vel_z`（默认 0.0）。
    - `G1AmpDeployEnvCfg` 中启用旋转命令范围：`command_ang_vel_z_range = (-0.2, 0.2)`。
    - `G1AmpDeployEnvCfg` 中显式设置课程阈值比例：`track_vel_curriculum_threshold_ratio = 0.8`。
    - `G1AmpDeployEnvCfg` 中设置旋转跟踪奖励权重：`rew_track_ang_vel_z = 1.0`。
    - **文件**: `g1_amp_env.py`
    - 将命令跟踪奖励拆分为两项：
      `rew_track_vel`（线速度 vx, vy）与 `rew_track_ang_vel_z`（角速度 wz）。
    - 总跟踪奖励改为 `rew_track_cmd_total = rew_track_vel + rew_track_ang_vel_z`，并用于总奖励与 curriculum 判据统计。
    - curriculum 阈值改为 `(rew_track_vel + rew_track_ang_vel_z) * track_vel_curriculum_threshold_ratio`。
    - 速度命令重采样逻辑增加 z 轴范围判定：即使 x/y 固定，也可单独采样 z 轴旋转命令。
    - 新增 TensorBoard 指标：
      `Reward / rew_track_ang_vel_z`、`Reward / error_track_ang_vel_z`。
- **Execution Record**:
```bash
python -m humanoid_amp.train \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --headless
```

```bash
python -m humanoid_amp.play \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --checkpoint logs/skrl/<run>/checkpoints/agent_<step>.pt \
  --num_envs 1 \
  --video \
  --video_length 300
```

## Bug Fix

- **Date**: 2026-02-26
- **Action**: 将 curriculum 触发逻辑改为 xy 与 z 轴独立判定、独立扩难度。
- **Details**:
    - **文件**: `g1_amp_env.py`
    - 将课程统计从单一总和拆分为：
      `_episode_track_vel_sum`（xy）与 `_episode_track_ang_vel_z_sum`（z）。
    - 触发逻辑改为双通道：
      - `xy` 达标（`avg_track_rew_xy > rew_track_vel * 0.8`）仅扩展 `command_lin_vel_x/y_range`。
      - `z` 达标（`avg_track_rew_z > rew_track_ang_vel_z * 0.8`）仅扩展 `command_ang_vel_z_range`。
    - 保留总开关 `curriculum_triggered`（任一通道触发即为 1），并新增独立日志：
      `curriculum_triggered_xy`、`curriculum_triggered_z`、
      `curriculum_threshold_xy`、`curriculum_threshold_z`、
      `curriculum_avg_track_rew_xy`、`curriculum_avg_track_rew_z`。
    - 为兼容旧面板，`curriculum_threshold` / `curriculum_avg_track_rew` / `curriculum_margin` 现在默认对齐 `xy` 通道，不再使用 xy+z 合并阈值。
- **Execution Record**:
```bash
python -m humanoid_amp.train \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --headless
```

## Feature Update

- **Date**: 2026-02-28
- **Action**: 为 `play` 增加“按参数文件固定 32 环境速度命令”的能力。
- **Details**:
    - **文件**: `play.py`
    - 新增参数 `--speed_config`，读取 JSON 配置并校验条目数必须等于 `num_envs`。
    - 支持三种条目格式：数字（`vx`）、列表（`[vx, vy, wz]`）和字典（`{"vx":...,"vy":...,"wz":...}`）。
    - 在建环境后调用环境接口注入固定速度，避免随机命令覆盖。
    - **文件**: `g1_amp_env.py`
    - 新增 `set_fixed_command_targets()` / `clear_fixed_command_targets()`。
    - 在 `_pre_physics_step()` 与 `_reset_idx()` 中增加固定命令覆盖逻辑，确保重置后仍保持指定速度。
    - **文件**: `configs/play_speed_32.example.json`
    - 新增 32 环境速度示例配置文件。
    - **文件**: `README.md`
    - 新增 `--speed_config` 用法说明与示例命令。
- **Execution Record**:
```bash
python -m humanoid_amp.play \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --num_envs 32 \
  --checkpoint logs/skrl/<run>/checkpoints/agent_<step>.pt \
  --speed_config configs/play_speed_32.example.json \
  --video \
  --video_length 300
```

## Feature Update

- **Date**: 2026-02-28
- **Action**: 将 `play` 速度配置 JSON 改为更直观的“每个 env 显式一行”写法。
- **Details**:
    - **文件**: `configs/play_speed_32.example.json`
    - 将原先按数组索引隐式对应 env 的写法，改为显式映射：`"env_0"` ~ `"env_31"`。
    - 每个 env 在同一行内声明：`{"vx": ..., "vy": ..., "wz": ...}`。
    - **文件**: `play.py`
    - `--speed_config` 解析逻辑新增对 dict 映射格式的支持：`{"commands": {"env_0": {...}}}`。
    - 继续兼容旧数组格式（索引 = env id）。
    - **文件**: `README.md`
    - 更新 speed config 说明，推荐显式 env 映射格式。
- **Execution Record**:
```bash
python -m humanoid_amp.play \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --num_envs 32 \
  --checkpoint logs/skrl/<run>/checkpoints/agent_<step>.pt \
  --speed_config configs/play_speed_32.example.json \
  --video \
  --video_length 300
```

## Config Update

- **Date**: 2026-02-28
- **Action**: 按 Deploy 任务最大课程范围重设 `play` 的 32 环境速度配置。
- **Details**:
    - **文件**: `configs/play_speed_32.example.json`
    - 依据 `g1_amp_env_cfg.py` 中 `G1AmpDeployEnvCfg` 的课程上限范围：
      `command_lin_vel_x_curriculum_limit_range = (-1.0, 5.0)`，
      `command_lin_vel_y_curriculum_limit_range = (-2.0, 2.0)`，
      `command_ang_vel_z_curriculum_limit_range = (-1.0, 1.0)`。
    - 将 32 个 `env_x` 显式映射到全区间覆盖：
      - `vx` 从 `-1.00` 到 `5.00` 递增分布；
      - `vy` 分 4 段覆盖 `-2.00 / -0.67 / 0.67 / 2.00`；
      - `wz` 循环覆盖 `-1.00 / -0.33 / 0.33 / 1.00`。
- **Execution Record**:
```bash
python -m humanoid_amp.play \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --num_envs 32 \
  --checkpoint logs/skrl/<run>/checkpoints/agent_<step>.pt \
  --speed_config configs/play_speed_32.example.json \
  --video \
  --video_length 300
```

## Feature Update

- **Date**: 2026-02-28
- **Action**: 将速度配置接入 `play_deploy.py`，默认自动透传给 `play.py`。
- **Details**:
    - **文件**: `play_deploy.py`
    - 在项目配置区新增 `SPEED_CONFIG = "configs/play_speed_32.example.json"`。
    - 构建 `humanoid_amp.play` 命令时，若该文件存在则自动追加：
      `--speed_config configs/play_speed_32.example.json`。
    - 若文件不存在，输出 warning 并继续执行，不阻塞 play。
    - 更新文件头部说明，明确脚本默认会尝试附加 `--speed_config`。
- **Execution Record**:
```bash
python play_deploy.py
```

```bash
python play_deploy.py logs/skrl/<run>/checkpoints/agent_<step>.pt
```

## Bug Fix

- **Date**: 2026-02-28
- **Action**: 修复 Deploy 训练中 z 轴角速度 curriculum 不易触发的问题（与 xy 轴解耦）。
- **Details**:
    - **文件**: `g1_amp_env.py`
    - 将 curriculum 阈值由“xy/z 共用 `track_vel_curriculum_threshold_ratio`”改为分轴读取：
      - xy: `track_vel_curriculum_threshold_ratio`
      - z: `track_ang_vel_curriculum_threshold_ratio`（缺省回退到 xy）
    - 将 curriculum 扩展步长由“统一 `track_vel_curriculum_delta`”改为支持 z 轴独立步长：
      - xy: `track_vel_curriculum_delta`
      - z: `track_ang_vel_curriculum_delta`（缺省回退到 xy）
    - 同步更新 `curriculum_threshold_z` 日志计算，避免日志与实际触发判据不一致。
    - **文件**: `g1_amp_env_cfg.py`
    - 新增可配置项：
      - `track_ang_vel_curriculum_threshold_ratio`
      - `track_ang_vel_curriculum_delta`
    - 在 `G1AmpDeployEnvCfg` 中设置
      `track_ang_vel_curriculum_threshold_ratio = 0.65`，
      让 z 轴在早期训练更容易触发课程扩展。
- **Execution Record**:
```bash
python -m humanoid_amp.train \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --headless
```

## Storage Check

- **Date**: 2026-03-03
- **Action**: 排查各用户目录空间占用，定位磁盘大户。
- **Details**:
    - 统计 `/home` 下用户目录占用并按大小排序。
    - 结果显示当前可统计范围内 `hz` 目录占用最大（约 `490G`）。
    - 尝试使用 `sudo` 获取完整用户占用失败（需要密码），因此其他用户当前仅统计到目录层级大小。
- **Execution Record**:
```bash
getent passwd | awk -F: '{print $1":"$6":"$7}'
ls -la /home
du -sh /home/* 2>/tmp/du_home_err.log | sort -h
cat /tmp/du_home_err.log
sudo -n du -sh /home/* | sort -h
```

## Feature Update

- **Date**: 2026-03-03
- **Action**: 新增 Curriculum 命令范围可视化脚本，并在指定 run 上生成图表。
- **Details**:
    - **文件**: `scripts/plot_curriculum_cmd.py`
    - 新增 tfevents 解析与绘图入口，读取以下 3 组 `Curriculum /` 指标：
      - `cmd_lin_vel_x_min/max`
      - `cmd_lin_vel_y_min/max`
      - `cmd_ang_vel_z_min/max`
    - 图形输出为 3 个子图（x/y/z），每个子图同时绘制：
      - `min` 边界线
      - `max` 边界线
      - `min~max` 区间带（fill band）
    - 默认输出目录：`<event_file_parent>/charts/`，默认文件名 `curriculum_cmd_ranges.png`。
    - 在 `g1_amp` conda 环境验证通过，并生成：
      `logs/skrl/g1_amp_dance/2026-02-28_01-40-17_ppo_torch/charts/curriculum_cmd_ranges.png`。
- **Execution Record**:
```bash
conda run -n g1_amp python scripts/plot_curriculum_cmd.py \
  --event_file logs/skrl/g1_amp_dance/2026-02-28_01-40-17_ppo_torch/events.out.tfevents.1772214025.rbm.2498113.0
```

## Feature Update

- **Date**: 2026-03-03
- **Action**: 为课程范围图新增可指定 iteration 区间的能力，并限制到 `0-3000000` 重绘。
- **Details**:
    - **文件**: `scripts/plot_curriculum_cmd.py`
    - 新增命令行参数：
      - `--step_min`
      - `--step_max`
    - 在绘图阶段通过 `set_xlim` 对 x 轴进行区间约束，可按指定 iteration 范围查看课程变化。
    - 按需求用 `--step_min 0 --step_max 3000000` 重新生成图：
      `logs/skrl/g1_amp_dance/2026-02-28_01-40-17_ppo_torch/charts/curriculum_cmd_ranges.png`。
- **Execution Record**:
```bash
conda run -n g1_amp python scripts/plot_curriculum_cmd.py \
  --event_file logs/skrl/g1_amp_dance/2026-02-28_01-40-17_ppo_torch/events.out.tfevents.1772214025.rbm.2498113.0 \
  --step_min 0 \
  --step_max 3000000
```

## Style Update

- **Date**: 2026-03-03
- **Action**: 优化 Curriculum 范围图配色，改为三类指标三种主题色。
- **Details**:
    - **文件**: `scripts/plot_curriculum_cmd.py`
    - 新增 `METRIC_THEME_COLORS`，为 3 个指标设置独立色板：
      - `cmd_lin_vel_x`: Ocean Blue
      - `cmd_lin_vel_y`: Sunset Orange
      - `cmd_ang_vel_z`: Forest Green
    - 每个子图内部保持同主题分层：
      - `min` 边界线
      - `max` 边界线
      - `range` 区间带
    - 子图标题增加主题名，便于快速识别不同指标。
    - 已按 `0-3000000` 区间重新生成图：
      `logs/skrl/g1_amp_dance/2026-02-28_01-40-17_ppo_torch/charts/curriculum_cmd_ranges.png`。
- **Execution Record**:
```bash
conda run -n g1_amp python scripts/plot_curriculum_cmd.py \
  --event_file logs/skrl/g1_amp_dance/2026-02-28_01-40-17_ppo_torch/events.out.tfevents.1772214025.rbm.2498113.0 \
  --step_min 0 \
  --step_max 3000000
```

## Style Update

- **Date**: 2026-03-03
- **Action**: 调整上下限颜色语义与图例顺序，统一为 `max -> min -> range`。
- **Details**:
    - **文件**: `scripts/plot_curriculum_cmd.py`
    - 颜色语义调整为：`max` 使用深色，`min` 使用浅色（三组主题色均同步）。
    - 绘图顺序改为先绘制 `max` 再绘制 `min`，与语义保持一致。
    - 每个子图图例显式重排为：`max`、`min`、`range`，保证和图形说明一致。
    - 重新生成图：
      `logs/skrl/g1_amp_dance/2026-02-28_01-40-17_ppo_torch/charts/curriculum_cmd_ranges.png`。
- **Execution Record**:
```bash
conda run -n g1_amp python scripts/plot_curriculum_cmd.py \
  --event_file logs/skrl/g1_amp_dance/2026-02-28_01-40-17_ppo_torch/events.out.tfevents.1772214025.rbm.2498113.0 \
  --step_min 0 \
  --step_max 3000000
```

## Style Update

- **Date**: 2026-03-03
- **Action**: 统一三个子图图例位置，固定为与中间图一致的右侧居中。
- **Details**:
    - **文件**: `scripts/plot_curriculum_cmd.py`
    - 将图例位置由自适应 `loc="best"` 改为固定 `loc="center right"`。
    - 三个子图的图例现在位置一致，均为右侧居中，同时保持顺序 `max -> min -> range`。
    - 已重新生成图：
      `logs/skrl/g1_amp_dance/2026-02-28_01-40-17_ppo_torch/charts/curriculum_cmd_ranges.png`。
- **Execution Record**:
```bash
conda run -n g1_amp python scripts/plot_curriculum_cmd.py \
  --event_file logs/skrl/g1_amp_dance/2026-02-28_01-40-17_ppo_torch/events.out.tfevents.1772214025.rbm.2498113.0 \
  --step_min 0 \
  --step_max 3000000
```

## Documentation Update

- **Date**: 2026-03-03
- **Action**: 精简 `README.md`，仅保留安装、训练、Play 三部分。
- **Details**:
    - 将 README 重构为三个最小必要章节：`安装`、`训练`、`Play`。
    - 统一命令展示格式，训练与 Play 使用多行参数，便于直接复制与修改。
    - 移除与核心上手流程无关的冗余表述。
- **Execution Record**:

```bash
pip install -e .

python -m humanoid_amp.train \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --headless

python -m humanoid_amp.play \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --num_envs 32 \
  --checkpoint logs/skrl/g1_amp_dance/2026-02-22_11-40-31_ppo_torch/checkpoints/agent_50000.pt
```

## 关键命令

- **[2026-03-03]** `git commit`: docs(readme): 精简安装训练Play说明 / simplify install train play docs

## Documentation Update

- **Date**: 2026-03-05
- **Action**: 新增 Velocity Tracking 全量正式评测计划文档（去除冒烟阶段）。
- **Details**:
    - **文件**: `docs/plan_velocity_tracking_eval.md`
    - 明确“最终目标=一次全量运行产出完整指标”，不包含冒烟测试。
    - 约束 checkpoint 为单点配置（`configs/eval_velocity_tracking.yaml` 的 `active_checkpoint`）。
    - 固化输出字段、评测口径、命令集合与验收标准。
- **Execution Record**:
    - 无运行命令（本次仅文档落地与流程规范化）。

## Evaluation Tooling

- **Date**: 2026-03-05
- **Action**: 新增 Velocity Tracking 全量评测配置与执行脚本。
- **Details**:
    - **文件**: `configs/eval_velocity_tracking.yaml`
      - 新增评测统一配置，`active_checkpoint` 作为 checkpoint 单点修改入口。
      - 固定评测关键参数：`num_envs=64`、`ramp/settle/record`、`max_vx/max_vy` 通过阈值。
    - **文件**: `scripts/eval/eval_vel_tracking_protocol.py`
      - 新增全量评测脚本，覆盖 `low_lin/high_lin/yaw_low/yaw_high/max_vx/max_vy/step_survival`。
      - 输出 `metrics_summary.csv`、`metrics_combo_details.csv`、`run_meta.json`。
      - 新增无 PyYAML 依赖的配置读取 fallback（支持简单 `key: value` 解析）。
      - 修复 done mask 维度问题（统一 reshape 为 1D）以避免索引错误。
- **Execution Record**:
```bash
python -m py_compile scripts/eval/eval_vel_tracking_protocol.py
```

## Evaluation Run / Troubleshooting

- **Date**: 2026-03-05
- **Action**: 启动 Velocity Tracking 全量正式评测并完成运行链路排障。
- **Details**:
    - 初次运行报错：缺少 `yaml` 模块；通过脚本 fallback 解析修复。
    - 运行报错：`--headless`、`--device` 与 `AppLauncher` 参数冲突；已移除重复参数定义。
    - 运行报错：`/tmp/isaaclab/logs` 无写权限；通过设置 `TMPDIR=/home/hz/g1/humanoid_amp/tmp` 规避。
    - 运行报错：`IndexError: too many indices`；修复 done mask 形状后重新启动。
- **Execution Record**:
```bash
TMPDIR=/home/hz/g1/humanoid_amp/tmp \
conda run -n g1_amp python scripts/eval/eval_vel_tracking_protocol.py \
  --config configs/eval_velocity_tracking.yaml \
  --headless
```

## Evaluation Stability Fix

- **Date**: 2026-03-05
- **Action**: 降低评测日志噪声并后台启动全量评测任务。
- **Details**:
    - **文件**: `scripts/eval/eval_vel_tracking_protocol.py`
    - 将 `quat_rotate_inverse` 替换为 `quat_apply_inverse`，避免 IsaacLab 弃用告警高频刷屏。
    - 重新启动正式评测任务并写入运行日志文件。
- **Execution Record**:
```bash
nohup env TMPDIR=/home/hz/g1/humanoid_amp/tmp \
  conda run -n g1_amp python scripts/eval/eval_vel_tracking_protocol.py \
  --config configs/eval_velocity_tracking.yaml \
  --headless > outputs/vel_tracking/eval_run.log 2>&1 &
```

## Configuration Change

- **Date**: 2026-03-05
- **Action**: 将 IsaacLab 默认日志临时目录从系统 `/tmp/isaaclab` 切换到项目本地可写目录。
- **Details**:
    - **文件**: `train.py`, `play.py`, `scripts/eval/eval_vel_tracking_protocol.py`
    - 新增 `_configure_tmpdir()`，默认设置 `TMPDIR=./tmp`（仅当外部未显式设置 `TMPDIR` 时生效）。
    - 启动前自动创建本地 `tmp/` 目录，避免系统 `/tmp/isaaclab/logs` 权限问题。
    - **文件**: `.gitignore`
      - 新增 `tmp/`，避免运行时临时文件进入版本控制。
- **Execution Record**:
```bash
python -m py_compile play.py train.py scripts/eval/eval_vel_tracking_protocol.py
```

## Environment Configuration

- **Date**: 2026-03-05
- **Action**: 切换为全局 TMPDIR 方案，统一 IsaacLab 临时日志目录到用户目录。
- **Details**:
    - 用户选择全局方案：`TMPDIR=$HOME/isaaclab/tmp`。
    - 已写入 `~/.bashrc`：
      - `export TMPDIR="$HOME/isaaclab/tmp"`
      - `mkdir -p "$TMPDIR"`
    - 已创建目录：`~/isaaclab/tmp`。
    - 回退项目内临时兜底改动，移除以下文件中的 `_configure_tmpdir()` 逻辑，避免重复策略：
      - `train.py`
      - `play.py`
      - `scripts/eval/eval_vel_tracking_protocol.py`
- **Execution Record**:
```bash
mkdir -p "$HOME/isaaclab/tmp"
```

## Environment Configuration

- **Date**: 2026-03-05
- **Action**: 修复 `conda run -n g1_amp` 未继承 `~/.bashrc` 导致 TMPDIR 失效的问题。
- **Details**:
    - 通过 `conda env config vars` 在 `g1_amp` 环境内持久设置：
      `TMPDIR=/home/hz/isaaclab/tmp`。
    - 验证 `conda run -n g1_amp python3 -c ...` 输出的 `tempfile.gettempdir()` 已切换到 `/home/hz/isaaclab/tmp`。
- **Execution Record**:
```bash
conda env config vars set TMPDIR=/home/hz/isaaclab/tmp -n g1_amp
conda env config vars list -n g1_amp
```

## Environment Configuration

- **Date**: 2026-03-05
- **Action**: 在 fish shell 中持久设置 TMPDIR 到用户目录。
- **Details**:
    - 执行 `set -Ux TMPDIR ~/isaaclab/tmp`，将 TMPDIR 设为 fish 全局持久变量。
    - 验证 `fish -lc 'python3 -c ...'` 输出 `tempfile.gettempdir()` 为 `/home/hz/isaaclab/tmp`。
- **Execution Record**:
```bash
mkdir -p /home/hz/isaaclab/tmp
fish -lc 'set -Ux TMPDIR ~/isaaclab/tmp'
```

## Skill / Environment Update

- **Date**: 2026-03-05
- **Action**: 统一 fish 配置并更新 `new-machine-setup` Skill 的 fish 同步规则。
- **Details**:
    - **系统配置**: 在 `~/.config/fish/config.fish` 追加 IsaacLab TMPDIR 配置：
      - `set -gx TMPDIR ~/isaaclab/tmp`
      - `mkdir -p ~/isaaclab/tmp`
    - **Skill 文件**: `/home/hz/.codex/skills/new-machine-setup/SKILL.md`
      - Step 4 增加“先询问是否安装/使用 fish shell”规则。
      - 若使用 fish，要求同步更新 `~/.bashrc` 与 `~/.config/fish/config.fish`。
      - 预做列表新增 fish 选项，完成态说明同步 fish 配置。
- **Execution Record**:
```bash
fish -lc 'echo TMPDIR=$TMPDIR; python3 -c "import tempfile; print(tempfile.gettempdir())"'
```

## Evaluation Visualization

- **Date**: 2026-03-05
- **Action**: 为 Velocity Tracking 评测新增图表输出，便于快速观察效果。
- **Details**:
    - **文件**: `scripts/eval/eval_vel_tracking_protocol.py`
    - 在原有 `metrics_summary.csv`、`metrics_combo_details.csv`、`run_meta.json` 基础上新增自动绘图：
      - `combo_survival.png`：每个 combo 的生存环境数量
      - `combo_lin_acc.png`：有效 combo 的线速度准确率
      - `summary_metrics.png`：关键汇总指标柱状图
      - `step_alive_rate.png`：step-response 各 phase 存活率曲线
- **Execution Record**:
```bash
python -m py_compile scripts/eval/eval_vel_tracking_protocol.py
```

## Experiment Record: Velocity Tracking Re-run (num_envs=100)

- **Date**: 2026-03-05
- **Model**: `logs/skrl/g1_amp_dance/2026-02-28_01-40-17_ppo_torch/checkpoints/agent_5555000.pt`
- **Configuration**:
    - `num_envs = 100`（`configs/eval_velocity_tracking.yaml`）
    - `task = Isaac-G1-AMP-Deploy-Direct-v0`
    - `seed = 42`
- **Phenomenon**:
    - 评测完成并生成完整结果与图表目录：`outputs/vel_tracking/agent_5555000_2026-03-05_15-43-36`。
    - 汇总指标：
      - `low_lin_lin_acc = 74.6403`
      - `high_lin_lin_acc = 88.0658`
      - `yaw_low_yaw_acc = 0.2735`
      - `yaw_high_yaw_acc = NaN`
      - `max_vx = 0.0`
      - `max_vy = 1.44`
      - `step_survival = 0.0`
- **Conclusion**:
    - 增加并行数量后整体结论未改变：高难度/高速项仍以失败为主，step 生存率为 0。
- **Execution Record**:
```bash
conda run -n g1_amp python scripts/eval/eval_vel_tracking_protocol.py \
  --config configs/eval_velocity_tracking.yaml \
  --headless
```
