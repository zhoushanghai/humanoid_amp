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

- **[2026-02-26]** `git commit`: feat(env): 接入地形课程学习 / add terrain curriculum

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

## Feature Adaptation

- **Date**: 2026-02-26 21:33:32 CST
- **Action**: 为 `Isaac-G1-AMP-Deploy-Direct-v0` 适配 Direct workflow 的 terrain curriculum，并将指标记录到 TensorBoard。
- **Details**:
    - **文件**: `g1_amp_env_cfg.py`
        - 新增 `terrain` 配置（`TerrainImporterCfg`）到 `G1AmpEnvCfg` 与 `G1AmpEnvCfg_CUSTOM`。
        - 新增 terrain curriculum 参数：
          `enable_terrain_curriculum`、
          `terrain_curriculum_threshold_ratio`、
          `terrain_curriculum_move_down_on_fall`。
        - `G1AmpDeployEnvCfg` 启用 terrain curriculum，并切换为 `terrain_type="generator"`（`ROUGH_TERRAINS_CFG`），设置 `max_init_terrain_level=1`。
        - 将 Deploy 的 `motion_file` 从绝对路径改为 `os.path.join(MOTIONS_DIR, "motion_config.yaml")`，提升可迁移性。
    - **文件**: `g1_amp_env.py`
        - `_setup_scene` 改为通过 `self.cfg.terrain.class_type(self.cfg.terrain)` 创建 terrain importer，不再固定 spawn ground plane。
        - reset 原点统一改用 `self._terrain.env_origins`。
        - 在 `_reset_idx` 新增 `_update_terrain_curriculum(...)` 调用：
          - timeout 且速度跟踪平均奖励超过阈值时 `move_up`；
          - 非 timeout（跌倒/提前终止）时按配置 `move_down`；
          - 通过 `self._terrain.update_env_origins(...)` 更新 terrain levels 与 env origins。
        - 新增 terrain 相关日志键并写入 `extras["log"]` 与 `agent.track_data(...)`：
          `terrain_level_mean/min/max`、
          `terrain_curriculum_avg_track_rew_xy`、
          `terrain_curriculum_threshold_xy`、
          `terrain_curriculum_move_up_count`、
          `terrain_curriculum_move_down_count`、
          `terrain_curriculum_timeout_ratio`。

## Execution Record

```bash
python -m py_compile \
  g1_amp_env.py \
  g1_amp_env_cfg.py

python -m humanoid_amp.train \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --headless
```

## Error Handling

- `ImportError: attempted relative import with no known parent package`
  - 场景：直接执行 `python - <<...` 导入 `g1_amp_env_cfg.py`。
  - 处理：改为包上下文导入（`from humanoid_amp...`）或使用模块方式运行。
- `ModuleNotFoundError: No module named 'gymnasium'`
  - 场景：当前终端 Python 环境缺少训练依赖，无法在本机直接做运行时实例化验证。
  - 处理：已完成语法级校验；运行级验证需在项目训练环境中执行上述 train 命令。

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

- **Date**: 2026-02-27
- **Action**: 为 `play_deploy.py` 增加“headless 推理并保存视频”快捷模式。
- **Details**:
    - **文件**: `play_deploy.py`
    - 新增参数解析函数 `parse_args()`，统一处理位置参数与功能开关。
    - 新增 `--headless-video`（等价于同时传递 `--headless` 与 `--video`）。
    - 新增 `--video_length`（默认 `300`），用于控制录制步数。
    - 保持原有用法兼容：
      `python play_deploy.py` 与 `python play_deploy.py <checkpoint>` 仍可直接使用。
- **Execution Record**:
```bash
python play_deploy.py --help

python -m py_compile \
  play_deploy.py

python play_deploy.py \
  --headless-video \
  --video_length 300
```

## Bug Fix

- **Date**: 2026-02-27
- **Action**: 修复 `headless + video` 录屏中镜头未拍到机器人问题。
- **Details**:
    - **文件**: `play.py`
    - 在视频模式下新增相机跟随逻辑：优先将 viewport 相机对准 `env_0` 机器人参考刚体，并在仿真循环中持续更新视角。
    - 当环境不支持该能力时打印警告并回退到默认相机，避免中断 play 流程。
    - **文件**: `play_deploy.py`
    - 新增 `--num_envs` 参数用于手动覆盖环境数量。
    - 在视频模式（`--video`/`--headless-video`）且未手动指定 `--num_envs` 时，默认使用 `num_envs=1`，降低多环境布局导致镜头偏离目标的风险。
- **Execution Record**:
```bash
python play_deploy.py --help

python -m py_compile \
  play_deploy.py \
  play.py

python play_deploy.py \
  --headless-video \
  --video_length 300
```

## Tool Enhancement

- **Date**: 2026-02-27
- **Action**: 优化 `play.py` 录屏命名，便于区分不同 checkpoint 的视频结果。
- **Details**:
    - **文件**: `play.py`
    - 视频录制参数新增 `name_prefix`，格式为 `{CheckpointName}_{Timestamp}`。
    - 实现方式：从 `resume_path` 提取 checkpoint 文件名，并拼接当前时间戳（`%Y-%m-%d_%H-%M-%S`）。
- **Execution Record**:
```bash
python -m py_compile \
  play.py \
  play_deploy.py
```

## Feature Update

- **Date**: 2026-02-27
- **Action**: 支持多环境地形场景的视频全景录制，避免仅看到单机器人。
- **Details**:
    - **文件**: `play.py`
    - 新增 `--video_camera_mode {follow,overview}` 参数。
    - `follow`: 镜头跟随 `env_0` 机器人（适合单环境特写）。
    - `overview`: 镜头自动拉远并对准环境原点中心（适合多环境地形总览）。
    - 多环境下优先可视化地形分布，若 `overview` 不可用自动回退到 `follow`。
    - **文件**: `play_deploy.py`
    - 视频模式默认不再强制 `num_envs=1`，恢复使用配置常量 `NUM_ENVS`（当前为 32）。
    - 新增 `--video_camera_mode` 透传参数。
    - 未手动指定时，自动策略为：`num_envs>1 => overview`，`num_envs==1 => follow`。
- **Execution Record**:
```bash
python play_deploy.py --help

python -m py_compile \
  play.py \
  play_deploy.py

python play_deploy.py \
  --headless-video \
  --num_envs 32 \
  --video_camera_mode overview \
  --video_length 300
```
