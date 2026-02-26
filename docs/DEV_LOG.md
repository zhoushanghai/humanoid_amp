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

- **[2026-02-23]** \`git commit\`: docs(docs): update DEV_LOG with training command and experiment results / 更新开发日志，新增训练命令及实验结果
- **[2026-02-25]** `git commit`: feat(play,exp): 新增部署脚本并记录S1结果 / add deploy play script and log S1 result


## Experiment Plan Update

- **Date**: 2026-02-25 22:00
- **Action**: 重构当前实验计划为“从不能训起点逐步加条件”的阶梯式设计。
- **Details**:
  - **文件**: `docs/@experiment_plan.md`
  - 由原先并列消融（E1~E6）改为顺序加条件（S0~S5）。
  - 明确失败起点 `S0`：`fixed_log_std=True` + 历史 `A+B+C` + reset 历史零初始化。
  - 将已完成结果映射为 `S1`：仅改 `fixed_log_std=False`，已可训，checkpoint 为 `logs/skrl/g1_amp_dance/2026-02-25_20-54-01_ppo_torch/checkpoints/agent_85000.pt`。
  - 新增目标输出：首个成功台阶与最小可训条件集（Minimal Trainable Conditions）。

## Experiment Plan Correction

- **Date**: 2026-02-25 22:10
- **Action**: 修正阶梯实验文档状态，重置本轮 S1 为未完成。
- **Details**:
  - **文件**: `docs/@experiment_plan.md`
  - 将 `S1` 从“已完成”改为“待验证”，`Checkpoint` 改为待补充。
  - Checklist 中 `S1` 由已勾选恢复为未勾选。
  - 明确说明：此前 `agent_85000.pt` 仅作为历史参考，不计入本轮 `S1` 完成判定。
  - `Next Action` 更新为按顺序执行 `S0 -> S1`。

## Experiment Plan Rewrite

- **Date**: 2026-02-25 22:20
- **Action**: 按“从不能训起点开始”的思路重写当前实验文档。
- **Details**:
  - **文件**: `docs/@experiment_plan.md`
  - 重构为阶梯式单变量流程：`S0 -> S1 -> S2 -> S3 -> S4 -> S5`。
  - 明确起点 `S0`（失败配置）与统一判定目标：`First Trainable Step`、`Minimal Trainable Conditions`。
  - 清空本轮完成状态，所有步骤恢复未开始（从零开始）。
  - 增加每步统一记录模板，保证实验可追踪与可复现。
- **Execution Record**:

```bash
python -m humanoid_amp.train \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --headless
```

## Experiment Update (S1 Start)

- **Date**: 2026-02-25 22:32
- **Action**: 启动 S1 测试（仅改 `fixed_log_std=False`）。
- **Details**:
  - **文件**: `agents/skrl_g1_deploy_amp_cfg.yaml`
  - 变更: `fixed_log_std: True -> False`。
  - 其余条件保持当前 S0 代码路径不变（Deploy 当前为 2-frame `base+command`，reset 历史零初始化）。
- **Execution Record**:

```bash
python -m humanoid_amp.train \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --headless
```

## Experiment Record

- **Date**: 2026-02-25 23:05
- **Model**: `logs/skrl/g1_amp_dance/2026-02-25_22-34-04_ppo_torch/checkpoints/agent_45000.pt`
- **Phenomenon**: S1（仅改 `fixed_log_std=False`，其余保持 S0）训练效果不行，完全训不出来。
- **Conclusion/Notes**: 仅解锁 `std` 学习不足以跨过可训阈值；下一步执行 S2，仅切换 reset 历史初始化为 `warm-start` 做对照。

- **Execution Record**:

```bash
python -m humanoid_amp.train \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --headless
```

## Experiment Update (S2 Start)

- **Date**: 2026-02-25 23:15
- **Action**: 启动 S2 测试（仅切换 reset 历史初始化为 warm-start）。
- **Details**:
  - **文件**: `g1_amp_env.py`
  - 新增 `_just_reset_mask`：在 reset 时标记环境，不再直接清零历史 buffer。
  - 在 `_get_observations` 中对刚 reset 的环境执行 warm-start：使用当前真实观测填满历史槽后再正常 shift。
  - 保持 S1 条件不变：`agents/skrl_g1_deploy_amp_cfg.yaml` 中 `fixed_log_std=False`。
- **Execution Record**:

```bash
python -m humanoid_amp.train \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --headless
```

## Experiment Record

- **Date**: 2026-02-25 23:15
- **Model**: `logs/skrl/g1_amp_dance/2026-02-25_23-08-39_ppo_torch/checkpoints/agent_10000.pt`
- **Phenomenon**: S2（仅切换 reset 历史初始化为 warm-start）效果仍不行，依然训不出来。
- **Conclusion/Notes**: warm-start 不是当前可训阈值的关键突破条件；继续执行 S3，改历史帧为 `A+B`（base_obs + last_actions）。

- **Execution Record**:

```bash
python -m humanoid_amp.train \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --headless
```

## Experiment Update (S3 Start)

- **Date**: 2026-02-25 23:15
- **Action**: 启动 S3 测试（仅改历史帧为 `A+B`）。
- **Details**:
  - **文件**: `g1_amp_env.py`
    - 历史帧构造由 `base_obs + command` 改为 `base_obs + last_actions`。
    - `actor_obs_per_frame` 由 `71 + 2` 改为 `71 + 29`。
  - **文件**: `g1_amp_env_cfg.py`
    - `observation_space` 由 `146` 调整为 `200`（2 帧，每帧 100 维）。
  - 其余条件保持 S2 不变：`fixed_log_std=False` + warm-start。
- **Execution Record**:

```bash
python -m humanoid_amp.train \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --headless
```

## 关键命令

- **[2026-02-25]** `git commit`: feat(env,exp): 记录S2失败并切换S3条件 / log S2 fail and switch S3

## Experiment Record

- **Date**: 2026-02-25 23:25
- **Model**: `待补充（S3 本轮 checkpoint 路径）`
- **Phenomenon**: S3（仅改历史帧为 `A+B`）效果仍不行，依然训不出来。
- **Conclusion/Notes**: `A+B` 历史帧仍未跨过可训阈值；继续执行 S4，改历史帧为 `A`（base_obs only）。

- **Execution Record**:

```bash
python -m humanoid_amp.train \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --headless
```

## Experiment Update (S4 Start)

- **Date**: 2026-02-25 23:25
- **Action**: 启动 S4 测试（仅改历史帧为 `A`）。
- **Details**:
  - **文件**: `g1_amp_env.py`
    - 历史帧构造由 `base_obs + last_actions` 改为 `base_obs`。
    - `actor_obs_per_frame` 由 `71 + 29` 改为 `71`。
  - **文件**: `g1_amp_env_cfg.py`
    - `observation_space` 由 `200` 调整为 `142`（2 帧，每帧 71 维）。
  - 其余条件保持 S3 不变：`fixed_log_std=False` + warm-start。
- **Execution Record**:

```bash
python -m humanoid_amp.train \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --headless
```

## Experiment Record

- **Date**: 2026-02-25 23:35
- **Model**: `待补充（S4 本轮 checkpoint 路径）`
- **Phenomenon**: S4（仅改历史帧为 `A`）仍然训不出来。
- **Conclusion/Notes**: 目前 S1~S4 均未跨过可训阈值，下一步建议先做单帧 sanity check（`num_actor_observations=1`）验证训练链路基本可训性。

- **Execution Record**:

```bash
python -m humanoid_amp.train \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --headless
```

## 关键命令

- **[2026-02-25]** `git commit`: feat(env,exp): 记录S4失败并转单帧排查 / log S4 fail and n1 check

## Code Update

- **Date**: 2026-02-26
- **Action**: 调整多帧 actor 观测拼接：当前帧加入 `last_actions` 和 `command`，历史帧保持 `base_obs`。
- **Details**:
  - **文件**: `g1_amp_env.py`
    - 多帧路径改为：`actor_obs = current_frame(A+B+C) + history(A)`。
    - 历史 buffer 由 `n` 帧改为 `(n-1)` 帧，仅存历史帧。
  - **文件**: `g1_amp_env_cfg.py`
    - `G1AmpDeployEnvCfg.observation_space` 由 `142` 更新为 `173`（`num_actor_observations=2`）。
- **Execution Record**:

```bash
# 本次为代码修改记录，无训练命令执行
```

## Experiment Record

- **Date**: 2026-02-26
- **Model**: `N/A（本次为代码定位与修复记录，未启动新训练）`
- **Phenomenon**: Deploy 多帧路径在 S4 代码中仅使用 `base_obs`，未把当前帧 `last_actions` 与 `command` 送入 policy，导致训练输入关键信息缺失。
- **Conclusion/Notes**: 本轮定位结论为“多帧路径缺少当前帧 `last_actions/command` 是训不出来的关键原因”。已修复为 `current_frame(A+B+C) + history(A)`；后续可做对照实验拆分验证 `last_actions` 与 `command` 的单独贡献。

## 关键命令

- **[2026-02-26]** `git commit`: fix(env): 修复多帧当前帧输入缺失 / restore current-frame inputs

## Code Update

- **Date**: 2026-02-26
- **Action**: 修复多帧输入“当前帧重复”问题，确保 policy 读取到真实历史帧（上一时刻及更早）。
- **Details**:
  - **文件**: `g1_amp_env.py`
    - 调整多帧时序：先读取 `actor_obs_history_buffer` 作为 `history_flatten` 拼接到 `actor_obs`，再将当前 `base_obs` 写回 history buffer。
    - 修复后 `num_actor_observations=2` 时输入为 `[当前A, 上一帧A]`，不再是 `[当前A, 当前A]`。
  - **文件**: `g1_amp_env_cfg.py`
    - 同步修正 `G1AmpDeployEnvCfg` 中多帧观测注释，避免与当前实现不一致。
- **Execution Record**:

```bash
# 本次为代码修改记录，无训练命令执行
```
