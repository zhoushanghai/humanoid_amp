# Velocity Tracking 全量正式评测计划

## 1. 目标

本计划以“**一次运行产出最终全量测试结果**”为唯一目标，不包含冒烟测试阶段。

最终产出 `metrics_summary.csv`，包含以下指标：

- `low_lin_lin_acc`, `low_lin_lin_acc_std`
- `high_lin_lin_acc`, `high_lin_lin_acc_std`
- `yaw_low_yaw_acc`, `yaw_low_yaw_acc_std`
- `yaw_high_yaw_acc`, `yaw_high_yaw_acc_std`
- `max_vx`, `max_vx_std`
- `max_vy`, `max_vy_std`
- `step_survival`

## 2. 范围与边界

### In Scope

- 完整执行 `low_lin / high_lin / yaw_low / yaw_high / max_vx / max_vy / step_survival` 全量评测。
- 导出汇总 CSV 与 combo 明细 CSV。
- checkpoint 路径只在一个配置文件中维护（单点修改）。

### Out of Scope

- 不改训练逻辑（`train.py`、reward、网络结构、训练超参）。
- 不做自动多 checkpoint 批量趋势分析。
- 不做阈值 pass/fail 判定（只输出原始指标）。

## 3. 单点配置（Checkpoint 只改一个地方）

新增配置文件：`configs/eval_velocity_tracking.yaml`

建议字段：

- `active_checkpoint`: 当前评测 checkpoint（唯一编码点）
- `task`: `Isaac-G1-AMP-Deploy-Direct-v0`
- `num_envs`: `64`
- `seed`: 固定值（复现）
- `device`: 例如 `cuda:0`
- `output_dir`: 结果输出目录

后续切换模型时，只改 `active_checkpoint`。

## 4. 实现方案

新增脚本：`scripts/eval/eval_vel_tracking_protocol.py`

输入：

- `--config configs/eval_velocity_tracking.yaml`

输出：

- `metrics_summary.csv`（最终汇总指标）
- `metrics_combo_details.csv`（每个 combo 明细）
- `run_meta.json`（运行参数、checkpoint、时间戳等）

## 5. 指标计算口径（严格一致）

每个 combo 执行流程：

1. `ramp`: 每 `0.5s` 命令增幅 `0.2`，直到目标值
2. `settle`: 到目标后稳定 `2.0s`
3. `record`: 记录 `10.0s` 用于统计

误差定义：

- `e_lin_t = ||v_cmd_xy - v_act_xy||_2`
- `e_yaw_t = |w_cmd_z - w_act_z|`
- `e_lin_bar = mean(e_lin_t)`（record 窗口）
- `e_yaw_bar = mean(e_yaw_t)`（record 窗口）

生存判定：

- 在该 combo 全阶段从未 `done` 记为 `survived=True`

聚合规则：

1. 先按 combo 分组
2. 每个 combo 仅使用 `survived=True` 的环境求均值
3. 再对“有生存样本”的 combo 求均值/std

专项规则：

- `max_vx` 通过条件：`survived=True` 且 `e_lin_bar < 0.5`
- `max_vy` 通过条件：`survived=True` 且 `e_lin_bar < 0.3`
- `step_survival`：完整四段阶跃序列中全程未 done 的环境占比

## 6. 全量测试命令集合

- `low_lin`:
  - `vx in {0.5, 1.0, 1.5, 2.0, 2.5}, vy=0, wz=0`
  - `vy in {-1.5, -1.0, -0.5, 0.5, 1.0, 1.5}, vx=0, wz=0`
- `high_lin`:
  - `vx in {3.0, 3.25, 3.5, 3.75, 4.0}, vy=0, wz=0`
  - `vy in {-2.5, -2.0, -1.5, 1.5, 2.0, 2.5}, vx=0, wz=0`
- `yaw_low`: `vx=1.0, vy=0, wz in {-1.0, -0.5, 0, 0.5, 1.0}`
- `yaw_high`: `vx=3.0, vy=0, wz in {-1.0, -0.5, 0, 0.5, 1.0}`
- `max_vx` 扫描：`[3.0, 3.5, ..., 6.0]`
- `max_vy` 扫描：`[1.0, 1.5, ..., 4.0]`
- `step_survival` 阶跃序列：
  1. `(0.0, 0.0, 0.0)` `3s`
  2. `(4.0, 0.0, 0.0)` `10s`
  3. `(0.0, 2.0, 0.0)` `10s`
  4. `(4.0, 0.0, 0.0)` `10s`

## 7. 执行与验收

执行流程：

1. 在 `configs/eval_velocity_tracking.yaml` 设置 `active_checkpoint`
2. 运行评测脚本（一次全量运行）
3. 检查输出文件与字段完整性
4. 记录到 `docs/DEV_LOG.md`

验收标准：

- 成功生成 `metrics_summary.csv`、`metrics_combo_details.csv`、`run_meta.json`
- `metrics_summary.csv` 13 个字段齐全，且为有效数值（非 NaN）
- 明细可追踪每个 combo 的 `n_env`、`n_survived`、`valid_combo`

