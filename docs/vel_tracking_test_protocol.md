# Velocity Tracking 测试执行文档（跨项目提取版）

## 1. 文档目的

本测试文档用于将 [`vel_tracking_metrics.md`](./vel_tracking_metrics.md) 中的评估标准提取为可执行流程。  
即使目标项目与原始项目不同，也可以复用相同指标口径进行横向对比。

## 2. 适用范围与输入要求

执行测试前，目标项目至少需要提供以下信号：

- 命令速度：`v_cmd_x`, `v_cmd_y`, `w_cmd_z`
- 实际速度：`v_act_x`, `v_act_y`, `w_act_z`
- 环境结束标记：`done`（用于生存判定）
- 采样步长或时间戳（用于按窗口统计）

## 3. 通用评估流程

每个测试命令（combo）按以下阶段执行：

1. 阶梯爬升：每 `0.5s` 增幅 `0.2` 直到目标命令
2. 稳定阶段：到目标后等待 `2.0s`
3. 记录阶段：记录 `10.0s` 数据用于统计

生存判定：

- 若环境在该评估阶段从未 `done`，记为 `survived=True`

误差定义：

- 线速度误差（每步）：`e_lin_t = ||v_cmd_xy - v_act_xy||_2`
- 偏航角速度误差（每步）：`e_yaw_t = |w_cmd_z - w_act_z|`
- 窗口平均误差：`e_lin_bar = mean_t(e_lin_t)`，`e_yaw_bar = mean_t(e_yaw_t)`

## 4. 测试项与计算规则

### 4.1 低速/高速线速度跟踪准确率

命令集合：

- `low_lin`:
  - `vx in {0.5, 1.0, 1.5, 2.0, 2.5}`, `vy=0`, `wz=0`
  - `vy in {-1.5,-1.0,-0.5,0.5,1.0,1.5}`, `vx=0`, `wz=0`
- `high_lin`:
  - `vx in {3.0,3.25,3.5,3.75,4.0}`, `vy=0`, `wz=0`
  - `vy in {-2.5,-2.0,-1.5,1.5,2.0,2.5}`, `vx=0`, `wz=0`

单环境准确率：

`acc_lin_i = clip(1 - e_lin_bar / max(||v_cmd_xy||_2, 0.1), 0, 1) * 100`

聚合方式（必须保持一致）：

1. 先按 combo 分组
2. 每个 combo 仅对 `survived=True` 的环境求均值
3. 再对“有生存样本”的 combo 求均值/std

输出字段：

- `low_lin_lin_acc`, `low_lin_lin_acc_std`
- `high_lin_lin_acc`, `high_lin_lin_acc_std`

### 4.2 低速/高速偏航跟踪准确率

命令集合：

- `yaw_low`: `vx=1.0`, `vy=0`, `wz in {-1.0,-0.5,0,0.5,1.0}`
- `yaw_high`: `vx=3.0`, `vy=0`, `wz in {-1.0,-0.5,0,0.5,1.0}`

单环境准确率：

`acc_yaw_i = clip(1 - e_yaw_bar / max(|w_cmd_z|, 0.1), 0, 1) * 100`

聚合方式：

- 与线速度一致（combo 内 survived 均值，再跨 combo 均值/std）

输出字段：

- `yaw_low_yaw_acc`, `yaw_low_yaw_acc_std`
- `yaw_high_yaw_acc`, `yaw_high_yaw_acc_std`

### 4.3 最大可达前向速度 `max_vx`

扫描列表：`[3.0, 3.5, ..., 6.0]` m/s  
测试命令：`(vx=spd, vy=0, wz=0)`

单环境通过条件：

- `survived=True`
- `e_lin_bar < 0.5` m/s

单环境最大值：

- `vx_max_i = max(|vx| where pass)`

输出字段：

- `max_vx = mean_i(vx_max_i)`
- `max_vx_std = std_i(vx_max_i)`

### 4.4 最大可达横向速度 `max_vy`

扫描列表：`[1.0, 1.5, ..., 4.0]` m/s  
测试命令：`(vx=0, vy=spd, wz=0)`

单环境通过条件：

- `survived=True`
- `e_lin_bar < 0.3` m/s

单环境最大值：

- `vy_max_i = max(|vy| where pass)`

输出字段：

- `max_vy = mean_i(vy_max_i)`
- `max_vy_std = std_i(vy_max_i)`

### 4.5 阶跃响应生存率 `step_survival`

固定序列：

1. `(0.0, 0.0, 0.0)` 持续 `3s`
2. `(4.0, 0.0, 0.0)` 持续 `10s`
3. `(0.0, 2.0, 0.0)` 持续 `10s`
4. `(4.0, 0.0, 0.0)` 持续 `10s`

指标定义：

- `step_survival = (# 全序列从未 done 的环境) / N`

建议附加诊断（可选）：

- 每个 phase 的 `alive_rate`
- 每个 phase 末尾 `5s` 的 `e_lin_bar` 与 `e_yaw_bar`

## 5. 最终结果表头（CSV）

```text
low_lin_lin_acc,low_lin_lin_acc_std,high_lin_lin_acc,high_lin_lin_acc_std,
yaw_low_yaw_acc,yaw_low_yaw_acc_std,yaw_high_yaw_acc,yaw_high_yaw_acc_std,
max_vx,max_vx_std,max_vy,max_vy_std,step_survival
```

## 6. 执行检查清单

- [ ] 已确认目标项目可读取 cmd/act 速度与 done 状态
- [ ] 已按统一的 ramp/settle/record 窗口执行所有 combo
- [ ] 线速度与偏航准确率采用 combo-level 聚合
- [ ] `max_vx` 使用 `e_lin_bar < 0.5` 阈值
- [ ] `max_vy` 使用 `e_lin_bar < 0.3` 阈值
- [ ] `step_survival` 基于完整四段阶跃序列统计
- [ ] 已导出标准字段 CSV

