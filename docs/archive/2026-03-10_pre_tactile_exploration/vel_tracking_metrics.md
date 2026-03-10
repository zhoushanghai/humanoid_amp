# Velocity Tracking 指标计算说明

本文档说明 `bash/eval/val/eval_all_vel_tracking.sh` 中汇总指标的计算方式。  
实际计算逻辑位于 `scripts/eval/eval_vel_tracking.py`，可视化与表格汇总位于 `scripts/eval/plot_vel_tracking_boxplot.py`。

## 1. 通用记号与采样窗口

对每个并行环境 \(i\)，在记录窗口内按时间步 \(t\) 计算：

- 线速度跟踪误差（每步）  
  \[
  e^{(i)}_{\text{lin}, t} = \left\| \mathbf{v}^{\text{cmd}}_{xy,t} - \mathbf{v}^{\text{act}}_{xy,t} \right\|_2
  \]
- 偏航角速度跟踪误差（每步）  
  \[
  e^{(i)}_{\text{yaw}, t} = \left| \omega^{\text{cmd}}_{z,t} - \omega^{\text{act}}_{z,t} \right|
  \]

并在窗口内取时间平均：

\[
\bar e^{(i)}_{\text{lin}} = \frac{1}{T}\sum_t e^{(i)}_{\text{lin}, t},\quad
\bar e^{(i)}_{\text{yaw}} = \frac{1}{T}\sum_t e^{(i)}_{\text{yaw}, t}
\]

评估采用“阶梯爬升 + 稳定 + 记录”流程：

- 每 `0.5s` 将命令幅值增加 `0.2`（`RAMP_INC=0.2`, `RAMP_DUR=0.5`）
- 到目标后稳定 `2s`（`SETTLE_S=2.0`）
- 再记录 `10s`（`RECORD_S=10.0`）

生存判定：若环境在整个评估阶段从未 `done`（未跌倒），则记为 `survived=True`。

## 2. 低/高速线速度跟踪准确率

### 2.1 命令集合

- `low_lin`:
  - \(v_x \in \{0.5, 1.0, 1.5, 2.0, 2.5\}\), \(v_y=0\), \(\omega_z=0\)
  - \(v_y \in \{-1.5,-1.0,-0.5,0.5,1.0,1.5\}\), \(v_x=0\), \(\omega_z=0\)
- `high_lin`:
  - \(v_x \in \{3.0, 3.25, 3.5, 3.75, 4.0\}\), \(v_y=0\), \(\omega_z=0\)
  - \(v_y \in \{-2.5,-2.0,-1.5,1.5,2.0,2.5\}\), \(v_x=0\), \(\omega_z=0\)

### 2.2 准确率定义

对每个环境：

\[
\text{acc}^{(i)}_{\text{lin}} =
\mathrm{clip}\!\left(
1 - \frac{\bar e^{(i)}_{\text{lin}}}{\max(\|\mathbf{v}^{\text{cmd}}_{xy}\|_2, 0.1)},
0, 1
\right)\times 100
\]

其中 `0.1` 是分母下限（`eps`），用于避免接近零命令时数值不稳定。

### 2.3 聚合方式

1. 先按命令组合（combo）分组；
2. 每个 combo 只用 `survived=True` 的环境求均值；
3. 对“有生存样本”的 combo 再求均值和标准差。

因此，最终 `low_lin_lin_acc` / `high_lin_lin_acc` 是 **combo-level mean**，而不是直接对所有环境一次性平均。

## 3. 低/高速偏航跟踪准确率

### 3.1 命令集合

- `yaw_low`: \(v_x=1.0,\ v_y=0,\ \omega_z \in \{-1.0,-0.5,0,0.5,1.0\}\)
- `yaw_high`: \(v_x=3.0,\ v_y=0,\ \omega_z \in \{-1.0,-0.5,0,0.5,1.0\}\)

### 3.2 准确率定义

对每个环境：

\[
\text{acc}^{(i)}_{\text{yaw}} =
\mathrm{clip}\!\left(
1 - \frac{\bar e^{(i)}_{\text{yaw}}}{\max(|\omega^{\text{cmd}}_z|, 0.1)},
0, 1
\right)\times 100
\]

聚合方式与线速度一致（先 combo 内生存样本均值，再跨 combo 统计均值/std）。

## 4. 最大可达前向速度 \(v_x\)

扫描速度列表（按绝对值升序）：`[3.0, 3.5, ..., 6.0]` m/s。  
每次测试时所有环境都执行同一命令：\((v_x=\text{spd}, v_y=0, \omega_z=0)\)。

单环境在该速度“通过”的条件：

- 该次测试中环境生存；
- 线速度平均误差 \(\bar e_{\text{lin}} < 0.5\) m/s。

单环境最大可达前向速度：

\[
v^{(i)}_{x,\max} = \max\{ |v_x| \mid \text{该环境在该 } v_x \text{ 通过} \}
\]

最终报告：

- `max_vx` = \(\text{mean}_i(v^{(i)}_{x,\max})\)
- `max_vx_std` = \(\text{std}_i(v^{(i)}_{x,\max})\)

## 5. 最大可达横向速度 \(|v_y|\)

扫描速度列表：`[1.0, 1.5, ..., 4.0]` m/s。  
每次测试命令：\((v_x=0, v_y=\text{spd}, \omega_z=0)\)。

通过条件：

- 环境生存；
- 线速度平均误差 \(\bar e_{\text{lin}} < 0.3\) m/s（阈值比 `max_vx` 更严格）。

单环境最大可达横向速度：

\[
|v_y|^{(i)}_{\max} = \max\{ |v_y| \mid \text{该环境在该 } v_y \text{ 通过} \}
\]

最终报告：

- `max_vy` = \(\text{mean}_i(|v_y|^{(i)}_{\max})\)
- `max_vy_std` = \(\text{std}_i(|v_y|^{(i)}_{\max})\)

## 6. Step-response survival rate

阶跃命令序列：

1. `(0.0, 0.0, 0.0)` 持续 `3s`
2. `(4.0, 0.0, 0.0)` 持续 `10s`
3. `(0.0, 2.0, 0.0)` 持续 `10s`
4. `(4.0, 0.0, 0.0)` 持续 `10s`

整体阶跃生存率定义为：

\[
\text{step\_survival} =
\frac{\#\{\text{在整个序列中从未 } done \text{ 的环境}\}}{N}
\]

代码还会记录每个 phase 的 `alive_rate`，以及该 phase 最后 `5s` 的平均线速度/偏航误差用于诊断，但主汇总指标是 `step_survival`。

## 7. CSV 关键字段对应

- 线速度准确率：`low_lin_lin_acc`, `high_lin_lin_acc`（及 `*_std`）
- 偏航准确率：`yaw_low_yaw_acc`, `yaw_high_yaw_acc`（及 `*_std`）
- 最大前向速度：`max_vx`, `max_vx_std`
- 最大横向速度：`max_vy`, `max_vy_std`
- 阶跃生存率：`step_survival`

