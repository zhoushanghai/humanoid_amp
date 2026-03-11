# Plan: Training Efficiency Optimization

- Date: 2026-03-11
- Task Name: `Training Efficiency Optimization`
- Related Task: `Isaac-G1-AMP-Poprioception-Direct-v0`
- Status: Implementation Completed, Validation Pending

## Objective

提升 `G1-AMP-Poprioception` 训练吞吐，优先降低 reset 期间的 CPU 开销与每步日志同步开销，同时尽量不破坏当前任务定义。

## Confirmed Direction

- ✅ ~~降低环境侧日志同步频率，而不仅仅是调低 TensorBoard writer 的写盘频率。~~
- ✅ ~~重构 reset 后的障碍物布局逻辑，避免每次 reset 都做 CPU 侧 rejection sampling。~~
- ✅ ~~保留足够的场景多样性，避免“所有环境永远固定成同一套场景”导致探索退化。~~

## Key Observation

当前最明显的吞吐瓶颈主要来自两处：

- 每步 reward 日志把 CUDA 张量转成 Python 标量并立即追踪。
- reset 时对障碍物布局进行 CPU 侧采样和 GPU/CPU 往返拷贝。

相关实现：

- [g1_amp_env.py](/home/hz/g1/humanoid_amp/g1_amp_env.py#L455)
- [g1_amp_poprioception_env.py](/home/hz/g1/humanoid_amp/g1_amp_poprioception_env.py#L226)
- [g1_amp_poprioception_scene.py](/home/hz/g1/humanoid_amp/g1_amp_poprioception_scene.py#L471)

## Plan

### Phase 1: Logging Throttle

- ✅ ~~新增环境侧日志节流机制，避免每步都执行 `.item()` 和 `agent.track_data(...)`。~~
- ✅ ~~区分“内部训练必须的张量计算”和“仅用于可视化的 Python 标量日志”。~~
- ✅ ~~保留核心指标，移除或降频次要指标。~~

说明：

- `agents/skrl_g1_amp_poprioception_cfg.yaml` 中的 `write_interval` 只控制 writer 写盘节奏，不等于消除了环境侧每步 `.item()` 的同步成本。

### Phase 2: Reset Simplification

- ✅ ~~方案 A：预生成有限数量的固定场景布局（scene bank），reset 时只做索引采样，不再重新 rejection sampling。~~
- [ ] 方案 B：把当前 `sample_episode_obstacle_layout(...)` 改造成 GPU 张量化采样。
- ✅ ~~对比 A / B 的工程复杂度与实际收益，优先落地更稳妥的一种。~~

推荐顺序：

- ✅ ~~先尝试 scene bank 方案。~~
- [ ] 若 scene bank 仍不足以提供多样性，再考虑 GPU 采样器。

## Non-goals

- [ ] 不在第一阶段引入新的感知传感器或 reward 设计。
- [ ] 不在第一阶段重构 AMP 训练主干。
- [ ] 不以“所有环境永远一模一样”为目标。

## Risks

- [ ] 若仅调低 TensorBoard writer 频率，而不改环境侧 `.item()`，收益会很有限。
- [ ] 若完全固定场景，策略可能过拟合少量布局，探索能力下降。
- [ ] 若直接上 GPU rejection sampling，代码复杂度和调试成本会明显上升。

## Validation

- [ ] 记录改动前后的迭代速度与 GPU-Util 变化。
- [ ] 记录每秒 reset 数量或平均 episode 长度变化。
- [ ] 验证 reward 曲线与行为是否保持可训练。
- [ ] 验证场景多样性是否足够，避免固定布局过拟合。

## Implementation Summary

- 已新增环境侧日志节流配置，base reward 与 proprioception reward 仅在指定步数导出 Python 标量日志。
- 已移除环境内重复的 `agent.track_data(...)` 调用，改为只通过 `infos["log"]` 交给 skrl trainer 统一记录。
- 已将 reset 障碍物布局从“每次 reset 现场 CPU rejection sampling”改为“启动时预生成 scene bank，reset 时按环境索引采样”，并增加总布局预算上限以避免大规模训练卡在初始化。
- 后续仍需补充训练速度、GPU-Util、reset 频率与行为质量的对比验证。
