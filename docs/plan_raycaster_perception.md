# Plan: RayCaster Perception for G1 AMP Poprioception

- Date: 2026-03-11
- Task Name: `RayCaster Perception`
- Related Task: `Isaac-G1-AMP-Poprioception-Direct-v0`
- Status: In Progress

## Objective

评估并逐步落地 Isaac Lab 官方 `RayCaster` / `MultiMeshRayCaster` 方案，用于替代或补充当前 `G1-AMP-Poprioception` 任务中的手写表面感知逻辑。

当前讨论目标不是立刻重写整套任务，而是先确定最稳妥的迁移路径：

- [ ] 明确 `RayCaster` 是用于替代“观测”、还是替代“奖励”、还是两者都替代。
- [ ] 确认是否保留当前 `ContactSensor` 的接触奖励主干。
- [ ] 确认是否保留当前 `surface_grid` / `geometry` 奖励，还是先做最小可行删减。

## Current Baseline

当前 proprioception 任务的探索奖励由三部分构成：

- `contact_count`
- `surface_grid`
- `geometry`

对应实现位置：

- [g1_amp_poprioception_env.py](/home/hz/g1/humanoid_amp/g1_amp_poprioception_env.py#L173)
- [g1_amp_poprioception_rewards.py](/home/hz/g1/humanoid_amp/g1_amp_poprioception_rewards.py#L21)
- [g1_amp_poprioception_rewards.py](/home/hz/g1/humanoid_amp/g1_amp_poprioception_rewards.py#L67)

当前环境的障碍物是多目标、按环境独立摆放的静态刚体：

- 每个环境 `1~3` 个有效障碍物，来自 `3` 个 slot、`2` 个候选类型。
- reset 时会重新采样位置和 yaw。
- 启动前会按环境采样不同尺寸。

相关实现位置：

- [g1_amp_poprioception_constants.py](/home/hz/g1/humanoid_amp/g1_amp_poprioception_constants.py#L19)
- [g1_amp_poprioception_scene.py](/home/hz/g1/humanoid_amp/g1_amp_poprioception_scene.py#L234)
- [g1_amp_poprioception_scene.py](/home/hz/g1/humanoid_amp/g1_amp_poprioception_scene.py#L395)

## Technical Decision

推荐优先采用“混合方案”，而不是一步到位完全替换：

- [ ] 第一阶段：引入 `MultiMeshRayCaster` 作为观测输入。
- [ ] 第一阶段：保留 `contact_count` 作为 tactile exploration 的核心奖励。
- [ ] 第一阶段：移除或下线 `surface_grid` 与 `geometry` 这两块手写感知奖励。
- [ ] 第二阶段：如果需要，再把 ray hit 做成低维 occupancy / novelty reward。

不推荐第一版就做“纯雷达替代全部奖励”，原因是：

- 当前任务的研究语义偏 tactile exploration。
- 若完全改为 ray-based reward，任务会从“触碰探索”漂移到“视距/邻近探索”。
- 一步同时替换观测和奖励，难以判断性能变化来自哪里。

## Sensor Choice

针对当前场景，优先考虑 `MultiMeshRayCaster`，而不是基础 `RayCaster`：

- [ ] 使用 `MultiMeshRayCasterCfg` 作为传感器配置入口。
- [ ] 为障碍物和房间墙体分别配置 raycast target。
- [ ] 对会在 reset 中改位姿的目标开启 `track_mesh_transforms=True`。

原因：

- 基础 `RayCaster` 当前只适合单 mesh / 静态 mesh。
- `MultiMeshRayCaster` 支持多目标和动态 mesh transform 跟踪。

参考实现与限制：

- [ray_caster_cfg.py](/home/hz/tiangong/IsaacLab/source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_cfg.py#L35)
- [ray_caster.py](/home/hz/tiangong/IsaacLab/source/isaaclab/isaaclab/sensors/ray_caster/ray_caster.py#L51)
- [multi_mesh_ray_caster_cfg.py](/home/hz/tiangong/IsaacLab/source/isaaclab/isaaclab/sensors/ray_caster/multi_mesh_ray_caster_cfg.py#L18)
- [multi_mesh_ray_caster.py](/home/hz/tiangong/IsaacLab/source/isaaclab/isaaclab/sensors/ray_caster/multi_mesh_ray_caster.py#L48)

## Migration Steps

- [ ] 在 `g1_amp_poprioception_scene.py` 中定义 raycaster target prim 表达式。
- [ ] 在 `g1_amp_poprioception_env.py` 中注册并持有 raycaster 传感器。
- [ ] 设计第一版 ray observation 结构，控制在可训练的小维度范围内。
- [ ] 在 policy observation 中接入 ray 观测，不修改 AMP discriminator 观测。
- [ ] 移除 `surface_grid_visited`、`candidate_cell_counts` 及对应 reward 路径。
- [ ] 保留 `compute_contact_pair_count_reward` 作为初始探索奖励。
- [ ] 对比修改前后的 step throughput、GPU util、训练稳定性。

## Risks

- [ ] 若 ray 数量过大，可能只是把瓶颈从 CPU 迁移到 GPU。
- [ ] 若直接删除 tactile reward，任务定义会发生变化。
- [ ] 若 ray observation 维度过高，可能拖慢训练或影响策略稳定性。
- [ ] 若 target prim 表达式配置不当，raycaster 可能无法正确跟踪 reset 后的障碍物。

## Validation

- [ ] 验证 ray hit 是否能稳定感知墙体和障碍物。
- [ ] 验证 reset 后障碍物位姿变化是否能被传感器正确反映。
- [ ] 验证 policy observation 维度和数据范围是否合理。
- [ ] 验证移除 `surface_grid` 后训练吞吐是否提升。
- [ ] 验证 reward 改动后策略是否仍然会主动接触障碍物。

