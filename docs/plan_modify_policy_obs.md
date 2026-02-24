# 计划：修改 Policy Observation - Sim2Real 优化

## 背景

将 Policy 观测从仿真精确信息改为真实机器人可直接获取的信息，减少 Sim2Real gap。

## 用户确认

- 历史帧配置：保持 5 帧 (`num_actor_observations=5`)
- AMP 观测：保持不变（判别器只在仿真中运行）

---

## 目标观测结构

| 观测项 | 维度 | 真实机器人来源 |
|--------|------|---------------|
| dof_positions | 29 | 关节编码器 |
| dof_velocities | 29 | 关节编码器 |
| projected_gravity | 3 | IMU |
| root_angular_velocities | 3 | IMU |
| last_actions | 29 | 已知 |
| command_target_speed | 2 | 已知 |
| **当前帧总计** | **95** | |

---

## 改动清单

### 1. g1_amp_env.py - compute_obs 函数（约第536-561行）

- [ ] 移除：root 高度 (1维)、tangent/normal (6维)、root_linear_velocities (3维)、key_body_positions (12维)
- [ ] 新增：projected_gravity (3维) - 使用 `quat_rotate_inverse(root_rotations, gravity)`

### 2. g1_amp_env.py - _get_observations 方法（约第175-242行）

- [x] 更新 `base_actor_obs` 计算逻辑（移除 key_body 相关维度）
- [x] 更新 `key_body_obs_size` 相关变量

### 3. g1_amp_env_cfg.py - G1AmpDeployEnvCfg 类（约第161-207行）

- [x] 更新基础观测维度：83 → 64
- [x] 更新 observation_space 计算

---

## 验证方法

```bash
# 训练
python -m humanoid_amp.train --task Isaac-G1-AMP-Deploy-Direct-v0 --headless

# 推理
python -m humanoid_amp.play --task Isaac-G1-AMP-Deploy-Direct-v0 --checkpoint <path> --num_envs 1 --video
```

---

## 实验记录 (Experiment Log)

### 实验①：修改 Policy 观测为 Sim2Real 友好版本

| 项目 | 内容 |
|------|------|
| **日期** | 2026-02-24 |
| **目标** | 将 Policy 观测从 114 维改为 95 维，移除仿真精确信息，保留真实机器人可直接获取的信息 |
| **改动** | 移除 root 高度、root 旋转(tangent/normal)、root 线速度、key_body；新增 projected_gravity |
| **状态** | ✅ 已完成代码修改 |

#### 执行记录

| 日期 | 操作 | 详情 |
|------|------|------|
| 2026-02-24 | 修改 compute_obs | 创建 compute_policy_obs(64维) 和 compute_amp_obs(83维) |
| 2026-02-24 | 修改 _get_observations | 调用新函数，添加 gravity 参数 |
| 2026-02-24 | 修改配置 | 添加 policy_base_obs_size=64 |

#### 结果

- **训练日志**：✅ 训练启动成功，环境正常运行
- **Observation Space**：475维 (95当前帧 + 4×95历史帧)
- **结论/备注**：代码修改完成，训练可正常运行

---

## 后续实验（待规划）

- [ ] 实验②：xxx
- [ ] 实验③：xxx
