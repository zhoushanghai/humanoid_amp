# 消融实验计划：历史帧观测对训练的影响

**目标**：逐步确定是历史帧的哪个组成部分导致 `num_actor_observations=2` 时训练失败。

**判断标准**：每个实验跑 **3~5 万步**，在 TensorBoard 观察：
- `Policy / Standard Deviation`：有没有**变化**（不是水平线）
- `Reward / Total reward`：有没有**上升趋势**

---

## 观测组成说明

当前 Actor 的单帧观测（102 维）由三部分构成：

| 符号 | 内容 | 维度 |
|---|---|---|
| **A** | base_actor_obs（关节角/角速度/根部高度/姿态/线速度/角速度） | 71 |
| **B** | last_actions（上一步关节位置指令） | 29 |
| **C** | command（目标速度 vx, vy） | 2 |

---

## 实验列表

### ① 基线（对照组）

- **当前帧**：A + B + C
- **历史帧**：无
- **observation_space**：102
- **num_actor_observations**：1
- **预期**：✅ 能训（已验证）
- **结果记录**：
  - Policy Std：正常变化 ✅
  - Reward 趋势：正常上升 ✅
  - **结论**：[x] 能训

---

### ② 加历史运动学（✅ 已通过）

- **当前帧**：A + B + C（102维，固定完整）
- **历史帧**：A 仅 base_obs（71维）
- **observation_space**：173
- **num_actor_observations**：2
- **配置**：`history_include_last_actions=False`, `history_include_command=False`
- **预期**：测试历史本身是否工作
- **Checkpoint**：`logs/skrl/g1_amp_dance/2026-02-23_17-26-04_ppo_torch/checkpoints/agent_185000.pt`
- **结果记录**：
  - Policy Std：正常变化（有训练动态） ✅
  - Reward 趋势：明显上升 ✅
  - **结论**：[x] 能训 / [ ] 不能训

---

### ③ 加历史运动学 + 历史动作 (✅ 已通过)

- **当前帧**：A + B + C
- **历史帧**：A + B（100 维）
- **observation_space**：202
- **num_actor_observations**：2
- **预期**：测试 last_actions 进入历史帧是否造成问题
- **Checkpoint**：`logs/skrl/g1_amp_dance/2026-02-23_19-13-39_ppo_torch/checkpoints/agent_10000.pt`
- **结果记录**：
  - Policy Std：正常变化 ✅
  - Reward 趋势：上升 ✅
  - **结论**：[x] 能训 / [ ] 不能训

---

### ④ 完整方案 (✅ 已通过)

- **当前帧**：A + B + C
- **历史帧**：A + B + C（102 维，与当前帧完全一致）
- **observation_space**：204
- **num_actor_observations**：2
- **预期**：✅ 机制修复后已能正常训练
- **Checkpoint**：`logs/skrl/g1_amp_dance/2026-02-23_19-24-57_ppo_torch/checkpoints/agent_10000.pt`
- **结果记录**：
  - Policy Std：正常变化 ✅
  - Reward 趋势：稳定上升 ✅
  - **结论**：[x] 能训

## 实施步骤 (Experiment ④)

### [MODIFY] [g1_amp_env_cfg.py](file:///home/hz/g1/humanoid_amp/g1_amp_env_cfg.py)

- 将 `G1AmpDeployEnvCfg` 中的 `history_include_command` 设置为 `True`。

## 验证计划

### 自动化验证
- **维度检查**：启动 `train.py`，确保 `observation_space` 为 `204`。
- **运行测试**：
  ```bash
  conda run -n g1_amp python -m humanoid_amp.train \
    --task Isaac-G1-AMP-Deploy-Direct-v0 \
    --num_envs 512 \
    --headless
  ```

---

### ⑤ 备选：去掉历史帧里的 last_actions（若③失败）

> 仅在实验③失败时执行

- **当前帧**：A + B + C
- **历史帧**：A + C（73 维，去掉 last_actions）
- **observation_space**：175
- **num_actor_observations**：2
- **预期**：排查 last_actions 是否为根因
- **Checkpoint**：`logs/skrl/g1_amp_dance/_______________/checkpoints/agent______.pt`
- **结果记录**：
  - Policy Std：
  - Reward 趋势：
  - **结论**：[ ] 能训 / [ ] 不能训

---

### ⑥ 备选：去掉历史帧里的 command（若③成功但④失败）

> 仅在实验③成功、④失败时执行

- **当前帧**：A + B + C
- **历史帧**：A + B（100 维，去掉 command）
- **observation_space**：202
- **num_actor_observations**：2
- **预期**：排查 command 历史是否为根因
- **Checkpoint**：`logs/skrl/g1_amp_dance/_______________/checkpoints/agent______.pt`
- **结果记录**：
  - Policy Std：
  - Reward 趋势：
  - **结论**：[ ] 能训 / [ ] 不能训

---

## 决策树

```
②能训？
├── 否 → 历史帧机制本身有 Bug（检查 warm-start、buffer 逻辑）
└── 是 → ③能训？
           ├── 否 → last_actions 进历史是根因 → 执行⑤确认
           └── 是 → ④能训？
                      ├── 是 → 问题已解决（可能 fixed_log_std 修复生效）
                      └── 否 → command 进历史是根因 → 执行⑥确认
```

---

## 配置修改参考

每次实验修改 `g1_amp_env_cfg.py` 中的 `observation_space` 和对应的 `g1_amp_env.py` 中 `per_frame_parts` 构建逻辑。

**训练命令**：
```bash
conda run -n g1_amp python -m humanoid_amp.train \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --num_envs 4096 \
  --headless
```
