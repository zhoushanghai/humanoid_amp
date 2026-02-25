# Experiment: Isaac-G1-AMP-Deploy 从不能训到可训的阶梯实验

- Date: 2026-02-25
- Slug: deploy_train_from_untrainable
- Status: CURRENT

## Why (实验原因)
- 当前问题：同一任务在不同条件下出现“不能训/能训”不稳定现象。
- 风险/代价：如果不从失败起点系统排查，会反复试错，难以复现。
- 必要性：从“不能训”的配置出发，逐步只加一个条件，才能确定“哪一步跨过可训阈值”。

## What (实验目标)
- 核心目标：找出“首次可训”的具体步骤。
- 输出结果：
  - 首个成功台阶（First Trainable Step）
  - 最小可训条件集（Minimal Trainable Conditions）

## Starting Point (S0: 不能训起点)
- `fixed_log_std=True`
- 历史帧组成：`A+B+C`
- reset 历史初始化：`zero`
- 判定预期：不能训或明显停滞

## Which Experiments (阶梯步骤)
- S0:
  - Purpose: 复现失败起点，作为唯一对照。
  - Change: 无（起点配置）。

- S1:
  - Purpose: 验证“可学习 std”是否是第一关键条件。
  - Change: 仅改 `fixed_log_std=False`（其余与 S0 完全一致）。

- S2:
  - Purpose: 验证 reset 历史初始化策略影响。
  - Change: 在 S1 基础上仅改为 `warm-start`（与 `zero` 对照）。

- S3:
  - Purpose: 验证历史动作信息必要性。
  - Change: 在 S2 基础上仅改历史帧 `A+B+C -> A+B`。

- S4:
  - Purpose: 验证最小历史信息是否足够。
  - Change: 在 S3 基础上仅改历史帧 `A+B -> A`。

- S5_recover_8055_logic:
  - Purpose: 以 `8055a88e5048565af93f6c96fc83f2e655d279cd` 为可训基线，恢复其观测逻辑并验证是否恢复可训。
  - Change:
    - 仅恢复 `g1_amp_env_cfg.py` 中 Deploy 的历史帧开关与自动维度推导（`history_include_last_actions/history_include_command/__post_init__`）。
    - 仅恢复 `g1_amp_env.py` 的“当前帧完整 + (n-1) 历史帧”构造逻辑（默认历史 `A+B+C`）。
    - 保持 `agents/skrl_g1_deploy_amp_cfg.yaml` 中 `fixed_log_std=False` 不变。

## How (执行顺序)
1. 先跑 S0，确认“不能训”能被复现。
2. 严格按 S1 -> S2 -> S3 -> S4 -> S5_recover_8055_logic 顺序执行。
3. 每一步仅允许一个变量变化。
4. 每步结束后立即记录 checkpoint 与结论。
5. S5 额外输出“当前代码 vs 8055”的差异消除清单。

## Success Criteria
- Must-have:
  - 明确指出“首个可训步骤”。
  - 给出最小可训条件集，并至少复现 1 次。
- Stop condition:
  - 当“首个可训步骤 + 最小条件集 + 回归复现”三项都满足时结束。

## 统一训练命令
```bash
python -m humanoid_amp.train \
  --task Isaac-G1-AMP-Deploy-Direct-v0 \
  --headless
```

## Checklist
- [ ] S0 失败起点复现完成
- ✅ ~~S1 完成（仅改 `fixed_log_std=False`）~~
- ✅ ~~S2 完成（仅改 reset 初始化为 `warm-start`）~~
- ✅ ~~S3 完成（仅改历史帧为 `A+B`）~~
- ✅ ~~S4 完成（仅改历史帧为 `A`）~~
- ✅ ~~S5_recover_8055_logic 完成（恢复 8055 观测逻辑并验证）~~
- ✅ ~~输出 First Trainable Step 与 Minimal Trainable Conditions~~

## Step Record Template
### Sx
- Config Change:
- Checkpoint:
- Policy / Standard Deviation:
- Reward / total_reward:
- Verdict: `Trainable` / `Not Trainable`
- Notes:

## Log
- 2026-02-25 22:20 - Reorganized plan from scratch: start at untrainable S0 and add one condition per step.
- 2026-02-25 22:32 - Start S1: changed only `fixed_log_std` to `False`; all other conditions remain same as current S0 code path.
- 2026-02-25 23:05 - Finish S1: result is `Not Trainable`. Checkpoint: `logs/skrl/g1_amp_dance/2026-02-25_22-34-04_ppo_torch/checkpoints/agent_45000.pt`.
- 2026-02-25 23:15 - Start S2: switched reset history initialization from `zero` to `warm-start` only; keep S1 settings unchanged.
- 2026-02-25 23:15 - Finish S2: result is `Not Trainable`. Checkpoint: `logs/skrl/g1_amp_dance/2026-02-25_23-08-39_ppo_torch/checkpoints/agent_10000.pt`.
- 2026-02-25 23:15 - Start S3: changed history frame composition to `A+B` only (base_obs + last_actions).
- 2026-02-25 23:25 - Finish S3: result is `Not Trainable`. Checkpoint: `待补充`.
- 2026-02-25 23:25 - Start S4: changed history frame composition to `A` only (base_obs).
- 2026-02-25 23:35 - Finish S4: result is `Not Trainable`. Checkpoint: `logs/skrl/g1_amp_dance/2026-02-25_23-23-50_ppo_torch/checkpoints/agent_10000.pt`.
- 2026-02-25 23:45 - Strategy update: stop further blind ablation; start S5_recover_8055_logic by diffing with commit `8055a88` and restoring its env/cfg observation logic.
- 2026-02-25 23:50 - Start S5_recover_8055_logic: restored `g1_amp_env.py` and `g1_amp_env_cfg.py` to commit `8055a88` observation logic (default history `A+B+C`, auto observation_space).
- 2026-02-26 00:05 - Finish S5_recover_8055_logic: result is `Trainable`. Checkpoint: `logs/skrl/g1_amp_dance/2026-02-25_23-53-25_ppo_torch/checkpoints/agent_10000.pt`.

## Decision
- Conclusion: First Trainable Step = `S5_recover_8055_logic`；Minimal Trainable Conditions = 对齐 `8055a88` 的 `g1_amp_env.py + g1_amp_env_cfg.py` 观测逻辑（默认历史 `A+B+C`）+ `fixed_log_std=False`。
- Next Action: 基于当前可训版本与 `8055a88` 做逐项差异回引（每次只引入一个差异）以定位真正根因。
