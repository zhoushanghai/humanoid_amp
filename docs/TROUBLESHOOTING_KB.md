# Troubleshooting Knowledge Base

用于沉淀项目内可复用的疑难排错经验。每条案例按“问题表现-根因分析-处理动作-验证结果-复用建议”记录。

## [CASE-20260305-001] Velocity Tracking 评测中视频与测试项错位、重置异常与中途崩溃

- 时间: 2026-03-05
- 场景/命令:
```bash
python scripts/eval/eval_vel_tracking_protocol.py \
  --config configs/eval_velocity_tracking.yaml \
  --headless \
  --video \
  --video_length 1200
```

### 问题表现

- 视频开头偶尔出现上一测试项动作，随后才突然 reset。
- `low_lin_vx_0.50` 切到 `low_lin_vx_1.00` 时，肉眼看起来没重置，像在上一速度基础上继续跑。
- 运行中途报错并退出：`Inplace update to inference tensor outside InferenceMode is not allowed`。

### 根因分析

- 根因 1（重置不生效）:
  - `skrl` 的 `IsaacLabWrapper.reset()` 有 `_reset_once` 缓存逻辑，默认只第一次真正调用底层 reset，后续直接返回缓存观测。
  - 证据文件: `/home/hz/miniconda3/envs/g1_amp/lib/python3.11/site-packages/skrl/envs/wrappers/torch/isaaclab_envs.py`。
- 根因 2（视频首帧串片）:
  - reset 后渲染存在 1~数帧滞后，新视频文件首帧可能抓到上一测试尾帧。
- 根因 3（中途崩溃）:
  - 评测循环把 `env.step()` 放进 `torch.inference_mode()`，导致环境内部出现 inference tensor。
  - 后续 reset 时的原地写入触发 PyTorch 限制并报错。

### 处理动作

- 文件: `scripts/eval/eval_vel_tracking_protocol.py`
- 关键改动:
  - 在 `_reset_env_for_new_test(...)` 中，重置前强制 `env._reset_once = True`（若属性存在），确保每个测试项都真实全量 reset。
  - 视频开始前先清理渲染缓存帧（`video_render_flush_frames`）。
  - 每段视频增加 reset 后零命令前导（`video_reset_lead_in_s`），保证开头先看到方阵再执行命令。
  - `_run_steps(...)` 改为仅策略推理使用 `torch.no_grad()`，`env.step()` 移出 inference_mode。
- 文件: `configs/eval_velocity_tracking.yaml`
- 关键改动:
  - `reset_between_combos: true`
  - `reset_sync_steps: 2`
  - `video_render_flush_frames: 8`
  - `video_reset_lead_in_s: 0.5`

### 验证结果

- 测试项切换时可看到稳定的“方阵起始 -> 执行命令”过程。
- 速度点切换不再沿用上一测试动作状态。
- 评测流程可跑完，不再出现 inference tensor reset 崩溃。

### 复用建议

- 凡是用 `skrl` + IsaacLab 包装器做分段评测，都默认检查 wrapper `reset` 是否缓存。
- 仿真视频分段录制时，始终加“渲染缓存清理 + 前导对齐段”。
- 推理优化只包策略前向，不要把 `env.step/reset` 放进 `torch.inference_mode()`。
