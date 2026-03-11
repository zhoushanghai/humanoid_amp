# BUG_EXPERIENCE

## Records

### 2026-03-11 09:32 - Proprioception play 偶发空房间导致看不到障碍物

- Symptom: `python -m humanoid_amp.play --task Isaac-G1-AMP-Poprioception-Direct-v0 ...` 录出来的视频里偶尔只有机器人和房间墙体，看不到任何障碍物。
- Root Cause: `play.py` 在加载 checkpoint 后会重新 `env.reset()`，而 `G1AmpPoprioceptionEnv` 会在 reset 时重新随机采样障碍物布局；当 `sample_episode_obstacle_layout(...)` 对所有启用 slot 都采样失败时，候选障碍物会保持默认隐藏位置 `z = -10.0`，于是当前 episode 变成空房间。
- Fix: 在 `g1_amp_poprioception_scene.py` 中保留原有 rejection sampling，但新增 fallback 逻辑；如果某个环境在一轮 reset 后没有任何 active obstacle，就从当前候选形状里选择 footprint 最小的障碍物，并把它放到房间角落的合法位置，保证至少有一个可见障碍物。
- Verification: 使用 `logs/skrl/g1_amp_poprioception/2026-03-11_01-04-47_amp_torch/checkpoints/agent_1050000.pt` 执行 top-down `play` 后，生成视频 `logs/skrl/g1_amp_poprioception/2026-03-11_01-04-47_amp_torch/videos/play/agent_1050000_2026-03-11_09-31-12-step-0.mp4`；抽帧检查 `tmp/obstacle_visible_check.png` 可见房间右上角出现蓝色圆柱障碍物。
- Prevention: 对依赖随机 rejection sampling 的场景生成逻辑，不要默认“失败就是空场景”；至少要在 debug / play 路径里提供保底可见物体，避免误判成渲染或 checkpoint 问题。
- Related Files: `g1_amp_poprioception_scene.py`, `g1_amp_poprioception_env.py`, `play.py`
- Related Commands: `python -m humanoid_amp.play --task Isaac-G1-AMP-Poprioception-Direct-v0 --algorithm AMP --checkpoint logs/skrl/g1_amp_poprioception/2026-03-11_01-04-47_amp_torch/checkpoints/agent_1050000.pt --num_envs 1 --video --video_length 60 --camera_view topdown --headless`

### 2026-03-11 00:06 - 短时 AMP 训练未产出可直接 play 的 checkpoint

- Symptom: `python -m humanoid_amp.train --task Isaac-G1-AMP-Poprioception-Direct-v0 --algorithm AMP --num_envs 64 --max_iterations 1 --headless` 跑完后，最新 run 目录下没有 `checkpoints/`，导致后续 `play` 只能手动指向旧模型。
- Root Cause: `agents/skrl_g1_amp_poprioception_cfg.yaml` 依赖周期性 checkpoint 保存，而短时 smoke run 在达到保存间隔前就结束了；`train.py` 退出前也没有补一个终态 checkpoint。
- Fix: 在 `train.py` 的 `runner.run()` 之后统一补存 `checkpoints/agent_last.pt`，保证任何成功结束的训练 run 都有一个可立即复用的 checkpoint；同时把 `__init__.py` 的本地 Gym 注册改成幂等方式，减少同进程重复导入时的重复注册风险。
- Verification: 重新运行短训练后，生成了 `logs/skrl/g1_amp_poprioception/2026-03-11_00-03-47_amp_torch/checkpoints/agent_last.pt`；随后执行 `python -m humanoid_amp.play --task Isaac-G1-AMP-Poprioception-Direct-v0 --algorithm AMP --num_envs 1 --video --video_length 30 --headless`，日志确认自动加载了最新 run 的 `agent_last.pt`，并产出视频 `logs/skrl/g1_amp_poprioception/2026-03-11_00-03-47_amp_torch/videos/play/Isaac-G1-AMP-Poprioception-Direct-v0_2026-03-11_00-04-19-step-0.mp4`。
- Prevention: 自定义训练入口不要只依赖周期性 checkpoint；只要存在 smoke test、短回归或人工中断后的复用需求，就应在训练结束时固定导出一个终态 checkpoint。
- Related Files: `train.py`, `__init__.py`
- Related Commands: `python -m humanoid_amp.train --task Isaac-G1-AMP-Poprioception-Direct-v0 --algorithm AMP --num_envs 64 --max_iterations 1 --headless`; `python -m humanoid_amp.play --task Isaac-G1-AMP-Poprioception-Direct-v0 --algorithm AMP --num_envs 1 --video --video_length 30 --headless`
