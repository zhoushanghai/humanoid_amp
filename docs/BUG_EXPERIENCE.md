# BUG_EXPERIENCE

## Records

### 2026-03-11 00:06 - 短时 AMP 训练未产出可直接 play 的 checkpoint

- Symptom: `python -m humanoid_amp.train --task Isaac-G1-AMP-Poprioception-Direct-v0 --algorithm AMP --num_envs 64 --max_iterations 1 --headless` 跑完后，最新 run 目录下没有 `checkpoints/`，导致后续 `play` 只能手动指向旧模型。
- Root Cause: `agents/skrl_g1_amp_poprioception_cfg.yaml` 依赖周期性 checkpoint 保存，而短时 smoke run 在达到保存间隔前就结束了；`train.py` 退出前也没有补一个终态 checkpoint。
- Fix: 在 `train.py` 的 `runner.run()` 之后统一补存 `checkpoints/agent_last.pt`，保证任何成功结束的训练 run 都有一个可立即复用的 checkpoint；同时把 `__init__.py` 的本地 Gym 注册改成幂等方式，减少同进程重复导入时的重复注册风险。
- Verification: 重新运行短训练后，生成了 `logs/skrl/g1_amp_poprioception/2026-03-11_00-03-47_amp_torch/checkpoints/agent_last.pt`；随后执行 `python -m humanoid_amp.play --task Isaac-G1-AMP-Poprioception-Direct-v0 --algorithm AMP --num_envs 1 --video --video_length 30 --headless`，日志确认自动加载了最新 run 的 `agent_last.pt`，并产出视频 `logs/skrl/g1_amp_poprioception/2026-03-11_00-03-47_amp_torch/videos/play/Isaac-G1-AMP-Poprioception-Direct-v0_2026-03-11_00-04-19-step-0.mp4`。
- Prevention: 自定义训练入口不要只依赖周期性 checkpoint；只要存在 smoke test、短回归或人工中断后的复用需求，就应在训练结束时固定导出一个终态 checkpoint。
- Related Files: `train.py`, `__init__.py`
- Related Commands: `python -m humanoid_amp.train --task Isaac-G1-AMP-Poprioception-Direct-v0 --algorithm AMP --num_envs 64 --max_iterations 1 --headless`; `python -m humanoid_amp.play --task Isaac-G1-AMP-Poprioception-Direct-v0 --algorithm AMP --num_envs 1 --video --video_length 30 --headless`
