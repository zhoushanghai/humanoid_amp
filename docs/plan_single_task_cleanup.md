# Plan: Single-Task Cleanup

- Date: 2026-03-11
- Task Name: `Single-Task Cleanup`
- Related Task: `Isaac-G1-AMP-Poprioception-Direct-v0`
- Status: Completed

## Objective

将仓库收敛为只支持 `Isaac-G1-AMP-Poprioception-Direct-v0` 的单任务项目，同时保留该任务训练所需的共享底座与工具。

## Checklist

- ✅ ~~清理 Gym task 注册，只保留 proprioception。~~
- ✅ ~~解除 `G1AmpPoprioceptionEnvCfg` 对 Deploy 配置链的继承依赖。~~
- ✅ ~~删除旧任务专属 agent 配置、脚本、评测配置与无关入口。~~
- ✅ ~~清理现行文档与 README，只保留 proprioception 入口。~~
- ✅ ~~归档当前仍在主路径下、但主要描述旧多任务体系的历史文档。~~
- ✅ ~~更新 `docs/DEV_LOG.md` 记录本次单任务化清理。~~
- ✅ ~~完成 `64 env` 与 `4096 env` 的训练 smoke 验证。~~

## Notes

- 历史事实保留在 `docs/DEV_LOG.md` 与 `docs/archive/*`。
- 运行时清理目标不包含继续优化 `4096 env` 性能，只要求当前入口不被清理破坏。

## Implementation Summary

- 运行时入口已经收敛到单一 Gym ID：`Isaac-G1-AMP-Poprioception-Direct-v0`。
- proprioception 配置已改为直接继承共享 `G1AmpEnvCfg`，不再依赖 Deploy / Walk / Dance / Custom 的旧配置链。
- 旧任务 agent 配置、旧 humanoid 环境、deploy/velocity-tracking 脚本、旧 motion 资产与主路径中的旧规划文档均已移除或归档。
- 现行 README 已更新为 proprioception-only 使用方式。
- 训练 smoke 验证已通过：
  - `64 env / max_iterations=1`
  - `4096 env / max_iterations=1`
