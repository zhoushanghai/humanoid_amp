# Plan: Auto-Compute observation_space from num_actor_observations

## 目标
让用户只需修改 `num_actor_observations` 一个参数，
`observation_space` 自动换算，无需手动同步。

## 实现方案

在 `G1AmpDeployEnvCfg` 中添加 `__post_init__` 方法，自动推导维度：

```python
def __post_init__(self):
    # 4 key bodies × 3 dims = 12，从 AMP obs 中去除
    KEY_BODY_OBS_SIZE = 4 * 3
    base_obs_size = self.amp_observation_space - KEY_BODY_OBS_SIZE  # 71
    command_size = 2 if self.rew_track_vel > 0.0 else 0              # 2
    per_frame = base_obs_size + self.action_space + command_size     # 102
    self.observation_space = per_frame * self.num_actor_observations
```

| num_actor_observations | 自动计算的 observation_space |
|---|---|
| 1 | 102 |
| 2 | 204 |
| 3 | 306 |
| N | 102 × N |

## Checklist

- [ ] 修改 `G1AmpDeployEnvCfg`：删除硬编码的 `observation_space`，添加 `__post_init__` <!-- id: 0 -->
- [ ] 验证配置正确加载 <!-- id: 1 -->
- [ ] 更新 DEV_LOG.md <!-- id: 2 -->
