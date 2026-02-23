# Adding last_actions to Policy History Input

The goal is to include `last_actions` in the observation history buffer for the policy network, ensuring the agent has temporal awareness of its previous actions.

## User Review Required

> [!IMPORTANT]
> This change will increase the `observation_space` dimension from **146** to **204**. 
> - Calculation: $(71 \text{ (base)} + 29 \text{ (last\_actions)} + 2 \text{ (command)}) \times 2 \text{ (frames)} = 204$.

## Proposed Changes

### Environment Configuration

#### [MODIFY] [g1_amp_env_cfg.py](file:///home/hz/g1/humanoid_amp/g1_amp_env_cfg.py)
- Update `observation_space` in `G1AmpDeployEnvCfg` from `146` to `204`.

---

### Environment Implementation

#### [MODIFY] [g1_amp_env.py](file:///home/hz/g1/humanoid_amp/g1_amp_env.py)
- Update `_get_observations` to include `self.last_actions` in the `per_frame_parts` list when `num_actor_observations > 1`.


## Checklist

- [x] Update `observation_space` in `G1AmpDeployEnvCfg` <!-- id: 0 -->
- [x] Add `self.last_actions` to `per_frame_parts` in `_get_observations` <!-- id: 1 -->
- [ ] Verify changes with `play.py` <!-- id: 2 -->

## Verification Plan

### Automated Tests
- Run `python play.py` (or the relevant play script) to ensure no dimension mismatch errors occur during initialization.
- Check logs or print statements to verify the observation tensor shape is `(num_envs, 204)`.
