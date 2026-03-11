# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Purpose: Register the single remaining local Gym task for G1 AMP proprioception training.
Main contents: agent config import, idempotent Gym registration helper, and the proprioception task entry.
"""

import gymnasium as gym

from . import agents


def _safe_register(id: str, **kwargs) -> None:
    """Register a Gym env only once to avoid duplicate-registry warnings on repeated imports."""
    if id in gym.registry:
        return
    gym.register(id=id, **kwargs)


_safe_register(
    id="Isaac-G1-AMP-Poprioception-Direct-v0",
    entry_point=f"{__name__}.g1_amp_poprioception_env:G1AmpPoprioceptionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_poprioception_env_cfg:G1AmpPoprioceptionEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_g1_amp_poprioception_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_g1_amp_poprioception_cfg.yaml",
    },
)
