# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Purpose: Register local humanoid AMP Gym tasks without repeatedly overriding existing registry entries.
Main contents: agent config import, idempotent Gym registration helper, and local task registrations.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

def _safe_register(id: str, **kwargs) -> None:
    """Register a Gym env only once to avoid duplicate-registry warnings on repeated imports."""
    if id in gym.registry:
        return
    gym.register(id=id, **kwargs)


_safe_register(
    id="Isaac-Humanoid-AMP-Dance-Direct-v0",
    entry_point=f"{__name__}.humanoid_amp_env:HumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_amp_env_cfg:HumanoidAmpDanceEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_dance_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_dance_amp_cfg.yaml",
    },
)

_safe_register(
    id="Isaac-Humanoid-AMP-Run-Direct-v0",
    entry_point=f"{__name__}.humanoid_amp_env:HumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_amp_env_cfg:HumanoidAmpRunEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_run_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_run_amp_cfg.yaml",
    },
)

_safe_register(
    id="Isaac-Humanoid-AMP-Walk-Direct-v0",
    entry_point=f"{__name__}.humanoid_amp_env:HumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_amp_env_cfg:HumanoidAmpWalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_walk_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_walk_amp_cfg.yaml",
    },
)

_safe_register(
    id="Isaac-G1-AMP-Dance-Direct-v0",
    entry_point=f"{__name__}.g1_amp_env:G1AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AmpDanceEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_g1_dance_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_g1_dance_amp_cfg.yaml",
    },
)

_safe_register(
    id="Isaac-G1-AMP-Walk-Direct-v0",
    entry_point=f"{__name__}.g1_amp_env:G1AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AmpWalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_g1_walk_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_g1_walk_amp_cfg.yaml",
    },
)

_safe_register(
    id="Isaac-G1-AMP-Custom-Direct-v0",
    entry_point=f"{__name__}.g1_amp_env:G1AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AmpCustomEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_g1_custom_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_g1_custom_amp_cfg.yaml",
    },
)

_safe_register(
    id="Isaac-G1-AMP-Deploy-Direct-v0",
    entry_point=f"{__name__}.g1_amp_env:G1AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AmpDeployEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_g1_deploy_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_g1_deploy_amp_cfg.yaml",
    },
)

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
