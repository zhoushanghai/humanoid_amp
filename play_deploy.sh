#!/usr/bin/env bash
# Usage: ./play_deploy.sh <checkpoint_path>
# Example:
#   ./play_deploy.sh /home/hz/g1/humanoid_amp/logs/skrl/g1_amp_dance/2026-02-22_18-27-11_ppo_torch/checkpoints/agent_20000.pt

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_path>"
    exit 1
fi

CHECKPOINT="$1"

python -m humanoid_amp.play \
    --task Isaac-G1-AMP-Deploy-Direct-v0 \
    --num_envs 32 \
    --checkpoint "$CHECKPOINT"
