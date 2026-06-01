#!/bin/bash
# Phase 5: HyperDA target historical adaptation entrypoint.
#
# This wrapper preregisters the target adaptation protocol. The full optimizer
# runner is intentionally separate from source-stage training: target_eval
# labels are never used for adaptation, selection, or normalization.
#
# Protocol:
#   target_train=2015-2021
#   target_val=2022
#   target_eval=2023-2025
#   freeze_hypernetwork=true
#   trainable=target_latent,adapter_coefficient_residuals,residual_gain
#
# Usage:
#   bash run/phase5_hyperda_target_adapt.sh <source_checkpoint> US-R1 0 1

set -euo pipefail

SOURCE_CHECKPOINT="${1:?source checkpoint path is required}"
TARGET_REGION="${2:-US-R1}"
SEED="${3:-0}"
export CUDA_VISIBLE_DEVICES="${4:-1}"

cd "$(dirname "$0")/.."

echo "============================================"
echo "Phase 5 HyperDA Target Historical Adaptation"
echo "  source_checkpoint=${SOURCE_CHECKPOINT}"
echo "  target_region=${TARGET_REGION}"
echo "  seed=${SEED}"
echo "  adaptation_setting=target_full_train"
echo "  target_train=2015-2021"
echo "  target_val=2022"
echo "  target_eval=2023-2025"
echo "  freeze_hypernetwork=true"
echo "  trainable=target_latent,adapter_coefficient_residuals,residual_gain"
echo "  target_eval labels are never used for adaptation"
echo "  split_artifact=artifacts/splits/US_loro_target_train_splits.json"
echo "============================================"

PYTHONPATH=. python scripts/train/train_hyperda_target_adapt.py \
    --source_checkpoint "${SOURCE_CHECKPOINT}" \
    --target_region "${TARGET_REGION}" \
    --adaptation_setting target_full_train \
    --seed "${SEED}" \
    --device cuda \
    --target_latent_dim 32 \
    --batch_size 8 \
    --max_epochs 20 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    --num_workers 0 \
    --use_lat_weighted_loss \
    --lambda_prior 1e-4 \
    --lambda_latent 1e-4 \
    --lambda_gain 1e-3 \
    --lambda_gain_smooth 1e-3 \
    --log_every_steps 50 \
    --checkpoint_every 5
