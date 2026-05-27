#!/bin/bash
# Phase 4B: Prompt-conditioned shared backbone (FiLMConditionalResUNet + RegionPromptEncoder)
# Target: US-R1, K=0 (no target labels), seed=0
#
# Usage:
#   bash run/phase4_prompt_conditioned.sh              # default: US-R1 K=0 seed=0 GPU1
#   bash run/phase4_prompt_conditioned.sh US-R2 0 1    # custom region, K, seed
#   bash run/phase4_prompt_conditioned.sh US-R1 0 0 0  # GPU0 explicitly
#
# Prerequisites:
#   - Splits: artifacts/splits/US_loro_kdate_splits.json
#   - DA.nc: /fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc
#   - Region masks: artifacts/regions/US_region_masks.nc

set -euo pipefail

TARGET_REGION="${1:-US-R1}"
K="${2:-0}"
SEED="${3:-0}"
export CUDA_VISIBLE_DEVICES="${4:-1}"

cd "$(dirname "$0")/.."

echo "============================================"
echo "Phase 4 Prompt-Conditioned Shared Backbone"
echo "  target_region=${TARGET_REGION}"
echo "  K=${K}"
echo "  seed=${SEED}"
echo "  width=32 prompt_dim=64 lr=3e-4 batch_size=8 accum_steps=4"
echo "  lat_weighted_loss=True zero_init=True inc_norm=True amp=True"
echo "============================================"

PYTHONPATH=. python scripts/train/train_prompt_conditioned_shared.py \
    --target_region "${TARGET_REGION}" \
    --K "${K}" \
    --seed "${SEED}" \
    --device cuda \
    --amp \
    --accum_steps 4 \
    --zero_raw_increment_init \
    --target_increment_normalization \
    --use_lat_weighted_loss \
    --batch_size 16 \
    --max_epochs 50 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    --num_workers 0 \
    --width 32 \
    --prompt_dim 64 \
    --log_every_steps 100 \
    --eval_every_epochs 1 \
    --checkpoint_every 10 \
    --selection_metric source_val_transfer_safe_score \

echo "Done: ${TARGET_REGION} K=${K} seed=${SEED}"