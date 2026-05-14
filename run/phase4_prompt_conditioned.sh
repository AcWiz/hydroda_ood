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
echo "  config=configs/model_resunet_main.yaml"
echo "============================================"

PYTHONPATH=. python scripts/train/train_prompt_conditioned_shared.py \
    --config configs/model_resunet_main.yaml \
    --target_region "${TARGET_REGION}" \
    --K "${K}" \
    --seed "${SEED}" \
    --device cuda \
    --amp \
    --accum_steps 4 \
    --zero_raw_increment_init \
    --target_increment_normalization \
    --batch_size 8 \
    --max_epochs 31 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    --num_workers 0 \
    --width 32 \
    --prompt_dim 64 \
    --log_every_steps 100 \
    --eval_every_epochs 1 \
    --run_name "" \
    --resume_from "artifacts/runs/phase4_prompt_conditioned/phase4_prompt_conditioned_prompt_conditioned_US-R1_w32_e30_lr0.001_norm_zero_s0_20260513_092533/checkpoints/last.pt" 

echo "Done: ${TARGET_REGION} K=${K} seed=${SEED}"