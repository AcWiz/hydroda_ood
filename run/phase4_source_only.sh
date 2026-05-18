#!/bin/bash
# Phase 4 full training: Source-only backbone, US-R1, K=0, seed=0
# Config: configs/model_resunet_main.yaml (width=32, max_epochs=30, accum_steps=4)
#
# Usage:
#   bash run/phase4_source_only.sh              # default: US-R1 K=0 seed=0
#   bash run/phase4_source_only.sh US-R2 0 1    # custom region, K, seed
#
# Prerequisites:
#   - Splits regenerated: artifacts/splits/US_loro_kdate_splits.json
#   - DA.nc at /fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc
#   - Region masks at artifacts/regions/US_region_masks.nc

set -euo pipefail

TARGET_REGION="${1:-US-R1}"
K="${2:-0}"
SEED="${3:-0}"
export CUDA_VISIBLE_DEVICES="${4:-1}"

cd "$(dirname "$0")/.."

echo "============================================"
echo "Phase 4 Full Training"
echo "  target_region=${TARGET_REGION}"
echo "  K=${K}"
echo "  seed=${SEED}"
echo "  config=configs/model_resunet_main.yaml"
echo "============================================"

PYTHONPATH=. python scripts/train/train_source_only_backbone.py \
    --config configs/model_resunet_main.yaml \
    --target_region "${TARGET_REGION}" \
    --K "${K}" \
    --seed "${SEED}" \
    --device cuda \
    --amp \
    --lr 3e-4 --weight_decay 1e-4 --grad_clip 1.0 \
    --accum_steps 4 \
    --checkpoint_every 5 \
    --max_epochs 50 \

echo "Done: ${TARGET_REGION} K=${K} seed=${SEED}"
