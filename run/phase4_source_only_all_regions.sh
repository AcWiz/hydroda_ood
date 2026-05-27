#!/bin/bash
# Phase 4 extension: Source-only all-regions baseline
# Trains on ALL US-R1..R6 (2015-2020), reports per-region source_val (2021)
#
# Usage:
#   bash run/phase4_source_only_all_regions.sh          # default: K=0 seed=0
#   bash run/phase4_source_only_all_regions.sh 0 1       # K=0, seed=1
#
# Prerequisites:
#   - Splits: artifacts/splits/US_loro_kdate_splits.json
#   - DA.nc at /fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc
#   - Region masks at artifacts/regions/US_region_masks.nc

set -euo pipefail

K="${1:-0}"
SEED="${2:-0}"
export CUDA_VISIBLE_DEVICES="${3:-1}"

cd "$(dirname "$0")/.."

echo "============================================"
echo "Phase 4 All-Regions Source-Only Training"
echo "  K=${K}"
echo "  seed=${SEED}"
echo "  config=configs/model_resunet_main.yaml"
echo "============================================"

PYTHONPATH=. python scripts/train/train_source_only_all_regions.py \
    --config configs/model_resunet_main.yaml \
    --K "${K}" \
    --seed "${SEED}" \
    --device cuda \
    --amp \
    --lr 3e-4 --weight_decay 1e-4 --grad_clip 1.0 \
    --accum_steps 4 \
    --max_epochs 50

echo "Done: all-regions K=${K} seed=${SEED}"
