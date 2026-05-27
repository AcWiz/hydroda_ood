#!/bin/bash
# Phase 4: Region-Specific Source-Only Baseline
# Trains one SmallResUNet per region (R1-R6), each trained ONLY on that region's
# source_fit data (2015-2020). Auto-evaluates on source_val (2021) and
# target_query (2023-2025) after training.
#
# Usage:
#   bash run/phase4_source_only_region_specific.sh          # default: K=0 seed=0
#   bash run/phase4_source_only_region_specific.sh 0 1       # K=0, seed=1
#   bash run/phase4_source_only_region_specific.sh 4 0       # K=4, seed=0
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

REGIONS=("US-R1" "US-R2" "US-R3" "US-R4" "US-R5" "US-R6")

echo "============================================"
echo "Phase 4 Region-Specific Source-Only Training"
echo "  K=${K}"
echo "  seed=${SEED}"
echo "  regions=${REGIONS[*]}"
echo "  config=configs/model_resunet_main.yaml"
echo "============================================"

for region in "${REGIONS[@]}"; do
    echo ""
    echo "--------------------------------------------"
    echo "Training region-specific model: ${region}"
    echo "--------------------------------------------"

    PYTHONPATH=. python scripts/train/train_source_only_region_specific.py \
        --config configs/model_resunet_main.yaml \
        --target_region "${region}" \
        --K "${K}" \
        --seed "${SEED}" \
        --device cuda \
        --amp

    echo "Done: ${region}"
done

echo ""
echo "============================================"
echo "All region-specific models complete."
echo "  K=${K}  seed=${SEED}"
echo "  regions=${REGIONS[*]}"
echo "============================================"
