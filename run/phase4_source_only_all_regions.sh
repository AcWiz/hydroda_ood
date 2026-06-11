#!/bin/bash
# Phase 4 legacy_all_regions_sanity baseline.
# Trains one SmallResUNet on ALL US-R1..R6 2015-2021 labels and reports
# per-region target_eval/source_val metrics.
# This is not_paper_facing_ood_global because it includes the target region's
# 2015-2021 labels.
#
# Usage:
#   bash run/phase4_source_only_all_regions.sh          # default: K=0 seed=0
#   bash run/phase4_source_only_all_regions.sh 4 1       # K=4, seed=1
#
# Prerequisites:
#   - Splits: artifacts/splits/US_loro_zero_few_shot_splits.json
#   - DA.nc at /fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc
#   - Region masks at artifacts/regions/US_region_masks.nc

set -euo pipefail

K="${1:-0}"
SEED="${2:-0}"
export CUDA_VISIBLE_DEVICES="${3:-1}"
if [[ "${K}" == "0" ]]; then
    ADAPTATION_SETTING="zero_shot_context"
else
    ADAPTATION_SETTING="few_shot_k${K}"
fi

cd "$(dirname "$0")/.."

echo "============================================"
echo "Phase 4 All-Regions Source-Only Training"
echo "  method=legacy_all_regions_sanity"
echo "  status=legacy_sanity_not_paper_facing_ood_global"
echo "  K=${K}"
echo "  seed=${SEED}"
echo "  adaptation_setting=${ADAPTATION_SETTING}"
echo "  recipe=width32 norm+zero latw batch16 accum4 epoch50 lr3e-4"
echo "  config=configs/model_resunet_main.yaml"
echo "============================================"

PYTHONPATH=. python scripts/train/train_source_only_all_regions.py \
    --config configs/model_resunet_main.yaml \
    --adaptation_setting "${ADAPTATION_SETTING}" \
    --K "${K}" \
    --seed "${SEED}" \
    --device cuda \
    --amp \
    --zero_raw_increment_init \
    --target_increment_normalization \
    --use_lat_weighted_loss \
    --batch_size 16 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    --accum_steps 4 \
    --max_epochs 50 \
    --checkpoint_every 10 \
    --selection_metric source_val_loss

echo "Done: all-regions K=${K} seed=${SEED}"
