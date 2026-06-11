#!/bin/bash
# Phase 4 target_full_history_region_oracle scratch sanity baseline.
# This is not the paper-facing LORO source-only baseline: each SmallResUNet is
# trained on that same region's 2015-2021 labels. Auto-evaluates on source_val
# (2022) and target_eval
# (2023-2025) after training.
# This is an oracle_upper_bound_internal_only, not a V4.4 source-trained
# region-specific specialist bank.
#
# Usage:
#   bash run/phase4_source_only_region_specific.sh          # default: K=0 seed=0
#   bash run/phase4_source_only_region_specific.sh 0 1       # K=0, seed=1
#   bash run/phase4_source_only_region_specific.sh 4 0       # K=4, seed=0
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

REGIONS=("US-R1" "US-R2" "US-R3" "US-R4" "US-R5" "US-R6")

echo "============================================"
echo "Phase 4 Region-Specific Source-Only Training"
echo "  method=target_full_history_region_oracle"
echo "  K=${K}"
echo "  seed=${SEED}"
echo "  regions=${REGIONS[*]}"
echo "  adaptation_setting=${ADAPTATION_SETTING}"
echo "  status=secondary_internal_not_paper_main"
echo "  oracle_status=oracle_upper_bound_internal_only"
echo "  recipe=width32 norm+zero latw batch16 accum4 epoch50 lr3e-4"
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
        --adaptation_setting "${ADAPTATION_SETTING}" \
        --K "${K}" \
        --seed "${SEED}" \
        --device cuda \
        --amp \
        --zero_raw_increment_init \
        --target_increment_normalization \
        --use_lat_weighted_loss \
        --batch_size 16 \
        --max_epochs 50 \
        --lr 3e-4 \
        --weight_decay 1e-4 \
        --grad_clip 1.0 \
        --accum_steps 4 \
        --checkpoint_every 10 \
        --selection_metric source_val_loss

    echo "Done: ${region}"
done

echo ""
echo "============================================"
echo "All region-specific models complete."
echo "  K=${K}  seed=${SEED}"
echo "  regions=${REGIONS[*]}"
echo "============================================"
