#!/bin/bash
# Phase 4A: source_pooled_global_backbone.
# US-only transition global baseline: leave-one-region-out pooled source model.
# No target-region labels used for training; no prompt encoder.
#
# Usage:
#   bash run/phase4_source_only.sh              # default: US-R1 seed=0 GPU1
#   bash run/phase4_source_only.sh US-R2 0 1    # custom region, seed, GPU
#
# Prerequisites:
#   - Splits: artifacts/splits/US_loro_zero_few_shot_splits.json
#   - DA.nc: /fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc
#   - Region masks: artifacts/regions/US_region_masks.nc

set -euo pipefail

TARGET_REGION="${1:-US-R1}"
SEED="${2:-0}"
export CUDA_VISIBLE_DEVICES="${3:-0}"

cd "$(dirname "$0")/.."

echo "============================================"
echo "Phase 4 Source-Only Backbone"
echo "  method=source_pooled_global_backbone"
echo "  status=paper_main_transition_us_loro"
echo "  scope=US-only transition global"
echo "  target_region=${TARGET_REGION}"
echo "  seed=${SEED}"
echo "  adaptation_setting=zero_shot_context  K=0"
echo "  source_fit=2015-2021 source_val=2022"
echo "  split_artifact=artifacts/splits/US_loro_zero_few_shot_splits.json"
echo "  train_domains=all US source regions excluding target_region"
echo "  width=32 lr=3e-4 batch_size=16 accum_steps=4"
echo "  lat_weighted_loss=True zero_init=True inc_norm=True amp=True"
echo "============================================"

PYTHONPATH=. python scripts/train/train_source_only_backbone.py \
    --target_region "${TARGET_REGION}" \
    --adaptation_setting zero_shot_context --K 0 \
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
    --log_every_steps 100 \
    --eval_every_epochs 1 \
    --checkpoint_every 10 \
    --selection_metric source_val_loss

echo "Done: ${TARGET_REGION} seed=${SEED}"
