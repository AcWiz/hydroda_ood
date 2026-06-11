#!/bin/bash
# Phase 4 target_full_history_region_oracle finetune sanity baseline.
# This is not the paper-facing LORO source-only baseline or source-regime specialist bank.
# Initializes each region-specific SmallResUNet from a legacy all-regions checkpoint,
# then trains on that region's 2015-2021 labels only.
#
# Usage:
#   bash run/phase4_source_only_region_specific_finetune.sh
#   bash run/phase4_source_only_region_specific_finetune.sh /path/to/global.pt 0 1
#
# If the first argument is omitted, the latest checkpoint_best_source_val_safe_score.pt
# under artifacts/runs/phase4_source_only_all_regions is used.

set -euo pipefail

INIT_CHECKPOINT="${1:-}"
SEED="${2:-0}"
export CUDA_VISIBLE_DEVICES="${3:-0}"
K="${K:-0}"
if [[ "${K}" == "0" ]]; then
    ADAPTATION_SETTING="zero_shot_context"
else
    ADAPTATION_SETTING="few_shot_k${K}"
fi

cd "$(dirname "$0")/.."

if [[ -z "${INIT_CHECKPOINT}" ]]; then
    INIT_CHECKPOINT="$(find artifacts/runs/phase4_source_only_all_regions \
        -path '*/checkpoints/checkpoint_best_source_val_safe_score.pt' \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
fi

if [[ -z "${INIT_CHECKPOINT}" || ! -f "${INIT_CHECKPOINT}" ]]; then
    echo "ERROR: legacy all-regions checkpoint not found."
    echo "Run: bash run/phase4_source_only_all_regions.sh 0 ${SEED}"
    exit 1
fi

REGIONS=("US-R1" "US-R2" "US-R3" "US-R4" "US-R5" "US-R6")

echo "============================================"
echo "Phase 4 Region-Specific Finetune Baseline"
echo "  method=target_full_history_region_oracle"
echo "  init_from_checkpoint=${INIT_CHECKPOINT}"
echo "  seed=${SEED}"
echo "  regions=${REGIONS[*]}"
echo "  adaptation_setting=${ADAPTATION_SETTING}  K=${K}"
echo "  status=secondary_internal_not_paper_main"
echo "  oracle_status=oracle_upper_bound_internal_only"
echo "  recipe=width32 norm+zero latw batch16 accum4 epoch50 lr3e-4"
echo "============================================"

# for region in "${REGIONS[@]}"; do
#     echo ""
#     echo "--------------------------------------------"
#     echo "Finetuning region-specific model: ${region}"
#     echo "--------------------------------------------"

#     PYTHONPATH=. python scripts/train/train_source_only_region_specific.py \
#         --config configs/model_resunet_main.yaml \
#         --target_region "${region}" \
#         --adaptation_setting "${ADAPTATION_SETTING}" \
#         --K "${K}" \
#         --seed "${SEED}" \
#         --device cuda \
#         --amp \
#         --init_from_checkpoint "${INIT_CHECKPOINT}" \
#         --zero_raw_increment_init \
#         --target_increment_normalization \
#         --use_lat_weighted_loss \
#         --batch_size 16 \
#         --max_epochs 50 \
#         --lr 3e-4 \
#         --weight_decay 1e-4 \
#         --grad_clip 1.0 \
#         --accum_steps 4 \
#         --checkpoint_every 10 \
#         --selection_metric source_val_loss 

#     echo "Done: ${region}"
# done

# echo ""
# echo "============================================"
# echo "All region-specific finetune models complete."
# echo "  seed=${SEED}"
# echo "  init_from_checkpoint=${INIT_CHECKPOINT}"
# echo "  regions=${REGIONS[*]}"
# echo "============================================"




PYTHONPATH=. python scripts/train/train_source_only_region_specific.py \
        --config configs/model_resunet_main.yaml \
        --target_region US-R3 \
        --adaptation_setting "${ADAPTATION_SETTING}" \
        --K "${K}" \
        --seed "${SEED}" \
        --device cuda \
        --amp \
        --init_from_checkpoint "${INIT_CHECKPOINT}" \
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
        --selection_metric source_val_loss \
        --resume_from artifacts/runs/phase4_source_only_region_specific/phase4_source_only_region_specific_source_only_US-R3_w32_e50_lr0.0003_norm_nozero_s0_20260607_175040/checkpoints/checkpoint_epoch_029.pt


PYTHONPATH=. python scripts/train/train_source_only_region_specific.py \
        --config configs/model_resunet_main.yaml \
        --target_region US-R4 \
        --adaptation_setting "${ADAPTATION_SETTING}" \
        --K "${K}" \
        --seed "${SEED}" \
        --device cuda \
        --amp \
        --init_from_checkpoint "${INIT_CHECKPOINT}" \
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


PYTHONPATH=. python scripts/train/train_source_only_region_specific.py \
        --config configs/model_resunet_main.yaml \
        --target_region US-R5 \
        --adaptation_setting "${ADAPTATION_SETTING}" \
        --K "${K}" \
        --seed "${SEED}" \
        --device cuda \
        --amp \
        --init_from_checkpoint "${INIT_CHECKPOINT}" \
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


PYTHONPATH=. python scripts/train/train_source_only_region_specific.py \
        --config configs/model_resunet_main.yaml \
        --target_region US-R6 \
        --adaptation_setting "${ADAPTATION_SETTING}" \
        --K "${K}" \
        --seed "${SEED}" \
        --device cuda \
        --amp \
        --init_from_checkpoint "${INIT_CHECKPOINT}" \
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
