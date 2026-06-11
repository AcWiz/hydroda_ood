#!/bin/bash
# Phase 5: HyperDA zero/few-shot target generalization.
#
# Main protocol:
#   source_fit=2015-2021
#   source_val=2022 for preregistered checkpoint/hyperparameter selection
#   target_context=2015-2021 input-side only
#   target_support=K labeled cycles for K in {0,4,12}
#   target_val=unused in main protocol
#   target_eval=2023-2025 final offline evaluation
#
# Usage:
#   bash run/phase5_hyperda_zero_few_shot.sh <source_checkpoint> US-R1 4 0 1
#   MAX_STEPS=1 bash run/phase5_hyperda_zero_few_shot.sh <source_checkpoint> US-R1 4 0 1

set -euo pipefail

SOURCE_CHECKPOINT="${1:-}"
TARGET_REGION="${2:-US-R1}"
K="${3:-0}"
SEED="${4:-0}"
export CUDA_VISIBLE_DEVICES="${5:-1}"
EXTRA_ARGS=()
if [[ "$#" -gt 5 ]]; then
    EXTRA_ARGS=("${@:6}")
fi

cd "$(dirname "$0")/.."

if [[ -z "${SOURCE_CHECKPOINT}" ]]; then
    SOURCE_CHECKPOINT="$(find artifacts/runs/phase4_prompt_conditioned \
        -path "*hyperda_basis_adapter_${TARGET_REGION}_*_s${SEED}_*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
        -type f 2>/dev/null | sort | tail -1)"
fi

if [[ -z "${SOURCE_CHECKPOINT}" || ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "ERROR: source HyperDA checkpoint not found." >&2
    echo "Provide it explicitly:" >&2
    echo "  bash run/phase5_hyperda_zero_few_shot.sh <source_checkpoint> ${TARGET_REGION} ${K} ${SEED} ${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
fi

if [[ "${K}" == "0" ]]; then
    ADAPTATION_SETTING="zero_shot_context"
    DEFAULT_STEPS=0
    DEFAULT_LR="1e-3"
    DEFAULT_ANCHOR_ALPHA="0.0"
elif [[ "${K}" == "4" ]]; then
    ADAPTATION_SETTING="few_shot_k4"
    DEFAULT_STEPS=100
    DEFAULT_LR="${LR_K4:-1e-3}"
    DEFAULT_ANCHOR_ALPHA="${ANCHOR_ALPHA_K4:-0.75}"
elif [[ "${K}" == "12" ]]; then
    ADAPTATION_SETTING="few_shot_k12"
    DEFAULT_STEPS=80
    DEFAULT_LR="${LR_K12:-3e-4}"
    DEFAULT_ANCHOR_ALPHA="${ANCHOR_ALPHA_K12:-0.25}"
else
    echo "ERROR: K must be one of 0, 4, 12; got ${K}" >&2
    exit 2
fi

ADAPTATION_STEPS="${MAX_STEPS:-${ADAPTATION_STEPS:-${DEFAULT_STEPS}}}"
ADAPT_RECIPE="${ADAPT_RECIPE:-source_anchor}"
ANCHOR_ALPHA="${ANCHOR_ALPHA:-${DEFAULT_ANCHOR_ALPHA}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-${DEFAULT_LR}}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
TARGET_LATENT_DIM="${TARGET_LATENT_DIM:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

echo "============================================"
echo "Phase 5 HyperDA Zero/Few-Shot Generalization"
echo "  source_checkpoint=${SOURCE_CHECKPOINT}"
echo "  target_region=${TARGET_REGION}"
echo "  K=${K} adaptation_setting=${ADAPTATION_SETTING}"
echo "  seed=${SEED}"
echo "  target_context=2015-2021 input-side only"
echo "  target_support=K labeled cycles"
echo "  target_val=unused_in_main_protocol"
echo "  target_eval=2023-2025 final offline evaluation"
echo "  split_artifact=artifacts/splits/US_loro_zero_few_shot_splits.json"
echo "  model_selection_source=source_val_preregistered"
echo "  adapt_recipe=${ADAPT_RECIPE} anchor_alpha=${ANCHOR_ALPHA}"
echo "  adaptation_steps=${ADAPTATION_STEPS} batch_size=${BATCH_SIZE} lr=${LR}"
echo "  output_dir=${OUTPUT_DIR:-auto}"
echo "============================================"

PYTHONPATH=. python scripts/train/train_hyperda_few_shot_adapt.py \
    --source_checkpoint "${SOURCE_CHECKPOINT}" \
    --target_region "${TARGET_REGION}" \
    --K "${K}" \
    --adaptation_setting "${ADAPTATION_SETTING}" \
    --seed "${SEED}" \
    --device cuda \
    --target_latent_dim "${TARGET_LATENT_DIM}" \
    --adaptation_steps "${ADAPTATION_STEPS}" \
    --adapt_recipe "${ADAPT_RECIPE}" \
    --anchor_alpha "${ANCHOR_ALPHA}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --grad_clip "${GRAD_CLIP}" \
    --num_workers "${NUM_WORKERS}" \
    --use_lat_weighted_loss \
    $(if [[ -n "${OUTPUT_DIR}" ]]; then echo "--output_dir ${OUTPUT_DIR}"; fi) \
    "${EXTRA_ARGS[@]}"
