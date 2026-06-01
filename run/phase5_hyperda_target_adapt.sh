#!/bin/bash
# Phase 5: HyperDA target historical adaptation entrypoint.
#
# This wrapper preregisters the target adaptation protocol. The full optimizer
# runner is intentionally separate from source-stage training: target_eval
# labels are never used for adaptation, selection, or normalization.
#
# Protocol:
#   target_train=2015-2021
#   target_val=2022
#   target_eval=2023-2025
#   freeze_hypernetwork=true
#   trainable=target_latent,adapter_coefficient_residuals,residual_gain
#
# Usage:
#   bash run/phase5_hyperda_target_adapt.sh
#   bash run/phase5_hyperda_target_adapt.sh <source_checkpoint> US-R1 0 1
#
# Optional environment overrides:
#   MAX_EPOCHS=5 BATCH_SIZE=4 LR=5e-4 bash run/phase5_hyperda_target_adapt.sh
#   MAX_TRAIN_BATCHES=1 MAX_VAL_BATCHES=1 bash run/phase5_hyperda_target_adapt.sh

set -euo pipefail

SOURCE_CHECKPOINT="${1:-}"
TARGET_REGION="${2:-US-R1}"
SEED="${3:-0}"
export CUDA_VISIBLE_DEVICES="${4:-1}"

cd "$(dirname "$0")/.."

if [[ -z "${SOURCE_CHECKPOINT}" ]]; then
    SOURCE_CHECKPOINT="$(find artifacts/runs/phase4_prompt_conditioned \
        -path "*hyperda_basis_adapter_${TARGET_REGION}_*_s${SEED}_*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
        -type f 2>/dev/null | sort | tail -1)"
fi

if [[ -z "${SOURCE_CHECKPOINT}" || ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "ERROR: source HyperDA checkpoint not found." >&2
    echo "Provide it explicitly:" >&2
    echo "  bash run/phase5_hyperda_target_adapt.sh <source_checkpoint> ${TARGET_REGION} ${SEED} ${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
fi

MAX_EPOCHS="${MAX_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
TARGET_LATENT_DIM="${TARGET_LATENT_DIM:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LAMBDA_PRIOR="${LAMBDA_PRIOR:-1e-4}"
LAMBDA_LATENT="${LAMBDA_LATENT:-1e-4}"
LAMBDA_GAIN="${LAMBDA_GAIN:-1e-3}"
LAMBDA_GAIN_SMOOTH="${LAMBDA_GAIN_SMOOTH:-1e-3}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

echo "============================================"
echo "Phase 5 HyperDA Target Historical Adaptation"
echo "  source_checkpoint=${SOURCE_CHECKPOINT}"
echo "  target_region=${TARGET_REGION}"
echo "  seed=${SEED}"
echo "  adaptation_setting=target_full_train"
echo "  target_train=2015-2021"
echo "  target_val=2022"
echo "  target_eval=2023-2025"
echo "  freeze_hypernetwork=true"
echo "  trainable=target_latent,adapter_coefficient_residuals,residual_gain"
echo "  target_eval labels are never used for adaptation"
echo "  split_artifact=artifacts/splits/US_loro_target_train_splits.json"
echo "  max_epochs=${MAX_EPOCHS} batch_size=${BATCH_SIZE} lr=${LR}"
echo "  target_latent_dim=${TARGET_LATENT_DIM}"
echo "  max_train_batches=${MAX_TRAIN_BATCHES} max_val_batches=${MAX_VAL_BATCHES}"
echo "  output_dir=${OUTPUT_DIR:-auto}"
echo "============================================"

PYTHONPATH=. python scripts/train/train_hyperda_target_adapt.py \
    --source_checkpoint "${SOURCE_CHECKPOINT}" \
    --target_region "${TARGET_REGION}" \
    --adaptation_setting target_full_train \
    --seed "${SEED}" \
    --device cuda \
    --target_latent_dim "${TARGET_LATENT_DIM}" \
    --batch_size "${BATCH_SIZE}" \
    --max_epochs "${MAX_EPOCHS}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --grad_clip "${GRAD_CLIP}" \
    --num_workers "${NUM_WORKERS}" \
    --use_lat_weighted_loss \
    --lambda_prior "${LAMBDA_PRIOR}" \
    --lambda_latent "${LAMBDA_LATENT}" \
    --lambda_gain "${LAMBDA_GAIN}" \
    --lambda_gain_smooth "${LAMBDA_GAIN_SMOOTH}" \
    --log_every_steps 50 \
    --checkpoint_every 5 \
    --max_train_batches "${MAX_TRAIN_BATCHES}" \
    --max_val_batches "${MAX_VAL_BATCHES}" \
    $(if [[ -n "${OUTPUT_DIR}" ]]; then echo "--output_dir ${OUTPUT_DIR}"; fi)
