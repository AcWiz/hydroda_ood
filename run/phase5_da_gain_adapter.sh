#!/bin/bash
# Phase 5: Regime-aware DA gain adapter target historical adaptation.
#
# This wrapper uses the Hydro-MSR candidate residual and learns a soft
# DA-evidence-conditioned gain for selective residual updates.

set -euo pipefail

SOURCE_CHECKPOINT="${1:-}"
TARGET_REGION="${2:-US-R1}"
SEED="${3:-0}"
export CUDA_VISIBLE_DEVICES="${4:-1}"
EXTRA_ARGS=()
if [[ "$#" -gt 4 ]]; then
    EXTRA_ARGS=("${@:5}")
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
    echo "  bash run/phase5_da_gain_adapter.sh <source_checkpoint> ${TARGET_REGION} ${SEED} ${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
fi

MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
TARGET_LATENT_DIM="${TARGET_LATENT_DIM:-32}"
ENABLE_TARGET_SPATIAL_REFINE="${ENABLE_TARGET_SPATIAL_REFINE:-1}"
TARGET_SPATIAL_REFINE_HIDDEN="${TARGET_SPATIAL_REFINE_HIDDEN:-16}"
TARGET_SPATIAL_REFINE_ROOTZONE="${TARGET_SPATIAL_REFINE_ROOTZONE:-1}"
TARGET_SPATIAL_REFINE_INPUT="${TARGET_SPATIAL_REFINE_INPUT:-normalized}"
TARGET_SPATIAL_REFINE_TYPE="${TARGET_SPATIAL_REFINE_TYPE:-hydro_msr_gain}"
TARGET_SPATIAL_REFINE_GAIN_SPAN="${TARGET_SPATIAL_REFINE_GAIN_SPAN:-0.25}"
HYDRO_MSR_HIDDEN="${HYDRO_MSR_HIDDEN:-16}"
ENABLE_HYDRO_MSR_DA_FILM="${ENABLE_HYDRO_MSR_DA_FILM:-0}"
ENABLE_DA_REGIME_GAIN_MIXER="${ENABLE_DA_REGIME_GAIN_MIXER:-1}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cuda}"
LAMBDA_PRIOR="${LAMBDA_PRIOR:-1e-4}"
LAMBDA_LATENT="${LAMBDA_LATENT:-1e-4}"
LAMBDA_GAIN="${LAMBDA_GAIN:-1e-3}"
LAMBDA_GAIN_SMOOTH="${LAMBDA_GAIN_SMOOTH:-1e-3}"
LAMBDA_ANALYSIS="${LAMBDA_ANALYSIS:-0.25}"
SURFACE_WEIGHT="${SURFACE_WEIGHT:-3.0}"
ROOTZONE_WEIGHT="${ROOTZONE_WEIGHT:-1.0}"
SELECTION_ROOTZONE_WEIGHT="${SELECTION_ROOTZONE_WEIGHT:-1.0}"
TARGET_SELECTION_METRIC="${TARGET_SELECTION_METRIC:-combined_val_wrmse}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RESUME_FROM="${RESUME_FROM:-}"
RUN_NAME="${RUN_NAME:-phase5_hydro_msr_gain_${TARGET_REGION}_s${SEED}}"

RESUME_ARGS=()
HAS_RESUME_ARG=0
for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "$arg" == "--resume_from" || "$arg" == --resume_from=* ]]; then
        HAS_RESUME_ARG=1
    fi
done
if [[ -n "${RESUME_FROM}" && "${HAS_RESUME_ARG}" == "0" ]]; then
    RESUME_ARGS=(--resume_from "${RESUME_FROM}")
fi
RESUME_LABEL="${RESUME_FROM:-}"
if [[ -z "${RESUME_LABEL}" && "${HAS_RESUME_ARG}" == "1" ]]; then
    RESUME_LABEL="from extra args"
fi

echo "============================================"
echo "Phase 5 Regime-aware DA Gain Adapter"
echo "  source_checkpoint=${SOURCE_CHECKPOINT}"
echo "  target_region=${TARGET_REGION}"
echo "  seed=${SEED}"
echo "  adaptation_setting=target_full_train"
echo "  target_train=2015-2021"
echo "  target_val=2022"
echo "  target_eval=2023-2025"
echo "  freeze_hypernetwork=true"
echo "  staged_schedule=global_target_modules:${STAGE1_EPOCHS},spatial_gain:remaining"
echo "  target_eval labels are never used for adaptation"
echo "  max_epochs=${MAX_EPOCHS} batch_size=${BATCH_SIZE} lr=${LR}"
echo "  device=${DEVICE}"
echo "  target_spatial_refine_type=${TARGET_SPATIAL_REFINE_TYPE} refine_input=${TARGET_SPATIAL_REFINE_INPUT}"
echo "  target_spatial_refine_gain_span=${TARGET_SPATIAL_REFINE_GAIN_SPAN}"
echo "  hydro_msr_hidden=${HYDRO_MSR_HIDDEN} enable_hydro_msr_da_film=${ENABLE_HYDRO_MSR_DA_FILM}"
echo "  enable_da_regime_gain_mixer=${ENABLE_DA_REGIME_GAIN_MIXER} stage1_epochs=${STAGE1_EPOCHS}"
echo "  target_selection_metric=${TARGET_SELECTION_METRIC} selection_rootzone_weight=${SELECTION_ROOTZONE_WEIGHT}"
echo "  run_name=${RUN_NAME}"
echo "  output_dir=${OUTPUT_DIR:-auto}"
echo "  resume_from=${RESUME_LABEL:-none}"
echo "============================================"

PYTHONPATH=. python scripts/train/train_hyperda_target_adapt.py \
    --source_checkpoint "${SOURCE_CHECKPOINT}" \
    --target_region "${TARGET_REGION}" \
    --adaptation_setting target_full_train \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --run_name "${RUN_NAME}" \
    --target_latent_dim "${TARGET_LATENT_DIM}" \
    $(if [[ "${ENABLE_TARGET_SPATIAL_REFINE}" == "1" ]]; then echo "--enable_target_spatial_refine"; fi) \
    --target_spatial_refine_hidden "${TARGET_SPATIAL_REFINE_HIDDEN}" \
    $(if [[ "${TARGET_SPATIAL_REFINE_ROOTZONE}" == "1" ]]; then echo "--target_spatial_refine_rootzone"; fi) \
    --target_spatial_refine_input "${TARGET_SPATIAL_REFINE_INPUT}" \
    --target_spatial_refine_type "${TARGET_SPATIAL_REFINE_TYPE}" \
    --target_spatial_refine_gain_span "${TARGET_SPATIAL_REFINE_GAIN_SPAN}" \
    --hydro_msr_hidden "${HYDRO_MSR_HIDDEN}" \
    $(if [[ "${ENABLE_HYDRO_MSR_DA_FILM}" == "1" ]]; then echo "--enable_hydro_msr_da_film"; fi) \
    $(if [[ "${ENABLE_DA_REGIME_GAIN_MIXER}" == "1" ]]; then echo "--enable_da_regime_gain_mixer"; fi) \
    --stage1_epochs "${STAGE1_EPOCHS}" \
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
    --lambda_analysis "${LAMBDA_ANALYSIS}" \
    --surface_weight "${SURFACE_WEIGHT}" \
    --rootzone_weight "${ROOTZONE_WEIGHT}" \
    --selection_rootzone_weight "${SELECTION_ROOTZONE_WEIGHT}" \
    --target_selection_metric "${TARGET_SELECTION_METRIC}" \
    --log_every_steps 50 \
    --checkpoint_every 5 \
    --max_train_batches "${MAX_TRAIN_BATCHES}" \
    --max_val_batches "${MAX_VAL_BATCHES}" \
    $(if [[ -n "${OUTPUT_DIR}" ]]; then echo "--output_dir ${OUTPUT_DIR}"; fi) \
    "${RESUME_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
