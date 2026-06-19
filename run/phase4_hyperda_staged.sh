#!/bin/bash
# Phase 4C staged HyperDA source-side training.
#
# Stage 1 is an existing source-only SmallResUNet checkpoint. Stage 2 loads
# that checkpoint as the frozen source base and trains only prompt encoder,
# FiLM, and HyperDA basis-adapter generation modules on source_fit/source_val.
#
# Usage:
#   bash run/phase4_hyperda_staged.sh /path/to/source.pt US-R1 0 0
#   bash run/phase4_hyperda_staged.sh auto US-R1 0 0
#   bash run/phase4_hyperda_staged.sh auto US-R1 0 0 --dry-run

set -euo pipefail

DRY_RUN=0
POSITIONAL=()
for arg in "$@"; do
    case "${arg}" in
        --dry-run)
            DRY_RUN=1
            ;;
        *)
            POSITIONAL+=("${arg}")
            ;;
    esac
done

if [[ "${#POSITIONAL[@]}" -gt 4 ]]; then
    echo "ERROR: Too many positional arguments: ${POSITIONAL[*]}" >&2
    exit 2
fi

SOURCE_CHECKPOINT="${POSITIONAL[0]:-auto}"
TARGET_REGION="${POSITIONAL[1]:-US-R1}"
SEED="${POSITIONAL[2]:-0}"
export CUDA_VISIBLE_DEVICES="${POSITIONAL[3]:-0}"

cd "$(dirname "$0")/.."
export PYTHONPATH=".:${PYTHONPATH:-}"

SPLITS_JSON="${SPLITS_JSON:-artifacts/splits/US_loro_zero_few_shot_splits.json}"
SPLIT_MANIFEST_PATH="${SPLIT_MANIFEST_PATH:-${SPLITS_JSON}}"
PROTOCOL_ID="${PROTOCOL_ID:-hyperda_v4_4_zero_few_shot_generalization_2015_2025_context2015_2021_sourceval2022_eval2023_2025}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/runs/phase4_hyperda_staged}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-phase4_hyperda_staged_${TARGET_REGION}_s${SEED}_${TIMESTAMP}}"
WIDTH="${WIDTH:-32}"
PROMPT_DIM="${PROMPT_DIM:-64}"
HYPER_N_BASIS="${HYPER_N_BASIS:-8}"
HYPER_ADAPTER_BOTTLENECK="${HYPER_ADAPTER_BOTTLENECK:-32}"
HYPER_ADAPTER_SCALE="${HYPER_ADAPTER_SCALE:-1.0}"
SOURCE_EPISODE_PROMPT_POLICY="${SOURCE_EPISODE_PROMPT_POLICY:-context_monthly_prototype}"
SOURCE_ANCHOR_BLEND_CALIBRATION="${SOURCE_ANCHOR_BLEND_CALIBRATION:-1}"
HYPER_OUTPUT_HEAD_RESIDUAL="${HYPER_OUTPUT_HEAD_RESIDUAL:-1}"
ZERO_SHOT_PRIOR_FORM="${ZERO_SHOT_PRIOR_FORM:-source_base_residual_reliability_gated}"
SOURCE_RESIDUAL_RHO="${SOURCE_RESIDUAL_RHO:-1.0}"
SOURCE_RESIDUAL_GATE="${SOURCE_RESIDUAL_GATE:-prompt_reliability_scalar}"
SOURCE_RESIDUAL_GATE_INIT="${SOURCE_RESIDUAL_GATE_INIT:-0.95}"
SOURCE_RESIDUAL_RELIABILITY_DIM="${SOURCE_RESIDUAL_RELIABILITY_DIM:-5}"
DATASET_BACKEND="${DATASET_BACKEND:-auto}"
TENSOR_CACHE_DIR="${TENSOR_CACHE_DIR:-artifacts/region_crops/US}"
MAX_YEAR_CACHE_ENTRIES="${MAX_YEAR_CACHE_ENTRIES:-1}"
TENSOR_CACHE_LOAD_MODE="${TENSOR_CACHE_LOAD_MODE:-eager}"
TRAIN_BATCH_SAMPLER="${TRAIN_BATCH_SAMPLER:-random}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-4}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-1}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-100}"
SOURCE_PROTOTYPE_CACHE_DIR="${SOURCE_PROTOTYPE_CACHE_DIR:-}"
SOURCE_PROTOTYPE_CACHE_MODE="${SOURCE_PROTOTYPE_CACHE_MODE:-off}"
TRAINABLE_SCOPE="source_base_frozen_adapter_film"
RUN_DIR="${OUTPUT_ROOT}/${TARGET_REGION}/${RUN_NAME}"

resolve_auto_source_checkpoint() {
    local target_region="$1"
    local seed="$2"
    local pattern="artifacts/runs/phase4_source_only/*${target_region}_*s${seed}_*/checkpoints/checkpoint_best_source_val_safe_score.pt"
    local checkpoint
    checkpoint="$(find artifacts/runs/phase4_source_only \
        -path "*/checkpoints/checkpoint_best_source_val_safe_score.pt" \
        -path "*${target_region}_*s${seed}_*" \
        -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
    if [[ -z "${checkpoint}" ]]; then
        echo "ERROR: auto source checkpoint not found for target=${target_region} seed=${seed}" >&2
        echo "Looked under pattern: ${pattern}" >&2
        exit 2
    fi
    echo "${checkpoint}"
}

run_or_print() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY_RUN:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

if [[ "${SOURCE_CHECKPOINT}" == "auto" ]]; then
    SOURCE_CHECKPOINT="$(resolve_auto_source_checkpoint "${TARGET_REGION}" "${SEED}")"
fi

if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "ERROR: source checkpoint not found: ${SOURCE_CHECKPOINT}" >&2
    exit 2
fi

RESOLVED_DATASET_BACKEND="${DATASET_BACKEND}"
if [[ "${DATASET_BACKEND}" == "auto" ]]; then
    if [[ -f "${TENSOR_CACHE_DIR}/manifest_region_crops_US.json" ]]; then
        RESOLVED_DATASET_BACKEND="tensor_cache"
    else
        RESOLVED_DATASET_BACKEND="netcdf"
    fi
fi
if [[ "${RESOLVED_DATASET_BACKEND}" != "netcdf" && "${RESOLVED_DATASET_BACKEND}" != "tensor_cache" ]]; then
    echo "ERROR: DATASET_BACKEND must be auto, netcdf, or tensor_cache; got ${DATASET_BACKEND}" >&2
    exit 2
fi
if [[ "${RESOLVED_DATASET_BACKEND}" == "tensor_cache" && ! -f "${TENSOR_CACHE_DIR}/manifest_region_crops_US.json" ]]; then
    echo "ERROR: tensor cache manifest not found: ${TENSOR_CACHE_DIR}/manifest_region_crops_US.json" >&2
    exit 2
fi

if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${RUN_DIR}"
fi

echo "============================================"
echo "Phase 4 staged HyperDA source-stage training"
echo "  target_region=${TARGET_REGION}"
echo "  seed=${SEED}"
echo "  gpu=${CUDA_VISIBLE_DEVICES}"
echo "  stage1_method=source_pooled_global_backbone"
echo "  stage2_frozen=source_base_backbone_and_head"
echo "  stage2_trainable=prompt_encoder,film,basis_adapter_generation,residual_head,reliability_gate_when_enabled"
echo "  adaptation_setting=zero_shot_context  K=0"
echo "  source_fit=2015-2021 source_val=2022"
echo "  target_context=2015-2021 input-side only"
echo "  target_val=unused_in_main_protocol"
echo "  target_eval=2023-2025"
echo "  target_labels=none"
echo "  target_eval_input_stats_used_for_update=false"
echo "  split_artifact=${SPLITS_JSON}"
echo "  source_base_checkpoint=${SOURCE_CHECKPOINT}"
echo "  trainable_scope=${TRAINABLE_SCOPE}"
echo "  source_episode_prompt_policy=${SOURCE_EPISODE_PROMPT_POLICY}"
echo "  source_anchor_blend_calibration=${SOURCE_ANCHOR_BLEND_CALIBRATION}"
echo "  hyper_output_head_residual=${HYPER_OUTPUT_HEAD_RESIDUAL}"
echo "  zero_shot_prior_form=${ZERO_SHOT_PRIOR_FORM}"
echo "  source_residual_rho=${SOURCE_RESIDUAL_RHO}"
echo "  source_residual_gate=${SOURCE_RESIDUAL_GATE}"
echo "  source_residual_gate_init=${SOURCE_RESIDUAL_GATE_INIT}"
echo "  dataset_backend=${RESOLVED_DATASET_BACKEND}"
echo "  tensor_cache_dir=${TENSOR_CACHE_DIR}"
echo "  max_year_cache_entries=${MAX_YEAR_CACHE_ENTRIES}"
echo "  tensor_cache_load_mode=${TENSOR_CACHE_LOAD_MODE}"
echo "  train_batch_sampler=${TRAIN_BATCH_SAMPLER}"
echo "  num_workers=${NUM_WORKERS}"
echo "  eval_every_epochs=${EVAL_EVERY_EPOCHS}"
echo "  log_every_steps=${LOG_EVERY_STEPS}"
echo "  source_prototype_cache_dir=${SOURCE_PROTOTYPE_CACHE_DIR:-none}"
echo "  source_prototype_cache_mode=${SOURCE_PROTOTYPE_CACHE_MODE}"
echo "  model_type=hyperda_basis_adapter width=${WIDTH} prompt_dim=${PROMPT_DIM}"
echo "  hyper_n_basis=${HYPER_N_BASIS} hyper_adapter_bottleneck=${HYPER_ADAPTER_BOTTLENECK}"
echo "  selection_metric=source_val_transfer_safe_score"
echo "  output_dir=${RUN_DIR}"
echo "  dry_run=${DRY_RUN}"
echo "============================================"

cmd=(
    python scripts/train/train_prompt_conditioned_shared.py
    --target_region "${TARGET_REGION}"
    --adaptation_setting zero_shot_context
    --K 0
    --seed "${SEED}"
    --device cuda
    --amp
    --accum_steps "${ACCUM_STEPS}"
    --target_increment_normalization
    --use_lat_weighted_loss
    --batch_size "${BATCH_SIZE}"
    --max_epochs "${MAX_EPOCHS}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --grad_clip 1.0
    --num_workers "${NUM_WORKERS}"
    --width "${WIDTH}"
    --prompt_dim "${PROMPT_DIM}"
    --model_type hyperda_basis_adapter
    --hyper_n_basis "${HYPER_N_BASIS}"
    --hyper_adapter_bottleneck "${HYPER_ADAPTER_BOTTLENECK}"
    --hyper_adapter_scale "${HYPER_ADAPTER_SCALE}"
    --source_episode_prompt_policy "${SOURCE_EPISODE_PROMPT_POLICY}"
    --source_anchor_blend_calibration "${SOURCE_ANCHOR_BLEND_CALIBRATION}"
    --hyper_output_head_residual "${HYPER_OUTPUT_HEAD_RESIDUAL}"
    --zero_shot_prior_form "${ZERO_SHOT_PRIOR_FORM}"
    --source_residual_rho "${SOURCE_RESIDUAL_RHO}"
    --source_residual_gate "${SOURCE_RESIDUAL_GATE}"
    --source_residual_gate_init "${SOURCE_RESIDUAL_GATE_INIT}"
    --source_residual_reliability_dim "${SOURCE_RESIDUAL_RELIABILITY_DIM}"
    --init_from_source_base_checkpoint "${SOURCE_CHECKPOINT}"
    --trainable_scope source_base_frozen_adapter_film
    --log_every_steps "${LOG_EVERY_STEPS}"
    --eval_every_epochs "${EVAL_EVERY_EPOCHS}"
    --checkpoint_every 10
    --selection_metric source_val_transfer_safe_score
    --splits_json "${SPLITS_JSON}"
    --split_manifest_path "${SPLIT_MANIFEST_PATH}"
    --protocol_freeze_id "${PROTOCOL_ID}"
    --dataset_backend "${RESOLVED_DATASET_BACKEND}"
    --tensor_cache_dir "${TENSOR_CACHE_DIR}"
    --max_year_cache_entries "${MAX_YEAR_CACHE_ENTRIES}"
    --tensor_cache_load_mode "${TENSOR_CACHE_LOAD_MODE}"
    --train_batch_sampler "${TRAIN_BATCH_SAMPLER}"
    --source_prototype_cache_mode "${SOURCE_PROTOTYPE_CACHE_MODE}"
    --output_dir "${RUN_DIR}"
)

if [[ -n "${SOURCE_PROTOTYPE_CACHE_DIR}" ]]; then
    cmd+=(--source_prototype_cache_dir "${SOURCE_PROTOTYPE_CACHE_DIR}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    run_or_print "${cmd[@]}"
else
    "${cmd[@]}" 2>&1 | tee "${RUN_DIR}/train_log.txt"
fi

echo "Done: staged HyperDA ${TARGET_REGION} seed=${SEED}"
