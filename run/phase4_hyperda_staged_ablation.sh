#!/bin/bash
# Phase 4C staged HyperDA-SAFE V1 source-side ablation wrapper.
#
# Usage:
#   ABLATION_ID=M2_rank_gated_dora bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M2_1_rank_gated_dora_stable bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M2_2_source_saliency_prior bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M2_3_source_safe_residual_hyperda bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M2_5a_da_aware_prompt_only bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M2_5b_da_aware_conservative_router bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M2_2_source_saliency_prior HYPER_SOURCE_SALIENCY_PRIOR_PATH=/path/prior.pt bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0 --dry-run

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

ABLATION_ID="${ABLATION_ID:-M0_current}"
SPLITS_JSON="${SPLITS_JSON:-artifacts/splits/US_loro_zero_few_shot_splits.json}"
SPLIT_MANIFEST_PATH="${SPLIT_MANIFEST_PATH:-${SPLITS_JSON}}"
PROTOCOL_ID="${PROTOCOL_ID:-hyperda_v4_4_zero_few_shot_generalization_2015_2025_context2015_2021_sourceval2022_eval2023_2025}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/runs/phase4_hyperda_staged_ablation}"
BUILD_ABLATION_TABLE="${BUILD_ABLATION_TABLE:-1}"
ABLATION_TABLE_OUTPUT_DIR="${ABLATION_TABLE_OUTPUT_DIR:-reports/ablations/hyperda_staged_v1}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
WIDTH="${WIDTH:-32}"
PROMPT_DIM="${PROMPT_DIM:-64}"
HYPER_N_BASIS="${HYPER_N_BASIS:-8}"
HYPER_ADAPTER_BOTTLENECK="${HYPER_ADAPTER_BOTTLENECK:-32}"
HYPER_ADAPTER_SCALE="${HYPER_ADAPTER_SCALE:-1.0}"
HYPER_RANK_GATE_TOP_K="${HYPER_RANK_GATE_TOP_K:-4}"
USER_SET_HYPER_RANK_GATE_TEMPERATURE_INIT="${HYPER_RANK_GATE_TEMPERATURE_INIT+x}"
HYPER_RANK_GATE_TEMPERATURE_INIT="${HYPER_RANK_GATE_TEMPERATURE_INIT:-1.0}"
HYPER_ADAPTER_PARAM_STYLE="${HYPER_ADAPTER_PARAM_STYLE:-basis_1x1}"
HYPER_RELIABILITY_INIT="${HYPER_RELIABILITY_INIT:-0.95}"
HYPER_SOURCE_SALIENCY_PRIOR_PATH="${HYPER_SOURCE_SALIENCY_PRIOR_PATH:-}"
HYPER_SOURCE_SALIENCY_PRIOR_DIR="${HYPER_SOURCE_SALIENCY_PRIOR_DIR:-artifacts/priors/source_basis_saliency}"
HYPER_SOURCE_SALIENCY_AUTO_BUILD="${HYPER_SOURCE_SALIENCY_AUTO_BUILD:-1}"
HYPER_SOURCE_SALIENCY_MAX_BATCHES="${HYPER_SOURCE_SALIENCY_MAX_BATCHES:-16}"
HYPER_SOURCE_SALIENCY_BATCH_SIZE="${HYPER_SOURCE_SALIENCY_BATCH_SIZE:-4}"
HYPER_SOURCE_SALIENCY_PRIOR_BETA="${HYPER_SOURCE_SALIENCY_PRIOR_BETA:-0.0}"
HYPER_SOURCE_SALIENCY_PRIOR_APPLICATION="${HYPER_SOURCE_SALIENCY_PRIOR_APPLICATION:-soft_regularization_metadata}"
HYPER_PROMPT_MANIFOLD_RELIABILITY="${HYPER_PROMPT_MANIFOLD_RELIABILITY:-0}"
HYPER_PROMPT_MANIFOLD_RELIABILITY_STRENGTH="${HYPER_PROMPT_MANIFOLD_RELIABILITY_STRENGTH:-0.0}"
HYPER_RESIDUAL_MAGNITUDE_PENALTY="${HYPER_RESIDUAL_MAGNITUDE_PENALTY:-0.0}"
HYPER_COEFF_ENTROPY_FLOOR="${HYPER_COEFF_ENTROPY_FLOOR:-0.0}"
HYPER_COEFF_ENTROPY_PENALTY="${HYPER_COEFF_ENTROPY_PENALTY:-0.0}"
CONTEXT_ENCODER="${CONTEXT_ENCODER:-current_mean_std}"
M2_3_INIT_FROM_M2_1_CHECKPOINT="${M2_3_INIT_FROM_M2_1_CHECKPOINT-auto}"
TARGET_EVAL_INPUT_STATS_USED_FOR_UPDATE="false"
SOURCE_EPISODE_PROMPT_POLICY="${SOURCE_EPISODE_PROMPT_POLICY:-context_monthly_prototype}"
SOURCE_ANCHOR_BLEND_CALIBRATION="${SOURCE_ANCHOR_BLEND_CALIBRATION:-1}"
HYPER_OUTPUT_HEAD_RESIDUAL="${HYPER_OUTPUT_HEAD_RESIDUAL:-1}"
ZERO_SHOT_PRIOR_FORM="${ZERO_SHOT_PRIOR_FORM:-direct_hyper}"
SOURCE_RESIDUAL_RHO="${SOURCE_RESIDUAL_RHO:-0.0}"
SOURCE_RESIDUAL_GATE="${SOURCE_RESIDUAL_GATE:-prompt_reliability_scalar}"
SOURCE_RESIDUAL_GATE_INIT="${SOURCE_RESIDUAL_GATE_INIT:-0.95}"
SOURCE_RESIDUAL_RELIABILITY_DIM="${SOURCE_RESIDUAL_RELIABILITY_DIM:-5}"
DATASET_BACKEND="${DATASET_BACKEND:-auto}"
TENSOR_CACHE_DIR="${TENSOR_CACHE_DIR:-artifacts/region_crops/US}"
MAX_YEAR_CACHE_ENTRIES="${MAX_YEAR_CACHE_ENTRIES:-2}"
TENSOR_CACHE_LOAD_MODE="${TENSOR_CACHE_LOAD_MODE:-mmap}"
TRAIN_BATCH_SAMPLER="${TRAIN_BATCH_SAMPLER:-source_region_year_grouped}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
USER_SET_USE_AMP="${USE_AMP+x}"
USER_SET_BATCH_SIZE="${BATCH_SIZE+x}"
USER_SET_ACCUM_STEPS="${ACCUM_STEPS+x}"
USER_SET_LR="${LR+x}"
USE_AMP="${USE_AMP:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ACCUM_STEPS="${ACCUM_STEPS:-2}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-5}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-200}"
SOURCE_PROTOTYPE_CACHE_DIR="${SOURCE_PROTOTYPE_CACHE_DIR:-artifacts/cache/source_context_monthly_prototypes}"
SOURCE_PROTOTYPE_CACHE_MODE="${SOURCE_PROTOTYPE_CACHE_MODE:-read_write}"
TRAINABLE_SCOPE="source_base_frozen_adapter_film"

case "${ABLATION_ID}" in
    M0_current)
        HYPER_COEFF_GENERATOR=per_adapter
        HYPER_RELIABILITY_GATE=none
        HYPER_ENABLE_FILM=1
        HYPER_ENABLE_ADAPTERS=1
        ;;
    M1_shared_coeff)
        HYPER_COEFF_GENERATOR=shared_layer_aware
        HYPER_RELIABILITY_GATE=none
        HYPER_ENABLE_FILM=1
        HYPER_ENABLE_ADAPTERS=1
        ;;
    M2_shared_coeff_gate)
        HYPER_COEFF_GENERATOR=shared_layer_aware
        HYPER_RELIABILITY_GATE=prompt_scalar
        HYPER_ENABLE_FILM=1
        HYPER_ENABLE_ADAPTERS=1
        ;;
    M2_rank_gated_dora)
        HYPER_COEFF_GENERATOR=shared_layer_aware_rank_gated
        HYPER_RELIABILITY_GATE=prompt_scalar
        HYPER_ADAPTER_PARAM_STYLE=dora_like_gain
        HYPER_ENABLE_FILM=1
        HYPER_ENABLE_ADAPTERS=1
        ;;
    M2_1_rank_gated_dora_stable)
        HYPER_COEFF_GENERATOR=shared_layer_aware_rank_gated_stable
        HYPER_RELIABILITY_GATE=prompt_scalar
        HYPER_ADAPTER_PARAM_STYLE=dora_like_gain_bounded
        ZERO_SHOT_PRIOR_FORM=direct_hyper
        SOURCE_RESIDUAL_RHO=0.0
        HYPER_RESIDUAL_MAGNITUDE_PENALTY=0.0
        HYPER_COEFF_ENTROPY_FLOOR=0.0
        HYPER_COEFF_ENTROPY_PENALTY=0.0
        if [[ -z "${USER_SET_HYPER_RANK_GATE_TEMPERATURE_INIT}" ]]; then
            HYPER_RANK_GATE_TEMPERATURE_INIT=2.0
        fi
        if [[ -z "${USER_SET_USE_AMP}" ]]; then
            USE_AMP=0
        fi
        if [[ -z "${USER_SET_BATCH_SIZE}" ]]; then
            BATCH_SIZE=64
        fi
        if [[ -z "${USER_SET_ACCUM_STEPS}" ]]; then
            ACCUM_STEPS=2
        fi
        if [[ -z "${USER_SET_LR}" ]]; then
            LR=2e-4
        fi
        HYPER_ENABLE_FILM=1
        HYPER_ENABLE_ADAPTERS=1
        ;;
    M2_2_source_saliency_prior)
        HYPER_COEFF_GENERATOR=shared_layer_aware_rank_gated_stable
        HYPER_RELIABILITY_GATE=prompt_scalar
        HYPER_ADAPTER_PARAM_STYLE=dora_like_gain_bounded
        if [[ -z "${USER_SET_HYPER_RANK_GATE_TEMPERATURE_INIT}" ]]; then
            HYPER_RANK_GATE_TEMPERATURE_INIT=2.0
        fi
        if [[ -z "${USER_SET_USE_AMP}" ]]; then
            USE_AMP=0
        fi
        if [[ -z "${USER_SET_BATCH_SIZE}" ]]; then
            BATCH_SIZE=64
        fi
        if [[ -z "${USER_SET_ACCUM_STEPS}" ]]; then
            ACCUM_STEPS=2
        fi
        if [[ -z "${USER_SET_LR}" ]]; then
            LR=2e-4
        fi
        if [[ "${HYPER_SOURCE_SALIENCY_PRIOR_BETA}" == "0.0" || "${HYPER_SOURCE_SALIENCY_PRIOR_BETA}" == "0" ]]; then
            HYPER_SOURCE_SALIENCY_PRIOR_BETA=0.5
        fi
        HYPER_SOURCE_SALIENCY_PRIOR_APPLICATION=legacy_gate_logit_bias_before_topk
        if [[ "${HYPER_PROMPT_MANIFOLD_RELIABILITY}" == "0" ]]; then
            HYPER_PROMPT_MANIFOLD_RELIABILITY=1
        fi
        if [[ "${HYPER_PROMPT_MANIFOLD_RELIABILITY_STRENGTH}" == "0.0" || "${HYPER_PROMPT_MANIFOLD_RELIABILITY_STRENGTH}" == "0" ]]; then
            HYPER_PROMPT_MANIFOLD_RELIABILITY_STRENGTH=0.5
        fi
        HYPER_ENABLE_FILM=1
        HYPER_ENABLE_ADAPTERS=1
        ;;
    M2_3_source_safe_residual_hyperda)
        HYPER_COEFF_GENERATOR=shared_layer_aware_rank_gated_stable
        HYPER_RELIABILITY_GATE=prompt_scalar
        HYPER_ADAPTER_PARAM_STYLE=dora_like_gain_bounded
        if [[ -z "${USER_SET_HYPER_RANK_GATE_TEMPERATURE_INIT}" ]]; then
            HYPER_RANK_GATE_TEMPERATURE_INIT=2.0
        fi
        if [[ -z "${USER_SET_USE_AMP}" ]]; then
            USE_AMP=0
        fi
        if [[ -z "${USER_SET_BATCH_SIZE}" ]]; then
            BATCH_SIZE=64
        fi
        if [[ -z "${USER_SET_ACCUM_STEPS}" ]]; then
            ACCUM_STEPS=2
        fi
        if [[ -z "${USER_SET_LR}" ]]; then
            LR=2e-4
        fi
        ZERO_SHOT_PRIOR_FORM=source_base_residual_reliability_gated
        SOURCE_RESIDUAL_GATE=prompt_reliability_scalar
        HYPER_SOURCE_SALIENCY_PRIOR_APPLICATION=soft_regularization_metadata
        if [[ "${HYPER_RESIDUAL_MAGNITUDE_PENALTY}" == "0.0" || "${HYPER_RESIDUAL_MAGNITUDE_PENALTY}" == "0" ]]; then
            HYPER_RESIDUAL_MAGNITUDE_PENALTY=0.001
        fi
        if [[ "${HYPER_COEFF_ENTROPY_FLOOR}" == "0.0" || "${HYPER_COEFF_ENTROPY_FLOOR}" == "0" ]]; then
            HYPER_COEFF_ENTROPY_FLOOR=0.5
        fi
        if [[ "${HYPER_COEFF_ENTROPY_PENALTY}" == "0.0" || "${HYPER_COEFF_ENTROPY_PENALTY}" == "0" ]]; then
            HYPER_COEFF_ENTROPY_PENALTY=0.0001
        fi
        HYPER_ENABLE_FILM=1
        HYPER_ENABLE_ADAPTERS=1
        ;;
    M2_5a_da_aware_prompt_only)
        HYPER_COEFF_GENERATOR=shared_layer_aware_rank_gated_stable
        HYPER_RELIABILITY_GATE=prompt_scalar
        HYPER_ADAPTER_PARAM_STYLE=dora_like_gain_bounded
        CONTEXT_ENCODER=robust_input_side_da_diagnostics
        ZERO_SHOT_PRIOR_FORM=direct_hyper
        SOURCE_RESIDUAL_RHO=0.0
        HYPER_RESIDUAL_MAGNITUDE_PENALTY=0.0
        HYPER_COEFF_ENTROPY_FLOOR=0.0
        HYPER_COEFF_ENTROPY_PENALTY=0.0
        if [[ -z "${USER_SET_HYPER_RANK_GATE_TEMPERATURE_INIT}" ]]; then
            HYPER_RANK_GATE_TEMPERATURE_INIT=2.0
        fi
        if [[ -z "${USER_SET_USE_AMP}" ]]; then
            USE_AMP=0
        fi
        if [[ -z "${USER_SET_BATCH_SIZE}" ]]; then
            BATCH_SIZE=64
        fi
        if [[ -z "${USER_SET_ACCUM_STEPS}" ]]; then
            ACCUM_STEPS=2
        fi
        if [[ -z "${USER_SET_LR}" ]]; then
            LR=2e-4
        fi
        HYPER_ENABLE_FILM=1
        HYPER_ENABLE_ADAPTERS=1
        ;;
    M2_5b_da_aware_conservative_router)
        HYPER_COEFF_GENERATOR=shared_layer_aware_rank_gated_stable
        HYPER_RELIABILITY_GATE=prompt_scalar
        HYPER_ADAPTER_PARAM_STYLE=dora_like_gain_bounded
        CONTEXT_ENCODER=robust_input_side_da_diagnostics_raw
        ZERO_SHOT_PRIOR_FORM=source_base_residual_reliability_gated
        SOURCE_RESIDUAL_RHO=1.0
        SOURCE_RESIDUAL_GATE=prompt_reliability_scalar
        SOURCE_RESIDUAL_GATE_INIT=0.90
        HYPER_SOURCE_SALIENCY_PRIOR_APPLICATION=soft_regularization_metadata
        HYPER_RESIDUAL_MAGNITUDE_PENALTY=0.0
        HYPER_COEFF_ENTROPY_FLOOR=0.0
        HYPER_COEFF_ENTROPY_PENALTY=0.0
        HYPER_PROMPT_MANIFOLD_RELIABILITY=1
        HYPER_PROMPT_MANIFOLD_RELIABILITY_STRENGTH=0.25
        if [[ -z "${USER_SET_HYPER_RANK_GATE_TEMPERATURE_INIT}" ]]; then
            HYPER_RANK_GATE_TEMPERATURE_INIT=2.0
        fi
        if [[ -z "${USER_SET_USE_AMP}" ]]; then
            USE_AMP=0
        fi
        if [[ -z "${USER_SET_BATCH_SIZE}" ]]; then
            BATCH_SIZE=64
        fi
        if [[ -z "${USER_SET_ACCUM_STEPS}" ]]; then
            ACCUM_STEPS=2
        fi
        if [[ -z "${USER_SET_LR}" ]]; then
            LR=2e-4
        fi
        HYPER_ENABLE_FILM=1
        HYPER_ENABLE_ADAPTERS=1
        ;;
    M3_film_only)
        HYPER_COEFF_GENERATOR=per_adapter
        HYPER_RELIABILITY_GATE=none
        HYPER_ENABLE_FILM=1
        HYPER_ENABLE_ADAPTERS=0
        ;;
    M4_adapter_only)
        HYPER_COEFF_GENERATOR=shared_layer_aware
        HYPER_RELIABILITY_GATE=none
        HYPER_ENABLE_FILM=0
        HYPER_ENABLE_ADAPTERS=1
        ;;
    *)
        echo "ERROR: Unsupported ABLATION_ID=${ABLATION_ID}" >&2
        echo "Expected one of: M0_current M1_shared_coeff M2_shared_coeff_gate M2_rank_gated_dora M2_1_rank_gated_dora_stable M2_2_source_saliency_prior M2_3_source_safe_residual_hyperda M2_5a_da_aware_prompt_only M2_5b_da_aware_conservative_router M3_film_only M4_adapter_only" >&2
        echo "M2.4 is a Stage 3 K=0 target-context diagnostic, not a source-stage ablation." >&2
        exit 2
        ;;
esac

RUN_NAME="${RUN_NAME:-phase4_hyperda_staged_ablation_${ABLATION_ID}_${TARGET_REGION}_s${SEED}_${TIMESTAMP}}"
RUN_DIR="${OUTPUT_ROOT}/${ABLATION_ID}/${TARGET_REGION}/${RUN_NAME}"

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

resolve_auto_m2_1_saliency_source_checkpoint() {
    local target_region="$1"
    local seed="$2"
    find artifacts/runs/phase4_hyperda_staged_ablation/M2_1_rank_gated_dora_stable \
        -path "*/${target_region}/*s${seed}*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
        -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-
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

if [[ "${ABLATION_ID}" == "M2_2_source_saliency_prior" ]]; then
    if [[ -z "${HYPER_SOURCE_SALIENCY_PRIOR_PATH}" ]]; then
        HYPER_SOURCE_SALIENCY_PRIOR_PATH="${HYPER_SOURCE_SALIENCY_PRIOR_DIR}/${TARGET_REGION}_s${SEED}.pt"
    fi
fi

if [[ "${ABLATION_ID}" == "M2_3_source_safe_residual_hyperda" && "${M2_3_INIT_FROM_M2_1_CHECKPOINT}" == "auto" ]]; then
    M2_3_INIT_FROM_M2_1_CHECKPOINT="$(resolve_auto_m2_1_saliency_source_checkpoint "${TARGET_REGION}" "${SEED}")"
fi
if [[ "${ABLATION_ID}" == "M2_3_source_safe_residual_hyperda" && -n "${M2_3_INIT_FROM_M2_1_CHECKPOINT}" && ! -f "${M2_3_INIT_FROM_M2_1_CHECKPOINT}" ]]; then
    echo "ERROR: M2.3 M2.1 source prior checkpoint not found: ${M2_3_INIT_FROM_M2_1_CHECKPOINT}" >&2
    echo "Run M2.1 first or set M2_3_INIT_FROM_M2_1_CHECKPOINT= to train from the staged source base only." >&2
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

if [[ "${ABLATION_ID}" == "M2_2_source_saliency_prior" && "${DRY_RUN}" != "1" && ! -f "${HYPER_SOURCE_SALIENCY_PRIOR_PATH}" ]]; then
    M2_1_SALIENCY_SOURCE_CHECKPOINT="$(resolve_auto_m2_1_saliency_source_checkpoint "${TARGET_REGION}" "${SEED}")"
    if [[ -z "${M2_1_SALIENCY_SOURCE_CHECKPOINT}" ]]; then
        echo "ERROR: M2_2_source_saliency_prior source saliency prior not found: ${HYPER_SOURCE_SALIENCY_PRIOR_PATH}" >&2
        echo "" >&2
        echo "No matching M2.1 stable HyperDA source-stage checkpoint was found under:" >&2
        echo "  artifacts/runs/phase4_hyperda_staged_ablation/M2_1_rank_gated_dora_stable/${TARGET_REGION}" >&2
        echo "" >&2
        echo "Run M2.1 first, then build the prior:" >&2
        echo "  ABLATION_ID=M2_1_rank_gated_dora_stable bash $0 auto ${TARGET_REGION} ${SEED} ${CUDA_VISIBLE_DEVICES}" >&2
        echo "  PYTHONPATH=. python scripts/train/build_source_basis_saliency_prior.py \\" >&2
        echo "    --source_checkpoint /path/to/M2_1/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt \\" >&2
        echo "    --target_region \"${TARGET_REGION}\" \\" >&2
        echo "    --output_path \"${HYPER_SOURCE_SALIENCY_PRIOR_PATH}\" \\" >&2
        echo "    --source_split source_fit" >&2
        echo "" >&2
        echo "Or pass an existing artifact explicitly:" >&2
        echo "  HYPER_SOURCE_SALIENCY_PRIOR_PATH=/path/to/prior.pt ABLATION_ID=M2_2_source_saliency_prior bash $0 ${SOURCE_CHECKPOINT} ${TARGET_REGION} ${SEED} ${CUDA_VISIBLE_DEVICES}" >&2
        exit 2
    fi
    if [[ "${HYPER_SOURCE_SALIENCY_AUTO_BUILD}" == "1" || "${HYPER_SOURCE_SALIENCY_AUTO_BUILD,,}" == "true" ]]; then
        echo "M2.2 source saliency prior not found; building it from M2.1 stable checkpoint:"
        echo "  source_checkpoint=${M2_1_SALIENCY_SOURCE_CHECKPOINT}"
        echo "  output_path=${HYPER_SOURCE_SALIENCY_PRIOR_PATH}"
        mkdir -p "$(dirname "${HYPER_SOURCE_SALIENCY_PRIOR_PATH}")"
        python scripts/train/build_source_basis_saliency_prior.py \
            --source_checkpoint "${M2_1_SALIENCY_SOURCE_CHECKPOINT}" \
            --target_region "${TARGET_REGION}" \
            --output_path "${HYPER_SOURCE_SALIENCY_PRIOR_PATH}" \
            --source_split source_fit \
            --seed "${SEED}" \
            --dataset_backend "${RESOLVED_DATASET_BACKEND}" \
            --tensor_cache_dir "${TENSOR_CACHE_DIR}" \
            --max_year_cache_entries "${MAX_YEAR_CACHE_ENTRIES}" \
            --tensor_cache_load_mode "${TENSOR_CACHE_LOAD_MODE}" \
            --batch_size "${HYPER_SOURCE_SALIENCY_BATCH_SIZE}" \
            --max_batches "${HYPER_SOURCE_SALIENCY_MAX_BATCHES}" \
            --num_workers "${NUM_WORKERS}" \
            --device cuda
    else
        echo "ERROR: M2_2_source_saliency_prior source saliency prior not found: ${HYPER_SOURCE_SALIENCY_PRIOR_PATH}" >&2
        echo "" >&2
        echo "Auto-build is disabled by HYPER_SOURCE_SALIENCY_AUTO_BUILD=${HYPER_SOURCE_SALIENCY_AUTO_BUILD}." >&2
        echo "Build it manually with:" >&2
        echo "  PYTHONPATH=. python scripts/train/build_source_basis_saliency_prior.py \\" >&2
        echo "    --source_checkpoint \"${M2_1_SALIENCY_SOURCE_CHECKPOINT}\" \\" >&2
        echo "    --target_region \"${TARGET_REGION}\" \\" >&2
        echo "    --output_path \"${HYPER_SOURCE_SALIENCY_PRIOR_PATH}\" \\" >&2
        echo "    --source_split source_fit" >&2
        exit 2
    fi
fi

if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${RUN_DIR}"
fi

echo "============================================"
echo "Phase 4 staged HyperDA-SAFE V1 ablation"
echo "  ablation_id=${ABLATION_ID}"
echo "  target_region=${TARGET_REGION}"
echo "  seed=${SEED}"
echo "  gpu=${CUDA_VISIBLE_DEVICES}"
echo "  stage1_method=source_pooled_global_backbone"
echo "  stage2_frozen=source_base_backbone_and_head"
echo "  stage2_trainable=prompt_encoder,film,basis_adapter_generation,reliability_gate_when_enabled"
echo "  adaptation_setting=zero_shot_context  K=0"
echo "  source_fit=2015-2021 source_val=2022"
echo "  target_context=2015-2021 input-side only"
echo "  target_val=unused_in_main_protocol"
echo "  target_eval=2023-2025"
echo "  target_labels=none"
echo "  split_artifact=${SPLITS_JSON}"
echo "  source_base_checkpoint=${SOURCE_CHECKPOINT}"
if [[ "${ABLATION_ID}" == "M2_3_source_safe_residual_hyperda" ]]; then
    echo "  m2_3_init_from_m2_1_checkpoint=${M2_3_INIT_FROM_M2_1_CHECKPOINT:-none}"
fi
echo "  target_labels_used_for_adaptation=false"
echo "  target_eval_input_stats_used_for_update=false"
echo "  trainable_scope=${TRAINABLE_SCOPE}"
echo "  context_encoder=${CONTEXT_ENCODER}"
if [[ "${CONTEXT_ENCODER}" == "robust_input_side_da_diagnostics" ]]; then
    echo "  prompt_diagnostics=DA-aware prompt/router diagnostics from input-side fields only"
    echo "  prompt_diagnostic_input_domain=normalized_input_side_legacy"
elif [[ "${CONTEXT_ENCODER}" == "robust_input_side_da_diagnostics_raw" ]]; then
    echo "  prompt_diagnostics=DA-aware prompt/router diagnostics from raw input-side fields only"
    echo "  prompt_diagnostic_input_domain=raw_input_side"
fi
if [[ "${ABLATION_ID}" == "M2_5a_da_aware_prompt_only" ]]; then
    echo "  m2_5a_failure_interpretation=source_val_improved_target_k0_degraded"
    echo "  diagnostic_status=negative_diagnostic_non_strict_prompt_only"
elif [[ "${ABLATION_ID}" == "M2_5b_da_aware_conservative_router" ]]; then
    echo "  m2_5a_failure_interpretation=source_val_improved_target_k0_degraded"
    echo "  diagnostic_status=da_aware_conservative_router_source_stage_diagnostic"
fi
echo "  hyper_coeff_generator=${HYPER_COEFF_GENERATOR}"
echo "  hyper_reliability_gate=${HYPER_RELIABILITY_GATE}"
echo "  hyper_reliability_init=${HYPER_RELIABILITY_INIT}"
echo "  hyper_rank_gate_top_k=${HYPER_RANK_GATE_TOP_K}"
echo "  hyper_rank_gate_temperature_init=${HYPER_RANK_GATE_TEMPERATURE_INIT}"
echo "  hyper_adapter_param_style=${HYPER_ADAPTER_PARAM_STYLE}"
echo "  hyper_source_saliency_prior_path=${HYPER_SOURCE_SALIENCY_PRIOR_PATH:-none}"
echo "  hyper_source_saliency_auto_build=${HYPER_SOURCE_SALIENCY_AUTO_BUILD}"
echo "  hyper_source_saliency_max_batches=${HYPER_SOURCE_SALIENCY_MAX_BATCHES}"
echo "  hyper_source_saliency_batch_size=${HYPER_SOURCE_SALIENCY_BATCH_SIZE}"
echo "  hyper_source_saliency_prior_beta=${HYPER_SOURCE_SALIENCY_PRIOR_BETA}"
echo "  hyper_source_saliency_prior_application=${HYPER_SOURCE_SALIENCY_PRIOR_APPLICATION}"
echo "  hyper_prompt_manifold_reliability=${HYPER_PROMPT_MANIFOLD_RELIABILITY}"
echo "  hyper_prompt_manifold_reliability_strength=${HYPER_PROMPT_MANIFOLD_RELIABILITY_STRENGTH}"
echo "  hyper_residual_magnitude_penalty=${HYPER_RESIDUAL_MAGNITUDE_PENALTY}"
echo "  hyper_coeff_entropy_floor=${HYPER_COEFF_ENTROPY_FLOOR}"
echo "  hyper_coeff_entropy_penalty=${HYPER_COEFF_ENTROPY_PENALTY}"
echo "  hyper_enable_film=${HYPER_ENABLE_FILM}"
echo "  hyper_enable_adapters=${HYPER_ENABLE_ADAPTERS}"
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
echo "  use_amp=${USE_AMP}"
echo "  eval_every_epochs=${EVAL_EVERY_EPOCHS}"
echo "  log_every_steps=${LOG_EVERY_STEPS}"
echo "  source_prototype_cache_dir=${SOURCE_PROTOTYPE_CACHE_DIR:-none}"
echo "  source_prototype_cache_mode=${SOURCE_PROTOTYPE_CACHE_MODE}"
echo "  model_type=hyperda_basis_adapter width=${WIDTH} prompt_dim=${PROMPT_DIM}"
echo "  hyper_n_basis=${HYPER_N_BASIS} hyper_adapter_bottleneck=${HYPER_ADAPTER_BOTTLENECK}"
echo "  batch_size=${BATCH_SIZE} accum_steps=${ACCUM_STEPS} lr=${LR}"
echo "  selection_metric=source_val_transfer_safe_score"
echo "  output_dir=${RUN_DIR}"
echo "  build_ablation_table=${BUILD_ABLATION_TABLE}"
echo "  ablation_table_output_dir=${ABLATION_TABLE_OUTPUT_DIR}"
echo "  dry_run=${DRY_RUN}"
echo "============================================"

cmd=(
    python scripts/train/train_prompt_conditioned_shared.py
    --target_region "${TARGET_REGION}"
    --adaptation_setting zero_shot_context
    --K 0
    --seed "${SEED}"
    --device cuda
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
    --context_encoder "${CONTEXT_ENCODER}"
    --model_type hyperda_basis_adapter
    --hyper_n_basis "${HYPER_N_BASIS}"
    --hyper_adapter_bottleneck "${HYPER_ADAPTER_BOTTLENECK}"
    --hyper_adapter_scale "${HYPER_ADAPTER_SCALE}"
    --hyper_coeff_generator "${HYPER_COEFF_GENERATOR}"
    --hyper_rank_gate_top_k "${HYPER_RANK_GATE_TOP_K}"
    --hyper_rank_gate_temperature_init "${HYPER_RANK_GATE_TEMPERATURE_INIT}"
    --hyper_adapter_param_style "${HYPER_ADAPTER_PARAM_STYLE}"
    --hyper_reliability_gate "${HYPER_RELIABILITY_GATE}"
    --hyper_reliability_init "${HYPER_RELIABILITY_INIT}"
    --hyper_source_saliency_prior_path "${HYPER_SOURCE_SALIENCY_PRIOR_PATH}"
    --hyper_source_saliency_prior_beta "${HYPER_SOURCE_SALIENCY_PRIOR_BETA}"
    --hyper_source_saliency_prior_application "${HYPER_SOURCE_SALIENCY_PRIOR_APPLICATION}"
    --hyper_prompt_manifold_reliability "${HYPER_PROMPT_MANIFOLD_RELIABILITY}"
    --hyper_prompt_manifold_reliability_strength "${HYPER_PROMPT_MANIFOLD_RELIABILITY_STRENGTH}"
    --hyper_enable_film "${HYPER_ENABLE_FILM}"
    --hyper_enable_adapters "${HYPER_ENABLE_ADAPTERS}"
    --hyper_residual_magnitude_penalty "${HYPER_RESIDUAL_MAGNITUDE_PENALTY}"
    --hyper_coeff_entropy_floor "${HYPER_COEFF_ENTROPY_FLOOR}"
    --hyper_coeff_entropy_penalty "${HYPER_COEFF_ENTROPY_PENALTY}"
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

if [[ "${USE_AMP}" == "1" || "${USE_AMP,,}" == "true" ]]; then
    cmd+=(--amp)
fi

if [[ "${ABLATION_ID}" == "M2_3_source_safe_residual_hyperda" && -n "${M2_3_INIT_FROM_M2_1_CHECKPOINT}" ]]; then
    cmd+=(--init_from_prompt_checkpoint "${M2_3_INIT_FROM_M2_1_CHECKPOINT}")
fi

if [[ -n "${SOURCE_PROTOTYPE_CACHE_DIR}" ]]; then
    cmd+=(--source_prototype_cache_dir "${SOURCE_PROTOTYPE_CACHE_DIR}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    run_or_print "${cmd[@]}"
else
    "${cmd[@]}" 2>&1 | tee "${RUN_DIR}/train_log.txt"
fi

table_cmd=(
    python scripts/analysis/build_hyperda_staged_ablation_table.py
    --runs_root "${OUTPUT_ROOT}"
    --output_dir "${ABLATION_TABLE_OUTPUT_DIR}"
    --target_region "${TARGET_REGION}"
    --seed "${SEED}"
)

if [[ "${BUILD_ABLATION_TABLE}" == "1" || "${BUILD_ABLATION_TABLE,,}" == "true" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
        run_or_print "${table_cmd[@]}"
    else
        "${table_cmd[@]}"
    fi
fi

echo "Done: staged HyperDA ablation ${ABLATION_ID} ${TARGET_REGION} seed=${SEED}"
