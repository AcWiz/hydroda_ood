#!/bin/bash
# Stage 3 source-side SAFE policy calibration + US-R1 seed0 eval.
#
# Default source prior:
#   M2_1_rank_gated_dora_stable
#   stable rank-gated bounded-DoRA HyperDA prior + SAFE refinement
#
# Usage:
#   bash run/stage3_calibrate_safe_policy_and_eval_us_r1_seed0.sh [source_checkpoint] [cuda_device] [output_base]
#   SOURCE_QUERY_MAX_SAMPLES=2 EVAL_MAX_SAMPLES=2 K_LIST="0 4 12" \
#     bash run/stage3_calibrate_safe_policy_and_eval_us_r1_seed0.sh

set -euo pipefail

SOURCE_CHECKPOINT="${1:-${SOURCE_CHECKPOINT:-}}"
CUDA_DEVICE="${2:-${CUDA_VISIBLE_DEVICES:-0}}"
RUN_ROOT="${3:-artifacts/runs/stage3_safe_us_r1_seed0_$(date -u +%Y%m%dT%H%M%SZ)}"
TARGET_REGION="${TARGET_REGION:-US-R1}"
SEED="${SEED:-0}"
SOURCE_QUERY_MAX_SAMPLES="${SOURCE_QUERY_MAX_SAMPLES:-256}"
TARGET_CONTEXT_MAX_SAMPLES="${TARGET_CONTEXT_MAX_SAMPLES:-0}"
EVIDENCE_LEVEL="${EVIDENCE_LEVEL:-weaker}"
SOURCE_HELDOUT_CHECKPOINT_MAP="${SOURCE_HELDOUT_CHECKPOINT_MAP:-}"
PSEUDO_TARGET_REGIONS="${PSEUDO_TARGET_REGIONS:-}"
CANDIDATE_SET="${CANDIDATE_SET:-stage3_k0_m2_4a_variable_v1}"
SPLITS_JSON="${SPLITS_JSON:-artifacts/splits/US_loro_zero_few_shot_splits.json}"
ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-${BATCH_SIZE:-8}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE:-8}}"

cd "$(dirname "$0")/.."

if [[ "${TARGET_REGION}" != "US-R1" || "${SEED}" != "0" ]]; then
    echo "ERROR: this wrapper is intentionally scoped to US-R1 seed0. Got TARGET_REGION=${TARGET_REGION} SEED=${SEED}." >&2
    exit 2
fi

resolve_auto_m2_1_source_checkpoint() {
    local target_region="$1"
    local seed="$2"
    find artifacts/runs/phase4_hyperda_staged_ablation/M2_1_rank_gated_dora_stable \
        -path "*/${target_region}/*s${seed}*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
        -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-
}

if [[ -z "${SOURCE_CHECKPOINT}" ]]; then
    SOURCE_CHECKPOINT="$(resolve_auto_m2_1_source_checkpoint "${TARGET_REGION}" "${SEED}")"
fi

if [[ -z "${SOURCE_CHECKPOINT}" || ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "ERROR: M2_1_source_checkpoint not found for ${TARGET_REGION} seed ${SEED}." >&2
    echo "Provide it explicitly or set SOURCE_CHECKPOINT." >&2
    echo "Expected latest checkpoint under:" >&2
    echo "  artifacts/runs/phase4_hyperda_staged_ablation/M2_1_rank_gated_dora_stable/${TARGET_REGION}" >&2
    echo "Train the promoted stable source prior first:" >&2
    echo "  ABLATION_ID=M2_1_rank_gated_dora_stable bash run/phase4_hyperda_staged_ablation.sh auto ${TARGET_REGION} ${SEED} ${CUDA_DEVICE}" >&2
    exit 2
fi

CALIBRATION_ROOT="${RUN_ROOT}/calibration"
SOURCE_ROWS_DIR="${CALIBRATION_ROOT}/source_rows"
CALIBRATION_DIR="${CALIBRATION_ROOT}/policy"
EVAL_OUTPUT_DIR="${RUN_ROOT}/target_eval"
mkdir -p "${SOURCE_ROWS_DIR}" "${CALIBRATION_DIR}" "${EVAL_OUTPUT_DIR}"
PRECOMPUTED_CALIBRATION_ROWS="${PRECOMPUTED_CALIBRATION_ROWS:-}"

ROW_BUILDER_ARGS=(
    --source_checkpoint "${SOURCE_CHECKPOINT}"
    --final_target_region "${TARGET_REGION}"
    --seed "${SEED}"
    --candidate_set "${CANDIDATE_SET}"
    --source_query_max_samples "${SOURCE_QUERY_MAX_SAMPLES}"
    --target_context_max_samples "${TARGET_CONTEXT_MAX_SAMPLES}"
    --evidence_level "${EVIDENCE_LEVEL}"
    --output_dir "${SOURCE_ROWS_DIR}"
    --splits_json "${SPLITS_JSON}"
    --cuda_device "${CUDA_DEVICE}"
    --adapt_batch_size "${ADAPT_BATCH_SIZE}"
    --eval_batch_size "${EVAL_BATCH_SIZE}"
)
CALIBRATE_ARGS=(
    --calibration_rows "${SOURCE_ROWS_DIR}/calibration_rows.csv"
    --output_dir "${CALIBRATION_DIR}"
    --final_target_region "${TARGET_REGION}"
    --seed "${SEED}"
    --source_checkpoint "${SOURCE_CHECKPOINT}"
    --candidate_set "${CANDIDATE_SET}"
    --calibration_stage final
    --source_query_max_samples "${SOURCE_QUERY_MAX_SAMPLES}"
    --source_rows_root "${SOURCE_ROWS_DIR}"
)

if [[ -n "${PSEUDO_TARGET_REGIONS}" ]]; then
    ROW_BUILDER_ARGS+=(--pseudo_target_regions "${PSEUDO_TARGET_REGIONS}")
    CALIBRATE_ARGS+=(--pseudo_target_regions "${PSEUDO_TARGET_REGIONS}")
fi

if [[ "${EVIDENCE_LEVEL}" == "strict" ]]; then
    if [[ -z "${SOURCE_HELDOUT_CHECKPOINT_MAP}" ]]; then
        echo "ERROR: EVIDENCE_LEVEL=strict requires SOURCE_HELDOUT_CHECKPOINT_MAP." >&2
        echo "EVIDENCE_LEVEL=strict SOURCE_HELDOUT_CHECKPOINT_MAP=/path/to/map.json bash run/stage3_calibrate_safe_policy_and_eval_us_r1_seed0.sh" >&2
        exit 2
    fi
    ROW_BUILDER_ARGS+=(--source_heldout_checkpoint_map "${SOURCE_HELDOUT_CHECKPOINT_MAP}")
elif [[ "${EVIDENCE_LEVEL}" == "weaker" ]]; then
    ROW_BUILDER_ARGS+=(--allow_in_checkpoint_source_episodes)
    CALIBRATE_ARGS+=(--allow_in_checkpoint_source_episodes)
else
    echo "ERROR: EVIDENCE_LEVEL must be weaker or strict; got ${EVIDENCE_LEVEL}" >&2
    exit 2
fi

echo "============================================"
echo "Stage 3 source-side SAFE policy calibration + US-R1 seed0 eval"
echo "  source_prior=M2_1_rank_gated_dora_stable"
echo "  method=stable rank-gated bounded-DoRA HyperDA prior + SAFE refinement"
echo "  source_checkpoint=${SOURCE_CHECKPOINT}"
echo "  target_region=${TARGET_REGION}"
echo "  seed=${SEED}"
echo "  evidence_level=${EVIDENCE_LEVEL}"
echo "  source_query_max_samples=${SOURCE_QUERY_MAX_SAMPLES} (0 = full source_val)"
echo "  target_context_max_samples=${TARGET_CONTEXT_MAX_SAMPLES} (0 = full target_context)"
echo "  candidate_set=${CANDIDATE_SET}"
echo "  source_rows_dir=${SOURCE_ROWS_DIR}"
echo "  calibration_dir=${CALIBRATION_DIR}"
echo "  eval_output_dir=${EVAL_OUTPUT_DIR}"
echo "  cuda_device=${CUDA_DEVICE}"
echo "============================================"

if [[ -n "${PRECOMPUTED_CALIBRATION_ROWS}" ]]; then
    cp -f "${PRECOMPUTED_CALIBRATION_ROWS}" "${SOURCE_ROWS_DIR}/calibration_rows.csv"
elif [[ "${CANDIDATE_SET}" == "stage3_k0_m2_4a_variable_v1" ]]; then
    echo "ERROR: CANDIDATE_SET=stage3_k0_m2_4a_variable_v1 requires PRECOMPUTED_CALIBRATION_ROWS with source-episode M2.4a candidate metrics." >&2
    echo "The existing source-row builder only generates K-shot SAFE rows and must not synthesize residual-shrinkage rows without source-base residual records." >&2
    exit 2
else
    PYTHONPATH=. CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python scripts/eval/run_stage3_source_safe_policy_calibration.py \
        "${ROW_BUILDER_ARGS[@]}" \
        2>&1 | tee "${CALIBRATION_ROOT}/source_row_builder.log"
fi

PYTHONPATH=. python scripts/eval/calibrate_source_safe_guard.py \
    "${CALIBRATE_ARGS[@]}" \
    2>&1 | tee "${CALIBRATION_ROOT}/policy_calibration.log"

if [[ ! -f "${CALIBRATION_DIR}/safe_policy.json" ]]; then
    echo "ERROR: expected policy JSON not written: ${CALIBRATION_DIR}/safe_policy.json" >&2
    exit 2
fi

SAFE_POLICY_JSON="${KSHOT_SAFE_POLICY_JSON:-}" \
    STAGE3_K0_CONTEXT_SHRINKAGE="${STAGE3_K0_CONTEXT_SHRINKAGE:-1}" \
    STAGE3_K0_CONTEXT_SHRINKAGE_POLICY="${STAGE3_K0_CONTEXT_SHRINKAGE_POLICY:-source_episode_calibrated_v1}" \
    STAGE3_K0_CONTEXT_SHRINKAGE_POLICY_JSON="${STAGE3_K0_CONTEXT_SHRINKAGE_POLICY_JSON:-${CALIBRATION_DIR}/safe_policy.json}" \
    K_LIST="${K_LIST:-0}" \
    EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}" \
    BATCH_SIZE="${BATCH_SIZE:-1}" \
    ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    bash run/hyperda_safe_us_r1_seed0.sh \
        "${SOURCE_CHECKPOINT}" \
        "${CUDA_DEVICE}" \
        "${EVAL_OUTPUT_DIR}"

echo "safe_policy_json=${CALIBRATION_DIR}/safe_policy.json"
echo "target_eval_output=${EVAL_OUTPUT_DIR}"
