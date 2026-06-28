#!/bin/bash
# Stage 3 HyperDA posterior evaluation wrapper.
#
# Paper-facing mode is strict by default:
#   - K-shot runs require SAFE_POLICY_JSON from source-side SAFE calibration.
#   - target_val is unused in the main protocol.
#   - target_eval is final offline evaluation only.
#
# Diagnostic mode can be requested with STAGE3_STRICT_PAPER_POLICY=0. Missing
# SAFE_POLICY_JSON then marks K-shot runs as diagnostic_no_source_safe_policy_json
# and the lower-level runner records the Stage-3 fallback/decision metadata.
#
# Usage:
#   SAFE_POLICY_JSON=/path/to/safe_policy.json \
#     bash run/stage3_hyperda_posterior_eval.sh auto US-R1 0 1
#   STAGE3_STRICT_PAPER_POLICY=0 K_LIST="0" EVAL_MAX_SAMPLES=2 \
#     bash run/stage3_hyperda_posterior_eval.sh /path/to/source.pt US-R1 0 1

set -euo pipefail

SOURCE_CHECKPOINT="${1:-}"
TARGET_REGION="${2:-US-R1}"
SEED="${3:-0}"
CUDA_DEVICE="${4:-${CUDA_VISIBLE_DEVICES:-1}}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

cd "$(dirname "$0")/.."

MODE="${MODE:-full}"
STAGE3_STRICT_PAPER_POLICY="${STAGE3_STRICT_PAPER_POLICY:-1}"
STAGE3_POSTERIOR_POLICY="${STAGE3_POSTERIOR_POLICY:-conservative_coeff_posterior}"
ADAPT_RECIPE="${ADAPT_RECIPE:-source_anchor}"
ADAPT_SCOPE="${ADAPT_SCOPE:-coeff_only}"
FREEZE_MONTHLY_GAIN="${FREEZE_MONTHLY_GAIN:-1}"
SUPPORT_GATE="${SUPPORT_GATE:-auto}"
SUPPORT_GATE_MIN_DELTA="${SUPPORT_GATE_MIN_DELTA:-0.0}"
SUPPORT_GATE_ROOTZONE_TOLERANCE="${SUPPORT_GATE_ROOTZONE_TOLERANCE:-0.0}"
SUPPORT_LOSS_REDUCTION="${SUPPORT_LOSS_REDUCTION:-global_pixel}"
TARGET_CONTEXT_MAX_SAMPLES="${TARGET_CONTEXT_MAX_SAMPLES:-0}"
STAGE3_CONTEXT_TTA="${STAGE3_CONTEXT_TTA:-none}"
SAFE_POLICY_JSON="${SAFE_POLICY_JSON:-}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-stage3_posterior_eval}"

if [[ "${MODE}" == "smoke" ]]; then
    K_LIST="${K_LIST:-0}"
    EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-2}"
    BATCH_SIZE="${BATCH_SIZE:-1}"
    ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-${BATCH_SIZE}}"
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
    REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT:-0}"
else
    K_LIST="${K_LIST:-0 4 12}"
    EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
    BATCH_SIZE="${BATCH_SIZE:-1}"
    ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-${BATCH_SIZE}}"
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
    if [[ "${STAGE3_STRICT_PAPER_POLICY}" == "1" || "${STAGE3_STRICT_PAPER_POLICY,,}" == "true" ]]; then
        REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT:-1}"
    else
        REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT:-0}"
    fi
fi

OUTPUT_BASE="${5:-artifacts/runs/stage3_hyperda_posterior/${TARGET_REGION}_s${SEED}_${MODE}_${TIMESTAMP}}"

resolve_source_checkpoint() {
    local target_region="$1"
    local seed="$2"
    local checkpoint=""

    checkpoint="$(find "artifacts/runs/phase4_hyperda_staged/${target_region}" \
        -path "*s${seed}*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
        -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
    if [[ -n "${checkpoint}" ]]; then
        echo "${checkpoint}"
        return 0
    fi

    checkpoint="$(find artifacts/runs/phase4_hyperda_staged_ablation \
        -path "*/${target_region}/*s${seed}*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
        -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
    if [[ -n "${checkpoint}" ]]; then
        echo "${checkpoint}"
        return 0
    fi

    find artifacts/runs/phase4_prompt_conditioned \
        -path "*hyperda_basis_adapter_${target_region}_*_s${seed}_*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
        -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-
}

if [[ -z "${SOURCE_CHECKPOINT}" || "${SOURCE_CHECKPOINT}" == "auto" ]]; then
    SOURCE_CHECKPOINT="$(resolve_source_checkpoint "${TARGET_REGION}" "${SEED}")"
fi

if [[ -z "${SOURCE_CHECKPOINT}" || ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "ERROR: source HyperDA checkpoint not found." >&2
    echo "Usage:" >&2
    echo "  SAFE_POLICY_JSON=/path/to/safe_policy.json bash run/stage3_hyperda_posterior_eval.sh auto ${TARGET_REGION} ${SEED} ${CUDA_DEVICE}" >&2
    echo "  bash run/stage3_hyperda_posterior_eval.sh /path/to/checkpoint.pt ${TARGET_REGION} ${SEED} ${CUDA_DEVICE}" >&2
    exit 2
fi

has_kshot=0
for K in ${K_LIST}; do
    if [[ "${K}" == "4" || "${K}" == "12" ]]; then
        has_kshot=1
    elif [[ "${K}" != "0" ]]; then
        echo "ERROR: K_LIST may contain only 0, 4, 12; got ${K}" >&2
        exit 2
    fi
done

PAPER_FACING_RUN=0
DIAGNOSTIC_RUN_REASON=""
if [[ "${STAGE3_STRICT_PAPER_POLICY}" == "1" || "${STAGE3_STRICT_PAPER_POLICY,,}" == "true" ]]; then
    PAPER_FACING_RUN=1
    if [[ "${has_kshot}" == "1" && -z "${SAFE_POLICY_JSON}" ]]; then
        echo "ERROR: STAGE3_STRICT_PAPER_POLICY=1 K-shot posterior eval requires SAFE_POLICY_JSON." >&2
        echo "Provide source-side SAFE policy calibration export, or set STAGE3_STRICT_PAPER_POLICY=0 for diagnostic_no_source_safe_policy_json runs." >&2
        exit 2
    fi
else
    PAPER_FACING_RUN=0
    if [[ "${has_kshot}" == "1" && -z "${SAFE_POLICY_JSON}" ]]; then
        DIAGNOSTIC_RUN_REASON="diagnostic_no_source_safe_policy_json"
    else
        DIAGNOSTIC_RUN_REASON="diagnostic_strict_policy_disabled"
    fi
fi

echo "============================================"
echo "Stage 3 HyperDA Posterior Eval"
echo "  source_checkpoint=${SOURCE_CHECKPOINT}"
echo "  target_region=${TARGET_REGION}"
echo "  seed=${SEED}"
echo "  mode=${MODE}"
echo "  K_LIST=${K_LIST}"
echo "  target_context=2015-2021 input-side only"
echo "  stage3_context_tta=${STAGE3_CONTEXT_TTA}"
echo "  target_support=K labeled cycles"
echo "  target_val=unused_in_main_protocol"
echo "  target_eval=2023-2025 final offline evaluation"
echo "  model_selection_source=source_val_preregistered"
echo "  posterior_policy=${STAGE3_POSTERIOR_POLICY}"
echo "  posterior_trainable=adapter_coefficient_residuals_only"
echo "  legacy_diagnostic_policy=safe_operator_ablation"
echo "  adapt_recipe=${ADAPT_RECIPE}"
echo "  adapt_scope=${ADAPT_SCOPE}"
echo "  freeze_monthly_gain=${FREEZE_MONTHLY_GAIN}"
echo "  support_gate=${SUPPORT_GATE} min_delta=${SUPPORT_GATE_MIN_DELTA} rootzone_tolerance=${SUPPORT_GATE_ROOTZONE_TOLERANCE}"
echo "  safe_policy_json=${SAFE_POLICY_JSON:-<none>}"
echo "  strict_paper_policy=${STAGE3_STRICT_PAPER_POLICY}"
echo "  paper_facing_run=${PAPER_FACING_RUN}"
echo "  diagnostic_run_reason=${DIAGNOSTIC_RUN_REASON:-<none>}"
echo "  require_safe_policy_json_for_kshot=${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT}"
echo "  expected_metadata=stage3_posterior_state_dict,stage3_source_prior_unchanged"
echo "  eval_max_samples=${EVAL_MAX_SAMPLES}"
echo "  batch_size=${BATCH_SIZE}"
echo "  adapt_batch_size=${ADAPT_BATCH_SIZE}"
echo "  eval_batch_size=${EVAL_BATCH_SIZE}"
echo "  cuda_device=${CUDA_DEVICE}"
echo "  output_base=${OUTPUT_BASE}"
echo "============================================"

MODE="${MODE}" \
K_LIST="${K_LIST}" \
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES}" \
BATCH_SIZE="${BATCH_SIZE}" \
ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
ADAPT_RECIPE="${ADAPT_RECIPE}" \
ADAPT_SCOPE="${ADAPT_SCOPE}" \
STAGE3_POSTERIOR_POLICY="${STAGE3_POSTERIOR_POLICY}" \
SUPPORT_GATE="${SUPPORT_GATE}" \
SUPPORT_GATE_MIN_DELTA="${SUPPORT_GATE_MIN_DELTA}" \
SUPPORT_GATE_ROOTZONE_TOLERANCE="${SUPPORT_GATE_ROOTZONE_TOLERANCE}" \
SUPPORT_LOSS_REDUCTION="${SUPPORT_LOSS_REDUCTION}" \
FREEZE_MONTHLY_GAIN="${FREEZE_MONTHLY_GAIN}" \
TARGET_CONTEXT_MAX_SAMPLES="${TARGET_CONTEXT_MAX_SAMPLES}" \
STAGE3_CONTEXT_TTA="${STAGE3_CONTEXT_TTA}" \
SAFE_POLICY_JSON="${SAFE_POLICY_JSON}" \
REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT}" \
SCHEDULE_LABEL="${SCHEDULE_LABEL}" \
bash run/phase5_hyperda_zero_few_shot_eval.sh \
    "${SOURCE_CHECKPOINT}" \
    "${TARGET_REGION}" \
    "${SEED}" \
    "${CUDA_DEVICE}" \
    "${OUTPUT_BASE}"
