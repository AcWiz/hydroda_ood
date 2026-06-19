#!/bin/bash
# Full diagnostic Stage 3 inference wrapper for staged HyperDA.
#
# This runs the complete target_eval path by default:
#   K_LIST=0 4 12, EVAL_MAX_SAMPLES=0, conservative posterior policy.
# It is meant to find Stage 3 integration/evaluation issues early. It is not a
# paper-facing run unless SAFE_POLICY_JSON is supplied and strict policy is set.

set -euo pipefail

SOURCE_CHECKPOINT="${1:-}"
TARGET_REGION="${2:-US-R1}"
SEED="${3:-0}"
CUDA_DEVICE="${4:-${CUDA_VISIBLE_DEVICES:-1}}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

cd "$(dirname "$0")/.."

MODE="${MODE:-full}"
K_LIST="${K_LIST:-0 4 12}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
STAGE3_STRICT_PAPER_POLICY="${STAGE3_STRICT_PAPER_POLICY:-0}"
REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT:-0}"
STAGE3_POSTERIOR_POLICY="${STAGE3_POSTERIOR_POLICY:-conservative_coeff_posterior}"
SUPPORT_GATE="${SUPPORT_GATE:-auto}"
STAGE3_K0_CONTEXT_SHRINKAGE="${STAGE3_K0_CONTEXT_SHRINKAGE:-0}"
STAGE3_K0_CONTEXT_SHRINKAGE_RHO_CAP="${STAGE3_K0_CONTEXT_SHRINKAGE_RHO_CAP:-1.0}"
STAGE3_K0_CONTEXT_SHRINKAGE_POLICY="${STAGE3_K0_CONTEXT_SHRINKAGE_POLICY:-variable_reliability_v1}"
STAGE3_K0_CONTEXT_SHRINKAGE_SURFACE_RHO_CAP="${STAGE3_K0_CONTEXT_SHRINKAGE_SURFACE_RHO_CAP:-${STAGE3_K0_CONTEXT_SHRINKAGE_RHO_CAP}}"
STAGE3_K0_CONTEXT_SHRINKAGE_ROOTZONE_RHO_CAP="${STAGE3_K0_CONTEXT_SHRINKAGE_ROOTZONE_RHO_CAP:-${STAGE3_K0_CONTEXT_SHRINKAGE_RHO_CAP}}"
STAGE3_K0_CONTEXT_SHRINKAGE_POLICY_JSON="${STAGE3_K0_CONTEXT_SHRINKAGE_POLICY_JSON:-}"
SNAPSHOT_STAGE2_CHECKPOINT="${SNAPSHOT_STAGE2_CHECKPOINT:-1}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-diagnostic_full_inference}"

if [[ -z "${SOURCE_CHECKPOINT}" || "${SOURCE_CHECKPOINT}" == "auto" ]]; then
    SOURCE_CHECKPOINT="$(find "artifacts/runs/phase4_hyperda_staged/${TARGET_REGION}" \
        -path "*s${SEED}*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
        -type f 2>/dev/null | sort | tail -1)"
fi

if [[ -z "${SOURCE_CHECKPOINT}" || ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "ERROR: source HyperDA checkpoint not found." >&2
    echo "Usage:" >&2
    echo "  bash run/stage3_hyperda_posterior_full_inference.sh auto ${TARGET_REGION} ${SEED} ${CUDA_DEVICE}" >&2
    echo "  bash run/stage3_hyperda_posterior_full_inference.sh /path/to/checkpoint.pt ${TARGET_REGION} ${SEED} ${CUDA_DEVICE}" >&2
    exit 2
fi

RUN_SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT}"
if [[ "${SNAPSHOT_STAGE2_CHECKPOINT}" == "1" || "${SNAPSHOT_STAGE2_CHECKPOINT,,}" == "true" ]]; then
    SNAPSHOT_DIR="artifacts/tmp/stage3_full_inference_snapshots"
    mkdir -p "${SNAPSHOT_DIR}"
    SNAPSHOT_PATH="${SNAPSHOT_DIR}/${TARGET_REGION}_s${SEED}_stage2_best_${TIMESTAMP}_snapshot.pt"
    cp -f "${SOURCE_CHECKPOINT}" "${SNAPSHOT_PATH}"
    RUN_SOURCE_CHECKPOINT="${SNAPSHOT_PATH}"
fi

OUTPUT_BASE="${5:-artifacts/runs/stage3_hyperda_posterior/${TARGET_REGION}_s${SEED}_stage2_full_inference_${TIMESTAMP}}"

echo "============================================"
echo "Stage 3 HyperDA Posterior Full Inference"
echo "  source_checkpoint=${SOURCE_CHECKPOINT}"
echo "  run_source_checkpoint=${RUN_SOURCE_CHECKPOINT}"
echo "  snapshot=${SNAPSHOT_STAGE2_CHECKPOINT}"
echo "  target_region=${TARGET_REGION}"
echo "  seed=${SEED}"
echo "  mode=${MODE}"
echo "  K_LIST=${K_LIST}"
echo "  EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  ADAPT_BATCH_SIZE=${ADAPT_BATCH_SIZE}"
echo "  EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "  stage3_posterior_policy=${STAGE3_POSTERIOR_POLICY}"
echo "  stage3_k0_context_shrinkage=${STAGE3_K0_CONTEXT_SHRINKAGE} policy=${STAGE3_K0_CONTEXT_SHRINKAGE_POLICY} rho_cap=${STAGE3_K0_CONTEXT_SHRINKAGE_RHO_CAP} surface_rho_cap=${STAGE3_K0_CONTEXT_SHRINKAGE_SURFACE_RHO_CAP} rootzone_rho_cap=${STAGE3_K0_CONTEXT_SHRINKAGE_ROOTZONE_RHO_CAP} policy_json=${STAGE3_K0_CONTEXT_SHRINKAGE_POLICY_JSON:-<none>}"
echo "  support_gate=${SUPPORT_GATE}"
echo "  schedule_label=${SCHEDULE_LABEL}"
echo "  strict_paper_policy=${STAGE3_STRICT_PAPER_POLICY}"
echo "  require_safe_policy_json_for_kshot=${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT}"
echo "  cuda_device=${CUDA_DEVICE}"
echo "  output_base=${OUTPUT_BASE}"
echo "============================================"

MODE="${MODE}" \
K_LIST="${K_LIST}" \
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES}" \
BATCH_SIZE="${BATCH_SIZE}" \
ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
STAGE3_STRICT_PAPER_POLICY="${STAGE3_STRICT_PAPER_POLICY}" \
REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT}" \
STAGE3_POSTERIOR_POLICY="${STAGE3_POSTERIOR_POLICY}" \
STAGE3_K0_CONTEXT_SHRINKAGE="${STAGE3_K0_CONTEXT_SHRINKAGE}" \
STAGE3_K0_CONTEXT_SHRINKAGE_RHO_CAP="${STAGE3_K0_CONTEXT_SHRINKAGE_RHO_CAP}" \
STAGE3_K0_CONTEXT_SHRINKAGE_POLICY="${STAGE3_K0_CONTEXT_SHRINKAGE_POLICY}" \
STAGE3_K0_CONTEXT_SHRINKAGE_SURFACE_RHO_CAP="${STAGE3_K0_CONTEXT_SHRINKAGE_SURFACE_RHO_CAP}" \
STAGE3_K0_CONTEXT_SHRINKAGE_ROOTZONE_RHO_CAP="${STAGE3_K0_CONTEXT_SHRINKAGE_ROOTZONE_RHO_CAP}" \
STAGE3_K0_CONTEXT_SHRINKAGE_POLICY_JSON="${STAGE3_K0_CONTEXT_SHRINKAGE_POLICY_JSON}" \
SUPPORT_GATE="${SUPPORT_GATE}" \
SCHEDULE_LABEL="${SCHEDULE_LABEL}" \
bash run/stage3_hyperda_posterior_eval.sh \
    "${RUN_SOURCE_CHECKPOINT}" \
    "${TARGET_REGION}" \
    "${SEED}" \
    "${CUDA_DEVICE}" \
    "${OUTPUT_BASE}"
