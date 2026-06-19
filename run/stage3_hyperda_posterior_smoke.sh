#!/bin/bash
# Diagnostic Stage 3 smoke wrapper for the current staged HyperDA checkpoint.
#
# Defaults are intentionally lightweight:
#   K_LIST=0, no target-label update, EVAL_MAX_SAMPLES=2, batch size 1.
# This is not a paper-facing K4/K12 entrypoint. Use
# run/stage3_hyperda_posterior_eval.sh with SAFE_POLICY_JSON for paper runs.

set -euo pipefail

SOURCE_CHECKPOINT="${1:-}"
TARGET_REGION="${2:-US-R1}"
SEED="${3:-0}"
CUDA_DEVICE="${4:-${CUDA_VISIBLE_DEVICES:-1}}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

cd "$(dirname "$0")/.."

MODE="${MODE:-smoke}"
K_LIST="${K_LIST:-0}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
STAGE3_STRICT_PAPER_POLICY="${STAGE3_STRICT_PAPER_POLICY:-0}"
REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT:-0}"
SNAPSHOT_STAGE2_CHECKPOINT="${SNAPSHOT_STAGE2_CHECKPOINT:-1}"

if [[ -z "${SOURCE_CHECKPOINT}" || "${SOURCE_CHECKPOINT}" == "auto" ]]; then
    SOURCE_CHECKPOINT="$(find "artifacts/runs/phase4_hyperda_staged/${TARGET_REGION}" \
        -path "*s${SEED}*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
        -type f 2>/dev/null | sort | tail -1)"
fi

if [[ -z "${SOURCE_CHECKPOINT}" || ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "ERROR: source HyperDA checkpoint not found." >&2
    echo "Usage:" >&2
    echo "  bash run/stage3_hyperda_posterior_smoke.sh auto ${TARGET_REGION} ${SEED} ${CUDA_DEVICE}" >&2
    echo "  bash run/stage3_hyperda_posterior_smoke.sh /path/to/checkpoint.pt ${TARGET_REGION} ${SEED} ${CUDA_DEVICE}" >&2
    exit 2
fi

RUN_SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT}"
if [[ "${SNAPSHOT_STAGE2_CHECKPOINT}" == "1" || "${SNAPSHOT_STAGE2_CHECKPOINT,,}" == "true" ]]; then
    SNAPSHOT_DIR="artifacts/tmp/stage3_smoke_snapshots"
    mkdir -p "${SNAPSHOT_DIR}"
    SNAPSHOT_PATH="${SNAPSHOT_DIR}/${TARGET_REGION}_s${SEED}_stage2_best_${TIMESTAMP}_snapshot.pt"
    cp -f "${SOURCE_CHECKPOINT}" "${SNAPSHOT_PATH}"
    RUN_SOURCE_CHECKPOINT="${SNAPSHOT_PATH}"
fi

OUTPUT_BASE="${5:-artifacts/runs/stage3_hyperda_posterior/${TARGET_REGION}_s${SEED}_stage2_smoke_${TIMESTAMP}}"

echo "============================================"
echo "Stage 3 HyperDA Posterior Smoke"
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
bash run/stage3_hyperda_posterior_eval.sh \
    "${RUN_SOURCE_CHECKPOINT}" \
    "${TARGET_REGION}" \
    "${SEED}" \
    "${CUDA_DEVICE}" \
    "${OUTPUT_BASE}"
