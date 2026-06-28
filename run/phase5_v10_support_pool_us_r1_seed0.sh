#!/usr/bin/env bash
# Run the v10 support-pool nested closed-form K-shot diagnostic for US-R1 seed0.
#
# This is exploratory only. Candidate selection uses target_support exact mixed
# predictor loss; target_eval remains final offline evaluation with no selection.
#
# Usage:
#   bash run/phase5_v10_support_pool_us_r1_seed0.sh
#   CUDA_DEVICE=0 K_LIST="0 4 12" bash run/phase5_v10_support_pool_us_r1_seed0.sh

set -euo pipefail

cd "$(dirname "$0")/.."

TARGET_REGION="${TARGET_REGION:-US-R1}"
SEED="${SEED:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
K_LIST="${K_LIST:-0 4 12}"
OUTPUT_BASE="${OUTPUT_BASE:-artifacts/runs/phase5_hyperda_zero_few_shot_eval/US-R1_s0_v10_support_pool_nested}"

STAGE3_KSHOT_MODE=diagnostic_linearized_coeff_ridge_v10_support_pool_nested \
STAGE3_CONTEXT_TTA=none \
K_LIST="${K_LIST}" \
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}" \
TARGET_CONTEXT_MAX_SAMPLES="${TARGET_CONTEXT_MAX_SAMPLES:-0}" \
EVAL_OUTPUT_LEVEL="${EVAL_OUTPUT_LEVEL:-compact}" \
ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-1}" \
bash run/phase5_hyperda_zero_few_shot_eval.sh "" "${TARGET_REGION}" "${SEED}" "${CUDA_DEVICE}" "${OUTPUT_BASE}"
