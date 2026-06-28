#!/usr/bin/env bash
# Run the v7 balanced closed-form K-shot diagnostic over multiple regions/seeds.
#
# This is an exploratory diagnostic/future-extension runner. It does not use
# target_eval for tuning; target_eval remains final offline evaluation only.
#
# Usage:
#   bash run/phase5_v7_balanced_kshot_multiseed.sh
#   REGIONS="US-R2 US-R3" SEEDS="0" CUDA_DEVICE=1 bash run/phase5_v7_balanced_kshot_multiseed.sh

set -euo pipefail

cd "$(dirname "$0")/.."

REGIONS="${REGIONS:-US-R2 US-R3 US-R4 US-R5 US-R6}"
SEEDS="${SEEDS:-0 1 2}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
K_LIST="${K_LIST:-0 4 12}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/runs/phase5_hyperda_zero_few_shot_eval}"

for region in ${REGIONS}; do
  for seed in ${SEEDS}; do
    output_base="${OUTPUT_ROOT}/${region}_s${seed}_v7_balanced_nested"
    echo ">>> v7 balanced K-shot diagnostic: region=${region} seed=${seed} output=${output_base}"
    STAGE3_KSHOT_MODE=diagnostic_linearized_coeff_ridge_v7_balanced_nested \
    STAGE3_CONTEXT_TTA=none \
    K_LIST="${K_LIST}" \
    EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}" \
    TARGET_CONTEXT_MAX_SAMPLES="${TARGET_CONTEXT_MAX_SAMPLES:-0}" \
    EVAL_OUTPUT_LEVEL="${EVAL_OUTPUT_LEVEL:-compact}" \
    ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-1}" \
    bash run/phase5_hyperda_zero_few_shot_eval.sh "" "${region}" "${seed}" "${CUDA_DEVICE}" "${output_base}"
  done
done
