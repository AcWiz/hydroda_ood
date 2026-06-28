#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

STAGE3_KSHOT_MODE=diagnostic_finetune_support_gain_v14_nested \
STAGE3_CONTEXT_TTA=none \
K_LIST="0 4 12" \
EVAL_MAX_SAMPLES=0 \
TARGET_CONTEXT_MAX_SAMPLES=0 \
EVAL_OUTPUT_LEVEL=compact \
ADAPT_BATCH_SIZE=1 \
bash run/phase5_hyperda_zero_few_shot_eval.sh "" US-R1 0 1 \
  artifacts/runs/phase5_hyperda_zero_few_shot_eval/US-R1_s0_v14_finetune_support_gain
