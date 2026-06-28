#!/usr/bin/env bash
# Rebuild artifact-driven Stage 3 K-shot overview for an existing run.
#
# Usage:
#   bash run/rebuild_stage3_kshot_overview.sh
#   bash run/rebuild_stage3_kshot_overview.sh <output_base> <target_region> <seed> "<k_list>"

set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_BASE="${1:-artifacts/runs/phase5_hyperda_zero_few_shot_eval/US-R1_s0_20260626T034455Z}"
TARGET_REGION="${2:-US-R1}"
SEED="${3:-0}"
K_LIST="${4:-0 4 12}"

export PYTHONPATH="${PYTHONPATH:-.}"

python scripts/eval/stage3_kshot_overview.py \
  --output_base "${OUTPUT_BASE}" \
  --target_region "${TARGET_REGION}" \
  --seed "${SEED}" \
  --k_list "${K_LIST}"
