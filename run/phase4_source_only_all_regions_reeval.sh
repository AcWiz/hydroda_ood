#!/bin/bash
# Phase 4 Source-Only All-Regions Re-Evaluation
#
# Re-evaluates the existing all-regions checkpoint with fixed latw metrics.
# Runs eval_source_only_all_regions.py for both source_test and target_eval.
#
# Usage:
#   bash run/phase4_source_only_all_regions_reeval.sh
#   bash run/phase4_source_only_all_regions_reeval.sh /path/to/checkpoint.pt
#
# Output: {run_dir}/results/{split_type}/

set -euo pipefail

CHECKPOINT_PATH="${1:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

cd "$(dirname "$0")/.."

# Auto-detect latest all-regions checkpoint
if [[ -z "$CHECKPOINT_PATH" ]]; then
    LATEST_RUN=$(ls -td artifacts/runs/phase4_source_only_all_regions/*/ 2>/dev/null | head -1)
    if [[ -z "$LATEST_RUN" ]]; then
        echo "ERROR: No all-regions run found. Provide checkpoint path manually."
        exit 1
    fi
    CHECKPOINT_PATH="${LATEST_RUN}checkpoints/checkpoint_best_source_val_safe_score.pt"
    echo "[auto-detect] Using checkpoint: ${CHECKPOINT_PATH}"
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}"
    exit 1
fi

CHECKPOINT_DIR=$(dirname "$CHECKPOINT_PATH")
RUN_DIR=$(dirname "$CHECKPOINT_DIR")
CHECKPOINT_NAME=$(basename "$CHECKPOINT_PATH" .pt)
OUTPUT_BASE="${RUN_DIR}/results/${CHECKPOINT_NAME}"

echo "============================================"
echo "Phase 4 All-Regions Re-Evaluation"
echo "  checkpoint=${CHECKPOINT_PATH}"
echo "  output_base=${OUTPUT_BASE}"
echo "  device=gpu:${CUDA_VISIBLE_DEVICES}"
echo "============================================"

mkdir -p "${OUTPUT_BASE}"

# Run 1: source_test
echo ""
echo ">>> [1/2] Evaluating source_test (all 6 regions)..."
PYTHONPATH=. python scripts/eval/eval_source_only_all_regions.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --split_type source_test \
    --adaptation_setting zero_shot_context --K 0 \
    --seed 0 \
    --device cuda \
    --output_dir "${OUTPUT_BASE}" \
    2>&1 | tee "${OUTPUT_BASE}/log_source_test.txt"

# Run 2: target_eval
echo ""
echo ">>> [2/2] Evaluating target_eval (all 6 regions)..."
PYTHONPATH=. python scripts/eval/eval_source_only_all_regions.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --split_type target_eval \
    --adaptation_setting zero_shot_context --K 0 \
    --seed 0 \
    --device cuda \
    --output_dir "${OUTPUT_BASE}" \
    2>&1 | tee "${OUTPUT_BASE}/log_target_eval.txt"

echo ""
echo "============================================"
echo "Re-evaluation complete."
echo "  source_test: ${OUTPUT_BASE}/source_test"
echo "  target_eval: ${OUTPUT_BASE}/target_eval"
echo "============================================"
