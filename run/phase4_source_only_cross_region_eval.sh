#!/bin/bash
# Phase 4 Source-Only Cross-Region Evaluation
#
# For each pair (model trained on Y, evaluate on X where X != Y):
#   Runs evaluate_checkpoint.py with --target_region X --split_type target_eval
#   Output: <Y_run>/results/<ckpt>/target_eval/US-R{X}/
#
# This enables fair Src_RS computation: for target region X,
# Src_RS = weighted aggregate of models Y1..Y5 each evaluated on X's test data.
#
# Usage:
#   bash run/phase4_source_only_cross_region_eval.sh
#
# Total evaluations: 6 models x 5 other regions = 30 passes

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

cd "$(dirname "$0")/.."

REGIONS=("US-R1" "US-R2" "US-R3" "US-R4" "US-R5" "US-R6")
BASE_DIR="artifacts/runs/phase4_source_only_region_specific"

echo "================================================================"
echo "Phase 4 Cross-Region Evaluation"
echo "Base directory: ${BASE_DIR}"
echo "Source model regions: ${REGIONS[*]}"
echo "Started at $(date)"
echo "================================================================"

TOTAL=0
SKIPPED=0

for src_region in "${REGIONS[@]}"; do
    # Find the run directory for this source region model
    RUN_DIR=$(ls -td ${BASE_DIR}/phase4_source_only_region_specific_source_only_${src_region}_* 2>/dev/null | head -1)
    if [[ -z "$RUN_DIR" ]]; then
        echo "WARNING: No run found for ${src_region}, skipping all cross-region evals from this model."
        continue
    fi

    CHECKPOINT="${RUN_DIR}/checkpoints/checkpoint_best_source_val_safe_score.pt"
    if [[ ! -f "$CHECKPOINT" ]]; then
        CHECKPOINT="${RUN_DIR}/checkpoints/best.pt"
    fi
    if [[ ! -f "$CHECKPOINT" ]]; then
        echo "WARNING: No checkpoint found for ${src_region}, skipping."
        continue
    fi

    CHECKPOINT_NAME=$(basename "$CHECKPOINT" .pt)
    OUTPUT_BASE="${RUN_DIR}/results/${CHECKPOINT_NAME}"

    echo ""
    echo "=== Source model: ${src_region} (checkpoint: ${CHECKPOINT_NAME}) ==="

    for tgt_region in "${REGIONS[@]}"; do
        # Skip self-evaluation (already done in re-eval)
        if [[ "$tgt_region" == "$src_region" ]]; then
            continue
        fi

        EVAL_DIR="${OUTPUT_BASE}/target_eval/${tgt_region}"

        # Skip if already evaluated
        if [[ -f "${EVAL_DIR}/summary.json" ]] && [[ -f "${EVAL_DIR}/metrics_long.csv" ]]; then
            echo "  [${src_region} -> ${tgt_region}] Already evaluated, skipping."
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        echo "  [${src_region} -> ${tgt_region}] Evaluating..."
        TOTAL=$((TOTAL + 1))

        PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \
            --checkpoint "$CHECKPOINT" \
            --target_region "${tgt_region}" \
            --adaptation_setting target_full_train \
            --seed 0 \
            --split_type target_eval \
            --predictor_type source_only \
            --device cuda \
            --output_dir "${OUTPUT_BASE}/target_eval/" \
            2>&1 | tee "${OUTPUT_BASE}/log_cross_${tgt_region}.txt"

        echo "  [${src_region} -> ${tgt_region}] Done."
    done
done

echo ""
echo "================================================================"
echo "Cross-region evaluation complete at $(date)"
echo "  Total new evaluations: ${TOTAL}"
echo "  Skipped (already done): ${SKIPPED}"
echo "================================================================"
