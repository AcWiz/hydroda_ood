#!/bin/bash
# Phase 4 Source-Only Region-Specific Re-Evaluation
#
# Re-evaluates each region-specific checkpoint with fixed latw metrics.
# Uses evaluate_checkpoint.py for each region × split_type combination.
#
# Usage:
#   bash run/phase4_source_only_region_specific_reeval.sh
#
# Output: {run_dir}/results/{split_type}/{region}/

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

cd "$(dirname "$0")/.."

REGIONS=("US-R1" "US-R2" "US-R3" "US-R4" "US-R5" "US-R6")
SPLIT_TYPES=("source_test" "target_eval")
BASE_DIR="artifacts/runs/phase4_source_only_region_specific"

echo "================================================================"
echo "Phase 4 Region-Specific Re-Evaluation"
echo "Base directory: ${BASE_DIR}"
echo "Regions: ${REGIONS[*]}"
echo "Split types: ${SPLIT_TYPES[*]}"
echo "Started at $(date)"
echo "================================================================"

for region in "${REGIONS[@]}"; do
    # Find the run directory for this region
    RUN_DIR=$(ls -td ${BASE_DIR}/phase4_source_only_region_specific_source_only_${region}_* 2>/dev/null | head -1)
    if [[ -z "$RUN_DIR" ]]; then
        echo "WARNING: No run found for ${region}, skipping."
        continue
    fi

    CHECKPOINT="${RUN_DIR}/checkpoints/checkpoint_best_source_val_safe_score.pt"
    if [[ ! -f "$CHECKPOINT" ]]; then
        CHECKPOINT="${RUN_DIR}/checkpoints/best.pt"
    fi
    if [[ ! -f "$CHECKPOINT" ]]; then
        echo "WARNING: No checkpoint found for ${region}, skipping."
        continue
    fi

    CHECKPOINT_NAME=$(basename "$CHECKPOINT" .pt)
    OUTPUT_BASE="${RUN_DIR}/results/${CHECKPOINT_NAME}"

    echo ""
    echo "--- ${region} (checkpoint: ${CHECKPOINT_NAME}) ---"

    for split_type in "${SPLIT_TYPES[@]}"; do
        EVAL_DIR="${OUTPUT_BASE}/${split_type}"
        mkdir -p "${EVAL_DIR}"

        echo "  [${region}] ${split_type} ..."
        PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \
            --checkpoint "$CHECKPOINT" \
            --target_region "${region}" \
            --adaptation_setting target_full_train \
            --seed 0 \
            --split_type "${split_type}" \
            --predictor_type source_only \
            --device cuda \
            --output_dir "${EVAL_DIR}" \
            2>&1 | tee "${OUTPUT_BASE}/log_${split_type}.txt"
        echo "  [${region}] ${split_type} done."
    done
done

echo ""
echo "================================================================"
echo "All region-specific re-evaluations complete at $(date)"
echo "================================================================"
