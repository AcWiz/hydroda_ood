#!/bin/bash
# Per-region evaluation for a trained source-only all-regions checkpoint.
#
# Evaluates on both source_val (2021) and target_query (2023-2025) splits,
# producing per-region metrics_long.csv, metrics_by_region.csv, per_region_summary.json
# under results/source_val/ and results/target_query/.
#
# Usage:
#   bash run/phase4_source_only_all_regions_eval.sh                                    # default checkpoint
#   bash run/phase4_source_only_all_regions_eval.sh path/to/checkpoint.pt              # custom checkpoint
#   bash run/phase4_source_only_all_regions_eval.sh path/to/checkpoint.pt 0 1          # K=0, seed=1

set -euo pipefail

CHECKPOINT="${1:-artifacts/runs/phase4_source_only_all_regions/phase4_source_only_all_regions_source_only_US-ALL_w32_e50_lr0.0003_nonorm_s0_20260525_162315/checkpoints/checkpoint_best_source_val_safe_score.pt}"
K="${2:-0}"
SEED="${3:-0}"
export CUDA_VISIBLE_DEVICES="${4:-1}"

cd "$(dirname "$0")/.."

echo "============================================"
echo "Phase 4 All-Regions Per-Region Evaluation"
echo "  checkpoint: ${CHECKPOINT}"
echo "  K=${K}  seed=${SEED}"
echo "============================================"

# ---- source_val (2021) ----
echo ""
echo ">>> [1/2] Evaluating on source_val (2021)..."
PYTHONPATH=. python scripts/eval/eval_source_only_all_regions.py \
    --checkpoint "${CHECKPOINT}" \
    --split_type source_val \
    --K "${K}" \
    --seed "${SEED}" \
    --device cuda

# ---- target_query (2023-2025) ----
echo ""
echo ">>> [2/2] Evaluating on target_query (2023-2025)..."
PYTHONPATH=. python scripts/eval/eval_source_only_all_regions.py \
    --checkpoint "${CHECKPOINT}" \
    --split_type target_query \
    --K "${K}" \
    --seed "${SEED}" \
    --device cuda

echo ""
echo "Done: all-regions evaluation K=${K} seed=${SEED}"
