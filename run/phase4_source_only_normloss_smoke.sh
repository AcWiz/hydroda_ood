#!/bin/bash
# Phase 4 Source-Only: Normalized-Loss Smoke Training
#
# Trains source-only backbone with per-variable increment normalization
# for 3 epochs as a smoke test. Compares against nonorm checkpoint and
# forecast-only baseline to diagnose rootzone skill.
#
# Usage:
#   bash run/phase4_source_only_normloss_smoke.sh
#
# Output:
#   artifacts/runs/phase4_source_only/phase4_source_only_source_only_US-R1_w32_e3_lr0.0003_norm_s0_*/
#   artifacts/results/phase4_source_only_inference_normloss_smoke/

set -euo pipefail

cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo "Phase 4 Source-Only: NormLoss Smoke Training"
echo "  timestamp=${TIMESTAMP}"
echo "============================================"

# ── Step 1: Smoke Training ──────────────────────────────────────────────────
echo ""
echo ">>> [1/3] Running smoke training (3 epochs, per_variable_increment_std)..."

PYTHONPATH=. python scripts/train/train_source_only_backbone.py \
    --target_region US-R1 \
    --K 0 \
    --seed 0 \
    --width 32 \
    --max_epochs 3 \
    --batch_size 16 \
    --accum_steps 4 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    --target_normalization_mode per_variable_increment_std \
    --device cuda \
    --amp \
    --wandb_mode disabled \
    --log_every_steps 100 \
    --eval_every_epochs 1 \
    2>&1 | tee "artifacts/source_only_rootzone_diagnosis/normloss_smoke_train_${TIMESTAMP}.log"

echo ""
echo "Smoke training complete."

# ── Find the latest smoke run directory ─────────────────────────────────────
LATEST_RUN=$(ls -td artifacts/runs/phase4_source_only/phase4_source_only_source_only_US-R1_w32_e3_lr0.0003_norm_s0_* 2>/dev/null | head -1)

if [[ -z "$LATEST_RUN" ]]; then
    echo "ERROR: No smoke run directory found."
    exit 1
fi

CHECKPOINT="${LATEST_RUN}/checkpoints/best.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "WARNING: best.pt not found, trying last.pt"
    CHECKPOINT="${LATEST_RUN}/checkpoints/last.pt"
fi

echo "  smoke run: ${LATEST_RUN}"
echo "  checkpoint: ${CHECKPOINT}"

# ── Step 2: Source Val Inference ───────────────────────────────────────────
OUTPUT_BASE="artifacts/results/phase4_source_only_inference_normloss_smoke/${TIMESTAMP}"

echo ""
echo ">>> [2/3] Running source_val inference..."

EVAL_SOURCE_DIR="${OUTPUT_BASE}/source_val"
mkdir -p "${EVAL_SOURCE_DIR}"

PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \
    --checkpoint "$CHECKPOINT" \
    --target_region US-R1 \
    --K 0 \
    --seed 0 \
    --split_type source_val \
    --predictor_type source_only \
    --device cuda \
    --output_dir "${EVAL_SOURCE_DIR}" \
    2>&1 | tee "${OUTPUT_BASE}/log_source_val.txt"

# ── Step 3: Target Query Inference ─────────────────────────────────────────
echo ""
echo ">>> [3/3] Running target_query inference..."

EVAL_TARGET_DIR="${OUTPUT_BASE}/target_query"
mkdir -p "${EVAL_TARGET_DIR}"

PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \
    --checkpoint "$CHECKPOINT" \
    --target_region US-R1 \
    --K 0 \
    --seed 0 \
    --split_type target_query \
    --predictor_type source_only \
    --device cuda \
    --output_dir "${EVAL_TARGET_DIR}" \
    2>&1 | tee "${OUTPUT_BASE}/log_target_query.txt"

# ── Parse and print comparison table ────────────────────────────────────────
echo ""
echo "============================================"
echo "Phase 4 NormLoss Smoke Results"
echo "============================================"

SOURCE_SUMMARY="${EVAL_SOURCE_DIR}/US-R1/summary.json"
TARGET_SUMMARY="${EVAL_TARGET_DIR}/US-R1/summary.json"

export SOURCE_SUMMARY TARGET_SUMMARY LATEST_RUN OUTPUT_BASE

python3 - <<'PYTHON_SCRIPT'
import json
from pathlib import Path
import os, sys

output_base = Path(os.environ.get("OUTPUT_BASE", "artifacts/results/phase4_source_only_inference_normloss_smoke/latest"))
run_dir = Path(os.environ.get("LATEST_RUN", "N/A"))

source_path = Path(os.environ.get("SOURCE_SUMMARY", ""))
target_path = Path(os.environ.get("TARGET_SUMMARY", ""))

def load(path):
    if path and Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None

src = load(source_path)
tgt = load(target_path)

def fmt(v, d=4):
    return f"{v:.{d}f}" if isinstance(v, (int, float)) and v == v else "N/A"

print(f"\nNormLoss Smoke Checkpoint: {run_dir}")

if src and tgt:
    print()
    print("| Metric       | Surface (source_val) | Surface (target_query) | Rootzone (source_val) | Rootzone (target_query) |")
    print("|:-------------|--------------------:|----------------------:|---------------------:|----------------------:|")
    for metric in ["skill_mean", "skill_std", "rmse_mean", "corr_mean"]:
        s_surf = src["surface"].get(metric)
        t_surf = tgt["surface"].get(metric)
        s_root = src["rootzone"].get(metric)
        t_root = tgt["rootzone"].get(metric)
        print(f"| {metric:<13} | {fmt(s_surf):>20} | {fmt(t_surf):>20} | {fmt(s_root):>18} | {fmt(t_root):>20} |")

    print()
    print("Compare with nonorm baseline:")
    print("  nonorm source_val    surface skill=+0.44 rootzone skill=-1.61")
    print("  nonorm target_query  surface skill=-1.95 rootzone skill=-12.21")
    print("  normloss smoke       surface skill=?     rootzone skill=?")

elif src:
    print("WARNING: target_query summary not found.")
elif tgt:
    print("WARNING: source_val summary not found.")
else:
    print("ERROR: Neither summary found.")
PYTHON_SCRIPT

echo ""
echo "Done."
echo "  Training log:  artifacts/source_only_rootzone_diagnosis/normloss_smoke_train_${TIMESTAMP}.log"
echo "  Results:       ${OUTPUT_BASE}/"
