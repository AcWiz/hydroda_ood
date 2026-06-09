#!/bin/bash
# Phase 5 HyperDA target-adapt inference on target_eval.
#
# Usage:
#   bash run/phase5_hyperda_target_adapt_inference.sh
#   bash run/phase5_hyperda_target_adapt_inference.sh /path/to/checkpoint_best_target_val_surface_wrmse.pt US-R1 0 1
#   bash run/phase5_hyperda_target_adapt_inference.sh /path/to/checkpoint_best_target_val_loss.pt US-R1 0 1 /path/to/output_dir
#
# Evaluates the target-adapted HyperDA checkpoint selected on target_val=2022
# against target_eval=2023-2025. target_eval labels are used only for final
# offline metric computation.

set -euo pipefail

CHECKPOINT_PATH="${1:-}"
TARGET_REGION="${2:-US-R1}"
SEED="${3:-0}"
export CUDA_VISIBLE_DEVICES="${4:-${CUDA_VISIBLE_DEVICES:-1}}"
OUTPUT_DIR_OVERRIDE="${5:-}"

cd "$(dirname "$0")/.."

if [[ -z "$CHECKPOINT_PATH" ]]; then
    SURFACE_CHECKPOINT=$(find artifacts/runs/phase5_hyperda_target_adapt \
        -path "*hyperda_target_adapt_${TARGET_REGION}_*_s${SEED}_*/checkpoints/checkpoint_best_target_val_surface_wrmse.pt" \
        -type f 2>/dev/null | sort | tail -1)
    LOSS_CHECKPOINT=$(find artifacts/runs/phase5_hyperda_target_adapt \
        -path "*hyperda_target_adapt_${TARGET_REGION}_*_s${SEED}_*/checkpoints/checkpoint_best_target_val_loss.pt" \
        -type f 2>/dev/null | sort | tail -1)
    CHECKPOINT_PATH="${SURFACE_CHECKPOINT:-${LOSS_CHECKPOINT}}"
    if [[ -z "$CHECKPOINT_PATH" ]]; then
        echo "ERROR: No Phase 5 target-adapt checkpoint found. Provide checkpoint path manually." >&2
        echo "  bash run/phase5_hyperda_target_adapt_inference.sh <checkpoint_best_target_val_surface_wrmse.pt> ${TARGET_REGION} ${SEED} ${CUDA_VISIBLE_DEVICES}" >&2
        exit 1
    fi
    echo "[auto-detect] Using checkpoint: ${CHECKPOINT_PATH}"
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}" >&2
    exit 1
fi

CHECKPOINT_DIR=$(dirname "$CHECKPOINT_PATH")
RUN_DIR=$(dirname "$CHECKPOINT_DIR")
CHECKPOINT_NAME=$(basename "$CHECKPOINT_PATH" .pt)
if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
    OUTPUT_BASE="$OUTPUT_DIR_OVERRIDE"
else
    OUTPUT_BASE="${RUN_DIR}/results/${CHECKPOINT_NAME}"
fi
PROTOCOL_ID="hyperda_v4_3_historical_target_adapt_2015_2025_train2015_2021_val2022_test2023_2025"

echo "============================================"
echo "Phase 5 HyperDA Target-Adapt Inference"
echo "  checkpoint=${CHECKPOINT_PATH}"
echo "  target_region=${TARGET_REGION}  adaptation_setting=target_full_train  seed=${SEED}"
echo "  target_train=2015-2021"
echo "  target_val=2022 checkpoint selection"
echo "  target_eval=2023-2025 final evaluation only"
echo "  protocol=${PROTOCOL_ID}"
echo "  output_base=${OUTPUT_BASE}"
echo "  device=gpu:${CUDA_VISIBLE_DEVICES}"
echo "============================================"

mkdir -p "${OUTPUT_BASE}"

echo ""
echo ">>> Evaluating target_eval on target region..."
EVAL_TARGET_DIR="${OUTPUT_BASE}/target_eval"
PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --target_region "${TARGET_REGION}" \
    --adaptation_setting target_full_train \
    --seed "${SEED}" \
    --split_type target_eval \
    --predictor_type hyperda_target_adapt \
    --device cuda \
    --output_dir "${EVAL_TARGET_DIR}" \
    2>&1 | tee "${OUTPUT_BASE}/log_target_eval.txt"

echo ""
echo "============================================"
echo "Phase 5 HyperDA Target-Adapt Results"
echo "============================================"

python3 - "${OUTPUT_BASE}" "${TARGET_REGION}" "${PROTOCOL_ID}" <<'PYTHON_SCRIPT'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
target_region = sys.argv[2]
protocol_id = sys.argv[3]
summary_path = run_dir / "target_eval" / target_region / "summary.json"

if not summary_path.exists():
    print(f"WARNING: Missing summary: {summary_path}")
    sys.exit(0)

with open(summary_path) as f:
    summary = json.load(f)

def fmt(v, decimals=10):
    if v is None:
        return "N/A"
    if isinstance(v, (int, float)):
        return f"{v:.{decimals}f}"
    return str(v)

print()
print("```")
print(f"Run ID:      {run_dir.name}")
print("Method:      hyperda_target_adapt")
print(f"Checkpoint: {summary.get('checkpoint', 'N/A')}")
print(f"Protocol:    {protocol_id}")
print(f"Target:      {target_region}  adaptation_setting=target_full_train")
print("Split:       target_eval=2023-2025")
print("```")
print()
print("| Metric       | Surface (target_eval) | Rootzone (target_eval) |")
print("|:-------------|---------------------:|----------------------:|")
for metric, label in [
    ("skill_primary", "skill_primary"),
    ("skill_latw_primary", "skill_latw_pri"),
    ("skill_median", "skill_median"),
    ("rmse_latw_mean", "WRMSE"),
    ("corr_latw_mean", "Corr_latw"),
]:
    print(
        f"| {label:<13} | {fmt(summary['surface'].get(metric)):>21} | "
        f"{fmt(summary['rootzone'].get(metric)):>21} |"
    )

print()
print("## Evaluation Details")
print("| Split       | Samples | Metric Rows | Eval Time |")
print("|:------------|--------:|------------:|----------:|")
elapsed = summary.get("eval_time_s")
elapsed_text = f"{elapsed:.1f}s" if isinstance(elapsed, (int, float)) else "N/A"
print(
    f"| target_eval | {summary.get('n_samples_evaluated', '?'):>6} | "
    f"{summary.get('n_metric_rows', '?'):>10} | {elapsed_text:>8} |"
)
print()
print(f"Full results: `{run_dir}/`")
print()
PYTHON_SCRIPT

echo "Done."
