#!/bin/bash
# Phase 4B Prompt-Conditioned Inference: source_val vs target_eval
#
# Usage:
#   bash run/phase4_prompt_conditioned_inference.sh
#   bash run/phase4_prompt_conditioned_inference.sh /path/to/checkpoint.pt US-R1 0 1
#
# Evaluates the prompt-conditioned shared backbone on:
#   - source_val:   2022 held-out source regions R2-R6
#   - target_eval:  2023-2025 target-region pixels

set -euo pipefail

CHECKPOINT_PATH="${1:-}"
TARGET_REGION="${2:-US-R1}"
SEED="${3:-0}"
export CUDA_VISIBLE_DEVICES="${4:-${CUDA_VISIBLE_DEVICES:-1}}"

cd "$(dirname "$0")/.."

# Auto-detect latest prompt-conditioned checkpoint if not provided.
if [[ -z "$CHECKPOINT_PATH" ]]; then
    LATEST_RUN=$(ls -td artifacts/runs/phase4_prompt_conditioned/phase4_prompt_conditioned_prompt_conditioned_* 2>/dev/null | head -1)
    if [[ -z "$LATEST_RUN" ]]; then
        echo "ERROR: No phase4_prompt_conditioned run found. Provide checkpoint path manually."
        exit 1
    fi
    if [[ -f "${LATEST_RUN}/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" ]]; then
        CHECKPOINT_PATH="${LATEST_RUN}/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt"
    else
        CHECKPOINT_PATH="${LATEST_RUN}/checkpoints/best.pt"
    fi
    echo "[auto-detect] Using checkpoint: ${CHECKPOINT_PATH}"
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}"
    exit 1
fi

CHECKPOINT_DIR=$(dirname "$CHECKPOINT_PATH")
RUN_ID=$(basename "$(dirname "$CHECKPOINT_DIR")")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="artifacts/results/phase4_prompt_conditioned_inference/${RUN_ID}_${TIMESTAMP}"
PROTOCOL_ID="hyperda_v4_3_historical_target_adapt_2015_2025_train2015_2021_val2022_test2023_2025"

echo "============================================"
echo "Phase 4B Prompt-Conditioned Inference"
echo "  checkpoint=${CHECKPOINT_PATH}"
echo "  target_region=${TARGET_REGION}  adaptation_setting=target_full_train  seed=${SEED}"
echo "  protocol=${PROTOCOL_ID}"
echo "  output_base=${OUTPUT_BASE}"
echo "  device=gpu:${CUDA_VISIBLE_DEVICES}"
echo "============================================"

mkdir -p "${OUTPUT_BASE}"

echo ""
echo ">>> [1/2] Evaluating source_val on held-out source regions..."
EVAL_SOURCE_DIR="${OUTPUT_BASE}/source_val"
PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --target_region "${TARGET_REGION}" \
    --adaptation_setting target_full_train \
    --seed "${SEED}" \
    --split_type source_val \
    --predictor_type prompt_conditioned \
    --device cuda \
    --output_dir "${EVAL_SOURCE_DIR}" \
    2>&1 | tee "${OUTPUT_BASE}/log_source_val.txt"

echo ""
echo ">>> [2/2] Evaluating target_eval on target region..."
EVAL_TARGET_DIR="${OUTPUT_BASE}/target_eval"
PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --target_region "${TARGET_REGION}" \
    --adaptation_setting target_full_train \
    --seed "${SEED}" \
    --split_type target_eval \
    --predictor_type prompt_conditioned \
    --device cuda \
    --output_dir "${EVAL_TARGET_DIR}" \
    2>&1 | tee "${OUTPUT_BASE}/log_target_eval.txt"

echo ""
echo "============================================"
echo "Phase 4B Prompt-Conditioned Inference Results"
echo "============================================"

python3 - "${OUTPUT_BASE}" "${TARGET_REGION}" "${PROTOCOL_ID}" <<'PYTHON_SCRIPT'
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
target_region = sys.argv[2]
protocol_id = sys.argv[3]

source_summary = run_dir / "source_val" / target_region / "summary.json"
target_summary = run_dir / "target_eval" / target_region / "summary.json"

def load_summary(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

src = load_summary(source_summary)
tgt = load_summary(target_summary)

print()
print("```")
print(f"Run ID:      {run_dir.name}")
print("Method:      prompt_conditioned_shared_backbone")
print(f"Checkpoint: {src['checkpoint'] if src else tgt['checkpoint'] if tgt else 'N/A'}")
print(f"Protocol:    {protocol_id}")
print(f"Target:      {target_region}  adaptation_setting=target_full_train")
print("```")
print()

def fmt(v, decimals=4):
    if v is None:
        return "N/A"
    if isinstance(v, (int, float)):
        return f"{v:.{decimals}f}"
    return str(v)

if src and tgt:
    print("| Metric       | Surface (source_val) | Surface (target_eval) | Rootzone (source_val) | Rootzone (target_eval) |")
    print("|:-------------|--------------------:|----------------------:|---------------------:|----------------------:|")
    for metric in ["skill_mean", "skill_std", "rmse_mean", "corr_mean"]:
        print(
            f"| {metric:<13} | {fmt(src['surface'].get(metric)):>20} | "
            f"{fmt(tgt['surface'].get(metric)):>20} | "
            f"{fmt(src['rootzone'].get(metric)):>19} | "
            f"{fmt(tgt['rootzone'].get(metric)):>21} |"
        )

    gap_surf = src["surface"]["skill_mean"] - tgt["surface"]["skill_mean"]
    gap_root = src["rootzone"]["skill_mean"] - tgt["rootzone"]["skill_mean"]
    print()
    print("## Evaluation Gap (source_val - target_eval)")
    print("|              |    Surface    |   Rootzone   |")
    print("|:-------------|-------------:|-------------:|")
    print(f"| skill_gap    | {fmt(gap_surf, 2):>10} | {fmt(gap_root, 2):>10} |")

    def extract_time(log_path):
        text = log_path.read_text() if log_path.exists() else ""
        m = re.search(r"Evaluation (?:done|completed) in ([\d.]+)s", text)
        return float(m.group(1)) if m else None

    print()
    print("## Evaluation Details")
    print("| Split       | Samples | Metric Rows | Eval Time |")
    print("|:------------|--------:|------------:|----------:|")
    for split_name, summary, log_name in [
        ("source_val", src, "log_source_val.txt"),
        ("target_eval", tgt, "log_target_eval.txt"),
    ]:
        elapsed = extract_time(run_dir / log_name)
        print(
            f"| {split_name:<11} | {summary.get('n_samples_evaluated', '?'):>6} | "
            f"{summary.get('n_metric_rows', '?'):>10} | "
            f"{fmt(elapsed, 1) + 's' if elapsed else 'N/A':>8} |"
        )
    print()
    print(f"Full results: `artifacts/results/phase4_prompt_conditioned_inference/{run_dir.name}/`")
else:
    if not src:
        print(f"WARNING: Missing summary: {source_summary}")
    if not tgt:
        print(f"WARNING: Missing summary: {target_summary}")

print()
PYTHON_SCRIPT

echo "Done."
