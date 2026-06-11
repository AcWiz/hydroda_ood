#!/bin/bash
# Phase 4C HyperDA Inference: source_test vs target_eval
#
# Usage:
#   bash run/phase4_hyperda_inference.sh
#   bash run/phase4_hyperda_inference.sh /path/to/checkpoint.pt US-R1 0 1
#   bash run/phase4_hyperda_inference.sh /path/to/checkpoint.pt US-R1 0 1 /path/to/output_dir
#
# Evaluates the HyperDA basis-adapter checkpoint on:
#   - source_test:  2023-2025 held-out source regions
#   - target_eval:  2023-2025 target-region pixels

set -euo pipefail

CHECKPOINT_PATH="${1:-}"
TARGET_REGION="${2:-US-R1}"
SEED="${3:-0}"
export CUDA_VISIBLE_DEVICES="${4:-${CUDA_VISIBLE_DEVICES:-1}}"
OUTPUT_DIR_OVERRIDE="${5:-}"
TARGET_CONTEXT_PROMPT="${TARGET_CONTEXT_PROMPT:-1}"
TARGET_TRAIN_RESIDUAL_GAIN_CALIBRATION="${TARGET_TRAIN_RESIDUAL_GAIN_CALIBRATION:-0}"
ALLOW_LEGACY_TARGET_LABEL_CALIBRATION="${ALLOW_LEGACY_TARGET_LABEL_CALIBRATION:-0}"

cd "$(dirname "$0")/.."

# Auto-detect latest HyperDA checkpoint if not provided.
if [[ -z "$CHECKPOINT_PATH" ]]; then
    LATEST_RUN=$(ls -td artifacts/runs/phase4_prompt_conditioned/phase4_prompt_conditioned_hyperda_basis_adapter_* 2>/dev/null | head -1)
    if [[ -z "$LATEST_RUN" ]]; then
        echo "ERROR: No HyperDA run found. Provide checkpoint path manually."
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
RUN_DIR=$(dirname "$CHECKPOINT_DIR")
CHECKPOINT_NAME=$(basename "$CHECKPOINT_PATH" .pt)
if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
    OUTPUT_BASE="$OUTPUT_DIR_OVERRIDE"
else
    OUTPUT_BASE="${RUN_DIR}/results/${CHECKPOINT_NAME}"
fi
PROTOCOL_ID="hyperda_v4_4_zero_few_shot_generalization_2015_2025_context2015_2021_sourceval2022_eval2023_2025"

echo "============================================"
echo "Phase 4C HyperDA Inference"
echo "  checkpoint=${CHECKPOINT_PATH}"
echo "  target_region=${TARGET_REGION}  adaptation_setting=zero_shot_context  seed=${SEED}"
echo "  protocol=${PROTOCOL_ID}"
echo "  output_base=${OUTPUT_BASE}"
echo "  target_context_prompt=${TARGET_CONTEXT_PROMPT}"
echo "  target_train_residual_gain_calibration=${TARGET_TRAIN_RESIDUAL_GAIN_CALIBRATION}"
echo "  device=gpu:${CUDA_VISIBLE_DEVICES}"
echo "============================================"

mkdir -p "${OUTPUT_BASE}"

echo ""
echo ">>> [1/2] Evaluating source_test on held-out source regions..."
EVAL_SOURCE_DIR="${OUTPUT_BASE}/source_test"
PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --target_region "${TARGET_REGION}" \
    --adaptation_setting zero_shot_context \
    --K 0 \
    --seed "${SEED}" \
    --split_type source_test \
    --predictor_type prompt_conditioned \
    --device cuda \
    --output_dir "${EVAL_SOURCE_DIR}" \
    2>&1 | tee "${OUTPUT_BASE}/log_source_test.txt"

echo ""
echo ">>> [2/2] Evaluating target_eval on target region..."
EVAL_TARGET_DIR="${OUTPUT_BASE}/target_eval"
PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --target_region "${TARGET_REGION}" \
    --adaptation_setting zero_shot_context \
    --K 0 \
    --seed "${SEED}" \
    --split_type target_eval \
    --predictor_type prompt_conditioned \
    --device cuda \
    --output_dir "${EVAL_TARGET_DIR}" \
    $(if [[ "${TARGET_CONTEXT_PROMPT}" == "1" ]]; then echo "--target_context_prompt"; fi) \
    $(if [[ "${TARGET_TRAIN_RESIDUAL_GAIN_CALIBRATION}" == "1" ]]; then echo "--target_train_residual_gain_calibration"; fi) \
    $(if [[ "${ALLOW_LEGACY_TARGET_LABEL_CALIBRATION}" == "1" ]]; then echo "--allow_legacy_target_label_calibration"; fi) \
    2>&1 | tee "${OUTPUT_BASE}/log_target_eval.txt"

echo ""
echo "============================================"
echo "Phase 4C HyperDA Inference Results"
echo "============================================"

python3 - "${OUTPUT_BASE}" "${TARGET_REGION}" "${PROTOCOL_ID}" <<'PYTHON_SCRIPT'
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
target_region = sys.argv[2]
protocol_id = sys.argv[3]

source_summary = run_dir / "source_test" / target_region / "summary.json"
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
print("Method:      hyperda_basis_adapter_shared")
print(f"Checkpoint: {src['checkpoint'] if src else tgt['checkpoint'] if tgt else 'N/A'}")
print(f"Protocol:    {protocol_id}")
print(f"Target:      {target_region}  adaptation_setting=zero_shot_context")
print("```")
print()

def fmt(v, decimals=10):
    if v is None:
        return "N/A"
    if isinstance(v, (int, float)):
        return f"{v:.{decimals}f}"
    return str(v)

if src and tgt:
    print("| Metric       | Surface (source_test) | Surface (target_eval) | Rootzone (source_test) | Rootzone (target_eval) |")
    print("|:-------------|--------------------:|----------------------:|---------------------:|----------------------:|")
    for metric, label in [("skill_primary", "skill_primary"), ("skill_latw_primary", "skill_latw_pri"), ("skill_median", "skill_median"), ("rmse_latw_mean", "WRMSE"), ("corr_latw_mean", "Corr_latw")]:
        print(
            f"| {label:<13} | {fmt(src['surface'].get(metric)):>20} | "
            f"{fmt(tgt['surface'].get(metric)):>20} | "
            f"{fmt(src['rootzone'].get(metric)):>19} | "
            f"{fmt(tgt['rootzone'].get(metric)):>21} |"
        )

    gap_surf = src["surface"]["skill_primary"] - tgt["surface"]["skill_primary"]
    gap_root = src["rootzone"]["skill_primary"] - tgt["rootzone"]["skill_primary"]
    print()
    print("## Evaluation Gap (source_test - target_eval)")
    print("|              |    Surface    |   Rootzone   |")
    print("|:-------------|-------------:|-------------:|")
    print(f"| skill_gap    | {fmt(gap_surf, 2):>10} | {fmt(gap_root, 2):>10} |")

    def extract_time(log_path):
        text = log_path.read_text() if log_path.exists() else ""
        m = re.search(r"Evaluation (?:done|completed) in ([\\d.]+)s", text)
        return float(m.group(1)) if m else None

    print()
    print("## Evaluation Details")
    print("| Split       | Samples | Metric Rows | Eval Time |")
    print("|:------------|--------:|------------:|----------:|")
    for split_name, summary, log_name in [
        ("source_test", src, "log_source_test.txt"),
        ("target_eval", tgt, "log_target_eval.txt"),
    ]:
        elapsed = extract_time(run_dir / log_name)
        print(
            f"| {split_name:<11} | {summary.get('n_samples_evaluated', '?'):>6} | "
            f"{summary.get('n_metric_rows', '?'):>10} | "
            f"{fmt(elapsed, 1) + 's' if elapsed else 'N/A':>8} |"
        )
    print()
    print(f"Full results: `{run_dir}/`")
else:
    if not src:
        print(f"WARNING: Missing summary: {source_summary}")
    if not tgt:
        print(f"WARNING: Missing summary: {target_summary}")

print()
PYTHON_SCRIPT

echo "Done."
