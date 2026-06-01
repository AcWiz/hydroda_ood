#!/bin/bash
# Phase 4 Source-Only Inference: In-domain (source_test) vs OOD (target_eval)
#
# Usage:
#   bash run/phase4_source_only_inference.sh
#   bash run/phase4_source_only_inference.sh /path/to/checkpoint.pt
#
# Evaluates the source-only backbone on:
#   - source_test:  2023-2025 held-out source regions R2-R6 (in-domain baseline, NOT target)
#   - target_eval:  2023-2025 US-R1 target pixels (OOD signal)

set -euo pipefail

CHECKPOINT_PATH="${1:-}"
OUTPUT_DIR_OVERRIDE="${2:-}"

cd "$(dirname "$0")/.."

# Auto-detect latest checkpoint if not provided
if [[ -z "$CHECKPOINT_PATH" ]]; then
    LATEST_RUN=$(ls -td artifacts/runs/phase4_source_only/phase4_source_only_source_only_US-R1_w32_e30_lr0.0003_nonorm_nozero_s0_* 2>/dev/null | head -1)
    if [[ -z "$LATEST_RUN" ]]; then
        echo "ERROR: No phase4_source_only run found. Provide checkpoint path manually."
        exit 1
    fi
    CHECKPOINT_PATH="${LATEST_RUN}/checkpoints/best.pt"
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

# Fixed evaluation parameters matching training
TARGET_REGION="US-R1"
K="0"
SEED="0"
DEVICE="${CUDA_VISIBLE_DEVICES:-1}"

echo "============================================"
echo "Phase 4 Source-Only Inference"
echo "  checkpoint=${CHECKPOINT_PATH}"
echo "  target_region=${TARGET_REGION}  K=${K}  seed=${SEED}"
echo "  output_base=${OUTPUT_BASE}"
echo "  device=gpu:${DEVICE}"
echo "============================================"

# Create output directory
mkdir -p "${OUTPUT_BASE}"

# Run 1: source_test (in-domain)
echo ""
echo ">>> [1/2] Evaluating source_test on source regions R2-R6 (in-domain)..."
EVAL_SOURCE_DIR="${OUTPUT_BASE}/source_test"
PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --target_region "${TARGET_REGION}" \
    --K "${K}" \
    --seed "${SEED}" \
    --split_type source_test \
    --predictor_type source_only \
    --device cuda \
    --output_dir "${EVAL_SOURCE_DIR}" \
    2>&1 | tee "${OUTPUT_BASE}/log_source_test.txt"

SOURCE_SUMMARY="${EVAL_SOURCE_DIR}/${TARGET_REGION}/summary.json"

# Run 2: target_eval (OOD)
echo ""
echo ">>> [2/2] Evaluating target_eval on target region US-R1 (OOD)..."
EVAL_TARGET_DIR="${OUTPUT_BASE}/target_eval"
PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --target_region "${TARGET_REGION}" \
    --K "${K}" \
    --seed "${SEED}" \
    --split_type target_eval \
    --predictor_type source_only \
    --device cuda \
    --output_dir "${EVAL_TARGET_DIR}" \
    2>&1 | tee "${OUTPUT_BASE}/log_target_eval.txt"

TARGET_SUMMARY="${EVAL_TARGET_DIR}/${TARGET_REGION}/summary.json"

# Parse and print comparison table
echo ""
echo "============================================"
echo "Phase 4 Source-Only Inference Results"
echo "============================================"

for summary in "$SOURCE_SUMMARY" "$TARGET_SUMMARY"; do
    if [[ ! -f "$summary" ]]; then
        echo "WARNING: Missing summary: $summary"
    fi
done

python3 - "${OUTPUT_BASE}" "${TARGET_REGION}" <<'PYTHON_SCRIPT'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
target_region = sys.argv[2]

source_summary = run_dir / "source_test" / target_region / "summary.json"
target_summary = run_dir / "target_eval" / target_region / "summary.json"

def load_summary(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

src = load_summary(source_summary)
tgt = load_summary(target_summary)

# ── Header ────────────────────────────────────────────────────────────────────
print()
print("```")
print(f"Run ID:      {run_dir.name}")
print(f"Method:      source_only_backbone")
print(f"Checkpoint: {src['checkpoint'] if src else tgt['checkpoint'] if tgt else 'N/A'}")
print(f"Protocol:    hyperda_v4_3_historical_target_adapt_2015_2025_train2015_2021_val2022_test2023_2025")
print(f"Target:      {target_region}  adaptation_setting=target_full_train")
print("```")
print()

# ── Helper ──────────────────────────────────────────────────────────────────────
def fmt(v, decimals=10):
    if v is None:
        return "N/A"
    if isinstance(v, (int, float)):
        return f"{v:.{decimals}f}"
    return str(v)

# ── Main metrics table ────────────────────────────────────────────────────────
if src and tgt:
    print("| Metric       | Surface (source_test) | Surface (target_eval) | Rootzone (source_test) | Rootzone (target_eval) |")
    print("|:-------------|--------------------:|----------------------:|---------------------:|----------------------:|")
    # Note: RMSE column uses WRMSE (latw), Corr uses Corr_latw
    for metric, label in [("skill_primary", "skill_primary"), ("skill_latw_primary", "skill_latw_pri"), ("skill_median", "skill_median"), ("rmse_latw_mean", "WRMSE"), ("corr_latw_mean", "Corr_latw")]:
        s_surf = src["surface"].get(metric)
        t_surf = tgt["surface"].get(metric)
        s_root = src["rootzone"].get(metric)
        t_root = tgt["rootzone"].get(metric)
        print(f"| {label:<13} | {fmt(s_surf):>20} | {fmt(t_surf):>20} | {fmt(s_root):>18} | {fmt(t_root):>20} |")

    # ── OOD gap section ────────────────────────────────────────────────────────
    ood_gap_surf = (src["surface"]["skill_primary"] - tgt["surface"]["skill_primary"]) if src and tgt else None
    ood_gap_root = (src["rootzone"]["skill_primary"] - tgt["rootzone"]["skill_primary"]) if src and tgt else None
    print()
    print("## OOD Gap (source_test − target_eval, positive = OOD degradation)")
    print("|              |    Surface    |   Rootzone   |")
    print("|:-------------|-------------:|-------------:|")
    print(f"| skill_gap    | {fmt(ood_gap_surf, 2):>10} | {fmt(ood_gap_root, 2):>10} |")
    print()
    print("> Note: positive gap = target_eval worse than source_test (OOD degradation); negative gap = target better than expected (rare).")

    # ── Evaluation metadata ─────────────────────────────────────────────────────
    n_src  = src.get("n_samples_evaluated", "?")
    n_tgt  = tgt.get("n_samples_evaluated", "?")
    mr_src = src.get("n_metric_rows", "?")
    mr_tgt = tgt.get("n_metric_rows", "?")
    # Try to extract eval time from log files
    import re
    def extract_time(log_path):
        m = re.search(r"Evaluation (?:done|completed) in ([\d.]+)s", log_path.read_text() if log_path.exists() else "")
        return float(m.group(1)) if m else None

    log_src = run_dir / "log_source_test.txt"
    log_tgt = run_dir / "log_target_eval.txt"
    t_src = extract_time(log_src)
    t_tgt = extract_time(log_tgt)

    print()
    print("## Evaluation Details")
    print("| Split         | Samples | Metric Rows | Eval Time |")
    print("|:--------------|--------:|------------:|----------:|")
    print(f"| source_test   | {n_src:>6} | {mr_src:>10} | {fmt(t_src, 1) + 's' if t_src else 'N/A':>8} |")
    print(f"| target_eval   | {n_tgt:>6} | {mr_tgt:>10} | {fmt(t_tgt, 1) + 's' if t_tgt else 'N/A':>8} |")
    print()
    print(f"Full results: `{run_dir}/`")

elif src:
    print("> WARNING: target_eval summary not found. Showing source_test only.")
    print()
    print("| Metric       | Surface (source_test) | Rootzone (source_test) |")
    print("|:-------------|--------------------:|---------------------:|")
    for metric, label in [("skill_primary", "skill_primary"), ("skill_latw_primary", "skill_latw_pri"), ("skill_median", "skill_median"), ("rmse_latw_mean", "WRMSE"), ("corr_latw_mean", "Corr_latw")]:
        s_surf = src["surface"].get(metric)
        s_root = src["rootzone"].get(metric)
        print(f"| {label:<13} | {fmt(s_surf):>20} | {fmt(s_root):>20} |")
elif tgt:
    print("> WARNING: source_test summary not found. Showing target_eval only.")
    print()
    print("| Metric       | Surface (target_eval) | Rootzone (target_eval) |")
    print("|:-------------|----------------------:|----------------------:|")
    for metric, label in [("skill_primary", "skill_primary"), ("skill_latw_primary", "skill_latw_pri"), ("skill_median", "skill_median"), ("rmse_latw_mean", "WRMSE"), ("corr_latw_mean", "Corr_latw")]:
        t_surf = tgt["surface"].get(metric)
        t_root = tgt["rootzone"].get(metric)
        print(f"| {label:<13} | {fmt(t_surf):>21} | {fmt(t_root):>21} |")
else:
    print("ERROR: Neither summary found.")
    print(f"  source_test:  {source_summary}")
    print(f"  target_eval:  {target_summary}")

print()
PYTHON_SCRIPT

echo "Done."
