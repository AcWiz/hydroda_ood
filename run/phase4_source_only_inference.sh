#!/bin/bash
# Phase 4 Source-Only Inference: In-domain (source_val) vs OOD (target_query)
#
# Usage:
#   bash run/phase4_source_only_inference.sh
#   bash run/phase4_source_only_inference.sh /path/to/checkpoint.pt
#
# Evaluates the source-only backbone on:
#   - source_val: 2021 held-out source regions R2-R6 (in-domain baseline, NOT target)
#   - target_query: 2023-2025 US-R1 target pixels (OOD signal)

set -euo pipefail

CHECKPOINT_PATH="${1:-}"

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

# Extract run ID for output naming
RUN_ID=$(basename "$(dirname "$CHECKPOINT_PATH")")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="artifacts/results/phase4_source_only_inference/${RUN_ID}_${TIMESTAMP}"

# Fixed evaluation parameters matching training
TARGET_REGION="US-R1"
K="0"
SEED="0"
DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

echo "============================================"
echo "Phase 4 Source-Only Inference"
echo "  checkpoint=${CHECKPOINT_PATH}"
echo "  target_region=${TARGET_REGION}  K=${K}  seed=${SEED}"
echo "  output_base=${OUTPUT_BASE}"
echo "  device=gpu:${DEVICE}"
echo "============================================"

# Create output directory
mkdir -p "${OUTPUT_BASE}"

# Run 1: source_val (in-domain)
echo ""
echo ">>> [1/2] Evaluating source_val on source regions R2-R6 (in-domain)..."
EVAL_SOURCE_DIR="${OUTPUT_BASE}/source_val"
PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --target_region "${TARGET_REGION}" \
    --K "${K}" \
    --seed "${SEED}" \
    --split_type source_val \
    --predictor_type source_only \
    --device cuda \
    --output_dir "${EVAL_SOURCE_DIR}" \
    2>&1 | tee "${OUTPUT_BASE}/log_source_val.txt"

SOURCE_SUMMARY="${EVAL_SOURCE_DIR}/${TARGET_REGION}/summary.json"

# Run 2: target_query (OOD)
echo ""
echo ">>> [2/2] Evaluating target_query on target region US-R1 (OOD)..."
EVAL_TARGET_DIR="${OUTPUT_BASE}/target_query"
PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --target_region "${TARGET_REGION}" \
    --K "${K}" \
    --seed "${SEED}" \
    --split_type target_query \
    --predictor_type source_only \
    --device cuda \
    --output_dir "${EVAL_TARGET_DIR}" \
    2>&1 | tee "${OUTPUT_BASE}/log_target_query.txt"

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

python3 - <<'PYTHON_SCRIPT'
import json
import sys
from pathlib import Path

output_base = Path("artifacts/results/phase4_source_only_inference").absolute()
run_dirs = sorted(output_base.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
run_dir = run_dirs[0] if run_dirs else None

if run_dir is None:
    print("ERROR: No inference output directory found.")
    sys.exit(1)

source_summary = run_dir / "source_val" / "US-R1" / "summary.json"
target_summary = run_dir / "target_query" / "US-R1" / "summary.json"

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
print(f"Protocol:    hyperda_v4_final_2015_2025_context2022_query2023_2025_k0_4_12")
print(f"Target:      US-R1  K=0  seed=0")
print("```")
print()

# ── Helper ──────────────────────────────────────────────────────────────────────
def fmt(v, decimals=4):
    if v is None:
        return "N/A"
    if isinstance(v, (int, float)):
        return f"{v:.{decimals}f}"
    return str(v)

# ── Main metrics table ────────────────────────────────────────────────────────
if src and tgt:
    print("| Metric       | Surface (source_val) | Surface (target_query) | Rootzone (source_val) | Rootzone (target_query) |")
    print("|:-------------|--------------------:|----------------------:|---------------------:|----------------------:|")
    for metric in ["skill_mean", "skill_std", "rmse_mean", "corr_mean"]:
        s_surf = src["surface"].get(metric)
        t_surf = tgt["surface"].get(metric)
        s_root = src["rootzone"].get(metric)
        t_root = tgt["rootzone"].get(metric)
        print(f"| {metric:<13} | {fmt(s_surf):>20} | {fmt(t_surf):>20} | {fmt(s_root):>18} | {fmt(t_root):>20} |")

    # ── OOD gap section ────────────────────────────────────────────────────────
    ood_gap_surf = (src["surface"]["skill_mean"] - tgt["surface"]["skill_mean"]) if src and tgt else None
    ood_gap_root = (src["rootzone"]["skill_mean"] - tgt["rootzone"]["skill_mean"]) if src and tgt else None
    print()
    print("## OOD Gap (source_val − target_query, positive = OOD degradation)")
    print("|              |    Surface    |   Rootzone   |")
    print("|:-------------|-------------:|-------------:|")
    print(f"| skill_gap    | {fmt(ood_gap_surf, 2):>10} | {fmt(ood_gap_root, 2):>10} |")
    print()
    print("> Note: positive gap = target_query worse than source_val (OOD degradation); negative gap = target better than expected (rare).")

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

    log_src = run_dir / "log_source_val.txt"
    log_tgt = run_dir / "log_target_query.txt"
    t_src = extract_time(log_src)
    t_tgt = extract_time(log_tgt)

    print()
    print("## Evaluation Details")
    print("| Split         | Samples | Metric Rows | Eval Time |")
    print("|:--------------|--------:|------------:|----------:|")
    print(f"| source_val    | {n_src:>6} | {mr_src:>10} | {fmt(t_src, 1) + 's' if t_src else 'N/A':>8} |")
    print(f"| target_query  | {n_tgt:>6} | {mr_tgt:>10} | {fmt(t_tgt, 1) + 's' if t_tgt else 'N/A':>8} |")
    print()
    print(f"Full results: `artifacts/results/phase4_source_only_inference/{run_dir.name}/`")

elif src:
    print("> WARNING: target_query summary not found. Showing source_val only.")
    print()
    print("| Metric       | Surface (source_val) | Rootzone (source_val) |")
    print("|:-------------|--------------------:|---------------------:|")
    for metric in ["skill_mean", "skill_std", "rmse_mean", "corr_mean"]:
        s_surf = src["surface"].get(metric)
        s_root = src["rootzone"].get(metric)
        print(f"| {metric:<13} | {fmt(s_surf):>20} | {fmt(s_root):>20} |")
elif tgt:
    print("> WARNING: source_val summary not found. Showing target_query only.")
    print()
    print("| Metric       | Surface (target_query) | Rootzone (target_query) |")
    print("|:-------------|----------------------:|----------------------:|")
    for metric in ["skill_mean", "skill_std", "rmse_mean", "corr_mean"]:
        t_surf = tgt["surface"].get(metric)
        t_root = tgt["rootzone"].get(metric)
        print(f"| {metric:<13} | {fmt(t_surf):>21} | {fmt(t_root):>21} |")
else:
    print("ERROR: Neither summary found.")
    print(f"  source_val:   {source_summary}")
    print(f"  target_query: {target_summary}")

print()
PYTHON_SCRIPT

echo "Done."
