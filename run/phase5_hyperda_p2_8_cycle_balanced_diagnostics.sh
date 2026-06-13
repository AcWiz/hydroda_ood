#!/bin/bash
# Phase 5 P2.8: missing cycle-balanced target_eval diagnostic rows.
#
# Diagnostic-only. These rows are interpretation-only and must not be read by
# scripts/eval/calibrate_source_safe_guard.py.
#
# Usage:
#   bash run/phase5_hyperda_p2_8_cycle_balanced_diagnostics.sh <source_checkpoint> US-R1 0 1

set -euo pipefail

SOURCE_CHECKPOINT="${1:-}"
TARGET_REGION="${2:-US-R1}"
SEED="${3:-0}"
CUDA_DEVICE="${4:-${CUDA_VISIBLE_DEVICES:-1}}"
OUTPUT_BASE="${5:-artifacts/runs/phase5_hyperda_p2_8_cycle_balanced_diagnostics/${TARGET_REGION}_s${SEED}_$(date -u +%Y%m%dT%H%M%SZ)}"

CURRENT_SPLITS_JSON="${CURRENT_SPLITS_JSON:-artifacts/splits/US_loro_zero_few_shot_splits.json}"
ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"

K4_LR="${K4_LR:-1e-3}"
K4_STEPS="${K4_STEPS:-100}"
K4_ANCHOR_ALPHA="${K4_ANCHOR_ALPHA:-0.75}"
K12_LR="${K12_LR:-3e-4}"
K12_STEPS="${K12_STEPS:-80}"
K12_ANCHOR_ALPHA="${K12_ANCHOR_ALPHA:-0.25}"
ADAPT_WEIGHT_DECAY="${ADAPT_WEIGHT_DECAY:-1e-4}"
ADAPT_GRAD_CLIP="${ADAPT_GRAD_CLIP:-1.0}"

TRUST_TOTAL_RADIUS_DEFAULT="${TRUST_TOTAL_RADIUS_DEFAULT:-3.8}"
TRUST_PROMPT_RADIUS_DEFAULT="${TRUST_PROMPT_RADIUS_DEFAULT:-3.6}"
TRUST_GAIN_RADIUS_DEFAULT="${TRUST_GAIN_RADIUS_DEFAULT:-0.33}"
TRUST_COEFF_RADIUS_DEFAULT="${TRUST_COEFF_RADIUS_DEFAULT:-0.68}"
TRUST_SPATIAL_RADIUS_DEFAULT="${TRUST_SPATIAL_RADIUS_DEFAULT:-0.0}"

cd "$(dirname "$0")/.."

if [[ -z "${SOURCE_CHECKPOINT}" ]]; then
    SOURCE_CHECKPOINT="$(find artifacts/runs/phase4_prompt_conditioned \
        -path "*hyperda_basis_adapter_${TARGET_REGION}_*_s${SEED}_*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
        -type f 2>/dev/null | sort | tail -1)"
fi

if [[ -z "${SOURCE_CHECKPOINT}" || ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "ERROR: source HyperDA checkpoint not found." >&2
    exit 2
fi

mkdir -p "${OUTPUT_BASE}"

run_row() {
    local run_id="$1"
    local k="$2"
    local schedule_label="$3"
    local adapt_lr="$4"
    local adapt_steps="$5"
    local anchor_alpha="$6"
    local trust_mode="$7"
    local mix_rho="$8"
    local row_dir="${OUTPUT_BASE}/${run_id}"

    echo ""
    echo ">>> ${run_id}: diagnostic_only=true target_eval_usage=diagnostic_only_not_for_calibration"
    K_LIST="${k}" \
    ADAPT_SCOPE="all" \
    ADAPT_SOLVER="adamw" \
    SCHEDULE_LABEL="${schedule_label}" \
    ADAPT_LR="${adapt_lr}" \
    ADAPT_MAX_STEPS="${adapt_steps}" \
    ADAPT_ANCHOR_ALPHA="${anchor_alpha}" \
    ADAPT_WEIGHT_DECAY="${ADAPT_WEIGHT_DECAY}" \
    ADAPT_GRAD_CLIP="${ADAPT_GRAD_CLIP}" \
    ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES}" \
    SPLITS_JSON="${CURRENT_SPLITS_JSON}" \
    TRUST_REGION_MODE="${trust_mode}" \
    TRUST_TOTAL_RADIUS="${TRUST_TOTAL_RADIUS_DEFAULT}" \
    TRUST_PROMPT_RADIUS="${TRUST_PROMPT_RADIUS_DEFAULT}" \
    TRUST_GAIN_RADIUS="${TRUST_GAIN_RADIUS_DEFAULT}" \
    TRUST_COEFF_RADIUS="${TRUST_COEFF_RADIUS_DEFAULT}" \
    TRUST_SPATIAL_RADIUS="${TRUST_SPATIAL_RADIUS_DEFAULT}" \
    SUPPORT_LOSS_REDUCTION=cycle_balanced \
    ADAPT_MIX_RHO="${mix_rho}" \
    bash run/phase5_hyperda_zero_few_shot_eval.sh \
        "${SOURCE_CHECKPOINT}" \
        "${TARGET_REGION}" \
        "${SEED}" \
        "${CUDA_DEVICE}" \
        "${row_dir}" \
        2>&1 | tee "${row_dir}.log"

    python3 - "${row_dir}" "${run_id}" <<'PYTHON_SCRIPT'
import csv
import json
import sys
from pathlib import Path

row_dir = Path(sys.argv[1])
run_id = sys.argv[2]
overview = row_dir / "overview.csv"
rows = []
if overview.exists():
    with overview.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["run_id"] = run_id
            row["diagnostic_only"] = "true"
            row["target_eval_usage"] = "diagnostic_only_not_for_calibration"
            row["calibration_usage"] = "forbidden"
            rows.append(row)
(row_dir / "p2_8_cycle_balanced_diagnostic_row.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
PYTHON_SCRIPT
}

run_row "A0_k12_all_original_cycle_balanced" "12" "original_K12_cycle_balanced" "${K12_LR}" "${K12_STEPS}" "${K12_ANCHOR_ALPHA}" "none" "1.0"
run_row "A1_k12_all_k4_schedule_cycle_balanced" "12" "K4_schedule_on_K12_cycle_balanced" "${K4_LR}" "${K4_STEPS}" "${K4_ANCHOR_ALPHA}" "none" "1.0"
run_row "A2_k12_all_k4_schedule_cycle_balanced_trust_rho_0_5" "12" "K4_schedule_on_K12_cycle_balanced_trust_rho_0_5" "${K4_LR}" "${K4_STEPS}" "${K4_ANCHOR_ALPHA}" "groupwise" "0.5"
run_row "A3_k4_all_original_cycle_balanced" "4" "original_K4_cycle_balanced" "${K4_LR}" "${K4_STEPS}" "${K4_ANCHOR_ALPHA}" "none" "1.0"

python3 - "${OUTPUT_BASE}" <<'PYTHON_SCRIPT'
import csv
import json
import sys
from pathlib import Path

output_base = Path(sys.argv[1])
rows = []
for path in sorted(output_base.glob("A*/p2_8_cycle_balanced_diagnostic_row.json")):
    rows.extend(json.loads(path.read_text(encoding="utf-8")))
fieldnames = [
    "run_id",
    "diagnostic_only",
    "target_eval_usage",
    "calibration_usage",
    "target_region",
    "seed",
    "K",
    "schedule_label",
    "trust_region_mode",
    "adapt_mix_rho",
    "support_loss_reduction",
    "surface_skill_primary",
    "rootzone_skill_primary",
    "prediction_content_hash",
    "metric_values_content_hash",
]
csv_path = output_base / "p2_8_cycle_balanced_diagnostic_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
(output_base / "p2_8_cycle_balanced_diagnostic_summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
lines = [
    "# P2.8 Cycle-Balanced Target-Eval Diagnostics",
    "",
    "diagnostic_only=true",
    "target_eval_usage=diagnostic_only_not_for_calibration",
    "",
    "|" + "|".join(fieldnames[:11]) + "|",
    "|" + "|".join([":--"] * 11) + "|",
]
for row in rows:
    lines.append("|" + "|".join(str(row.get(h, "")) for h in fieldnames[:11]) + "|")
(output_base / "p2_8_cycle_balanced_diagnostic_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {csv_path}")
PYTHON_SCRIPT

echo ""
echo "P2.8 diagnostic-only artifacts:"
echo "  ${OUTPUT_BASE}/p2_8_cycle_balanced_diagnostic_summary.csv"
echo "  ${OUTPUT_BASE}/p2_8_cycle_balanced_diagnostic_summary.md"
