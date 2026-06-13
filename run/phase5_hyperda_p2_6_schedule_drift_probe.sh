#!/bin/bash
# Phase 5 P2.6: schedule/drift diagnostic matrix for HyperDA few-shot adaptation.
#
# Diagnostic-only. Do not use target_eval to select a final schedule.
#
# Usage:
#   bash run/phase5_hyperda_p2_6_schedule_drift_probe.sh <source_checkpoint> US-R1 0 1

set -euo pipefail

SOURCE_CHECKPOINT="${1:-}"
TARGET_REGION="${2:-US-R1}"
SEED="${3:-0}"
CUDA_DEVICE="${4:-${CUDA_VISIBLE_DEVICES:-1}}"
OUTPUT_BASE="${5:-artifacts/runs/phase5_hyperda_p2_6_schedule_drift/${TARGET_REGION}_s${SEED}_$(date -u +%Y%m%dT%H%M%SZ)}"

CURRENT_SPLITS_JSON="${CURRENT_SPLITS_JSON:-artifacts/splits/US_loro_zero_few_shot_splits.json}"
NESTED_SPLITS_JSON="${NESTED_SPLITS_JSON:-artifacts/runs/phase5_hyperda_8_5_comparable_probe/US-R1_s0_20260612T005651Z/nested_support/US-R1_s0_K12_nested_splits.json}"
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
    local split_label="$3"
    local splits_json="$4"
    local adapt_scope="$5"
    local schedule_label="$6"
    local adapt_lr="$7"
    local adapt_steps="$8"
    local anchor_alpha="$9"
    local row_dir="${OUTPUT_BASE}/${run_id}"

    echo ""
    echo ">>> ${run_id}: K=${k} split=${split_label} scope=${adapt_scope} schedule=${schedule_label}"
    K_LIST="${k}" \
    ADAPT_SCOPE="${adapt_scope}" \
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
    SPLITS_JSON="${splits_json}" \
    bash run/phase5_hyperda_zero_few_shot_eval.sh \
        "${SOURCE_CHECKPOINT}" \
        "${TARGET_REGION}" \
        "${SEED}" \
        "${CUDA_DEVICE}" \
        "${row_dir}" \
        2>&1 | tee "${row_dir}.log"

    python3 - "${row_dir}" "${run_id}" "${split_label}" "${schedule_label}" <<'PYTHON_SCRIPT'
import csv
import json
import sys
from pathlib import Path

row_dir = Path(sys.argv[1])
run_id = sys.argv[2]
split_label = sys.argv[3]
schedule_label = sys.argv[4]
overview = row_dir / "overview.csv"
rows = []
if overview.exists():
    with overview.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["run_id"] = run_id
            row["split_type_diagnostic"] = split_label
            row["schedule_label"] = row.get("schedule_label") or schedule_label
            rows.append(row)
out = row_dir / "p2_6_row.json"
out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
PYTHON_SCRIPT
}

run_row "A0_k0_identity_base" "0" "current" "${CURRENT_SPLITS_JSON}" "all" "identity_base" "1e-3" "0" "0.0"
run_row "A1_k4_all_original_schedule" "4" "current" "${CURRENT_SPLITS_JSON}" "all" "original_K4" "${K4_LR}" "${K4_STEPS}" "${K4_ANCHOR_ALPHA}"
run_row "A2_k12_all_original_schedule" "12" "current" "${CURRENT_SPLITS_JSON}" "all" "original_K12" "${K12_LR}" "${K12_STEPS}" "${K12_ANCHOR_ALPHA}"
run_row "A3_k12_all_k4_schedule" "12" "current" "${CURRENT_SPLITS_JSON}" "all" "K4_schedule_on_K12" "${K4_LR}" "${K4_STEPS}" "${K4_ANCHOR_ALPHA}"
run_row "A4_k4_all_k12_schedule" "4" "current" "${CURRENT_SPLITS_JSON}" "all" "K12_schedule_on_K4" "${K12_LR}" "${K12_STEPS}" "${K12_ANCHOR_ALPHA}"
run_row "A5_k12_prompt_only_original_schedule" "12" "current" "${CURRENT_SPLITS_JSON}" "prompt_only" "original_K12_prompt_only" "${K12_LR}" "${K12_STEPS}" "${K12_ANCHOR_ALPHA}"
run_row "A6_k12_prompt_only_k4_schedule" "12" "current" "${CURRENT_SPLITS_JSON}" "prompt_only" "K4_schedule_on_K12_prompt_only" "${K4_LR}" "${K4_STEPS}" "${K4_ANCHOR_ALPHA}"
run_row "A7_k12_all_nested_k4_schedule" "12" "nested" "${NESTED_SPLITS_JSON}" "all" "K4_schedule_on_K12_nested" "${K4_LR}" "${K4_STEPS}" "${K4_ANCHOR_ALPHA}"

python3 - "${OUTPUT_BASE}" <<'PYTHON_SCRIPT'
import csv
import json
import math
import sys
from pathlib import Path

output_base = Path(sys.argv[1])
rows = []
for path in sorted(output_base.glob("A*/p2_6_row.json")):
    rows.extend(json.loads(path.read_text(encoding="utf-8")))

def as_float(value):
    try:
        if value in ("", None):
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None

def overall(row):
    surface = as_float(row.get("surface_skill_primary"))
    rootzone = as_float(row.get("rootzone_skill_primary"))
    if surface is None or rootzone is None:
        return None
    return (surface + rootzone) / 2.0

base = next((r for r in rows if r.get("run_id") == "A0_k0_identity_base"), None)
k4 = next((r for r in rows if r.get("run_id") == "A1_k4_all_original_schedule"), None)
base_skill = overall(base or {})
k4_skill = overall(k4 or {})

for row in rows:
    skill = overall(row)
    row["overall_skill"] = "" if skill is None else f"{skill:.12g}"
    row["delta_vs_K0"] = "" if skill is None or base_skill is None else f"{skill - base_skill:.12g}"
    row["delta_vs_K4_original"] = "" if skill is None or k4_skill is None else f"{skill - k4_skill:.12g}"

fieldnames = [
    "run_id",
    "target_region",
    "seed",
    "split_type_diagnostic",
    "K",
    "adapt_scope",
    "schedule_label",
    "lr",
    "requested_lr",
    "adaptation_steps",
    "requested_max_steps",
    "anchor_alpha",
    "requested_anchor_alpha",
    "weight_decay",
    "grad_clip",
    "effective_support_passes",
    "support_batch_count",
    "adapt_batch_size",
    "eval_batch_size",
    "standard_support_loss_before_full_support",
    "standard_support_loss_after_full_support",
    "surface_skill_primary",
    "rootzone_skill_primary",
    "overall_skill",
    "delta_vs_K0",
    "delta_vs_K4_original",
    "target_parameter_l2_drift_pre_anchor_target_prompt",
    "target_parameter_l2_drift_post_anchor_target_prompt",
    "target_parameter_l2_drift_pre_anchor_monthly_gain",
    "target_parameter_l2_drift_post_anchor_monthly_gain",
    "target_parameter_l2_drift_pre_anchor_adapter_coeff_bottleneck",
    "target_parameter_l2_drift_pre_anchor_adapter_coeff_dec2",
    "target_parameter_l2_drift_pre_anchor_adapter_coeff_dec1",
    "target_parameter_l2_drift_post_anchor_adapter_coeff_bottleneck",
    "target_parameter_l2_drift_post_anchor_adapter_coeff_dec2",
    "target_parameter_l2_drift_post_anchor_adapter_coeff_dec1",
    "support_dates_hash",
    "prediction_content_hash",
    "metric_values_content_hash",
    "actual_optimizer_steps",
    "optimizer_param_count",
    "requires_grad_param_count",
    "target_labels_loaded_for_adaptation",
    "target_labels_used_for_adaptation",
]

csv_path = output_base / "p2_6_schedule_drift_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

json_path = output_base / "p2_6_schedule_drift_summary.json"
json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

md_headers = [
    "run_id",
    "K",
    "split_type_diagnostic",
    "adapt_scope",
    "schedule_label",
    "lr",
    "adaptation_steps",
    "anchor_alpha",
    "effective_support_passes",
    "standard_support_loss_before_full_support",
    "standard_support_loss_after_full_support",
    "overall_skill",
    "delta_vs_K0",
    "delta_vs_K4_original",
    "target_parameter_l2_drift_post_anchor_target_prompt",
    "target_parameter_l2_drift_post_anchor_monthly_gain",
    "support_dates_hash",
    "prediction_content_hash",
    "metric_values_content_hash",
]
def fmt(v):
    if v is None:
        return ""
    text = str(v)
    return text[:12] if len(text) > 24 and "hash" in text.lower() else text

lines = [
    "# P2.6 Schedule/Drift Diagnostic Summary",
    "",
    "|" + "|".join(md_headers) + "|",
    "|" + "|".join([":--"] * len(md_headers)) + "|",
]
for row in rows:
    lines.append("|" + "|".join(fmt(row.get(h, "")) for h in md_headers) + "|")
(output_base / "p2_6_schedule_drift_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Wrote {csv_path}")
PYTHON_SCRIPT

echo ""
echo "P2.6 diagnostic artifacts:"
echo "  ${OUTPUT_BASE}/p2_6_schedule_drift_summary.csv"
echo "  ${OUTPUT_BASE}/p2_6_schedule_drift_summary.md"
