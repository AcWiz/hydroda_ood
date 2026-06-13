#!/bin/bash
# Phase 5 P2.7/P5-lite: conservative adaptation guard diagnostic matrix.
#
# Diagnostic-only. Do not use target_eval to select final rho, radii, or schedule.
#
# Usage:
#   bash run/phase5_hyperda_p2_7_conservative_guard_probe.sh <source_checkpoint> US-R1 0 1

set -euo pipefail

SOURCE_CHECKPOINT="${1:-}"
TARGET_REGION="${2:-US-R1}"
SEED="${3:-0}"
CUDA_DEVICE="${4:-${CUDA_VISIBLE_DEVICES:-1}}"
OUTPUT_BASE="${5:-artifacts/runs/phase5_hyperda_p2_7_conservative_guard/${TARGET_REGION}_s${SEED}_$(date -u +%Y%m%dT%H%M%SZ)}"

CURRENT_SPLITS_JSON="${CURRENT_SPLITS_JSON:-artifacts/splits/US_loro_zero_few_shot_splits.json}"
ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
RUN_OPTIONAL="${RUN_OPTIONAL:-0}"

K4_LR="${K4_LR:-1e-3}"
K4_STEPS="${K4_STEPS:-100}"
K4_ANCHOR_ALPHA="${K4_ANCHOR_ALPHA:-0.75}"
K12_LR="${K12_LR:-3e-4}"
K12_STEPS="${K12_STEPS:-80}"
K12_ANCHOR_ALPHA="${K12_ANCHOR_ALPHA:-0.25}"
ADAPT_WEIGHT_DECAY="${ADAPT_WEIGHT_DECAY:-1e-4}"
ADAPT_GRAD_CLIP="${ADAPT_GRAD_CLIP:-1.0}"

# Mild K4-like diagnostic caps. These are fixed/preregistered for this probe,
# not selected with target_eval.
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
    local adapt_scope="$3"
    local schedule_label="$4"
    local adapt_lr="$5"
    local adapt_steps="$6"
    local anchor_alpha="$7"
    local trust_mode="$8"
    local mix_rho="$9"
    local support_reduction="${10}"
    local row_dir="${OUTPUT_BASE}/${run_id}"

    echo ""
    echo ">>> ${run_id}: K=${k} scope=${adapt_scope} schedule=${schedule_label} trust=${trust_mode} rho=${mix_rho} support_reduction=${support_reduction}"
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
    SPLITS_JSON="${CURRENT_SPLITS_JSON}" \
    TRUST_REGION_MODE="${trust_mode}" \
    TRUST_TOTAL_RADIUS="${TRUST_TOTAL_RADIUS_DEFAULT}" \
    TRUST_PROMPT_RADIUS="${TRUST_PROMPT_RADIUS_DEFAULT}" \
    TRUST_GAIN_RADIUS="${TRUST_GAIN_RADIUS_DEFAULT}" \
    TRUST_COEFF_RADIUS="${TRUST_COEFF_RADIUS_DEFAULT}" \
    TRUST_SPATIAL_RADIUS="${TRUST_SPATIAL_RADIUS_DEFAULT}" \
    SUPPORT_LOSS_REDUCTION="${support_reduction}" \
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
            rows.append(row)
(row_dir / "p2_7_row.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
PYTHON_SCRIPT
}

run_row "A0_k0_identity_base" "0" "all" "identity_base" "1e-3" "0" "0.0" "none" "1.0" "global_pixel"
run_row "A1_k4_all_original" "4" "all" "original_K4" "${K4_LR}" "${K4_STEPS}" "${K4_ANCHOR_ALPHA}" "none" "1.0" "global_pixel"
run_row "A2_k12_all_original" "12" "all" "original_K12" "${K12_LR}" "${K12_STEPS}" "${K12_ANCHOR_ALPHA}" "none" "1.0" "global_pixel"
run_row "A3_k12_all_original_rho_0_5" "12" "all" "original_K12_rho_0_5" "${K12_LR}" "${K12_STEPS}" "${K12_ANCHOR_ALPHA}" "none" "0.5" "global_pixel"
run_row "A4_k12_all_original_rho_0_25" "12" "all" "original_K12_rho_0_25" "${K12_LR}" "${K12_STEPS}" "${K12_ANCHOR_ALPHA}" "none" "0.25" "global_pixel"
run_row "A5_k12_all_k4_schedule" "12" "all" "K4_schedule_on_K12" "${K4_LR}" "${K4_STEPS}" "${K4_ANCHOR_ALPHA}" "none" "1.0" "global_pixel"
run_row "A6_k12_all_k4_schedule_trust" "12" "all" "K4_schedule_on_K12_trust" "${K4_LR}" "${K4_STEPS}" "${K4_ANCHOR_ALPHA}" "groupwise" "1.0" "global_pixel"
run_row "A7_k12_all_k4_schedule_trust_rho_0_5" "12" "all" "K4_schedule_on_K12_trust_rho_0_5" "${K4_LR}" "${K4_STEPS}" "${K4_ANCHOR_ALPHA}" "groupwise" "0.5" "global_pixel"
run_row "A8_k12_prompt_only_original" "12" "prompt_only" "original_K12_prompt_only" "${K12_LR}" "${K12_STEPS}" "${K12_ANCHOR_ALPHA}" "none" "1.0" "global_pixel"
run_row "A9_k12_prompt_only_trust_rho_0_5" "12" "prompt_only" "original_K12_prompt_only_trust_rho_0_5" "${K12_LR}" "${K12_STEPS}" "${K12_ANCHOR_ALPHA}" "groupwise" "0.5" "global_pixel"

if [[ "${RUN_OPTIONAL}" == "1" || "${RUN_OPTIONAL,,}" == "true" ]]; then
    run_row "A10_k12_all_original_cycle_balanced" "12" "all" "original_K12_cycle_balanced" "${K12_LR}" "${K12_STEPS}" "${K12_ANCHOR_ALPHA}" "none" "1.0" "cycle_balanced"
    run_row "A11_k12_all_k4_schedule_cycle_balanced_trust" "12" "all" "K4_schedule_on_K12_cycle_balanced_trust" "${K4_LR}" "${K4_STEPS}" "${K4_ANCHOR_ALPHA}" "groupwise" "1.0" "cycle_balanced"
    run_row "A12_k4_all_original_trust" "4" "all" "original_K4_trust" "${K4_LR}" "${K4_STEPS}" "${K4_ANCHOR_ALPHA}" "groupwise" "1.0" "global_pixel"
fi

python3 - "${OUTPUT_BASE}" <<'PYTHON_SCRIPT'
import csv
import json
import math
import sys
from pathlib import Path

output_base = Path(sys.argv[1])
rows = []
for path in sorted(output_base.glob("A*/p2_7_row.json")):
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

base_skill = overall(next((r for r in rows if r.get("run_id") == "A0_k0_identity_base"), {}) or {})
k4_skill = overall(next((r for r in rows if r.get("run_id") == "A1_k4_all_original"), {}) or {})

for row in rows:
    skill = overall(row)
    row["overall_skill"] = "" if skill is None else f"{skill:.12g}"
    row["delta_vs_K0"] = "" if skill is None or base_skill is None else f"{skill - base_skill:.12g}"
    row["delta_vs_K4"] = "" if skill is None or k4_skill is None else f"{skill - k4_skill:.12g}"

fieldnames = [
    "run_id",
    "target_region",
    "seed",
    "K",
    "adapt_scope",
    "schedule_label",
    "trust_region_mode",
    "trust_total_radius",
    "trust_prompt_radius",
    "trust_gain_radius",
    "trust_coeff_radius",
    "trust_spatial_radius",
    "adapt_mix_rho",
    "support_loss_reduction",
    "surface_skill_primary",
    "rootzone_skill_primary",
    "overall_skill",
    "delta_vs_K0",
    "delta_vs_K4",
    "standard_support_loss_before_full_support",
    "standard_support_loss_after_full_support",
    "support_cycle_loss_improvement_mean",
    "support_cycle_loss_improvement_std",
    "support_gradient_cosine_mean",
    "support_gradient_cosine_min",
    "support_gradient_negative_fraction",
    "target_parameter_l2_drift_pre_anchor_target_prompt",
    "target_parameter_l2_drift_post_anchor_target_prompt",
    "target_parameter_l2_drift_pre_anchor_monthly_gain",
    "target_parameter_l2_drift_post_anchor_monthly_gain",
    "trust_projection_pre_step_drift_max_total",
    "trust_projection_post_step_drift_max_total",
    "prediction_content_hash",
    "zero_shot_prediction_content_hash",
    "adapted_pre_mix_prediction_content_hash",
    "final_mixed_prediction_content_hash",
    "support_dates_hash",
    "target_context_dates_hash",
    "target_eval_dates_hash",
    "source_checkpoint_sha256",
    "split_manifest_sha256",
]

csv_path = output_base / "p2_7_conservative_guard_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

json_path = output_base / "p2_7_conservative_guard_summary.json"
json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

headers = fieldnames[:36]
lines = [
    "# P2.7/P5-lite Conservative Guard Diagnostic Summary",
    "",
    "|" + "|".join(headers) + "|",
    "|" + "|".join([":--"] * len(headers)) + "|",
]
for row in rows:
    lines.append("|" + "|".join(str(row.get(h, ""))[:20] for h in headers) + "|")
(output_base / "p2_7_conservative_guard_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Wrote {csv_path}")
PYTHON_SCRIPT

echo ""
echo "P2.7 diagnostic artifacts:"
echo "  ${OUTPUT_BASE}/p2_7_conservative_guard_summary.csv"
echo "  ${OUTPUT_BASE}/p2_7_conservative_guard_summary.md"
