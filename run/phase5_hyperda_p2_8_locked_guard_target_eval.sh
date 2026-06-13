#!/bin/bash
# Phase 5 P2.8: locked guard target_eval pass.
#
# Runs K0, K4 all-original, K12 all-original, and one selected K12 guarded
# configuration. If selected rho is rule-based, rho is computed from target_support gradient diagnostics
# after adaptation and before evaluation, never from target_eval.
#
# Usage:
#   bash run/phase5_hyperda_p2_8_locked_guard_target_eval.sh <source_checkpoint> <selected_guard_config.yaml> US-R1 0 1

set -euo pipefail

SOURCE_CHECKPOINT="${1:-}"
SELECTED_GUARD_CONFIG="${2:-}"
TARGET_REGION="${3:-US-R1}"
SEED="${4:-0}"
CUDA_DEVICE="${5:-${CUDA_VISIBLE_DEVICES:-1}}"
OUTPUT_BASE="${6:-artifacts/runs/phase5_hyperda_p2_8_locked_guard_target_eval/${TARGET_REGION}_s${SEED}_$(date -u +%Y%m%dT%H%M%SZ)}"

CURRENT_SPLITS_JSON="${CURRENT_SPLITS_JSON:-artifacts/splits/US_loro_zero_few_shot_splits.json}"
ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
K_LIST="0 4 12"

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

if [[ -z "${SELECTED_GUARD_CONFIG}" || ! -f "${SELECTED_GUARD_CONFIG}" ]]; then
    echo "ERROR: selected_guard_config file not found." >&2
    exit 2
fi

mkdir -p "${OUTPUT_BASE}"

echo "============================================"
echo "Phase 5 P2.8 Locked Guard Target Eval"
echo "  source_checkpoint=${SOURCE_CHECKPOINT}"
echo "  selected_guard_config=${SELECTED_GUARD_CONFIG}"
echo "  target_region=${TARGET_REGION}"
echo "  seed=${SEED}"
echo "  baseline K_LIST=\"${K_LIST}\""
echo "  eval_split=target_eval"
echo "  rule-based rho source=target_support gradient diagnostics after adaptation before eval; never from target_eval"
echo "============================================"

run_baseline_row() {
    local run_id="$1"
    local k="$2"
    local lr="$3"
    local steps="$4"
    local alpha="$5"
    local schedule="$6"
    local row_dir="${OUTPUT_BASE}/${run_id}"
    K_LIST="${k}" \
    ADAPT_SCOPE="all" \
    ADAPT_SOLVER="adamw" \
    SCHEDULE_LABEL="${schedule}" \
    ADAPT_LR="${lr}" \
    ADAPT_MAX_STEPS="${steps}" \
    ADAPT_ANCHOR_ALPHA="${alpha}" \
    ADAPT_WEIGHT_DECAY="${ADAPT_WEIGHT_DECAY}" \
    ADAPT_GRAD_CLIP="${ADAPT_GRAD_CLIP}" \
    ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES}" \
    SPLITS_JSON="${CURRENT_SPLITS_JSON}" \
    TRUST_REGION_MODE="none" \
    SUPPORT_LOSS_REDUCTION="global_pixel" \
    ADAPT_MIX_RHO="1.0" \
    bash run/phase5_hyperda_zero_few_shot_eval.sh \
        "${SOURCE_CHECKPOINT}" \
        "${TARGET_REGION}" \
        "${SEED}" \
        "${CUDA_DEVICE}" \
        "${row_dir}" \
        2>&1 | tee "${row_dir}.log"
}

run_baseline_row "K0_identity_base" "0" "1e-3" "0" "0.0" "identity_base"
run_baseline_row "K4_all_original" "4" "${K4_LR}" "${K4_STEPS}" "${K4_ANCHOR_ALPHA}" "original_K4"
run_baseline_row "K12_all_original" "12" "${K12_LR}" "${K12_STEPS}" "${K12_ANCHOR_ALPHA}" "original_K12"

GUARD_ENV_JSON="${OUTPUT_BASE}/selected_guard_env.json"
PYTHONPATH=. python3 - "${SELECTED_GUARD_CONFIG}" "${GUARD_ENV_JSON}" <<'PYTHON_SCRIPT'
import json
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
env = {
    "candidate_id": cfg.get("candidate_id", ""),
    "guard_config_hash": cfg.get("guard_config_hash", ""),
    "schedule_label": cfg.get("schedule_label", ""),
    "support_loss_reduction": cfg.get("support_loss_reduction", "global_pixel"),
    "rho_policy": cfg.get("rho_policy", "fixed_1.0"),
    "adapt_mix_rho": cfg.get("adapt_mix_rho"),
    "trust_region_mode": cfg.get("trust_region_mode", "none"),
    "trust_total_radius": cfg.get("trust_total_radius", 0.0),
    "trust_prompt_radius": cfg.get("trust_prompt_radius", 0.0),
    "trust_gain_radius": cfg.get("trust_gain_radius", 0.0),
    "trust_coeff_radius": cfg.get("trust_coeff_radius", 0.0),
    "trust_spatial_radius": cfg.get("trust_spatial_radius", 0.0),
    "lr": cfg.get("lr"),
    "adaptation_steps": cfg.get("adaptation_steps"),
    "anchor_alpha": cfg.get("anchor_alpha"),
}
out_path.write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
PYTHON_SCRIPT

read_cfg() {
    python3 - "${GUARD_ENV_JSON}" "$1" <<'PYTHON_SCRIPT'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
value = payload.get(sys.argv[2], "")
print("" if value is None else value)
PYTHON_SCRIPT
}

GUARD_CANDIDATE_ID="$(read_cfg candidate_id)"
GUARD_HASH="$(read_cfg guard_config_hash)"
GUARD_SCHEDULE="$(read_cfg schedule_label)"
GUARD_SUPPORT_LOSS_REDUCTION="$(read_cfg support_loss_reduction)"
GUARD_RHO_POLICY="$(read_cfg rho_policy)"
GUARD_FIXED_RHO="$(read_cfg adapt_mix_rho)"
GUARD_TRUST_MODE="$(read_cfg trust_region_mode)"
GUARD_TRUST_TOTAL="$(read_cfg trust_total_radius)"
GUARD_TRUST_PROMPT="$(read_cfg trust_prompt_radius)"
GUARD_TRUST_GAIN="$(read_cfg trust_gain_radius)"
GUARD_TRUST_COEFF="$(read_cfg trust_coeff_radius)"
GUARD_TRUST_SPATIAL="$(read_cfg trust_spatial_radius)"
GUARD_LR="$(read_cfg lr)"
GUARD_STEPS="$(read_cfg adaptation_steps)"
GUARD_ALPHA="$(read_cfg anchor_alpha)"

GUARD_RUN_DIR="${OUTPUT_BASE}/K12_selected_guarded_${GUARD_HASH:0:12}"
ADAPT_DIR="${GUARD_RUN_DIR}/K12/adapt"
EVAL_DIR="${GUARD_RUN_DIR}/K12/eval"
mkdir -p "${ADAPT_DIR}" "${EVAL_DIR}"

echo ""
echo ">>> selected_guard_config ${GUARD_CANDIDATE_ID} hash=${GUARD_HASH}"
ADAPT_RECIPE="source_anchor" \
ADAPT_SCOPE="all" \
ADAPT_SOLVER="adamw" \
SCHEDULE_LABEL="${GUARD_SCHEDULE}" \
TRUST_REGION_MODE="${GUARD_TRUST_MODE}" \
TRUST_TOTAL_RADIUS="${GUARD_TRUST_TOTAL}" \
TRUST_PROMPT_RADIUS="${GUARD_TRUST_PROMPT}" \
TRUST_GAIN_RADIUS="${GUARD_TRUST_GAIN}" \
TRUST_COEFF_RADIUS="${GUARD_TRUST_COEFF}" \
TRUST_SPATIAL_RADIUS="${GUARD_TRUST_SPATIAL}" \
SUPPORT_LOSS_REDUCTION="${GUARD_SUPPORT_LOSS_REDUCTION}" \
ADAPT_ANCHOR_ALPHA="${GUARD_ALPHA}" \
ADAPT_LR="${GUARD_LR}" \
ADAPT_MAX_STEPS="${GUARD_STEPS}" \
ADAPT_WEIGHT_DECAY="${ADAPT_WEIGHT_DECAY}" \
ADAPT_GRAD_CLIP="${ADAPT_GRAD_CLIP}" \
BATCH_SIZE="${ADAPT_BATCH_SIZE}" \
SPLITS_JSON="${CURRENT_SPLITS_JSON}" \
OUTPUT_DIR="${ADAPT_DIR}" \
bash run/phase5_hyperda_zero_few_shot.sh \
    "${SOURCE_CHECKPOINT}" \
    "${TARGET_REGION}" \
    "12" \
    "${SEED}" \
    "${CUDA_DEVICE}" \
    2>&1 | tee "${ADAPT_DIR}/adapt.log"

if [[ "${GUARD_RHO_POLICY}" == fixed_* ]]; then
    GUARD_EVAL_RHO="${GUARD_FIXED_RHO}"
else
    GUARD_EVAL_RHO="$(PYTHONPATH=. python3 - "${ADAPT_DIR}/metadata.json" "${GUARD_RHO_POLICY}" <<'PYTHON_SCRIPT'
import json
import sys
from scripts.eval.calibrate_source_safe_guard import compute_rho_for_policy
metadata = json.load(open(sys.argv[1], encoding="utf-8"))
diagnostics = metadata.get("support_gradient_diagnostics", {})
print(compute_rho_for_policy(sys.argv[2], diagnostics))
PYTHON_SCRIPT
)"
fi

ADAPTED_CHECKPOINT="${ADAPT_DIR}/checkpoints/checkpoint_final_preregistered.pt"
if [[ ! -f "${ADAPTED_CHECKPOINT}" ]]; then
    echo "ERROR: expected adapted checkpoint not found: ${ADAPTED_CHECKPOINT}" >&2
    exit 2
fi

PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \
    --checkpoint "${ADAPTED_CHECKPOINT}" \
    --target_region "${TARGET_REGION}" \
    --adaptation_setting few_shot_k12 \
    --K 12 \
    --seed "${SEED}" \
    --split_type target_eval \
    --splits_json "${CURRENT_SPLITS_JSON}" \
    --predictor_type hyperda_target_adapt \
    --device cuda \
    --output_dir "${EVAL_DIR}" \
    --max_samples "${EVAL_MAX_SAMPLES}" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --adapt_mix_rho "${GUARD_EVAL_RHO}" \
    2>&1 | tee "${EVAL_DIR}/eval.log"

python3 - "${OUTPUT_BASE}" "${TARGET_REGION}" "${GUARD_CANDIDATE_ID}" "${GUARD_HASH}" "${GUARD_RHO_POLICY}" "${GUARD_EVAL_RHO}" <<'PYTHON_SCRIPT'
import csv
import json
import sys
from pathlib import Path

output_base = Path(sys.argv[1])
target_region = sys.argv[2]
candidate_id = sys.argv[3]
guard_hash = sys.argv[4]
rho_policy = sys.argv[5]
rho = sys.argv[6]
rows = []

def first_present(mapping, keys):
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None

def as_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def overall_skill_from_values(surface, rootzone):
    surface_f = as_float(surface)
    rootzone_f = as_float(rootzone)
    if surface_f is None or rootzone_f is None:
        return None
    return (surface_f + rootzone_f) / 2.0

def summary_skill(summary, variable):
    block = summary.get(variable, {}) or {}
    return first_present(block, ["skill_primary", "skill_latw_primary", "skill_global", "skill_latw_global"])

def enrich_skill_row(row):
    surface = first_present(row, ["surface_skill_primary", "surface_skill_latw_primary", "surface_skill_global"])
    rootzone = first_present(row, ["rootzone_skill_primary", "rootzone_skill_latw_primary", "rootzone_skill_global"])
    row["surface_skill_primary"] = surface
    row["rootzone_skill_primary"] = rootzone
    row["overall_skill_primary"] = overall_skill_from_values(surface, rootzone)
    return row

for overview in sorted(output_base.glob("*/overview.csv")):
    with overview.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["p2_8_role"] = "baseline"
            rows.append(enrich_skill_row(row))
guard_dir = output_base / f"K12_selected_guarded_{guard_hash[:12]}"
summary_path = guard_dir / "K12" / "eval" / target_region / "summary.json"
adapt_path = guard_dir / "K12" / "adapt" / "metadata.json"
summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
adapt = json.loads(adapt_path.read_text(encoding="utf-8")) if adapt_path.exists() else {}
if summary or adapt:
    trust_diag = adapt.get("trust_projection_diagnostics", {}) or {}
    drift = adapt.get("target_parameter_l2_drift", {}) or {}
    pre_anchor_drift = adapt.get("target_parameter_l2_drift_pre_anchor", {}) or {}
    post_anchor_drift = adapt.get("target_parameter_l2_drift_post_anchor", drift) or {}
    support_diag = adapt.get("support_gradient_diagnostics", {}) or {}
    selected_row = enrich_skill_row(
        {
            "p2_8_role": "selected_guarded",
            "selected_guard_candidate_id": candidate_id,
            "guard_config_hash": guard_hash,
            "selected_guard_config_hash": guard_hash,
            "rho_policy": rho_policy,
            "target_region": target_region,
            "seed": summary.get("seed", adapt.get("seed")),
            "K": 12,
            "adaptation_setting": "few_shot_k12",
            "adapt_scope": adapt.get("adapt_scope", "all"),
            "adapt_solver": adapt.get("adapt_solver", "adamw"),
            "schedule_label": adapt.get("schedule_label", ""),
            "trust_region_mode": adapt.get("trust_region_mode", ""),
            "trust_total_radius": adapt.get("trust_total_radius", ""),
            "trust_prompt_radius": adapt.get("trust_prompt_radius", ""),
            "trust_gain_radius": adapt.get("trust_gain_radius", ""),
            "trust_coeff_radius": adapt.get("trust_coeff_radius", ""),
            "trust_spatial_radius": adapt.get("trust_spatial_radius", ""),
            "support_loss_reduction": adapt.get("support_loss_reduction", ""),
            "adapt_mix_rho": rho,
            "lr": adapt.get("lr"),
            "adaptation_steps": adapt.get("adaptation_steps"),
            "anchor_alpha": adapt.get("anchor_alpha"),
            "weight_decay": adapt.get("weight_decay"),
            "grad_clip": adapt.get("grad_clip"),
            "surface_skill_primary": summary_skill(summary, "surface"),
            "rootzone_skill_primary": summary_skill(summary, "rootzone"),
            "support_loss_before": adapt.get("support_loss_before"),
            "support_loss_after": adapt.get("support_loss_after"),
            "support_final_loss": adapt.get("support_final_loss"),
            "support_loss_delta": adapt.get("support_loss_delta"),
            "standard_support_loss_before_full_support": adapt.get("standard_support_loss_before_full_support"),
            "standard_support_loss_after_full_support": adapt.get("standard_support_loss_after_full_support"),
            "standard_support_loss_delta_full_support": adapt.get("standard_support_loss_delta_full_support"),
            "support_gradient_cosine_mean": first_present(adapt, ["support_gradient_cosine_mean"]) or support_diag.get("support_gradient_cosine_mean"),
            "support_gradient_cosine_min": first_present(adapt, ["support_gradient_cosine_min"]) or support_diag.get("support_gradient_cosine_min"),
            "support_gradient_negative_fraction": first_present(adapt, ["support_gradient_negative_fraction"]) or support_diag.get("support_gradient_negative_fraction"),
            "support_cycle_loss_improvement_mean": first_present(adapt, ["support_cycle_loss_improvement_mean"]) or support_diag.get("support_cycle_loss_improvement_mean"),
            "support_cycle_loss_improvement_std": first_present(adapt, ["support_cycle_loss_improvement_std"]) or support_diag.get("support_cycle_loss_improvement_std"),
            "trust_projection_applied_count": trust_diag.get("trust_projection_applied_count"),
            "trust_projection_pre_step_drift_max_total": trust_diag.get("trust_projection_pre_step_drift_max_total"),
            "trust_projection_post_step_drift_max_total": trust_diag.get("trust_projection_post_step_drift_max_total"),
            "trust_projection_pre_step_drift_last_total": trust_diag.get("trust_projection_pre_step_drift_last_total"),
            "trust_projection_post_step_drift_last_total": trust_diag.get("trust_projection_post_step_drift_last_total"),
            "target_parameter_l2_drift_total": drift.get("total"),
            "target_parameter_l2_drift_pre_anchor_total": pre_anchor_drift.get("total"),
            "target_parameter_l2_drift_post_anchor_total": post_anchor_drift.get("total"),
            "target_parameter_l2_drift_pre_anchor_target_prompt": pre_anchor_drift.get("target_prompt"),
            "target_parameter_l2_drift_post_anchor_target_prompt": post_anchor_drift.get("target_prompt"),
            "target_parameter_l2_drift_pre_anchor_monthly_gain": first_present(pre_anchor_drift, ["monthly_gain", "monthly_residual_gain"]),
            "target_parameter_l2_drift_post_anchor_monthly_gain": first_present(post_anchor_drift, ["monthly_gain", "monthly_residual_gain"]),
            "target_support_count": adapt.get("target_support_count"),
            "target_support_dates": ",".join(adapt.get("target_support_dates", []) or []),
            "target_labels_loaded_for_adaptation": adapt.get("target_labels_loaded_for_adaptation"),
            "target_labels_used_for_adaptation": adapt.get("target_labels_used_for_adaptation"),
            "prediction_content_hash": summary.get("prediction_content_hash", ""),
            "metric_values_content_hash": summary.get("metric_values_content_hash", ""),
            "target_context_dates_hash": summary.get("target_context_dates_hash", adapt.get("target_context_dates_hash", "")),
            "target_support_dates_hash": summary.get("target_support_dates_hash", adapt.get("target_support_dates_hash", "")),
            "target_eval_dates_hash": summary.get("target_eval_dates_hash", ""),
            "split_manifest_sha256": summary.get("split_manifest_sha256", adapt.get("split_manifest_sha256", "")),
            "source_checkpoint_sha256": adapt.get("source_checkpoint_sha256", summary.get("source_checkpoint_sha256", "")),
            "checkpoint": str(adapt_path.parent / "checkpoints" / "checkpoint_final_preregistered.pt"),
            "summary": str(summary_path),
            "adapt_metadata": str(adapt_path),
        }
    )
    rows.append(selected_row)

baseline_by_k = {str(row.get("K")): row for row in rows if row.get("p2_8_role") == "baseline"}
k0_overall = as_float((baseline_by_k.get("0") or {}).get("overall_skill_primary"))
k4_overall = as_float((baseline_by_k.get("4") or {}).get("overall_skill_primary"))
for row in rows:
    overall = as_float(row.get("overall_skill_primary"))
    row["delta_vs_K0_overall_skill"] = overall - k0_overall if overall is not None and k0_overall is not None else None
    row["delta_vs_K4_overall_skill"] = overall - k4_overall if overall is not None and k4_overall is not None else None
    row["negative_transfer_vs_K0"] = (
        row["delta_vs_K0_overall_skill"] < 0.0 if row["delta_vs_K0_overall_skill"] is not None else None
    )
    row["negative_transfer_vs_K4"] = (
        row["delta_vs_K4_overall_skill"] < 0.0 if row["delta_vs_K4_overall_skill"] is not None else None
    )
fieldnames = sorted({key for row in rows for key in row})
csv_path = output_base / "p2_8_locked_guard_target_eval_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
(output_base / "p2_8_locked_guard_target_eval_summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {csv_path}")
PYTHON_SCRIPT

echo ""
echo "P2.8 locked target_eval artifacts:"
echo "  ${OUTPUT_BASE}/p2_8_locked_guard_target_eval_summary.csv"
