#!/bin/bash
# Phase 5: one-click HyperDA zero/few-shot adaptation and target evaluation.
#
# Usage:
#   bash run/phase5_hyperda_zero_few_shot_eval.sh [source_checkpoint] [target_region] [seed] [cuda_device] [output_base]
#   bash run/phase5_hyperda_zero_few_shot_eval.sh /path/to/source.pt US-R1 0 1
#   bash run/phase5_hyperda_zero_few_shot_eval.sh "" US-R1 0 1
#   MAX_STEPS=1 EVAL_MAX_SAMPLES=2 K_LIST="0 4" \
#     bash run/phase5_hyperda_zero_few_shot_eval.sh /path/to/source.pt US-R1 0 1

set -euo pipefail

SOURCE_CHECKPOINT="${1:-}"
TARGET_REGION="${2:-US-R1}"
SEED="${3:-0}"
export CUDA_VISIBLE_DEVICES="${4:-${CUDA_VISIBLE_DEVICES:-1}}"
OUTPUT_BASE="${5:-artifacts/runs/phase5_hyperda_zero_few_shot_eval/${TARGET_REGION}_s${SEED}_$(date -u +%Y%m%dT%H%M%SZ)}"
K_LIST="${K_LIST:-0 4 12}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
ADAPT_RECIPE="${ADAPT_RECIPE:-source_anchor}"
ANCHOR_ALPHA_K4="${ANCHOR_ALPHA_K4:-0.75}"
ANCHOR_ALPHA_K12="${ANCHOR_ALPHA_K12:-0.25}"
LR_K12="${LR_K12:-3e-4}"
MAX_STEPS_K12="${MAX_STEPS_K12:-80}"

cd "$(dirname "$0")/.."

if [[ -z "${SOURCE_CHECKPOINT}" ]]; then
    SOURCE_CHECKPOINT="$(find artifacts/runs/phase4_prompt_conditioned \
        -path "*hyperda_basis_adapter_${TARGET_REGION}_*_s${SEED}_*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
        -type f 2>/dev/null | sort | tail -1)"
fi

if [[ -z "${SOURCE_CHECKPOINT}" || ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "ERROR: source HyperDA checkpoint not found." >&2
    echo "Provide it explicitly:" >&2
    echo "  bash run/phase5_hyperda_zero_few_shot_eval.sh /path/to/source.pt ${TARGET_REGION} ${SEED} ${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_BASE}"

echo "============================================"
echo "Phase 5 HyperDA Zero/Few-Shot Adapt + Eval"
echo "  source_checkpoint=${SOURCE_CHECKPOINT}"
echo "  target_region=${TARGET_REGION}"
echo "  seed=${SEED}"
echo "  K_LIST=${K_LIST}"
echo "  ADAPT_RECIPE=${ADAPT_RECIPE}"
echo "  eval_split=target_eval"
echo "  eval_max_samples=${EVAL_MAX_SAMPLES}"
echo "  output_base=${OUTPUT_BASE}"
echo "  device=gpu:${CUDA_VISIBLE_DEVICES}"
echo "============================================"

for K in ${K_LIST}; do
    if [[ "${K}" == "0" ]]; then
        ADAPTATION_SETTING="zero_shot_context"
        ADAPTATION_MAX_STEPS="0"
        ADAPTATION_LR="${LR_K0:-${LR:-1e-3}}"
        ANCHOR_ALPHA="0.0"
    elif [[ "${K}" == "4" ]]; then
        ADAPTATION_SETTING="few_shot_k4"
        ADAPTATION_MAX_STEPS="${MAX_STEPS_K4:-${MAX_STEPS:-100}}"
        ADAPTATION_LR="${LR_K4:-${LR:-1e-3}}"
        ANCHOR_ALPHA="${ANCHOR_ALPHA_K4}"
    elif [[ "${K}" == "12" ]]; then
        ADAPTATION_SETTING="few_shot_k12"
        ADAPTATION_MAX_STEPS="${MAX_STEPS_K12:-${MAX_STEPS:-80}}"
        ADAPTATION_LR="${LR_K12:-${LR:-3e-4}}"
        ANCHOR_ALPHA="${ANCHOR_ALPHA_K12}"
    else
        echo "ERROR: K_LIST may contain only 0, 4, 12; got ${K}" >&2
        exit 2
    fi

    K_DIR="${OUTPUT_BASE}/K${K}"
    ADAPT_DIR="${K_DIR}/adapt"
    EVAL_DIR="${K_DIR}/eval"
    mkdir -p "${ADAPT_DIR}" "${EVAL_DIR}"

    echo ""
    echo ">>> [K=${K}] Adaptation: ${ADAPTATION_SETTING}"
    ADAPT_RECIPE="${ADAPT_RECIPE}" ANCHOR_ALPHA="${ANCHOR_ALPHA}" LR="${ADAPTATION_LR}" MAX_STEPS="${ADAPTATION_MAX_STEPS}" OUTPUT_DIR="${ADAPT_DIR}" bash run/phase5_hyperda_zero_few_shot.sh \
        "${SOURCE_CHECKPOINT}" \
        "${TARGET_REGION}" \
        "${K}" \
        "${SEED}" \
        "${CUDA_VISIBLE_DEVICES}" \
        2>&1 | tee "${ADAPT_DIR}/adapt.log"

    ADAPTED_CHECKPOINT="${ADAPT_DIR}/checkpoints/checkpoint_final_preregistered.pt"
    if [[ ! -f "${ADAPTED_CHECKPOINT}" ]]; then
        echo "ERROR: expected adapted checkpoint not found: ${ADAPTED_CHECKPOINT}" >&2
        exit 2
    fi

    echo ""
    echo ">>> [K=${K}] Evaluation on target_eval"
    PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \
        --checkpoint "${ADAPTED_CHECKPOINT}" \
        --target_region "${TARGET_REGION}" \
        --adaptation_setting "${ADAPTATION_SETTING}" \
        --K "${K}" \
        --seed "${SEED}" \
        --split_type target_eval \
        --predictor_type hyperda_target_adapt \
        --device cuda \
        --output_dir "${EVAL_DIR}" \
        --max_samples "${EVAL_MAX_SAMPLES}" \
        2>&1 | tee "${EVAL_DIR}/eval.log"
done

python3 - "${OUTPUT_BASE}" "${TARGET_REGION}" "${K_LIST}" <<'PYTHON_SCRIPT'
import csv
import json
import math
import sys
from pathlib import Path

output_base = Path(sys.argv[1])
target_region = sys.argv[2]
k_values = sys.argv[3].split()
settings = {
    "0": "zero_shot_context",
    "4": "few_shot_k4",
    "12": "few_shot_k12",
}

def fmt(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "NA"
    if isinstance(value, (int, float)):
        return f"{value:.10f}"
    if value is None:
        return "NA"
    return str(value)

def first_present(mapping, keys):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None

def metric_block(summary, variable):
    block = summary.get(variable, {}) or {}
    return {
        f"{variable}_skill_primary": first_present(block, ["skill_primary", "skill_global"]),
        f"{variable}_skill_latw_primary": first_present(block, ["skill_latw_primary", "skill_latw_global"]),
        f"{variable}_skill_global": block.get("skill_global"),
        f"{variable}_skill_latw_global": block.get("skill_latw_global"),
        f"{variable}_skill_median": block.get("skill_median"),
        f"{variable}_rmse_latw": first_present(block, ["rmse_latw_mean", "analysis_rmse_latw_mean"]),
        f"{variable}_rmse_mean": block.get("rmse_mean"),
        f"{variable}_corr_latw": block.get("corr_latw_mean"),
        f"{variable}_corr_mean": block.get("corr_mean"),
    }

def print_markdown_table(rows, columns):
    headers = [column[0] if isinstance(column, tuple) else column for column in columns]
    keys = [column[1] if isinstance(column, tuple) else column for column in columns]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join([":--"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(fmt(row.get(key)) for key in keys) + " |")

rows = []
for k in k_values:
    summary_path = output_base / f"K{k}" / "eval" / target_region / "summary.json"
    adapt_metadata_path = output_base / f"K{k}" / "adapt" / "metadata.json"
    metrics_long_path = output_base / f"K{k}" / "eval" / target_region / "metrics_long.csv"
    metrics_by_region_path = output_base / f"K{k}" / "eval" / target_region / "metrics_by_region.csv"
    metrics_by_season_path = output_base / f"K{k}" / "eval" / target_region / "metrics_by_season.csv"
    checkpoint_path = output_base / f"K{k}" / "adapt" / "checkpoints" / "checkpoint_final_preregistered.pt"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        status = "ok"
    else:
        summary = {}
        status = "missing_summary"
    if adapt_metadata_path.exists():
        with open(adapt_metadata_path, encoding="utf-8") as f:
            adapt_metadata = json.load(f)
    else:
        adapt_metadata = {}
    drift = adapt_metadata.get("target_parameter_l2_drift", {}) or {}
    row = {
        "target_region": target_region,
        "K": k,
        "adaptation_setting": settings.get(k, ""),
        "adapt_recipe": adapt_metadata.get("adapt_recipe"),
        "status": status,
        "method": summary.get("method"),
        "split_type": summary.get("split_type"),
        "seed": summary.get("seed"),
        "adaptation_steps": adapt_metadata.get("adaptation_steps"),
        "lr": adapt_metadata.get("lr"),
        "anchor_alpha": adapt_metadata.get("anchor_alpha"),
        "source_anchor_hyperparameter_source": adapt_metadata.get("source_anchor_hyperparameter_source"),
        "trainable_parameter_count": adapt_metadata.get("trainable_parameter_count"),
        "support_final_loss": adapt_metadata.get("support_final_loss"),
        "support_loss_delta": adapt_metadata.get("support_loss_delta"),
        "target_parameter_l2_drift_total": drift.get("total"),
        "target_parameter_l2_drift_target_prompt": drift.get("target_prompt"),
        "target_parameter_l2_drift_adapter_coefficients": drift.get("adapter_coefficient_residuals"),
        "target_parameter_l2_drift_monthly_gain": drift.get("monthly_residual_gain"),
        "target_support_dates": ",".join(adapt_metadata.get("target_support_dates", []) or []),
        "n_samples_evaluated": summary.get("n_samples_evaluated"),
        "n_metric_rows": summary.get("n_metric_rows"),
        "eval_time_s": summary.get("eval_time_s"),
        "protocol_freeze_id": summary.get("protocol_freeze_id"),
        "split_manifest_sha256": summary.get("split_manifest_sha256"),
        "target_context_dates_hash": summary.get("target_context_dates_hash"),
        "target_support_dates_hash": summary.get("target_support_dates_hash"),
        "target_eval_dates_hash": summary.get("target_eval_dates_hash"),
        "checkpoint": str(checkpoint_path),
        "adapt_metadata": str(adapt_metadata_path),
        "summary": str(summary_path),
        "metrics_long": str(metrics_long_path),
        "metrics_by_region": str(metrics_by_region_path),
        "metrics_by_season": str(metrics_by_season_path),
    }
    row.update(metric_block(summary, "surface"))
    row.update(metric_block(summary, "rootzone"))
    rows.append(row)

csv_path = output_base / "overview.csv"
fieldnames = [
    "target_region",
    "K",
    "adaptation_setting",
    "adapt_recipe",
    "status",
    "method",
    "split_type",
    "seed",
    "adaptation_steps",
    "lr",
    "anchor_alpha",
    "source_anchor_hyperparameter_source",
    "trainable_parameter_count",
    "support_final_loss",
    "support_loss_delta",
    "target_parameter_l2_drift_total",
    "target_parameter_l2_drift_target_prompt",
    "target_parameter_l2_drift_adapter_coefficients",
    "target_parameter_l2_drift_monthly_gain",
    "target_support_dates",
    "n_samples_evaluated",
    "n_metric_rows",
    "surface_skill_primary",
    "surface_skill_latw_primary",
    "surface_skill_global",
    "surface_skill_latw_global",
    "surface_skill_median",
    "surface_rmse_latw",
    "surface_rmse_mean",
    "surface_corr_latw",
    "surface_corr_mean",
    "rootzone_skill_primary",
    "rootzone_skill_latw_primary",
    "rootzone_skill_global",
    "rootzone_skill_latw_global",
    "rootzone_skill_median",
    "rootzone_rmse_latw",
    "rootzone_rmse_mean",
    "rootzone_corr_latw",
    "rootzone_corr_mean",
    "eval_time_s",
    "protocol_freeze_id",
    "split_manifest_sha256",
    "target_context_dates_hash",
    "target_support_dates_hash",
    "target_eval_dates_hash",
    "checkpoint",
    "adapt_metadata",
    "summary",
    "metrics_long",
    "metrics_by_region",
    "metrics_by_season",
]
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

json_path = output_base / "overview.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)

md_path = output_base / "overview.md"
headers = [
    "target_region",
    "K",
    "adaptation_setting",
    "adapt_recipe",
    "status",
    "adaptation_steps",
    "lr",
    "anchor_alpha",
    "trainable_parameter_count",
    "support_final_loss",
    "support_loss_delta",
    "target_parameter_l2_drift_total",
    "surface_skill_primary",
    "surface_skill_latw_primary",
    "surface_rmse_latw",
    "surface_corr_latw",
    "rootzone_skill_primary",
    "rootzone_skill_latw_primary",
    "rootzone_rmse_latw",
    "rootzone_corr_latw",
    "n_samples_evaluated",
    "n_metric_rows",
    "summary",
    "metrics_by_region",
]
lines = [
    "# HyperDA Zero/Few-Shot Target Eval Overview",
    "",
    f"Target region: `{target_region}`",
    "",
    "|" + "|".join(headers) + "|",
    "|" + "|".join([":--"] * len(headers)) + "|",
]
for row in rows:
    lines.append("|" + "|".join(fmt(row.get(h)) for h in headers) + "|")
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

terminal_columns = [
    ("K", "K"),
    ("setting", "adaptation_setting"),
    ("status", "status"),
    ("surface_WRMSE", "surface_rmse_latw"),
    ("rootzone_WRMSE", "rootzone_rmse_latw"),
]

print()
print("============================================")
print("HyperDA Zero/Few-Shot Target Eval Overview")
print("============================================")
print(f"Overview CSV: {csv_path}")
print(f"Overview MD:  {md_path}")
print(f"Overview JSON: {json_path}")
print()
print("Quick target_eval WRMSE table:")
print_markdown_table(rows, terminal_columns)
PYTHON_SCRIPT

echo ""
echo "Done. Outputs are under: ${OUTPUT_BASE}"
