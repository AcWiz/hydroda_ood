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
ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-${BATCH_SIZE:-8}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE:-8}}"
SPLITS_JSON="${SPLITS_JSON:-artifacts/splits/US_loro_zero_few_shot_splits.json}"
ADAPT_RECIPE="${ADAPT_RECIPE:-source_anchor}"
ADAPT_SCOPE="${ADAPT_SCOPE:-all}"
ADAPT_SOLVER="${ADAPT_SOLVER:-adamw}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-}"
FREEZE_MONTHLY_GAIN="${FREEZE_MONTHLY_GAIN:-0}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1.0}"
RIDGE_CLIP_COEFF_NORM="${RIDGE_CLIP_COEFF_NORM:-1.0}"
RIDGE_TRUST_REGION_RADIUS="${RIDGE_TRUST_REGION_RADIUS:-1.0}"
RIDGE_MAX_FEATURE_PIXELS="${RIDGE_MAX_FEATURE_PIXELS:-20000}"
RIDGE_STANDARDIZE_FEATURES="${RIDGE_STANDARDIZE_FEATURES:-0}"
TRUST_REGION_MODE="${TRUST_REGION_MODE:-none}"
TRUST_TOTAL_RADIUS="${TRUST_TOTAL_RADIUS:-0.0}"
TRUST_PROMPT_RADIUS="${TRUST_PROMPT_RADIUS:-0.0}"
TRUST_GAIN_RADIUS="${TRUST_GAIN_RADIUS:-0.0}"
TRUST_COEFF_RADIUS="${TRUST_COEFF_RADIUS:-0.0}"
TRUST_SPATIAL_RADIUS="${TRUST_SPATIAL_RADIUS:-0.0}"
SUPPORT_LOSS_REDUCTION="${SUPPORT_LOSS_REDUCTION:-global_pixel}"
ADAPT_MIX_RHO="${ADAPT_MIX_RHO:-1.0}"
AUDIT_IDENTITY="${AUDIT_IDENTITY:-0}"
AUDIT_IDENTITY_TOLERANCE="${AUDIT_IDENTITY_TOLERANCE:-1e-8}"
ANCHOR_ALPHA_K4="${ANCHOR_ALPHA_K4:-0.75}"
ANCHOR_ALPHA_K12="${ANCHOR_ALPHA_K12:-0.25}"
LR_K12="${LR_K12:-3e-4}"
MAX_STEPS_K12="${MAX_STEPS_K12:-80}"
ADAPT_LR_OVERRIDE="${ADAPT_LR:-}"
ADAPT_MAX_STEPS_OVERRIDE="${ADAPT_MAX_STEPS:-}"
ADAPT_ANCHOR_ALPHA_OVERRIDE="${ADAPT_ANCHOR_ALPHA:-}"
ADAPT_WEIGHT_DECAY_OVERRIDE="${ADAPT_WEIGHT_DECAY:-${WEIGHT_DECAY:-}}"
ADAPT_GRAD_CLIP_OVERRIDE="${ADAPT_GRAD_CLIP:-${GRAD_CLIP:-}}"

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
echo "  ADAPT_SCOPE=${ADAPT_SCOPE}"
echo "  ADAPT_SOLVER=${ADAPT_SOLVER}"
echo "  SCHEDULE_LABEL=${SCHEDULE_LABEL}"
echo "  FREEZE_MONTHLY_GAIN=${FREEZE_MONTHLY_GAIN}"
echo "  RIDGE_LAMBDA=${RIDGE_LAMBDA} RIDGE_CLIP_COEFF_NORM=${RIDGE_CLIP_COEFF_NORM} RIDGE_TRUST_REGION_RADIUS=${RIDGE_TRUST_REGION_RADIUS} RIDGE_MAX_FEATURE_PIXELS=${RIDGE_MAX_FEATURE_PIXELS} RIDGE_STANDARDIZE_FEATURES=${RIDGE_STANDARDIZE_FEATURES}"
echo "  TRUST_REGION_MODE=${TRUST_REGION_MODE} TRUST_TOTAL_RADIUS=${TRUST_TOTAL_RADIUS} TRUST_PROMPT_RADIUS=${TRUST_PROMPT_RADIUS} TRUST_GAIN_RADIUS=${TRUST_GAIN_RADIUS} TRUST_COEFF_RADIUS=${TRUST_COEFF_RADIUS} TRUST_SPATIAL_RADIUS=${TRUST_SPATIAL_RADIUS}"
echo "  SUPPORT_LOSS_REDUCTION=${SUPPORT_LOSS_REDUCTION}"
echo "  ADAPT_MIX_RHO=${ADAPT_MIX_RHO}"
echo "  AUDIT_IDENTITY=${AUDIT_IDENTITY} tolerance=${AUDIT_IDENTITY_TOLERANCE}"
echo "  eval_split=target_eval"
echo "  eval_max_samples=${EVAL_MAX_SAMPLES}"
echo "  adapt_batch_size=${ADAPT_BATCH_SIZE}"
echo "  eval_batch_size=${EVAL_BATCH_SIZE}"
echo "  adapt_lr_override=${ADAPT_LR_OVERRIDE:-<none>}"
echo "  adapt_max_steps_override=${ADAPT_MAX_STEPS_OVERRIDE:-<none>}"
echo "  adapt_anchor_alpha_override=${ADAPT_ANCHOR_ALPHA_OVERRIDE:-<none>}"
echo "  splits_json=${SPLITS_JSON}"
echo "  output_base=${OUTPUT_BASE}"
echo "  device=gpu:${CUDA_VISIBLE_DEVICES}"
echo "============================================"

for K in ${K_LIST}; do
    ADAPT_SCOPE_FOR_K="${ADAPT_SCOPE}"
    ADAPT_SOLVER_FOR_K="${ADAPT_SOLVER}"
    AUDIT_IDENTITY_FOR_K="0"
    if [[ "${K}" == "0" ]]; then
        ADAPTATION_SETTING="zero_shot_context"
        ADAPTATION_MAX_STEPS="0"
        ADAPTATION_LR="${LR_K0:-${LR:-1e-3}}"
        ANCHOR_ALPHA="0.0"
        ADAPT_SOLVER_FOR_K="adamw"
    elif [[ "${K}" == "4" ]]; then
        ADAPTATION_SETTING="few_shot_k4"
        ADAPTATION_MAX_STEPS="${ADAPT_MAX_STEPS_OVERRIDE:-${MAX_STEPS_K4:-${MAX_STEPS:-100}}}"
        ADAPTATION_LR="${ADAPT_LR_OVERRIDE:-${LR_K4:-${LR:-1e-3}}}"
        ANCHOR_ALPHA="${ADAPT_ANCHOR_ALPHA_OVERRIDE:-${ANCHOR_ALPHA_K4}}"
    elif [[ "${K}" == "12" ]]; then
        ADAPTATION_SETTING="few_shot_k12"
        ADAPTATION_MAX_STEPS="${ADAPT_MAX_STEPS_OVERRIDE:-${MAX_STEPS_K12:-${MAX_STEPS:-80}}}"
        ADAPTATION_LR="${ADAPT_LR_OVERRIDE:-${LR_K12:-${LR:-3e-4}}}"
        ANCHOR_ALPHA="${ADAPT_ANCHOR_ALPHA_OVERRIDE:-${ANCHOR_ALPHA_K12}}"
    else
        echo "ERROR: K_LIST may contain only 0, 4, 12; got ${K}" >&2
        exit 2
    fi
    if [[ "${AUDIT_IDENTITY}" == "1" || "${AUDIT_IDENTITY,,}" == "true" ]]; then
        if [[ "${K}" == "12" ]]; then
            ADAPT_SCOPE_FOR_K="none"
            ADAPT_SOLVER_FOR_K="adamw"
            ADAPTATION_MAX_STEPS="0"
            ANCHOR_ALPHA="0.0"
            AUDIT_IDENTITY_FOR_K="1"
        fi
    fi

    K_DIR="${OUTPUT_BASE}/K${K}"
    ADAPT_DIR="${K_DIR}/adapt"
    EVAL_DIR="${K_DIR}/eval"
    mkdir -p "${ADAPT_DIR}" "${EVAL_DIR}"

    echo ""
    echo ">>> [K=${K}] Adaptation: ${ADAPTATION_SETTING}"
    ADAPT_RECIPE="${ADAPT_RECIPE}" ADAPT_SCOPE="${ADAPT_SCOPE_FOR_K}" ADAPT_SOLVER="${ADAPT_SOLVER_FOR_K}" SCHEDULE_LABEL="${SCHEDULE_LABEL}" FREEZE_MONTHLY_GAIN="${FREEZE_MONTHLY_GAIN}" RIDGE_LAMBDA="${RIDGE_LAMBDA}" RIDGE_CLIP_COEFF_NORM="${RIDGE_CLIP_COEFF_NORM}" RIDGE_TRUST_REGION_RADIUS="${RIDGE_TRUST_REGION_RADIUS}" RIDGE_MAX_FEATURE_PIXELS="${RIDGE_MAX_FEATURE_PIXELS}" RIDGE_STANDARDIZE_FEATURES="${RIDGE_STANDARDIZE_FEATURES}" TRUST_REGION_MODE="${TRUST_REGION_MODE}" TRUST_TOTAL_RADIUS="${TRUST_TOTAL_RADIUS}" TRUST_PROMPT_RADIUS="${TRUST_PROMPT_RADIUS}" TRUST_GAIN_RADIUS="${TRUST_GAIN_RADIUS}" TRUST_COEFF_RADIUS="${TRUST_COEFF_RADIUS}" TRUST_SPATIAL_RADIUS="${TRUST_SPATIAL_RADIUS}" SUPPORT_LOSS_REDUCTION="${SUPPORT_LOSS_REDUCTION}" AUDIT_IDENTITY="${AUDIT_IDENTITY_FOR_K}" AUDIT_IDENTITY_TOLERANCE="${AUDIT_IDENTITY_TOLERANCE}" ADAPT_ANCHOR_ALPHA="${ANCHOR_ALPHA}" ADAPT_LR="${ADAPTATION_LR}" ADAPT_MAX_STEPS="${ADAPTATION_MAX_STEPS}" ADAPT_WEIGHT_DECAY="${ADAPT_WEIGHT_DECAY_OVERRIDE}" ADAPT_GRAD_CLIP="${ADAPT_GRAD_CLIP_OVERRIDE}" ADAPT_LAMBDA_PRIOR="${ADAPT_LAMBDA_PRIOR:-${LAMBDA_PRIOR:-}}" ADAPT_LAMBDA_LATENT="${ADAPT_LAMBDA_LATENT:-${LAMBDA_LATENT:-}}" ADAPT_LAMBDA_GAIN="${ADAPT_LAMBDA_GAIN:-${LAMBDA_GAIN:-}}" ADAPT_LAMBDA_GAIN_SMOOTH="${ADAPT_LAMBDA_GAIN_SMOOTH:-${LAMBDA_GAIN_SMOOTH:-}}" ADAPT_LAMBDA_ANALYSIS="${ADAPT_LAMBDA_ANALYSIS:-${LAMBDA_ANALYSIS:-}}" BATCH_SIZE="${ADAPT_BATCH_SIZE}" SPLITS_JSON="${SPLITS_JSON}" OUTPUT_DIR="${ADAPT_DIR}" bash run/phase5_hyperda_zero_few_shot.sh \
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
        --splits_json "${SPLITS_JSON}" \
        --predictor_type hyperda_target_adapt \
        --device cuda \
        --output_dir "${EVAL_DIR}" \
        --max_samples "${EVAL_MAX_SAMPLES}" \
        --batch_size "${EVAL_BATCH_SIZE}" \
        --adapt_mix_rho "${ADAPT_MIX_RHO}" \
        2>&1 | tee "${EVAL_DIR}/eval.log"
done

compare_identity_audit() {
python3 - "${OUTPUT_BASE}" "${TARGET_REGION}" "${AUDIT_IDENTITY_TOLERANCE}" <<'PYTHON_SCRIPT'
import json
import math
import sys
from pathlib import Path

output_base = Path(sys.argv[1])
target_region = sys.argv[2]
tolerance = float(sys.argv[3])
audit_path = output_base / "identity_audit.json"

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def nested_get(mapping, path):
    value = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value

k0_summary_path = output_base / "K0" / "eval" / target_region / "summary.json"
k12_summary_path = output_base / "K12" / "eval" / target_region / "summary.json"
k0_meta_path = output_base / "K0" / "adapt" / "metadata.json"
k12_meta_path = output_base / "K12" / "adapt" / "metadata.json"
required = [k0_summary_path, k12_summary_path, k0_meta_path, k12_meta_path]
missing = [str(path) for path in required if not path.exists()]
if missing:
    result = {
        "status": "failed",
        "reason": "missing_required_artifact",
        "missing": missing,
        "tolerance": tolerance,
    }
    audit_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(f"identity audit missing artifacts: {missing}")

k0_summary = load_json(k0_summary_path)
k12_summary = load_json(k12_summary_path)
k0_meta = load_json(k0_meta_path)
k12_meta = load_json(k12_meta_path)

metric_paths = [
    ("surface_skill_primary", ("surface", "skill_primary")),
    ("surface_skill_latw_primary", ("surface", "skill_latw_primary")),
    ("surface_rmse_latw_mean", ("surface", "rmse_latw_mean")),
    ("surface_corr_latw_mean", ("surface", "corr_latw_mean")),
    ("rootzone_skill_primary", ("rootzone", "skill_primary")),
    ("rootzone_skill_latw_primary", ("rootzone", "skill_latw_primary")),
    ("rootzone_rmse_latw_mean", ("rootzone", "rmse_latw_mean")),
    ("rootzone_corr_latw_mean", ("rootzone", "corr_latw_mean")),
    ("prediction_content_hash", ("prediction_content_hash",)),
    ("metric_values_content_hash", ("metric_values_content_hash",)),
]
metric_diffs = {}
max_abs_diff = 0.0
failures = []
for name, path in metric_paths:
    v0 = nested_get(k0_summary, path)
    v12 = nested_get(k12_summary, path)
    if v0 is None or v12 is None:
        failures.append(f"missing_metric:{name}")
        metric_diffs[name] = {"K0": v0, "K12": v12, "abs_diff": None}
        continue
    if name.endswith("_hash"):
        match = str(v12) == str(v0)
        metric_diffs[name] = {"K0": str(v0), "K12": str(v12), "match": match}
        if not match:
            failures.append(f"hash_diff:{name}")
        continue
    diff = abs(float(v12) - float(v0))
    if math.isfinite(diff):
        max_abs_diff = max(max_abs_diff, diff)
    metric_diffs[name] = {"K0": float(v0), "K12": float(v12), "abs_diff": diff}
    if diff > tolerance:
        failures.append(f"metric_diff:{name}:{diff:.12g}")

hash_checks = [
    "target_context_dates_hash",
    "target_eval_dates_hash",
    "split_manifest_sha256",
]
hash_matches = {}
for key in hash_checks:
    left = k0_summary.get(key) or k0_meta.get(key)
    right = k12_summary.get(key) or k12_meta.get(key)
    hash_matches[key] = {"K0": left, "K12": right, "match": left == right}
    if left != right:
        failures.append(f"hash_mismatch:{key}")

source_left = k0_meta.get("source_checkpoint_sha256")
source_right = k12_meta.get("source_checkpoint_sha256")
hash_matches["source_checkpoint_sha256"] = {
    "K0": source_left,
    "K12": source_right,
    "match": source_left == source_right,
}
if source_left != source_right:
    failures.append("hash_mismatch:source_checkpoint_sha256")

if k12_meta.get("adapt_scope") != "none":
    failures.append("K12_adapt_scope_not_none")
if int(k12_meta.get("adaptation_steps") or 0) != 0:
    failures.append("K12_adaptation_steps_not_zero")
if abs(float(k12_meta.get("anchor_alpha") or 0.0)) > 1e-12:
    failures.append("K12_anchor_alpha_not_zero")
if bool(k12_meta.get("target_labels_used_for_adaptation")):
    failures.append("K12_target_labels_used")
if bool(k12_meta.get("target_labels_loaded_for_adaptation")):
    failures.append("K12_target_labels_loaded")
if float((k12_meta.get("target_parameter_l2_drift") or {}).get("total") or 0.0) > tolerance:
    failures.append("K12_parameter_drift_nonzero")

result = {
    "status": "passed" if not failures else "failed",
    "tolerance": tolerance,
    "max_abs_metric_diff": max_abs_diff,
    "metric_diffs": metric_diffs,
    "hash_matches": hash_matches,
    "K0_metadata": str(k0_meta_path),
    "K12_metadata": str(k12_meta_path),
    "K0_summary": str(k0_summary_path),
    "K12_summary": str(k12_summary_path),
    "failures": failures,
}
audit_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
if failures:
    raise SystemExit(f"identity audit failed; see {audit_path}")
print(f"Identity audit passed: max_abs_metric_diff={max_abs_diff:.12g} tolerance={tolerance}")
print(f"Identity audit JSON: {audit_path}")
PYTHON_SCRIPT
}

if [[ "${AUDIT_IDENTITY}" == "1" || "${AUDIT_IDENTITY,,}" == "true" ]]; then
    compare_identity_audit
fi

python3 - "${OUTPUT_BASE}" "${TARGET_REGION}" "${K_LIST}" "${SEED}" <<'PYTHON_SCRIPT'
import csv
import json
import math
import sys
from pathlib import Path

output_base = Path(sys.argv[1])
target_region = sys.argv[2]
k_values = sys.argv[3].split()
requested_seed = int(sys.argv[4])
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

def file_sha256(path):
    import hashlib
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def overall_skill(summary):
    surface = (summary.get("surface", {}) or {}).get("skill_primary")
    rootzone = (summary.get("rootzone", {}) or {}).get("skill_primary")
    if surface is None or rootzone is None:
        return None
    return (float(surface) + float(rootzone)) / 2.0

def latest_source_only_summary(target_region, seed):
    preferred = sorted(Path("artifacts/runs/phase4_source_only").glob(
        f"phase4_source_only_source_only_{target_region}_w32_e50_lr0.0003_norm_zero_s{seed}_*/"
        f"results/checkpoint_best_source_val_safe_score/target_eval/{target_region}/summary.json"
    ))
    candidates = preferred or sorted(Path("artifacts/runs/phase4_source_only").glob(
        f"phase4_source_only_source_only_{target_region}_w32_e50_lr0.0003_norm_zero_s{seed}_*/"
        f"results/checkpoint_*/target_eval/{target_region}/summary.json"
    ))
    if not candidates:
        return None, None
    path = candidates[-1]
    with open(path, encoding="utf-8") as f:
        return json.load(f), str(path)

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

def records_hash(records):
    import hashlib
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def load_split_entry_for_seed(k, seed):
    split_path = Path("artifacts/splits/US_loro_zero_few_shot_splits.json")
    if not split_path.exists():
        return {}
    with open(split_path, encoding="utf-8") as f:
        data = json.load(f)
    setting = settings.get(str(k), "")
    for entry in data.get("splits", []):
        if entry.get("target_region_id") != target_region:
            continue
        try:
            entry_seed = int(entry.get("seed", -1))
            wanted_seed = int(seed)
        except Exception:
            entry_seed = str(entry.get("seed", ""))
            wanted_seed = str(seed)
        if entry_seed != wanted_seed:
            continue
        if entry.get("adaptation_setting") != setting:
            continue
        return entry
    return {}

def date_strs(records):
    return [str(record.get("date_str", "")) for record in records if isinstance(record, dict) and record.get("date_str")]

def support_subset(left_dates, right_dates):
    return set(left_dates) <= set(right_dates)

def write_nested_support_diagnostic(seed):
    split_path = Path("artifacts/splits/US_loro_zero_few_shot_splits.json")
    k4_entry = load_split_entry_for_seed(4, seed)
    k12_entry = load_split_entry_for_seed(12, seed)
    k4_records = list(k4_entry.get("target_support_dates", []) or [])
    k12_records = list(k12_entry.get("target_support_dates", []) or [])
    k4_dates = date_strs(k4_records)
    k12_dates = date_strs(k12_records)
    subset = support_subset(k4_dates, k12_dates)
    result = {
        "target_region": target_region,
        "seed": seed,
        "k4_support_dates": k4_dates,
        "k12_support_dates": k12_dates,
        "k4_support_dates_hash": k4_entry.get("target_support_dates_hash", records_hash(k4_records)),
        "k12_support_dates_hash": k12_entry.get("target_support_dates_hash", records_hash(k12_records)),
        "k4_support_subset_of_k12": subset,
        "nested_manifest_path": "",
        "nested_command_path": "",
        "note": "current K4 support is nested in current K12 support" if subset else "current K4 support is not nested in current K12 support",
    }
    if not subset and k4_records and k12_records:
        seen = {(record.get("time_index"), record.get("datetime_str", record.get("date_str"))) for record in k4_records}
        additional = []
        for record in k12_records:
            key = (record.get("time_index"), record.get("datetime_str", record.get("date_str")))
            if key in seen:
                continue
            additional.append(record)
            seen.add(key)
            if len(additional) >= 8:
                break
        nested_records = k4_records + additional
        nested_dir = output_base / "nested_support"
        nested_dir.mkdir(parents=True, exist_ok=True)
        nested_support_path = nested_dir / f"{target_region}_s{seed}_K12_nested_support.json"
        nested_payload = {
            "schema_version": "hyperda_k12_nested_support_diagnostic_v1",
            "target_region": target_region,
            "seed": seed,
            "source": "derived_from_frozen_v4_4_manifest_without_target_eval",
            "k4_support_dates": k4_records,
            "k12_current_support_dates": k12_records,
            "k12_nested_support_dates": nested_records,
            "k12_nested_support_dates_hash": records_hash(nested_records),
            "selection_rule": "K4 support dates plus first 8 current K12 support dates not already in K4 order",
            "leakage_note": "uses only frozen target_support records from 2015-2021; target_eval labels/features are not used",
        }
        nested_support_path.write_text(json.dumps(nested_payload, indent=2) + "\n", encoding="utf-8")
        nested_split_path = nested_dir / f"{target_region}_s{seed}_K12_nested_splits.json"
        with open(split_path, encoding="utf-8") as f:
            split_data = json.load(f)
        nested_split_data = json.loads(json.dumps(split_data))
        for entry in nested_split_data.get("splits", []):
            if entry.get("target_region_id") != target_region:
                continue
            if int(entry.get("seed", -1)) != int(seed):
                continue
            if entry.get("adaptation_setting") != "few_shot_k12":
                continue
            support_hash = records_hash(nested_records)
            entry["target_support_dates"] = nested_records
            entry["target_train_dates"] = nested_records
            entry["target_adaptation_dates"] = nested_records
            entry["target_support_cycle_count"] = len(nested_records)
            entry["target_train_cycle_count"] = len(nested_records)
            entry["target_adaptation_cycle_count"] = len(nested_records)
            entry["target_support_dates_hash"] = support_hash
            entry["target_train_dates_hash"] = support_hash
            entry["target_adaptation_dates_hash"] = support_hash
            entry["support_dates_hash"] = support_hash
            entry["diagnostic_split_note"] = "K12 nested diagnostic: K4 support dates plus 8 safe K12 support dates"
            break
        nested_split_path.write_text(json.dumps(nested_split_data, indent=2) + "\n", encoding="utf-8")
        command_path = nested_dir / "commands.md"
        command_path.write_text(
            "# K12 Nested Support Diagnostic\n\n"
            "This command uses a run-local split manifest derived from the frozen V4.4 manifest. "
            "It does not modify the frozen split artifact and does not use target_eval for support selection.\n\n"
            "```bash\n"
            f"K_LIST=\"12\" ADAPT_SCOPE=all SPLITS_JSON=\"{nested_split_path}\" "
            f"ADAPT_BATCH_SIZE=\"${{ADAPT_BATCH_SIZE:-8}}\" EVAL_BATCH_SIZE=\"${{EVAL_BATCH_SIZE:-8}}\" "
            f"bash run/phase5_hyperda_zero_few_shot_eval.sh \"$SOURCE_CHECKPOINT\" {target_region} {seed} "
            f"\"${{CUDA_VISIBLE_DEVICES:-1}}\" \"{output_base}/K12_nested\"\n"
            "```\n\n"
            f"- Nested support artifact: `{nested_support_path}`\n"
            f"- Nested split manifest: `{nested_split_path}`\n"
            f"- Nested support hash: `{nested_payload['k12_nested_support_dates_hash']}`\n",
            encoding="utf-8",
        )
        result["nested_manifest_path"] = str(nested_split_path)
        result["nested_support_artifact_path"] = str(nested_support_path)
        result["nested_command_path"] = str(command_path)
        result["k12_nested_support_dates"] = date_strs(nested_records)
        result["k12_nested_support_dates_hash"] = nested_payload["k12_nested_support_dates_hash"]
    support_diag_path = output_base / "support_set_diagnostic.json"
    support_diag_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result

support_diagnostic = write_nested_support_diagnostic(requested_seed)

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
    row_seed = summary.get("seed", adapt_metadata.get("seed"))
    if row_seed is None:
        row_seed = ""
    source_only_summary, source_only_summary_path = latest_source_only_summary(target_region, row_seed)
    source_only_surface = (source_only_summary.get("surface", {}) or {}).get("skill_primary") if source_only_summary else None
    source_only_rootzone = (source_only_summary.get("rootzone", {}) or {}).get("skill_primary") if source_only_summary else None
    source_only_overall = overall_skill(source_only_summary) if source_only_summary else None
    drift = adapt_metadata.get("target_parameter_l2_drift", {}) or {}
    pre_anchor_drift = adapt_metadata.get("target_parameter_l2_drift_pre_anchor", {}) or {}
    post_anchor_drift = adapt_metadata.get("target_parameter_l2_drift_post_anchor", drift) or {}
    trust_diag = adapt_metadata.get("trust_projection_diagnostics", {}) or {}
    row = {
        "target_region": target_region,
        "K": k,
        "adaptation_setting": settings.get(k, ""),
        "adapt_recipe": adapt_metadata.get("adapt_recipe"),
        "adapt_scope": adapt_metadata.get("adapt_scope"),
        "adapt_solver": adapt_metadata.get("adapt_solver"),
        "trust_region_mode": adapt_metadata.get("trust_region_mode"),
        "trust_total_radius": adapt_metadata.get("trust_total_radius"),
        "trust_prompt_radius": adapt_metadata.get("trust_prompt_radius"),
        "trust_gain_radius": adapt_metadata.get("trust_gain_radius"),
        "trust_coeff_radius": adapt_metadata.get("trust_coeff_radius"),
        "trust_spatial_radius": adapt_metadata.get("trust_spatial_radius"),
        "support_loss_reduction": adapt_metadata.get("support_loss_reduction"),
        "adapt_mix_rho": summary.get("adapt_mix_rho"),
        "freeze_monthly_gain": adapt_metadata.get("freeze_monthly_gain"),
        "audit_identity": adapt_metadata.get("audit_identity"),
        "status": status,
        "method": summary.get("method"),
        "split_type": summary.get("split_type"),
        "split_file": summary.get("split_file"),
        "seed": row_seed,
        "adaptation_steps": adapt_metadata.get("adaptation_steps"),
        "schedule_label": adapt_metadata.get("schedule_label"),
        "max_steps_requested": first_present(adapt_metadata, ["max_steps_requested", "adaptation_steps"]),
        "requested_lr": adapt_metadata.get("requested_lr"),
        "requested_max_steps": adapt_metadata.get("requested_max_steps"),
        "requested_anchor_alpha": adapt_metadata.get("requested_anchor_alpha"),
        "requested_weight_decay": adapt_metadata.get("requested_weight_decay"),
        "requested_grad_clip": adapt_metadata.get("requested_grad_clip"),
        "actual_optimizer_steps": first_present(adapt_metadata, ["actual_optimizer_steps", "optimizer_steps_run"]),
        "optimizer_steps_run": adapt_metadata.get("optimizer_steps_run"),
        "support_batch_count": adapt_metadata.get("support_batch_count"),
        "effective_support_passes": adapt_metadata.get("effective_support_passes"),
        "adapt_batch_size": adapt_metadata.get("adapt_batch_size"),
        "eval_batch_size": summary.get("eval_batch_size"),
        "lr": adapt_metadata.get("lr"),
        "weight_decay": adapt_metadata.get("weight_decay"),
        "grad_clip": adapt_metadata.get("grad_clip"),
        "anchor_alpha": adapt_metadata.get("anchor_alpha"),
        "lambda_prior": adapt_metadata.get("lambda_prior"),
        "lambda_latent": adapt_metadata.get("lambda_latent"),
        "lambda_gain": adapt_metadata.get("lambda_gain"),
        "lambda_gain_smooth": adapt_metadata.get("lambda_gain_smooth"),
        "lambda_analysis": adapt_metadata.get("lambda_analysis"),
        "source_checkpoint_sha256": adapt_metadata.get("source_checkpoint_sha256"),
        "adapted_checkpoint_sha256": file_sha256(checkpoint_path),
        "source_anchor_hyperparameter_source": adapt_metadata.get("source_anchor_hyperparameter_source"),
        "trainable_parameter_count": adapt_metadata.get("trainable_parameter_count"),
        "requires_grad_parameter_count": first_present(adapt_metadata, ["requires_grad_parameter_count", "requires_grad_param_count", "trainable_parameter_count"]),
        "requires_grad_param_count": first_present(adapt_metadata, ["requires_grad_param_count", "requires_grad_parameter_count", "trainable_parameter_count"]),
        "optimizer_parameter_count": first_present(adapt_metadata, ["optimizer_parameter_count", "optimizer_param_count"]),
        "optimizer_param_count": first_present(adapt_metadata, ["optimizer_param_count", "optimizer_parameter_count"]),
        "target_parameter_count_by_group": json.dumps(adapt_metadata.get("target_parameter_count_by_group", {}) or {}, sort_keys=True),
        "source_only_summary": source_only_summary_path,
        "source_only_surface_skill_primary": source_only_surface,
        "source_only_rootzone_skill_primary": source_only_rootzone,
        "source_only_overall_skill_primary": source_only_overall,
        "support_loss_before": adapt_metadata.get("support_loss_before"),
        "support_loss_after": adapt_metadata.get("support_loss_after"),
        "support_final_loss": adapt_metadata.get("support_final_loss"),
        "support_loss_delta": adapt_metadata.get("support_loss_delta"),
        "standard_support_loss_before_full_support": adapt_metadata.get("standard_support_loss_before_full_support"),
        "standard_support_loss_after_full_support": adapt_metadata.get("standard_support_loss_after_full_support"),
        "standard_support_loss_delta_full_support": adapt_metadata.get("standard_support_loss_delta_full_support"),
        "standard_support_objective_before_full_support": adapt_metadata.get("standard_support_objective_before_full_support"),
        "standard_support_objective_after_full_support": adapt_metadata.get("standard_support_objective_after_full_support"),
        "standard_support_increment_loss_before_full_support": adapt_metadata.get("standard_support_increment_loss_before_full_support"),
        "standard_support_increment_loss_after_full_support": adapt_metadata.get("standard_support_increment_loss_after_full_support"),
        "standard_support_analysis_loss_before_full_support": adapt_metadata.get("standard_support_analysis_loss_before_full_support"),
        "standard_support_analysis_loss_after_full_support": adapt_metadata.get("standard_support_analysis_loss_after_full_support"),
        "standard_support_regularization_loss_before_full_support": adapt_metadata.get("standard_support_regularization_loss_before_full_support"),
        "standard_support_regularization_loss_after_full_support": adapt_metadata.get("standard_support_regularization_loss_after_full_support"),
        "support_gradient_cosine_mean": adapt_metadata.get("support_gradient_cosine_mean"),
        "support_gradient_cosine_min": adapt_metadata.get("support_gradient_cosine_min"),
        "support_gradient_negative_fraction": adapt_metadata.get("support_gradient_negative_fraction"),
        "support_cycle_loss_improvement_mean": adapt_metadata.get("support_cycle_loss_improvement_mean"),
        "support_cycle_loss_improvement_std": adapt_metadata.get("support_cycle_loss_improvement_std"),
        "trust_projection_applied_count": trust_diag.get("trust_projection_applied_count"),
        "trust_projection_pre_step_drift_max_total": trust_diag.get("trust_projection_pre_step_drift_max_total"),
        "trust_projection_post_step_drift_max_total": trust_diag.get("trust_projection_post_step_drift_max_total"),
        "trust_projection_pre_step_drift_last_total": trust_diag.get("trust_projection_pre_step_drift_last_total"),
        "trust_projection_post_step_drift_last_total": trust_diag.get("trust_projection_post_step_drift_last_total"),
        "ridge_design_loss_before_sampled_pixels": adapt_metadata.get("ridge_design_loss_before_sampled_pixels"),
        "ridge_design_loss_after_sampled_pixels": adapt_metadata.get("ridge_design_loss_after_sampled_pixels"),
        "ridge_design_loss_delta_sampled_pixels": adapt_metadata.get("ridge_design_loss_delta_sampled_pixels"),
        "ridge_lambda": adapt_metadata.get("ridge_lambda"),
        "ridge_clip_coeff_norm": adapt_metadata.get("ridge_clip_coeff_norm"),
        "ridge_trust_region_radius": adapt_metadata.get("ridge_trust_region_radius"),
        "ridge_max_feature_pixels": adapt_metadata.get("ridge_max_feature_pixels"),
        "ridge_standardize_features": adapt_metadata.get("ridge_standardize_features"),
        "ridge_status": adapt_metadata.get("ridge_status"),
        "ridge_coefficient_norm": adapt_metadata.get("ridge_coefficient_norm"),
        "ridge_coeff_norm": adapt_metadata.get("ridge_coeff_norm"),
        "ridge_delta_norm": adapt_metadata.get("ridge_delta_norm"),
        "ridge_coeff_delta_norm": adapt_metadata.get("ridge_coeff_delta_norm"),
        "ridge_raw_delta_norm": adapt_metadata.get("ridge_raw_delta_norm"),
        "ridge_clip_applied": adapt_metadata.get("ridge_clip_applied"),
        "ridge_trust_region_clipped": adapt_metadata.get("ridge_trust_region_clipped"),
        "ridge_support_count": adapt_metadata.get("ridge_support_count"),
        "ridge_masked_pixel_count": adapt_metadata.get("ridge_masked_pixel_count"),
        "ridge_masked_observation_count": adapt_metadata.get("ridge_masked_observation_count"),
        "ridge_feature_pixel_count": adapt_metadata.get("ridge_feature_pixel_count"),
        "ridge_feature_observation_count": adapt_metadata.get("ridge_feature_observation_count"),
        "ridge_feature_dim": adapt_metadata.get("ridge_feature_dim"),
        "ridge_condition_number": adapt_metadata.get("ridge_condition_number"),
        "ridge_rank": adapt_metadata.get("ridge_rank"),
        "target_parameter_l2_drift_total": drift.get("total"),
        "target_parameter_l2_drift_target_prompt": drift.get("target_prompt"),
        "target_parameter_l2_drift_adapter_coeff_bottleneck": drift.get("adapter_coeff_bottleneck"),
        "target_parameter_l2_drift_adapter_coeff_dec2": drift.get("adapter_coeff_dec2"),
        "target_parameter_l2_drift_adapter_coeff_dec1": drift.get("adapter_coeff_dec1"),
        "target_parameter_l2_drift_monthly_gain": first_present(drift, ["monthly_gain", "monthly_residual_gain"]),
        "target_parameter_l2_drift_spatial_refine": first_present(drift, ["spatial_refine", "target_spatial_refine"]),
        "target_parameter_l2_drift_other": first_present(drift, ["other_target_specific", "other_target_parameters"]),
        "target_parameter_l2_drift_pre_anchor_total": pre_anchor_drift.get("total"),
        "target_parameter_l2_drift_pre_anchor_target_prompt": pre_anchor_drift.get("target_prompt"),
        "target_parameter_l2_drift_pre_anchor_adapter_coeff_bottleneck": pre_anchor_drift.get("adapter_coeff_bottleneck"),
        "target_parameter_l2_drift_pre_anchor_adapter_coeff_dec2": pre_anchor_drift.get("adapter_coeff_dec2"),
        "target_parameter_l2_drift_pre_anchor_adapter_coeff_dec1": pre_anchor_drift.get("adapter_coeff_dec1"),
        "target_parameter_l2_drift_pre_anchor_monthly_gain": first_present(pre_anchor_drift, ["monthly_gain", "monthly_residual_gain"]),
        "target_parameter_l2_drift_post_anchor_total": post_anchor_drift.get("total"),
        "target_parameter_l2_drift_post_anchor_target_prompt": post_anchor_drift.get("target_prompt"),
        "target_parameter_l2_drift_post_anchor_adapter_coeff_bottleneck": post_anchor_drift.get("adapter_coeff_bottleneck"),
        "target_parameter_l2_drift_post_anchor_adapter_coeff_dec2": post_anchor_drift.get("adapter_coeff_dec2"),
        "target_parameter_l2_drift_post_anchor_adapter_coeff_dec1": post_anchor_drift.get("adapter_coeff_dec1"),
        "target_parameter_l2_drift_post_anchor_monthly_gain": first_present(post_anchor_drift, ["monthly_gain", "monthly_residual_gain"]),
        "target_support_count": adapt_metadata.get("target_support_count"),
        "target_labels_loaded_for_adaptation": adapt_metadata.get("target_labels_loaded_for_adaptation"),
        "target_labels_used_for_adaptation": adapt_metadata.get("target_labels_used_for_adaptation"),
        "target_support_dates": ",".join(adapt_metadata.get("target_support_dates", []) or []),
        "support_dates": ",".join(adapt_metadata.get("target_support_dates", []) or []),
        "support_dates_hash": first_present(summary, ["support_dates_hash", "target_support_dates_hash"]),
        "k4_support_subset_of_k12": support_diagnostic.get("k4_support_subset_of_k12"),
        "k4_support_dates": ",".join(support_diagnostic.get("k4_support_dates", []) or []),
        "k12_support_dates": ",".join(support_diagnostic.get("k12_support_dates", []) or []),
        "k12_nested_support_dates": ",".join(support_diagnostic.get("k12_nested_support_dates", []) or []),
        "k12_nested_support_dates_hash": support_diagnostic.get("k12_nested_support_dates_hash"),
        "nested_support_manifest": support_diagnostic.get("nested_manifest_path"),
        "nested_support_command": support_diagnostic.get("nested_command_path"),
        "n_samples_evaluated": summary.get("n_samples_evaluated"),
        "n_metric_rows": summary.get("n_metric_rows"),
        "prediction_content_hash": summary.get("prediction_content_hash"),
        "zero_shot_prediction_content_hash": summary.get("zero_shot_prediction_content_hash"),
        "adapted_pre_mix_prediction_content_hash": summary.get("adapted_pre_mix_prediction_content_hash"),
        "final_mixed_prediction_content_hash": summary.get("final_mixed_prediction_content_hash"),
        "mix_mean_abs_change_from_k0": summary.get("mix_mean_abs_change_from_k0"),
        "mix_max_abs_change_from_k0": summary.get("mix_max_abs_change_from_k0"),
        "mix_mean_abs_change_from_adapted": summary.get("mix_mean_abs_change_from_adapted"),
        "mix_max_abs_change_from_adapted": summary.get("mix_max_abs_change_from_adapted"),
        "prediction_record_count": summary.get("prediction_record_count"),
        "metric_content_hash": summary.get("metric_content_hash"),
        "metric_row_content_hash": summary.get("metric_row_content_hash"),
        "metric_values_content_hash": summary.get("metric_values_content_hash"),
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
    "adapt_scope",
    "adapt_solver",
    "trust_region_mode",
    "trust_total_radius",
    "trust_prompt_radius",
    "trust_gain_radius",
    "trust_coeff_radius",
    "trust_spatial_radius",
    "support_loss_reduction",
    "adapt_mix_rho",
    "freeze_monthly_gain",
    "audit_identity",
    "status",
    "method",
    "split_type",
    "split_file",
    "seed",
    "adaptation_steps",
    "schedule_label",
    "max_steps_requested",
    "requested_lr",
    "requested_max_steps",
    "requested_anchor_alpha",
    "requested_weight_decay",
    "requested_grad_clip",
    "actual_optimizer_steps",
    "optimizer_steps_run",
    "support_batch_count",
    "effective_support_passes",
    "adapt_batch_size",
    "eval_batch_size",
    "lr",
    "weight_decay",
    "grad_clip",
    "anchor_alpha",
    "lambda_prior",
    "lambda_latent",
    "lambda_gain",
    "lambda_gain_smooth",
    "lambda_analysis",
    "source_checkpoint_sha256",
    "adapted_checkpoint_sha256",
    "source_anchor_hyperparameter_source",
    "trainable_parameter_count",
    "requires_grad_parameter_count",
    "requires_grad_param_count",
    "optimizer_parameter_count",
    "optimizer_param_count",
    "target_parameter_count_by_group",
    "source_only_summary",
    "source_only_surface_skill_primary",
    "source_only_rootzone_skill_primary",
    "source_only_overall_skill_primary",
    "support_loss_before",
    "support_loss_after",
    "support_final_loss",
    "support_loss_delta",
    "standard_support_loss_before_full_support",
    "standard_support_loss_after_full_support",
    "standard_support_loss_delta_full_support",
    "standard_support_objective_before_full_support",
    "standard_support_objective_after_full_support",
    "standard_support_increment_loss_before_full_support",
    "standard_support_increment_loss_after_full_support",
    "standard_support_analysis_loss_before_full_support",
    "standard_support_analysis_loss_after_full_support",
    "standard_support_regularization_loss_before_full_support",
    "standard_support_regularization_loss_after_full_support",
    "support_gradient_cosine_mean",
    "support_gradient_cosine_min",
    "support_gradient_negative_fraction",
    "support_cycle_loss_improvement_mean",
    "support_cycle_loss_improvement_std",
    "trust_projection_applied_count",
    "trust_projection_pre_step_drift_max_total",
    "trust_projection_post_step_drift_max_total",
    "trust_projection_pre_step_drift_last_total",
    "trust_projection_post_step_drift_last_total",
    "ridge_design_loss_before_sampled_pixels",
    "ridge_design_loss_after_sampled_pixels",
    "ridge_design_loss_delta_sampled_pixels",
    "ridge_lambda",
    "ridge_clip_coeff_norm",
    "ridge_trust_region_radius",
    "ridge_max_feature_pixels",
    "ridge_standardize_features",
    "ridge_status",
    "ridge_coefficient_norm",
    "ridge_coeff_norm",
    "ridge_delta_norm",
    "ridge_coeff_delta_norm",
    "ridge_raw_delta_norm",
    "ridge_clip_applied",
    "ridge_trust_region_clipped",
    "ridge_support_count",
    "ridge_masked_pixel_count",
    "ridge_masked_observation_count",
    "ridge_feature_pixel_count",
    "ridge_feature_observation_count",
    "ridge_feature_dim",
    "ridge_condition_number",
    "ridge_rank",
    "target_parameter_l2_drift_total",
    "target_parameter_l2_drift_target_prompt",
    "target_parameter_l2_drift_adapter_coeff_bottleneck",
    "target_parameter_l2_drift_adapter_coeff_dec2",
    "target_parameter_l2_drift_adapter_coeff_dec1",
    "target_parameter_l2_drift_monthly_gain",
    "target_parameter_l2_drift_spatial_refine",
    "target_parameter_l2_drift_other",
    "target_parameter_l2_drift_pre_anchor_total",
    "target_parameter_l2_drift_pre_anchor_target_prompt",
    "target_parameter_l2_drift_pre_anchor_adapter_coeff_bottleneck",
    "target_parameter_l2_drift_pre_anchor_adapter_coeff_dec2",
    "target_parameter_l2_drift_pre_anchor_adapter_coeff_dec1",
    "target_parameter_l2_drift_pre_anchor_monthly_gain",
    "target_parameter_l2_drift_post_anchor_total",
    "target_parameter_l2_drift_post_anchor_target_prompt",
    "target_parameter_l2_drift_post_anchor_adapter_coeff_bottleneck",
    "target_parameter_l2_drift_post_anchor_adapter_coeff_dec2",
    "target_parameter_l2_drift_post_anchor_adapter_coeff_dec1",
    "target_parameter_l2_drift_post_anchor_monthly_gain",
    "target_support_count",
    "target_labels_loaded_for_adaptation",
    "target_labels_used_for_adaptation",
    "target_support_dates",
    "support_dates",
    "support_dates_hash",
    "k4_support_subset_of_k12",
    "k4_support_dates",
    "k12_support_dates",
    "k12_nested_support_dates",
    "k12_nested_support_dates_hash",
    "nested_support_manifest",
    "nested_support_command",
    "n_samples_evaluated",
    "n_metric_rows",
    "prediction_content_hash",
    "zero_shot_prediction_content_hash",
    "adapted_pre_mix_prediction_content_hash",
    "final_mixed_prediction_content_hash",
    "mix_mean_abs_change_from_k0",
    "mix_max_abs_change_from_k0",
    "mix_mean_abs_change_from_adapted",
    "mix_max_abs_change_from_adapted",
    "prediction_record_count",
    "metric_content_hash",
    "metric_row_content_hash",
    "metric_values_content_hash",
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
    "adapt_scope",
    "adapt_solver",
    "freeze_monthly_gain",
    "status",
    "schedule_label",
    "trust_region_mode",
    "adapt_mix_rho",
    "support_loss_reduction",
    "adaptation_steps",
    "max_steps_requested",
    "requested_lr",
    "requested_anchor_alpha",
    "actual_optimizer_steps",
    "optimizer_steps_run",
    "support_batch_count",
    "effective_support_passes",
    "adapt_batch_size",
    "eval_batch_size",
    "lr",
    "weight_decay",
    "grad_clip",
    "anchor_alpha",
    "trainable_parameter_count",
    "requires_grad_param_count",
    "optimizer_param_count",
    "support_loss_before",
    "support_loss_after",
    "standard_support_loss_before_full_support",
    "standard_support_loss_after_full_support",
    "support_gradient_cosine_mean",
    "support_gradient_cosine_min",
    "support_gradient_negative_fraction",
    "support_cycle_loss_improvement_mean",
    "support_cycle_loss_improvement_std",
    "trust_projection_post_step_drift_max_total",
    "support_final_loss",
    "support_loss_delta",
    "k4_support_subset_of_k12",
    "ridge_status",
    "ridge_coefficient_norm",
    "ridge_feature_observation_count",
    "ridge_masked_observation_count",
    "target_parameter_l2_drift_total",
    "target_parameter_l2_drift_pre_anchor_total",
    "target_parameter_l2_drift_post_anchor_total",
    "target_parameter_l2_drift_pre_anchor_target_prompt",
    "target_parameter_l2_drift_post_anchor_target_prompt",
    "target_parameter_l2_drift_monthly_gain",
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
    "prediction_content_hash",
    "zero_shot_prediction_content_hash",
    "adapted_pre_mix_prediction_content_hash",
    "final_mixed_prediction_content_hash",
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
