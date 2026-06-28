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
STAGE3_KSHOT_MODE="${STAGE3_KSHOT_MODE:-paper_safe}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
EVAL_OUTPUT_LEVEL="${EVAL_OUTPUT_LEVEL:-compact}"
REQUIRE_SOURCE_GATE_JSON_FOR_TARGET_EVAL="${REQUIRE_SOURCE_GATE_JSON_FOR_TARGET_EVAL:-0}"
SOURCE_GATE_JSON="${SOURCE_GATE_JSON:-${M3_13_SOURCE_GATE_JSON:-}}"
ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-${BATCH_SIZE:-8}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE:-8}}"
SPLITS_JSON="${SPLITS_JSON:-artifacts/splits/US_loro_zero_few_shot_splits.json}"
ADAPT_RECIPE="${ADAPT_RECIPE:-source_anchor}"
ADAPT_SCOPE_WAS_SET="${ADAPT_SCOPE+x}"
STAGE3_POSTERIOR_POLICY_WAS_SET="${STAGE3_POSTERIOR_POLICY+x}"
SUPPORT_GATE_WAS_SET="${SUPPORT_GATE+x}"
SUPPORT_GATE_MIN_DELTA_WAS_SET="${SUPPORT_GATE_MIN_DELTA+x}"
SUPPORT_GATE_ROOTZONE_TOLERANCE_WAS_SET="${SUPPORT_GATE_ROOTZONE_TOLERANCE+x}"
SUPPORT_LOSS_REDUCTION_WAS_SET="${SUPPORT_LOSS_REDUCTION+x}"
FREEZE_MONTHLY_GAIN_WAS_SET="${FREEZE_MONTHLY_GAIN+x}"
ADAPT_SOLVER_WAS_SET="${ADAPT_SOLVER+x}"
RIDGE_LAMBDA_WAS_SET="${RIDGE_LAMBDA+x}"
RIDGE_CLIP_COEFF_NORM_WAS_SET="${RIDGE_CLIP_COEFF_NORM+x}"
RIDGE_TRUST_REGION_RADIUS_WAS_SET="${RIDGE_TRUST_REGION_RADIUS+x}"
RIDGE_STANDARDIZE_FEATURES_WAS_SET="${RIDGE_STANDARDIZE_FEATURES+x}"
RIDGE_WEIGHTING_WAS_SET="${RIDGE_WEIGHTING+x}"
ADAPT_SCOPE="${ADAPT_SCOPE:-safe_operator}"
STAGE3_POSTERIOR_POLICY="${STAGE3_POSTERIOR_POLICY:-safe_operator_ablation}"
SUPPORT_GATE="${SUPPORT_GATE:-policy_default}"
SUPPORT_GATE_MIN_DELTA="${SUPPORT_GATE_MIN_DELTA:-0.0}"
SUPPORT_GATE_ROOTZONE_TOLERANCE="${SUPPORT_GATE_ROOTZONE_TOLERANCE:-0.0}"
SAFE_POLICY_JSON="${SAFE_POLICY_JSON:-}"
SAFE_POLICY_CACHE_ROOT="${SAFE_POLICY_CACHE_ROOT:-artifacts/runs/stage3_source_safe_policy_cache}"
AUTO_GENERATE_SAFE_POLICY="${AUTO_GENERATE_SAFE_POLICY:-0}"
SAFE_POLICY_CANDIDATE_SET="${SAFE_POLICY_CANDIDATE_SET:-stage3_conservative_v1}"
SAFE_POLICY_CALIBRATION_STAGE="${SAFE_POLICY_CALIBRATION_STAGE:-coarse}"
SAFE_POLICY_SOURCE_QUERY_MAX_SAMPLES="${SAFE_POLICY_SOURCE_QUERY_MAX_SAMPLES:-256}"
SAFE_POLICY_TARGET_CONTEXT_MAX_SAMPLES="${SAFE_POLICY_TARGET_CONTEXT_MAX_SAMPLES:-${TARGET_CONTEXT_MAX_SAMPLES:-0}}"
SAFE_POLICY_EVIDENCE_LEVEL="${SAFE_POLICY_EVIDENCE_LEVEL:-weaker}"
SAFE_POLICY_PSEUDO_TARGET_REGIONS="${SAFE_POLICY_PSEUDO_TARGET_REGIONS:-}"
SAFE_POLICY_KSHOT_UPDATE_REQUIREMENT="${SAFE_POLICY_KSHOT_UPDATE_REQUIREMENT:-nonzero_update}"
SAFE_POLICY_ALLOW_IN_CHECKPOINT_SOURCE_EPISODES="${SAFE_POLICY_ALLOW_IN_CHECKPOINT_SOURCE_EPISODES:-1}"
SOURCE_HELDOUT_CHECKPOINT_MAP="${SOURCE_HELDOUT_CHECKPOINT_MAP:-}"
REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT_WAS_SET="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT+x}"
REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT:-1}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-}"
SUPPORT_LOSS_REDUCTION="${SUPPORT_LOSS_REDUCTION:-global_pixel}"
FREEZE_MONTHLY_GAIN="${FREEZE_MONTHLY_GAIN:-0}"
ADAPT_SOLVER="${ADAPT_SOLVER:-adamw}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1.0}"
RIDGE_CLIP_COEFF_NORM="${RIDGE_CLIP_COEFF_NORM:-1.0}"
RIDGE_TRUST_REGION_RADIUS="${RIDGE_TRUST_REGION_RADIUS:-1.0}"
RIDGE_MAX_FEATURE_PIXELS="${RIDGE_MAX_FEATURE_PIXELS:-20000}"
RIDGE_STANDARDIZE_FEATURES="${RIDGE_STANDARDIZE_FEATURES:-0}"
RIDGE_WEIGHTING="${RIDGE_WEIGHTING:-global_pixel_l2}"
TARGET_CONTEXT_MAX_SAMPLES="${TARGET_CONTEXT_MAX_SAMPLES:-0}"
STAGE3_CONTEXT_TTA="${STAGE3_CONTEXT_TTA:-none}"
CONTEXT_TTA_RESIDUAL_SCALE="${CONTEXT_TTA_RESIDUAL_SCALE:-0.05}"
CONTEXT_TTA_RESIDUAL_CLIP_L2="${CONTEXT_TTA_RESIDUAL_CLIP_L2:-0.0}"
DIAGNOSTIC_KSHOT_STRENGTH="${DIAGNOSTIC_KSHOT_STRENGTH:-strong}"
EVAL_RAW_ADAPTED_BEFORE_MIX="${EVAL_RAW_ADAPTED_BEFORE_MIX:-1}"
ADAPT_MIX_RHO_WAS_SET="${ADAPT_MIX_RHO+x}"
ADAPT_MIX_RHO="${ADAPT_MIX_RHO:-}"
if [[ -n "${ADAPT_MIX_RHO_WAS_SET}" ]]; then
    ADAPT_MIX_RHO_SOURCE="explicit override"
else
    ADAPT_MIX_RHO_SOURCE="policy-derived adapt_mix_rho for K-shot; diagnostic K-shot defaults to 0.0"
fi
AUDIT_IDENTITY="${AUDIT_IDENTITY:-0}"
AUDIT_IDENTITY_TOLERANCE="${AUDIT_IDENTITY_TOLERANCE:-1e-8}"
ANCHOR_ALPHA_K4="${ANCHOR_ALPHA_K4:-0.75}"
ANCHOR_ALPHA_K12="${ANCHOR_ALPHA_K12:-0.25}"
LR_K12="${LR_K12:-3e-4}"
MAX_STEPS_K12="${MAX_STEPS_K12:-100}"
ADAPT_LR_OVERRIDE="${ADAPT_LR:-}"
ADAPT_MAX_STEPS_OVERRIDE="${ADAPT_MAX_STEPS:-}"
ADAPT_ANCHOR_ALPHA_OVERRIDE="${ADAPT_ANCHOR_ALPHA:-}"
ADAPT_WEIGHT_DECAY_OVERRIDE="${ADAPT_WEIGHT_DECAY:-${WEIGHT_DECAY:-}}"
ADAPT_GRAD_CLIP_OVERRIDE="${ADAPT_GRAD_CLIP:-${GRAD_CLIP:-}}"

cd "$(dirname "$0")/.."

if [[ -z "${SOURCE_CHECKPOINT}" ]]; then
    SOURCE_CHECKPOINT="$(find "artifacts/runs/phase4_hyperda_staged_ablation/M3_1_hyperda_trust_medium" \
        -path "*/${TARGET_REGION}/*s${SEED}*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
        -type f 2>/dev/null | sort | tail -1 || true)"
fi

if [[ -z "${SOURCE_CHECKPOINT}" ]]; then
    SOURCE_CHECKPOINT="$(find "artifacts/runs/phase4_hyperda_staged_ablation/M2_1_rank_gated_dora_stable" \
        -path "*/${TARGET_REGION}/*s${SEED}*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
        -type f 2>/dev/null | sort | tail -1 || true)"
fi

if [[ -z "${SOURCE_CHECKPOINT}" ]]; then
    SOURCE_CHECKPOINT="$(find "artifacts/runs/phase4_hyperda_staged/${TARGET_REGION}" \
        -path "*s${SEED}*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
        -type f 2>/dev/null | sort | tail -1 || true)"
fi

if [[ -z "${SOURCE_CHECKPOINT}" ]]; then
    SOURCE_CHECKPOINT="$(find artifacts/runs/phase4_prompt_conditioned \
        -path "*hyperda_basis_adapter_${TARGET_REGION}_*_s${SEED}_*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
        -type f 2>/dev/null | sort | tail -1 || true)"
fi

if [[ -z "${SOURCE_CHECKPOINT}" || ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "ERROR: source HyperDA checkpoint not found." >&2
    echo "Provide it explicitly:" >&2
    echo "  bash run/phase5_hyperda_zero_few_shot_eval.sh /path/to/source.pt ${TARGET_REGION} ${SEED} ${CUDA_VISIBLE_DEVICES}" >&2
    echo "Or train the staged source prior first:" >&2
    echo "  bash run/phase4_hyperda_staged.sh auto ${TARGET_REGION} ${SEED} ${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
fi

if [[ "${REQUIRE_SOURCE_GATE_JSON_FOR_TARGET_EVAL}" == "1" || "${REQUIRE_SOURCE_GATE_JSON_FOR_TARGET_EVAL,,}" == "true" ]]; then
    if [[ -z "${SOURCE_GATE_JSON}" || ! -f "${SOURCE_GATE_JSON}" ]]; then
        echo "ERROR: target_eval requires SOURCE_GATE_JSON when REQUIRE_SOURCE_GATE_JSON_FOR_TARGET_EVAL=1." >&2
        echo "Set SOURCE_GATE_JSON=/path/to/source_gate.json produced by the source-side gate." >&2
        exit 2
    fi
    python3 - "${SOURCE_GATE_JSON}" <<'PYTHON_SCRIPT'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(encoding="utf-8") as f:
    gate = json.load(f)
if not gate.get("source_gate_pass"):
    raise SystemExit(f"source gate did not pass: {path}")
if "target_eval_allowed" in gate and not gate.get("target_eval_allowed"):
    raise SystemExit(f"source gate does not allow target_eval: {path}")
if gate.get("method_id") == "M3_13_phys_gain_guarded_hypertrust":
    if gate.get("schema_version") != "m3_13_source_gate_report_v1":
        raise SystemExit(f"unsupported M3_13 source gate schema: {gate.get('schema_version')!r}")
    if gate.get("identity_diagnostic"):
        raise SystemExit(f"M3_13 identity diagnostic fallback cannot enter target_eval: {path}")
    eta_positive = (
        float(gate.get("selected_eta_surface", 0.0)) > 0.0
        or float(gate.get("selected_eta_rootzone", 0.0)) > 0.0
    )
    if not eta_positive:
        raise SystemExit(f"M3_13 target_eval requires a positive selected eta: {path}")
PYTHON_SCRIPT
fi

mkdir -p "${OUTPUT_BASE}"

if [[ "${STAGE3_KSHOT_MODE}" != "paper_safe" && "${STAGE3_KSHOT_MODE}" != "diagnostic_direct_kshot" && "${STAGE3_KSHOT_MODE}" != "diagnostic_direct_kshot_v2" && "${STAGE3_KSHOT_MODE}" != "diagnostic_conservative_kshot_v3" && "${STAGE3_KSHOT_MODE}" != "diagnostic_support_gain_v1" && "${STAGE3_KSHOT_MODE}" != "diagnostic_support_gain_v2" && "${STAGE3_KSHOT_MODE}" != "diagnostic_support_gain_v3_stable" && "${STAGE3_KSHOT_MODE}" != "diagnostic_support_gain_v4_nested_stable" && "${STAGE3_KSHOT_MODE}" != "diagnostic_support_gain_v12_nested_cv" && "${STAGE3_KSHOT_MODE}" != "diagnostic_support_gain_v13_k12_aggressive_calibration_pool" && "${STAGE3_KSHOT_MODE}" != "diagnostic_finetune_support_gain_v14_nested" && "${STAGE3_KSHOT_MODE}" != "diagnostic_support_affine_v1_nested" && "${STAGE3_KSHOT_MODE}" != "diagnostic_safe_operator_v5_nested" && "${STAGE3_KSHOT_MODE}" != "diagnostic_linearized_coeff_ridge_v6_nested" && "${STAGE3_KSHOT_MODE}" != "diagnostic_linearized_coeff_ridge_v7_balanced_nested" && "${STAGE3_KSHOT_MODE}" != "diagnostic_linearized_coeff_ridge_v8_hybrid_nested" && "${STAGE3_KSHOT_MODE}" != "diagnostic_linearized_coeff_ridge_v9_guarded_nested" && "${STAGE3_KSHOT_MODE}" != "diagnostic_linearized_coeff_ridge_v10_support_pool_nested" && "${STAGE3_KSHOT_MODE}" != "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested" ]]; then
    echo "ERROR: STAGE3_KSHOT_MODE must be a supported paper_safe or diagnostic mode; got ${STAGE3_KSHOT_MODE}" >&2
    exit 2
fi

if [[ "${STAGE3_KSHOT_MODE}" == "diagnostic_direct_kshot" || "${STAGE3_KSHOT_MODE}" == "diagnostic_direct_kshot_v2" || "${STAGE3_KSHOT_MODE}" == "diagnostic_conservative_kshot_v3" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v1" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v2" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v3_stable" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v4_nested_stable" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v12_nested_cv" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v13_k12_aggressive_calibration_pool" || "${STAGE3_KSHOT_MODE}" == "diagnostic_finetune_support_gain_v14_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_affine_v1_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_safe_operator_v5_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v6_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v7_balanced_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v8_hybrid_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v9_guarded_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v10_support_pool_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested" ]]; then
    if [[ -z "${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT_WAS_SET}" ]]; then
        REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="0"
    fi
    ADAPT_MIX_RHO_SOURCE="${STAGE3_KSHOT_MODE} fixed diagnostic rho unless explicitly overridden"
fi

resolve_safe_policy_cache_path() {
python3 - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" <<'PYTHON_SCRIPT'
import hashlib
import json
import sys
from pathlib import Path

(
    source_checkpoint,
    splits_json,
    target_region,
    seed,
    candidate_set,
    calibration_stage,
    source_query_max_samples,
    target_context_max_samples,
    evidence_level,
    pseudo_target_regions,
    kshot_update_requirement,
    cache_root,
) = sys.argv[1:13]


def sha256_file(path_text: str) -> str:
    path = Path(path_text)
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


payload = {
    "schema_version": "stage3_safe_policy_cache_v1",
    "source_checkpoint": str(Path(source_checkpoint).resolve()),
    "source_checkpoint_sha256": sha256_file(source_checkpoint),
    "splits_json": str(Path(splits_json).resolve()),
    "splits_json_sha256": sha256_file(splits_json),
    "target_region": target_region,
    "seed": str(seed),
    "candidate_set": candidate_set,
    "calibration_stage": calibration_stage,
    "source_query_max_samples": str(source_query_max_samples),
    "target_context_max_samples": str(target_context_max_samples),
    "evidence_level": evidence_level,
    "pseudo_target_regions": pseudo_target_regions,
    "kshot_policy_update_requirement": kshot_update_requirement,
    "calibration_scripts": [
        "scripts/eval/run_stage3_source_safe_policy_calibration.py",
        "scripts/eval/calibrate_source_safe_guard.py",
    ],
}
cache_hash = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
cache_key = f"{target_region}_s{seed}_{cache_hash[:16]}"
print(cache_key)
print(str(Path(cache_root) / cache_key))
print(json.dumps(payload, sort_keys=True))
PYTHON_SCRIPT
}

write_safe_policy_cache_manifest() {
python3 - "$1" "$2" "$3" "$4" <<'PYTHON_SCRIPT'
import json
import sys
from pathlib import Path

cache_dir, cache_key, payload_json, safe_policy_json = sys.argv[1:5]
cache_path = Path(cache_dir)
payload = json.loads(payload_json)
manifest = {
    **payload,
    "safe_policy_cache_key": cache_key,
    "safe_policy_cache_dir": str(cache_path),
    "safe_policy_json": safe_policy_json,
    "cache_manifest": str(cache_path / "safe_policy_cache_manifest.json"),
}
cache_path.mkdir(parents=True, exist_ok=True)
(cache_path / "safe_policy_cache_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PYTHON_SCRIPT
}

generate_cached_safe_policy() {
    local cache_dir="$1"
    local source_rows_dir="${cache_dir}/source_rows"
    local row_builder_extra_args=()
    local exporter_extra_args=()

    if [[ -n "${SAFE_POLICY_PSEUDO_TARGET_REGIONS}" ]]; then
        row_builder_extra_args+=(--pseudo_target_regions "${SAFE_POLICY_PSEUDO_TARGET_REGIONS}")
        exporter_extra_args+=(--pseudo_target_regions "${SAFE_POLICY_PSEUDO_TARGET_REGIONS}")
    fi

    if [[ "${SAFE_POLICY_EVIDENCE_LEVEL}" == "strict" ]]; then
        if [[ -z "${SOURCE_HELDOUT_CHECKPOINT_MAP}" ]]; then
            echo "ERROR: SAFE_POLICY_EVIDENCE_LEVEL=strict requires SOURCE_HELDOUT_CHECKPOINT_MAP." >&2
            exit 2
        fi
        row_builder_extra_args+=(--source_heldout_checkpoint_map "${SOURCE_HELDOUT_CHECKPOINT_MAP}")
    elif [[ "${SAFE_POLICY_ALLOW_IN_CHECKPOINT_SOURCE_EPISODES}" == "1" || "${SAFE_POLICY_ALLOW_IN_CHECKPOINT_SOURCE_EPISODES,,}" == "true" ]]; then
        row_builder_extra_args+=(--allow_in_checkpoint_source_episodes)
        exporter_extra_args+=(--allow_in_checkpoint_source_episodes)
    fi

    echo "Generating cached source-side SAFE policy:"
    echo "  cache_dir=${cache_dir}"
    echo "  source_rows_dir=${source_rows_dir}"
    PYTHONPATH=. python scripts/eval/run_stage3_source_safe_policy_calibration.py \
        --source_checkpoint "${SOURCE_CHECKPOINT}" \
        --final_target_region "${TARGET_REGION}" \
        --seed "${SEED}" \
        --candidate_set "${SAFE_POLICY_CANDIDATE_SET}" \
        --source_query_max_samples "${SAFE_POLICY_SOURCE_QUERY_MAX_SAMPLES}" \
        --target_context_max_samples "${SAFE_POLICY_TARGET_CONTEXT_MAX_SAMPLES}" \
        --evidence_level "${SAFE_POLICY_EVIDENCE_LEVEL}" \
        --output_dir "${source_rows_dir}" \
        --splits_json "${SPLITS_JSON}" \
        --cuda_device "${CUDA_VISIBLE_DEVICES}" \
        --adapt_batch_size "${ADAPT_BATCH_SIZE}" \
        --eval_batch_size "${EVAL_BATCH_SIZE}" \
        --skip_existing \
        "${row_builder_extra_args[@]}"

    PYTHONPATH=. python scripts/eval/calibrate_source_safe_guard.py \
        --input_roots "${source_rows_dir}" \
        --output_dir "${cache_dir}" \
        --final_target_region "${TARGET_REGION}" \
        --seed "${SEED}" \
        --source_checkpoint "${SOURCE_CHECKPOINT}" \
        --candidate_set "${SAFE_POLICY_CANDIDATE_SET}" \
        --calibration_stage "${SAFE_POLICY_CALIBRATION_STAGE}" \
        --source_query_max_samples "${SAFE_POLICY_SOURCE_QUERY_MAX_SAMPLES}" \
        --source_rows_root "${source_rows_dir}" \
        --kshot_policy_update_requirement "${SAFE_POLICY_KSHOT_UPDATE_REQUIREMENT}" \
        "${exporter_extra_args[@]}"
}

HAS_KSHOT="0"
for K_VALUE_FOR_POLICY in ${K_LIST}; do
    if [[ "${K_VALUE_FOR_POLICY}" != "0" ]]; then
        HAS_KSHOT="1"
    fi
done

SAFE_POLICY_CACHE_KEY=""
SAFE_POLICY_CACHE_DIR=""
SAFE_POLICY_CACHE_MANIFEST_JSON=""
SAFE_POLICY_CACHE_STATUS="not_used"
if [[ "${STAGE3_KSHOT_MODE}" == "paper_safe" && "${HAS_KSHOT}" == "1" && -z "${SAFE_POLICY_JSON}" ]]; then
    mapfile -t SAFE_POLICY_CACHE_INFO < <(resolve_safe_policy_cache_path \
        "${SOURCE_CHECKPOINT}" \
        "${SPLITS_JSON}" \
        "${TARGET_REGION}" \
        "${SEED}" \
        "${SAFE_POLICY_CANDIDATE_SET}" \
        "${SAFE_POLICY_CALIBRATION_STAGE}" \
        "${SAFE_POLICY_SOURCE_QUERY_MAX_SAMPLES}" \
        "${SAFE_POLICY_TARGET_CONTEXT_MAX_SAMPLES}" \
        "${SAFE_POLICY_EVIDENCE_LEVEL}" \
        "${SAFE_POLICY_PSEUDO_TARGET_REGIONS}" \
        "${SAFE_POLICY_KSHOT_UPDATE_REQUIREMENT}" \
        "${SAFE_POLICY_CACHE_ROOT}")
    SAFE_POLICY_CACHE_KEY="${SAFE_POLICY_CACHE_INFO[0]}"
    SAFE_POLICY_CACHE_DIR="${SAFE_POLICY_CACHE_INFO[1]}"
    SAFE_POLICY_CACHE_MANIFEST_JSON="${SAFE_POLICY_CACHE_INFO[2]}"
    CACHED_SAFE_POLICY_JSON="${SAFE_POLICY_CACHE_DIR}/safe_policy.json"
    if [[ -f "${CACHED_SAFE_POLICY_JSON}" ]]; then
        SAFE_POLICY_JSON="${CACHED_SAFE_POLICY_JSON}"
        SAFE_POLICY_CACHE_STATUS="reused cached safe_policy.json"
    elif [[ "${AUTO_GENERATE_SAFE_POLICY}" == "1" || "${AUTO_GENERATE_SAFE_POLICY,,}" == "true" ]]; then
        generate_cached_safe_policy "${SAFE_POLICY_CACHE_DIR}"
        if [[ ! -f "${CACHED_SAFE_POLICY_JSON}" ]]; then
            echo "ERROR: SAFE policy generation completed but did not create ${CACHED_SAFE_POLICY_JSON}" >&2
            exit 2
        fi
        SAFE_POLICY_JSON="${CACHED_SAFE_POLICY_JSON}"
        SAFE_POLICY_CACHE_STATUS="generated cached safe_policy.json"
    else
        echo "ERROR: paper-facing K-shot requires SAFE_POLICY_JSON from source-side episode calibration." >&2
        echo "No cached safe_policy.json found at: ${CACHED_SAFE_POLICY_JSON}" >&2
        echo "Options:" >&2
        echo "  1) set SAFE_POLICY_JSON=/path/to/safe_policy.json" >&2
        echo "  2) set AUTO_GENERATE_SAFE_POLICY=1 to build and cache it" >&2
        echo "  3) set STAGE3_KSHOT_MODE=diagnostic_direct_kshot for non-paper diagnostics" >&2
        exit 2
    fi
    write_safe_policy_cache_manifest \
        "${SAFE_POLICY_CACHE_DIR}" \
        "${SAFE_POLICY_CACHE_KEY}" \
        "${SAFE_POLICY_CACHE_MANIFEST_JSON}" \
        "${SAFE_POLICY_JSON}"
fi

resolve_policy_adapt_mix_rho() {
python3 - "$1" "$2" "$3" <<'PYTHON_SCRIPT'
import json
import sys
from pathlib import Path

policy_path, adaptation_setting, k_value = sys.argv[1:4]
if not policy_path:
    print("1.0" if int(k_value) == 0 else "0.0")
    raise SystemExit(0)

path = Path(policy_path)
with path.open(encoding="utf-8") as f:
    policy = json.load(f)

policies = policy.get("policies", {})
if not isinstance(policies, dict):
    print("1.0" if int(k_value) == 0 else "0.0")
    raise SystemExit(0)

for key in (adaptation_setting, f"K{int(k_value)}", f"k{int(k_value)}"):
    selected = policies.get(key)
    if isinstance(selected, dict) and selected.get("adapt_mix_rho") is not None:
        print(selected["adapt_mix_rho"])
        raise SystemExit(0)

print("1.0" if int(k_value) == 0 else "0.0")
PYTHON_SCRIPT
}

prepare_nested_k12_support_manifest() {
python3 - "$1" "$2" "$3" "$4" <<'PYTHON_SCRIPT'
import hashlib
import json
import sys
from pathlib import Path

split_path = Path(sys.argv[1])
output_base = Path(sys.argv[2])
target_region = sys.argv[3]
seed = int(sys.argv[4])
policy = "run_local_k12_nested_k4_plus_8_original_k12_nonduplicate"


def records_hash(records):
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_entry(data, setting):
    for entry in data.get("splits", []):
        if entry.get("target_region_id") != target_region:
            continue
        try:
            entry_seed = int(entry.get("seed", -1))
        except Exception:
            continue
        if entry_seed != seed:
            continue
        if entry.get("adaptation_setting") == setting:
            return entry
    raise SystemExit(f"missing split entry for {target_region} seed={seed} setting={setting}")


with split_path.open(encoding="utf-8") as f:
    split_data = json.load(f)

k4_entry = load_entry(split_data, "few_shot_k4")
k12_entry = load_entry(split_data, "few_shot_k12")
k4_records = list(k4_entry.get("target_support_dates", []) or [])
k12_records = list(k12_entry.get("target_support_dates", []) or [])
if not k4_records or not k12_records:
    raise SystemExit("cannot build nested K12 manifest without both K4 and K12 support records")

seen = set()
nested_records = []
for record in k4_records:
    key = (record.get("time_index"), record.get("datetime_str", record.get("date_str")))
    if key in seen:
        continue
    seen.add(key)
    nested_records.append(record)
for record in k12_records:
    key = (record.get("time_index"), record.get("datetime_str", record.get("date_str")))
    if key in seen:
        continue
    seen.add(key)
    nested_records.append(record)
    if len(nested_records) >= 12:
        break
if len(nested_records) != 12:
    raise SystemExit(f"expected 12 nested K12 support records, got {len(nested_records)}")

nested_hash = records_hash(nested_records)
nested_dir = output_base / "nested_support"
nested_dir.mkdir(parents=True, exist_ok=True)
nested_support_path = nested_dir / f"{target_region}_s{seed}_K12_nested_support.json"
nested_split_path = nested_dir / f"{target_region}_s{seed}_K12_nested_splits.json"
nested_support_payload = {
    "schema_version": "hyperda_k12_nested_support_diagnostic_v2",
    "target_region": target_region,
    "seed": seed,
    "support_nesting_policy": policy,
    "source": "derived_from_frozen_v4_4_manifest_without_target_eval",
    "k4_support_dates": k4_records,
    "k12_original_support_dates": k12_records,
    "k12_nested_support_dates": nested_records,
    "nested_support_dates_hash": nested_hash,
    "target_eval_usage": "final_eval_only_no_selection",
    "leakage_note": "uses only target_support records from 2015-2021; target_eval is not used for selection",
}
nested_support_path.write_text(json.dumps(nested_support_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

nested_split_data = json.loads(json.dumps(split_data))
updated = False
for entry in nested_split_data.get("splits", []):
    if entry.get("target_region_id") != target_region:
        continue
    try:
        entry_seed = int(entry.get("seed", -1))
    except Exception:
        continue
    if entry_seed != seed:
        continue
    if entry.get("adaptation_setting") != "few_shot_k12":
        continue
    entry["target_support_dates"] = nested_records
    entry["target_train_dates"] = nested_records
    entry["target_adaptation_dates"] = nested_records
    entry["target_support_cycle_count"] = len(nested_records)
    entry["target_train_cycle_count"] = len(nested_records)
    entry["target_adaptation_cycle_count"] = len(nested_records)
    entry["target_support_dates_hash"] = nested_hash
    entry["target_train_dates_hash"] = nested_hash
    entry["target_adaptation_dates_hash"] = nested_hash
    entry["support_dates_hash"] = nested_hash
    entry["support_nesting_policy"] = policy
    entry["nested_support_dates_hash"] = nested_hash
    entry["nested_support_manifest"] = str(nested_support_path)
    entry["diagnostic_split_note"] = "K12 nested stable diagnostic: K4 support plus 8 original K12 nonduplicate support dates"
    updated = True
    break
if not updated:
    raise SystemExit("failed to update K12 split entry")
nested_split_path.write_text(json.dumps(nested_split_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(str(nested_split_path))
print(str(nested_support_path))
print(nested_hash)
PYTHON_SCRIPT
}

NESTED_K12_SPLITS_JSON=""
NESTED_K12_SUPPORT_JSON=""
NESTED_K12_SUPPORT_DATES_HASH=""
if [[ "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v4_nested_stable" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v12_nested_cv" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v13_k12_aggressive_calibration_pool" || "${STAGE3_KSHOT_MODE}" == "diagnostic_finetune_support_gain_v14_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_affine_v1_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_safe_operator_v5_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v6_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v7_balanced_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v8_hybrid_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v9_guarded_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v10_support_pool_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested" ]]; then
    mapfile -t NESTED_K12_INFO < <(prepare_nested_k12_support_manifest \
        "${SPLITS_JSON}" \
        "${OUTPUT_BASE}" \
        "${TARGET_REGION}" \
        "${SEED}")
    NESTED_K12_SPLITS_JSON="${NESTED_K12_INFO[0]}"
    NESTED_K12_SUPPORT_JSON="${NESTED_K12_INFO[1]}"
    NESTED_K12_SUPPORT_DATES_HASH="${NESTED_K12_INFO[2]}"
fi

echo "============================================"
echo "Phase 5 HyperDA Zero/Few-Shot Adapt + Eval"
echo "  source_checkpoint=${SOURCE_CHECKPOINT}"
echo "  target_region=${TARGET_REGION}"
echo "  seed=${SEED}"
echo "  K_LIST=${K_LIST}"
echo "  STAGE3_KSHOT_MODE=${STAGE3_KSHOT_MODE}"
echo "  EVAL_OUTPUT_LEVEL=${EVAL_OUTPUT_LEVEL}"
echo "  ADAPT_RECIPE=${ADAPT_RECIPE}"
echo "  ADAPT_SCOPE=${ADAPT_SCOPE} (SAFE Prompt+Coeff+Gain)"
echo "  STAGE3_POSTERIOR_POLICY=${STAGE3_POSTERIOR_POLICY}"
echo "  SUPPORT_GATE=${SUPPORT_GATE} min_delta=${SUPPORT_GATE_MIN_DELTA} rootzone_tolerance=${SUPPORT_GATE_ROOTZONE_TOLERANCE}"
echo "  SAFE_POLICY_JSON=${SAFE_POLICY_JSON:-<none>}"
echo "  SAFE_POLICY_CACHE_ROOT=${SAFE_POLICY_CACHE_ROOT}"
echo "  SAFE_POLICY_CACHE_KEY=${SAFE_POLICY_CACHE_KEY:-<none>}"
echo "  SAFE_POLICY_CACHE_STATUS=${SAFE_POLICY_CACHE_STATUS}"
echo "  AUTO_GENERATE_SAFE_POLICY=${AUTO_GENERATE_SAFE_POLICY}"
echo "  REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT=${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT}"
echo "  policy_source=source_side_episode_calibration for K-shot when SAFE_POLICY_JSON is provided"
echo "  K-shot strict policy requires nonzero target_support update when REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT=1"
echo "  adapt_solver=${ADAPT_SOLVER}"
echo "  ridge_lambda=${RIDGE_LAMBDA} ridge_clip_coeff_norm=${RIDGE_CLIP_COEFF_NORM} ridge_trust_region_radius=${RIDGE_TRUST_REGION_RADIUS} ridge_max_feature_pixels=${RIDGE_MAX_FEATURE_PIXELS} ridge_standardize_features=${RIDGE_STANDARDIZE_FEATURES} ridge_weighting=${RIDGE_WEIGHTING}"
echo "  SCHEDULE_LABEL=${SCHEDULE_LABEL}"
echo "  SUPPORT_LOSS_REDUCTION=${SUPPORT_LOSS_REDUCTION}"
echo "  FREEZE_MONTHLY_GAIN=${FREEZE_MONTHLY_GAIN}"
echo "  TARGET_CONTEXT_MAX_SAMPLES=${TARGET_CONTEXT_MAX_SAMPLES} (0 = full target_context)"
echo "  STAGE3_CONTEXT_TTA=${STAGE3_CONTEXT_TTA} (target_context input-side only; target_eval frozen read-only)"
echo "  CONTEXT_TTA_RESIDUAL_SCALE=${CONTEXT_TTA_RESIDUAL_SCALE} clip_l2=${CONTEXT_TTA_RESIDUAL_CLIP_L2}"
echo "  DIAGNOSTIC_KSHOT_STRENGTH=${DIAGNOSTIC_KSHOT_STRENGTH}"
echo "  ADAPT_MIX_RHO=${ADAPT_MIX_RHO} (${ADAPT_MIX_RHO_SOURCE})"
echo "  AUDIT_IDENTITY=${AUDIT_IDENTITY} tolerance=${AUDIT_IDENTITY_TOLERANCE}"
echo "  eval_split=target_eval"
echo "  require_source_gate_json_for_target_eval=${REQUIRE_SOURCE_GATE_JSON_FOR_TARGET_EVAL}"
echo "  source_gate_json=${SOURCE_GATE_JSON:-<none>}"
echo "  eval_max_samples=${EVAL_MAX_SAMPLES}"
echo "  adapt_batch_size=${ADAPT_BATCH_SIZE}"
echo "  eval_batch_size=${EVAL_BATCH_SIZE}"
echo "  adapt_lr_override=${ADAPT_LR_OVERRIDE:-<none>}"
echo "  adapt_max_steps_override=${ADAPT_MAX_STEPS_OVERRIDE:-<none>}"
echo "  adapt_anchor_alpha_override=${ADAPT_ANCHOR_ALPHA_OVERRIDE:-<none>}"
echo "  splits_json=${SPLITS_JSON}"
echo "  nested_k12_splits_json=${NESTED_K12_SPLITS_JSON:-<none>}"
echo "  nested_k12_support_hash=${NESTED_K12_SUPPORT_DATES_HASH:-<none>}"
echo "  output_base=${OUTPUT_BASE}"
echo "  device=gpu:${CUDA_VISIBLE_DEVICES}"
echo "============================================"

for K in ${K_LIST}; do
    ADAPT_SCOPE_FOR_K="${ADAPT_SCOPE}"
    ADAPT_SOLVER_FOR_K="${ADAPT_SOLVER}"
    RIDGE_LAMBDA_FOR_K="${RIDGE_LAMBDA}"
    RIDGE_CLIP_COEFF_NORM_FOR_K="${RIDGE_CLIP_COEFF_NORM}"
    RIDGE_TRUST_REGION_RADIUS_FOR_K="${RIDGE_TRUST_REGION_RADIUS}"
    RIDGE_STANDARDIZE_FEATURES_FOR_K="${RIDGE_STANDARDIZE_FEATURES}"
    RIDGE_WEIGHTING_FOR_K="${RIDGE_WEIGHTING}"
    AUDIT_IDENTITY_FOR_K="0"
    if [[ "${K}" == "0" ]]; then
        ADAPTATION_SETTING="zero_shot_context"
        ADAPTATION_MAX_STEPS="0"
        ADAPTATION_LR="${LR_K0:-${LR:-1e-3}}"
        ANCHOR_ALPHA="0.0"
        if [[ "${STAGE3_POSTERIOR_POLICY}" == "conservative_coeff_posterior" ]]; then
            ADAPT_SCOPE_FOR_K="none"
        fi
    elif [[ "${K}" == "4" ]]; then
        ADAPTATION_SETTING="few_shot_k4"
        ADAPTATION_MAX_STEPS="${ADAPT_MAX_STEPS_OVERRIDE:-${MAX_STEPS_K4:-${MAX_STEPS:-100}}}"
        ADAPTATION_LR="${ADAPT_LR_OVERRIDE:-${LR_K4:-${LR:-1e-3}}}"
        ANCHOR_ALPHA="${ADAPT_ANCHOR_ALPHA_OVERRIDE:-${ANCHOR_ALPHA_K4}}"
    elif [[ "${K}" == "12" ]]; then
        ADAPTATION_SETTING="few_shot_k12"
        ADAPTATION_MAX_STEPS="${ADAPT_MAX_STEPS_OVERRIDE:-${MAX_STEPS_K12:-${MAX_STEPS:-100}}}"
        ADAPTATION_LR="${ADAPT_LR_OVERRIDE:-${LR_K12:-${LR:-3e-4}}}"
        ANCHOR_ALPHA="${ADAPT_ANCHOR_ALPHA_OVERRIDE:-${ANCHOR_ALPHA_K12}}"
    else
        echo "ERROR: K_LIST may contain only 0, 4, 12; got ${K}" >&2
        exit 2
    fi
    if [[ -n "${ADAPT_MIX_RHO_WAS_SET}" ]]; then
        ADAPT_MIX_RHO_FOR_K="${ADAPT_MIX_RHO}"
    elif [[ "${K}" != "0" && "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v10_support_pool_nested" ]]; then
        if [[ "${K}" == "4" ]]; then
            ADAPT_MIX_RHO_FOR_K="0.35"
        else
            ADAPT_MIX_RHO_FOR_K="0.45"
        fi
    elif [[ "${K}" != "0" && "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested" ]]; then
        if [[ "${K}" == "4" ]]; then
            ADAPT_MIX_RHO_FOR_K="0.20"
        else
            ADAPT_MIX_RHO_FOR_K="0.30"
        fi
    elif [[ "${K}" != "0" && ( "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v6_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v7_balanced_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v8_hybrid_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v9_guarded_nested" ) ]]; then
        if [[ "${K}" == "4" && "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
            ADAPT_MIX_RHO_FOR_K="0.35"
        elif [[ "${K}" == "4" ]]; then
            ADAPT_MIX_RHO_FOR_K="0.50"
        elif [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
            ADAPT_MIX_RHO_FOR_K="0.50"
        else
            ADAPT_MIX_RHO_FOR_K="0.65"
        fi
    elif [[ "${K}" != "0" && ( "${STAGE3_KSHOT_MODE}" == "diagnostic_conservative_kshot_v3" || "${STAGE3_KSHOT_MODE}" == "diagnostic_safe_operator_v5_nested" ) ]]; then
        if [[ "${K}" == "4" && "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
            ADAPT_MIX_RHO_FOR_K="0.35"
        elif [[ "${K}" == "4" ]]; then
            ADAPT_MIX_RHO_FOR_K="0.50"
        elif [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
            ADAPT_MIX_RHO_FOR_K="0.45"
        else
            ADAPT_MIX_RHO_FOR_K="0.60"
        fi
    elif [[ "${K}" != "0" && "${STAGE3_KSHOT_MODE}" == "diagnostic_finetune_support_gain_v14_nested" ]]; then
        ADAPT_MIX_RHO_FOR_K="1.0"
    elif [[ "${K}" != "0" && ( "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v1" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v2" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v3_stable" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v4_nested_stable" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v12_nested_cv" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v13_k12_aggressive_calibration_pool" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_affine_v1_nested" ) ]]; then
        ADAPT_MIX_RHO_FOR_K="1.0"
    elif [[ "${K}" != "0" && ( "${STAGE3_KSHOT_MODE}" == "diagnostic_direct_kshot" || "${STAGE3_KSHOT_MODE}" == "diagnostic_direct_kshot_v2" ) ]]; then
        ADAPT_MIX_RHO_FOR_K="1.0"
    else
        ADAPT_MIX_RHO_FOR_K="$(resolve_policy_adapt_mix_rho "${SAFE_POLICY_JSON}" "${ADAPTATION_SETTING}" "${K}")"
    fi
    if [[ "${AUDIT_IDENTITY}" == "1" || "${AUDIT_IDENTITY,,}" == "true" ]]; then
        if [[ "${K}" == "12" ]]; then
            ADAPT_SCOPE_FOR_K="none"
            ADAPTATION_MAX_STEPS="0"
            ANCHOR_ALPHA="0.0"
            AUDIT_IDENTITY_FOR_K="1"
        fi
    fi
    if [[ "${K}" != "0" && "${STAGE3_KSHOT_MODE}" == "diagnostic_direct_kshot" && "${AUDIT_IDENTITY_FOR_K}" != "1" ]]; then
        ADAPT_SCOPE_FOR_K="${ADAPT_SCOPE:-safe_operator}"
        ANCHOR_ALPHA="${ADAPT_ANCHOR_ALPHA_OVERRIDE:-1.0}"
        SUPPORT_GATE_FOR_K="off"
        SUPPORT_LOSS_REDUCTION_FOR_K="cycle_balanced"
        REQUIRE_SAFE_POLICY_FOR_K="0"
    elif [[ "${K}" != "0" && "${STAGE3_KSHOT_MODE}" == "diagnostic_direct_kshot_v2" && "${AUDIT_IDENTITY_FOR_K}" != "1" ]]; then
        ADAPT_SCOPE_FOR_K="${ADAPT_SCOPE:-safe_operator}"
        ANCHOR_ALPHA="${ADAPT_ANCHOR_ALPHA_OVERRIDE:-1.0}"
        ADAPTATION_LR="${ADAPT_LR_OVERRIDE:-${LR:-1e-3}}"
        if [[ -z "${ADAPT_MAX_STEPS_OVERRIDE}" ]]; then
            if [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" && "${K}" == "4" ]]; then
                ADAPTATION_MAX_STEPS="50"
            elif [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" && "${K}" == "12" ]]; then
                ADAPTATION_MAX_STEPS="100"
            elif [[ "${K}" == "4" ]]; then
                ADAPTATION_MAX_STEPS="100"
            else
                ADAPTATION_MAX_STEPS="200"
            fi
        fi
        SUPPORT_GATE_FOR_K="off"
        SUPPORT_LOSS_REDUCTION_FOR_K="cycle_balanced"
        REQUIRE_SAFE_POLICY_FOR_K="0"
    elif [[ "${K}" != "0" && ( "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v1" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v2" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v3_stable" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v4_nested_stable" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v12_nested_cv" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v13_k12_aggressive_calibration_pool" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_affine_v1_nested" ) && "${AUDIT_IDENTITY_FOR_K}" != "1" ]]; then
        ADAPT_SCOPE_FOR_K="none"
        STAGE3_POSTERIOR_POLICY="source_calibrated_mix"
        ANCHOR_ALPHA="0.0"
        ADAPTATION_MAX_STEPS="0"
        ADAPTATION_LR="${ADAPT_LR_OVERRIDE:-${LR:-1e-3}}"
        SUPPORT_GATE_FOR_K="off"
        SUPPORT_LOSS_REDUCTION_FOR_K="cycle_balanced"
        if [[ -z "${FREEZE_MONTHLY_GAIN_WAS_SET}" ]]; then FREEZE_MONTHLY_GAIN="1"; fi
        REQUIRE_SAFE_POLICY_FOR_K="0"
    elif [[ "${K}" != "0" && "${STAGE3_KSHOT_MODE}" == "diagnostic_finetune_support_gain_v14_nested" && "${AUDIT_IDENTITY_FOR_K}" != "1" ]]; then
        if [[ -z "${ADAPT_SCOPE_WAS_SET}" ]]; then
            ADAPT_SCOPE_FOR_K="coeff_only"
        fi
        if [[ -z "${STAGE3_POSTERIOR_POLICY_WAS_SET}" ]]; then
            STAGE3_POSTERIOR_POLICY="conservative_coeff_posterior"
        fi
        if [[ -z "${ADAPT_MAX_STEPS_OVERRIDE}" ]]; then
            if [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" && "${K}" == "4" ]]; then
                ADAPTATION_MAX_STEPS="20"
            elif [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" && "${K}" == "12" ]]; then
                ADAPTATION_MAX_STEPS="40"
            elif [[ "${K}" == "4" ]]; then
                ADAPTATION_MAX_STEPS="40"
            else
                ADAPTATION_MAX_STEPS="80"
            fi
        fi
        ADAPTATION_LR="${ADAPT_LR_OVERRIDE:-${LR:-3e-4}}"
        if [[ -z "${ADAPT_ANCHOR_ALPHA_OVERRIDE}" ]]; then
            if [[ "${K}" == "4" && "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
                ANCHOR_ALPHA="0.25"
            elif [[ "${K}" == "4" ]]; then
                ANCHOR_ALPHA="0.35"
            elif [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
                ANCHOR_ALPHA="0.35"
            else
                ANCHOR_ALPHA="0.50"
            fi
        fi
        if [[ -z "${ADAPT_SOLVER_WAS_SET}" ]]; then ADAPT_SOLVER_FOR_K="adamw"; fi
        if [[ -z "${SUPPORT_GATE_WAS_SET}" ]]; then SUPPORT_GATE_FOR_K="auto"; else SUPPORT_GATE_FOR_K="${SUPPORT_GATE}"; fi
        if [[ -z "${SUPPORT_LOSS_REDUCTION_WAS_SET}" ]]; then SUPPORT_LOSS_REDUCTION_FOR_K="cycle_balanced"; else SUPPORT_LOSS_REDUCTION_FOR_K="${SUPPORT_LOSS_REDUCTION}"; fi
        if [[ -z "${SUPPORT_GATE_MIN_DELTA_WAS_SET}" ]]; then SUPPORT_GATE_MIN_DELTA="1e-8"; fi
        if [[ -z "${SUPPORT_GATE_ROOTZONE_TOLERANCE_WAS_SET}" ]]; then SUPPORT_GATE_ROOTZONE_TOLERANCE="1e-8"; fi
        if [[ -z "${FREEZE_MONTHLY_GAIN_WAS_SET}" ]]; then FREEZE_MONTHLY_GAIN="1"; fi
        REQUIRE_SAFE_POLICY_FOR_K="0"
    elif [[ "${K}" != "0" && "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v10_support_pool_nested" && "${AUDIT_IDENTITY_FOR_K}" != "1" ]]; then
        if [[ -z "${ADAPT_SCOPE_WAS_SET}" ]]; then
            ADAPT_SCOPE_FOR_K="coeff_only"
        fi
        if [[ -z "${STAGE3_POSTERIOR_POLICY_WAS_SET}" ]]; then
            STAGE3_POSTERIOR_POLICY="conservative_coeff_posterior"
        fi
        ADAPTATION_MAX_STEPS="0"
        ADAPTATION_LR="${ADAPT_LR_OVERRIDE:-0.0}"
        if [[ -z "${ADAPT_ANCHOR_ALPHA_OVERRIDE}" ]]; then
            if [[ "${K}" == "4" ]]; then
                ANCHOR_ALPHA="0.30"
            else
                ANCHOR_ALPHA="0.45"
            fi
        fi
        if [[ -z "${ADAPT_SOLVER_WAS_SET}" ]]; then ADAPT_SOLVER_FOR_K="ridge_coeff"; fi
        if [[ -z "${SUPPORT_GATE_WAS_SET}" ]]; then SUPPORT_GATE_FOR_K="auto"; else SUPPORT_GATE_FOR_K="${SUPPORT_GATE}"; fi
        if [[ -z "${SUPPORT_LOSS_REDUCTION_WAS_SET}" ]]; then SUPPORT_LOSS_REDUCTION_FOR_K="cycle_balanced"; else SUPPORT_LOSS_REDUCTION_FOR_K="${SUPPORT_LOSS_REDUCTION}"; fi
        if [[ -z "${SUPPORT_GATE_MIN_DELTA_WAS_SET}" ]]; then
            if [[ "${K}" == "4" ]]; then
                SUPPORT_GATE_MIN_DELTA="1e-4"
            else
                SUPPORT_GATE_MIN_DELTA="1e-3"
            fi
        fi
        if [[ -z "${SUPPORT_GATE_ROOTZONE_TOLERANCE_WAS_SET}" ]]; then SUPPORT_GATE_ROOTZONE_TOLERANCE="0.0"; fi
        if [[ -z "${FREEZE_MONTHLY_GAIN_WAS_SET}" ]]; then FREEZE_MONTHLY_GAIN="1"; fi
        if [[ -z "${RIDGE_STANDARDIZE_FEATURES_WAS_SET}" ]]; then RIDGE_STANDARDIZE_FEATURES_FOR_K="1"; fi
        if [[ -z "${RIDGE_WEIGHTING_WAS_SET}" ]]; then
            if [[ "${K}" == "4" ]]; then
                RIDGE_WEIGHTING_FOR_K="cycle_variable_balanced_huber"
            else
                RIDGE_WEIGHTING_FOR_K="global_pixel_l2"
            fi
        fi
        if [[ -z "${RIDGE_LAMBDA_WAS_SET}" ]]; then
            if [[ "${K}" == "4" ]]; then
                RIDGE_LAMBDA_FOR_K="4.0"
            else
                RIDGE_LAMBDA_FOR_K="2.0"
            fi
        fi
        if [[ -z "${RIDGE_TRUST_REGION_RADIUS_WAS_SET}" ]]; then
            if [[ "${K}" == "4" ]]; then
                RIDGE_TRUST_REGION_RADIUS_FOR_K="0.10"
            else
                RIDGE_TRUST_REGION_RADIUS_FOR_K="0.18"
            fi
        fi
        if [[ -z "${RIDGE_CLIP_COEFF_NORM_WAS_SET}" ]]; then
            RIDGE_CLIP_COEFF_NORM_FOR_K="${RIDGE_TRUST_REGION_RADIUS_FOR_K}"
        fi
        REQUIRE_SAFE_POLICY_FOR_K="0"
    elif [[ "${K}" != "0" && "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested" && "${AUDIT_IDENTITY_FOR_K}" != "1" ]]; then
        if [[ -z "${ADAPT_SCOPE_WAS_SET}" ]]; then
            ADAPT_SCOPE_FOR_K="coeff_only"
        fi
        if [[ -z "${STAGE3_POSTERIOR_POLICY_WAS_SET}" ]]; then
            STAGE3_POSTERIOR_POLICY="conservative_coeff_posterior"
        fi
        ADAPTATION_MAX_STEPS="0"
        ADAPTATION_LR="${ADAPT_LR_OVERRIDE:-0.0}"
        if [[ -z "${ADAPT_ANCHOR_ALPHA_OVERRIDE}" ]]; then
            if [[ "${K}" == "4" ]]; then
                ANCHOR_ALPHA="0.20"
            else
                ANCHOR_ALPHA="0.30"
            fi
        fi
        if [[ -z "${ADAPT_SOLVER_WAS_SET}" ]]; then ADAPT_SOLVER_FOR_K="ridge_coeff"; fi
        if [[ -z "${SUPPORT_GATE_WAS_SET}" ]]; then SUPPORT_GATE_FOR_K="auto"; else SUPPORT_GATE_FOR_K="${SUPPORT_GATE}"; fi
        if [[ -z "${SUPPORT_LOSS_REDUCTION_WAS_SET}" ]]; then SUPPORT_LOSS_REDUCTION_FOR_K="cycle_balanced"; else SUPPORT_LOSS_REDUCTION_FOR_K="${SUPPORT_LOSS_REDUCTION}"; fi
        if [[ -z "${SUPPORT_GATE_MIN_DELTA_WAS_SET}" ]]; then
            if [[ "${K}" == "4" ]]; then
                SUPPORT_GATE_MIN_DELTA="5e-6"
            else
                SUPPORT_GATE_MIN_DELTA="2e-6"
            fi
        fi
        if [[ -z "${SUPPORT_GATE_ROOTZONE_TOLERANCE_WAS_SET}" ]]; then SUPPORT_GATE_ROOTZONE_TOLERANCE="0.0"; fi
        if [[ -z "${FREEZE_MONTHLY_GAIN_WAS_SET}" ]]; then FREEZE_MONTHLY_GAIN="1"; fi
        if [[ -z "${RIDGE_STANDARDIZE_FEATURES_WAS_SET}" ]]; then RIDGE_STANDARDIZE_FEATURES_FOR_K="1"; fi
        if [[ -z "${RIDGE_WEIGHTING_WAS_SET}" ]]; then
            if [[ "${K}" == "4" ]]; then
                RIDGE_WEIGHTING_FOR_K="cycle_variable_balanced_huber"
            else
                RIDGE_WEIGHTING_FOR_K="global_pixel_l2"
            fi
        fi
        if [[ -z "${RIDGE_LAMBDA_WAS_SET}" ]]; then
            if [[ "${K}" == "4" ]]; then
                RIDGE_LAMBDA_FOR_K="8.0"
            else
                RIDGE_LAMBDA_FOR_K="6.0"
            fi
        fi
        if [[ -z "${RIDGE_TRUST_REGION_RADIUS_WAS_SET}" ]]; then
            if [[ "${K}" == "4" ]]; then
                RIDGE_TRUST_REGION_RADIUS_FOR_K="0.06"
            else
                RIDGE_TRUST_REGION_RADIUS_FOR_K="0.10"
            fi
        fi
        if [[ -z "${RIDGE_CLIP_COEFF_NORM_WAS_SET}" ]]; then
            RIDGE_CLIP_COEFF_NORM_FOR_K="${RIDGE_TRUST_REGION_RADIUS_FOR_K}"
        fi
        REQUIRE_SAFE_POLICY_FOR_K="0"
    elif [[ "${K}" != "0" && ( "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v6_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v7_balanced_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v8_hybrid_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v9_guarded_nested" ) && "${AUDIT_IDENTITY_FOR_K}" != "1" ]]; then
        if [[ -z "${ADAPT_SCOPE_WAS_SET}" ]]; then
            ADAPT_SCOPE_FOR_K="coeff_only"
        fi
        if [[ -z "${STAGE3_POSTERIOR_POLICY_WAS_SET}" ]]; then
            STAGE3_POSTERIOR_POLICY="conservative_coeff_posterior"
        fi
        ADAPTATION_MAX_STEPS="0"
        ADAPTATION_LR="${ADAPT_LR_OVERRIDE:-0.0}"
        if [[ -z "${ADAPT_ANCHOR_ALPHA_OVERRIDE}" ]]; then
            if [[ "${K}" == "4" && "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
                ANCHOR_ALPHA="0.30"
            elif [[ "${K}" == "4" ]]; then
                ANCHOR_ALPHA="0.40"
            elif [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
                ANCHOR_ALPHA="0.45"
            else
                ANCHOR_ALPHA="0.60"
            fi
        fi
        if [[ -z "${ADAPT_SOLVER_WAS_SET}" ]]; then ADAPT_SOLVER_FOR_K="ridge_coeff"; fi
        if [[ -z "${SUPPORT_GATE_WAS_SET}" ]]; then SUPPORT_GATE_FOR_K="auto"; else SUPPORT_GATE_FOR_K="${SUPPORT_GATE}"; fi
        if [[ -z "${SUPPORT_LOSS_REDUCTION_WAS_SET}" ]]; then SUPPORT_LOSS_REDUCTION_FOR_K="cycle_balanced"; else SUPPORT_LOSS_REDUCTION_FOR_K="${SUPPORT_LOSS_REDUCTION}"; fi
        if [[ -z "${SUPPORT_GATE_MIN_DELTA_WAS_SET}" ]]; then
            if [[ "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v9_guarded_nested" && "${K}" == "12" && "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
                SUPPORT_GATE_MIN_DELTA="2e-3"
            elif [[ "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v9_guarded_nested" && "${K}" == "12" ]]; then
                SUPPORT_GATE_MIN_DELTA="3e-3"
            else
                SUPPORT_GATE_MIN_DELTA="1e-8"
            fi
        fi
        if [[ -z "${SUPPORT_GATE_ROOTZONE_TOLERANCE_WAS_SET}" ]]; then SUPPORT_GATE_ROOTZONE_TOLERANCE="1e-8"; fi
        if [[ -z "${FREEZE_MONTHLY_GAIN_WAS_SET}" ]]; then FREEZE_MONTHLY_GAIN="1"; fi
        if [[ -z "${RIDGE_STANDARDIZE_FEATURES_WAS_SET}" ]]; then RIDGE_STANDARDIZE_FEATURES_FOR_K="1"; fi
        if [[ "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v7_balanced_nested" && -z "${RIDGE_WEIGHTING_WAS_SET}" ]]; then RIDGE_WEIGHTING_FOR_K="cycle_variable_balanced_huber"; fi
        if [[ ( "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v8_hybrid_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v9_guarded_nested" ) && -z "${RIDGE_WEIGHTING_WAS_SET}" ]]; then
            if [[ "${K}" == "4" ]]; then
                RIDGE_WEIGHTING_FOR_K="cycle_variable_balanced_huber"
            else
                RIDGE_WEIGHTING_FOR_K="global_pixel_l2"
            fi
        fi
        if [[ -z "${RIDGE_LAMBDA_WAS_SET}" ]]; then
            if [[ "${K}" == "4" && "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
                RIDGE_LAMBDA_FOR_K="4.0"
            elif [[ "${K}" == "4" ]]; then
                RIDGE_LAMBDA_FOR_K="2.0"
            elif [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
                RIDGE_LAMBDA_FOR_K="2.0"
            else
                RIDGE_LAMBDA_FOR_K="1.0"
            fi
        fi
        if [[ -z "${RIDGE_TRUST_REGION_RADIUS_WAS_SET}" ]]; then
            if [[ "${K}" == "4" && "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
                RIDGE_TRUST_REGION_RADIUS_FOR_K="0.10"
            elif [[ "${K}" == "4" ]]; then
                RIDGE_TRUST_REGION_RADIUS_FOR_K="0.18"
            elif [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
                RIDGE_TRUST_REGION_RADIUS_FOR_K="0.18"
            else
                RIDGE_TRUST_REGION_RADIUS_FOR_K="0.28"
            fi
        fi
        if [[ -z "${RIDGE_CLIP_COEFF_NORM_WAS_SET}" ]]; then
            if [[ "${K}" == "4" && "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
                RIDGE_CLIP_COEFF_NORM_FOR_K="0.15"
            elif [[ "${K}" == "4" ]]; then
                RIDGE_CLIP_COEFF_NORM_FOR_K="0.25"
            elif [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
                RIDGE_CLIP_COEFF_NORM_FOR_K="0.25"
            else
                RIDGE_CLIP_COEFF_NORM_FOR_K="0.40"
            fi
        fi
        REQUIRE_SAFE_POLICY_FOR_K="0"
    elif [[ "${K}" != "0" && ( "${STAGE3_KSHOT_MODE}" == "diagnostic_conservative_kshot_v3" || "${STAGE3_KSHOT_MODE}" == "diagnostic_safe_operator_v5_nested" ) && "${AUDIT_IDENTITY_FOR_K}" != "1" ]]; then
        if [[ -z "${ADAPT_SCOPE_WAS_SET}" ]]; then
            ADAPT_SCOPE_FOR_K="coeff_only"
        fi
        if [[ -z "${STAGE3_POSTERIOR_POLICY_WAS_SET}" ]]; then
            STAGE3_POSTERIOR_POLICY="conservative_coeff_posterior"
        fi
        if [[ -z "${ADAPT_MAX_STEPS_OVERRIDE}" ]]; then
            if [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" && "${K}" == "4" ]]; then
                ADAPTATION_MAX_STEPS="20"
            elif [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" && "${K}" == "12" ]]; then
                ADAPTATION_MAX_STEPS="40"
            elif [[ "${K}" == "4" ]]; then
                ADAPTATION_MAX_STEPS="40"
            else
                ADAPTATION_MAX_STEPS="80"
            fi
        fi
        ADAPTATION_LR="${ADAPT_LR_OVERRIDE:-${LR:-3e-4}}"
        if [[ -z "${ADAPT_ANCHOR_ALPHA_OVERRIDE}" ]]; then
            if [[ "${K}" == "4" && "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
                ANCHOR_ALPHA="0.25"
            elif [[ "${K}" == "4" ]]; then
                ANCHOR_ALPHA="0.35"
            elif [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
                ANCHOR_ALPHA="0.35"
            else
                ANCHOR_ALPHA="0.50"
            fi
        fi
        if [[ -z "${SUPPORT_GATE_WAS_SET}" ]]; then SUPPORT_GATE_FOR_K="auto"; else SUPPORT_GATE_FOR_K="${SUPPORT_GATE}"; fi
        if [[ -z "${SUPPORT_LOSS_REDUCTION_WAS_SET}" ]]; then SUPPORT_LOSS_REDUCTION_FOR_K="cycle_balanced"; else SUPPORT_LOSS_REDUCTION_FOR_K="${SUPPORT_LOSS_REDUCTION}"; fi
        if [[ -z "${SUPPORT_GATE_MIN_DELTA_WAS_SET}" ]]; then SUPPORT_GATE_MIN_DELTA="1e-8"; fi
        if [[ -z "${SUPPORT_GATE_ROOTZONE_TOLERANCE_WAS_SET}" ]]; then SUPPORT_GATE_ROOTZONE_TOLERANCE="1e-8"; fi
        if [[ -z "${FREEZE_MONTHLY_GAIN_WAS_SET}" ]]; then FREEZE_MONTHLY_GAIN="1"; fi
        REQUIRE_SAFE_POLICY_FOR_K="0"
    else
        SUPPORT_GATE_FOR_K="${SUPPORT_GATE}"
        SUPPORT_LOSS_REDUCTION_FOR_K="${SUPPORT_LOSS_REDUCTION}"
    fi
    REQUIRE_SAFE_POLICY_FOR_K="0"
    if [[ "${K}" != "0" && ( "${STAGE3_KSHOT_MODE}" == "diagnostic_direct_kshot" || "${STAGE3_KSHOT_MODE}" == "diagnostic_direct_kshot_v2" || "${STAGE3_KSHOT_MODE}" == "diagnostic_conservative_kshot_v3" || "${STAGE3_KSHOT_MODE}" == "diagnostic_safe_operator_v5_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_finetune_support_gain_v14_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v6_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v7_balanced_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v8_hybrid_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v9_guarded_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v10_support_pool_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v1" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v2" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v3_stable" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v4_nested_stable" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v12_nested_cv" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v13_k12_aggressive_calibration_pool" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_affine_v1_nested" ) ]]; then
        REQUIRE_SAFE_POLICY_FOR_K="0"
    elif [[ "${K}" != "0" && -n "${SAFE_POLICY_JSON}" ]]; then
        REQUIRE_SAFE_POLICY_FOR_K="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT}"
    elif [[ "${K}" != "0" && "${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT}" == "1" ]]; then
        echo "ERROR: paper-facing K=${K} requires SAFE_POLICY_JSON from source-side episode calibration." >&2
        echo "Set SAFE_POLICY_JSON=/path/to/safe_policy.json, or set STAGE3_KSHOT_MODE=diagnostic_direct_kshot for non-paper diagnostics." >&2
        exit 2
    elif [[ "${K}" != "0" && "${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT,,}" == "true" ]]; then
        echo "ERROR: paper-facing K=${K} requires SAFE_POLICY_JSON from source-side episode calibration." >&2
        echo "Set SAFE_POLICY_JSON=/path/to/safe_policy.json, or set STAGE3_KSHOT_MODE=diagnostic_direct_kshot for non-paper diagnostics." >&2
        exit 2
    fi

    K_DIR="${OUTPUT_BASE}/K${K}"
    ADAPT_DIR="${K_DIR}/adapt"
    EVAL_DIR="${K_DIR}/eval"
    mkdir -p "${ADAPT_DIR}" "${EVAL_DIR}"
    SPLITS_JSON_FOR_K="${SPLITS_JSON}"
    if [[ "${K}" == "12" && -n "${NESTED_K12_SPLITS_JSON}" ]]; then
        SPLITS_JSON_FOR_K="${NESTED_K12_SPLITS_JSON}"
    fi
    K4_REFERENCE_CHECKPOINT_FOR_K="${K4_REFERENCE_CHECKPOINT:-}"
    if [[ "${K}" == "12" && ( "${STAGE3_KSHOT_MODE}" == "diagnostic_safe_operator_v5_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_finetune_support_gain_v14_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v6_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v7_balanced_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v8_hybrid_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v9_guarded_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v10_support_pool_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v12_nested_cv" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v13_k12_aggressive_calibration_pool" ) && -z "${K4_REFERENCE_CHECKPOINT_FOR_K}" ]]; then
        K4_REFERENCE_CHECKPOINT_FOR_K="${OUTPUT_BASE}/K4/adapt/checkpoints/checkpoint_final_preregistered.pt"
        if [[ ! -f "${K4_REFERENCE_CHECKPOINT_FOR_K}" ]]; then
            echo "ERROR: ${STAGE3_KSHOT_MODE} K12 requires a K4 reference checkpoint." >&2
            echo "Run K_LIST with K4 before K12, or set K4_REFERENCE_CHECKPOINT=/path/to/checkpoint_final_preregistered.pt." >&2
            exit 2
        fi
    fi

    echo ""
    echo ">>> [K=${K}] Adaptation: ${ADAPTATION_SETTING}"
    echo "    splits_json_for_k=${SPLITS_JSON_FOR_K}"
    echo "    k4_reference_checkpoint=${K4_REFERENCE_CHECKPOINT_FOR_K:-<none>}"
    echo "    resolved_policy: solver=${ADAPT_SOLVER_FOR_K} steps=${ADAPTATION_MAX_STEPS} lr=${ADAPTATION_LR} anchor_alpha=${ANCHOR_ALPHA} mix_rho=${ADAPT_MIX_RHO_FOR_K}"
    echo "    resolved_ridge: lambda=${RIDGE_LAMBDA_FOR_K} clip_coeff_norm=${RIDGE_CLIP_COEFF_NORM_FOR_K} trust_region_radius=${RIDGE_TRUST_REGION_RADIUS_FOR_K} standardize_features=${RIDGE_STANDARDIZE_FEATURES_FOR_K} weighting=${RIDGE_WEIGHTING_FOR_K}"
    ADAPT_RECIPE="${ADAPT_RECIPE}" ADAPT_SCOPE="${ADAPT_SCOPE_FOR_K}" ADAPT_SOLVER="${ADAPT_SOLVER_FOR_K}" RIDGE_LAMBDA="${RIDGE_LAMBDA_FOR_K}" RIDGE_CLIP_COEFF_NORM="${RIDGE_CLIP_COEFF_NORM_FOR_K}" RIDGE_TRUST_REGION_RADIUS="${RIDGE_TRUST_REGION_RADIUS_FOR_K}" RIDGE_MAX_FEATURE_PIXELS="${RIDGE_MAX_FEATURE_PIXELS}" RIDGE_STANDARDIZE_FEATURES="${RIDGE_STANDARDIZE_FEATURES_FOR_K}" RIDGE_WEIGHTING="${RIDGE_WEIGHTING_FOR_K}" STAGE3_KSHOT_MODE="${STAGE3_KSHOT_MODE}" DIAGNOSTIC_KSHOT_STRENGTH="${DIAGNOSTIC_KSHOT_STRENGTH}" STAGE3_POSTERIOR_POLICY="${STAGE3_POSTERIOR_POLICY}" SUPPORT_GATE="${SUPPORT_GATE_FOR_K}" SUPPORT_GATE_MIN_DELTA="${SUPPORT_GATE_MIN_DELTA}" SUPPORT_GATE_ROOTZONE_TOLERANCE="${SUPPORT_GATE_ROOTZONE_TOLERANCE}" FREEZE_MONTHLY_GAIN="${FREEZE_MONTHLY_GAIN}" TARGET_CONTEXT_MAX_SAMPLES="${TARGET_CONTEXT_MAX_SAMPLES}" STAGE3_CONTEXT_TTA="${STAGE3_CONTEXT_TTA}" CONTEXT_TTA_RESIDUAL_SCALE="${CONTEXT_TTA_RESIDUAL_SCALE}" CONTEXT_TTA_RESIDUAL_CLIP_L2="${CONTEXT_TTA_RESIDUAL_CLIP_L2}" SAFE_POLICY_JSON="${SAFE_POLICY_JSON}" REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_FOR_K}" SCHEDULE_LABEL="${SCHEDULE_LABEL}" SUPPORT_LOSS_REDUCTION="${SUPPORT_LOSS_REDUCTION_FOR_K}" AUDIT_IDENTITY="${AUDIT_IDENTITY_FOR_K}" AUDIT_IDENTITY_TOLERANCE="${AUDIT_IDENTITY_TOLERANCE}" ADAPT_ANCHOR_ALPHA="${ANCHOR_ALPHA}" ADAPT_MIX_RHO="${ADAPT_MIX_RHO_FOR_K}" ADAPT_LR="${ADAPTATION_LR}" ADAPT_MAX_STEPS="${ADAPTATION_MAX_STEPS}" ADAPT_WEIGHT_DECAY="${ADAPT_WEIGHT_DECAY_OVERRIDE}" ADAPT_GRAD_CLIP="${ADAPT_GRAD_CLIP_OVERRIDE}" ADAPT_LAMBDA_PRIOR="${ADAPT_LAMBDA_PRIOR:-${LAMBDA_PRIOR:-}}" ADAPT_LAMBDA_LATENT="${ADAPT_LAMBDA_LATENT:-${LAMBDA_LATENT:-}}" ADAPT_LAMBDA_GAIN="${ADAPT_LAMBDA_GAIN:-${LAMBDA_GAIN:-}}" ADAPT_LAMBDA_GAIN_SMOOTH="${ADAPT_LAMBDA_GAIN_SMOOTH:-${LAMBDA_GAIN_SMOOTH:-}}" ADAPT_LAMBDA_ANALYSIS="${ADAPT_LAMBDA_ANALYSIS:-${LAMBDA_ANALYSIS:-}}" K4_REFERENCE_CHECKPOINT="${K4_REFERENCE_CHECKPOINT_FOR_K}" BATCH_SIZE="${ADAPT_BATCH_SIZE}" SPLITS_JSON="${SPLITS_JSON_FOR_K}" OUTPUT_DIR="${ADAPT_DIR}" bash run/phase5_hyperda_zero_few_shot.sh \
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
    ADAPT_MIX_RHO_FOR_EVAL="${ADAPT_MIX_RHO_FOR_K}"
    if [[ "${K}" != "0" && ( "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v10_support_pool_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested" ) ]]; then
        CHECKPOINT_MIX_RHO="$(python3 - "${ADAPTED_CHECKPOINT}" <<'PYTHON_SCRIPT'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
config = checkpoint.get("config", {})
print(config.get("adapt_mix_rho", ""))
PYTHON_SCRIPT
)"
        if [[ -n "${CHECKPOINT_MIX_RHO}" ]]; then
            ADAPT_MIX_RHO_FOR_EVAL="${CHECKPOINT_MIX_RHO}"
        fi
    elif [[ "${K}" == "12" && ( "${STAGE3_KSHOT_MODE}" == "diagnostic_safe_operator_v5_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_finetune_support_gain_v14_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v6_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v7_balanced_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v8_hybrid_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v9_guarded_nested" ) ]]; then
        CHECKPOINT_MIX_RHO="$(python3 - "${ADAPTED_CHECKPOINT}" <<'PYTHON_SCRIPT'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
config = checkpoint.get("config", {})
decision = str(config.get("stage3_posterior_decision", ""))
if decision == "fallback_to_k4_reference":
    print(config.get("adapt_mix_rho", ""))
else:
    print("")
PYTHON_SCRIPT
)"
        if [[ -n "${CHECKPOINT_MIX_RHO}" ]]; then
            ADAPT_MIX_RHO_FOR_EVAL="${CHECKPOINT_MIX_RHO}"
        fi
    fi

    RAW_ADAPTED_EVAL_ARGS=()
    if [[ "${EVAL_RAW_ADAPTED_BEFORE_MIX}" == "1" ]]; then
        RAW_ADAPTED_EVAL_ARGS+=(--eval_raw_adapted_before_mix)
    fi
    SUPPORT_GAIN_EVAL_ARGS=()
    if [[ "${K}" != "0" && "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v1" ]]; then
        SUPPORT_GAIN_EVAL_ARGS+=(--target_train_residual_gain_calibration --allow_legacy_target_label_calibration)
    fi

    echo ""
    echo ">>> [K=${K}] Evaluation on target_eval (adapt_mix_rho=${ADAPT_MIX_RHO_FOR_EVAL})"
    PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \
        --checkpoint "${ADAPTED_CHECKPOINT}" \
        --target_region "${TARGET_REGION}" \
        --adaptation_setting "${ADAPTATION_SETTING}" \
        --K "${K}" \
        --seed "${SEED}" \
        --split_type target_eval \
        --splits_json "${SPLITS_JSON_FOR_K}" \
        --predictor_type hyperda_target_adapt \
        --device cuda \
        --output_dir "${EVAL_DIR}" \
        --max_samples "${EVAL_MAX_SAMPLES}" \
        --batch_size "${EVAL_BATCH_SIZE}" \
        --adapt_mix_rho "${ADAPT_MIX_RHO_FOR_EVAL}" \
        --output_level "${EVAL_OUTPUT_LEVEL}" \
        "${RAW_ADAPTED_EVAL_ARGS[@]}" \
        "${SUPPORT_GAIN_EVAL_ARGS[@]}" \
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
requested_k_values = sys.argv[3].split()
requested_seed = int(sys.argv[4])
settings = {
    "0": "zero_shot_context",
    "4": "few_shot_k4",
    "12": "few_shot_k12",
}
k_values = []
seen_k = set()
for item in requested_k_values:
    if item in settings and item not in seen_k:
        seen_k.add(item)
for child in output_base.glob("K*"):
    suffix = child.name[1:] if child.name.startswith("K") else ""
    if child.is_dir() and suffix in settings:
        seen_k.add(suffix)
k_values = [k for k in ("0", "4", "12") if k in seen_k]

def fmt(value):
    if isinstance(value, bool):
        return "true" if value else "false"
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

def nested_present(*mappings_and_paths):
    for mapping, path in mappings_and_paths:
        value = mapping
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if value is not None:
            return value
    return None

def support_affine_coeff(metadata, variable, key):
    affine = metadata.get("support_affine_calibration", {})
    if not isinstance(affine, dict):
        return None
    coeffs = affine.get("support_affine_coefficients", {})
    if not isinstance(coeffs, dict):
        return None
    block = coeffs.get(variable, {})
    if not isinstance(block, dict):
        return None
    return block.get(key)

def stage3_decision_from_artifacts(summary, adapt_metadata):
    return nested_present(
        (adapt_metadata, ("stage3_posterior_decision",)),
        (adapt_metadata, ("stage3_posterior_state", "stage3_posterior_decision")),
        (summary, ("stage3_protocol", "stage3_posterior_decision")),
    )

def stage3_overview_status(k, summary_status, summary, adapt_metadata):
    if summary_status != "ok":
        return summary_status
    decision = stage3_decision_from_artifacts(summary, adapt_metadata)
    paper_facing = paper_facing_from_artifacts(k, summary, adapt_metadata)
    try:
        k_int = int(k)
    except Exception:
        k_int = -1
    if k_int == 0:
        return "ok"
    if decision == "rejected_to_k0_anchor":
        return "rejected_to_k0_anchor"
    if decision == "no_update":
        return "source_policy_no_update_k0_equivalent"
    if decision == "accepted" and paper_facing is False:
        return "diagnostic_accepted"
    return "ok"

def paper_facing_from_artifacts(k, summary, adapt_metadata):
    decision = stage3_decision_from_artifacts(summary, adapt_metadata)
    try:
        k_int = int(k)
    except Exception:
        k_int = -1
    if k_int > 0 and decision in {"rejected_to_k0_anchor", "no_update"}:
        return False
    return nested_present(
        (adapt_metadata, ("paper_facing_run",)),
        (summary, ("stage3_protocol", "paper_facing_run")),
    )

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
    nested_dir = output_base / "nested_support"
    existing_nested_support_path = nested_dir / f"{target_region}_s{seed}_K12_nested_support.json"
    existing_nested_split_path = nested_dir / f"{target_region}_s{seed}_K12_nested_splits.json"
    if existing_nested_support_path.exists() and existing_nested_split_path.exists():
        try:
            existing_payload = json.loads(existing_nested_support_path.read_text(encoding="utf-8"))
        except Exception:
            existing_payload = {}
        if existing_payload.get("support_nesting_policy") == "run_local_k12_nested_k4_plus_8_original_k12_nonduplicate":
            result["nested_manifest_path"] = str(existing_nested_split_path)
            result["nested_support_artifact_path"] = str(existing_nested_support_path)
            result["k12_nested_support_dates"] = date_strs(existing_payload.get("k12_nested_support_dates", []) or [])
            result["k12_nested_support_dates_hash"] = existing_payload.get(
                "nested_support_dates_hash",
                existing_payload.get("k12_nested_support_dates_hash"),
            )
            result["support_nesting_policy"] = existing_payload.get("support_nesting_policy")
            support_diag_path = output_base / "support_set_diagnostic.json"
            support_diag_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            return result
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
            f"K_LIST=\"12\" ADAPT_SCOPE=safe_operator SPLITS_JSON=\"{nested_split_path}\" "
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
    region_eval_dir = output_base / f"K{k}" / "eval" / target_region
    summary_path = region_eval_dir / "summary.json"
    adapt_metadata_path = output_base / f"K{k}" / "adapt" / "metadata.json"
    metrics_long_csv_path = region_eval_dir / "metrics_long.csv"
    metrics_long_gz_path = region_eval_dir / "metrics_long.csv.gz"
    metrics_long_path = (
        metrics_long_csv_path
        if metrics_long_csv_path.exists()
        else metrics_long_gz_path
        if metrics_long_gz_path.exists()
        else metrics_long_csv_path
    )
    metrics_by_region_path = region_eval_dir / "metrics_by_region.csv"
    metrics_by_season_path = region_eval_dir / "metrics_by_season.csv"
    checkpoint_path = output_base / f"K{k}" / "adapt" / "checkpoints" / "checkpoint_final_preregistered.pt"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        summary_status = "ok"
    else:
        summary = {}
        summary_status = "missing_summary"
    if adapt_metadata_path.exists():
        with open(adapt_metadata_path, encoding="utf-8") as f:
            adapt_metadata = json.load(f)
    else:
        adapt_metadata = {}
    status = stage3_overview_status(k, summary_status, summary, adapt_metadata)
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
    row = {
        "target_region": target_region,
        "K": k,
        "adaptation_setting": settings.get(k, ""),
        "stage3_kshot_mode": adapt_metadata.get("stage3_kshot_mode", (summary.get("stage3_protocol", {}) or {}).get("stage3_kshot_mode")),
        "adapt_recipe": adapt_metadata.get("adapt_recipe"),
        "adapt_scope": adapt_metadata.get("adapt_scope"),
        "adapt_solver": adapt_metadata.get("adapt_solver"),
        "support_loss_reduction": adapt_metadata.get("support_loss_reduction"),
        "adapt_mix_rho": summary.get("adapt_mix_rho"),
        "audit_identity": adapt_metadata.get("audit_identity"),
        "status": status,
        "method": summary.get("method"),
        "paper_facing_run": paper_facing_from_artifacts(k, summary, adapt_metadata),
        "diagnostic_run_reason": adapt_metadata.get("diagnostic_run_reason"),
        "policy_source": adapt_metadata.get("policy_source"),
        "safe_policy_json_sha256": first_present(adapt_metadata, ["safe_policy_json_sha256", "safe_policy_hash"]),
        "source_policy_candidate_id": adapt_metadata.get("source_policy_candidate_id"),
        "stage3_posterior_decision": stage3_decision_from_artifacts(summary, adapt_metadata),
        "stage3_acceptance_basis": nested_present(
            (adapt_metadata, ("stage3_acceptance_basis",)),
            (adapt_metadata, ("stage3_posterior_state", "stage3_acceptance_basis")),
            (summary, ("stage3_protocol", "stage3_acceptance_basis")),
        ),
        "support_gate_status": nested_present(
            (adapt_metadata, ("support_gate_status",)),
            (adapt_metadata, ("stage3_posterior_state", "support_gate_status")),
            (summary, ("stage3_protocol", "support_gate_status")),
        ),
        "support_only_gate_status": nested_present(
            (adapt_metadata, ("support_only_gate_status",)),
            (adapt_metadata, ("stage3_posterior_state", "support_only_gate_status")),
            (summary, ("stage3_protocol", "support_only_gate_status")),
        ),
        "support_gate_reject_reason": json.dumps(
            nested_present(
                (adapt_metadata, ("support_gate_reject_reason",)),
                (adapt_metadata, ("stage3_posterior_state", "support_gate_reject_reason")),
                (summary, ("stage3_protocol", "support_gate_reject_reason")),
            )
            or [],
            sort_keys=True,
        ),
        "k0_anchor_state_hash": nested_present(
            (adapt_metadata, ("k0_anchor_state_hash",)),
            (adapt_metadata, ("stage3_posterior_state", "k0_anchor_state_hash")),
        ),
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
        "staged_source_checkpoint_sha256": adapt_metadata.get("staged_source_checkpoint_sha256"),
        "source_stage_checkpoint_provenance": adapt_metadata.get("source_stage_checkpoint_provenance"),
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
        "target_parameter_l2_drift_total": drift.get("total"),
        "target_parameter_l2_drift_target_prompt": drift.get("target_prompt"),
        "target_parameter_l2_drift_adapter_coeff_bottleneck": drift.get("adapter_coeff_bottleneck"),
        "target_parameter_l2_drift_adapter_coeff_dec2": drift.get("adapter_coeff_dec2"),
        "target_parameter_l2_drift_adapter_coeff_dec1": drift.get("adapter_coeff_dec1"),
        "target_parameter_l2_drift_monthly_gain": first_present(drift, ["monthly_gain", "monthly_residual_gain"]),
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
        "support_nesting_policy": nested_present(
            (adapt_metadata, ("support_nesting_policy",)),
            (adapt_metadata, ("support_gain_calibration", "support_nesting_policy")),
            (summary, ("stage3_protocol", "support_nesting_policy")),
        ),
        "nested_support_dates_hash": nested_present(
            (adapt_metadata, ("nested_support_dates_hash",)),
            (adapt_metadata, ("support_gain_calibration", "nested_support_dates_hash")),
            (summary, ("stage3_protocol", "nested_support_dates_hash")),
        ),
        "n_samples_evaluated": summary.get("n_samples_evaluated"),
        "n_metric_rows": summary.get("n_metric_rows"),
        "prediction_content_hash": summary.get("prediction_content_hash"),
        "zero_shot_prediction_content_hash": summary.get("zero_shot_prediction_content_hash"),
        "adapted_pre_mix_prediction_content_hash": summary.get("adapted_pre_mix_prediction_content_hash"),
        "raw_adapted_prediction_content_hash": summary.get("raw_adapted_prediction_content_hash"),
        "post_gate_prediction_content_hash": summary.get("post_gate_prediction_content_hash"),
        "final_mixed_prediction_content_hash": summary.get("final_mixed_prediction_content_hash"),
        "raw_to_k0_mean_abs_delta": summary.get("raw_to_k0_mean_abs_delta"),
        "post_gate_to_k0_mean_abs_delta": summary.get("post_gate_to_k0_mean_abs_delta"),
        "final_mix_to_k0_mean_abs_delta": summary.get("final_mix_to_k0_mean_abs_delta"),
        "mix_mean_abs_change_from_k0": summary.get("mix_mean_abs_change_from_k0"),
        "mix_max_abs_change_from_k0": summary.get("mix_max_abs_change_from_k0"),
        "mix_mean_abs_change_from_adapted": summary.get("mix_mean_abs_change_from_adapted"),
        "mix_max_abs_change_from_adapted": summary.get("mix_max_abs_change_from_adapted"),
        "context_tta_mode": nested_present(
            (summary, ("context_tta_mode",)),
            (adapt_metadata, ("context_tta_mode",)),
            (summary, ("target_prompt", "context_tta_mode")),
            (adapt_metadata, ("target_context_prompt_state", "context_tta_mode")),
        ),
        "context_tta_residual_scale": nested_present(
            (adapt_metadata, ("context_tta_residual_scale",)),
            (adapt_metadata, ("target_context_prompt_state", "context_tta_residual_scale")),
        ),
        "context_tta_residual_clip_l2": nested_present(
            (adapt_metadata, ("context_tta_residual_clip_l2",)),
            (adapt_metadata, ("target_context_prompt_state", "context_tta_residual_clip_l2")),
        ),
        "context_tta_effective": nested_present(
            (summary, ("context_tta_effective",)),
            (adapt_metadata, ("context_tta_effective",)),
            (summary, ("target_prompt", "context_tta_effective")),
            (adapt_metadata, ("target_context_prompt_state", "context_tta_effective")),
        ),
        "context_tta_source_stat_status": nested_present(
            (summary, ("context_tta_source_stat_status",)),
            (adapt_metadata, ("context_tta_source_stat_status",)),
            (summary, ("target_prompt", "context_tta_source_stat_status")),
            (adapt_metadata, ("target_context_prompt_state", "context_tta_source_stat_status")),
        ),
        "prompt_l2_delta_mean": nested_present(
            (summary, ("prompt_l2_delta_mean",)),
            (adapt_metadata, ("prompt_l2_delta_mean",)),
            (summary, ("target_prompt", "prompt_l2_delta_mean")),
            (adapt_metadata, ("target_context_prompt_state", "prompt_l2_delta_mean")),
        ),
        "prediction_delta_vs_no_tta": nested_present(
            (summary, ("prediction_delta_vs_no_tta",)),
            (adapt_metadata, ("prediction_delta_vs_no_tta",)),
        ),
        "support_gain_alpha_surface": nested_present(
            (adapt_metadata, ("support_gain_calibration", "best_alpha_surface")),
            (adapt_metadata, ("residual_gain_alpha_surface",)),
            (summary, ("target_train_residual_gain_calibration", "best_alpha_surface")),
        ),
        "support_gain_alpha_rootzone": nested_present(
            (adapt_metadata, ("support_gain_calibration", "best_alpha_rootzone")),
            (adapt_metadata, ("residual_gain_alpha_rootzone",)),
            (summary, ("target_train_residual_gain_calibration", "best_alpha_rootzone")),
        ),
        "support_gain_selection_score": nested_present(
            (adapt_metadata, ("support_gain_calibration", "selection_score")),
            (summary, ("target_train_residual_gain_calibration", "selection_score")),
        ),
        "support_gain_selection_rule": nested_present(
            (adapt_metadata, ("support_gain_calibration", "selection_rule")),
            (summary, ("target_train_residual_gain_calibration", "selection_rule")),
        ),
        "support_gain_best_alpha_raw": nested_present(
            (adapt_metadata, ("support_gain_calibration", "best_alpha_raw")),
            (summary, ("target_train_residual_gain_calibration", "best_alpha_raw")),
        ),
        "support_gain_stable_candidate_alphas": json.dumps(
            nested_present(
                (adapt_metadata, ("support_gain_calibration", "stable_candidate_alphas")),
                (summary, ("target_train_residual_gain_calibration", "stable_candidate_alphas")),
            )
            or [],
            sort_keys=True,
        ),
        "support_gain_selection_margin": nested_present(
            (adapt_metadata, ("support_gain_calibration", "selection_margin")),
            (summary, ("target_train_residual_gain_calibration", "selection_margin")),
        ),
        "support_gain_stability_tolerance": nested_present(
            (adapt_metadata, ("support_gain_calibration", "stability_tolerance")),
            (summary, ("target_train_residual_gain_calibration", "stability_tolerance")),
        ),
        "support_gain_paired_support_se_capped": nested_present(
            (adapt_metadata, ("support_gain_calibration", "paired_support_se_capped")),
            (summary, ("target_train_residual_gain_calibration", "paired_support_se_capped")),
        ),
        "support_gain_target_eval_usage": nested_present(
            (adapt_metadata, ("support_gain_calibration", "target_eval_usage")),
            (summary, ("target_train_residual_gain_calibration", "target_eval_usage")),
        ),
        "support_affine_surface_a": support_affine_coeff(adapt_metadata, "surface", "a"),
        "support_affine_surface_b": support_affine_coeff(adapt_metadata, "surface", "b"),
        "support_affine_rootzone_a": support_affine_coeff(adapt_metadata, "rootzone", "a"),
        "support_affine_rootzone_b": support_affine_coeff(adapt_metadata, "rootzone", "b"),
        "effective_calibration_dof": nested_present(
            (adapt_metadata, ("support_affine_calibration", "effective_calibration_dof")),
            (summary, ("stage3_protocol", "support_affine_calibration", "effective_calibration_dof")),
        ),
        "support_affine_target_eval_usage": nested_present(
            (adapt_metadata, ("support_affine_calibration", "target_eval_usage")),
            (summary, ("stage3_protocol", "support_affine_calibration", "target_eval_usage")),
        ),
        "k_specific_prediction_changed": False,
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

previous_prediction_hash = ""
for row in rows:
    current_prediction_hash = str(row.get("prediction_content_hash") or "")
    row["k_specific_prediction_changed"] = bool(
        previous_prediction_hash and current_prediction_hash and current_prediction_hash != previous_prediction_hash
    )
    if current_prediction_hash:
        previous_prediction_hash = current_prediction_hash

csv_path = output_base / "overview.csv"
fieldnames = [
    "target_region",
    "K",
    "adaptation_setting",
    "stage3_kshot_mode",
    "adapt_recipe",
    "adapt_scope",
    "adapt_solver",
    "support_loss_reduction",
    "adapt_mix_rho",
    "audit_identity",
    "status",
    "method",
    "paper_facing_run",
    "diagnostic_run_reason",
    "policy_source",
    "safe_policy_json_sha256",
    "source_policy_candidate_id",
    "stage3_posterior_decision",
    "stage3_acceptance_basis",
    "support_gate_status",
    "support_only_gate_status",
    "support_gate_reject_reason",
    "k0_anchor_state_hash",
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
    "staged_source_checkpoint_sha256",
    "source_stage_checkpoint_provenance",
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
    "target_parameter_l2_drift_total",
    "target_parameter_l2_drift_target_prompt",
    "target_parameter_l2_drift_adapter_coeff_bottleneck",
    "target_parameter_l2_drift_adapter_coeff_dec2",
    "target_parameter_l2_drift_adapter_coeff_dec1",
    "target_parameter_l2_drift_monthly_gain",
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
    "support_nesting_policy",
    "nested_support_dates_hash",
    "n_samples_evaluated",
    "n_metric_rows",
    "prediction_content_hash",
    "zero_shot_prediction_content_hash",
    "adapted_pre_mix_prediction_content_hash",
    "raw_adapted_prediction_content_hash",
    "post_gate_prediction_content_hash",
    "final_mixed_prediction_content_hash",
    "raw_to_k0_mean_abs_delta",
    "post_gate_to_k0_mean_abs_delta",
    "final_mix_to_k0_mean_abs_delta",
    "mix_mean_abs_change_from_k0",
    "mix_max_abs_change_from_k0",
    "mix_mean_abs_change_from_adapted",
    "mix_max_abs_change_from_adapted",
    "context_tta_mode",
    "context_tta_residual_scale",
    "context_tta_residual_clip_l2",
    "context_tta_effective",
    "context_tta_source_stat_status",
    "prompt_l2_delta_mean",
    "prediction_delta_vs_no_tta",
    "support_gain_alpha_surface",
    "support_gain_alpha_rootzone",
    "support_gain_selection_score",
    "support_gain_selection_rule",
    "support_gain_best_alpha_raw",
    "support_gain_stable_candidate_alphas",
    "support_gain_selection_margin",
    "support_gain_stability_tolerance",
    "support_gain_paired_support_se_capped",
    "support_gain_target_eval_usage",
    "support_affine_surface_a",
    "support_affine_surface_b",
    "support_affine_rootzone_a",
    "support_affine_rootzone_b",
    "effective_calibration_dof",
    "support_affine_target_eval_usage",
    "k_specific_prediction_changed",
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
    "stage3_kshot_mode",
    "adapt_recipe",
    "adapt_scope",
    "adapt_solver",
    "status",
    "method",
    "paper_facing_run",
    "stage3_posterior_decision",
    "stage3_acceptance_basis",
    "support_gate_status",
    "support_gate_reject_reason",
    "diagnostic_run_reason",
    "policy_source",
    "schedule_label",
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
    "support_final_loss",
    "support_loss_delta",
    "k4_support_subset_of_k12",
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
    "raw_adapted_prediction_content_hash",
    "post_gate_prediction_content_hash",
    "adapted_pre_mix_prediction_content_hash",
    "final_mixed_prediction_content_hash",
    "raw_to_k0_mean_abs_delta",
    "final_mix_to_k0_mean_abs_delta",
    "context_tta_effective",
    "prompt_l2_delta_mean",
    "prediction_delta_vs_no_tta",
    "support_gain_alpha_surface",
    "support_gain_alpha_rootzone",
    "support_gain_selection_score",
    "support_gain_selection_rule",
    "support_gain_best_alpha_raw",
    "support_gain_stable_candidate_alphas",
    "support_gain_selection_margin",
    "support_gain_stability_tolerance",
    "support_gain_paired_support_se_capped",
    "support_affine_surface_a",
    "support_affine_surface_b",
    "support_affine_rootzone_a",
    "support_affine_rootzone_b",
    "effective_calibration_dof",
    "support_affine_target_eval_usage",
    "k_specific_prediction_changed",
    "support_nesting_policy",
    "nested_support_dates_hash",
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
    ("method", "method"),
    ("decision", "stage3_posterior_decision"),
    ("paper", "paper_facing_run"),
    ("rho", "adapt_mix_rho"),
    ("raw_delta", "raw_to_k0_mean_abs_delta"),
    ("final_delta", "final_mix_to_k0_mean_abs_delta"),
    ("tta_effective", "context_tta_effective"),
    ("prompt_delta", "prompt_l2_delta_mean"),
    ("tta_pred_delta", "prediction_delta_vs_no_tta"),
    ("gain_s", "support_gain_alpha_surface"),
    ("gain_r", "support_gain_alpha_rootzone"),
    ("gain_rule", "support_gain_selection_rule"),
    ("aff_s_a", "support_affine_surface_a"),
    ("aff_r_a", "support_affine_rootzone_a"),
    ("k_changed", "k_specific_prediction_changed"),
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
if any(row.get("stage3_posterior_decision") == "rejected_to_k0_anchor" for row in rows):
    print()
    print("Note: K-shot rows marked rejected_to_k0_anchor are K0-equivalent fallback, not accepted few-shot adaptation.")
if any(int(row.get("K") or 0) > 0 and row.get("stage3_posterior_decision") == "no_update" for row in rows):
    print()
    print("Note: K-shot rows marked no_update are source-policy-selected K0-equivalent diagnostics, not few-shot improvements.")
PYTHON_SCRIPT

echo ""
echo "Done. Outputs are under: ${OUTPUT_BASE}"
