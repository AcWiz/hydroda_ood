#!/bin/bash
# Phase 5 P2.8: source-safe conservative guard calibration.
#
# Calibration query split is source_val pseudo-query only. This wrapper does
# not pass target_eval artifacts to the calibrator.
#
# Usage:
#   ALLOW_IN_CHECKPOINT_SOURCE_EPISODES=1 \
#   bash run/phase5_hyperda_p2_8_source_safe_guard_calibration.sh <source_checkpoint> US-R1 0

set -euo pipefail

SOURCE_CHECKPOINT="${1:-}"
FINAL_TARGET_REGION="${2:-US-R1}"
SEED="${3:-0}"
OUTPUT_BASE="${4:-artifacts/runs/phase5_hyperda_p2_8_source_safe_guard_calibration/${FINAL_TARGET_REGION}_s${SEED}_$(date -u +%Y%m%dT%H%M%SZ)}"

CALIBRATION_ROWS="${CALIBRATION_ROWS:-}"
CALIBRATION_INPUT_ROOTS="${CALIBRATION_INPUT_ROOTS:-artifacts/runs/phase5_hyperda_p2_8_source_safe_guard_calibration_source_rows}"
CHECKPOINT_SOURCE_REGIONS="${CHECKPOINT_SOURCE_REGIONS:-}"
ALLOW_IN_CHECKPOINT_SOURCE_EPISODES="${ALLOW_IN_CHECKPOINT_SOURCE_EPISODES:-0}"
MAX_EPISODES="${MAX_EPISODES:-0}"
CANDIDATE_SET="${CANDIDATE_SET:-compact_v1}"
CALIBRATION_STAGE="${CALIBRATION_STAGE:-coarse}"
SOURCE_QUERY_MAX_SAMPLES="${SOURCE_QUERY_MAX_SAMPLES:-256}"
RESUME="${RESUME:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN_MANIFEST="${DRY_RUN_MANIFEST:-0}"
CALIB_MAX_QUERY_SAMPLES="${CALIB_MAX_QUERY_SAMPLES:-${SOURCE_QUERY_MAX_SAMPLES}}"
BUILD_SOURCE_ROWS="${BUILD_SOURCE_ROWS:-0}"
PSEUDO_TARGET_REGIONS="${PSEUDO_TARGET_REGIONS:-}"
TOP_CANDIDATE_IDS="${TOP_CANDIDATE_IDS:-}"
CUDA_DEVICE="${CUDA_DEVICE:-${CUDA_VISIBLE_DEVICES:-1}}"
CURRENT_SPLITS_JSON="${CURRENT_SPLITS_JSON:-artifacts/splits/US_loro_zero_few_shot_splits.json}"
ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
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
        -path "*hyperda_basis_adapter_${FINAL_TARGET_REGION}_*_s${SEED}_*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
        -type f 2>/dev/null | sort | tail -1)"
fi

if [[ -z "${SOURCE_CHECKPOINT}" || ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "ERROR: source HyperDA checkpoint not found." >&2
    exit 2
fi

mkdir -p "${OUTPUT_BASE}"

ALLOW_ARGS=()
if [[ "${ALLOW_IN_CHECKPOINT_SOURCE_EPISODES}" == "1" || "${ALLOW_IN_CHECKPOINT_SOURCE_EPISODES,,}" == "true" ]]; then
    ALLOW_ARGS=(--allow_in_checkpoint_source_episodes)
fi

ROW_ARGS=()
if [[ -n "${CALIBRATION_ROWS}" ]]; then
    read -r -a _rows <<< "${CALIBRATION_ROWS}"
    ROW_ARGS=(--calibration_rows "${_rows[@]}")
fi

ROOT_ARGS=()
if [[ -n "${CALIBRATION_INPUT_ROOTS}" ]]; then
    read -r -a _roots <<< "${CALIBRATION_INPUT_ROOTS}"
    ROOT_ARGS=(--input_roots "${_roots[@]}")
fi

echo "============================================"
echo "Phase 5 P2.8 Source-Safe Guard Calibration"
echo "  source_checkpoint=${SOURCE_CHECKPOINT}"
echo "  final_target_region=${FINAL_TARGET_REGION}"
echo "  seed=${SEED}"
echo "  calibration_query_split=source_val"
echo "  support_labels=source_pseudo_target_K12_support_only"
echo "  target_eval_usage=never_read_by_calibration"
echo "  CANDIDATE_SET=${CANDIDATE_SET}"
echo "  CALIBRATION_STAGE=${CALIBRATION_STAGE}"
echo "  SOURCE_QUERY_MAX_SAMPLES=${SOURCE_QUERY_MAX_SAMPLES}"
echo "  RESUME=${RESUME}"
echo "  SKIP_EXISTING=${SKIP_EXISTING}"
echo "  DRY_RUN_MANIFEST=${DRY_RUN_MANIFEST}"
echo "  ALLOW_IN_CHECKPOINT_SOURCE_EPISODES=${ALLOW_IN_CHECKPOINT_SOURCE_EPISODES}"
echo "  BUILD_SOURCE_ROWS=${BUILD_SOURCE_ROWS}"
echo "  PSEUDO_TARGET_REGIONS=${PSEUDO_TARGET_REGIONS}"
echo "  MAX_EPISODES=${MAX_EPISODES}"
echo "  CALIB_MAX_QUERY_SAMPLES=${CALIB_MAX_QUERY_SAMPLES}"
echo "  output_base=${OUTPUT_BASE}"
echo "============================================"

write_resume_manifest_for_root() {
    local source_rows_root="$1"
    if [[ -z "${PSEUDO_TARGET_REGIONS}" ]]; then
        echo "ERROR: manifest writing requires PSEUDO_TARGET_REGIONS, e.g. US-R2,US-R3." >&2
        exit 2
    fi
    PYTHONPATH=. python3 - "${source_rows_root}" "${PSEUDO_TARGET_REGIONS}" "${CANDIDATE_SET}" "${OUTPUT_BASE}" "${CALIB_MAX_QUERY_SAMPLES}" "${CALIBRATION_STAGE}" "${TOP_CANDIDATE_IDS}" <<'PYTHON_SCRIPT'
import sys
from pathlib import Path
from scripts.eval.calibrate_source_safe_guard import build_resume_manifest, required_gpu_row_configs, _write_csv

root = Path(sys.argv[1])
regions = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
candidate_set = sys.argv[3]
output = Path(sys.argv[4])
expected_sample_budget = int(sys.argv[5])
stage = sys.argv[6]
top_ids = [item.strip() for item in sys.argv[7].split(",") if item.strip()]
manifest = build_resume_manifest(
    base_candidates=required_gpu_row_configs(
        candidate_set=candidate_set,
        top_candidate_ids=top_ids if stage == "final" else (),
    ),
    pseudo_target_regions=regions,
    source_rows_root=root,
    base_command_prefix="RESUME=1 SKIP_EXISTING=1 bash run/phase5_hyperda_p2_8_source_safe_guard_calibration.sh",
    expected_sample_budget=expected_sample_budget,
)
_write_csv(output / "candidate_manifest.csv", manifest["candidate_manifest"])
_write_csv(output / "completed_rows.csv", manifest["completed_rows"])
_write_csv(output / "missing_rows.csv", manifest["missing_rows"])
_write_csv(output / "invalid_existing_rows.csv", manifest["invalid_existing_rows"])
(output / "resume_commands.md").write_text(manifest["resume_commands_md"], encoding="utf-8")
(output / "estimated_remaining_rows").write_text(str(manifest["estimated_remaining_rows"]) + "\n", encoding="utf-8")
PYTHON_SCRIPT
}

if [[ "${DRY_RUN_MANIFEST}" == "1" || "${DRY_RUN_MANIFEST,,}" == "true" ]]; then
    if [[ "${CALIBRATION_STAGE}" == "final" && -z "${TOP_CANDIDATE_IDS}" ]]; then
        echo "ERROR: CALIBRATION_STAGE=final requires TOP_CANDIDATE_IDS from the coarse summary." >&2
        exit 2
    fi
    DRY_RUN_ROWS_ROOT="${OUTPUT_BASE}/source_val_candidate_rows"
    if [[ "${BUILD_SOURCE_ROWS}" != "1" && "${BUILD_SOURCE_ROWS,,}" != "true" ]]; then
        if [[ -n "${CALIBRATION_INPUT_ROOTS}" && -d "${CALIBRATION_INPUT_ROOTS}" ]]; then
            DRY_RUN_ROWS_ROOT="${CALIBRATION_INPUT_ROOTS}"
        elif [[ -d "artifacts/runs/phase5_hyperda_p2_8_source_safe_guard_calibration/US-R1_s0_20260612T145549Z/source_val_candidate_rows" ]]; then
            DRY_RUN_ROWS_ROOT="artifacts/runs/phase5_hyperda_p2_8_source_safe_guard_calibration/US-R1_s0_20260612T145549Z/source_val_candidate_rows"
        fi
    fi
    write_resume_manifest_for_root "${DRY_RUN_ROWS_ROOT}"
    echo "DRY_RUN_MANIFEST=1 wrote manifests and exited before adaptation/evaluation."
    exit 0
fi

if [[ "${BUILD_SOURCE_ROWS}" == "1" || "${BUILD_SOURCE_ROWS,,}" == "true" ]]; then
    if [[ -z "${PSEUDO_TARGET_REGIONS}" ]]; then
        echo "ERROR: BUILD_SOURCE_ROWS=1 requires PSEUDO_TARGET_REGIONS, e.g. US-R2,US-R3." >&2
        exit 2
    fi

    SOURCE_ROWS_ROOT="${OUTPUT_BASE}/source_val_candidate_rows"
    mkdir -p "${SOURCE_ROWS_ROOT}"
    IFS=',' read -r -a _pseudo_regions <<< "${PSEUDO_TARGET_REGIONS}"
    episode_count=0
    _resume_manifest_trap_active=0

    write_resume_manifest() {
        write_resume_manifest_for_root "${SOURCE_ROWS_ROOT}"
    }

    on_source_rows_error() {
        local status=$?
        if [[ "${_resume_manifest_trap_active}" == "1" ]]; then
            echo "ERROR: source-row build failed with status ${status}; writing resume manifest before exit." >&2
            write_resume_manifest || true
        fi
        exit "${status}"
    }
    trap on_source_rows_error ERR
    _resume_manifest_trap_active=1

    artifact_valid_for_hash() {
        local row_dir="$1"
        local expected_hash="$2"
        [[ -f "${row_dir}/source_safe_candidate_rows.csv" ]] || return 1
        PYTHONPATH=. python3 - "${row_dir}/source_safe_candidate_rows.csv" "${expected_hash}" "${CALIB_MAX_QUERY_SAMPLES}" <<'PYTHON_SCRIPT'
import sys
from scripts.eval.calibrate_source_safe_guard import _strict_existing_row_invalid_reasons, load_calibration_rows

rows = load_calibration_rows([sys.argv[1]])
ok = any(
    not _strict_existing_row_invalid_reasons(
        row,
        expected_hash=sys.argv[2],
        expected_sample_budget=int(sys.argv[3]),
    )
    for row in rows
)
raise SystemExit(0 if ok else 1)
PYTHON_SCRIPT
    }

    if [[ "${CALIBRATION_STAGE}" == "final" && -z "${TOP_CANDIDATE_IDS}" ]]; then
        echo "ERROR: CALIBRATION_STAGE=final requires TOP_CANDIDATE_IDS from the coarse summary." >&2
        exit 2
    fi

    run_source_row() {
        local pseudo_region="$1"
        local run_id="$2"
        local k="$3"
        local schedule_label="$4"
        local lr="$5"
        local steps="$6"
        local alpha="$7"
        local trust_policy="$8"
        local trust_mode="$9"
        local trust_total="${10}"
        local trust_prompt="${11}"
        local trust_gain="${12}"
        local trust_coeff="${13}"
        local trust_spatial="${14}"
        local rho_policy="${15}"
        local requested_rho="${16}"
        local loss_reduction="${17}"
        local candidate_config_hash="${18:-}"
        local base_config_id="${19:-}"
        local row_dir="${SOURCE_ROWS_ROOT}/${pseudo_region}/${run_id}"
        local target_adaptation_setting
        if [[ "${k}" == "0" ]]; then
            target_adaptation_setting="zero_shot_context"
        else
            target_adaptation_setting="few_shot_k${k}"
        fi
        mkdir -p "${row_dir}"
        if [[ "${RESUME}" == "1" && "${SKIP_EXISTING}" == "1" && -n "${candidate_config_hash}" ]]; then
            if artifact_valid_for_hash "${row_dir}" "${candidate_config_hash}"; then
                echo ">>> skip existing valid source row pseudo_region=${pseudo_region} run_id=${run_id}"
                return 0
            fi
        fi

        K_LIST="${k}" \
        ADAPT_SCOPE="all" \
        ADAPT_SOLVER="adamw" \
        SCHEDULE_LABEL="${schedule_label}" \
        ADAPT_LR="${lr}" \
        ADAPT_MAX_STEPS="${steps}" \
        ADAPT_ANCHOR_ALPHA="${alpha}" \
        ADAPT_WEIGHT_DECAY="${ADAPT_WEIGHT_DECAY}" \
        ADAPT_GRAD_CLIP="${ADAPT_GRAD_CLIP}" \
        ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE}" \
        BATCH_SIZE="${ADAPT_BATCH_SIZE}" \
        SPLITS_JSON="${CURRENT_SPLITS_JSON}" \
        TRUST_REGION_MODE="${trust_mode}" \
        TRUST_TOTAL_RADIUS="${trust_total}" \
        TRUST_PROMPT_RADIUS="${trust_prompt}" \
        TRUST_GAIN_RADIUS="${trust_gain}" \
        TRUST_COEFF_RADIUS="${trust_coeff}" \
        TRUST_SPATIAL_RADIUS="${trust_spatial}" \
        SUPPORT_LOSS_REDUCTION="${loss_reduction}" \
        OUTPUT_DIR="${row_dir}/K${k}/adapt" \
        bash run/phase5_hyperda_zero_few_shot.sh \
            "${SOURCE_CHECKPOINT}" \
            "${pseudo_region}" \
            "${k}" \
            "${SEED}" \
            "${CUDA_DEVICE}" \
            2>&1 | tee "${row_dir}/adapt.log"

        local adapted_checkpoint="${row_dir}/K${k}/adapt/checkpoints/checkpoint_final_preregistered.pt"
        if [[ ! -f "${adapted_checkpoint}" ]]; then
            echo "ERROR: missing adapted checkpoint ${adapted_checkpoint}" >&2
            exit 2
        fi
        local eval_rho="${requested_rho}"
        if [[ "${rho_policy}" == rule_* ]]; then
            eval_rho="$(PYTHONPATH=. python3 - "${row_dir}/K${k}/adapt/metadata.json" "${rho_policy}" <<'PYTHON_SCRIPT'
import json
import sys
from scripts.eval.calibrate_source_safe_guard import compute_rho_for_policy

metadata = json.load(open(sys.argv[1], encoding="utf-8"))
diagnostics = metadata.get("support_gradient_diagnostics", {})
print(compute_rho_for_policy(sys.argv[2], diagnostics))
PYTHON_SCRIPT
)"
        fi

        PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \
            --checkpoint "${adapted_checkpoint}" \
            --target_region "${pseudo_region}" \
            --adaptation_setting "${target_adaptation_setting}" \
            --K "${k}" \
            --seed "${SEED}" \
            --split_type source_val \
            --active_region_override "${pseudo_region}" \
            --splits_json "${CURRENT_SPLITS_JSON}" \
            --predictor_type hyperda_target_adapt \
            --device cuda \
            --output_dir "${row_dir}/eval" \
            --max_samples "${CALIB_MAX_QUERY_SAMPLES}" \
            --batch_size "${EVAL_BATCH_SIZE}" \
            --adapt_mix_rho "${eval_rho}" \
            --prediction_record_path "${row_dir}/prediction_records.jsonl" \
            2>&1 | tee "${row_dir}/eval.log"

        python3 - "${row_dir}" "${pseudo_region}" "${run_id}" "${rho_policy}" "${trust_policy}" "${candidate_config_hash}" "${base_config_id}" "${CALIB_MAX_QUERY_SAMPLES}" <<'PYTHON_SCRIPT'
import csv
import json
import sys
from pathlib import Path

row_dir = Path(sys.argv[1])
pseudo_region = sys.argv[2]
run_id = sys.argv[3]
rho_policy = sys.argv[4]
trust_policy = sys.argv[5]
candidate_config_hash = sys.argv[6]
base_config_id = sys.argv[7]
calib_max_query_samples = sys.argv[8]
overview_rows = []
summary_paths = sorted((row_dir / "eval").glob("*/summary.json"))
adapt_paths = sorted((row_dir).glob("K*/adapt/metadata.json"))
summary = json.loads(summary_paths[0].read_text(encoding="utf-8")) if summary_paths else {}
adapt = json.loads(adapt_paths[0].read_text(encoding="utf-8")) if adapt_paths else {}
surface = (summary.get("surface", {}) or {}).get("skill_primary")
rootzone = (summary.get("rootzone", {}) or {}).get("skill_primary")
row = {
    "episode_id": f"{pseudo_region}_S{summary.get('seed', '')}",
    "pseudo_target_region": pseudo_region,
    "query_role": "source_val_pseudo_query",
    "split_type": "source_val",
    "run_id": run_id,
    "candidate_id": run_id,
    "base_config_id": base_config_id,
    "candidate_config_hash": candidate_config_hash,
    "K": summary.get("K", adapt.get("K")),
    "seed": summary.get("seed", adapt.get("seed")),
    "adapt_scope": adapt.get("adapt_scope", "all"),
    "adapt_solver": adapt.get("adapt_solver", "adamw"),
    "schedule_label": adapt.get("schedule_label", ""),
    "support_loss_reduction": adapt.get("support_loss_reduction", "global_pixel"),
    "rho_policy": rho_policy,
    "trust_policy": trust_policy,
    "trust_region_mode": adapt.get("trust_region_mode", "none"),
    "adapt_mix_rho": summary.get("adapt_mix_rho"),
    "lr": adapt.get("lr"),
    "adaptation_steps": adapt.get("adaptation_steps"),
    "anchor_alpha": adapt.get("anchor_alpha"),
    "surface_skill_primary": surface,
    "rootzone_skill_primary": rootzone,
    "target_parameter_l2_drift_post_anchor_total": (adapt.get("target_parameter_l2_drift_post_anchor", {}) or {}).get("total"),
    "target_parameter_l2_drift_post_anchor_target_prompt": (adapt.get("target_parameter_l2_drift_post_anchor", {}) or {}).get("target_prompt"),
    "target_parameter_l2_drift_post_anchor_monthly_gain": (adapt.get("target_parameter_l2_drift_post_anchor", {}) or {}).get("monthly_gain"),
    "target_parameter_l2_drift_post_anchor_adapter_coeff_bottleneck": (adapt.get("target_parameter_l2_drift_post_anchor", {}) or {}).get("adapter_coeff_bottleneck"),
    "target_parameter_l2_drift_post_anchor_adapter_coeff_dec2": (adapt.get("target_parameter_l2_drift_post_anchor", {}) or {}).get("adapter_coeff_dec2"),
    "target_parameter_l2_drift_post_anchor_adapter_coeff_dec1": (adapt.get("target_parameter_l2_drift_post_anchor", {}) or {}).get("adapter_coeff_dec1"),
    "support_gradient_negative_fraction": adapt.get("support_gradient_negative_fraction"),
    "support_gradient_cosine_min": adapt.get("support_gradient_cosine_min"),
    "source_checkpoint_sha256": adapt.get("source_checkpoint_sha256", summary.get("source_checkpoint_sha256", "")),
    "split_manifest_sha256": adapt.get("split_manifest_sha256", summary.get("split_manifest_sha256", "")),
    "target_eval_dates_hash": "not_used_source_val_query",
    "prediction_record_path": str(row_dir / "prediction_records.jsonl"),
    "prediction_content_hash": summary.get("prediction_content_hash", ""),
    "calib_max_query_samples": calib_max_query_samples,
}
fieldnames = list(row)
csv_path = row_dir / "source_safe_candidate_rows.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(row)
(row_dir / "source_safe_candidate_rows.json").write_text(json.dumps([row], indent=2) + "\n", encoding="utf-8")
PYTHON_SCRIPT
    }

    selected_regions=()
    for pseudo_region_raw in "${_pseudo_regions[@]}"; do
        pseudo_region="$(echo "${pseudo_region_raw}" | xargs)"
        [[ -z "${pseudo_region}" ]] && continue
        if [[ "${MAX_EPISODES}" != "0" && "${episode_count}" -ge "${MAX_EPISODES}" ]]; then
            break
        fi
        episode_count=$((episode_count + 1))
        selected_regions+=("${pseudo_region}")
        run_source_row "${pseudo_region}" "K0_identity" "0" "identity_base" "1e-3" "0" "0.0" "none" "none" "0.0" "0.0" "0.0" "0.0" "0.0" "fixed_1.0" "1.0" "global_pixel" "K0_identity_static" "K0"
        run_source_row "${pseudo_region}" "K4_original" "4" "original_K4" "${K4_LR}" "${K4_STEPS}" "${K4_ANCHOR_ALPHA}" "none" "none" "0.0" "0.0" "0.0" "0.0" "0.0" "fixed_1.0" "1.0" "global_pixel" "K4_original_static" "K4"
    done

    TRUST_RADII_JSON="${SOURCE_ROWS_ROOT}/trust_radii_source_k4_original.json"
    PYTHONPATH=. python3 - "${SOURCE_ROWS_ROOT}" "${TRUST_RADII_JSON}" <<'PYTHON_SCRIPT'
import csv
import json
import sys
from pathlib import Path
from scripts.eval.calibrate_source_safe_guard import derive_trust_radii

rows = []
for path in sorted(Path(sys.argv[1]).glob("*/K4_original/source_safe_candidate_rows.csv")):
    with path.open(newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))
Path(sys.argv[2]).write_text(json.dumps(derive_trust_radii(rows), indent=2) + "\n", encoding="utf-8")
PYTHON_SCRIPT

    for pseudo_region in "${selected_regions[@]}"; do
        while IFS= read -r candidate; do
            trust_policy="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["trust_policy"])' "${candidate}")"
            trust_mode="$([[ "${trust_policy}" == "none" ]] && echo none || echo groupwise)"
            trust_total="$(python3 - "${TRUST_RADII_JSON}" "${trust_policy}" total <<'PYTHON_SCRIPT'
import json, sys
r=json.load(open(sys.argv[1], encoding="utf-8"))
print(r.get(sys.argv[2], {}).get(sys.argv[3], 0.0))
PYTHON_SCRIPT
)"
            trust_prompt="$(python3 - "${TRUST_RADII_JSON}" "${trust_policy}" prompt <<'PYTHON_SCRIPT'
import json, sys
r=json.load(open(sys.argv[1], encoding="utf-8"))
print(r.get(sys.argv[2], {}).get(sys.argv[3], 0.0))
PYTHON_SCRIPT
)"
            trust_gain="$(python3 - "${TRUST_RADII_JSON}" "${trust_policy}" gain <<'PYTHON_SCRIPT'
import json, sys
r=json.load(open(sys.argv[1], encoding="utf-8"))
print(r.get(sys.argv[2], {}).get(sys.argv[3], 0.0))
PYTHON_SCRIPT
)"
            trust_coeff="$(python3 - "${TRUST_RADII_JSON}" "${trust_policy}" coeff <<'PYTHON_SCRIPT'
import json, sys
r=json.load(open(sys.argv[1], encoding="utf-8"))
print(r.get(sys.argv[2], {}).get(sys.argv[3], 0.0))
PYTHON_SCRIPT
)"
            trust_spatial="$(python3 - "${TRUST_RADII_JSON}" "${trust_policy}" spatial <<'PYTHON_SCRIPT'
import json, sys
r=json.load(open(sys.argv[1], encoding="utf-8"))
print(r.get(sys.argv[2], {}).get(sys.argv[3], 0.0))
PYTHON_SCRIPT
)"
            run_source_row \
                "${pseudo_region}" \
                "$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["candidate_id"])' "${candidate}")" \
                "12" \
                "$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["schedule_label"])' "${candidate}")" \
                "$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["lr"])' "${candidate}")" \
                "$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["adaptation_steps"])' "${candidate}")" \
                "$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["anchor_alpha"])' "${candidate}")" \
                "${trust_policy}" \
                "${trust_mode}" \
                "${trust_total}" "${trust_prompt}" "${trust_gain}" "${trust_coeff}" "${trust_spatial}" \
                "$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["rho_policy"])' "${candidate}")" \
                "$(python3 -c 'import json,sys; from scripts.eval.calibrate_source_safe_guard import compute_rho_for_policy; print(compute_rho_for_policy(json.loads(sys.argv[1])["rho_policy"], {}))' "${candidate}")" \
                "$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["support_loss_reduction"])' "${candidate}")" \
                "$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["candidate_config_hash"])' "${candidate}")" \
                "$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["base_config_id"])' "${candidate}")"
        done < <(PYTHONPATH=. python3 - "${CANDIDATE_SET}" "${CALIBRATION_STAGE}" "${TOP_CANDIDATE_IDS}" <<'PYTHON_SCRIPT'
import json
import sys
from scripts.eval.calibrate_source_safe_guard import base_configs_for_logical_candidate_ids

top_ids = [item.strip() for item in sys.argv[3].split(",") if item.strip()]
for candidate in base_configs_for_logical_candidate_ids(
    candidate_set=sys.argv[1],
    candidate_ids=top_ids if sys.argv[2] == "final" else (),
):
    print(json.dumps(candidate, separators=(",", ":")))
PYTHON_SCRIPT
)
    done
    write_resume_manifest
    _resume_manifest_trap_active=0
    trap - ERR
    CALIBRATION_INPUT_ROOTS="${SOURCE_ROWS_ROOT}"
    ROOT_ARGS=(--input_roots "${SOURCE_ROWS_ROOT}")
fi

if [[ "${CALIBRATION_STAGE}" == "final" && -z "${TOP_CANDIDATE_IDS}" && -f "${OUTPUT_BASE}/source_safe_calibration_summary.json" ]]; then
    TOP_CANDIDATE_IDS="$(PYTHONPATH=. python3 - "${OUTPUT_BASE}/source_safe_calibration_summary.json" <<'PYTHON_SCRIPT'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
print(",".join(summary.get("top5_candidate_ids", [])))
PYTHON_SCRIPT
)"
fi

PYTHONPATH=. python scripts/eval/calibrate_source_safe_guard.py \
    --source_checkpoint "${SOURCE_CHECKPOINT}" \
    --checkpoint_source_regions "${CHECKPOINT_SOURCE_REGIONS}" \
    --final_target_region "${FINAL_TARGET_REGION}" \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_BASE}" \
    --write_candidate_plan "${OUTPUT_BASE}/candidate_plan.json" \
    --candidate_set "${CANDIDATE_SET}" \
    --calibration_stage "${CALIBRATION_STAGE}" \
    --source_query_max_samples "${CALIB_MAX_QUERY_SAMPLES}" \
    --pseudo_target_regions "${PSEUDO_TARGET_REGIONS}" \
    --source_rows_root "${OUTPUT_BASE}/source_val_candidate_rows" \
    --top_candidate_ids "${TOP_CANDIDATE_IDS}" \
    "${ALLOW_ARGS[@]}" \
    "${ROW_ARGS[@]}" \
    "${ROOT_ARGS[@]}"

echo ""
echo "P2.8 source-safe calibration artifacts:"
echo "  ${OUTPUT_BASE}/selected_guard_config.yaml"
echo "  ${OUTPUT_BASE}/source_safe_calibration_summary.md"
echo "  ${OUTPUT_BASE}/coarse_source_safe_calibration_summary.md"
echo "  ${OUTPUT_BASE}/final_source_safe_calibration_summary.md"
echo "  ${OUTPUT_BASE}/candidate_rankings.csv"
echo "  ${OUTPUT_BASE}/final_candidate_rankings.csv"
echo "  ${OUTPUT_BASE}/top5_stability.csv"
echo "  ${OUTPUT_BASE}/top5_candidates.csv"
echo "  ${OUTPUT_BASE}/leave_one_source_region_out_stability.csv"
echo "  ${OUTPUT_BASE}/candidate_manifest.csv"
echo "  ${OUTPUT_BASE}/completed_rows.csv"
echo "  ${OUTPUT_BASE}/missing_rows.csv"
echo "  ${OUTPUT_BASE}/invalid_existing_rows.csv"
echo "  ${OUTPUT_BASE}/resume_commands.md"
