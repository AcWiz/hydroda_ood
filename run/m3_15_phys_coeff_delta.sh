#!/usr/bin/env bash
set -euo pipefail

# M3_15: source-safe interpolation between frozen M3_1 and an M3_1-anchored
# physics coefficient-delta branch. Eta is selected only on source_val.
#
# Usage:
#   bash run/m3_15_phys_coeff_delta.sh select-eta US-R1 0 0
#   bash run/m3_15_phys_coeff_delta.sh eval-source US-R1 0 0
#   bash run/m3_15_phys_coeff_delta.sh eval-target US-R1 0 0
#   bash run/m3_15_phys_coeff_delta.sh all US-R1 0 0

MODE="${1:-all}"
TARGET_REGION="${2:-US-R1}"
SEED="${3:-0}"
CUDA_ID="${4:-0}"

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_ID}"
export PYTHONPATH=".:${PYTHONPATH:-}"

find_m3_1_checkpoint() {
  find "artifacts/runs/phase4_hyperda_staged_ablation/M3_1_hyperda_trust_medium" \
    -path "*/${TARGET_REGION}/*s${SEED}*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
    -type f 2>/dev/null | sort | tail -1 || true
}

find_m3_15_checkpoint() {
  find "artifacts/runs/phase4_hyperda_staged_ablation/M3_15_m31_anchored_source_safe_phys_coeff_delta" \
    -path "*/${TARGET_REGION}/*s${SEED}*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
    -type f 2>/dev/null | sort | tail -1 || true
}

M3_1_CHECKPOINT="${M3_1_CHECKPOINT:-$(find_m3_1_checkpoint)}"
PHYS_COEFF_CHECKPOINT="${PHYS_COEFF_CHECKPOINT:-$(find_m3_15_checkpoint)}"

if [[ -z "${M3_1_CHECKPOINT}" || ! -f "${M3_1_CHECKPOINT}" ]]; then
  echo "ERROR: M3_15 requires the M3_1 anchor checkpoint for ${TARGET_REGION} seed ${SEED}." >&2
  echo "Set M3_1_CHECKPOINT=/path/to/M3_1/checkpoint_best_source_val_transfer_safe_score.pt." >&2
  exit 2
fi
if [[ -z "${PHYS_COEFF_CHECKPOINT}" || ! -f "${PHYS_COEFF_CHECKPOINT}" ]]; then
  echo "ERROR: M3_15 requires the trained phys coefficient-delta checkpoint for ${TARGET_REGION} seed ${SEED}." >&2
  echo "Run: ABLATION_ID=M3_15_m31_anchored_source_safe_phys_coeff_delta bash run/phase4_hyperda_staged_ablation.sh auto ${TARGET_REGION} ${SEED} ${CUDA_ID}" >&2
  exit 2
fi

MAX_SAMPLES="${MAX_SAMPLES:-0}"
if [[ "${MAX_SAMPLES}" == "0" ]]; then
  DEFAULT_SAMPLE_TAG="full"
else
  DEFAULT_SAMPLE_TAG="ms${MAX_SAMPLES}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-artifacts/runs/M3_15_m31_anchored_source_safe_phys_coeff_delta/${TARGET_REGION}_s${SEED}_${DEFAULT_SAMPLE_TAG}}"
SELECTION_PATH="${SELECTION_PATH:-${OUTPUT_BASE}/source_val_eta_selection.json}"
GATE_REPORT_PATH="${GATE_REPORT_PATH:-${OUTPUT_BASE}/source_gate.json}"
DEVICE="${DEVICE:-cuda}"
REQUIRE_GPU="${REQUIRE_GPU:-0}"
ETA_GRID="${ETA_GRID:-0,0.1,0.25,0.5,1.0}"
ANCHOR_DUAL_CVAR="${ANCHOR_DUAL_CVAR:-0.446573390549}"
MIN_DUAL_CVAR_DELTA="${MIN_DUAL_CVAR_DELTA:-0.001}"
MIN_BEST_VAR_RMSE_REL_IMPROVE="${MIN_BEST_VAR_RMSE_REL_IMPROVE:-0.001}"
MAX_OTHER_VAR_RMSE_REL_DEGRADE="${MAX_OTHER_VAR_RMSE_REL_DEGRADE:-0.0005}"
MAX_REGION_RMSE_REL_DEGRADE="${MAX_REGION_RMSE_REL_DEGRADE:-0.003}"
MAX_SEASON_RMSE_REL_DEGRADE="${MAX_SEASON_RMSE_REL_DEGRADE:-0.003}"
OUTPUT_LEVEL="${OUTPUT_LEVEL:-compact}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

mkdir -p "${OUTPUT_BASE}"

COMMON_ARGS=(
  --m3_1_checkpoint "${M3_1_CHECKPOINT}"
  --phys_coeff_checkpoint "${PHYS_COEFF_CHECKPOINT}"
  --target_region "${TARGET_REGION}"
  --K 0
  --seed "${SEED}"
  --device "${DEVICE}"
  --max_samples "${MAX_SAMPLES}"
)
if [[ "${REQUIRE_GPU}" == "1" ]]; then
  COMMON_ARGS+=(--require_gpu)
fi

select_eta() {
  python scripts/eval/m3_15_phys_coeff_delta.py select-eta \
    "${COMMON_ARGS[@]}" \
    --selection_out "${SELECTION_PATH}" \
    --eta_grid "${ETA_GRID}" \
    --anchor_dual_cvar "${ANCHOR_DUAL_CVAR}" \
    --min_dual_cvar_delta "${MIN_DUAL_CVAR_DELTA}" \
    --min_best_variable_rmse_relative_improve "${MIN_BEST_VAR_RMSE_REL_IMPROVE}" \
    --max_other_variable_rmse_relative_degrade "${MAX_OTHER_VAR_RMSE_REL_DEGRADE}" \
    --max_region_rmse_relative_degrade "${MAX_REGION_RMSE_REL_DEGRADE}" \
    --max_season_rmse_relative_degrade "${MAX_SEASON_RMSE_REL_DEGRADE}"
}

write_gate_report() {
  python scripts/eval/m3_15_phys_coeff_delta.py write-gate-report \
    --target_region "${TARGET_REGION}" \
    --K 0 \
    --seed "${SEED}" \
    --selection "${SELECTION_PATH}" \
    --gate_report_out "${GATE_REPORT_PATH}"
}

eval_split() {
  local split_type="$1"
  local out_dir="$2"
  python scripts/eval/m3_15_phys_coeff_delta.py evaluate \
    "${COMMON_ARGS[@]}" \
    --selection "${SELECTION_PATH}" \
    --source_gate "${GATE_REPORT_PATH}" \
    --split_type "${split_type}" \
    --output_level "${OUTPUT_LEVEL}" \
    --output_dir "${out_dir}"
}

ensure_selection() {
  if [[ "${FORCE_REBUILD}" == "1" || ! -f "${SELECTION_PATH}" ]]; then
    select_eta
  else
    echo "Reusing M3_15 eta selection cache: ${SELECTION_PATH}"
  fi
  write_gate_report
}

source_gate_passes() {
  python - "$GATE_REPORT_PATH" <<'PY'
import json, sys
path = sys.argv[1]
gate = json.load(open(path, encoding="utf-8"))
ok = (
    gate.get("schema_version") == "m3_15_source_gate_report_v1"
    and gate.get("method_id") == "M3_15_m31_anchored_source_safe_phys_coeff_delta"
    and bool(gate.get("source_gate_pass"))
    and bool(gate.get("target_eval_allowed"))
    and not bool(gate.get("identity_diagnostic"))
    and (
        float(gate.get("selected_eta_surface", 0.0)) > 0.0
        or float(gate.get("selected_eta_rootzone", 0.0)) > 0.0
    )
)
raise SystemExit(0 if ok else 1)
PY
}

case "${MODE}" in
  select-eta)
    ensure_selection
    ;;
  eval-source)
    ensure_selection
    eval_split source_val "${OUTPUT_BASE}/source_val_eval"
    ;;
  eval-target)
    ensure_selection
    if ! source_gate_passes; then
      echo "ERROR: M3_15 source gate did not pass; refusing target_eval." >&2
      echo "Gate report: ${GATE_REPORT_PATH}" >&2
      exit 3
    fi
    eval_split target_eval "${OUTPUT_BASE}/target_eval"
    ;;
  all)
    ensure_selection
    eval_split source_val "${OUTPUT_BASE}/source_val_eval"
    if source_gate_passes; then
      eval_split target_eval "${OUTPUT_BASE}/target_eval"
    else
      echo "M3_15 source gate did not pass; target_eval intentionally not run."
      echo "Gate report: ${GATE_REPORT_PATH}"
    fi
    ;;
  *)
    echo "ERROR: unknown mode ${MODE}. Expected select-eta, eval-source, eval-target, all." >&2
    exit 2
    ;;
esac

echo "M3_15 artifacts: ${OUTPUT_BASE}"
