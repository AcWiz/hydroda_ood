#!/usr/bin/env bash
set -euo pipefail

# M3_11: signed DA-gain physical proposal + residual trust blend around a
# frozen M3_1 anchor.
#
# Usage:
#   bash run/m3_11_signed_da_gain_residual_trust.sh build-bank US-R2 0 0
#   bash run/m3_11_signed_da_gain_residual_trust.sh select-eta US-R2 0 0
#   bash run/m3_11_signed_da_gain_residual_trust.sh eval-source US-R2 0 0
#   bash run/m3_11_signed_da_gain_residual_trust.sh eval-target US-R2 0 0
#   bash run/m3_11_signed_da_gain_residual_trust.sh all US-R2 0 0
#
# MAX_SAMPLES=0 means full split. Existing bank/selection caches are reused
# unless FORCE_REBUILD=1 is set.

MODE="${1:-all}"
TARGET_REGION="${2:-US-R2}"
SEED="${3:-0}"
CUDA_ID="${4:-0}"

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_ID}"
export PYTHONPATH=".:${PYTHONPATH:-}"

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-}"
if [[ -z "${SOURCE_CHECKPOINT}" ]]; then
    SOURCE_CHECKPOINT="$(find "artifacts/runs/phase4_hyperda_staged_ablation/M3_1_hyperda_trust_medium" \
        -path "*/${TARGET_REGION}/*s${SEED}*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
        -type f 2>/dev/null | sort | tail -1 || true)"
fi
if [[ -z "${SOURCE_CHECKPOINT}" || ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "ERROR: M3_11 requires an M3_1 anchor checkpoint for ${TARGET_REGION} seed ${SEED}." >&2
    echo "Set SOURCE_CHECKPOINT=/path/to/checkpoint_best_source_val_transfer_safe_score.pt or run:" >&2
    echo "  ABLATION_ID=M3_1_hyperda_trust_medium bash run/phase4_hyperda_staged_ablation.sh auto ${TARGET_REGION} ${SEED} ${CUDA_ID}" >&2
    exit 2
fi

MAX_SAMPLES="${MAX_SAMPLES:-0}"
if [[ "${MAX_SAMPLES}" == "0" ]]; then
  DEFAULT_SAMPLE_TAG="full"
else
  DEFAULT_SAMPLE_TAG="ms${MAX_SAMPLES}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-artifacts/runs/M3_11_signed_da_gain_residual_trust/${TARGET_REGION}_s${SEED}_${DEFAULT_SAMPLE_TAG}}"
BANK_PATH="${BANK_PATH:-${OUTPUT_BASE}/source_fit_signed_da_gain_bank.json}"
SELECTION_PATH="${SELECTION_PATH:-${OUTPUT_BASE}/source_val_eta_selection.json}"
GATE_REPORT_PATH="${GATE_REPORT_PATH:-${OUTPUT_BASE}/source_gate_report.json}"
DEVICE="${DEVICE:-cuda}"
REQUIRE_GPU="${REQUIRE_GPU:-0}"
ETA_GRID="${ETA_GRID:-0,0.025,0.05,0.10}"
PROPOSAL_CLIP_SCALE="${PROPOSAL_CLIP_SCALE:-1.0}"
RESIDUAL_CLIP_SCALE="${RESIDUAL_CLIP_SCALE:-}"
MIN_DUAL_CVAR_DELTA="${MIN_DUAL_CVAR_DELTA:-0.002}"
MAX_VARIABLE_RMSE_REL_DEGRADE="${MAX_VARIABLE_RMSE_REL_DEGRADE:-0.001}"
MAX_REGION_RMSE_REL_DEGRADE="${MAX_REGION_RMSE_REL_DEGRADE:-0.005}"
OUTPUT_LEVEL="${OUTPUT_LEVEL:-compact}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-200}"
SOURCE_NEIGHBOR_TOP_M="${SOURCE_NEIGHBOR_TOP_M:-4}"

mkdir -p "${OUTPUT_BASE}"

COMMON_ARGS=(
  --checkpoint "${SOURCE_CHECKPOINT}"
  --target_region "${TARGET_REGION}"
  --K 0
  --seed "${SEED}"
  --device "${DEVICE}"
  --max_samples "${MAX_SAMPLES}"
)
if [[ "${REQUIRE_GPU}" == "1" ]]; then
  COMMON_ARGS+=(--require_gpu)
fi

RESIDUAL_ARGS=()
if [[ -n "${RESIDUAL_CLIP_SCALE}" ]]; then
  RESIDUAL_ARGS+=(--residual_clip_scale "${RESIDUAL_CLIP_SCALE}")
fi

build_bank() {
  python scripts/eval/m3_11_signed_da_gain_residual_trust.py build-bank \
    "${COMMON_ARGS[@]}" \
    --bank_out "${BANK_PATH}" \
    --source_neighbor_top_m "${SOURCE_NEIGHBOR_TOP_M}" \
    --progress_every "${PROGRESS_EVERY}"
}

select_eta() {
  python scripts/eval/m3_11_signed_da_gain_residual_trust.py select-eta \
    "${COMMON_ARGS[@]}" \
    --bank "${BANK_PATH}" \
    --selection_out "${SELECTION_PATH}" \
    --eta_grid "${ETA_GRID}" \
    --proposal_clip_scale "${PROPOSAL_CLIP_SCALE}" \
    "${RESIDUAL_ARGS[@]}" \
    --min_dual_cvar_delta "${MIN_DUAL_CVAR_DELTA}" \
    --max_variable_rmse_relative_degrade "${MAX_VARIABLE_RMSE_REL_DEGRADE}" \
    --max_region_rmse_relative_degrade "${MAX_REGION_RMSE_REL_DEGRADE}"
}

write_gate_report() {
  python scripts/eval/m3_11_signed_da_gain_residual_trust.py write-gate-report \
    --target_region "${TARGET_REGION}" \
    --K 0 \
    --seed "${SEED}" \
    --selection "${SELECTION_PATH}" \
    --gate_report_out "${GATE_REPORT_PATH}"
}

eval_split() {
  local split_type="$1"
  local out_dir="$2"
  shift 2
  python scripts/eval/m3_11_signed_da_gain_residual_trust.py evaluate \
    "${COMMON_ARGS[@]}" \
    --bank "${BANK_PATH}" \
    --selection "${SELECTION_PATH}" \
    --split_type "${split_type}" \
    --proposal_clip_scale "${PROPOSAL_CLIP_SCALE}" \
    "${RESIDUAL_ARGS[@]}" \
    --output_level "${OUTPUT_LEVEL}" \
    --output_dir "${out_dir}" \
    "$@"
}

ensure_bank() {
  if [[ "${FORCE_REBUILD}" == "1" || ! -f "${BANK_PATH}" ]]; then
    build_bank
  else
    echo "Reusing M3_11 signed DA gain bank cache: ${BANK_PATH}"
  fi
}

ensure_selection() {
  ensure_bank
  if [[ "${FORCE_REBUILD}" == "1" || ! -f "${SELECTION_PATH}" ]]; then
    select_eta
  else
    echo "Reusing M3_11 eta selection cache: ${SELECTION_PATH}"
  fi
  write_gate_report
}

source_gate_passes() {
  python - "$SELECTION_PATH" <<'PY'
import json, sys
path = sys.argv[1]
selection = json.load(open(path, encoding="utf-8"))
eta_positive = float(selection.get("selected_eta_surface", 0.0)) > 0.0 or float(selection.get("selected_eta_rootzone", 0.0)) > 0.0
raise SystemExit(0 if selection.get("source_gate_pass") and eta_positive else 1)
PY
}

case "${MODE}" in
  build-bank)
    if [[ "${FORCE_REBUILD}" == "1" || ! -f "${BANK_PATH}" ]]; then
      build_bank
    else
      echo "Reusing M3_11 signed DA gain bank cache: ${BANK_PATH}"
    fi
    ;;
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
      echo "ERROR: M3_11 source gate did not pass; refusing target_eval." >&2
      echo "Gate report: ${GATE_REPORT_PATH}" >&2
      exit 3
    fi
    eval_split target_eval "${OUTPUT_BASE}/target_eval" --write_experiment_card
    ;;
  all)
    ensure_selection
    eval_split source_val "${OUTPUT_BASE}/source_val_eval"
    if source_gate_passes; then
      eval_split target_eval "${OUTPUT_BASE}/target_eval" --write_experiment_card
    else
      echo "M3_11 source gate did not pass; target_eval intentionally not run."
      echo "Gate report: ${GATE_REPORT_PATH}"
    fi
    ;;
  *)
    echo "ERROR: unknown mode ${MODE}. Expected build-bank, select-eta, eval-source, eval-target, all." >&2
    exit 2
    ;;
esac

echo "M3_11 artifacts: ${OUTPUT_BASE}"
