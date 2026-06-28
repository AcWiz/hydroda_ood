#!/usr/bin/env bash
set -euo pipefail

# M3_10a: source-side DA gain bank + post-hoc bounded router.
# Usage:
#   bash run/m3_10a_da_gain_router_lite.sh build-bank US-R1 0 0
#   bash run/m3_10a_da_gain_router_lite.sh select-eta US-R1 0 0
#   bash run/m3_10a_da_gain_router_lite.sh eval-source US-R1 0 0
#   bash run/m3_10a_da_gain_router_lite.sh eval-target US-R1 0 0
#   bash run/m3_10a_da_gain_router_lite.sh all US-R1 0 0
#
# MAX_SAMPLES=0 means full source split. Existing bank/selection caches are
# reused unless FORCE_REBUILD=1 is set.

MODE="${1:-all}"
TARGET_REGION="${2:-US-R1}"
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
    echo "ERROR: M3_1 source checkpoint not found. Set SOURCE_CHECKPOINT=/path/to/checkpoint.pt" >&2
    exit 2
fi

MAX_SAMPLES="${MAX_SAMPLES:-0}"
if [[ "${MAX_SAMPLES}" == "0" ]]; then
  DEFAULT_SAMPLE_TAG="full"
else
  DEFAULT_SAMPLE_TAG="ms${MAX_SAMPLES}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-artifacts/runs/M3_10a_da_gain_router_lite/${TARGET_REGION}_s${SEED}_${DEFAULT_SAMPLE_TAG}}"
BANK_PATH="${BANK_PATH:-${OUTPUT_BASE}/source_fit_da_gain_bank.json}"
SELECTION_PATH="${SELECTION_PATH:-${OUTPUT_BASE}/source_val_eta_selection.json}"
DEVICE="${DEVICE:-cuda}"
REQUIRE_GPU="${REQUIRE_GPU:-0}"
ETA_GRID="${ETA_GRID:-0,0.025,0.05,0.10}"
PROPOSAL_CLIP_SCALE="${PROPOSAL_CLIP_SCALE:-1.0}"
OUTPUT_LEVEL="${OUTPUT_LEVEL:-compact}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-200}"

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

build_bank() {
  python scripts/eval/m3_10a_da_gain_router_lite.py build-bank \
    "${COMMON_ARGS[@]}" \
    --bank_out "${BANK_PATH}" \
    --progress_every "${PROGRESS_EVERY}"
}

select_eta() {
  python scripts/eval/m3_10a_da_gain_router_lite.py select-eta \
    "${COMMON_ARGS[@]}" \
    --bank "${BANK_PATH}" \
    --selection_out "${SELECTION_PATH}" \
    --eta_grid "${ETA_GRID}" \
    --proposal_clip_scale "${PROPOSAL_CLIP_SCALE}"
}

eval_split() {
  local split_type="$1"
  local out_dir="$2"
  python scripts/eval/m3_10a_da_gain_router_lite.py evaluate \
    "${COMMON_ARGS[@]}" \
    --bank "${BANK_PATH}" \
    --selection "${SELECTION_PATH}" \
    --split_type "${split_type}" \
    --proposal_clip_scale "${PROPOSAL_CLIP_SCALE}" \
    --output_level "${OUTPUT_LEVEL}" \
    --output_dir "${out_dir}"
}

ensure_bank() {
  if [[ "${FORCE_REBUILD}" == "1" || ! -f "${BANK_PATH}" ]]; then
    build_bank
  else
    echo "Reusing DA gain bank cache: ${BANK_PATH}"
  fi
}

ensure_selection() {
  ensure_bank
  if [[ "${FORCE_REBUILD}" == "1" || ! -f "${SELECTION_PATH}" ]]; then
    select_eta
  else
    echo "Reusing DA gain eta selection cache: ${SELECTION_PATH}"
  fi
}

case "${MODE}" in
  build-bank)
    if [[ "${FORCE_REBUILD}" == "1" || ! -f "${BANK_PATH}" ]]; then
      build_bank
    else
      echo "Reusing DA gain bank cache: ${BANK_PATH}"
    fi
    ;;
  select-eta)
    ensure_bank
    if [[ "${FORCE_REBUILD}" == "1" || ! -f "${SELECTION_PATH}" ]]; then
      select_eta
    else
      echo "Reusing DA gain eta selection cache: ${SELECTION_PATH}"
    fi
    ;;
  eval-source)
    ensure_selection
    eval_split source_val "${OUTPUT_BASE}/source_val_eval"
    ;;
  eval-target)
    ensure_selection
    eval_split target_eval "${OUTPUT_BASE}/target_eval"
    ;;
  all)
    ensure_selection
    eval_split source_val "${OUTPUT_BASE}/source_val_eval"
    ;;
  *)
    echo "ERROR: unknown mode ${MODE}. Expected build-bank, select-eta, eval-source, eval-target, all." >&2
    exit 2
    ;;
esac

echo "M3_10a artifacts: ${OUTPUT_BASE}"
