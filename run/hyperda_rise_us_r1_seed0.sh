#!/usr/bin/env bash
# HyperDA-RISE internal diagnostic wrapper.
# Status: internal_diagnostic_not_paper_main.
# Retrieval-Informed Self-supervised Expert Operator Composition.

set -euo pipefail

TARGET_REGION="${TARGET_REGION:-US-R1}"
SEED="${SEED:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
K_LIST="${K_LIST:-0 4 12}"
DRY_RUN=0
RUN_MODE="smoke"
MAX_CONTEXT_SAMPLES="${MAX_CONTEXT_SAMPLES:-2}"
MAX_SUPPORT_SAMPLES="${MAX_SUPPORT_SAMPLES:-2}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-2}"
TEMPERATURE="${TEMPERATURE:-0.2}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1.0}"
OUTPUT_BASE=""

DEFAULT_SOURCE_POOLED_CHECKPOINT="${DEFAULT_SOURCE_POOLED_CHECKPOINT:-artifacts/runs/phase4_source_only/latest/checkpoints/checkpoint_best_source_val_safe_score.pt}"
DEFAULT_HYPERDA_K0_CHECKPOINT="${DEFAULT_HYPERDA_K0_CHECKPOINT:-artifacts/runs/phase4_hyperda_staged/latest/checkpoints/checkpoint_best_source_val_safe_score.pt}"
DEFAULT_CANDIDATE_SPECS="${DEFAULT_CANDIDATE_SPECS:-forecast:forecast_only:zero source_pooled:source_only:${DEFAULT_SOURCE_POOLED_CHECKPOINT} hyperda_k0:prompt_conditioned:${DEFAULT_HYPERDA_K0_CHECKPOINT}}"
CANDIDATE_SPECS="${CANDIDATE_SPECS:-${DEFAULT_CANDIDATE_SPECS}}"

usage() {
    cat <<'EOF'
Usage: bash run/hyperda_rise_us_r1_seed0.sh [options]

HyperDA-RISE diagnostic wrapper. The defaults are embedded in this script.

Options:
  --dry-run                    Print commands without executing.
  --full                       Use all context/support/eval samples.
  --target-region REGION       Target region. Default US-R1.
  --seed SEED                  Split seed. Default 0.
  --cuda-device ID             CUDA_VISIBLE_DEVICES value. Default 0.
  --k-list "0 4 12"            K values to evaluate.
  --candidate-specs SPECS      Space-separated expert_id:predictor_type:checkpoint specs.
  --max-context-samples N      Smoke cap; 0 means full.
  --max-support-samples N      Smoke cap; 0 means full.
  --max-eval-samples N         Smoke cap; 0 means full.
  --output-base DIR            Output directory.
  --help                       Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --full)
            RUN_MODE="full"
            MAX_CONTEXT_SAMPLES=0
            MAX_SUPPORT_SAMPLES=0
            MAX_EVAL_SAMPLES=0
            shift
            ;;
        --target-region)
            TARGET_REGION="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --cuda-device)
            CUDA_DEVICE="$2"
            shift 2
            ;;
        --k-list)
            K_LIST="$2"
            shift 2
            ;;
        --candidate-specs)
            CANDIDATE_SPECS="$2"
            shift 2
            ;;
        --max-context-samples)
            MAX_CONTEXT_SAMPLES="$2"
            shift 2
            ;;
        --max-support-samples)
            MAX_SUPPORT_SAMPLES="$2"
            shift 2
            ;;
        --max-eval-samples)
            MAX_EVAL_SAMPLES="$2"
            shift 2
            ;;
        --output-base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

if [[ -z "${OUTPUT_BASE}" ]]; then
    OUTPUT_BASE="artifacts/runs/hyperda_rise/${TARGET_REGION}_s${SEED}_${RUN_MODE}"
fi

echo "============================================"
echo "HyperDA-RISE"
echo "  status=internal_diagnostic_not_paper_main"
echo "  Retrieval-Informed Self-supervised Expert Operator Composition"
echo "  DRY_RUN=${DRY_RUN}"
echo "  run_mode=${RUN_MODE}"
echo "  target_region=${TARGET_REGION}"
echo "  seed=${SEED}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  K_LIST=${K_LIST}"
echo "  output_base=${OUTPUT_BASE}"
echo "  max_context_samples=${MAX_CONTEXT_SAMPLES}"
echo "  max_support_samples=${MAX_SUPPORT_SAMPLES}"
echo "  max_eval_samples=${MAX_EVAL_SAMPLES}"
echo "  target_context=2015-2021 input-side only"
echo "  source_val 2022 candidate WRMSE"
echo "  target_support=K labeled DA cycles"
echo "  target_val=unused_in_main_protocol"
echo "  target_eval=2023-2025 final offline evaluation"
echo "  DEFAULT_SOURCE_POOLED_CHECKPOINT=${DEFAULT_SOURCE_POOLED_CHECKPOINT}"
echo "  DEFAULT_HYPERDA_K0_CHECKPOINT=${DEFAULT_HYPERDA_K0_CHECKPOINT}"
echo "  DEFAULT_CANDIDATE_SPECS=${DEFAULT_CANDIDATE_SPECS}"
echo "  CANDIDATE_SPECS=${CANDIDATE_SPECS}"
echo "============================================"

if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${OUTPUT_BASE}/router"
fi

TRAIN_CMD=(
    env PYTHONPATH=. python scripts/train/train_hyperda_rise_router.py
    --target_region "${TARGET_REGION}"
    --seed "${SEED}"
    --output_dir "${OUTPUT_BASE}/router"
    --temperature "${TEMPERATURE}"
    --max_context_samples "${MAX_CONTEXT_SAMPLES}"
    --max_eval_samples "${MAX_EVAL_SAMPLES}"
    --device "cuda"
)
for SPEC in ${CANDIDATE_SPECS}; do
    TRAIN_CMD+=(--candidate "${SPEC}")
done

printf 'train_command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'
if [[ "${DRY_RUN}" != "1" ]]; then
    "${TRAIN_CMD[@]}"
fi

for K in ${K_LIST}; do
    EVAL_DIR="${OUTPUT_BASE}/K${K}"
    EVAL_CMD=(
        env PYTHONPATH=. python scripts/eval/eval_hyperda_rise.py
        --router_prior "${OUTPUT_BASE}/router/router_prior.json"
        --target_region "${TARGET_REGION}"
        --K "${K}"
        --seed "${SEED}"
        --output_dir "${EVAL_DIR}"
        --device "cuda"
        --max_eval_samples "${MAX_EVAL_SAMPLES}"
        --max_context_samples "${MAX_CONTEXT_SAMPLES}"
        --max_support_samples "${MAX_SUPPORT_SAMPLES}"
        --ridge_lambda "${RIDGE_LAMBDA}"
        --temperature "${TEMPERATURE}"
    )
    printf 'eval_command_K%s:' "${K}"
    printf ' %q' "${EVAL_CMD[@]}"
    printf '\n'
    if [[ "${DRY_RUN}" != "1" ]]; then
        "${EVAL_CMD[@]}"
    fi
done
