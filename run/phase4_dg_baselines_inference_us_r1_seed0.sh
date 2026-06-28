#!/usr/bin/env bash
# Phase 4 DG baseline inference wrapper.
#
# Default target: evaluate completed SWAD and MixStyle US-R1 seed0 full runs.
# Runs source_test and target_eval through scripts/eval/evaluate_checkpoint.py.

set -euo pipefail

TARGET_REGION="US-R1"
SEED="0"
CUDA_DEVICE="0"
METHOD_LIST=(swad mixstyle)
SPLIT_LIST=(source_test target_eval)
RUN_MODE="full"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OUTPUT_LEVEL="${EVAL_OUTPUT_LEVEL:-compact}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
    cat <<'EOF'
Usage: bash run/phase4_dg_baselines_inference_us_r1_seed0.sh [options]

Options:
  --dry-run                 Print commands without executing.
  --method-list METHODS...  Methods: swad mixstyle disam udim moment_align iu ssa_reg tca self_bootstrap.
  --splits SPLITS...        Splits: source_test target_eval. Default both.
  --target-region REGION   Target region. Default US-R1.
  --seed SEED              Split seed. Default 0.
  --cuda-device ID         CUDA_VISIBLE_DEVICES value. Default 0.
  --max-samples N          Evaluation sample cap. Default 0 = all.
  --output-level LEVEL     compact, long, or full. Default compact.
  --help                   Show this help.

Environment overrides:
  EVAL_BATCH_SIZE=8
  MAX_SAMPLES=0
  EVAL_OUTPUT_LEVEL=compact
  DRY_RUN=1
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --method-list)
            shift
            METHOD_LIST=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                METHOD_LIST+=("$1")
                shift
            done
            ;;
        --splits)
            shift
            SPLIT_LIST=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SPLIT_LIST+=("$1")
                shift
            done
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
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --output-level)
            OUTPUT_LEVEL="$2"
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

if [[ ${#METHOD_LIST[@]} -eq 0 ]]; then
    echo "ERROR: --method-list must include at least one method." >&2
    exit 2
fi
if [[ ${#SPLIT_LIST[@]} -eq 0 ]]; then
    echo "ERROR: --splits must include at least one split." >&2
    exit 2
fi

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

method_id_for() {
    case "$1" in
        swad) echo "swad_source_pooled_global_backbone" ;;
        mixstyle) echo "mixstyle_source_pooled_global_backbone" ;;
        disam) echo "disam_source_domain_sharpness_alignment" ;;
        udim) echo "udim_unknown_domain_inconsistency_minimization" ;;
        moment_align) echo "moment_alignment_source_domain_invariance" ;;
        iu) echo "identify_unlearn_source_domain_gradient_ascent" ;;
        ssa_reg) echo "ssa_reg_target_context_subspace_alignment" ;;
        tca) echo "tca_target_context_correlation_alignment" ;;
        self_bootstrap) echo "self_bootstrap_target_context_consistency_tta" ;;
        *)
            echo "ERROR: unsupported method '$1'. Supported: swad mixstyle disam udim moment_align iu ssa_reg tca self_bootstrap" >&2
            exit 2
            ;;
    esac
}

checkpoint_name_for() {
    case "$1" in
        swad) echo "checkpoint_swad.pt" ;;
        mixstyle) echo "checkpoint_best_source_val_safe_score.pt" ;;
        disam|udim|moment_align|iu|ssa_reg|tca|self_bootstrap) echo "best.pt" ;;
        *)
            echo "ERROR: unsupported method '$1'" >&2
            exit 2
            ;;
    esac
}

find_checkpoint() {
    local method_name="$1"
    local method_id="$2"
    local preferred_name
    preferred_name="$(checkpoint_name_for "${method_name}")"
    local run_dir="artifacts/runs/phase4_dg_baselines/${method_id}/${TARGET_REGION}_s${SEED}_${RUN_MODE}"
    local preferred="${run_dir}/checkpoints/${preferred_name}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '%s\n' "${preferred}"
        return 0
    fi
    if [[ -f "${preferred}" ]]; then
        printf '%s\n' "${preferred}"
        return 0
    fi
    local fallback="${run_dir}/checkpoints/checkpoint_best_source_val_safe_score.pt"
    if [[ -f "${fallback}" ]]; then
        printf '%s\n' "${fallback}"
        return 0
    fi
    fallback="${run_dir}/checkpoints/best.pt"
    if [[ -f "${fallback}" ]]; then
        printf '%s\n' "${fallback}"
        return 0
    fi
    echo "ERROR: no checkpoint found for ${method_id} at ${run_dir}/checkpoints" >&2
    return 1
}

echo "============================================"
echo "Phase 4 DG Baseline Inference"
echo "  DRY_RUN=${DRY_RUN}"
echo "  target_region=${TARGET_REGION}"
echo "  seed=${SEED}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  methods=${METHOD_LIST[*]}"
echo "  splits=${SPLIT_LIST[*]}"
echo "  eval_batch_size=${EVAL_BATCH_SIZE}"
echo "  max_samples=${MAX_SAMPLES}"
echo "  output_level=${OUTPUT_LEVEL}"
echo "  target_eval=2023-2025 final offline evaluation"
echo "============================================"

for METHOD_NAME in "${METHOD_LIST[@]}"; do
    METHOD_ID="$(method_id_for "${METHOD_NAME}")"
    CHECKPOINT_PATH="$(find_checkpoint "${METHOD_NAME}" "${METHOD_ID}")"
    CHECKPOINT_STEM="$(basename "${CHECKPOINT_PATH}" .pt)"
    OUTPUT_BASE="artifacts/runs/phase4_dg_baselines/${METHOD_ID}/${TARGET_REGION}_s${SEED}_${RUN_MODE}/results/${CHECKPOINT_STEM}"

    echo
    echo "method_id=${METHOD_ID}"
    echo "checkpoint=${CHECKPOINT_PATH}"
    echo "output_base=${OUTPUT_BASE}"

    for SPLIT_TYPE in "${SPLIT_LIST[@]}"; do
        case "${SPLIT_TYPE}" in
            source_test|target_eval) ;;
            *)
                echo "ERROR: unsupported split '${SPLIT_TYPE}'. Supported: source_test target_eval" >&2
                exit 2
                ;;
        esac
        EVAL_DIR="${OUTPUT_BASE}/${SPLIT_TYPE}"
        CMD=(
            env PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py
            --checkpoint "${CHECKPOINT_PATH}"
            --target_region "${TARGET_REGION}"
            --adaptation_setting zero_shot_context --K 0
            --seed "${SEED}"
            --split_type "${SPLIT_TYPE}"
            --predictor_type source_only
            --device cuda
            --batch_size "${EVAL_BATCH_SIZE}"
            --max_samples "${MAX_SAMPLES}"
            --output_level "${OUTPUT_LEVEL}"
            --output_dir "${EVAL_DIR}"
        )
        echo
        printf 'command:'
        printf ' %q' "${CMD[@]}"
        printf '\n'
        if [[ "${DRY_RUN}" != "1" ]]; then
            mkdir -p "${OUTPUT_BASE}"
            "${CMD[@]}" 2>&1 | tee "${OUTPUT_BASE}/log_${SPLIT_TYPE}.txt"
        fi
    done
done
