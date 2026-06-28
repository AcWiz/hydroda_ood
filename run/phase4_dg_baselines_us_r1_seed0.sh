#!/usr/bin/env bash
# Phase 4 source-only DG baseline training wrapper.
#
# Runs V4.4 US-only LORO zero-shot source-side DG baselines through
# scripts/train/train_source_only_backbone.py. Target-context methods are
# explicit diagnostics and are not included in the default method list.

set -euo pipefail

TARGET_REGION="US-R1"
SEED="0"
CUDA_DEVICE="0"
RUN_MODE="dev"
DRY_RUN="${DRY_RUN:-0}"
METHOD_LIST=(swad mixstyle disam udim moment_align iu)

usage() {
    cat <<'EOF'
Usage: bash run/phase4_dg_baselines_us_r1_seed0.sh [options]

Options:
  --dry-run                 Print commands without executing.
  --full                    Use full run settings. Default is dev settings.
  --method-list METHODS...  Methods: swad mixstyle disam udim moment_align iu ssa_reg tca self_bootstrap.
  --target-region REGION    Target region. Default US-R1.
  --seed SEED               Split seed. Default 0.
  --cuda-device ID          CUDA_VISIBLE_DEVICES value. Default 0.
  --help                    Show this help.

Environment overrides:
  DRY_RUN=1
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

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

if [[ "${RUN_MODE}" == "full" ]]; then
    BATCH_SIZE="${BATCH_SIZE:-8}"
    MAX_EPOCHS="${MAX_EPOCHS:-50}"
    LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-100}"
else
    BATCH_SIZE="${BATCH_SIZE:-2}"
    MAX_EPOCHS="${MAX_EPOCHS:-1}"
    LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-10}"
fi

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

append_method_params() {
    local method_name="$1"
    case "${method_name}" in
        swad)
            CMD+=(--swad_start_epoch 10 --swad_tolerance 0.02 --swad_patience 3)
            ;;
        mixstyle)
            CMD+=(--mixstyle_p 0.5 --mixstyle_alpha 0.1 --mixstyle_layers enc1,enc2)
            ;;
        disam)
            CMD+=(--disam_rho 0.05 --disam_lambda 0.1)
            ;;
        udim)
            CMD+=(--udim_rho 0.05 --udim_lambda 0.1)
            ;;
        moment_align)
            CMD+=(--moment_align_lambda 0.01 --moment_align_feature_layer bottleneck --moment_align_order 2)
            ;;
        iu)
            CMD+=(
                --iu_lambda 0.001
                --iu_feature_layer bottleneck
                --iu_top_fraction 0.25
                --iu_sample_top_fraction 0.5
                --iu_score_cap 10.0
            )
            ;;
        ssa_reg)
            CMD+=(--ssa_reg_lambda 0.01 --ssa_reg_feature_layer bottleneck --ssa_reg_rank 8)
            ;;
        tca)
            CMD+=(--tca_lambda 0.01 --tca_feature_layer bottleneck)
            ;;
        self_bootstrap)
            CMD+=(
                --self_bootstrap_lambda 0.01
                --self_bootstrap_noise_std 0.01
                --self_bootstrap_channel_dropout_p 0.05
            )
            ;;
    esac
}

echo "============================================"
echo "Phase 4 DG Baseline Training"
echo "  DRY_RUN=${DRY_RUN}"
echo "  run_mode=${RUN_MODE}"
echo "  target_region=${TARGET_REGION}"
echo "  seed=${SEED}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  methods=${METHOD_LIST[*]}"
echo "  source_fit=2015-2021 source_val=2022"
echo "  target_eval=2023-2025 final offline evaluation"
echo "  target_context=unused for default source-only DG methods"
echo "  source_domain_grouping=pooled sample masks; no 5x source-region episode expansion"
echo "============================================"

for METHOD_NAME in "${METHOD_LIST[@]}"; do
    METHOD_ID="$(method_id_for "${METHOD_NAME}")"
    OUTPUT_DIR="artifacts/runs/phase4_dg_baselines/${METHOD_ID}/${TARGET_REGION}_s${SEED}_${RUN_MODE}"

    CMD=(
        env PYTHONPATH=. python scripts/train/train_source_only_backbone.py
        --target_region "${TARGET_REGION}"
        --adaptation_setting zero_shot_context --K 0
        --seed "${SEED}"
        --device cuda
        --amp
        --zero_raw_increment_init
        --target_increment_normalization
        --use_lat_weighted_loss
        --batch_size "${BATCH_SIZE}"
        --max_epochs "${MAX_EPOCHS}"
        --lr 3e-4
        --weight_decay 1e-4
        --grad_clip 1.0
        --num_workers 0
        --width 32
        --log_every_steps "${LOG_EVERY_STEPS}"
        --eval_every_epochs 1
        --checkpoint_every 10
        --selection_metric source_val_loss
        --dg_method "${METHOD_NAME}"
        --output_dir "${OUTPUT_DIR}"
    )
    append_method_params "${METHOD_NAME}"

    echo
    echo "method_id=${METHOD_ID}"
    echo "output_dir=${OUTPUT_DIR}"
    printf 'command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'

    if [[ "${DRY_RUN}" != "1" ]]; then
        mkdir -p "${OUTPUT_DIR}"
        "${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/console_train.log"
    fi
done
