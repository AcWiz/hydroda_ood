#!/bin/bash
# Phase 5: HyperDA zero/few-shot target generalization.
#
# Main protocol:
#   source_fit=2015-2021
#   source_val=2022 for preregistered checkpoint/hyperparameter selection
#   target_context=2015-2021 input-side only
#   target_support=K labeled cycles for K in {0,4,12}
#   target_val=unused in main protocol
#   target_eval=2023-2025 final offline evaluation
#
# Usage:
#   bash run/phase5_hyperda_zero_few_shot.sh <source_checkpoint> US-R1 4 0 1
#   MAX_STEPS=1 bash run/phase5_hyperda_zero_few_shot.sh <source_checkpoint> US-R1 4 0 1

set -euo pipefail

SOURCE_CHECKPOINT="${1:-}"
TARGET_REGION="${2:-US-R1}"
K="${3:-0}"
SEED="${4:-0}"
export CUDA_VISIBLE_DEVICES="${5:-1}"
EXTRA_ARGS=()
if [[ "$#" -gt 5 ]]; then
    EXTRA_ARGS=("${@:6}")
fi

cd "$(dirname "$0")/.."

if [[ -z "${SOURCE_CHECKPOINT}" ]]; then
    SOURCE_CHECKPOINT="$(find "artifacts/runs/phase4_hyperda_staged/${TARGET_REGION}" \
        -path "*s${SEED}*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
        -type f 2>/dev/null | sort | tail -1)"
fi

if [[ -z "${SOURCE_CHECKPOINT}" ]]; then
    SOURCE_CHECKPOINT="$(find artifacts/runs/phase4_prompt_conditioned \
        -path "*hyperda_basis_adapter_${TARGET_REGION}_*_s${SEED}_*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
        -type f 2>/dev/null | sort | tail -1)"
fi

if [[ -z "${SOURCE_CHECKPOINT}" || ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "ERROR: source HyperDA checkpoint not found." >&2
    echo "Provide it explicitly:" >&2
    echo "  bash run/phase5_hyperda_zero_few_shot.sh <source_checkpoint> ${TARGET_REGION} ${K} ${SEED} ${CUDA_VISIBLE_DEVICES}" >&2
    echo "Or train the staged source prior first:" >&2
    echo "  bash run/phase4_hyperda_staged.sh auto ${TARGET_REGION} ${SEED} ${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
fi

if [[ "${K}" == "0" ]]; then
    ADAPTATION_SETTING="zero_shot_context"
    DEFAULT_STEPS=0
    DEFAULT_LR="1e-3"
    DEFAULT_ANCHOR_ALPHA="0.0"
elif [[ "${K}" == "4" ]]; then
    ADAPTATION_SETTING="few_shot_k4"
    DEFAULT_STEPS=100
    DEFAULT_LR="${LR_K4:-1e-3}"
    DEFAULT_ANCHOR_ALPHA="${ANCHOR_ALPHA_K4:-0.75}"
elif [[ "${K}" == "12" ]]; then
    ADAPTATION_SETTING="few_shot_k12"
    DEFAULT_STEPS=80
    DEFAULT_LR="${LR_K12:-3e-4}"
    DEFAULT_ANCHOR_ALPHA="${ANCHOR_ALPHA_K12:-0.25}"
else
    echo "ERROR: K must be one of 0, 4, 12; got ${K}" >&2
    exit 2
fi

ADAPTATION_STEPS="${ADAPT_MAX_STEPS:-${MAX_STEPS:-${ADAPTATION_STEPS:-${DEFAULT_STEPS}}}}"
ADAPT_RECIPE="${ADAPT_RECIPE:-source_anchor}"
ANCHOR_ALPHA="${ADAPT_ANCHOR_ALPHA:-${ANCHOR_ALPHA:-${DEFAULT_ANCHOR_ALPHA}}}"
ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-${BATCH_SIZE:-8}}"
LR="${ADAPT_LR:-${LR:-${DEFAULT_LR}}}"
WEIGHT_DECAY="${ADAPT_WEIGHT_DECAY:-${WEIGHT_DECAY:-1e-4}}"
GRAD_CLIP="${ADAPT_GRAD_CLIP:-${GRAD_CLIP:-1.0}}"
LAMBDA_PRIOR="${ADAPT_LAMBDA_PRIOR:-${LAMBDA_PRIOR:-1e-3}}"
LAMBDA_LATENT="${ADAPT_LAMBDA_LATENT:-${LAMBDA_LATENT:-1e-3}}"
LAMBDA_GAIN="${ADAPT_LAMBDA_GAIN:-${LAMBDA_GAIN:-1e-2}}"
LAMBDA_GAIN_SMOOTH="${ADAPT_LAMBDA_GAIN_SMOOTH:-${LAMBDA_GAIN_SMOOTH:-1e-3}}"
LAMBDA_ANALYSIS="${ADAPT_LAMBDA_ANALYSIS:-${LAMBDA_ANALYSIS:-0.25}}"
SCHEDULE_LABEL="${SCHEDULE_LABEL:-}"
TARGET_LATENT_DIM="${TARGET_LATENT_DIM:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
SPLITS_JSON="${SPLITS_JSON:-artifacts/splits/US_loro_zero_few_shot_splits.json}"
ADAPT_SCOPE="${ADAPT_SCOPE:-safe_operator}"
STAGE3_POSTERIOR_POLICY="${STAGE3_POSTERIOR_POLICY:-safe_operator_ablation}"
SUPPORT_GATE="${SUPPORT_GATE:-policy_default}"
SUPPORT_GATE_MIN_DELTA="${SUPPORT_GATE_MIN_DELTA:-0.0}"
SUPPORT_GATE_ROOTZONE_TOLERANCE="${SUPPORT_GATE_ROOTZONE_TOLERANCE:-0.0}"
SUPPORT_LOSS_REDUCTION="${SUPPORT_LOSS_REDUCTION:-global_pixel}"
FREEZE_MONTHLY_GAIN="${FREEZE_MONTHLY_GAIN:-0}"
SAFE_POLICY_JSON="${SAFE_POLICY_JSON:-}"
REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT:-0}"
AUDIT_IDENTITY="${AUDIT_IDENTITY:-0}"
AUDIT_IDENTITY_TOLERANCE="${AUDIT_IDENTITY_TOLERANCE:-1e-8}"
TARGET_CONTEXT_MAX_SAMPLES="${TARGET_CONTEXT_MAX_SAMPLES:-0}"
AUDIT_ARGS=()
if [[ "${AUDIT_IDENTITY}" == "1" || "${AUDIT_IDENTITY,,}" == "true" ]]; then
    AUDIT_ARGS=(--audit_identity)
fi
POLICY_ARGS=()
if [[ -n "${SAFE_POLICY_JSON}" ]]; then
    POLICY_ARGS+=(--safe_policy_json "${SAFE_POLICY_JSON}")
fi
if [[ "${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT}" == "1" || "${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT,,}" == "true" ]]; then
    POLICY_ARGS+=(--require_safe_policy_json_for_kshot)
fi

echo "============================================"
echo "Phase 5 HyperDA Zero/Few-Shot Generalization"
echo "  source_checkpoint=${SOURCE_CHECKPOINT}"
echo "  source_stage_checkpoint_provenance=phase4_hyperda_staged"
echo "  target_region=${TARGET_REGION}"
echo "  K=${K} adaptation_setting=${ADAPTATION_SETTING}"
echo "  seed=${SEED}"
echo "  target_context=2015-2021 input-side only"
echo "  target_support=K labeled cycles"
echo "  target_val=unused_in_main_protocol"
echo "  target_eval=2023-2025 final offline evaluation"
echo "  split_artifact=artifacts/splits/US_loro_zero_few_shot_splits.json"
echo "  active_splits_json=${SPLITS_JSON}"
echo "  model_selection_source=source_val_preregistered"
echo "  adapt_recipe=${ADAPT_RECIPE} anchor_alpha=${ANCHOR_ALPHA}"
echo "  adapt_scope=${ADAPT_SCOPE} (SAFE Prompt+Coeff+Gain unless conservative Stage 3 overrides) adapt_solver=adamw audit_identity=${AUDIT_IDENTITY}"
echo "  stage3_posterior_policy=${STAGE3_POSTERIOR_POLICY} support_gate=${SUPPORT_GATE} min_delta=${SUPPORT_GATE_MIN_DELTA} rootzone_tolerance=${SUPPORT_GATE_ROOTZONE_TOLERANCE}"
echo "  freeze_monthly_gain=${FREEZE_MONTHLY_GAIN}"
echo "  safe_policy_json=${SAFE_POLICY_JSON:-<none>} require_safe_policy_json_for_kshot=${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT}"
echo "  policy_source=source_side_episode_calibration when SAFE_POLICY_JSON is provided"
echo "  target_context_max_samples=${TARGET_CONTEXT_MAX_SAMPLES} (0 = full target_context)"
echo "  support_loss_reduction=${SUPPORT_LOSS_REDUCTION}"
echo "  schedule_label=${SCHEDULE_LABEL}"
echo "  adaptation_steps=${ADAPTATION_STEPS} adapt_batch_size=${ADAPT_BATCH_SIZE} lr=${LR} weight_decay=${WEIGHT_DECAY} grad_clip=${GRAD_CLIP}"
echo "  lambda_prior=${LAMBDA_PRIOR} lambda_latent=${LAMBDA_LATENT} lambda_gain=${LAMBDA_GAIN} lambda_gain_smooth=${LAMBDA_GAIN_SMOOTH} lambda_analysis=${LAMBDA_ANALYSIS}"
echo "  output_dir=${OUTPUT_DIR:-auto}"
echo "============================================"

PYTHONPATH=. python scripts/train/train_hyperda_few_shot_adapt.py \
    --source_checkpoint "${SOURCE_CHECKPOINT}" \
    --target_region "${TARGET_REGION}" \
    --K "${K}" \
    --adaptation_setting "${ADAPTATION_SETTING}" \
    --seed "${SEED}" \
    --device cuda \
    --target_latent_dim "${TARGET_LATENT_DIM}" \
    --splits_json "${SPLITS_JSON}" \
    --adaptation_steps "${ADAPTATION_STEPS}" \
    --schedule_label "${SCHEDULE_LABEL}" \
    --adapt_recipe "${ADAPT_RECIPE}" \
    --anchor_alpha "${ANCHOR_ALPHA}" \
    --adapt_scope "${ADAPT_SCOPE}" \
    --stage3_posterior_policy "${STAGE3_POSTERIOR_POLICY}" \
    --support_gate "${SUPPORT_GATE}" \
    --support_gate_min_delta "${SUPPORT_GATE_MIN_DELTA}" \
    --support_gate_rootzone_tolerance "${SUPPORT_GATE_ROOTZONE_TOLERANCE}" \
    --adapt_solver adamw \
    --support_loss_reduction "${SUPPORT_LOSS_REDUCTION}" \
    --audit_identity_tolerance "${AUDIT_IDENTITY_TOLERANCE}" \
    --target_context_max_samples "${TARGET_CONTEXT_MAX_SAMPLES}" \
    $(if [[ "${FREEZE_MONTHLY_GAIN}" == "1" || "${FREEZE_MONTHLY_GAIN,,}" == "true" ]]; then echo "--freeze_monthly_gain"; fi) \
    "${POLICY_ARGS[@]}" \
    --batch_size "${ADAPT_BATCH_SIZE}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --grad_clip "${GRAD_CLIP}" \
    --lambda_prior "${LAMBDA_PRIOR}" \
    --lambda_latent "${LAMBDA_LATENT}" \
    --lambda_gain "${LAMBDA_GAIN}" \
    --lambda_gain_smooth "${LAMBDA_GAIN_SMOOTH}" \
    --lambda_analysis "${LAMBDA_ANALYSIS}" \
    --num_workers "${NUM_WORKERS}" \
    --use_lat_weighted_loss \
    $(if [[ -n "${OUTPUT_DIR}" ]]; then echo "--output_dir ${OUTPUT_DIR}"; fi) \
    "${AUDIT_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
