#!/bin/bash
# Phase 4C staged HyperDA / HyperDA-TRUST source-side ablation wrapper.
#
# Usage:
#   ABLATION_ID=M2_rank_gated_dora bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M2_1_rank_gated_dora_stable bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M2_2_source_saliency_prior bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M2_3_source_safe_residual_hyperda bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M2_5a_da_aware_prompt_only bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M2_5b_da_aware_conservative_router bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M2_6_source_manifold_guarded_prior bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M3_0_hyperda_trust_light bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M3_1a_trust_medium_dualalpha bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M3_2_hyperda_trust_raw_reliability bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M3_2_phys_trust_raw_da_query bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M3_2a_phys_trust_raw_query_fixed bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M3_4_phys_trust_blended_query bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M3_5_phys_agreement_guarded_trust bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M3_5b_phys_agreement_floor_guard bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M3_6_phys_token_operator_droppath_trust RESUME_FROM_M3_1_BEST=1 TRAINABLE_SCOPE=phys_context_only MAX_EPOCHS=5 EVAL_EVERY_EPOCHS=5 bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M3_7_phys_formula_consistency_guarded_trust RESUME_FROM_M3_1_BEST=1 TRAINABLE_SCOPE=none MAX_EPOCHS=0 EVAL_EVERY_EPOCHS=1 bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M3_8_phys_formula_operator_trust RESUME_FROM_M3_1_BEST=1 TRAINABLE_SCOPE=phys_formula_context_only MAX_EPOCHS=3 EVAL_EVERY_EPOCHS=1 LR=1e-4 USE_AMP=0 bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M3_8b_phys_formula_light_guarded_trust bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M3_8c_phys_formula_light_operator_only_trust bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M3_9_phys_formula_enhanced_trust RESUME_FROM_M3_1_BEST=1 TRAINABLE_SCOPE=phys_formula_context_only MAX_EPOCHS=3 EVAL_EVERY_EPOCHS=1 LR=1e-4 USE_AMP=0 SOURCE_FIT_MAX_BATCHES_PER_EPOCH=384 bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M3_12_phys_gain_basis_hypertrust bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M3_13_phys_gain_guarded_hypertrust RESUME_FROM_M3_1_BEST=1 TRAINABLE_SCOPE=phys_gain_guard_only MAX_EPOCHS=0 EVAL_EVERY_EPOCHS=1 bash run/phase4_hyperda_staged_ablation.sh auto US-R2 0 0
#   ABLATION_ID=M3_14_source_trained_phys_formula_gain_hypertrust bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M3_15_m31_anchored_source_safe_phys_coeff_delta bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M3_16_source_only_phys_m3trust_lite bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M3_3_hyperda_trust_selection_only bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   ABLATION_ID=M2_2_source_saliency_prior HYPER_SOURCE_SALIENCY_PRIOR_PATH=/path/prior.pt bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
#   bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0 --dry-run

set -euo pipefail

DRY_RUN=0
POSITIONAL=()
for arg in "$@"; do
    case "${arg}" in
        --dry-run)
            DRY_RUN=1
            ;;
        *)
            POSITIONAL+=("${arg}")
            ;;
    esac
done

if [[ "${#POSITIONAL[@]}" -gt 4 ]]; then
    echo "ERROR: Too many positional arguments: ${POSITIONAL[*]}" >&2
    exit 2
fi

SOURCE_CHECKPOINT="${POSITIONAL[0]:-auto}"
TARGET_REGION="${POSITIONAL[1]:-US-R1}"
SEED="${POSITIONAL[2]:-0}"
export CUDA_VISIBLE_DEVICES="${POSITIONAL[3]:-0}"

cd "$(dirname "$0")/.."
export PYTHONPATH=".:${PYTHONPATH:-}"

ABLATION_ID="${ABLATION_ID:-M0_current}"
SPLITS_JSON="${SPLITS_JSON:-artifacts/splits/US_loro_zero_few_shot_splits.json}"
SPLIT_MANIFEST_PATH="${SPLIT_MANIFEST_PATH:-${SPLITS_JSON}}"
PROTOCOL_ID="${PROTOCOL_ID:-hyperda_v4_4_zero_few_shot_generalization_2015_2025_context2015_2021_sourceval2022_eval2023_2025}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/runs/phase4_hyperda_staged_ablation}"
BUILD_ABLATION_TABLE="${BUILD_ABLATION_TABLE:-1}"
ABLATION_TABLE_OUTPUT_DIR="${ABLATION_TABLE_OUTPUT_DIR:-reports/ablations/hyperda_staged_v1}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
WIDTH="${WIDTH:-32}"
PROMPT_DIM="${PROMPT_DIM:-64}"
HYPER_N_BASIS="${HYPER_N_BASIS:-8}"
HYPER_ADAPTER_BOTTLENECK="${HYPER_ADAPTER_BOTTLENECK:-32}"
HYPER_ADAPTER_SCALE="${HYPER_ADAPTER_SCALE:-1.0}"
HYPER_RANK_GATE_TOP_K="${HYPER_RANK_GATE_TOP_K:-4}"
USER_SET_HYPER_RANK_GATE_TEMPERATURE_INIT="${HYPER_RANK_GATE_TEMPERATURE_INIT+x}"
HYPER_RANK_GATE_TEMPERATURE_INIT="${HYPER_RANK_GATE_TEMPERATURE_INIT:-1.0}"
HYPER_ADAPTER_PARAM_STYLE="${HYPER_ADAPTER_PARAM_STYLE:-basis_1x1}"
HYPER_RELIABILITY_INIT="${HYPER_RELIABILITY_INIT:-0.95}"
HYPER_SOURCE_SALIENCY_PRIOR_PATH="${HYPER_SOURCE_SALIENCY_PRIOR_PATH:-}"
HYPER_SOURCE_SALIENCY_PRIOR_DIR="${HYPER_SOURCE_SALIENCY_PRIOR_DIR:-artifacts/priors/source_basis_saliency}"
HYPER_SOURCE_SALIENCY_AUTO_BUILD="${HYPER_SOURCE_SALIENCY_AUTO_BUILD:-1}"
HYPER_SOURCE_SALIENCY_MAX_BATCHES="${HYPER_SOURCE_SALIENCY_MAX_BATCHES:-16}"
HYPER_SOURCE_SALIENCY_BATCH_SIZE="${HYPER_SOURCE_SALIENCY_BATCH_SIZE:-4}"
HYPER_SOURCE_SALIENCY_PRIOR_BETA="${HYPER_SOURCE_SALIENCY_PRIOR_BETA:-0.0}"
HYPER_SOURCE_SALIENCY_PRIOR_APPLICATION="${HYPER_SOURCE_SALIENCY_PRIOR_APPLICATION:-soft_regularization_metadata}"
HYPER_PROMPT_MANIFOLD_RELIABILITY="${HYPER_PROMPT_MANIFOLD_RELIABILITY:-0}"
HYPER_PROMPT_MANIFOLD_RELIABILITY_STRENGTH="${HYPER_PROMPT_MANIFOLD_RELIABILITY_STRENGTH:-0.0}"
HYPER_SOURCE_MANIFOLD_GUARD="${HYPER_SOURCE_MANIFOLD_GUARD:-0}"
HYPER_SOURCE_MANIFOLD_GUARD_STRENGTH="${HYPER_SOURCE_MANIFOLD_GUARD_STRENGTH:-0.25}"
HYPER_SOURCE_MANIFOLD_GUARD_DISTANCE_KEY="${HYPER_SOURCE_MANIFOLD_GUARD_DISTANCE_KEY:-source_manifold_distance_bounded}"
HYPER_SOURCE_MANIFOLD_GUARD_MIN_MULTIPLIER="${HYPER_SOURCE_MANIFOLD_GUARD_MIN_MULTIPLIER:-0.0}"
SOURCE_MANIFOLD_GUARD_CALIBRATION="${SOURCE_MANIFOLD_GUARD_CALIBRATION:-disabled}"
HYPER_SOURCE_TRUST_ROUTING="${HYPER_SOURCE_TRUST_ROUTING:-0}"
USER_SET_HYPER_SOURCE_TRUST_STRENGTH="${HYPER_SOURCE_TRUST_STRENGTH+x}"
HYPER_SOURCE_TRUST_STRENGTH="${HYPER_SOURCE_TRUST_STRENGTH:-0.0}"
HYPER_SOURCE_TRUST_TOP_M="${HYPER_SOURCE_TRUST_TOP_M:-4}"
HYPER_SOURCE_TRUST_VARIABLE_GATE="${HYPER_SOURCE_TRUST_VARIABLE_GATE:-0}"
SOURCE_TRUST_BANK_CALIBRATION="${SOURCE_TRUST_BANK_CALIBRATION:-disabled}"
SOURCE_TRUST_QUERY_MODE="${SOURCE_TRUST_QUERY_MODE:-prompt_embedding}"
SOURCE_TRUST_QUERY_BLEND_LAMBDA="${SOURCE_TRUST_QUERY_BLEND_LAMBDA:-0.0}"
HYPER_PHYS_AGREEMENT_GUARD="${HYPER_PHYS_AGREEMENT_GUARD:-0}"
HYPER_PHYS_AGREEMENT_GUARD_STRENGTH="${HYPER_PHYS_AGREEMENT_GUARD_STRENGTH:-1.0}"
HYPER_PHYS_AGREEMENT_GUARD_MIN_MULTIPLIER="${HYPER_PHYS_AGREEMENT_GUARD_MIN_MULTIPLIER:-0.0}"
HYPER_PHYS_AGREEMENT_GUARD_RISK_RULE="${HYPER_PHYS_AGREEMENT_GUARD_RISK_RULE:-or}"
HYPER_PHYS_CONTEXT_MODULATION="${HYPER_PHYS_CONTEXT_MODULATION:-0}"
PHYS_CONTEXT_SOURCE="${PHYS_CONTEXT_SOURCE:-raw_input_side_da_diagnostics}"
HYPER_PHYS_FORMULA_OPERATOR="${HYPER_PHYS_FORMULA_OPERATOR:-0}"
PHYS_FORMULA_MODE="${PHYS_FORMULA_MODE:-enkf_rt_vertical_temp}"
PHYS_FORMULA_SOURCE="${PHYS_FORMULA_SOURCE:-raw_input_side_formula_v2}"
USER_SET_HYPER_PHYS_DELTA_SCALE="${HYPER_PHYS_DELTA_SCALE+x}"
HYPER_PHYS_DELTA_SCALE="${HYPER_PHYS_DELTA_SCALE:-0.25}"
USER_SET_HYPER_PHYS_GATE_INIT="${HYPER_PHYS_GATE_INIT+x}"
HYPER_PHYS_GATE_INIT="${HYPER_PHYS_GATE_INIT:-0.90}"
USER_SET_HYPER_OPERATOR_DROPPATH_P="${HYPER_OPERATOR_DROPPATH_P+x}"
HYPER_OPERATOR_DROPPATH_P="${HYPER_OPERATOR_DROPPATH_P:-0.10}"
USER_SET_HYPER_PHYS_CONSISTENCY_GUARD="${HYPER_PHYS_CONSISTENCY_GUARD+x}"
HYPER_PHYS_CONSISTENCY_GUARD="${HYPER_PHYS_CONSISTENCY_GUARD:-0}"
USER_SET_PHYS_CONSISTENCY_GUARD_MODE="${PHYS_CONSISTENCY_GUARD_MODE+x}"
PHYS_CONSISTENCY_GUARD_MODE="${PHYS_CONSISTENCY_GUARD_MODE:-enkf_rt_vertical}"
USER_SET_PHYS_CONSISTENCY_SOURCE="${PHYS_CONSISTENCY_SOURCE+x}"
PHYS_CONSISTENCY_SOURCE="${PHYS_CONSISTENCY_SOURCE:-raw_input_side_formula}"
USER_SET_PHYS_CONSISTENCY_MIN_SURFACE="${PHYS_CONSISTENCY_MIN_SURFACE+x}"
PHYS_CONSISTENCY_MIN_SURFACE="${PHYS_CONSISTENCY_MIN_SURFACE:-0.95}"
USER_SET_PHYS_CONSISTENCY_MIN_ROOTZONE="${PHYS_CONSISTENCY_MIN_ROOTZONE+x}"
PHYS_CONSISTENCY_MIN_ROOTZONE="${PHYS_CONSISTENCY_MIN_ROOTZONE:-0.90}"
USER_SET_PHYS_CONSISTENCY_STRENGTH_SURFACE="${PHYS_CONSISTENCY_STRENGTH_SURFACE+x}"
PHYS_CONSISTENCY_STRENGTH_SURFACE="${PHYS_CONSISTENCY_STRENGTH_SURFACE:-0.10}"
USER_SET_PHYS_CONSISTENCY_STRENGTH_ROOTZONE="${PHYS_CONSISTENCY_STRENGTH_ROOTZONE+x}"
PHYS_CONSISTENCY_STRENGTH_ROOTZONE="${PHYS_CONSISTENCY_STRENGTH_ROOTZONE:-0.15}"
USER_SET_HYPER_PHYS_GAIN_BASIS_RESIDUAL="${HYPER_PHYS_GAIN_BASIS_RESIDUAL+x}"
HYPER_PHYS_GAIN_BASIS_RESIDUAL="${HYPER_PHYS_GAIN_BASIS_RESIDUAL:-0}"
HYPER_PHYS_GAIN_BASIS_COEFF_SCALE="${HYPER_PHYS_GAIN_BASIS_COEFF_SCALE:-0.05}"
HYPER_PHYS_GAIN_BASIS_RESIDUAL_CLIP="${HYPER_PHYS_GAIN_BASIS_RESIDUAL_CLIP:-0.25}"
HYPER_PHYS_GAIN_BASIS_BETA_INIT="${HYPER_PHYS_GAIN_BASIS_BETA_INIT:-0.50}"
USER_SET_HYPER_PHYS_CONSISTENCY_REGULARIZATION_WEIGHT="${HYPER_PHYS_CONSISTENCY_REGULARIZATION_WEIGHT+x}"
HYPER_PHYS_CONSISTENCY_REGULARIZATION_WEIGHT="${HYPER_PHYS_CONSISTENCY_REGULARIZATION_WEIGHT:-0.0}"
M3_14_ALLOW_FROZEN_CONFIRMATION="${M3_14_ALLOW_FROZEN_CONFIRMATION:-${FROZEN_CONFIRMATION:-0}}"
M3_16_ALLOW_FROZEN_CONFIRMATION="${M3_16_ALLOW_FROZEN_CONFIRMATION:-${FROZEN_CONFIRMATION:-0}}"
SELECTION_METRIC="${SELECTION_METRIC:-source_val_transfer_safe_score}"
HYPER_RESIDUAL_MAGNITUDE_PENALTY="${HYPER_RESIDUAL_MAGNITUDE_PENALTY:-0.0}"
HYPER_COEFF_ENTROPY_FLOOR="${HYPER_COEFF_ENTROPY_FLOOR:-0.0}"
HYPER_COEFF_ENTROPY_PENALTY="${HYPER_COEFF_ENTROPY_PENALTY:-0.0}"
CONTEXT_ENCODER="${CONTEXT_ENCODER:-current_mean_std}"
M2_3_INIT_FROM_M2_1_CHECKPOINT="${M2_3_INIT_FROM_M2_1_CHECKPOINT-auto}"
USER_SET_RESUME_FROM_M3_1_BEST="${RESUME_FROM_M3_1_BEST+x}"
RESUME_FROM_M3_1_BEST="${RESUME_FROM_M3_1_BEST:-0}"
M3_6_INIT_FROM_M3_1_CHECKPOINT="${M3_6_INIT_FROM_M3_1_CHECKPOINT:-auto}"
M3_7_INIT_FROM_M3_1_CHECKPOINT="${M3_7_INIT_FROM_M3_1_CHECKPOINT:-auto}"
M3_8_INIT_FROM_M3_1_CHECKPOINT="${M3_8_INIT_FROM_M3_1_CHECKPOINT:-auto}"
M3_9_INIT_FROM_M3_1_CHECKPOINT="${M3_9_INIT_FROM_M3_1_CHECKPOINT:-auto}"
M3_13_INIT_FROM_M3_1_CHECKPOINT="${M3_13_INIT_FROM_M3_1_CHECKPOINT:-auto}"
M3_15_INIT_FROM_M3_1_CHECKPOINT="${M3_15_INIT_FROM_M3_1_CHECKPOINT:-auto}"
TARGET_EVAL_INPUT_STATS_USED_FOR_UPDATE="false"
SOURCE_EPISODE_PROMPT_POLICY="${SOURCE_EPISODE_PROMPT_POLICY:-context_monthly_prototype}"
SOURCE_ANCHOR_BLEND_CALIBRATION="${SOURCE_ANCHOR_BLEND_CALIBRATION:-1}"
HYPER_OUTPUT_HEAD_RESIDUAL="${HYPER_OUTPUT_HEAD_RESIDUAL:-1}"
ZERO_SHOT_PRIOR_FORM="${ZERO_SHOT_PRIOR_FORM:-direct_hyper}"
SOURCE_RESIDUAL_RHO="${SOURCE_RESIDUAL_RHO:-0.0}"
SOURCE_RESIDUAL_GATE="${SOURCE_RESIDUAL_GATE:-prompt_reliability_scalar}"
SOURCE_RESIDUAL_GATE_INIT="${SOURCE_RESIDUAL_GATE_INIT:-0.95}"
SOURCE_RESIDUAL_RELIABILITY_DIM="${SOURCE_RESIDUAL_RELIABILITY_DIM:-5}"
DATASET_BACKEND="${DATASET_BACKEND:-auto}"
TENSOR_CACHE_DIR="${TENSOR_CACHE_DIR:-artifacts/region_crops/US}"
MAX_YEAR_CACHE_ENTRIES="${MAX_YEAR_CACHE_ENTRIES:-2}"
TENSOR_CACHE_LOAD_MODE="${TENSOR_CACHE_LOAD_MODE:-mmap}"
TRAIN_BATCH_SAMPLER="${TRAIN_BATCH_SAMPLER:-source_region_year_grouped}"
USER_SET_SOURCE_FIT_MAX_BATCHES_PER_EPOCH="${SOURCE_FIT_MAX_BATCHES_PER_EPOCH+x}"
SOURCE_FIT_MAX_BATCHES_PER_EPOCH="${SOURCE_FIT_MAX_BATCHES_PER_EPOCH:-0}"
NUM_WORKERS="${NUM_WORKERS:-2}"
USER_SET_MAX_EPOCHS="${MAX_EPOCHS+x}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
USER_SET_USE_AMP="${USE_AMP+x}"
USER_SET_BATCH_SIZE="${BATCH_SIZE+x}"
USER_SET_ACCUM_STEPS="${ACCUM_STEPS+x}"
USER_SET_LR="${LR+x}"
USE_AMP="${USE_AMP:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ACCUM_STEPS="${ACCUM_STEPS:-2}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
USER_SET_EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS+x}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-5}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-200}"
SOURCE_PROTOTYPE_CACHE_DIR="${SOURCE_PROTOTYPE_CACHE_DIR:-artifacts/cache/source_context_monthly_prototypes}"
SOURCE_PROTOTYPE_CACHE_MODE="${SOURCE_PROTOTYPE_CACHE_MODE:-read_write}"
SOURCE_TRUST_BANK_CACHE_DIR="${SOURCE_TRUST_BANK_CACHE_DIR:-artifacts/cache/source_trust_banks}"
SOURCE_TRUST_BANK_CACHE_MODE="${SOURCE_TRUST_BANK_CACHE_MODE:-read_write}"
USER_SET_TRAINABLE_SCOPE="${TRAINABLE_SCOPE+x}"
TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-source_base_frozen_adapter_film}"

case "${ABLATION_ID}" in
    M0_current)
        HYPER_COEFF_GENERATOR=per_adapter
        HYPER_RELIABILITY_GATE=none
        HYPER_ENABLE_FILM=1
        HYPER_ENABLE_ADAPTERS=1
        ;;
    M1_shared_coeff)
        HYPER_COEFF_GENERATOR=shared_layer_aware
        HYPER_RELIABILITY_GATE=none
        HYPER_ENABLE_FILM=1
        HYPER_ENABLE_ADAPTERS=1
        ;;
    M2_shared_coeff_gate)
        HYPER_COEFF_GENERATOR=shared_layer_aware
        HYPER_RELIABILITY_GATE=prompt_scalar
        HYPER_ENABLE_FILM=1
        HYPER_ENABLE_ADAPTERS=1
        ;;
    M2_rank_gated_dora)
        HYPER_COEFF_GENERATOR=shared_layer_aware_rank_gated
        HYPER_RELIABILITY_GATE=prompt_scalar
        HYPER_ADAPTER_PARAM_STYLE=dora_like_gain
        HYPER_ENABLE_FILM=1
        HYPER_ENABLE_ADAPTERS=1
        ;;
    M2_1_rank_gated_dora_stable)
        HYPER_COEFF_GENERATOR=shared_layer_aware_rank_gated_stable
        HYPER_RELIABILITY_GATE=prompt_scalar
        HYPER_ADAPTER_PARAM_STYLE=dora_like_gain_bounded
        ZERO_SHOT_PRIOR_FORM=direct_hyper
        SOURCE_RESIDUAL_RHO=0.0
        HYPER_RESIDUAL_MAGNITUDE_PENALTY=0.0
        HYPER_COEFF_ENTROPY_FLOOR=0.0
        HYPER_COEFF_ENTROPY_PENALTY=0.0
        if [[ -z "${USER_SET_HYPER_RANK_GATE_TEMPERATURE_INIT}" ]]; then
            HYPER_RANK_GATE_TEMPERATURE_INIT=2.0
        fi
        if [[ -z "${USER_SET_USE_AMP}" ]]; then
            USE_AMP=0
        fi
        if [[ -z "${USER_SET_BATCH_SIZE}" ]]; then
            BATCH_SIZE=64
        fi
        if [[ -z "${USER_SET_ACCUM_STEPS}" ]]; then
            ACCUM_STEPS=2
        fi
        if [[ -z "${USER_SET_LR}" ]]; then
            LR=2e-4
        fi
        HYPER_ENABLE_FILM=1
        HYPER_ENABLE_ADAPTERS=1
        ;;
    M2_2_source_saliency_prior)
        HYPER_COEFF_GENERATOR=shared_layer_aware_rank_gated_stable
        HYPER_RELIABILITY_GATE=prompt_scalar
        HYPER_ADAPTER_PARAM_STYLE=dora_like_gain_bounded
        if [[ -z "${USER_SET_HYPER_RANK_GATE_TEMPERATURE_INIT}" ]]; then
            HYPER_RANK_GATE_TEMPERATURE_INIT=2.0
        fi
        if [[ -z "${USER_SET_USE_AMP}" ]]; then
            USE_AMP=0
        fi
        if [[ -z "${USER_SET_BATCH_SIZE}" ]]; then
            BATCH_SIZE=64
        fi
        if [[ -z "${USER_SET_ACCUM_STEPS}" ]]; then
            ACCUM_STEPS=2
        fi
        if [[ -z "${USER_SET_LR}" ]]; then
            LR=2e-4
        fi
        if [[ "${HYPER_SOURCE_SALIENCY_PRIOR_BETA}" == "0.0" || "${HYPER_SOURCE_SALIENCY_PRIOR_BETA}" == "0" ]]; then
            HYPER_SOURCE_SALIENCY_PRIOR_BETA=0.5
        fi
        HYPER_SOURCE_SALIENCY_PRIOR_APPLICATION=legacy_gate_logit_bias_before_topk
        if [[ "${HYPER_PROMPT_MANIFOLD_RELIABILITY}" == "0" ]]; then
            HYPER_PROMPT_MANIFOLD_RELIABILITY=1
        fi
        if [[ "${HYPER_PROMPT_MANIFOLD_RELIABILITY_STRENGTH}" == "0.0" || "${HYPER_PROMPT_MANIFOLD_RELIABILITY_STRENGTH}" == "0" ]]; then
            HYPER_PROMPT_MANIFOLD_RELIABILITY_STRENGTH=0.5
        fi
        HYPER_ENABLE_FILM=1
        HYPER_ENABLE_ADAPTERS=1
        ;;
    M2_3_source_safe_residual_hyperda)
        HYPER_COEFF_GENERATOR=shared_layer_aware_rank_gated_stable
        HYPER_RELIABILITY_GATE=prompt_scalar
        HYPER_ADAPTER_PARAM_STYLE=dora_like_gain_bounded
        if [[ -z "${USER_SET_HYPER_RANK_GATE_TEMPERATURE_INIT}" ]]; then
            HYPER_RANK_GATE_TEMPERATURE_INIT=2.0
        fi
        if [[ -z "${USER_SET_USE_AMP}" ]]; then
            USE_AMP=0
        fi
        if [[ -z "${USER_SET_BATCH_SIZE}" ]]; then
            BATCH_SIZE=64
        fi
        if [[ -z "${USER_SET_ACCUM_STEPS}" ]]; then
            ACCUM_STEPS=2
        fi
        if [[ -z "${USER_SET_LR}" ]]; then
            LR=2e-4
        fi
        ZERO_SHOT_PRIOR_FORM=source_base_residual_reliability_gated
        SOURCE_RESIDUAL_GATE=prompt_reliability_scalar
        HYPER_SOURCE_SALIENCY_PRIOR_APPLICATION=soft_regularization_metadata
        if [[ "${HYPER_RESIDUAL_MAGNITUDE_PENALTY}" == "0.0" || "${HYPER_RESIDUAL_MAGNITUDE_PENALTY}" == "0" ]]; then
            HYPER_RESIDUAL_MAGNITUDE_PENALTY=0.001
        fi
        if [[ "${HYPER_COEFF_ENTROPY_FLOOR}" == "0.0" || "${HYPER_COEFF_ENTROPY_FLOOR}" == "0" ]]; then
            HYPER_COEFF_ENTROPY_FLOOR=0.5
        fi
        if [[ "${HYPER_COEFF_ENTROPY_PENALTY}" == "0.0" || "${HYPER_COEFF_ENTROPY_PENALTY}" == "0" ]]; then
            HYPER_COEFF_ENTROPY_PENALTY=0.0001
        fi
        HYPER_ENABLE_FILM=1
        HYPER_ENABLE_ADAPTERS=1
        ;;
    M2_5a_da_aware_prompt_only)
        HYPER_COEFF_GENERATOR=shared_layer_aware_rank_gated_stable
        HYPER_RELIABILITY_GATE=prompt_scalar
        HYPER_ADAPTER_PARAM_STYLE=dora_like_gain_bounded
        CONTEXT_ENCODER=robust_input_side_da_diagnostics
        ZERO_SHOT_PRIOR_FORM=direct_hyper
        SOURCE_RESIDUAL_RHO=0.0
        HYPER_RESIDUAL_MAGNITUDE_PENALTY=0.0
        HYPER_COEFF_ENTROPY_FLOOR=0.0
        HYPER_COEFF_ENTROPY_PENALTY=0.0
        if [[ -z "${USER_SET_HYPER_RANK_GATE_TEMPERATURE_INIT}" ]]; then
            HYPER_RANK_GATE_TEMPERATURE_INIT=2.0
        fi
        if [[ -z "${USER_SET_USE_AMP}" ]]; then
            USE_AMP=0
        fi
        if [[ -z "${USER_SET_BATCH_SIZE}" ]]; then
            BATCH_SIZE=64
        fi
        if [[ -z "${USER_SET_ACCUM_STEPS}" ]]; then
            ACCUM_STEPS=2
        fi
        if [[ -z "${USER_SET_LR}" ]]; then
            LR=2e-4
        fi
        HYPER_ENABLE_FILM=1
        HYPER_ENABLE_ADAPTERS=1
        ;;
    M2_5b_da_aware_conservative_router)
        HYPER_COEFF_GENERATOR=shared_layer_aware_rank_gated_stable
        HYPER_RELIABILITY_GATE=prompt_scalar
        HYPER_ADAPTER_PARAM_STYLE=dora_like_gain_bounded
        CONTEXT_ENCODER=robust_input_side_da_diagnostics_raw
        ZERO_SHOT_PRIOR_FORM=source_base_residual_reliability_gated
        SOURCE_RESIDUAL_RHO=1.0
        SOURCE_RESIDUAL_GATE=prompt_reliability_scalar
        SOURCE_RESIDUAL_GATE_INIT=0.90
        HYPER_SOURCE_SALIENCY_PRIOR_APPLICATION=soft_regularization_metadata
        HYPER_RESIDUAL_MAGNITUDE_PENALTY=0.0
        HYPER_COEFF_ENTROPY_FLOOR=0.0
        HYPER_COEFF_ENTROPY_PENALTY=0.0
        HYPER_PROMPT_MANIFOLD_RELIABILITY=1
        HYPER_PROMPT_MANIFOLD_RELIABILITY_STRENGTH=0.25
        if [[ -z "${USER_SET_HYPER_RANK_GATE_TEMPERATURE_INIT}" ]]; then
            HYPER_RANK_GATE_TEMPERATURE_INIT=2.0
        fi
        if [[ -z "${USER_SET_USE_AMP}" ]]; then
            USE_AMP=0
        fi
        if [[ -z "${USER_SET_BATCH_SIZE}" ]]; then
            BATCH_SIZE=64
        fi
        if [[ -z "${USER_SET_ACCUM_STEPS}" ]]; then
            ACCUM_STEPS=2
        fi
        if [[ -z "${USER_SET_LR}" ]]; then
            LR=2e-4
        fi
        HYPER_ENABLE_FILM=1
        HYPER_ENABLE_ADAPTERS=1
        ;;
    M2_6_source_manifold_guarded_prior)
        HYPER_COEFF_GENERATOR=shared_layer_aware_rank_gated_stable
        HYPER_RELIABILITY_GATE=prompt_scalar
        HYPER_ADAPTER_PARAM_STYLE=dora_like_gain_bounded
        CONTEXT_ENCODER=current_mean_std
        ZERO_SHOT_PRIOR_FORM=source_base_residual_reliability_gated
        SOURCE_RESIDUAL_RHO=1.0
        SOURCE_RESIDUAL_GATE=prompt_reliability_scalar
        SOURCE_RESIDUAL_GATE_INIT=0.95
        HYPER_SOURCE_SALIENCY_PRIOR_APPLICATION=soft_regularization_metadata
        HYPER_RESIDUAL_MAGNITUDE_PENALTY=0.0
        HYPER_COEFF_ENTROPY_FLOOR=0.0
        HYPER_COEFF_ENTROPY_PENALTY=0.0
        HYPER_SOURCE_MANIFOLD_GUARD=1
        SOURCE_MANIFOLD_GUARD_CALIBRATION=source_fit_source_val_only
        HYPER_SOURCE_MANIFOLD_GUARD_DISTANCE_KEY=source_manifold_distance_bounded
        if [[ -z "${USER_SET_HYPER_RANK_GATE_TEMPERATURE_INIT}" ]]; then
            HYPER_RANK_GATE_TEMPERATURE_INIT=2.0
        fi
        if [[ -z "${USER_SET_USE_AMP}" ]]; then
            USE_AMP=0
        fi
        if [[ -z "${USER_SET_BATCH_SIZE}" ]]; then
            BATCH_SIZE=64
        fi
        if [[ -z "${USER_SET_ACCUM_STEPS}" ]]; then
            ACCUM_STEPS=2
        fi
        if [[ -z "${USER_SET_LR}" ]]; then
            LR=2e-4
        fi
        HYPER_ENABLE_FILM=1
        HYPER_ENABLE_ADAPTERS=1
        ;;
    M3_0_hyperda_trust_light|M3_1_hyperda_trust_medium|M3_2_hyperda_trust_raw_reliability|M3_2_phys_trust_raw_da_query|M3_2a_phys_trust_raw_query_fixed|M3_4_phys_trust_blended_query|M3_5_phys_agreement_guarded_trust|M3_5b_phys_agreement_floor_guard|M3_6_phys_token_operator_droppath_trust|M3_7_phys_formula_consistency_guarded_trust|M3_8_phys_formula_operator_trust|M3_8b_phys_formula_light_guarded_trust|M3_8c_phys_formula_light_operator_only_trust|M3_9_phys_formula_enhanced_trust|M3_12_phys_gain_basis_hypertrust|M3_13_phys_gain_guarded_hypertrust|M3_14_source_trained_phys_formula_gain_hypertrust|M3_15_m31_anchored_source_safe_phys_coeff_delta|M3_16_source_only_phys_m3trust_lite|M3_3_hyperda_trust_selection_only|M3_1a_trust_medium_dualalpha|M3_1b_trust_mid_high|M3_1c_trust_medium_local|M3_1d_trust_medium_broad)
        HYPER_COEFF_GENERATOR=shared_layer_aware_rank_gated_stable
        HYPER_RELIABILITY_GATE=prompt_scalar
        HYPER_ADAPTER_PARAM_STYLE=dora_like_gain_bounded
        CONTEXT_ENCODER=current_mean_std
        ZERO_SHOT_PRIOR_FORM=source_base_residual_reliability_gated
        SOURCE_RESIDUAL_RHO=1.0
        SOURCE_RESIDUAL_GATE=prompt_reliability_scalar
        SOURCE_RESIDUAL_GATE_INIT=0.95
        HYPER_SOURCE_SALIENCY_PRIOR_APPLICATION=soft_regularization_metadata
        HYPER_RESIDUAL_MAGNITUDE_PENALTY=0.0
        HYPER_COEFF_ENTROPY_FLOOR=0.0
        HYPER_COEFF_ENTROPY_PENALTY=0.0
        HYPER_SOURCE_TRUST_TOP_M=4
        SOURCE_TRUST_BANK_CALIBRATION=source_fit_source_val_only
        HYPER_SOURCE_TRUST_VARIABLE_GATE=1
        SELECTION_METRIC=source_val_dual_variable_cvar_safe_score
        if [[ "${ABLATION_ID}" == "M3_0_hyperda_trust_light" ]]; then
            HYPER_SOURCE_TRUST_ROUTING=1
            HYPER_SOURCE_TRUST_STRENGTH=0.25
        elif [[ "${ABLATION_ID}" == "M3_1_hyperda_trust_medium" || "${ABLATION_ID}" == "M3_1a_trust_medium_dualalpha" ]]; then
            HYPER_SOURCE_TRUST_ROUTING=1
            HYPER_SOURCE_TRUST_STRENGTH=0.50
        elif [[ "${ABLATION_ID}" == "M3_1b_trust_mid_high" ]]; then
            HYPER_SOURCE_TRUST_ROUTING=1
            HYPER_SOURCE_TRUST_STRENGTH=0.375
        elif [[ "${ABLATION_ID}" == "M3_1c_trust_medium_local" ]]; then
            HYPER_SOURCE_TRUST_ROUTING=1
            HYPER_SOURCE_TRUST_STRENGTH=0.50
            HYPER_SOURCE_TRUST_TOP_M=2
        elif [[ "${ABLATION_ID}" == "M3_1d_trust_medium_broad" ]]; then
            HYPER_SOURCE_TRUST_ROUTING=1
            HYPER_SOURCE_TRUST_STRENGTH=0.50
            HYPER_SOURCE_TRUST_TOP_M=6
        elif [[ "${ABLATION_ID}" == "M3_2_hyperda_trust_raw_reliability" || "${ABLATION_ID}" == "M3_2_phys_trust_raw_da_query" || "${ABLATION_ID}" == "M3_2a_phys_trust_raw_query_fixed" ]]; then
            HYPER_SOURCE_TRUST_ROUTING=1
            SOURCE_TRUST_QUERY_MODE=raw_input_side_da_diagnostics
            if [[ -z "${USER_SET_HYPER_SOURCE_TRUST_STRENGTH}" ]]; then
                HYPER_SOURCE_TRUST_STRENGTH=0.25
            fi
        elif [[ "${ABLATION_ID}" == "M3_4_phys_trust_blended_query" ]]; then
            HYPER_SOURCE_TRUST_ROUTING=1
            SOURCE_TRUST_QUERY_MODE=blended_prompt_raw_da_0p25
            SOURCE_TRUST_QUERY_BLEND_LAMBDA=0.25
            if [[ -z "${USER_SET_HYPER_SOURCE_TRUST_STRENGTH}" ]]; then
                HYPER_SOURCE_TRUST_STRENGTH=0.25
            fi
        elif [[ "${ABLATION_ID}" == "M3_5_phys_agreement_guarded_trust" ]]; then
            HYPER_SOURCE_TRUST_ROUTING=1
            HYPER_SOURCE_TRUST_STRENGTH=0.50
            HYPER_SOURCE_TRUST_TOP_M=4
            SOURCE_TRUST_QUERY_MODE=raw_input_side_da_diagnostics
            HYPER_PHYS_AGREEMENT_GUARD=1
            HYPER_PHYS_AGREEMENT_GUARD_STRENGTH=1.0
        elif [[ "${ABLATION_ID}" == "M3_5b_phys_agreement_floor_guard" ]]; then
            HYPER_SOURCE_TRUST_ROUTING=1
            HYPER_SOURCE_TRUST_STRENGTH=0.50
            HYPER_SOURCE_TRUST_TOP_M=4
            SOURCE_TRUST_QUERY_MODE=raw_input_side_da_diagnostics
            HYPER_PHYS_AGREEMENT_GUARD=1
            HYPER_PHYS_AGREEMENT_GUARD_STRENGTH=1.0
            HYPER_PHYS_AGREEMENT_GUARD_MIN_MULTIPLIER=0.8
            HYPER_PHYS_AGREEMENT_GUARD_RISK_RULE=and
        elif [[ "${ABLATION_ID}" == "M3_6_phys_token_operator_droppath_trust" ]]; then
            HYPER_SOURCE_TRUST_ROUTING=1
            HYPER_SOURCE_TRUST_STRENGTH=0.50
            HYPER_SOURCE_TRUST_TOP_M=4
            SOURCE_TRUST_QUERY_MODE=prompt_embedding
            HYPER_PHYS_AGREEMENT_GUARD=0
            HYPER_PHYS_CONTEXT_MODULATION=1
            PHYS_CONTEXT_SOURCE=raw_input_side_da_diagnostics
        elif [[ "${ABLATION_ID}" == "M3_7_phys_formula_consistency_guarded_trust" ]]; then
            HYPER_SOURCE_TRUST_ROUTING=1
            HYPER_SOURCE_TRUST_STRENGTH=0.50
            HYPER_SOURCE_TRUST_TOP_M=4
            HYPER_SOURCE_TRUST_VARIABLE_GATE=1
            SOURCE_TRUST_QUERY_MODE=prompt_embedding
            HYPER_PHYS_AGREEMENT_GUARD=0
            HYPER_PHYS_CONTEXT_MODULATION=0
            HYPER_PHYS_CONSISTENCY_GUARD=1
            PHYS_CONSISTENCY_GUARD_MODE=enkf_rt_vertical
            PHYS_CONSISTENCY_SOURCE=raw_input_side_formula
            PHYS_CONSISTENCY_MIN_SURFACE=0.95
            PHYS_CONSISTENCY_MIN_ROOTZONE=0.90
            PHYS_CONSISTENCY_STRENGTH_SURFACE=0.10
            PHYS_CONSISTENCY_STRENGTH_ROOTZONE=0.15
        elif [[ "${ABLATION_ID}" == "M3_8_phys_formula_operator_trust" ]]; then
            HYPER_SOURCE_TRUST_ROUTING=1
            HYPER_SOURCE_TRUST_STRENGTH=0.50
            HYPER_SOURCE_TRUST_TOP_M=4
            HYPER_SOURCE_TRUST_VARIABLE_GATE=1
            SOURCE_TRUST_QUERY_MODE=prompt_embedding
            HYPER_PHYS_AGREEMENT_GUARD=0
            HYPER_PHYS_CONTEXT_MODULATION=1
            HYPER_PHYS_FORMULA_OPERATOR=1
            PHYS_CONTEXT_SOURCE=raw_input_side_formula_v2
            PHYS_FORMULA_MODE=enkf_rt_vertical_temp
            PHYS_FORMULA_SOURCE=raw_input_side_formula_v2
            if [[ -z "${USER_SET_HYPER_PHYS_DELTA_SCALE}" ]]; then
                HYPER_PHYS_DELTA_SCALE=0.10
            fi
            if [[ -z "${USER_SET_HYPER_PHYS_GATE_INIT}" ]]; then
                HYPER_PHYS_GATE_INIT=0.50
            fi
            if [[ -z "${USER_SET_HYPER_OPERATOR_DROPPATH_P}" ]]; then
                HYPER_OPERATOR_DROPPATH_P=0.10
            fi
            if [[ -z "${USER_SET_HYPER_PHYS_CONSISTENCY_GUARD}" ]]; then
                HYPER_PHYS_CONSISTENCY_GUARD=1
            fi
            if [[ -z "${USER_SET_PHYS_CONSISTENCY_GUARD_MODE}" ]]; then
                PHYS_CONSISTENCY_GUARD_MODE=surface_primary_enkf_rt_vertical_product
            fi
            if [[ -z "${USER_SET_PHYS_CONSISTENCY_SOURCE}" ]]; then
                PHYS_CONSISTENCY_SOURCE=raw_input_side_formula_v2
            fi
            if [[ -z "${USER_SET_PHYS_CONSISTENCY_MIN_SURFACE}" ]]; then
                PHYS_CONSISTENCY_MIN_SURFACE=0.97
            fi
            if [[ -z "${USER_SET_PHYS_CONSISTENCY_MIN_ROOTZONE}" ]]; then
                PHYS_CONSISTENCY_MIN_ROOTZONE=0.98
            fi
            if [[ -z "${USER_SET_PHYS_CONSISTENCY_STRENGTH_SURFACE}" ]]; then
                PHYS_CONSISTENCY_STRENGTH_SURFACE=0.05
            fi
            if [[ -z "${USER_SET_PHYS_CONSISTENCY_STRENGTH_ROOTZONE}" ]]; then
                PHYS_CONSISTENCY_STRENGTH_ROOTZONE=0.02
            fi
            if [[ -z "${USER_SET_TRAINABLE_SCOPE}" ]]; then
                TRAINABLE_SCOPE=phys_formula_context_only
            fi
            if [[ -z "${USER_SET_RESUME_FROM_M3_1_BEST}" ]]; then
                RESUME_FROM_M3_1_BEST=1
            fi
            if [[ -z "${USER_SET_LR}" ]]; then
                LR=1e-4
                USER_SET_LR=branch_default
            fi
            if [[ -z "${USER_SET_MAX_EPOCHS}" ]]; then
                MAX_EPOCHS=3
            fi
            if [[ -z "${USER_SET_EVAL_EVERY_EPOCHS}" ]]; then
                EVAL_EVERY_EPOCHS=1
            fi
        elif [[ "${ABLATION_ID}" == "M3_8b_phys_formula_light_guarded_trust" || "${ABLATION_ID}" == "M3_8c_phys_formula_light_operator_only_trust" ]]; then
            if [[ "${RESUME_FROM_M3_1_BEST}" == "1" || "${RESUME_FROM_M3_1_BEST,,}" == "true" ]]; then
                echo "ERROR: clean source-stage physics preset cannot warm-start from M3_1: ${ABLATION_ID}" >&2
                echo "Use only --init_from_source_base_checkpoint via SOURCE_CHECKPOINT; do not set RESUME_FROM_M3_1_BEST=1." >&2
                exit 2
            fi
            HYPER_SOURCE_TRUST_ROUTING=1
            HYPER_SOURCE_TRUST_STRENGTH=0.50
            HYPER_SOURCE_TRUST_TOP_M=4
            HYPER_SOURCE_TRUST_VARIABLE_GATE=1
            SOURCE_TRUST_QUERY_MODE=prompt_embedding
            HYPER_PHYS_AGREEMENT_GUARD=0
            HYPER_PHYS_CONTEXT_MODULATION=1
            HYPER_PHYS_FORMULA_OPERATOR=1
            PHYS_CONTEXT_SOURCE=raw_input_side_formula_v2
            PHYS_FORMULA_MODE=enkf_rt_vertical_temp
            PHYS_FORMULA_SOURCE=raw_input_side_formula_v2
            if [[ -z "${USER_SET_HYPER_PHYS_DELTA_SCALE}" ]]; then
                HYPER_PHYS_DELTA_SCALE=0.05
            fi
            if [[ -z "${USER_SET_HYPER_PHYS_GATE_INIT}" ]]; then
                HYPER_PHYS_GATE_INIT=0.35
            fi
            if [[ -z "${USER_SET_HYPER_OPERATOR_DROPPATH_P}" ]]; then
                HYPER_OPERATOR_DROPPATH_P=0.10
            fi
            if [[ "${ABLATION_ID}" == "M3_8b_phys_formula_light_guarded_trust" ]]; then
                if [[ -z "${USER_SET_HYPER_PHYS_CONSISTENCY_GUARD}" ]]; then
                    HYPER_PHYS_CONSISTENCY_GUARD=1
                fi
            else
                if [[ -z "${USER_SET_HYPER_PHYS_CONSISTENCY_GUARD}" ]]; then
                    HYPER_PHYS_CONSISTENCY_GUARD=0
                fi
            fi
            if [[ -z "${USER_SET_PHYS_CONSISTENCY_GUARD_MODE}" ]]; then
                PHYS_CONSISTENCY_GUARD_MODE=surface_primary_enkf_rt_vertical_product
            fi
            if [[ -z "${USER_SET_PHYS_CONSISTENCY_SOURCE}" ]]; then
                PHYS_CONSISTENCY_SOURCE=raw_input_side_formula_v2
            fi
            if [[ -z "${USER_SET_PHYS_CONSISTENCY_MIN_SURFACE}" ]]; then
                PHYS_CONSISTENCY_MIN_SURFACE=0.985
            fi
            if [[ -z "${USER_SET_PHYS_CONSISTENCY_MIN_ROOTZONE}" ]]; then
                PHYS_CONSISTENCY_MIN_ROOTZONE=0.99
            fi
            if [[ -z "${USER_SET_PHYS_CONSISTENCY_STRENGTH_SURFACE}" ]]; then
                PHYS_CONSISTENCY_STRENGTH_SURFACE=0.02
            fi
            if [[ -z "${USER_SET_PHYS_CONSISTENCY_STRENGTH_ROOTZONE}" ]]; then
                PHYS_CONSISTENCY_STRENGTH_ROOTZONE=0.01
            fi
            if [[ -n "${USER_SET_TRAINABLE_SCOPE}" && "${TRAINABLE_SCOPE}" != "source_base_frozen_adapter_film" ]]; then
                echo "ERROR: ${ABLATION_ID} is a clean source-stage preset and requires TRAINABLE_SCOPE=source_base_frozen_adapter_film" >&2
                exit 2
            fi
            TRAINABLE_SCOPE=source_base_frozen_adapter_film
        elif [[ "${ABLATION_ID}" == "M3_9_phys_formula_enhanced_trust" ]]; then
            HYPER_SOURCE_TRUST_ROUTING=1
            HYPER_SOURCE_TRUST_STRENGTH=0.50
            HYPER_SOURCE_TRUST_TOP_M=4
            HYPER_SOURCE_TRUST_VARIABLE_GATE=1
            SOURCE_TRUST_QUERY_MODE=prompt_embedding
            HYPER_PHYS_AGREEMENT_GUARD=0
            HYPER_PHYS_CONTEXT_MODULATION=1
            HYPER_PHYS_FORMULA_OPERATOR=1
            PHYS_CONTEXT_SOURCE=raw_input_side_formula_v3_enhanced
            PHYS_FORMULA_MODE=enkf_rt_vertical_temp
            PHYS_FORMULA_SOURCE=raw_input_side_formula_v3_enhanced
            if [[ -z "${USER_SET_HYPER_PHYS_DELTA_SCALE}" ]]; then
                HYPER_PHYS_DELTA_SCALE=0.05
            fi
            if [[ -z "${USER_SET_HYPER_PHYS_GATE_INIT}" ]]; then
                HYPER_PHYS_GATE_INIT=0.35
            fi
            if [[ -z "${USER_SET_HYPER_OPERATOR_DROPPATH_P}" ]]; then
                HYPER_OPERATOR_DROPPATH_P=0.10
            fi
            if [[ -z "${USER_SET_HYPER_PHYS_CONSISTENCY_GUARD}" ]]; then
                HYPER_PHYS_CONSISTENCY_GUARD=1
            fi
            if [[ -z "${USER_SET_PHYS_CONSISTENCY_GUARD_MODE}" ]]; then
                PHYS_CONSISTENCY_GUARD_MODE=surface_primary_enkf_rt_vertical_product
            fi
            if [[ -z "${USER_SET_PHYS_CONSISTENCY_SOURCE}" ]]; then
                PHYS_CONSISTENCY_SOURCE=raw_input_side_formula_v3_enhanced
            fi
            if [[ -z "${USER_SET_PHYS_CONSISTENCY_MIN_SURFACE}" ]]; then
                PHYS_CONSISTENCY_MIN_SURFACE=0.98
            fi
            if [[ -z "${USER_SET_PHYS_CONSISTENCY_MIN_ROOTZONE}" ]]; then
                PHYS_CONSISTENCY_MIN_ROOTZONE=0.99
            fi
            if [[ -z "${USER_SET_PHYS_CONSISTENCY_STRENGTH_SURFACE}" ]]; then
                PHYS_CONSISTENCY_STRENGTH_SURFACE=0.02
            fi
            if [[ -z "${USER_SET_PHYS_CONSISTENCY_STRENGTH_ROOTZONE}" ]]; then
                PHYS_CONSISTENCY_STRENGTH_ROOTZONE=0.01
            fi
            if [[ -z "${USER_SET_TRAINABLE_SCOPE}" ]]; then
                TRAINABLE_SCOPE=phys_formula_context_only
            fi
            if [[ -z "${USER_SET_RESUME_FROM_M3_1_BEST}" ]]; then
                RESUME_FROM_M3_1_BEST=1
            fi
            if [[ -z "${USER_SET_LR}" ]]; then
                LR=1e-4
                USER_SET_LR=branch_default
            fi
            if [[ -z "${USER_SET_MAX_EPOCHS}" ]]; then
                MAX_EPOCHS=3
            fi
            if [[ -z "${USER_SET_EVAL_EVERY_EPOCHS}" ]]; then
                EVAL_EVERY_EPOCHS=1
            fi
            if [[ "${SOURCE_FIT_MAX_BATCHES_PER_EPOCH}" == "0" ]]; then
                if [[ -z "${USER_SET_SOURCE_FIT_MAX_BATCHES_PER_EPOCH}" ]]; then
                    SOURCE_FIT_MAX_BATCHES_PER_EPOCH=384
                fi
            fi
        elif [[ "${ABLATION_ID}" == "M3_12_phys_gain_basis_hypertrust" ]]; then
            if [[ "${RESUME_FROM_M3_1_BEST}" == "1" || "${RESUME_FROM_M3_1_BEST,,}" == "true" ]]; then
                echo "ERROR: M3_12_phys_gain_basis_hypertrust cannot warm-start from M3_1." >&2
                echo "Use only --init_from_source_base_checkpoint via SOURCE_CHECKPOINT; do not set RESUME_FROM_M3_1_BEST=1." >&2
                exit 2
            fi
            HYPER_SOURCE_TRUST_ROUTING=1
            HYPER_SOURCE_TRUST_STRENGTH=0.50
            HYPER_SOURCE_TRUST_TOP_M=4
            HYPER_SOURCE_TRUST_VARIABLE_GATE=1
            SOURCE_TRUST_QUERY_MODE=prompt_embedding
            HYPER_PHYS_AGREEMENT_GUARD=0
            HYPER_PHYS_CONTEXT_MODULATION=0
            HYPER_PHYS_FORMULA_OPERATOR=0
            HYPER_PHYS_CONSISTENCY_GUARD=0
            HYPER_PHYS_GAIN_BASIS_RESIDUAL=1
            if [[ -n "${USER_SET_TRAINABLE_SCOPE}" && "${TRAINABLE_SCOPE}" != "source_base_frozen_adapter_film" ]]; then
                echo "ERROR: M3_12_phys_gain_basis_hypertrust requires TRAINABLE_SCOPE=source_base_frozen_adapter_film" >&2
                exit 2
            fi
            TRAINABLE_SCOPE=source_base_frozen_adapter_film
            if [[ -z "${USER_SET_USE_AMP}" ]]; then
                USE_AMP=0
            fi
            if [[ -z "${USER_SET_BATCH_SIZE}" ]]; then
                BATCH_SIZE=64
            fi
            if [[ -z "${USER_SET_ACCUM_STEPS}" ]]; then
                ACCUM_STEPS=2
            fi
            if [[ -z "${USER_SET_LR}" ]]; then
                LR=2e-4
            fi
            if [[ -z "${USER_SET_MAX_EPOCHS}" ]]; then
                MAX_EPOCHS=50
            fi
        elif [[ "${ABLATION_ID}" == "M3_13_phys_gain_guarded_hypertrust" ]]; then
            if [[ -n "${USER_SET_RESUME_FROM_M3_1_BEST}" && "${RESUME_FROM_M3_1_BEST}" != "1" && "${RESUME_FROM_M3_1_BEST,,}" != "true" ]]; then
                echo "ERROR: M3_13_phys_gain_guarded_hypertrust requires RESUME_FROM_M3_1_BEST=1." >&2
                exit 2
            fi
            if [[ -n "${USER_SET_TRAINABLE_SCOPE}" && "${TRAINABLE_SCOPE}" != "phys_gain_guard_only" ]]; then
                echo "ERROR: M3_13_phys_gain_guarded_hypertrust requires TRAINABLE_SCOPE=phys_gain_guard_only." >&2
                exit 2
            fi
            if [[ -n "${USER_SET_MAX_EPOCHS}" && "${MAX_EPOCHS}" != "0" ]]; then
                echo "ERROR: M3_13_phys_gain_guarded_hypertrust requires MAX_EPOCHS=0." >&2
                exit 2
            fi
            RESUME_FROM_M3_1_BEST=1
            TRAINABLE_SCOPE=phys_gain_guard_only
            MAX_EPOCHS=0
            EVAL_EVERY_EPOCHS=1
            USE_AMP=0
            HYPER_SOURCE_TRUST_ROUTING=1
            HYPER_SOURCE_TRUST_STRENGTH=0.50
            HYPER_SOURCE_TRUST_TOP_M=4
            HYPER_SOURCE_TRUST_VARIABLE_GATE=1
            SOURCE_TRUST_QUERY_MODE=prompt_embedding
            HYPER_PHYS_AGREEMENT_GUARD=0
            HYPER_PHYS_CONTEXT_MODULATION=0
            HYPER_PHYS_FORMULA_OPERATOR=0
            HYPER_PHYS_CONSISTENCY_GUARD=0
            HYPER_PHYS_GAIN_BASIS_RESIDUAL=0
            ZERO_SHOT_PRIOR_FORM=source_base_residual_reliability_gated
            SOURCE_RESIDUAL_RHO=1.0
        elif [[ "${ABLATION_ID}" == "M3_14_source_trained_phys_formula_gain_hypertrust" ]]; then
            if [[ "${TARGET_REGION}" != "US-R1" && "${M3_14_ALLOW_FROZEN_CONFIRMATION}" != "1" && "${M3_14_ALLOW_FROZEN_CONFIRMATION,,}" != "true" ]]; then
                echo "ERROR: M3_14 current physics ablation is restricted to US-R1." >&2
                echo "US-R2..US-R6 are deferred frozen-confirmation regions; set M3_14_ALLOW_FROZEN_CONFIRMATION=1 only after method freeze." >&2
                exit 2
            fi
            if [[ "${SEED}" != "0" && "${M3_14_ALLOW_FROZEN_CONFIRMATION}" != "1" && "${M3_14_ALLOW_FROZEN_CONFIRMATION,,}" != "true" ]]; then
                echo "ERROR: M3_14 current physics ablation is restricted to seed=0." >&2
                exit 2
            fi
            if [[ "${RESUME_FROM_M3_1_BEST}" == "1" || "${RESUME_FROM_M3_1_BEST,,}" == "true" ]]; then
                echo "ERROR: M3_14_source_trained_phys_formula_gain_hypertrust cannot warm-start from M3_1." >&2
                echo "Use only --init_from_source_base_checkpoint via SOURCE_CHECKPOINT; do not set RESUME_FROM_M3_1_BEST=1." >&2
                exit 2
            fi
            if [[ -n "${USER_SET_HYPER_PHYS_GAIN_BASIS_RESIDUAL}" && "${HYPER_PHYS_GAIN_BASIS_RESIDUAL}" != "0" && "${HYPER_PHYS_GAIN_BASIS_RESIDUAL,,}" != "false" ]]; then
                echo "ERROR: M3_14 forbids final-output physics residual branches; set HYPER_PHYS_GAIN_BASIS_RESIDUAL=0." >&2
                exit 2
            fi
            if [[ -n "${USER_SET_TRAINABLE_SCOPE}" && "${TRAINABLE_SCOPE}" != "source_base_frozen_adapter_film" ]]; then
                echo "ERROR: M3_14 requires TRAINABLE_SCOPE=source_base_frozen_adapter_film." >&2
                exit 2
            fi
            HYPER_SOURCE_TRUST_ROUTING=1
            HYPER_SOURCE_TRUST_STRENGTH=0.50
            HYPER_SOURCE_TRUST_TOP_M=4
            HYPER_SOURCE_TRUST_VARIABLE_GATE=1
            SOURCE_TRUST_QUERY_MODE=prompt_embedding
            HYPER_PHYS_AGREEMENT_GUARD=0
            HYPER_PHYS_CONTEXT_MODULATION=1
            HYPER_PHYS_FORMULA_OPERATOR=1
            PHYS_CONTEXT_SOURCE=raw_input_side_formula_gain
            PHYS_FORMULA_MODE=enkf_rt_vertical_temp
            PHYS_FORMULA_SOURCE=raw_input_side_formula_gain
            if [[ -z "${USER_SET_HYPER_PHYS_DELTA_SCALE}" ]]; then
                HYPER_PHYS_DELTA_SCALE=0.05
            fi
            if [[ -z "${USER_SET_HYPER_PHYS_GATE_INIT}" ]]; then
                HYPER_PHYS_GATE_INIT=0.50
            fi
            if [[ -z "${USER_SET_HYPER_OPERATOR_DROPPATH_P}" ]]; then
                HYPER_OPERATOR_DROPPATH_P=0.10
            fi
            if [[ -z "${USER_SET_HYPER_PHYS_CONSISTENCY_GUARD}" ]]; then
                HYPER_PHYS_CONSISTENCY_GUARD=0
            fi
            if [[ -z "${USER_SET_PHYS_CONSISTENCY_SOURCE}" ]]; then
                PHYS_CONSISTENCY_SOURCE=raw_input_side_formula_gain
            fi
            HYPER_PHYS_GAIN_BASIS_RESIDUAL=0
            if [[ -z "${USER_SET_HYPER_PHYS_CONSISTENCY_REGULARIZATION_WEIGHT}" ]]; then
                HYPER_PHYS_CONSISTENCY_REGULARIZATION_WEIGHT=0.01
            fi
            TRAINABLE_SCOPE=source_base_frozen_adapter_film
            if [[ -z "${USER_SET_USE_AMP}" ]]; then
                USE_AMP=0
            fi
            if [[ -z "${USER_SET_BATCH_SIZE}" ]]; then
                BATCH_SIZE=64
            fi
            if [[ -z "${USER_SET_ACCUM_STEPS}" ]]; then
                ACCUM_STEPS=2
            fi
            if [[ -z "${USER_SET_LR}" ]]; then
                LR=2e-4
            fi
            if [[ -z "${USER_SET_MAX_EPOCHS}" ]]; then
                MAX_EPOCHS=50
            fi
        elif [[ "${ABLATION_ID}" == "M3_15_m31_anchored_source_safe_phys_coeff_delta" ]]; then
            if [[ -n "${USER_SET_RESUME_FROM_M3_1_BEST}" && "${RESUME_FROM_M3_1_BEST}" != "1" && "${RESUME_FROM_M3_1_BEST,,}" != "true" ]]; then
                echo "ERROR: M3_15_m31_anchored_source_safe_phys_coeff_delta requires RESUME_FROM_M3_1_BEST=1." >&2
                exit 2
            fi
            if [[ -n "${USER_SET_TRAINABLE_SCOPE}" && "${TRAINABLE_SCOPE}" != "phys_coeff_delta_only" ]]; then
                echo "ERROR: M3_15 requires TRAINABLE_SCOPE=phys_coeff_delta_only." >&2
                exit 2
            fi
            if [[ -n "${USER_SET_HYPER_PHYS_GAIN_BASIS_RESIDUAL}" && "${HYPER_PHYS_GAIN_BASIS_RESIDUAL}" != "0" && "${HYPER_PHYS_GAIN_BASIS_RESIDUAL,,}" != "false" ]]; then
                echo "ERROR: M3_15 forbids final-output physics residual branches; set HYPER_PHYS_GAIN_BASIS_RESIDUAL=0." >&2
                exit 2
            fi
            if [[ -n "${USER_SET_HYPER_PHYS_CONSISTENCY_REGULARIZATION_WEIGHT}" && "${HYPER_PHYS_CONSISTENCY_REGULARIZATION_WEIGHT}" != "0" && "${HYPER_PHYS_CONSISTENCY_REGULARIZATION_WEIGHT}" != "0.0" ]]; then
                echo "ERROR: M3_15 keeps source-fit sign consistency loss diagnostic-only; set HYPER_PHYS_CONSISTENCY_REGULARIZATION_WEIGHT=0." >&2
                exit 2
            fi
            RESUME_FROM_M3_1_BEST=1
            TRAINABLE_SCOPE=phys_coeff_delta_only
            USE_AMP=0
            HYPER_SOURCE_TRUST_ROUTING=1
            HYPER_SOURCE_TRUST_STRENGTH=0.50
            HYPER_SOURCE_TRUST_TOP_M=4
            HYPER_SOURCE_TRUST_VARIABLE_GATE=1
            SOURCE_TRUST_QUERY_MODE=prompt_embedding
            HYPER_PHYS_AGREEMENT_GUARD=0
            HYPER_PHYS_CONTEXT_MODULATION=1
            HYPER_PHYS_FORMULA_OPERATOR=1
            PHYS_CONTEXT_SOURCE=raw_input_side_formula_gain
            PHYS_FORMULA_MODE=enkf_rt_vertical_temp
            PHYS_FORMULA_SOURCE=raw_input_side_formula_gain
            HYPER_PHYS_GAIN_BASIS_RESIDUAL=0
            HYPER_PHYS_CONSISTENCY_GUARD=0
            HYPER_PHYS_CONSISTENCY_REGULARIZATION_WEIGHT=0.0
            if [[ -z "${USER_SET_HYPER_PHYS_DELTA_SCALE}" ]]; then
                HYPER_PHYS_DELTA_SCALE=0.05
            fi
            if [[ -z "${USER_SET_HYPER_PHYS_GATE_INIT}" ]]; then
                HYPER_PHYS_GATE_INIT=0.50
            fi
            if [[ -z "${USER_SET_HYPER_OPERATOR_DROPPATH_P}" ]]; then
                HYPER_OPERATOR_DROPPATH_P=0.10
            fi
            if [[ -z "${USER_SET_MAX_EPOCHS}" ]]; then
                MAX_EPOCHS=5
            fi
            if [[ -z "${USER_SET_EVAL_EVERY_EPOCHS}" ]]; then
                EVAL_EVERY_EPOCHS=1
            fi
            if [[ -z "${USER_SET_LR}" ]]; then
                LR=1e-4
                USER_SET_LR=branch_default
            fi
        elif [[ "${ABLATION_ID}" == "M3_16_source_only_phys_m3trust_lite" ]]; then
            if [[ "${TARGET_REGION}" != "US-R1" && "${M3_16_ALLOW_FROZEN_CONFIRMATION}" != "1" && "${M3_16_ALLOW_FROZEN_CONFIRMATION,,}" != "true" ]]; then
                echo "ERROR: M3_16 current physics ablation is restricted to US-R1." >&2
                echo "US-R2..US-R6 are deferred frozen-confirmation regions; set M3_16_ALLOW_FROZEN_CONFIRMATION=1 only after method freeze." >&2
                exit 2
            fi
            if [[ "${SEED}" != "0" && "${M3_16_ALLOW_FROZEN_CONFIRMATION}" != "1" && "${M3_16_ALLOW_FROZEN_CONFIRMATION,,}" != "true" ]]; then
                echo "ERROR: M3_16 current physics ablation is restricted to seed=0." >&2
                exit 2
            fi
            if [[ "${RESUME_FROM_M3_1_BEST}" == "1" || "${RESUME_FROM_M3_1_BEST,,}" == "true" ]]; then
                echo "ERROR: M3_16_source_only_phys_m3trust_lite cannot warm-start from M3_1." >&2
                echo "Use only --init_from_source_base_checkpoint via SOURCE_CHECKPOINT; do not set RESUME_FROM_M3_1_BEST=1." >&2
                exit 2
            fi
            if [[ -n "${USER_SET_HYPER_PHYS_GAIN_BASIS_RESIDUAL}" && "${HYPER_PHYS_GAIN_BASIS_RESIDUAL}" != "0" && "${HYPER_PHYS_GAIN_BASIS_RESIDUAL,,}" != "false" ]]; then
                echo "ERROR: M3_16 forbids final-output physics residual branches; set HYPER_PHYS_GAIN_BASIS_RESIDUAL=0." >&2
                exit 2
            fi
            if [[ -n "${USER_SET_TRAINABLE_SCOPE}" && "${TRAINABLE_SCOPE}" != "source_base_frozen_adapter_film" ]]; then
                echo "ERROR: M3_16 requires TRAINABLE_SCOPE=source_base_frozen_adapter_film." >&2
                exit 2
            fi
            HYPER_SOURCE_TRUST_ROUTING=1
            HYPER_SOURCE_TRUST_STRENGTH=0.50
            HYPER_SOURCE_TRUST_TOP_M=4
            HYPER_SOURCE_TRUST_VARIABLE_GATE=1
            SOURCE_TRUST_QUERY_MODE=prompt_embedding
            HYPER_PHYS_AGREEMENT_GUARD=0
            HYPER_PHYS_CONTEXT_MODULATION=1
            HYPER_PHYS_FORMULA_OPERATOR=1
            PHYS_CONTEXT_SOURCE=raw_input_side_formula_gain
            PHYS_FORMULA_MODE=enkf_rt_vertical_temp
            PHYS_FORMULA_SOURCE=raw_input_side_formula_gain
            HYPER_PHYS_GAIN_BASIS_RESIDUAL=0
            HYPER_PHYS_CONSISTENCY_GUARD=0
            HYPER_PHYS_CONSISTENCY_REGULARIZATION_WEIGHT=0.0
            TRAINABLE_SCOPE=source_base_frozen_adapter_film
            if [[ -z "${USER_SET_HYPER_PHYS_DELTA_SCALE}" ]]; then
                HYPER_PHYS_DELTA_SCALE=0.03
            fi
            if [[ -z "${USER_SET_HYPER_PHYS_GATE_INIT}" ]]; then
                HYPER_PHYS_GATE_INIT=0.25
            fi
            if [[ -z "${USER_SET_HYPER_OPERATOR_DROPPATH_P}" ]]; then
                HYPER_OPERATOR_DROPPATH_P=0.10
            fi
            if [[ -z "${USER_SET_USE_AMP}" ]]; then
                USE_AMP=0
            fi
            if [[ -z "${USER_SET_BATCH_SIZE}" ]]; then
                BATCH_SIZE=64
            fi
            if [[ -z "${USER_SET_ACCUM_STEPS}" ]]; then
                ACCUM_STEPS=2
            fi
            if [[ -z "${USER_SET_LR}" ]]; then
                LR=2e-4
            fi
            if [[ -z "${USER_SET_MAX_EPOCHS}" ]]; then
                MAX_EPOCHS=50
            fi
        else
            HYPER_SOURCE_TRUST_ROUTING=0
            HYPER_SOURCE_TRUST_STRENGTH=0.0
            HYPER_SOURCE_TRUST_VARIABLE_GATE=0
        fi
        if [[ -z "${USER_SET_HYPER_RANK_GATE_TEMPERATURE_INIT}" ]]; then
            HYPER_RANK_GATE_TEMPERATURE_INIT=2.0
        fi
        if [[ -z "${USER_SET_USE_AMP}" ]]; then
            USE_AMP=0
        fi
        if [[ -z "${USER_SET_BATCH_SIZE}" ]]; then
            BATCH_SIZE=64
        fi
        if [[ -z "${USER_SET_ACCUM_STEPS}" ]]; then
            ACCUM_STEPS=2
        fi
        if [[ -z "${USER_SET_LR}" ]]; then
            LR=2e-4
        fi
        HYPER_ENABLE_FILM=1
        HYPER_ENABLE_ADAPTERS=1
        ;;
    M3_film_only)
        HYPER_COEFF_GENERATOR=per_adapter
        HYPER_RELIABILITY_GATE=none
        HYPER_ENABLE_FILM=1
        HYPER_ENABLE_ADAPTERS=0
        ;;
    M4_adapter_only)
        HYPER_COEFF_GENERATOR=shared_layer_aware
        HYPER_RELIABILITY_GATE=none
        HYPER_ENABLE_FILM=0
        HYPER_ENABLE_ADAPTERS=1
        ;;
    *)
        echo "ERROR: Unsupported ABLATION_ID=${ABLATION_ID}" >&2
        echo "Expected one of: M0_current M1_shared_coeff M2_shared_coeff_gate M2_rank_gated_dora M2_1_rank_gated_dora_stable M2_2_source_saliency_prior M2_3_source_safe_residual_hyperda M2_5a_da_aware_prompt_only M2_5b_da_aware_conservative_router M2_6_source_manifold_guarded_prior M3_0_hyperda_trust_light M3_1_hyperda_trust_medium M3_1a_trust_medium_dualalpha M3_1b_trust_mid_high M3_1c_trust_medium_local M3_1d_trust_medium_broad M3_2_hyperda_trust_raw_reliability M3_2_phys_trust_raw_da_query M3_2a_phys_trust_raw_query_fixed M3_4_phys_trust_blended_query M3_5_phys_agreement_guarded_trust M3_5b_phys_agreement_floor_guard M3_6_phys_token_operator_droppath_trust M3_7_phys_formula_consistency_guarded_trust M3_8_phys_formula_operator_trust M3_8b_phys_formula_light_guarded_trust M3_8c_phys_formula_light_operator_only_trust M3_9_phys_formula_enhanced_trust M3_12_phys_gain_basis_hypertrust M3_13_phys_gain_guarded_hypertrust M3_14_source_trained_phys_formula_gain_hypertrust M3_15_m31_anchored_source_safe_phys_coeff_delta M3_16_source_only_phys_m3trust_lite M3_3_hyperda_trust_selection_only M3_film_only M4_adapter_only" >&2
        exit 2
        ;;
esac

RUN_NAME="${RUN_NAME:-phase4_hyperda_staged_ablation_${ABLATION_ID}_${TARGET_REGION}_s${SEED}_${TIMESTAMP}}"
RUN_DIR="${OUTPUT_ROOT}/${ABLATION_ID}/${TARGET_REGION}/${RUN_NAME}"

resolve_auto_source_checkpoint() {
    local target_region="$1"
    local seed="$2"
    local pattern="artifacts/runs/phase4_source_only/*${target_region}_*s${seed}_*/checkpoints/checkpoint_best_source_val_safe_score.pt"
    local checkpoint
    checkpoint="$(find artifacts/runs/phase4_source_only \
        -path "*/checkpoints/checkpoint_best_source_val_safe_score.pt" \
        -path "*${target_region}_*s${seed}_*" \
        -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
    if [[ -z "${checkpoint}" ]]; then
        echo "ERROR: auto source checkpoint not found for target=${target_region} seed=${seed}" >&2
        echo "Looked under pattern: ${pattern}" >&2
        exit 2
    fi
    echo "${checkpoint}"
}

resolve_auto_m2_1_saliency_source_checkpoint() {
    local target_region="$1"
    local seed="$2"
    find artifacts/runs/phase4_hyperda_staged_ablation/M2_1_rank_gated_dora_stable \
        -path "*/${target_region}/*s${seed}*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
        -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-
}

resolve_auto_m3_1_best_checkpoint() {
    local target_region="$1"
    local seed="$2"
    find artifacts/runs/phase4_hyperda_staged_ablation/M3_1_hyperda_trust_medium \
        -path "*/${target_region}/*s${seed}*/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt" \
        -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-
}

run_or_print() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY_RUN:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

if [[ "${ABLATION_ID}" != "M3_15_m31_anchored_source_safe_phys_coeff_delta" && "${SOURCE_CHECKPOINT}" == "auto" ]]; then
    SOURCE_CHECKPOINT="$(resolve_auto_source_checkpoint "${TARGET_REGION}" "${SEED}")"
fi

if [[ "${ABLATION_ID}" != "M3_15_m31_anchored_source_safe_phys_coeff_delta" && ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "ERROR: source checkpoint not found: ${SOURCE_CHECKPOINT}" >&2
    exit 2
fi

if [[ "${ABLATION_ID}" == "M2_2_source_saliency_prior" ]]; then
    if [[ -z "${HYPER_SOURCE_SALIENCY_PRIOR_PATH}" ]]; then
        HYPER_SOURCE_SALIENCY_PRIOR_PATH="${HYPER_SOURCE_SALIENCY_PRIOR_DIR}/${TARGET_REGION}_s${SEED}.pt"
    fi
fi

if [[ "${ABLATION_ID}" == "M2_3_source_safe_residual_hyperda" && "${M2_3_INIT_FROM_M2_1_CHECKPOINT}" == "auto" ]]; then
    M2_3_INIT_FROM_M2_1_CHECKPOINT="$(resolve_auto_m2_1_saliency_source_checkpoint "${TARGET_REGION}" "${SEED}")"
fi
if [[ "${ABLATION_ID}" == "M2_3_source_safe_residual_hyperda" && -n "${M2_3_INIT_FROM_M2_1_CHECKPOINT}" && ! -f "${M2_3_INIT_FROM_M2_1_CHECKPOINT}" ]]; then
    echo "ERROR: M2.3 M2.1 source prior checkpoint not found: ${M2_3_INIT_FROM_M2_1_CHECKPOINT}" >&2
    echo "Run M2.1 first or set M2_3_INIT_FROM_M2_1_CHECKPOINT= to train from the staged source base only." >&2
    exit 2
fi
if [[ "${ABLATION_ID}" == "M3_6_phys_token_operator_droppath_trust" && ( "${RESUME_FROM_M3_1_BEST}" == "1" || "${RESUME_FROM_M3_1_BEST,,}" == "true" ) && "${M3_6_INIT_FROM_M3_1_CHECKPOINT}" == "auto" ]]; then
    M3_6_INIT_FROM_M3_1_CHECKPOINT="$(resolve_auto_m3_1_best_checkpoint "${TARGET_REGION}" "${SEED}")"
    if [[ -z "${M3_6_INIT_FROM_M3_1_CHECKPOINT}" ]]; then
        echo "ERROR: RESUME_FROM_M3_1_BEST=1 but no M3.1 best checkpoint was found for target=${TARGET_REGION} seed=${SEED}" >&2
        echo "Looked under artifacts/runs/phase4_hyperda_staged_ablation/M3_1_hyperda_trust_medium" >&2
        exit 2
    fi
fi
if [[ "${ABLATION_ID}" == "M3_6_phys_token_operator_droppath_trust" && -n "${M3_6_INIT_FROM_M3_1_CHECKPOINT}" && "${M3_6_INIT_FROM_M3_1_CHECKPOINT}" != "auto" && ! -f "${M3_6_INIT_FROM_M3_1_CHECKPOINT}" ]]; then
    echo "ERROR: M3.6 M3.1 source prior checkpoint not found: ${M3_6_INIT_FROM_M3_1_CHECKPOINT}" >&2
    echo "Run M3.1 first or set RESUME_FROM_M3_1_BEST=0 to train from the staged source base." >&2
    exit 2
fi
if [[ "${ABLATION_ID}" == "M3_7_phys_formula_consistency_guarded_trust" && ( "${RESUME_FROM_M3_1_BEST}" == "1" || "${RESUME_FROM_M3_1_BEST,,}" == "true" ) && "${M3_7_INIT_FROM_M3_1_CHECKPOINT}" == "auto" ]]; then
    M3_7_INIT_FROM_M3_1_CHECKPOINT="$(resolve_auto_m3_1_best_checkpoint "${TARGET_REGION}" "${SEED}")"
    if [[ -z "${M3_7_INIT_FROM_M3_1_CHECKPOINT}" ]]; then
        echo "ERROR: RESUME_FROM_M3_1_BEST=1 but no M3.1 best checkpoint was found for target=${TARGET_REGION} seed=${SEED}" >&2
        echo "Looked under artifacts/runs/phase4_hyperda_staged_ablation/M3_1_hyperda_trust_medium" >&2
        exit 2
    fi
fi
if [[ "${ABLATION_ID}" == "M3_7_phys_formula_consistency_guarded_trust" && -n "${M3_7_INIT_FROM_M3_1_CHECKPOINT}" && "${M3_7_INIT_FROM_M3_1_CHECKPOINT}" != "auto" && ! -f "${M3_7_INIT_FROM_M3_1_CHECKPOINT}" ]]; then
    echo "ERROR: M3.7 M3.1 source prior checkpoint not found: ${M3_7_INIT_FROM_M3_1_CHECKPOINT}" >&2
    echo "Run M3.1 first or set RESUME_FROM_M3_1_BEST=0 to evaluate from the staged source base." >&2
    exit 2
fi
if [[ "${ABLATION_ID}" == "M3_8_phys_formula_operator_trust" && ( "${RESUME_FROM_M3_1_BEST}" == "1" || "${RESUME_FROM_M3_1_BEST,,}" == "true" ) && "${M3_8_INIT_FROM_M3_1_CHECKPOINT}" == "auto" ]]; then
    M3_8_INIT_FROM_M3_1_CHECKPOINT="$(resolve_auto_m3_1_best_checkpoint "${TARGET_REGION}" "${SEED}")"
    if [[ -z "${M3_8_INIT_FROM_M3_1_CHECKPOINT}" ]]; then
        echo "ERROR: RESUME_FROM_M3_1_BEST=1 but no M3.1 best checkpoint was found for target=${TARGET_REGION} seed=${SEED}" >&2
        echo "Looked under artifacts/runs/phase4_hyperda_staged_ablation/M3_1_hyperda_trust_medium" >&2
        exit 2
    fi
fi
if [[ "${ABLATION_ID}" == "M3_8_phys_formula_operator_trust" && -n "${M3_8_INIT_FROM_M3_1_CHECKPOINT}" && "${M3_8_INIT_FROM_M3_1_CHECKPOINT}" != "auto" && ! -f "${M3_8_INIT_FROM_M3_1_CHECKPOINT}" ]]; then
    echo "ERROR: M3.8 M3.1 source prior checkpoint not found: ${M3_8_INIT_FROM_M3_1_CHECKPOINT}" >&2
    echo "Run M3.1 first or set RESUME_FROM_M3_1_BEST=0 to train from the staged source base." >&2
    exit 2
fi
if [[ "${ABLATION_ID}" == "M3_9_phys_formula_enhanced_trust" && ( "${RESUME_FROM_M3_1_BEST}" == "1" || "${RESUME_FROM_M3_1_BEST,,}" == "true" ) && "${M3_9_INIT_FROM_M3_1_CHECKPOINT}" == "auto" ]]; then
    M3_9_INIT_FROM_M3_1_CHECKPOINT="$(resolve_auto_m3_1_best_checkpoint "${TARGET_REGION}" "${SEED}")"
    if [[ -z "${M3_9_INIT_FROM_M3_1_CHECKPOINT}" ]]; then
        echo "ERROR: RESUME_FROM_M3_1_BEST=1 but no M3.1 best checkpoint was found for target=${TARGET_REGION} seed=${SEED}" >&2
        echo "Looked under artifacts/runs/phase4_hyperda_staged_ablation/M3_1_hyperda_trust_medium" >&2
        exit 2
    fi
fi
if [[ "${ABLATION_ID}" == "M3_9_phys_formula_enhanced_trust" && -n "${M3_9_INIT_FROM_M3_1_CHECKPOINT}" && "${M3_9_INIT_FROM_M3_1_CHECKPOINT}" != "auto" && ! -f "${M3_9_INIT_FROM_M3_1_CHECKPOINT}" ]]; then
    echo "ERROR: M3.9 M3.1 source prior checkpoint not found: ${M3_9_INIT_FROM_M3_1_CHECKPOINT}" >&2
    echo "Run M3.1 first or set RESUME_FROM_M3_1_BEST=0 to train from the staged source base." >&2
    exit 2
fi
if [[ "${ABLATION_ID}" == "M3_13_phys_gain_guarded_hypertrust" && ( "${RESUME_FROM_M3_1_BEST}" == "1" || "${RESUME_FROM_M3_1_BEST,,}" == "true" ) && "${M3_13_INIT_FROM_M3_1_CHECKPOINT}" == "auto" ]]; then
    M3_13_INIT_FROM_M3_1_CHECKPOINT="$(resolve_auto_m3_1_best_checkpoint "${TARGET_REGION}" "${SEED}")"
    if [[ -z "${M3_13_INIT_FROM_M3_1_CHECKPOINT}" ]]; then
        echo "ERROR: RESUME_FROM_M3_1_BEST=1 but no M3.1 best checkpoint was found for target=${TARGET_REGION} seed=${SEED}" >&2
        echo "Looked under artifacts/runs/phase4_hyperda_staged_ablation/M3_1_hyperda_trust_medium" >&2
        exit 2
    fi
fi
if [[ "${ABLATION_ID}" == "M3_13_phys_gain_guarded_hypertrust" && -n "${M3_13_INIT_FROM_M3_1_CHECKPOINT}" && "${M3_13_INIT_FROM_M3_1_CHECKPOINT}" != "auto" && ! -f "${M3_13_INIT_FROM_M3_1_CHECKPOINT}" ]]; then
    echo "ERROR: M3.13 M3.1 source prior checkpoint not found: ${M3_13_INIT_FROM_M3_1_CHECKPOINT}" >&2
    echo "Run M3.1 first or set M3_13_INIT_FROM_M3_1_CHECKPOINT=/path/to/M3_1_best.pt." >&2
    exit 2
fi
if [[ "${ABLATION_ID}" == "M3_15_m31_anchored_source_safe_phys_coeff_delta" && ( "${RESUME_FROM_M3_1_BEST}" == "1" || "${RESUME_FROM_M3_1_BEST,,}" == "true" ) && "${M3_15_INIT_FROM_M3_1_CHECKPOINT}" == "auto" ]]; then
    M3_15_INIT_FROM_M3_1_CHECKPOINT="$(resolve_auto_m3_1_best_checkpoint "${TARGET_REGION}" "${SEED}")"
    if [[ -z "${M3_15_INIT_FROM_M3_1_CHECKPOINT}" ]]; then
        echo "ERROR: RESUME_FROM_M3_1_BEST=1 but no M3.1 best checkpoint was found for target=${TARGET_REGION} seed=${SEED}" >&2
        echo "Looked under artifacts/runs/phase4_hyperda_staged_ablation/M3_1_hyperda_trust_medium" >&2
        exit 2
    fi
fi
if [[ "${ABLATION_ID}" == "M3_15_m31_anchored_source_safe_phys_coeff_delta" && -n "${M3_15_INIT_FROM_M3_1_CHECKPOINT}" && "${M3_15_INIT_FROM_M3_1_CHECKPOINT}" != "auto" && ! -f "${M3_15_INIT_FROM_M3_1_CHECKPOINT}" ]]; then
    echo "ERROR: M3.15 M3.1 source prior checkpoint not found: ${M3_15_INIT_FROM_M3_1_CHECKPOINT}" >&2
    echo "Run M3.1 first or set M3_15_INIT_FROM_M3_1_CHECKPOINT=/path/to/M3_1_best.pt." >&2
    exit 2
fi
RESOLVED_DATASET_BACKEND="${DATASET_BACKEND}"
if [[ "${DATASET_BACKEND}" == "auto" ]]; then
    if [[ -f "${TENSOR_CACHE_DIR}/manifest_region_crops_US.json" ]]; then
        RESOLVED_DATASET_BACKEND="tensor_cache"
    else
        RESOLVED_DATASET_BACKEND="netcdf"
    fi
fi
if [[ "${RESOLVED_DATASET_BACKEND}" != "netcdf" && "${RESOLVED_DATASET_BACKEND}" != "tensor_cache" ]]; then
    echo "ERROR: DATASET_BACKEND must be auto, netcdf, or tensor_cache; got ${DATASET_BACKEND}" >&2
    exit 2
fi
if [[ "${RESOLVED_DATASET_BACKEND}" == "tensor_cache" && ! -f "${TENSOR_CACHE_DIR}/manifest_region_crops_US.json" ]]; then
    echo "ERROR: tensor cache manifest not found: ${TENSOR_CACHE_DIR}/manifest_region_crops_US.json" >&2
    exit 2
fi

if [[ "${ABLATION_ID}" == "M2_2_source_saliency_prior" && "${DRY_RUN}" != "1" && ! -f "${HYPER_SOURCE_SALIENCY_PRIOR_PATH}" ]]; then
    M2_1_SALIENCY_SOURCE_CHECKPOINT="$(resolve_auto_m2_1_saliency_source_checkpoint "${TARGET_REGION}" "${SEED}")"
    if [[ -z "${M2_1_SALIENCY_SOURCE_CHECKPOINT}" ]]; then
        echo "ERROR: M2_2_source_saliency_prior source saliency prior not found: ${HYPER_SOURCE_SALIENCY_PRIOR_PATH}" >&2
        echo "" >&2
        echo "No matching M2.1 stable HyperDA source-stage checkpoint was found under:" >&2
        echo "  artifacts/runs/phase4_hyperda_staged_ablation/M2_1_rank_gated_dora_stable/${TARGET_REGION}" >&2
        echo "" >&2
        echo "Run M2.1 first, then build the prior:" >&2
        echo "  ABLATION_ID=M2_1_rank_gated_dora_stable bash $0 auto ${TARGET_REGION} ${SEED} ${CUDA_VISIBLE_DEVICES}" >&2
        echo "  PYTHONPATH=. python scripts/train/build_source_basis_saliency_prior.py \\" >&2
        echo "    --source_checkpoint /path/to/M2_1/checkpoints/checkpoint_best_source_val_transfer_safe_score.pt \\" >&2
        echo "    --target_region \"${TARGET_REGION}\" \\" >&2
        echo "    --output_path \"${HYPER_SOURCE_SALIENCY_PRIOR_PATH}\" \\" >&2
        echo "    --source_split source_fit" >&2
        echo "" >&2
        echo "Or pass an existing artifact explicitly:" >&2
        echo "  HYPER_SOURCE_SALIENCY_PRIOR_PATH=/path/to/prior.pt ABLATION_ID=M2_2_source_saliency_prior bash $0 ${SOURCE_CHECKPOINT} ${TARGET_REGION} ${SEED} ${CUDA_VISIBLE_DEVICES}" >&2
        exit 2
    fi
    if [[ "${HYPER_SOURCE_SALIENCY_AUTO_BUILD}" == "1" || "${HYPER_SOURCE_SALIENCY_AUTO_BUILD,,}" == "true" ]]; then
        echo "M2.2 source saliency prior not found; building it from M2.1 stable checkpoint:"
        echo "  source_checkpoint=${M2_1_SALIENCY_SOURCE_CHECKPOINT}"
        echo "  output_path=${HYPER_SOURCE_SALIENCY_PRIOR_PATH}"
        mkdir -p "$(dirname "${HYPER_SOURCE_SALIENCY_PRIOR_PATH}")"
        python scripts/train/build_source_basis_saliency_prior.py \
            --source_checkpoint "${M2_1_SALIENCY_SOURCE_CHECKPOINT}" \
            --target_region "${TARGET_REGION}" \
            --output_path "${HYPER_SOURCE_SALIENCY_PRIOR_PATH}" \
            --source_split source_fit \
            --seed "${SEED}" \
            --dataset_backend "${RESOLVED_DATASET_BACKEND}" \
            --tensor_cache_dir "${TENSOR_CACHE_DIR}" \
            --max_year_cache_entries "${MAX_YEAR_CACHE_ENTRIES}" \
            --tensor_cache_load_mode "${TENSOR_CACHE_LOAD_MODE}" \
            --batch_size "${HYPER_SOURCE_SALIENCY_BATCH_SIZE}" \
            --max_batches "${HYPER_SOURCE_SALIENCY_MAX_BATCHES}" \
            --num_workers "${NUM_WORKERS}" \
            --device cuda
    else
        echo "ERROR: M2_2_source_saliency_prior source saliency prior not found: ${HYPER_SOURCE_SALIENCY_PRIOR_PATH}" >&2
        echo "" >&2
        echo "Auto-build is disabled by HYPER_SOURCE_SALIENCY_AUTO_BUILD=${HYPER_SOURCE_SALIENCY_AUTO_BUILD}." >&2
        echo "Build it manually with:" >&2
        echo "  PYTHONPATH=. python scripts/train/build_source_basis_saliency_prior.py \\" >&2
        echo "    --source_checkpoint \"${M2_1_SALIENCY_SOURCE_CHECKPOINT}\" \\" >&2
        echo "    --target_region \"${TARGET_REGION}\" \\" >&2
        echo "    --output_path \"${HYPER_SOURCE_SALIENCY_PRIOR_PATH}\" \\" >&2
        echo "    --source_split source_fit" >&2
        exit 2
    fi
fi

if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${RUN_DIR}"
fi

echo "============================================"
echo "Phase 4 staged HyperDA / HyperDA-TRUST ablation"
echo "  ablation_id=${ABLATION_ID}"
echo "  target_region=${TARGET_REGION}"
echo "  seed=${SEED}"
echo "  gpu=${CUDA_VISIBLE_DEVICES}"
echo "  stage1_method=source_pooled_global_backbone"
echo "  stage2_frozen=source_base_backbone_and_head"
echo "  stage2_trainable=prompt_encoder,film,basis_adapter_generation,reliability_gate_when_enabled"
echo "  adaptation_setting=zero_shot_context  K=0"
echo "  source_fit=2015-2021 source_val=2022"
echo "  target_context=2015-2021 input-side only"
echo "  target_val=unused_in_main_protocol"
echo "  target_eval=2023-2025"
echo "  target_labels=none"
echo "  split_artifact=${SPLITS_JSON}"
echo "  source_base_checkpoint=${SOURCE_CHECKPOINT}"
if [[ "${ABLATION_ID}" == "M2_3_source_safe_residual_hyperda" ]]; then
    echo "  m2_3_init_from_m2_1_checkpoint=${M2_3_INIT_FROM_M2_1_CHECKPOINT:-none}"
elif [[ "${ABLATION_ID}" == "M3_6_phys_token_operator_droppath_trust" ]]; then
    echo "  m3_6_init_from_m3_1_checkpoint=${M3_6_INIT_FROM_M3_1_CHECKPOINT:-none}"
elif [[ "${ABLATION_ID}" == "M3_7_phys_formula_consistency_guarded_trust" ]]; then
    echo "  m3_7_init_from_m3_1_checkpoint=${M3_7_INIT_FROM_M3_1_CHECKPOINT:-none}"
elif [[ "${ABLATION_ID}" == "M3_8_phys_formula_operator_trust" ]]; then
    echo "  m3_8_init_from_m3_1_checkpoint=${M3_8_INIT_FROM_M3_1_CHECKPOINT:-none}"
elif [[ "${ABLATION_ID}" == "M3_8b_phys_formula_light_guarded_trust" || "${ABLATION_ID}" == "M3_8c_phys_formula_light_operator_only_trust" ]]; then
    echo "  source_stage_initialization=source_only_checkpoint_clean_hypernetwork_training"
    echo "  prompt_checkpoint_warm_start=none_for_clean_physics_candidate"
elif [[ "${ABLATION_ID}" == "M3_12_phys_gain_basis_hypertrust" ]]; then
    echo "  source_stage_initialization=source_only_checkpoint_clean_hypernetwork_training"
    echo "  prompt_checkpoint_warm_start=forbidden_for_phys_gain_basis_hypertrust"
elif [[ "${ABLATION_ID}" == "M3_9_phys_formula_enhanced_trust" ]]; then
    echo "  m3_9_init_from_m3_1_checkpoint=${M3_9_INIT_FROM_M3_1_CHECKPOINT:-none}"
elif [[ "${ABLATION_ID}" == "M3_13_phys_gain_guarded_hypertrust" ]]; then
    echo "  m3_13_init_from_m3_1_checkpoint=${M3_13_INIT_FROM_M3_1_CHECKPOINT:-none}"
    echo "  source_stage_initialization=M3_1_best_checkpoint_eval_only"
    echo "  prompt_checkpoint_warm_start=required_for_phys_gain_guarded_hypertrust"
elif [[ "${ABLATION_ID}" == "M3_15_m31_anchored_source_safe_phys_coeff_delta" ]]; then
    echo "  m3_15_init_from_m3_1_checkpoint=${M3_15_INIT_FROM_M3_1_CHECKPOINT:-none}"
    echo "  source_stage_initialization=M3_1_best_checkpoint_phys_coeff_delta_only"
    echo "  source_base_checkpoint_usage=not_loaded_for_m3_15"
    echo "  prompt_checkpoint_warm_start=required_for_m31_anchored_candidate"
elif [[ "${ABLATION_ID}" == "M3_14_source_trained_phys_formula_gain_hypertrust" ]]; then
    echo "  checkpoint_start=source_pooled_global_backbone"
    echo "  source_stage_initialization=source_only_checkpoint_clean_hypernetwork_training"
    echo "  prompt_checkpoint_warm_start=forbidden_for_source_trained_physics_candidate"
    echo "  current_ablation_policy=US-R1_seed0_K0_only"
elif [[ "${ABLATION_ID}" == "M3_16_source_only_phys_m3trust_lite" ]]; then
    echo "  checkpoint_start=source_pooled_global_backbone"
    echo "  source_stage_initialization=source_only_checkpoint_clean_hypernetwork_training"
    echo "  prompt_checkpoint_warm_start=forbidden_for_source_only_physics_mainline"
    echo "  current_ablation_policy=US-R1_seed0_K0_only"
fi
echo "  target_labels_used_for_adaptation=false"
echo "  target_eval_input_stats_used_for_update=false"
echo "  trainable_scope=${TRAINABLE_SCOPE}"
echo "  context_encoder=${CONTEXT_ENCODER}"
if [[ "${CONTEXT_ENCODER}" == "robust_input_side_da_diagnostics" ]]; then
    echo "  prompt_diagnostics=DA-aware prompt/router diagnostics from input-side fields only"
    echo "  prompt_diagnostic_input_domain=normalized_input_side_legacy"
elif [[ "${CONTEXT_ENCODER}" == "robust_input_side_da_diagnostics_raw" ]]; then
    echo "  prompt_diagnostics=DA-aware prompt/router diagnostics from raw input-side fields only"
    echo "  prompt_diagnostic_input_domain=raw_input_side"
fi
if [[ "${ABLATION_ID}" == "M2_5a_da_aware_prompt_only" ]]; then
    echo "  m2_5a_failure_interpretation=source_val_improved_target_k0_degraded"
    echo "  diagnostic_status=negative_diagnostic_non_strict_prompt_only"
elif [[ "${ABLATION_ID}" == "M2_5b_da_aware_conservative_router" ]]; then
    echo "  m2_5a_failure_interpretation=source_val_improved_target_k0_degraded"
    echo "  diagnostic_status=da_aware_conservative_router_source_stage_diagnostic"
elif [[ "${ABLATION_ID}" == "M2_6_source_manifold_guarded_prior" ]]; then
    echo "  diagnostic_status=diagnostic_source_manifold_guarded_prior"
    echo "  m2_1_anchor=context_encoder=current_mean_std,shared_layer_aware_rank_gated_stable,dora_like_gain_bounded,top_k=4,temperature=2.0,USE_AMP=0,LR=2e-4"
    echo "  source_manifold_guard_source=source_fit_source_val_only"
    echo "  target_eval_usage=final_eval_only_no_selection"
elif [[ "${ABLATION_ID}" == M3_* ]]; then
    if [[ "${ABLATION_ID}" == "M3_5_phys_agreement_guarded_trust" ]]; then
        echo "  diagnostic_status=phys_agreement_guarded_trust_source_gated_candidate"
        echo "  m3_1_anchor=trust_strength=0.50,top_m=4,context_encoder=current_mean_std"
        echo "  prompt_trust_geometry=prompt_embedding"
        echo "  phys_trust_usage=guard_only_shrink_no_enhance"
        echo "  phys_guard_reads=x,x_raw,month,region_mask_only"
        echo "  source_val_gate=reject_if_dual_cvar_more_than_0.005_below_M3_1"
        echo "  source_trust_query_used_as_neighbor_geometry=false"
    elif [[ "${ABLATION_ID}" == "M3_5b_phys_agreement_floor_guard" ]]; then
        echo "  diagnostic_status=phys_agreement_floor_guard_source_gated_candidate"
        echo "  m3_1_anchor=trust_strength=0.50,top_m=4,context_encoder=current_mean_std"
        echo "  prompt_trust_geometry=prompt_embedding"
        echo "  phys_trust_usage=guard_only_bounded_shrink_no_enhance"
        echo "  phys_guard_reads=x,x_raw,month,region_mask_only"
        echo "  phys_guard_risk_rule=${HYPER_PHYS_AGREEMENT_GUARD_RISK_RULE}"
        echo "  phys_guard_min_multiplier=${HYPER_PHYS_AGREEMENT_GUARD_MIN_MULTIPLIER}"
        echo "  source_val_gate=reject_if_dual_cvar_more_than_0.005_below_M3_1"
        echo "  source_trust_query_used_as_neighbor_geometry=false"
    elif [[ "${ABLATION_ID}" == "M3_6_phys_token_operator_droppath_trust" ]]; then
        echo "  diagnostic_status=phys_token_operator_droppath_trust_source_gated_candidate"
        echo "  stage2_candidate=true"
        echo "  m3_1_anchor=trust_strength=0.50,top_m=4,context_encoder=current_mean_std"
        echo "  prompt_trust_geometry=prompt_embedding"
        echo "  trust_routing_geometry=prompt_embedding"
        echo "  phys_token_usage=operator_coefficient_delta_only"
        echo "  phys_context_source=${PHYS_CONTEXT_SOURCE}"
        echo "  phys_delta_scale=${HYPER_PHYS_DELTA_SCALE}"
        echo "  phys_gate_init=${HYPER_PHYS_GATE_INIT}"
        echo "  operator_droppath_p=${HYPER_OPERATOR_DROPPATH_P}"
        echo "  operator_droppath_train_only=true"
        echo "  phys_delta_head_zero_init=true"
        echo "  source_val_gate=reject_if_dual_cvar_more_than_0.005_below_M3_1"
        echo "  source_trust_query_used_as_neighbor_geometry=false"
        echo "  warm_start_policy=M3_1_best_checkpoint_phys_branch_only_first_screen"
    elif [[ "${ABLATION_ID}" == "M3_7_phys_formula_consistency_guarded_trust" ]]; then
        echo "  diagnostic_status=phys_formula_consistency_guarded_trust_source_gated_candidate"
        echo "  stage2_candidate=false"
        echo "  eval_only_guard_supported=true"
        echo "  m3_1_anchor=trust_strength=0.50,top_m=4,context_encoder=current_mean_std"
        echo "  prompt_trust_geometry=prompt_embedding"
        echo "  trust_routing_geometry=prompt_embedding"
        echo "  phys_consistency_usage=raw_input_side_formula_variable_trust_gate_shrink_or_identity"
        echo "  phys_consistency_source=${PHYS_CONSISTENCY_SOURCE}"
        echo "  phys_guard_reads=x_raw,month,region_mask_only"
        echo "  source_val_gate=reject_if_dual_cvar_more_than_0.005_below_M3_1"
        echo "  target_eval_usage=final_eval_only_no_selection"
        echo "  source_trust_query_used_as_neighbor_geometry=false"
        echo "  warm_start_policy=M3_1_best_checkpoint_eval_only_source_gate"
    elif [[ "${ABLATION_ID}" == "M3_8_phys_formula_operator_trust" ]]; then
        echo "  diagnostic_status=phys_formula_operator_trust_source_gated_candidate"
        echo "  stage2_candidate=source_side_only"
        echo "  m3_1_anchor=trust_strength=0.50,top_m=4,context_encoder=current_mean_std"
        echo "  prompt_trust_geometry=prompt_embedding"
        echo "  trust_routing_geometry=prompt_embedding"
        echo "  phys_formula_usage=raw_input_side_formula_operator_coefficient_delta"
        echo "  phys_context_source=${PHYS_CONTEXT_SOURCE}"
        echo "  phys_formula_mode=${PHYS_FORMULA_MODE}"
        echo "  phys_formula_source=${PHYS_FORMULA_SOURCE}"
        echo "  phys_formula_delta_scale=${HYPER_PHYS_DELTA_SCALE}"
        echo "  phys_formula_gate_init=${HYPER_PHYS_GATE_INIT}"
        echo "  phys_formula_operator_droppath_p=${HYPER_OPERATOR_DROPPATH_P}"
        echo "  phys_delta_head_zero_init=true"
        echo "  phys_consistency_usage=surface_primary_product_variable_trust_gate_shrink_or_identity"
        echo "  phys_consistency_guard_mode=${PHYS_CONSISTENCY_GUARD_MODE}"
        echo "  phys_consistency_source=${PHYS_CONSISTENCY_SOURCE}"
        echo "  phys_guard_reads=x_raw,month,region_mask_only"
        echo "  channel_11_usage=diagnostic_only_not_hard_mask"
        echo "  source_val_gate=reject_if_dual_cvar_below_0.441573390549_or_rmse_degrades_gt_0p5pct_vs_M3_1"
        echo "  source_val_anchor_dual_cvar=0.446573390549"
        echo "  source_val_anchor_rmse_surface=0.004712299814"
        echo "  source_val_anchor_rmse_rootzone=0.000889948021"
        echo "  target_eval_usage=final_eval_only_no_selection"
        echo "  source_trust_query_used_as_neighbor_geometry=false"
        echo "  warm_start_policy=M3_1_best_checkpoint_phys_formula_branch_only_first_screen"
    elif [[ "${ABLATION_ID}" == "M3_8b_phys_formula_light_guarded_trust" || "${ABLATION_ID}" == "M3_8c_phys_formula_light_operator_only_trust" ]]; then
        if [[ "${ABLATION_ID}" == "M3_8b_phys_formula_light_guarded_trust" ]]; then
            echo "  diagnostic_status=phys_formula_light_guarded_trust_main_method_candidate"
            echo "  phys_consistency_usage=shrink_only_high_floor_variable_trust_gate"
        else
            echo "  diagnostic_status=phys_formula_light_operator_only_trust_guard_ablation"
            echo "  phys_consistency_usage=disabled_operator_only_guard_ablation"
        fi
        echo "  main_method_candidate=Physics-informed HyperDA-TRUST"
        echo "  stage2_candidate=source_side_only"
        echo "  m3_1_design_anchor=trust_strength=0.50,top_m=4,context_encoder=current_mean_std"
        echo "  prompt_trust_geometry=prompt_embedding"
        echo "  trust_routing_geometry=prompt_embedding"
        echo "  phys_formula_usage=raw_input_side_formula_operator_coefficient_delta"
        echo "  phys_context_source=${PHYS_CONTEXT_SOURCE}"
        echo "  phys_formula_mode=${PHYS_FORMULA_MODE}"
        echo "  phys_formula_source=${PHYS_FORMULA_SOURCE}"
        echo "  phys_formula_schema=phys_formula_operator_v2_input_side"
        echo "  phys_formula_delta_scale=${HYPER_PHYS_DELTA_SCALE}"
        echo "  phys_formula_gate_init=${HYPER_PHYS_GATE_INIT}"
        echo "  phys_formula_operator_droppath_p=${HYPER_OPERATOR_DROPPATH_P}"
        echo "  phys_delta_head_zero_init=true"
        echo "  phys_consistency_guard_mode=${PHYS_CONSISTENCY_GUARD_MODE}"
        echo "  phys_consistency_source=${PHYS_CONSISTENCY_SOURCE}"
        echo "  phys_guard_reads=x_raw,month,region_mask_only"
        echo "  channel_11_usage=diagnostic_only_not_hard_mask"
        echo "  source_val_selection_rule=dual_cvar_gte_M3_1_minus_0.005_choose_best_source_safe_score_tie_rmse"
        echo "  target_eval_acceptance=improve_one_variable_and_other_degrades_lte_0.2pct_or_both_degrade_lte_0.2pct_else_diagnostic"
        echo "  target_eval_usage=final_eval_only_no_selection"
        echo "  source_trust_query_used_as_neighbor_geometry=false"
        echo "  warm_start_policy=none_clean_source_only_checkpoint_full_hypernetwork_training"
    elif [[ "${ABLATION_ID}" == "M3_9_phys_formula_enhanced_trust" ]]; then
        echo "  diagnostic_status=phys_formula_enhanced_trust_source_gated_candidate"
        echo "  stage2_candidate=source_side_only"
        echo "  m3_1_anchor=trust_strength=0.50,top_m=4,context_encoder=current_mean_std"
        echo "  prompt_trust_geometry=prompt_embedding"
        echo "  trust_routing_geometry=prompt_embedding"
        echo "  phys_formula_usage=enhanced_raw_input_side_formula_operator_coefficient_delta"
        echo "  phys_context_source=${PHYS_CONTEXT_SOURCE}"
        echo "  phys_formula_mode=${PHYS_FORMULA_MODE}"
        echo "  phys_formula_source=${PHYS_FORMULA_SOURCE}"
        echo "  phys_formula_schema=phys_formula_operator_v2_enhanced_input_side"
        echo "  phys_formula_enhanced_features=tb_hv_normalized_innovation,innovation_asymmetry,polarization_mismatch,vegopacity,weak_obs,finite,temp_contrast,vertical_decoupling,hydraulic_gradient,channel11_diagnostic"
        echo "  phys_formula_delta_scale=${HYPER_PHYS_DELTA_SCALE}"
        echo "  phys_formula_gate_init=${HYPER_PHYS_GATE_INIT}"
        echo "  phys_formula_operator_droppath_p=${HYPER_OPERATOR_DROPPATH_P}"
        echo "  phys_delta_head_zero_init=true"
        echo "  phys_consistency_usage=shrink_only_high_floor_variable_trust_gate"
        echo "  phys_consistency_guard_mode=${PHYS_CONSISTENCY_GUARD_MODE}"
        echo "  phys_consistency_source=${PHYS_CONSISTENCY_SOURCE}"
        echo "  phys_guard_reads=x_raw,month,region_mask_only"
        echo "  channel_11_usage=diagnostic_only_not_hard_mask"
        echo "  source_side_cheap_screen=US-R1_source_val_only_cap384_no_target_eval"
        echo "  source_val_gate=cap384_select_best_rootzone_if_dual_cvar_within_0.0005_of_M3_8_V1_and_surface_degrade_lte_0.2pct_else_stop"
        echo "  full_confirmation_gate=dual_cvar_gte_0.441573390549_surface_rootzone_rmse_degrade_lte_0.5pct_vs_M3_1_leakage_clean"
        echo "  target_eval_usage=final_eval_only_no_selection"
        echo "  source_trust_query_used_as_neighbor_geometry=false"
        echo "  warm_start_policy=M3_1_best_checkpoint_phys_formula_branch_only_first_screen"
    elif [[ "${ABLATION_ID}" == "M3_12_phys_gain_basis_hypertrust" ]]; then
        echo "  diagnostic_status=rejected_negative_diagnostic"
        echo "  method_role=do_not_use_as_active_route_or_target_eval_tuning_basis"
        echo "  m3_1_design_anchor=trust_strength=0.50,top_m=4,context_encoder=current_mean_std"
        echo "  prompt_trust_geometry=prompt_embedding"
        echo "  trust_routing_geometry=prompt_embedding"
        echo "  phys_gain_basis_usage=raw_input_side_formula_basis_output_residual"
        echo "  phys_gain_basis_schema=phys_gain_basis_hypertrust_v1"
        echo "  phys_gain_basis_maps=B_H,B_V,B_pol,B_temp,B_vert"
        echo "  phys_gain_basis_formula=enkf_innovation_tau_omega_surface_rootzone_coupling"
        echo "  phys_gain_coeff_heads_zero_init=true"
        echo "  phys_gain_source_bank=source_fit_gain_priors_interpretation_metadata_only"
        echo "  source_gain_bank_uses=source_fit_only"
        echo "  forbidden_warm_start=M3_1_or_M2_1_prompt_checkpoint"
        echo "  channel_11_usage=diagnostic_only_not_hard_mask"
        echo "  source_val_gate=dual_cvar_gte_M3_1_minus_0.003_surface_rootzone_rmse_degrade_lte_0.3pct_worst_region_degrade_lte_0.5pct"
        echo "  target_eval_acceptance=improve_one_variable_rmse_gte_0.2pct_other_degrade_lte_0.2pct_or_interpretability_if_both_degrade_lte_0.5pct"
        echo "  target_eval_usage=final_eval_only_no_selection"
        echo "  source_trust_query_used_as_neighbor_geometry=false"
        echo "  warm_start_policy=none_clean_source_only_checkpoint_full_hypernetwork_training"
    elif [[ "${ABLATION_ID}" == "M3_13_phys_gain_guarded_hypertrust" ]]; then
        echo "  diagnostic_status=phys_gain_guarded_hypertrust_source_calibrated_no_harm_guard"
        echo "  method_role=M3_1_plus_source_calibrated_physics_gain_no_harm_guard"
        echo "  m3_1_anchor=trust_strength=0.50,top_m=4,context_encoder=current_mean_std"
        echo "  prompt_trust_geometry=prompt_embedding"
        echo "  trust_routing_geometry=prompt_embedding"
        echo "  guard_action=shrink_only_pred_m3_1_minus_source_base"
        echo "  guard_formula=pred=source_base+guard*(pred_M3_1-source_base)"
        echo "  guard_min=0.90"
        echo "  eta_grid=0,0.02,0.05,0.10"
        echo "  eta_zero_contract=exact_M3_1_identity"
        echo "  source_gain_bank_uses=source_fit_only"
        echo "  eta_selection_source=source_val_only"
        echo "  source_gate_json_required_for_target_eval=true"
        echo "  target_eval_policy=run_once_only_after_source_gate_passes"
        echo "  target_eval_usage=final_eval_only_no_selection"
        echo "  channel_11_usage=diagnostic_only_not_hard_mask"
        echo "  neural_training_epochs=0"
        echo "  neural_parameter_updates=0"
        echo "  warm_start_policy=M3_1_best_checkpoint_eval_only_source_gate"
    elif [[ "${ABLATION_ID}" == "M3_14_source_trained_phys_formula_gain_hypertrust" ]]; then
        echo "  diagnostic_status=source_trained_phys_formula_gain_hypertrust_design_candidate"
        echo "  method_role=M3_1_architecture_route_plus_source_trained_formula_gain_operator_logits"
        echo "  m3_1_design_anchor=trust_strength=0.50,top_m=4,context_encoder=current_mean_std"
        echo "  checkpoint_start=source_pooled_global_backbone"
        echo "  prompt_trust_geometry=prompt_embedding"
        echo "  trust_routing_geometry=prompt_embedding"
        echo "  phys_formula_usage=raw_input_side_formula_gain_bounded_operator_coefficient_logit_delta"
        echo "  phys_context_source=${PHYS_CONTEXT_SOURCE}"
        echo "  phys_formula_mode=${PHYS_FORMULA_MODE}"
        echo "  phys_formula_source=${PHYS_FORMULA_SOURCE}"
        echo "  phys_formula_schema=m3_14_raw_input_side_formula_gain_v1"
        echo "  phys_formula_features=d_H,d_V,m_H,m_V,gamma,rho_H,rho_V,B_pol,B_temp,B_vert,source_gain_prior_summaries,finite_coverage,channel11_diagnostic"
        echo "  phys_formula_delta_scale=${HYPER_PHYS_DELTA_SCALE}"
        echo "  phys_formula_gate_init=${HYPER_PHYS_GATE_INIT}"
        echo "  phys_formula_operator_droppath_p=${HYPER_OPERATOR_DROPPATH_P}"
        echo "  phys_delta_head_zero_init=true"
        echo "  final_output_residual_allowed=false"
        echo "  hyper_phys_gain_basis_residual=0"
        echo "  phys_consistency_usage=weak_source_fit_regularization_only_no_target_eval_tuning"
        echo "  phys_consistency_regularization_weight=${HYPER_PHYS_CONSISTENCY_REGULARIZATION_WEIGHT}"
        echo "  diagnostic_guard_default=disabled_shrink_only_if_future_enabled"
        echo "  source_val_selection_rule=source_val_not_weaker_than_M3_1_select_by_dual_variable_cvar_safe_score"
        echo "  target_eval_acceptance=US_R1_development_ablation_improve_one_variable_other_degrades_lte_0.2pct_else_diagnostic_only"
        echo "  target_eval_usage=US_R1_development_ablation_only_no_selection_until_method_freeze"
        echo "  current_region_policy=US-R1_seed0_K0_only_R2_to_R6_deferred_until_method_freeze"
        echo "  channel_11_usage=diagnostic_only_not_hard_mask"
        echo "  source_trust_query_used_as_neighbor_geometry=false"
        echo "  warm_start_policy=none_clean_source_only_checkpoint_full_hypernetwork_training"
    elif [[ "${ABLATION_ID}" == "M3_15_m31_anchored_source_safe_phys_coeff_delta" ]]; then
        echo "  diagnostic_status=m31_anchored_source_safe_phys_coeff_delta_candidate"
        echo "  method_role=M3_1_frozen_main_path_plus_small_physics_coefficient_delta_branch"
        echo "  m3_1_anchor=trust_strength=0.50,top_m=4,context_encoder=current_mean_std"
        echo "  checkpoint_start=M3_1_best_checkpoint"
        echo "  frozen_modules=M3_1_backbone,prompt_encoder,trust_routing,adapter_film_hypernetwork,formula_phys_context_encoder"
        echo "  trainable_modules=phys_operator_residual_only"
        echo "  prompt_trust_geometry=prompt_embedding"
        echo "  trust_routing_geometry=prompt_embedding"
        echo "  phys_formula_usage=raw_input_side_formula_gain_bounded_operator_coefficient_logit_delta"
        echo "  phys_context_source=${PHYS_CONTEXT_SOURCE}"
        echo "  phys_formula_mode=${PHYS_FORMULA_MODE}"
        echo "  phys_formula_source=${PHYS_FORMULA_SOURCE}"
        echo "  phys_formula_schema=m3_14_raw_input_side_formula_gain_v1"
        echo "  phys_formula_delta_scale=${HYPER_PHYS_DELTA_SCALE}"
        echo "  phys_formula_gate_init=${HYPER_PHYS_GATE_INIT}"
        echo "  phys_formula_operator_droppath_p=${HYPER_OPERATOR_DROPPATH_P}"
        echo "  phys_delta_head_zero_init=true"
        echo "  final_output_residual_allowed=false"
        echo "  hyper_phys_gain_basis_residual=0"
        echo "  phys_consistency_usage=diagnostic_only_no_source_fit_sign_loss"
        echo "  hyper_phys_consistency_regularization_weight=${HYPER_PHYS_CONSISTENCY_REGULARIZATION_WEIGHT}"
        echo "  prediction_interpolation=pred_v=(1-eta_v)*pred_m3_1_v+eta_v*pred_phys_coeff_v"
        echo "  eta_grid=0,0.1,0.25,0.5,1.0"
        echo "  eta_selection_source=source_val_only"
        echo "  eta_zero_contract=exact_M3_1_prediction_identity"
        echo "  source_gate=dual_cvar_gte_M3_1_plus_0.001_one_variable_rmse_improves_gte_0.1pct_other_degrades_lte_0.05pct_region_season_degrade_lte_0.3pct"
        echo "  source_val_anchor_dual_cvar=0.446573390549"
        echo "  identity_diagnostic_policy=if_best_eta_all_zero_refuse_target_eval"
        echo "  target_eval_policy=run_once_only_after_source_gate_passes"
        echo "  target_eval_usage=final_eval_only_no_selection_after_source_gate_passes"
        echo "  channel_11_usage=diagnostic_only_not_hard_mask"
        echo "  source_trust_query_used_as_neighbor_geometry=false"
        echo "  warm_start_policy=M3_1_best_checkpoint_phys_coeff_delta_branch_only"
    elif [[ "${ABLATION_ID}" == "M3_16_source_only_phys_m3trust_lite" ]]; then
        echo "  diagnostic_status=source_only_phys_m3trust_lite_mainline_candidate"
        echo "  method_role=M3_1_architecture_route_plus_source_only_lite_physics_operator_logits"
        echo "  active_stage2_physics_mainline=true"
        echo "  stage2_source_only_invariant=true"
        echo "  m3_1_design_anchor=trust_strength=0.50,top_m=4,context_encoder=current_mean_std"
        echo "  checkpoint_start=source_pooled_global_backbone"
        echo "  prompt_trust_geometry=prompt_embedding"
        echo "  trust_routing_geometry=prompt_embedding"
        echo "  phys_formula_usage=raw_input_side_formula_gain_bounded_operator_coefficient_logit_delta"
        echo "  phys_context_source=${PHYS_CONTEXT_SOURCE}"
        echo "  phys_formula_mode=${PHYS_FORMULA_MODE}"
        echo "  phys_formula_source=${PHYS_FORMULA_SOURCE}"
        echo "  phys_formula_schema=m3_14_raw_input_side_formula_gain_v1"
        echo "  phys_formula_features=d_H,d_V,m_H,m_V,gamma,rho_H,rho_V,B_pol,B_temp,B_vert,source_gain_prior_summaries,finite_coverage,channel11_diagnostic"
        echo "  phys_formula_delta_scale=${HYPER_PHYS_DELTA_SCALE}"
        echo "  phys_formula_gate_init=${HYPER_PHYS_GATE_INIT}"
        echo "  phys_formula_operator_droppath_p=${HYPER_OPERATOR_DROPPATH_P}"
        echo "  phys_delta_head_zero_init=true"
        echo "  final_output_residual_allowed=false"
        echo "  hyper_phys_gain_basis_residual=0"
        echo "  second_model_forward_allowed=false"
        echo "  phys_consistency_usage=disabled_no_source_fit_sign_loss"
        echo "  hyper_phys_consistency_regularization_weight=${HYPER_PHYS_CONSISTENCY_REGULARIZATION_WEIGHT}"
        echo "  source_fit_gain_bank_forbidden_roles=target_context,target_val,target_eval,target_full_train"
        echo "  source_val_gate=dual_cvar_gte_M3_1_no_obvious_variable_region_season_rmse_regression"
        echo "  target_eval_acceptance=improve_one_variable_and_other_degrades_lte_0.2pct"
        echo "  target_eval_usage=final_eval_only_once_after_source_val_gate_no_selection"
        echo "  current_region_policy=US-R1_seed0_K0_only_R2_to_R6_deferred_until_M3_16_freeze"
        echo "  channel_11_usage=diagnostic_only_not_hard_mask"
        echo "  source_trust_query_used_as_neighbor_geometry=false"
        echo "  warm_start_policy=none_clean_source_only_checkpoint_full_hypernetwork_training"
    else
        echo "  diagnostic_status=hyperda_trust_source_manifold_trust_routed_operator_generation"
    fi
    echo "  m2_1_anchor=context_encoder=current_mean_std,shared_layer_aware_rank_gated_stable,dora_like_gain_bounded,top_k=4,temperature=2.0,USE_AMP=0,LR=2e-4"
    echo "  trust_bank_source=source_fit_source_val_only"
    echo "  label_usage=none"
    echo "  target_eval_usage=final_eval_only_no_selection"
    echo "  alpha_selection_objective=dual_variable_cvar_safe_score"
    if [[ "${SOURCE_TRUST_QUERY_MODE}" == "raw_input_side_da_diagnostics" ]]; then
        echo "  has_separate_source_trust_query_required=true"
        echo "  source_trust_query_input_domain=raw_input_side"
    elif [[ "${SOURCE_TRUST_QUERY_MODE}" == "blended_prompt_raw_da_0p25" ]]; then
        echo "  has_separate_source_trust_query_required=true"
        echo "  source_trust_query_input_domain=blended_prompt_raw_input_side"
        echo "  source_trust_query_blend_lambda=${SOURCE_TRUST_QUERY_BLEND_LAMBDA}"
        echo "  main_prompt_unchanged_by_blended_query=true"
    fi
fi
echo "  hyper_coeff_generator=${HYPER_COEFF_GENERATOR}"
echo "  hyper_reliability_gate=${HYPER_RELIABILITY_GATE}"
echo "  hyper_reliability_init=${HYPER_RELIABILITY_INIT}"
echo "  hyper_rank_gate_top_k=${HYPER_RANK_GATE_TOP_K}"
echo "  hyper_rank_gate_temperature_init=${HYPER_RANK_GATE_TEMPERATURE_INIT}"
echo "  hyper_adapter_param_style=${HYPER_ADAPTER_PARAM_STYLE}"
echo "  hyper_source_saliency_prior_path=${HYPER_SOURCE_SALIENCY_PRIOR_PATH:-none}"
echo "  hyper_source_saliency_auto_build=${HYPER_SOURCE_SALIENCY_AUTO_BUILD}"
echo "  hyper_source_saliency_max_batches=${HYPER_SOURCE_SALIENCY_MAX_BATCHES}"
echo "  hyper_source_saliency_batch_size=${HYPER_SOURCE_SALIENCY_BATCH_SIZE}"
echo "  hyper_source_saliency_prior_beta=${HYPER_SOURCE_SALIENCY_PRIOR_BETA}"
echo "  hyper_source_saliency_prior_application=${HYPER_SOURCE_SALIENCY_PRIOR_APPLICATION}"
echo "  hyper_prompt_manifold_reliability=${HYPER_PROMPT_MANIFOLD_RELIABILITY}"
echo "  hyper_prompt_manifold_reliability_strength=${HYPER_PROMPT_MANIFOLD_RELIABILITY_STRENGTH}"
echo "  hyper_source_manifold_guard=${HYPER_SOURCE_MANIFOLD_GUARD}"
echo "  hyper_source_manifold_guard_strength=${HYPER_SOURCE_MANIFOLD_GUARD_STRENGTH}"
echo "  hyper_source_manifold_guard_distance_key=${HYPER_SOURCE_MANIFOLD_GUARD_DISTANCE_KEY}"
echo "  hyper_source_manifold_guard_min_multiplier=${HYPER_SOURCE_MANIFOLD_GUARD_MIN_MULTIPLIER}"
echo "  source_manifold_guard_calibration=${SOURCE_MANIFOLD_GUARD_CALIBRATION}"
echo "  hyper_source_trust_routing=${HYPER_SOURCE_TRUST_ROUTING}"
echo "  hyper_source_trust_strength=${HYPER_SOURCE_TRUST_STRENGTH}"
echo "  hyper_source_trust_top_m=${HYPER_SOURCE_TRUST_TOP_M}"
echo "  hyper_source_trust_variable_gate=${HYPER_SOURCE_TRUST_VARIABLE_GATE}"
echo "  source_trust_bank_calibration=${SOURCE_TRUST_BANK_CALIBRATION}"
echo "  source_trust_query_mode=${SOURCE_TRUST_QUERY_MODE}"
echo "  source_trust_query_blend_lambda=${SOURCE_TRUST_QUERY_BLEND_LAMBDA}"
echo "  hyper_phys_agreement_guard=${HYPER_PHYS_AGREEMENT_GUARD}"
echo "  hyper_phys_agreement_guard_strength=${HYPER_PHYS_AGREEMENT_GUARD_STRENGTH}"
echo "  hyper_phys_agreement_guard_min_multiplier=${HYPER_PHYS_AGREEMENT_GUARD_MIN_MULTIPLIER}"
echo "  hyper_phys_agreement_guard_risk_rule=${HYPER_PHYS_AGREEMENT_GUARD_RISK_RULE}"
echo "  hyper_phys_context_modulation=${HYPER_PHYS_CONTEXT_MODULATION}"
echo "  phys_context_source=${PHYS_CONTEXT_SOURCE}"
echo "  hyper_phys_formula_operator=${HYPER_PHYS_FORMULA_OPERATOR}"
echo "  phys_formula_mode=${PHYS_FORMULA_MODE}"
echo "  phys_formula_source=${PHYS_FORMULA_SOURCE}"
echo "  hyper_phys_delta_scale=${HYPER_PHYS_DELTA_SCALE}"
echo "  hyper_phys_gate_init=${HYPER_PHYS_GATE_INIT}"
echo "  hyper_operator_droppath_p=${HYPER_OPERATOR_DROPPATH_P}"
echo "  hyper_phys_consistency_guard=${HYPER_PHYS_CONSISTENCY_GUARD}"
echo "  phys_consistency_guard_mode=${PHYS_CONSISTENCY_GUARD_MODE}"
echo "  phys_consistency_source=${PHYS_CONSISTENCY_SOURCE}"
echo "  phys_consistency_min_surface=${PHYS_CONSISTENCY_MIN_SURFACE}"
echo "  phys_consistency_min_rootzone=${PHYS_CONSISTENCY_MIN_ROOTZONE}"
echo "  phys_consistency_strength_surface=${PHYS_CONSISTENCY_STRENGTH_SURFACE}"
echo "  phys_consistency_strength_rootzone=${PHYS_CONSISTENCY_STRENGTH_ROOTZONE}"
echo "  hyper_phys_gain_basis_residual=${HYPER_PHYS_GAIN_BASIS_RESIDUAL}"
echo "  hyper_phys_gain_basis_coeff_scale=${HYPER_PHYS_GAIN_BASIS_COEFF_SCALE}"
echo "  hyper_phys_gain_basis_residual_clip=${HYPER_PHYS_GAIN_BASIS_RESIDUAL_CLIP}"
echo "  hyper_phys_gain_basis_beta_init=${HYPER_PHYS_GAIN_BASIS_BETA_INIT}"
echo "  hyper_phys_consistency_regularization_weight=${HYPER_PHYS_CONSISTENCY_REGULARIZATION_WEIGHT}"
echo "  hyper_residual_magnitude_penalty=${HYPER_RESIDUAL_MAGNITUDE_PENALTY}"
echo "  hyper_coeff_entropy_floor=${HYPER_COEFF_ENTROPY_FLOOR}"
echo "  hyper_coeff_entropy_penalty=${HYPER_COEFF_ENTROPY_PENALTY}"
echo "  hyper_enable_film=${HYPER_ENABLE_FILM}"
echo "  hyper_enable_adapters=${HYPER_ENABLE_ADAPTERS}"
echo "  source_episode_prompt_policy=${SOURCE_EPISODE_PROMPT_POLICY}"
echo "  source_anchor_blend_calibration=${SOURCE_ANCHOR_BLEND_CALIBRATION}"
echo "  hyper_output_head_residual=${HYPER_OUTPUT_HEAD_RESIDUAL}"
echo "  zero_shot_prior_form=${ZERO_SHOT_PRIOR_FORM}"
echo "  source_residual_rho=${SOURCE_RESIDUAL_RHO}"
echo "  source_residual_gate=${SOURCE_RESIDUAL_GATE}"
echo "  source_residual_gate_init=${SOURCE_RESIDUAL_GATE_INIT}"
echo "  dataset_backend=${RESOLVED_DATASET_BACKEND}"
echo "  tensor_cache_dir=${TENSOR_CACHE_DIR}"
echo "  max_year_cache_entries=${MAX_YEAR_CACHE_ENTRIES}"
echo "  tensor_cache_load_mode=${TENSOR_CACHE_LOAD_MODE}"
echo "  train_batch_sampler=${TRAIN_BATCH_SAMPLER}"
echo "  source_fit_max_batches_per_epoch=${SOURCE_FIT_MAX_BATCHES_PER_EPOCH}"
if [[ "${SOURCE_FIT_MAX_BATCHES_PER_EPOCH}" != "0" ]]; then
    echo "  source_fit_fast_screen=enabled_source_fit_training_cap_full_source_val"
else
    echo "  source_fit_fast_screen=disabled_full_source_fit_training"
fi
echo "  num_workers=${NUM_WORKERS}"
echo "  use_amp=${USE_AMP}"
echo "  eval_every_epochs=${EVAL_EVERY_EPOCHS}"
echo "  log_every_steps=${LOG_EVERY_STEPS}"
echo "  source_prototype_cache_dir=${SOURCE_PROTOTYPE_CACHE_DIR:-none}"
echo "  source_prototype_cache_mode=${SOURCE_PROTOTYPE_CACHE_MODE}"
echo "  source_trust_bank_cache_dir=${SOURCE_TRUST_BANK_CACHE_DIR:-none}"
echo "  source_trust_bank_cache_mode=${SOURCE_TRUST_BANK_CACHE_MODE}"
echo "  model_type=hyperda_basis_adapter width=${WIDTH} prompt_dim=${PROMPT_DIM}"
echo "  hyper_n_basis=${HYPER_N_BASIS} hyper_adapter_bottleneck=${HYPER_ADAPTER_BOTTLENECK}"
echo "  batch_size=${BATCH_SIZE} accum_steps=${ACCUM_STEPS} lr=${LR}"
echo "  selection_metric=${SELECTION_METRIC}"
echo "  output_dir=${RUN_DIR}"
echo "  build_ablation_table=${BUILD_ABLATION_TABLE}"
echo "  ablation_table_output_dir=${ABLATION_TABLE_OUTPUT_DIR}"
echo "  dry_run=${DRY_RUN}"
echo "============================================"

cmd=(
    python scripts/train/train_prompt_conditioned_shared.py
    --target_region "${TARGET_REGION}"
    --adaptation_setting zero_shot_context
    --K 0
    --seed "${SEED}"
    --device cuda
    --accum_steps "${ACCUM_STEPS}"
    --target_increment_normalization
    --use_lat_weighted_loss
    --batch_size "${BATCH_SIZE}"
    --max_epochs "${MAX_EPOCHS}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --grad_clip 1.0
    --num_workers "${NUM_WORKERS}"
    --width "${WIDTH}"
    --prompt_dim "${PROMPT_DIM}"
    --context_encoder "${CONTEXT_ENCODER}"
    --model_type hyperda_basis_adapter
    --hyper_n_basis "${HYPER_N_BASIS}"
    --hyper_adapter_bottleneck "${HYPER_ADAPTER_BOTTLENECK}"
    --hyper_adapter_scale "${HYPER_ADAPTER_SCALE}"
    --hyper_coeff_generator "${HYPER_COEFF_GENERATOR}"
    --hyper_rank_gate_top_k "${HYPER_RANK_GATE_TOP_K}"
    --hyper_rank_gate_temperature_init "${HYPER_RANK_GATE_TEMPERATURE_INIT}"
    --hyper_adapter_param_style "${HYPER_ADAPTER_PARAM_STYLE}"
    --hyper_reliability_gate "${HYPER_RELIABILITY_GATE}"
    --hyper_reliability_init "${HYPER_RELIABILITY_INIT}"
    --hyper_source_saliency_prior_path "${HYPER_SOURCE_SALIENCY_PRIOR_PATH}"
    --hyper_source_saliency_prior_beta "${HYPER_SOURCE_SALIENCY_PRIOR_BETA}"
    --hyper_source_saliency_prior_application "${HYPER_SOURCE_SALIENCY_PRIOR_APPLICATION}"
    --hyper_prompt_manifold_reliability "${HYPER_PROMPT_MANIFOLD_RELIABILITY}"
    --hyper_prompt_manifold_reliability_strength "${HYPER_PROMPT_MANIFOLD_RELIABILITY_STRENGTH}"
    --hyper_source_manifold_guard "${HYPER_SOURCE_MANIFOLD_GUARD}"
    --hyper_source_manifold_guard_strength "${HYPER_SOURCE_MANIFOLD_GUARD_STRENGTH}"
    --hyper_source_manifold_guard_distance_key "${HYPER_SOURCE_MANIFOLD_GUARD_DISTANCE_KEY}"
    --hyper_source_manifold_guard_min_multiplier "${HYPER_SOURCE_MANIFOLD_GUARD_MIN_MULTIPLIER}"
    --source_manifold_guard_calibration "${SOURCE_MANIFOLD_GUARD_CALIBRATION}"
    --hyper_source_trust_routing "${HYPER_SOURCE_TRUST_ROUTING}"
    --hyper_source_trust_strength "${HYPER_SOURCE_TRUST_STRENGTH}"
    --hyper_source_trust_top_m "${HYPER_SOURCE_TRUST_TOP_M}"
    --hyper_source_trust_variable_gate "${HYPER_SOURCE_TRUST_VARIABLE_GATE}"
    --source_trust_bank_calibration "${SOURCE_TRUST_BANK_CALIBRATION}"
    --source_trust_query_mode "${SOURCE_TRUST_QUERY_MODE}"
    --hyper_phys_agreement_guard "${HYPER_PHYS_AGREEMENT_GUARD}"
    --hyper_phys_agreement_guard_strength "${HYPER_PHYS_AGREEMENT_GUARD_STRENGTH}"
    --hyper_phys_agreement_guard_min_multiplier "${HYPER_PHYS_AGREEMENT_GUARD_MIN_MULTIPLIER}"
    --hyper_phys_agreement_guard_risk_rule "${HYPER_PHYS_AGREEMENT_GUARD_RISK_RULE}"
    --hyper_phys_context_modulation "${HYPER_PHYS_CONTEXT_MODULATION}"
    --phys_context_source "${PHYS_CONTEXT_SOURCE}"
    --hyper_phys_formula_operator "${HYPER_PHYS_FORMULA_OPERATOR}"
    --phys_formula_mode "${PHYS_FORMULA_MODE}"
    --phys_formula_source "${PHYS_FORMULA_SOURCE}"
    --hyper_phys_delta_scale "${HYPER_PHYS_DELTA_SCALE}"
    --hyper_phys_gate_init "${HYPER_PHYS_GATE_INIT}"
    --hyper_operator_droppath_p "${HYPER_OPERATOR_DROPPATH_P}"
    --hyper_phys_consistency_guard "${HYPER_PHYS_CONSISTENCY_GUARD}"
    --phys_consistency_guard_mode "${PHYS_CONSISTENCY_GUARD_MODE}"
    --phys_consistency_source "${PHYS_CONSISTENCY_SOURCE}"
    --phys_consistency_min_surface "${PHYS_CONSISTENCY_MIN_SURFACE}"
    --phys_consistency_min_rootzone "${PHYS_CONSISTENCY_MIN_ROOTZONE}"
    --phys_consistency_strength_surface "${PHYS_CONSISTENCY_STRENGTH_SURFACE}"
    --phys_consistency_strength_rootzone "${PHYS_CONSISTENCY_STRENGTH_ROOTZONE}"
    --hyper_phys_gain_basis_residual "${HYPER_PHYS_GAIN_BASIS_RESIDUAL}"
    --hyper_phys_gain_basis_coeff_scale "${HYPER_PHYS_GAIN_BASIS_COEFF_SCALE}"
    --hyper_phys_gain_basis_residual_clip "${HYPER_PHYS_GAIN_BASIS_RESIDUAL_CLIP}"
    --hyper_phys_gain_basis_beta_init "${HYPER_PHYS_GAIN_BASIS_BETA_INIT}"
    --hyper_phys_consistency_regularization_weight "${HYPER_PHYS_CONSISTENCY_REGULARIZATION_WEIGHT}"
    --hyper_enable_film "${HYPER_ENABLE_FILM}"
    --hyper_enable_adapters "${HYPER_ENABLE_ADAPTERS}"
    --hyper_residual_magnitude_penalty "${HYPER_RESIDUAL_MAGNITUDE_PENALTY}"
    --hyper_coeff_entropy_floor "${HYPER_COEFF_ENTROPY_FLOOR}"
    --hyper_coeff_entropy_penalty "${HYPER_COEFF_ENTROPY_PENALTY}"
    --source_episode_prompt_policy "${SOURCE_EPISODE_PROMPT_POLICY}"
    --source_anchor_blend_calibration "${SOURCE_ANCHOR_BLEND_CALIBRATION}"
    --hyper_output_head_residual "${HYPER_OUTPUT_HEAD_RESIDUAL}"
    --zero_shot_prior_form "${ZERO_SHOT_PRIOR_FORM}"
    --source_residual_rho "${SOURCE_RESIDUAL_RHO}"
    --source_residual_gate "${SOURCE_RESIDUAL_GATE}"
    --source_residual_gate_init "${SOURCE_RESIDUAL_GATE_INIT}"
    --source_residual_reliability_dim "${SOURCE_RESIDUAL_RELIABILITY_DIM}"
    --trainable_scope "${TRAINABLE_SCOPE}"
    --log_every_steps "${LOG_EVERY_STEPS}"
    --eval_every_epochs "${EVAL_EVERY_EPOCHS}"
    --checkpoint_every 10
    --selection_metric "${SELECTION_METRIC}"
    --splits_json "${SPLITS_JSON}"
    --split_manifest_path "${SPLIT_MANIFEST_PATH}"
    --protocol_freeze_id "${PROTOCOL_ID}"
    --dataset_backend "${RESOLVED_DATASET_BACKEND}"
    --tensor_cache_dir "${TENSOR_CACHE_DIR}"
    --max_year_cache_entries "${MAX_YEAR_CACHE_ENTRIES}"
    --tensor_cache_load_mode "${TENSOR_CACHE_LOAD_MODE}"
    --train_batch_sampler "${TRAIN_BATCH_SAMPLER}"
    --source_fit_max_batches_per_epoch "${SOURCE_FIT_MAX_BATCHES_PER_EPOCH}"
    --source_prototype_cache_mode "${SOURCE_PROTOTYPE_CACHE_MODE}"
    --source_trust_bank_cache_mode "${SOURCE_TRUST_BANK_CACHE_MODE}"
    --output_dir "${RUN_DIR}"
)

if [[ "${ABLATION_ID}" != "M3_15_m31_anchored_source_safe_phys_coeff_delta" ]]; then
    cmd+=(--init_from_source_base_checkpoint "${SOURCE_CHECKPOINT}")
fi

if [[ "${USE_AMP}" == "1" || "${USE_AMP,,}" == "true" ]]; then
    cmd+=(--amp)
fi

if [[ "${ABLATION_ID}" == "M2_3_source_safe_residual_hyperda" && -n "${M2_3_INIT_FROM_M2_1_CHECKPOINT}" ]]; then
    cmd+=(--init_from_prompt_checkpoint "${M2_3_INIT_FROM_M2_1_CHECKPOINT}")
fi
if [[ "${ABLATION_ID}" == "M3_6_phys_token_operator_droppath_trust" && -n "${M3_6_INIT_FROM_M3_1_CHECKPOINT}" && "${M3_6_INIT_FROM_M3_1_CHECKPOINT}" != "auto" ]]; then
    cmd+=(--init_from_prompt_checkpoint "${M3_6_INIT_FROM_M3_1_CHECKPOINT}")
fi
if [[ "${ABLATION_ID}" == "M3_7_phys_formula_consistency_guarded_trust" && -n "${M3_7_INIT_FROM_M3_1_CHECKPOINT}" && "${M3_7_INIT_FROM_M3_1_CHECKPOINT}" != "auto" ]]; then
    cmd+=(--init_from_prompt_checkpoint "${M3_7_INIT_FROM_M3_1_CHECKPOINT}")
fi
if [[ "${ABLATION_ID}" == "M3_8_phys_formula_operator_trust" && -n "${M3_8_INIT_FROM_M3_1_CHECKPOINT}" && "${M3_8_INIT_FROM_M3_1_CHECKPOINT}" != "auto" ]]; then
    cmd+=(--init_from_prompt_checkpoint "${M3_8_INIT_FROM_M3_1_CHECKPOINT}")
fi
if [[ "${ABLATION_ID}" == "M3_9_phys_formula_enhanced_trust" && -n "${M3_9_INIT_FROM_M3_1_CHECKPOINT}" && "${M3_9_INIT_FROM_M3_1_CHECKPOINT}" != "auto" ]]; then
    cmd+=(--init_from_prompt_checkpoint "${M3_9_INIT_FROM_M3_1_CHECKPOINT}")
fi
if [[ "${ABLATION_ID}" == "M3_13_phys_gain_guarded_hypertrust" && -n "${M3_13_INIT_FROM_M3_1_CHECKPOINT}" && "${M3_13_INIT_FROM_M3_1_CHECKPOINT}" != "auto" ]]; then
    cmd+=(--init_from_prompt_checkpoint "${M3_13_INIT_FROM_M3_1_CHECKPOINT}")
fi
if [[ "${ABLATION_ID}" == "M3_15_m31_anchored_source_safe_phys_coeff_delta" && -n "${M3_15_INIT_FROM_M3_1_CHECKPOINT}" && "${M3_15_INIT_FROM_M3_1_CHECKPOINT}" != "auto" ]]; then
    cmd+=(--init_from_prompt_checkpoint "${M3_15_INIT_FROM_M3_1_CHECKPOINT}")
fi

if [[ -n "${SOURCE_PROTOTYPE_CACHE_DIR}" ]]; then
    cmd+=(--source_prototype_cache_dir "${SOURCE_PROTOTYPE_CACHE_DIR}")
fi
if [[ -n "${SOURCE_TRUST_BANK_CACHE_DIR}" ]]; then
    cmd+=(--source_trust_bank_cache_dir "${SOURCE_TRUST_BANK_CACHE_DIR}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    run_or_print "${cmd[@]}"
else
    "${cmd[@]}" 2>&1 | tee "${RUN_DIR}/train_log.txt"
fi

table_cmd=(
    python scripts/analysis/build_hyperda_staged_ablation_table.py
    --runs_root "${OUTPUT_ROOT}"
    --output_dir "${ABLATION_TABLE_OUTPUT_DIR}"
    --target_region "${TARGET_REGION}"
    --seed "${SEED}"
)

if [[ "${BUILD_ABLATION_TABLE}" == "1" || "${BUILD_ABLATION_TABLE,,}" == "true" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
        run_or_print "${table_cmd[@]}"
    else
        "${table_cmd[@]}"
    fi
fi

echo "Done: staged HyperDA ablation ${ABLATION_ID} ${TARGET_REGION} seed=${SEED}"
