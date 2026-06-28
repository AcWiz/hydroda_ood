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
    echo "  bash run/phase5_hyperda_zero_few_shot.sh <source_checkpoint> ${TARGET_REGION} ${K} ${SEED} ${CUDA_VISIBLE_DEVICES}" >&2
    echo "Or train the staged source prior first:" >&2
    echo "  bash run/phase4_hyperda_staged.sh auto ${TARGET_REGION} ${SEED} ${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
fi

STAGE3_KSHOT_MODE="${STAGE3_KSHOT_MODE:-diagnostic_direct_kshot}"
DIAGNOSTIC_KSHOT_STRENGTH="${DIAGNOSTIC_KSHOT_STRENGTH:-strong}"
if [[ "${K}" == "0" ]]; then
    ADAPTATION_SETTING="zero_shot_context"
    DEFAULT_STEPS=0
    DEFAULT_LR="1e-3"
    DEFAULT_ANCHOR_ALPHA="0.0"
elif [[ "${K}" == "4" ]]; then
    ADAPTATION_SETTING="few_shot_k4"
    if [[ "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v1" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v2" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v3_stable" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v4_nested_stable" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v12_nested_cv" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v13_k12_aggressive_calibration_pool" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_affine_v1_nested" ]]; then
        DEFAULT_STEPS=0
        DEFAULT_LR="${LR_K4:-1e-3}"
        DEFAULT_ANCHOR_ALPHA="0.0"
    elif [[ "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v10_support_pool_nested" ]]; then
        DEFAULT_STEPS=0
        DEFAULT_LR="${LR_K4:-0.0}"
        DEFAULT_ANCHOR_ALPHA="${ANCHOR_ALPHA_K4:-0.30}"
    elif [[ "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested" ]]; then
        DEFAULT_STEPS=0
        DEFAULT_LR="${LR_K4:-0.0}"
        DEFAULT_ANCHOR_ALPHA="${ANCHOR_ALPHA_K4:-0.20}"
    elif [[ "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v6_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v7_balanced_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v8_hybrid_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v9_guarded_nested" ]]; then
        DEFAULT_STEPS=0
        DEFAULT_LR="${LR_K4:-0.0}"
        if [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
            DEFAULT_ANCHOR_ALPHA="${ANCHOR_ALPHA_K4:-0.30}"
        else
            DEFAULT_ANCHOR_ALPHA="${ANCHOR_ALPHA_K4:-0.40}"
        fi
    elif [[ "${STAGE3_KSHOT_MODE}" == "diagnostic_conservative_kshot_v3" || "${STAGE3_KSHOT_MODE}" == "diagnostic_safe_operator_v5_nested" ]]; then
        if [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
            DEFAULT_STEPS=20
            DEFAULT_ANCHOR_ALPHA="${ANCHOR_ALPHA_K4:-0.25}"
        else
            DEFAULT_STEPS=40
            DEFAULT_ANCHOR_ALPHA="${ANCHOR_ALPHA_K4:-0.35}"
        fi
        DEFAULT_LR="${LR_K4:-3e-4}"
    elif [[ "${STAGE3_KSHOT_MODE}" == "diagnostic_direct_kshot_v2" ]]; then
        if [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
            DEFAULT_STEPS=50
        else
            DEFAULT_STEPS=100
        fi
        DEFAULT_LR="${LR_K4:-1e-3}"
        DEFAULT_ANCHOR_ALPHA="${ANCHOR_ALPHA_K4:-1.0}"
    else
        DEFAULT_STEPS=100
        DEFAULT_LR="${LR_K4:-1e-3}"
        DEFAULT_ANCHOR_ALPHA="${ANCHOR_ALPHA_K4:-0.75}"
    fi
elif [[ "${K}" == "12" ]]; then
    ADAPTATION_SETTING="few_shot_k12"
    if [[ "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v1" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v2" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v3_stable" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_affine_v1_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v4_nested_stable" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v12_nested_cv" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v13_k12_aggressive_calibration_pool" ]]; then
        DEFAULT_STEPS=0
        DEFAULT_LR="${LR_K12:-1e-3}"
        DEFAULT_ANCHOR_ALPHA="0.0"
    elif [[ "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v10_support_pool_nested" ]]; then
        DEFAULT_STEPS=0
        DEFAULT_LR="${LR_K12:-0.0}"
        DEFAULT_ANCHOR_ALPHA="${ANCHOR_ALPHA_K12:-0.45}"
    elif [[ "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested" ]]; then
        DEFAULT_STEPS=0
        DEFAULT_LR="${LR_K12:-0.0}"
        DEFAULT_ANCHOR_ALPHA="${ANCHOR_ALPHA_K12:-0.30}"
    elif [[ "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v6_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v7_balanced_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v8_hybrid_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v9_guarded_nested" ]]; then
        DEFAULT_STEPS=0
        DEFAULT_LR="${LR_K12:-0.0}"
        if [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
            DEFAULT_ANCHOR_ALPHA="${ANCHOR_ALPHA_K12:-0.45}"
        else
            DEFAULT_ANCHOR_ALPHA="${ANCHOR_ALPHA_K12:-0.60}"
        fi
    elif [[ "${STAGE3_KSHOT_MODE}" == "diagnostic_conservative_kshot_v3" || "${STAGE3_KSHOT_MODE}" == "diagnostic_safe_operator_v5_nested" ]]; then
        if [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
            DEFAULT_STEPS=40
            DEFAULT_ANCHOR_ALPHA="${ANCHOR_ALPHA_K12:-0.35}"
        else
            DEFAULT_STEPS=80
            DEFAULT_ANCHOR_ALPHA="${ANCHOR_ALPHA_K12:-0.50}"
        fi
        DEFAULT_LR="${LR_K12:-3e-4}"
    elif [[ "${STAGE3_KSHOT_MODE}" == "diagnostic_direct_kshot_v2" ]]; then
        if [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
            DEFAULT_STEPS=100
        else
            DEFAULT_STEPS=200
        fi
        DEFAULT_LR="${LR_K12:-1e-3}"
        DEFAULT_ANCHOR_ALPHA="${ANCHOR_ALPHA_K12:-1.0}"
    else
        DEFAULT_STEPS=100
        DEFAULT_LR="${LR_K12:-3e-4}"
        DEFAULT_ANCHOR_ALPHA="${ANCHOR_ALPHA_K12:-0.25}"
    fi
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
ADAPT_SCOPE_WAS_SET="${ADAPT_SCOPE+x}"
STAGE3_POSTERIOR_POLICY_WAS_SET="${STAGE3_POSTERIOR_POLICY+x}"
SUPPORT_GATE_WAS_SET="${SUPPORT_GATE+x}"
SUPPORT_GATE_MIN_DELTA_WAS_SET="${SUPPORT_GATE_MIN_DELTA+x}"
SUPPORT_GATE_ROOTZONE_TOLERANCE_WAS_SET="${SUPPORT_GATE_ROOTZONE_TOLERANCE+x}"
SUPPORT_LOSS_REDUCTION_WAS_SET="${SUPPORT_LOSS_REDUCTION+x}"
FREEZE_MONTHLY_GAIN_WAS_SET="${FREEZE_MONTHLY_GAIN+x}"
TRUST_REGION_MODE_WAS_SET="${TRUST_REGION_MODE+x}"
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
SUPPORT_LOSS_REDUCTION="${SUPPORT_LOSS_REDUCTION:-global_pixel}"
FREEZE_MONTHLY_GAIN="${FREEZE_MONTHLY_GAIN:-0}"
TRUST_REGION_MODE="${TRUST_REGION_MODE:-none}"
TRUST_TOTAL_RADIUS="${TRUST_TOTAL_RADIUS:-0.0}"
TRUST_PROMPT_RADIUS="${TRUST_PROMPT_RADIUS:-0.0}"
TRUST_GAIN_RADIUS="${TRUST_GAIN_RADIUS:-0.0}"
TRUST_COEFF_RADIUS="${TRUST_COEFF_RADIUS:-0.0}"
TRUST_SPATIAL_RADIUS="${TRUST_SPATIAL_RADIUS:-0.0}"
ADAPT_SOLVER="${ADAPT_SOLVER:-adamw}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1.0}"
RIDGE_CLIP_COEFF_NORM="${RIDGE_CLIP_COEFF_NORM:-1.0}"
RIDGE_TRUST_REGION_RADIUS="${RIDGE_TRUST_REGION_RADIUS:-1.0}"
RIDGE_MAX_FEATURE_PIXELS="${RIDGE_MAX_FEATURE_PIXELS:-20000}"
RIDGE_STANDARDIZE_FEATURES="${RIDGE_STANDARDIZE_FEATURES:-0}"
RIDGE_WEIGHTING="${RIDGE_WEIGHTING:-global_pixel_l2}"
SAFE_POLICY_JSON="${SAFE_POLICY_JSON:-}"
REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT:-0}"
AUDIT_IDENTITY="${AUDIT_IDENTITY:-0}"
AUDIT_IDENTITY_TOLERANCE="${AUDIT_IDENTITY_TOLERANCE:-1e-8}"
TARGET_CONTEXT_MAX_SAMPLES="${TARGET_CONTEXT_MAX_SAMPLES:-0}"
STAGE3_CONTEXT_TTA="${STAGE3_CONTEXT_TTA:-none}"
CONTEXT_TTA_RESIDUAL_SCALE="${CONTEXT_TTA_RESIDUAL_SCALE:-0.05}"
CONTEXT_TTA_RESIDUAL_CLIP_L2="${CONTEXT_TTA_RESIDUAL_CLIP_L2:-0.0}"
if [[ "${K}" != "0" && ( "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v1" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v2" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v3_stable" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v4_nested_stable" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v12_nested_cv" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_gain_v13_k12_aggressive_calibration_pool" || "${STAGE3_KSHOT_MODE}" == "diagnostic_support_affine_v1_nested" ) ]]; then
    if [[ -z "${ADAPT_SCOPE_WAS_SET}" ]]; then ADAPT_SCOPE="none"; fi
    if [[ -z "${STAGE3_POSTERIOR_POLICY_WAS_SET}" ]]; then STAGE3_POSTERIOR_POLICY="source_calibrated_mix"; fi
    if [[ -z "${SUPPORT_GATE_WAS_SET}" ]]; then SUPPORT_GATE="off"; fi
    if [[ -z "${SUPPORT_LOSS_REDUCTION_WAS_SET}" ]]; then SUPPORT_LOSS_REDUCTION="cycle_balanced"; fi
    if [[ -z "${FREEZE_MONTHLY_GAIN_WAS_SET}" ]]; then FREEZE_MONTHLY_GAIN="1"; fi
elif [[ "${K}" != "0" && ( "${STAGE3_KSHOT_MODE}" == "diagnostic_conservative_kshot_v3" || "${STAGE3_KSHOT_MODE}" == "diagnostic_safe_operator_v5_nested" ) ]]; then
    if [[ -z "${ADAPT_SCOPE_WAS_SET}" ]]; then ADAPT_SCOPE="coeff_only"; fi
    if [[ -z "${STAGE3_POSTERIOR_POLICY_WAS_SET}" ]]; then STAGE3_POSTERIOR_POLICY="conservative_coeff_posterior"; fi
    if [[ -z "${SUPPORT_GATE_WAS_SET}" ]]; then SUPPORT_GATE="auto"; fi
    if [[ -z "${SUPPORT_GATE_MIN_DELTA_WAS_SET}" ]]; then SUPPORT_GATE_MIN_DELTA="1e-8"; fi
    if [[ -z "${SUPPORT_GATE_ROOTZONE_TOLERANCE_WAS_SET}" ]]; then SUPPORT_GATE_ROOTZONE_TOLERANCE="1e-8"; fi
    if [[ -z "${SUPPORT_LOSS_REDUCTION_WAS_SET}" ]]; then SUPPORT_LOSS_REDUCTION="cycle_balanced"; fi
    if [[ -z "${FREEZE_MONTHLY_GAIN_WAS_SET}" ]]; then FREEZE_MONTHLY_GAIN="1"; fi
    if [[ -z "${TRUST_REGION_MODE_WAS_SET}" ]]; then TRUST_REGION_MODE="groupwise"; fi
    if [[ "${TRUST_COEFF_RADIUS}" == "0.0" || "${TRUST_COEFF_RADIUS}" == "0" ]]; then
        if [[ "${K}" == "4" && "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
            TRUST_COEFF_RADIUS="0.10"
        elif [[ "${K}" == "4" ]]; then
            TRUST_COEFF_RADIUS="0.20"
        elif [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
            TRUST_COEFF_RADIUS="0.15"
        else
            TRUST_COEFF_RADIUS="0.30"
        fi
    fi
elif [[ "${K}" != "0" && "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v10_support_pool_nested" ]]; then
    if [[ -z "${ADAPT_SCOPE_WAS_SET}" ]]; then ADAPT_SCOPE="coeff_only"; fi
    if [[ -z "${STAGE3_POSTERIOR_POLICY_WAS_SET}" ]]; then STAGE3_POSTERIOR_POLICY="conservative_coeff_posterior"; fi
    if [[ -z "${SUPPORT_GATE_WAS_SET}" ]]; then SUPPORT_GATE="auto"; fi
    if [[ -z "${SUPPORT_GATE_MIN_DELTA_WAS_SET}" ]]; then
        if [[ "${K}" == "4" ]]; then
            SUPPORT_GATE_MIN_DELTA="1e-4"
        else
            SUPPORT_GATE_MIN_DELTA="1e-3"
        fi
    fi
    if [[ -z "${SUPPORT_GATE_ROOTZONE_TOLERANCE_WAS_SET}" ]]; then SUPPORT_GATE_ROOTZONE_TOLERANCE="0.0"; fi
    if [[ -z "${SUPPORT_LOSS_REDUCTION_WAS_SET}" ]]; then SUPPORT_LOSS_REDUCTION="cycle_balanced"; fi
    if [[ -z "${FREEZE_MONTHLY_GAIN_WAS_SET}" ]]; then FREEZE_MONTHLY_GAIN="1"; fi
    if [[ -z "${TRUST_REGION_MODE_WAS_SET}" ]]; then TRUST_REGION_MODE="groupwise"; fi
    ADAPT_SOLVER="ridge_coeff"
    if [[ -z "${RIDGE_STANDARDIZE_FEATURES_WAS_SET}" ]]; then
        RIDGE_STANDARDIZE_FEATURES="1"
    fi
    if [[ -z "${RIDGE_WEIGHTING_WAS_SET}" ]]; then
        if [[ "${K}" == "4" ]]; then
            RIDGE_WEIGHTING="cycle_variable_balanced_huber"
        else
            RIDGE_WEIGHTING="global_pixel_l2"
        fi
    fi
    if [[ -z "${RIDGE_LAMBDA_WAS_SET}" ]]; then
        if [[ "${K}" == "4" ]]; then
            RIDGE_LAMBDA="4.0"
        else
            RIDGE_LAMBDA="2.0"
        fi
    fi
    if [[ -z "${RIDGE_TRUST_REGION_RADIUS_WAS_SET}" ]]; then
        if [[ "${K}" == "4" ]]; then
            RIDGE_TRUST_REGION_RADIUS="0.10"
        else
            RIDGE_TRUST_REGION_RADIUS="0.18"
        fi
    fi
    if [[ -z "${RIDGE_CLIP_COEFF_NORM_WAS_SET}" ]]; then
        RIDGE_CLIP_COEFF_NORM="${RIDGE_TRUST_REGION_RADIUS}"
    fi
    if [[ "${TRUST_COEFF_RADIUS}" == "0.0" || "${TRUST_COEFF_RADIUS}" == "0" ]]; then
        TRUST_COEFF_RADIUS="${RIDGE_TRUST_REGION_RADIUS}"
    fi
elif [[ "${K}" != "0" && "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested" ]]; then
    if [[ -z "${ADAPT_SCOPE_WAS_SET}" ]]; then ADAPT_SCOPE="coeff_only"; fi
    if [[ -z "${STAGE3_POSTERIOR_POLICY_WAS_SET}" ]]; then STAGE3_POSTERIOR_POLICY="conservative_coeff_posterior"; fi
    if [[ -z "${SUPPORT_GATE_WAS_SET}" ]]; then SUPPORT_GATE="auto"; fi
    if [[ -z "${SUPPORT_GATE_MIN_DELTA_WAS_SET}" ]]; then
        if [[ "${K}" == "4" ]]; then
            SUPPORT_GATE_MIN_DELTA="5e-6"
        else
            SUPPORT_GATE_MIN_DELTA="2e-6"
        fi
    fi
    if [[ -z "${SUPPORT_GATE_ROOTZONE_TOLERANCE_WAS_SET}" ]]; then SUPPORT_GATE_ROOTZONE_TOLERANCE="0.0"; fi
    if [[ -z "${SUPPORT_LOSS_REDUCTION_WAS_SET}" ]]; then SUPPORT_LOSS_REDUCTION="cycle_balanced"; fi
    if [[ -z "${FREEZE_MONTHLY_GAIN_WAS_SET}" ]]; then FREEZE_MONTHLY_GAIN="1"; fi
    if [[ -z "${TRUST_REGION_MODE_WAS_SET}" ]]; then TRUST_REGION_MODE="groupwise"; fi
    ADAPT_SOLVER="ridge_coeff"
    if [[ -z "${RIDGE_STANDARDIZE_FEATURES_WAS_SET}" ]]; then
        RIDGE_STANDARDIZE_FEATURES="1"
    fi
    if [[ -z "${RIDGE_WEIGHTING_WAS_SET}" ]]; then
        if [[ "${K}" == "4" ]]; then
            RIDGE_WEIGHTING="cycle_variable_balanced_huber"
        else
            RIDGE_WEIGHTING="global_pixel_l2"
        fi
    fi
    if [[ -z "${RIDGE_LAMBDA_WAS_SET}" ]]; then
        if [[ "${K}" == "4" ]]; then
            RIDGE_LAMBDA="8.0"
        else
            RIDGE_LAMBDA="6.0"
        fi
    fi
    if [[ -z "${RIDGE_TRUST_REGION_RADIUS_WAS_SET}" ]]; then
        if [[ "${K}" == "4" ]]; then
            RIDGE_TRUST_REGION_RADIUS="0.06"
        else
            RIDGE_TRUST_REGION_RADIUS="0.10"
        fi
    fi
    if [[ -z "${RIDGE_CLIP_COEFF_NORM_WAS_SET}" ]]; then
        RIDGE_CLIP_COEFF_NORM="${RIDGE_TRUST_REGION_RADIUS}"
    fi
    if [[ "${TRUST_COEFF_RADIUS}" == "0.0" || "${TRUST_COEFF_RADIUS}" == "0" ]]; then
        TRUST_COEFF_RADIUS="${RIDGE_TRUST_REGION_RADIUS}"
    fi
elif [[ "${K}" != "0" && ( "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v6_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v7_balanced_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v8_hybrid_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v9_guarded_nested" ) ]]; then
    if [[ -z "${ADAPT_SCOPE_WAS_SET}" ]]; then ADAPT_SCOPE="coeff_only"; fi
    if [[ -z "${STAGE3_POSTERIOR_POLICY_WAS_SET}" ]]; then STAGE3_POSTERIOR_POLICY="conservative_coeff_posterior"; fi
    if [[ -z "${SUPPORT_GATE_WAS_SET}" ]]; then SUPPORT_GATE="auto"; fi
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
    if [[ -z "${SUPPORT_LOSS_REDUCTION_WAS_SET}" ]]; then SUPPORT_LOSS_REDUCTION="cycle_balanced"; fi
    if [[ -z "${FREEZE_MONTHLY_GAIN_WAS_SET}" ]]; then FREEZE_MONTHLY_GAIN="1"; fi
    if [[ -z "${TRUST_REGION_MODE_WAS_SET}" ]]; then TRUST_REGION_MODE="groupwise"; fi
    ADAPT_SOLVER="ridge_coeff"
    if [[ -z "${RIDGE_STANDARDIZE_FEATURES_WAS_SET}" ]]; then
        RIDGE_STANDARDIZE_FEATURES="1"
    fi
    if [[ "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v7_balanced_nested" && -z "${RIDGE_WEIGHTING_WAS_SET}" ]]; then
        RIDGE_WEIGHTING="cycle_variable_balanced_huber"
    fi
    if [[ ( "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v8_hybrid_nested" || "${STAGE3_KSHOT_MODE}" == "diagnostic_linearized_coeff_ridge_v9_guarded_nested" ) && -z "${RIDGE_WEIGHTING_WAS_SET}" ]]; then
        if [[ "${K}" == "4" ]]; then
            RIDGE_WEIGHTING="cycle_variable_balanced_huber"
        else
            RIDGE_WEIGHTING="global_pixel_l2"
        fi
    fi
    if [[ -z "${RIDGE_LAMBDA_WAS_SET}" ]]; then
        if [[ "${K}" == "4" && "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
            RIDGE_LAMBDA="4.0"
        elif [[ "${K}" == "4" ]]; then
            RIDGE_LAMBDA="2.0"
        elif [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
            RIDGE_LAMBDA="2.0"
        else
            RIDGE_LAMBDA="1.0"
        fi
    fi
    if [[ -z "${RIDGE_TRUST_REGION_RADIUS_WAS_SET}" ]]; then
        if [[ "${K}" == "4" && "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
            RIDGE_TRUST_REGION_RADIUS="0.10"
        elif [[ "${K}" == "4" ]]; then
            RIDGE_TRUST_REGION_RADIUS="0.18"
        elif [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
            RIDGE_TRUST_REGION_RADIUS="0.18"
        else
            RIDGE_TRUST_REGION_RADIUS="0.28"
        fi
    fi
    if [[ -z "${RIDGE_CLIP_COEFF_NORM_WAS_SET}" ]]; then
        if [[ "${K}" == "4" && "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
            RIDGE_CLIP_COEFF_NORM="0.15"
        elif [[ "${K}" == "4" ]]; then
            RIDGE_CLIP_COEFF_NORM="0.25"
        elif [[ "${DIAGNOSTIC_KSHOT_STRENGTH}" == "medium" ]]; then
            RIDGE_CLIP_COEFF_NORM="0.25"
        else
            RIDGE_CLIP_COEFF_NORM="0.40"
        fi
    fi
    if [[ "${TRUST_COEFF_RADIUS}" == "0.0" || "${TRUST_COEFF_RADIUS}" == "0" ]]; then
        TRUST_COEFF_RADIUS="${RIDGE_TRUST_REGION_RADIUS}"
    fi
fi
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
K4_REFERENCE_ARGS=()
if [[ -n "${K4_REFERENCE_CHECKPOINT:-}" ]]; then
    K4_REFERENCE_ARGS+=(--k4_reference_checkpoint "${K4_REFERENCE_CHECKPOINT}")
fi
ADAPT_MIX_RHO_ARGS=()
if [[ -n "${ADAPT_MIX_RHO:-}" ]]; then
    ADAPT_MIX_RHO_ARGS+=(--adapt_mix_rho "${ADAPT_MIX_RHO}")
fi

echo "============================================"
echo "Phase 5 HyperDA Zero/Few-Shot Generalization"
echo "  source_checkpoint=${SOURCE_CHECKPOINT}"
echo "  k4_reference_checkpoint=${K4_REFERENCE_CHECKPOINT:-<none>}"
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
echo "  adapt_scope=${ADAPT_SCOPE} (SAFE Prompt+Coeff+Gain unless conservative Stage 3 overrides) adapt_solver=${ADAPT_SOLVER} audit_identity=${AUDIT_IDENTITY}"
echo "  stage3_posterior_policy=${STAGE3_POSTERIOR_POLICY} support_gate=${SUPPORT_GATE} min_delta=${SUPPORT_GATE_MIN_DELTA} rootzone_tolerance=${SUPPORT_GATE_ROOTZONE_TOLERANCE}"
echo "  stage3_kshot_mode=${STAGE3_KSHOT_MODE}"
echo "  diagnostic_kshot_strength=${DIAGNOSTIC_KSHOT_STRENGTH}"
echo "  freeze_monthly_gain=${FREEZE_MONTHLY_GAIN}"
echo "  trust_region_mode=${TRUST_REGION_MODE} total=${TRUST_TOTAL_RADIUS} prompt=${TRUST_PROMPT_RADIUS} gain=${TRUST_GAIN_RADIUS} coeff=${TRUST_COEFF_RADIUS} spatial=${TRUST_SPATIAL_RADIUS}"
echo "  adapt_solver=${ADAPT_SOLVER} ridge_lambda=${RIDGE_LAMBDA} ridge_clip_coeff_norm=${RIDGE_CLIP_COEFF_NORM} ridge_trust_region_radius=${RIDGE_TRUST_REGION_RADIUS} ridge_max_feature_pixels=${RIDGE_MAX_FEATURE_PIXELS} ridge_standardize_features=${RIDGE_STANDARDIZE_FEATURES} ridge_weighting=${RIDGE_WEIGHTING}"
echo "  safe_policy_json=${SAFE_POLICY_JSON:-<none>} require_safe_policy_json_for_kshot=${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT}"
echo "  policy_source=source_side_episode_calibration when SAFE_POLICY_JSON is provided"
echo "  K-shot strict policy requires nonzero target_support update when require_safe_policy_json_for_kshot=1"
echo "  target_context_max_samples=${TARGET_CONTEXT_MAX_SAMPLES} (0 = full target_context)"
echo "  stage3_context_tta=${STAGE3_CONTEXT_TTA} (target_context input-side only; target_eval frozen read-only)"
echo "  context_tta_residual_scale=${CONTEXT_TTA_RESIDUAL_SCALE} clip_l2=${CONTEXT_TTA_RESIDUAL_CLIP_L2}"
echo "  support_loss_reduction=${SUPPORT_LOSS_REDUCTION}"
echo "  schedule_label=${SCHEDULE_LABEL}"
echo "  adaptation_steps=${ADAPTATION_STEPS} adapt_batch_size=${ADAPT_BATCH_SIZE} lr=${LR} weight_decay=${WEIGHT_DECAY} grad_clip=${GRAD_CLIP}"
echo "  lambda_prior=${LAMBDA_PRIOR} lambda_latent=${LAMBDA_LATENT} lambda_gain=${LAMBDA_GAIN} lambda_gain_smooth=${LAMBDA_GAIN_SMOOTH} lambda_analysis=${LAMBDA_ANALYSIS}"
echo "  output_dir=${OUTPUT_DIR:-auto}"
echo "============================================"

PYTHONPATH=. python scripts/train/train_hyperda_few_shot_adapt.py \
    --source_checkpoint "${SOURCE_CHECKPOINT}" \
    "${K4_REFERENCE_ARGS[@]}" \
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
    "${ADAPT_MIX_RHO_ARGS[@]}" \
    --adapt_scope "${ADAPT_SCOPE}" \
    --stage3_kshot_mode "${STAGE3_KSHOT_MODE}" \
    --stage3_posterior_policy "${STAGE3_POSTERIOR_POLICY}" \
    --support_gate "${SUPPORT_GATE}" \
    --support_gate_min_delta "${SUPPORT_GATE_MIN_DELTA}" \
    --support_gate_rootzone_tolerance "${SUPPORT_GATE_ROOTZONE_TOLERANCE}" \
    --adapt_solver "${ADAPT_SOLVER}" \
    --support_loss_reduction "${SUPPORT_LOSS_REDUCTION}" \
    --trust_region_mode "${TRUST_REGION_MODE}" \
    --trust_total_radius "${TRUST_TOTAL_RADIUS}" \
    --trust_prompt_radius "${TRUST_PROMPT_RADIUS}" \
    --trust_gain_radius "${TRUST_GAIN_RADIUS}" \
    --trust_coeff_radius "${TRUST_COEFF_RADIUS}" \
    --trust_spatial_radius "${TRUST_SPATIAL_RADIUS}" \
    --ridge_lambda "${RIDGE_LAMBDA}" \
    --ridge_clip_coeff_norm "${RIDGE_CLIP_COEFF_NORM}" \
    --ridge_trust_region_radius "${RIDGE_TRUST_REGION_RADIUS}" \
    --ridge_max_feature_pixels "${RIDGE_MAX_FEATURE_PIXELS}" \
    --ridge_weighting "${RIDGE_WEIGHTING}" \
    $(if [[ "${RIDGE_STANDARDIZE_FEATURES}" == "1" || "${RIDGE_STANDARDIZE_FEATURES,,}" == "true" ]]; then echo "--ridge_standardize_features"; fi) \
    --audit_identity_tolerance "${AUDIT_IDENTITY_TOLERANCE}" \
    --target_context_max_samples "${TARGET_CONTEXT_MAX_SAMPLES}" \
    --context_tta_mode "${STAGE3_CONTEXT_TTA}" \
    --context_tta_residual_scale "${CONTEXT_TTA_RESIDUAL_SCALE}" \
    --context_tta_residual_clip_l2 "${CONTEXT_TTA_RESIDUAL_CLIP_L2}" \
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
