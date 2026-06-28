#!/bin/bash
# SAFE diagnostic US-R1 seed0 entrypoint.
# Method: Source-Anchored Few-Shot Operator Refinement.
# Current K-shot rows must be interpreted through stage3_posterior_decision;
# rejected_to_k0_anchor is K0-equivalent fallback, not few-shot improvement.
#
# Uses a source-side episode calibration export with
# policy_source=source_side_episode_calibration. If SAFE_POLICY_JSON is omitted,
# the delegated wrapper reuses a cached safe_policy.json or can build one when
# AUTO_GENERATE_SAFE_POLICY=1:
#   AUTO_GENERATE_SAFE_POLICY=1 bash run/hyperda_safe_us_r1_seed0.sh [source_checkpoint] [cuda_device] [output_base]

set -euo pipefail

SOURCE_CHECKPOINT="${1:-${SOURCE_CHECKPOINT:-}}"
TARGET_REGION="${TARGET_REGION:-US-R1}"
SEED="${SEED:-0}"
CUDA_DEVICE="${2:-${CUDA_VISIBLE_DEVICES:-0}}"
OUTPUT_BASE="${3:-artifacts/runs/hyperda_safe_us_r1_seed0/$(date -u +%Y%m%dT%H%M%SZ)}"
SAFE_POLICY_JSON="${SAFE_POLICY_JSON:-}"
SAFE_POLICY_CACHE_ROOT="${SAFE_POLICY_CACHE_ROOT:-artifacts/runs/stage3_source_safe_policy_cache}"
AUTO_GENERATE_SAFE_POLICY="${AUTO_GENERATE_SAFE_POLICY:-0}"
REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT:-1}"
STAGE3_KSHOT_MODE="${STAGE3_KSHOT_MODE:-paper_safe}"
K_LIST="${K_LIST:-0 4 12}"

cd "$(dirname "$0")/.."

export SAFE_POLICY_JSON="${SAFE_POLICY_JSON}"
export SAFE_POLICY_CACHE_ROOT="${SAFE_POLICY_CACHE_ROOT}"
export AUTO_GENERATE_SAFE_POLICY="${AUTO_GENERATE_SAFE_POLICY}"
export REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT}"
export STAGE3_KSHOT_MODE="${STAGE3_KSHOT_MODE}"
export MODE="${MODE:-full}"
export STAGE3_STRICT_PAPER_POLICY="${STAGE3_STRICT_PAPER_POLICY:-1}"
export STAGE3_POSTERIOR_POLICY="${STAGE3_POSTERIOR_POLICY:-conservative_coeff_posterior}"
export SUPPORT_GATE="${SUPPORT_GATE:-auto}"
export TARGET_REGION="${TARGET_REGION}"
export SEED="${SEED}"
export K_LIST="${K_LIST}"

echo "SAFE diagnostic: Source-Anchored Few-Shot Operator Refinement"
echo "  target_region=${TARGET_REGION}"
echo "  seed=${SEED}"
echo "  K_LIST=${K_LIST}"
echo "  stage3_kshot_mode=${STAGE3_KSHOT_MODE}"
echo "  safe_policy_json=${SAFE_POLICY_JSON:-<auto-cache>}"
echo "  safe_policy_cache_root=${SAFE_POLICY_CACHE_ROOT}"
echo "  auto_generate_safe_policy=${AUTO_GENERATE_SAFE_POLICY}"
echo "  target_context=2015-2021 input-side only"
echo "  target_support=K labeled cycles"
echo "  target_val=unused_in_main_protocol"
echo "  target_eval=2023-2025 final offline evaluation"
echo "  model_selection_source=source_val_preregistered"
echo "  kshot_claim_status=diagnostic_requires_stage3_posterior_decision"
echo "  rejected_to_k0_anchor=K0_equivalent_fallback_not_few_shot_improvement"
echo "  ADAPT_RECIPE=source_anchor"
echo "  ANCHOR_ALPHA_K4=0.75"
echo "  ANCHOR_ALPHA_K12=0.25"

bash run/phase5_hyperda_zero_few_shot_eval.sh \
    "${SOURCE_CHECKPOINT}" \
    "${TARGET_REGION}" \
    "${SEED}" \
    "${CUDA_DEVICE}" \
    "${OUTPUT_BASE}"
