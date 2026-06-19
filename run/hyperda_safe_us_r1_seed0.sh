#!/bin/bash
# HyperDA-SAFE paper-facing US-R1 seed0 entrypoint.
# Method: Source-Anchored Few-Shot Operator Refinement.
#
# Requires a source-side episode calibration export with
# policy_source=source_side_episode_calibration:
#   SAFE_POLICY_JSON=/path/to/safe_policy.json bash run/hyperda_safe_us_r1_seed0.sh [source_checkpoint] [cuda_device] [output_base]

set -euo pipefail

SOURCE_CHECKPOINT="${1:-${SOURCE_CHECKPOINT:-}}"
TARGET_REGION="${TARGET_REGION:-US-R1}"
SEED="${SEED:-0}"
CUDA_DEVICE="${2:-${CUDA_VISIBLE_DEVICES:-0}}"
OUTPUT_BASE="${3:-artifacts/runs/hyperda_safe_us_r1_seed0/$(date -u +%Y%m%dT%H%M%SZ)}"
SAFE_POLICY_JSON="${SAFE_POLICY_JSON:-}"
REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT:-1}"
K_LIST="${K_LIST:-0 4 12}"

cd "$(dirname "$0")/.."

if [[ -z "${SAFE_POLICY_JSON}" ]]; then
    echo "ERROR: HyperDA-SAFE paper-facing K4/K12 requires SAFE_POLICY_JSON." >&2
    echo "Provide source-side episode calibration export, e.g.:" >&2
    echo "  SAFE_POLICY_JSON=artifacts/runs/<source_safe_calibration>/safe_policy.json bash run/hyperda_safe_us_r1_seed0.sh" >&2
    exit 2
fi

export SAFE_POLICY_JSON="${SAFE_POLICY_JSON}"
export REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT="${REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT}"
export MODE="${MODE:-full}"
export STAGE3_STRICT_PAPER_POLICY="${STAGE3_STRICT_PAPER_POLICY:-1}"
export STAGE3_POSTERIOR_POLICY="${STAGE3_POSTERIOR_POLICY:-conservative_coeff_posterior}"
export SUPPORT_GATE="${SUPPORT_GATE:-auto}"
export TARGET_REGION="${TARGET_REGION}"
export SEED="${SEED}"
export K_LIST="${K_LIST}"

echo "HyperDA-SAFE: Source-Anchored Few-Shot Operator Refinement"
echo "  target_region=${TARGET_REGION}"
echo "  seed=${SEED}"
echo "  K_LIST=${K_LIST}"
echo "  target_context=2015-2021 input-side only"
echo "  target_support=K labeled cycles"
echo "  target_val=unused_in_main_protocol"
echo "  target_eval=2023-2025 final offline evaluation"
echo "  model_selection_source=source_val_preregistered"
echo "  ADAPT_RECIPE=source_anchor"
echo "  ANCHOR_ALPHA_K4=0.75"
echo "  ANCHOR_ALPHA_K12=0.25"

bash run/phase5_hyperda_zero_few_shot_eval.sh \
    "${SOURCE_CHECKPOINT}" \
    "${TARGET_REGION}" \
    "${SEED}" \
    "${CUDA_DEVICE}" \
    "${OUTPUT_BASE}"
