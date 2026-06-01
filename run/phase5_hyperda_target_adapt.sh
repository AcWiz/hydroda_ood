#!/bin/bash
# Phase 5: HyperDA target historical adaptation entrypoint.
#
# This wrapper preregisters the target adaptation protocol. The full optimizer
# runner is intentionally separate from source-stage training: target_eval
# labels are never used for adaptation, selection, or normalization.
#
# Protocol:
#   target_train=2015-2021
#   target_val=2022
#   target_eval=2023-2025
#   freeze_hypernetwork=true
#   trainable=target_latent,adapter_coefficient_residuals,residual_gain
#
# Usage:
#   bash run/phase5_hyperda_target_adapt.sh <source_checkpoint> US-R1 0 1

set -euo pipefail

SOURCE_CHECKPOINT="${1:?source checkpoint path is required}"
TARGET_REGION="${2:-US-R1}"
SEED="${3:-0}"
export CUDA_VISIBLE_DEVICES="${4:-1}"

cd "$(dirname "$0")/.."

echo "============================================"
echo "Phase 5 HyperDA Target Historical Adaptation"
echo "  source_checkpoint=${SOURCE_CHECKPOINT}"
echo "  target_region=${TARGET_REGION}"
echo "  seed=${SEED}"
echo "  adaptation_setting=target_full_train"
echo "  target_train=2015-2021"
echo "  target_val=2022"
echo "  target_eval=2023-2025"
echo "  freeze_hypernetwork=true"
echo "  trainable=target_latent,adapter_coefficient_residuals,residual_gain"
echo "  target_eval labels are never used for adaptation"
echo "  split_artifact=artifacts/splits/US_loro_target_train_splits.json"
echo "============================================"

PYTHONPATH=. python - <<'PY'
raise SystemExit(
    "HyperDA target adaptation modules are implemented, but the full dataset "
    "optimizer runner is not wired yet. Use this wrapper as the preregistered "
    "protocol contract until scripts/train/train_hyperda_target_adapt.py is added."
)
PY
