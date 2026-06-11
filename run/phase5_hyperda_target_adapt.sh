#!/bin/bash
# Compatibility wrapper for the legacy/internal full-target adaptation path.
#
# Paper-facing main entrypoint: run/phase5_hyperda_zero_few_shot.sh
# Historical implementation: run/legacy/phase5_hyperda_target_adapt.sh
#
# Contract markers retained for tests/docs:
#   adaptation_setting=target_full_train
#   status=legacy_internal_not_paper_main
#   target_train=2015-2021
#   target_val=2022
#   target_eval=2023-2025
#   freeze_hypernetwork=true
#   trainable=target_latent,adapter_coefficient_residuals,residual_gain,target_spatial_refine
#   target_eval labels are never used for adaptation
#   scripts/train/train_hyperda_target_adapt.py
#
# Forwarded knobs retained by the legacy script:
#   RESUME_FROM / --resume_from
#   ENABLE_TARGET_SPATIAL_REFINE / --enable_target_spatial_refine
#   ENABLE_TARGET_SPATIAL_REFINE="${ENABLE_TARGET_SPATIAL_REFINE:-1}"
#   TARGET_SPATIAL_REFINE_ROOTZONE="${TARGET_SPATIAL_REFINE_ROOTZONE:-1}"
#   TARGET_SPATIAL_REFINE_INPUT="${TARGET_SPATIAL_REFINE_INPUT:-normalized}"
#   --target_spatial_refine_input
#   TARGET_SELECTION_METRIC / --target_selection_metric
#   --surface_weight

set -euo pipefail

exec bash "$(dirname "$0")/legacy/phase5_hyperda_target_adapt.sh" "$@"
