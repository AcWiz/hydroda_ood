#!/bin/bash
# Compatibility wrapper for legacy/internal target-adapt inference.
#
# Historical implementation: run/legacy/phase5_hyperda_target_adapt_inference.sh
# Contract markers retained for tests/docs:
#   checkpoint_best_target_val_surface_wrmse.pt
#   checkpoint_best_target_val_loss.pt
#   SURFACE_CHECKPOINT
#   --predictor_type hyperda_target_adapt
#   --split_type target_eval

set -euo pipefail

exec bash "$(dirname "$0")/legacy/phase5_hyperda_target_adapt_inference.sh" "$@"
