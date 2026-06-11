#!/bin/bash
# Phase 5 scaffold: deterministic episode-prior HyperDA evaluation.
#
# This public entrypoint reserves the planned interface. It delegates to the
# zero/few-shot eval wrapper with ADAPT_RECIPE=episode_prior once the prior
# artifacts are implemented.

set -euo pipefail

export ADAPT_RECIPE=episode_prior

echo "ERROR: HyperDA episode-prior evaluation is scaffolded only." >&2
echo "Implement scripts/train/build_hyperda_source_episode_bank.py and scripts/train/train_hyperda_episode_prior.py before running:" >&2
echo "  ADAPT_RECIPE=episode_prior bash run/phase5_hyperda_zero_few_shot_eval.sh ..." >&2
exit 2
