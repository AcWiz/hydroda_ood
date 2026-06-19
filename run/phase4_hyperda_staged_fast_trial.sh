#!/bin/bash
# Fast single-GPU engineering trial for staged HyperDA source-stage training.
#
# This wrapper keeps the staged HyperDA protocol semantics from
# phase4_hyperda_staged.sh: one shared HyperDA model, source_fit/source_val
# source-region episodes, and no target labels. It only changes engineering
# defaults and writes to a separate run root.

set -euo pipefail

cd "$(dirname "$0")/.."

export OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/runs/phase4_hyperda_staged_fast_trial}"
export RUN_NAME="${RUN_NAME:-phase4_hyperda_staged_fast_trial_${2:-US-R1}_s${3:-0}_$(date +%Y%m%d_%H%M%S)}"
export DATASET_BACKEND="${DATASET_BACKEND:-auto}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export MAX_YEAR_CACHE_ENTRIES="${MAX_YEAR_CACHE_ENTRIES:-2}"
export TENSOR_CACHE_LOAD_MODE="${TENSOR_CACHE_LOAD_MODE:-mmap}"
export TRAIN_BATCH_SAMPLER="${TRAIN_BATCH_SAMPLER:-source_region_year_grouped}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export ACCUM_STEPS="${ACCUM_STEPS:-2}"
export EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-5}"
export LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-200}"
export SOURCE_PROTOTYPE_CACHE_MODE="${SOURCE_PROTOTYPE_CACHE_MODE:-read_write}"
export SOURCE_PROTOTYPE_CACHE_DIR="${SOURCE_PROTOTYPE_CACHE_DIR:-artifacts/cache/source_context_monthly_prototypes}"

exec bash run/phase4_hyperda_staged.sh "$@"
