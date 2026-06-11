# HydroDA-OOD Co-author Context

This is the compact working context for research and engineering collaboration.
Use it as the first project-specific document after `CLAUDE.md`.

## Current Paper Position

HydroDA-OOD is a leakage-controlled benchmark for neural land data
assimilation increment emulation. The model does not predict soil moisture from
scratch; it emulates the correction operator induced by an operational land DA
system:

```text
analysis_increment = analysis_soil_moisture - forecast_soil_moisture
pred_analysis      = forecast_soil_moisture + pred_increment
```

The method track is HyperDA: a hydroclimatic spatio-temporal prompt-conditioned
hypernetwork that generates target-specific lightweight DA increment operators.

## Frozen Main Protocol

The paper-facing protocol is V4.4 zero/few-shot target generalization:

```text
source_fit/train:     2015-01-01 to 2021-12-31
source_val:           2022-01-01 to 2022-12-31
target_context:       2015-01-01 to 2021-12-31 input-side only
target_support:       K labeled target DA cycles, K in {0,4,12}
target_val:           unused in main protocol
target_eval:          2023-01-01 to 2025-12-31
adaptation_setting:   zero_shot_context | few_shot_k4 | few_shot_k12
```

After source-domain training, the source backbone, prompt encoder, HyperDA
basis bank, and hypernetwork are frozen. K=0 uses only input-side
`target_context` to construct target-context monthly prompt prototypes. K=4/12
may update only lightweight target-specific variables on the K labeled support
cycles with fixed preregistered steps, while using the same monthly context
prototype policy in each forward pass. Final metrics must be computed only on
`target_eval`; `target_val` is not used for main model selection.

Few-shot target checkpoints use a source-anchor policy by default: save
`theta_init + alpha * (theta_adapt - theta_init)` for target-specific tensors
only. Current conservative preregistered defaults are K4 `alpha=0.75`,
`steps=100`, `lr=1e-3`; K12 `alpha=0.25`, `steps=80`, `lr=3e-4`. These values
must not be selected with target_val or target_eval labels.

`month` is a deployment-known month-of-year seasonal phase used only to select
the matching monthly prototype. It is not an absolute date label or a
target_eval model-selection signal.

Legacy aliases exist only for compatibility:

```text
target_train       -> legacy/internal full-target path
target_query       -> target_eval
target_full_train  -> explicit legacy opt-in only
```

## Leakage Rules

Target evaluation labels must never be used for training, prompt construction,
normalization, adaptation sample selection, early stopping, model selection,
hyperparameter selection, threshold calibration, prompt feature tuning, or
region definition.

Model selection, early stopping, and preregistered hyperparameters use
source-domain validation only. Normalization
statistics are computed from source fit data only unless a method-specific
contract explicitly states otherwise and excludes `target_eval`.

## Main Method Ladder

Paper main tables should compare:

```text
1. Forecast-only
2. Source-only backbone (`source_pooled_global_backbone`)
3. Prompt-conditioned shared backbone
4. Source-regime specialist bank (`source_regime_specialist_bank`) once
   cross-continent same-regime source regions are available
5. HyperDA K=0 zero-shot context prompt
6. HyperDA K=4 few-shot lightweight adaptation
7. HyperDA K=12 few-shot lightweight adaptation
```

Internal sanity checks such as source mean, target mean, monthly mean, ridge,
nearest-source specialists, and old K-shot curves do not belong in the main
table unless a later paper decision explicitly promotes them.

For the current US-only transition, the existing leave-one-region-out
source-only training path is the valid `source_pooled_global_backbone` analogue:
train on the five non-target US regions, validate on source_val 2022, evaluate
on the held-out target region. The old all-regions run is
`legacy_all_regions_sanity`, not OOD global. The old region-specific target
history run is `target_full_history_region_oracle`, not the new region-specific
baseline.

## Current Engineering Shape

Canonical implementation areas:

```text
hydroda/data/        protocol, dataset, leakage guard, audits
hydroda/splits/      zero/few-shot and legacy split manifests
hydroda/baselines/   forecast/source/prompt/simple sanity baselines
hydroda/models/      ResUNet, prompt encoder, adapters, HyperDA modules
hydroda/training/    shared training harness and checkpoint metadata
hydroda/evaluation/  long-table metric harness
scripts/data/        audit and split artifact builders
scripts/train/       source and prompt-conditioned training entrypoints
scripts/eval/        checkpoint and forecast-only evaluation entrypoints
```

## Codex Collaboration Protocol

Codex session discipline is defined in
`docs/CODEX_RESEARCH_OPERATING_PROTOCOL.md`. Use it as the runbook for
start-of-session context loading, experiment auditing, leakage review,
post-run summaries, and next-step planning.

Preferred main split artifact:

```text
artifacts/splits/US_loro_zero_few_shot_splits.json
```

The old artifacts `artifacts/splits/US_loro_target_train_splits.json` and
`artifacts/splits/US_loro_kdate_splits.json` are retained only for legacy
compatibility and historical reproduction.

## Near-term Research Priorities

1. Generate and freeze `US_loro_zero_few_shot_splits.json`.
2. Re-run forecast-only/source-only/prompt-conditioned baselines under
   `zero_shot_context` / `few_shot_k4` / `few_shot_k12`.
3. Verify HyperDA K=0/4/12 checkpoint metadata and no target-val construction.
4. Keep full-target results/code as legacy/internal reproduction only.

## Cleanup Policy

Use `docs/PROJECT_CLEANUP_AUDIT.md` to decide whether a file is active,
legacy, archive, generated, or disposable. Do not delete historical evidence
when a small summary or manifest is the only record of an experiment. Delete
Python caches, local long metrics tables, and W&B offline run payloads from git
tracking.
