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

The paper-facing protocol is V4.3 historical target-training-period adaptation:

```text
source_fit/train:     2015-01-01 to 2021-12-31
source_val:           2022-01-01 to 2022-12-31
target_train:         2015-01-01 to 2021-12-31
target_val:           2022-01-01 to 2022-12-31
target_eval:          2023-01-01 to 2025-12-31
adaptation_setting:   target_full_train
```

Main experiments may use all labeled target-domain samples in `target_train`
to build a target-specific prompt, adapter, LoRA state, generated parameter
set, or refinement state. Final metrics must be computed only on
`target_eval`.

Legacy aliases exist only for compatibility:

```text
target_context -> target_train
target_support -> target_train only in legacy few-shot code
target_query   -> target_eval
K              -> legacy few-shot ablation variable
```

## Leakage Rules

Target evaluation labels must never be used for training, prompt construction,
normalization, adaptation sample selection, early stopping, model selection,
hyperparameter selection, threshold calibration, prompt feature tuning, or
region definition.

Model selection and early stopping use source-domain validation only unless an
experiment is explicitly preregistered as a secondary ablation. Normalization
statistics are computed from source fit data only unless a method-specific
contract explicitly states otherwise and excludes `target_eval`.

## Main Method Ladder

Paper main tables should compare:

```text
1. Forecast-only
2. Source-only backbone
3. Prompt-conditioned shared backbone
4. Adapter tuning on full target_train
5. LoRA tuning on full target_train
6. HyperDA generated operator from full target_train
7. HyperDA generated operator plus full target_train calibration
8. HyperDA-Refine with full target_train
```

Internal sanity checks such as source mean, target mean, monthly mean, ridge,
nearest-source specialists, and old K-shot curves do not belong in the main
table unless a later paper decision explicitly promotes them.

## Current Engineering Shape

Canonical implementation areas:

```text
hydroda/data/        protocol, dataset, leakage guard, audits
hydroda/splits/      target_train and legacy K-date split manifests
hydroda/baselines/   forecast/source/prompt/simple sanity baselines
hydroda/models/      ResUNet, prompt encoder, adapters, HyperDA modules
hydroda/training/    shared training harness and checkpoint metadata
hydroda/evaluation/  long-table metric harness
scripts/data/        audit and split artifact builders
scripts/train/       source and prompt-conditioned training entrypoints
scripts/eval/        checkpoint and forecast-only evaluation entrypoints
```

Preferred main split artifact:

```text
artifacts/splits/US_loro_target_train_splits.json
```

The old artifact `artifacts/splits/US_loro_kdate_splits.json` is retained only
for legacy compatibility and historical reproduction.

## Near-term Research Priorities

1. Generate and freeze `US_loro_target_train_splits.json`.
2. Re-run forecast-only/source-only/prompt-conditioned baselines under
   `adaptation_setting=target_full_train`.
3. Implement and verify adapter and LoRA full-target-train adaptation.
4. Build HyperDA generated-operator and refine paths with the same leakage
   guards and metadata hashes.
5. Produce reviewer-facing tables that separate main protocol results from
   secondary K-shot ablations.

## Cleanup Policy

Use `docs/PROJECT_CLEANUP_AUDIT.md` to decide whether a file is active,
legacy, archive, generated, or disposable. Do not delete historical evidence
when a small summary or manifest is the only record of an experiment. Delete
Python caches, local long metrics tables, and W&B offline run payloads from git
tracking.
