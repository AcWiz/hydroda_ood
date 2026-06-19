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

The method track is HyperDA-SAFE: a hydroclimatic spatio-temporal
prompt-conditioned hypernetwork with Source-Anchored Few-Shot Operator
Refinement. It generates target-specific lightweight DA increment operators
and keeps few-shot updates close to the source-trained prior.

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
cycles with a source-side calibrated SAFE policy, while using the same monthly
context prototype policy in each forward pass. Final metrics must be computed only on
`target_eval`; `target_val` is not used for main model selection.

Few-shot target checkpoints use the HyperDA-SAFE source-anchor policy by default: save
`theta_init + alpha * (theta_adapt - theta_init)` for target-specific tensors
only. Paper-facing K=4/K=12 runs must provide `safe_policy.json` from
source-side episode calibration with
`policy_source=source_side_episode_calibration`; the policy fixes scope, steps,
learning rate, anchor alpha, and output blend `adapt_mix_rho` without reading
target_val or target_eval. Conservative diagnostic defaults update only adapter
coefficient residuals and keep monthly residual gain frozen unless a
source-side policy explicitly enables it. K-shot runs without policy or rejected
by the Stage 3 gate are saved as the K0 anchor state and must not use
paper-facing HyperDA-SAFE method IDs.

The active source-stage HyperDA entrypoint is `run/phase4_hyperda_staged.sh`.
It loads a Stage 1 source-only `source_pooled_global_backbone` checkpoint,
freezes the source base backbone/head, and trains only the prompt encoder,
FiLM, and basis-adapter generation modules with
`trainable_scope=source_base_frozen_adapter_film`. `run/phase4_hyperda.sh` is a
compatibility wrapper for this staged path, not a separate scratch method.
The current paper-facing HyperDA-SAFE source prior is
`M2_1_rank_gated_dora_stable`: stable rank-gated bounded-DoRA HyperDA prior +
SAFE refinement, using `shared_layer_aware_rank_gated_stable`,
`dora_like_gain_bounded`, temperature `2.0`, `USE_AMP=0`, and `LR=2e-4`.
The original `M2_rank_gated_dora` is
`retired_failed_exploration_not_paper_main` because of AMP skip/numerical
failure. `M2_2_source_saliency_prior` is a secondary diagnostic only until it
has source-side and target_eval evidence, and saliency/Fisher artifacts should
not change hard top-k routing unless a legacy diagnostic mode explicitly says
so. `M2_3_source_safe_residual_hyperda` is a negative/diagnostic source-safe
residual ablation: source-side safe score improved but current US-R1 K=0
target_eval RMSE degraded relative to M2.1, so it must not replace M2.1.
`M2_5a_da_aware_prompt_only` is retained as
`negative_diagnostic_non_strict_prompt_only`: saved artifacts show stronger
source_val behavior but worse US-R1 K=0 target_eval, and the run was not a
strict prompt-only contrast to M2.1 because it used `direct_hyper + rho=0.0`
while M2.1 used a source-base residual reliability-gated form. Its
DA-aware robust diagnostics were also computed in the normalized tensor domain, so TB
innovation and contrast were no longer physical raw O-F residual diagnostics.
`M2_5b_da_aware_conservative_router` is the follow-up diagnostic: it keeps the
M2.1 stable rank-gated bounded-DoRA source-base anchor, sets
`context_encoder=robust_input_side_da_diagnostics_raw`, computes DA prompt
diagnostics from raw input-side `x` while the backbone still consumes `x_norm`,
uses `zero_shot_prior_form=source_base_residual_reliability_gated`,
`source_residual_rho=1.0`, `SOURCE_RESIDUAL_GATE_INIT=0.90`, and enables
prompt-manifold reliability with strength `0.25`. It must not read target
labels, target_val, target_eval statistics, or use channel 11 as a loss,
metric, observation, or region hard mask.
`M2_4_target_context_conservative_hyperda` is a Stage 3 K=0
target-context conservative shrinkage diagnostic, not a source-stage
ablation: freeze the M2.1 prior, do no extra source fine-tune, and apply
target_context input-only reliability shrinkage calibrated from
leave-one-source-region pseudo-target episodes. It records
`target_labels_used_for_adaptation=false`,
`target_val_usage=unused_in_main_protocol`,
`target_eval_usage=final_eval_only_no_selection`, and
`target_eval_input_stats_used_for_update=false`. Hessian/Fisher/top-parameter
selection is not part of the mainline and remains a future source-side
ablation.

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
6. HyperDA-SAFE K=4 Source-Anchored Few-Shot Operator Refinement
7. HyperDA-SAFE K=12 Source-Anchored Few-Shot Operator Refinement
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

Retired failed explorations:

```text
phase6_surface_residual_ridge
phase6_bora_residual_adapter
phase7_hyperda_apo
```

Their status is `retired_failed_exploration_not_paper_main`. Existing artifacts
may be kept as internal evidence, but these methods are not a paper-main method
and should not appear as active run wrappers, configs, or protocol tests.

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
