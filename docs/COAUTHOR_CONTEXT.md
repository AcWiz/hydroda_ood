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

The current method track is HyperDA-TRUST: zero-shot target-specific DA
increment operator generation. HyperDA uses a source-trained
prompt-conditioned hypernetwork to generate lightweight target-specific DA
increment operators from input-side target context. HyperDA-TRUST adds
source-manifold trust routing to regularize generated operator coefficients
toward nearest source-neighborhood consensus without target labels.

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
`target_context` to construct target-context monthly prompt prototypes.
The current paper claim prioritizes K=0 zero-shot results. K=4/12 remain in the
experiment system as frozen SAFE diagnostics / future extension; current
K-shot results rejected updates to the K0 anchor, so they must not be described
as accepted few-shot adaptation. Final metrics must be computed only on
`target_eval`; `target_val` is not used for main model selection.

SAFE few-shot checkpoints, when run for diagnostics, use the source-anchor
policy: save `theta_init + alpha * (theta_adapt - theta_init)` for
target-specific tensors only. K=4/K=12 runs must provide `safe_policy.json`
from source-side episode calibration with
`policy_source=source_side_episode_calibration`; the policy fixes scope, steps,
learning rate, anchor alpha, and output blend `adapt_mix_rho` without reading
target_val or target_eval. Current K-shot rows that are
`rejected_to_k0_anchor` are K0-equivalent fallback, not few-shot improvement.

The active source-stage HyperDA entrypoint is `run/phase4_hyperda_staged.sh`.
It loads a Stage 1 source-only `source_pooled_global_backbone` checkpoint,
freezes the source base backbone/head, and trains only the prompt encoder,
FiLM, and basis-adapter generation modules with
`trainable_scope=source_base_frozen_adapter_film`. `run/phase4_hyperda.sh` is a
compatibility wrapper for this staged path, not a separate scratch method.
The current HyperDA Operator Generator is `M2_1_rank_gated_dora_stable`: stable
rank-gated bounded-DoRA source-trained HyperDA prior, using
`shared_layer_aware_rank_gated_stable`,
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
`M2_6_source_manifold_guarded_prior` is a diagnostic source-manifold guarded
prior, not a replacement for M2.1 unless gates pass. It returns to the M2.1
`current_mean_std` prompt route and M2.1 stable rank-gated bounded-DoRA settings,
uses `source_base_residual_reliability_gated` with `source_residual_rho=1.0` and
`SOURCE_RESIDUAL_GATE_INIT=0.95`, and adds only a source-fit/source-val
calibrated `source_manifold_distance_bounded` guard that shrinks adapter/HyperDA
residual branches without changing `source_base_forward(x)`. It is not M2.4
post-hoc target-context shrinkage and it must not tune shrinkage from target_eval.
The K=0 mainline remains the frozen M2.1 target-context HyperDA prior without
post-hoc residual shrinkage. Hessian, Fisher, top-parameter selection, and
other extra target-context guards are not part of the mainline and remain
future source-side ablations only if explicitly promoted later.
`HyperDA-TRUST` / M3 is the source-manifold trust-routing module:
Source-Manifold Trust-Routed Operator Generation. Unlike M2.6's scalar
source-manifold guard, it builds a source_fit/source_val-only trust bank of
source prompt embeddings, nearest-source distance quantiles, and
source-neighborhood coefficient consensus, then blends target-prompt operator
coefficients with nearest source consensus and records separate
surface/rootzone trust gates. Its source-side selection metric is
`source_val_dual_variable_cvar_safe_score`. M3 remains diagnostic/candidate
until source gates pass across enough regions; only preselected top candidates
receive final K=0 `target_eval`. Any tuning after seeing target_eval is
exploratory only. As of the 2026-06-21 US-R1 seed0 preregistered matrix,
`M3_1_hyperda_trust_medium` remains the current K=0 paper-facing
HyperDA-TRUST candidate; `M3_1a_trust_medium_dualalpha`,
`M3_1b_trust_mid_high`, and `M3_1c_trust_medium_local` are
`negative_or_neutral_preregistered_ablation` and must not replace M3_1.
`M3_1d_trust_medium_broad` should not be run as strict evidence unless it is
explicitly downgraded to exploratory. Evidence:
`reports/experiments/M3_1_plus_us_r1_seed0_ablation_decision_20260621.md`.
M3_4 raw/blended PhysTrust query-space routing is a negative diagnostic, not a
direction to continue. The follow-up `M3_5_phys_agreement_guarded_trust` keeps
M3_1 prompt-space trust routing as the main geometry (`trust_strength=0.50`,
`top_m=4`, `context_encoder=current_mean_std`) and uses raw PhysTrust only as a
source-side calibrated shrink/no-harm agreement guard. Agreement preserves M3_1
effective trust; disagreement or high raw physical distance can only shrink
trust/source-residual strength. M3_5 is rejected before target_eval if
source_val dual-variable CVaR falls more than 0.005 below M3_1. It reads only
input-side `x/x_raw/month/region_mask`; target labels, target_val, and
target_eval statistics remain forbidden.
The first M3_5 US-R1 seed0 source-side run was early-stopped because the guard
collapsed effective trust/residual strength instead of acting as a no-harm
check. The next low-complexity candidate is
`M3_5b_phys_agreement_floor_guard`: same M3_1 prompt-space trust routing and raw
diagnostics, but shrinkage requires both neighbor disagreement and high raw
physical OOD distance (`risk_rule=and`) and the guard multiplier is floored at
`0.8`. M3_5b remains source-gated and target_eval-final-only.
The next candidate is `M3_6_phys_token_operator_droppath_trust`: a Stage 2
supernetwork change, not Stage 3 calibration. It keeps M3_1 prompt-space trust
routing (`trust_strength=0.50`, `top_m=4`, `context_encoder=current_mean_std`)
and adds only a raw input-side physical context token that perturbs adapter
operator coefficient logits through a zero-initialized bounded residual branch.
Operator DropPath applies only to this new perturbation during training
(`p=0.10`) and is disabled at eval. The first screen warm-starts from the M3_1
best source checkpoint, freezes the existing M3_1 path, trains only the new
physical branch (`trainable_scope=phys_context_only`) for 5 epochs, and uses the
same source_val dual-variable CVaR gate. It reads only
`x/x_raw/month/region_mask`; target labels, target_val, and target_eval
statistics remain forbidden.
`M3_12_phys_gain_basis_hypertrust` is a
`rejected_negative_diagnostic`: US-R1 source_val safe score collapsed far below
M3_1 and the seen US-R1 target_eval RMSE degraded, so it must not be used as an
active route or tuning basis. The replacement diagnostic is
`M3_13_phys_gain_guarded_hypertrust`: start from the M3_1 best checkpoint,
freeze the full M3_1 path, build a source_fit-only physics-gain bank, select
eta only on source_val from `{0, 0.02, 0.05, 0.10}`, and only shrink
`pred_M3_1 - source_base` with `guard in [0.90, 1.00]`. If no positive eta
passes the source gate, eta=0 is an identity diagnostic and target_eval is
refused. M3_13 is now a guarded diagnostic, not the active physics route.
`M3_14_source_trained_phys_formula_gain_hypertrust` is now a
`rejected_negative_diagnostic`: its best source_val
`dual_variable_cvar_score=0.44455` did not protect the M3_1 anchor
(`0.446573`) and the seen US-R1 target_eval RMSE degraded. Do not run US-R2
through US-R6 for M3_14 and do not use its US-R1 target_eval as a tuning basis.
`M3_15_m31_anchored_source_safe_phys_coeff_delta` is retained as an
M3_1-warm-start diagnostic only; it is not the active Stage 2 physics mainline.
The active physics candidate is `M3_16_source_only_phys_m3trust_lite`: Stage 2
always starts from the Stage 1 `source_pooled_global_backbone` source-only
checkpoint via `--init_from_source_base_checkpoint`, never from M2_1/M3_1
prompt checkpoints. M3_16 keeps the M3_1 route
(`trust_strength=0.50`, `top_m=4`, `context_encoder=current_mean_std`,
prompt-space trust routing), uses
`trainable_scope=source_base_frozen_adapter_film`, and adds only a lightweight
raw input-side physics token to operator coefficient logits
(`phys_delta_scale=0.03`, `phys_gate_init=0.25`, Operator DropPath `p=0.10`).
It has no final-output `q_surface/q_rootzone` residual and no post-hoc eta
interpolation. Current screening is US-R1 seed0 K=0 only; US-R2 through US-R6
remain locked until M3_16 is frozen.

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
5. HyperDA Operator Generator K=0 zero-shot context prompt (`M2_1`)
6. HyperDA-TRUST K=0 source-manifold trust routing (`M3_1`)
```

Internal sanity checks such as source mean, target mean, monthly mean, ridge,
nearest-source specialists, and K-shot SAFE diagnostics do not belong in the
main table unless a later paper decision explicitly promotes accepted few-shot
updates.

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
3. Verify HyperDA K=0 and HyperDA-TRUST K=0 checkpoint metadata and no
   target-val construction; keep K=4/K=12 SAFE rows diagnostic unless accepted
   updates are demonstrated.
4. Keep full-target results/code as legacy/internal reproduction only.

## Cleanup Policy

Use `docs/PROJECT_CLEANUP_AUDIT.md` to decide whether a file is active,
legacy, archive, generated, or disposable. Do not delete historical evidence
when a small summary or manifest is the only record of an experiment. Delete
Python caches, local long metrics tables, and W&B offline run payloads from git
tracking.
