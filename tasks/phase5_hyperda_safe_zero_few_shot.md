# Phase 5 — HyperDA-TRUST Zero-Shot Target Generalization

## Active Protocol

This is the active Phase 5 task for the V4.4 paper-facing surface:

```text
HyperDA-TRUST: zero-shot target-specific DA operator generation
adaptation_setting: zero_shot_context | few_shot_k4 | few_shot_k12
K: 0 | 4 | 12 labeled target support cycles
split_manifest: artifacts/splits/US_loro_zero_few_shot_splits.json
target_eval: 2023-2025 final offline evaluation only
```

Current paper-facing claims are K=0 only. K=4/K=12 SAFE runs remain available
as diagnostic / frozen future-extension runs; current rows rejected target
updates to the K0 anchor and must be labeled `rejected_to_k0_anchor`, not
accepted few-shot adaptation.

HyRAO, K-date selection, `target_full_train`, target_val router selection,
phase6 residual ridge/BORA adapters, and phase7 APO are legacy, diagnostic, or
retired surfaces. They must not be used as paper-main methods or active run
entrypoints.

## Required Context

```text
context/01_RESEARCH_CONTRACT.md
context/06_BASELINES_AND_MODEL_ROADMAP.md
context/10_REVIEWER_RISK_CONTROL.md
specs/protocol_v4.yaml
specs/hyperda_v4.yaml
specs/experiment_schema.yaml
checklists/no_leakage_checklist.md
```

## Method Contract

The paper-facing HyperDA Operator Generator is fixed to
`M2_1_rank_gated_dora_stable`: stable rank-gated bounded-DoRA HyperDA prior with
`shared_layer_aware_rank_gated_stable`, `dora_like_gain_bounded`, temperature
`2.0`, `USE_AMP=0`, and `LR=2e-4`. The original `M2_rank_gated_dora` is
`retired_failed_exploration_not_paper_main` due to AMP skip/numerical failure.
`M2_2_source_saliency_prior` is a secondary diagnostic only.
`M2_3_source_safe_residual_hyperda` is retained as negative/diagnostic evidence
unless same-split target_eval evidence reverses the current K=0 degradation vs
M2.1. `M2_5a_da_aware_prompt_only` is retained as
`negative_diagnostic_non_strict_prompt_only`: saved artifacts show source_val
improved but K=0 target_eval degraded, the operator form was `direct_hyper +
rho=0.0` rather than M2.1's source-base residual anchor, and DA diagnostics were
computed in the normalized tensor domain. `M2_5b_da_aware_conservative_router`
is the follow-up diagnostic: set
`context_encoder=robust_input_side_da_diagnostics_raw`, compute DA prompt
diagnostics from raw input-side `x` while the backbone consumes `x_norm`, restore
`zero_shot_prior_form=source_base_residual_reliability_gated` with
`source_residual_rho=1.0`, set `SOURCE_RESIDUAL_GATE_INIT=0.90`, and enable
prompt-manifold reliability at strength `0.25`. It must not read target labels,
target_val, target_eval statistics, or use `base_valid_mask` as a
loss/metric/obs/region hard mask. K=0 remains the plain frozen M2.1
target-context HyperDA prior without post-hoc residual shrinkage.
Hessian/Fisher/top-parameter selection stays out of the mainline and is only a
future source-side ablation.

K=0 uses the frozen source-stage HyperDA prior and target_context monthly prompt
prototypes built from 2015-2021 input-side target data only.

`M3_1_hyperda_trust_medium` is the current HyperDA-TRUST K=0 candidate. It uses
a source_fit/source_val-only trust bank, nearest-source coefficient consensus,
and surface/rootzone trust gates to regularize target-prompt operator
coefficients without target labels. Its source-side selection metric is
`source_val_dual_variable_cvar_safe_score`.

`M3_5_phys_agreement_guarded_trust` is the conservative PhysTrust follow-up
candidate. It starts from M3_1 (`trust_strength=0.50`, `top_m=4`,
`context_encoder=current_mean_std`), keeps prompt-neighbor trust routing as the
only coefficient geometry, and uses raw PhysTrust diagnostics only as a
shrink/no-harm guard. If prompt and physical nearest-source neighborhoods
agree, effective trust remains M3_1-equivalent; if they disagree or raw
physical distance is high, effective trust/source-residual strength may only
decrease. Source-side gate: reject without target_eval if source_val
dual-variable CVaR is more than 0.005 below M3_1. The guard reads only
`x`, `x_raw`, `month`, and `region_mask`; target labels, target_val, and
target_eval statistics are forbidden.
`M3_5b_phys_agreement_floor_guard` is the low-complexity redesign after the
M3_5 source-side early stop. It keeps the same M3_1 prompt-neighbor trust
routing and raw PhysTrust diagnostics, but shrinkage requires both
prompt/physical neighbor disagreement and high raw physical OOD distance
(`risk_rule=and`), and the guard multiplier is floored at `0.8`. It is still
selected only by source_val dual-variable CVaR and target_eval is final-only.
`M3_6_phys_token_operator_droppath_trust` is the next Stage 2 candidate: keep
M3_1 prompt-space trust routing and add a raw input-side physical token only as
a zero-initialized operator-coefficient residual with train-mode Operator
DropPath. First screen: warm-start from M3_1 best, freeze existing M3_1
parameters, train only `phys_context_only` for 5 epochs, and use the same
source_val dual-variable CVaR gate before any target_eval.

Earlier physics formula candidates such as
`M3_8b_phys_formula_light_guarded_trust`,
`M3_8c_phys_formula_light_operator_only_trust`, M3_9, M3_11, and M3_12 remain
historical diagnostics. They should not be treated as the current active
physics route or as permission to run US-R2..US-R6 physics screens. Their
useful surviving rule is now a hard invariant: every active Stage 2
HyperDA/HyperDA-TRUST/physics source-stage candidate must start from the Stage
1 `source_pooled_global_backbone` checkpoint via
`--init_from_source_base_checkpoint`, not from an M2_1/M3_1 prompt checkpoint.
M3_1 warm-started physics branches are diagnostics only.

`M3_12_phys_gain_basis_hypertrust` is rejected as a negative diagnostic and
must not be used as an active route or target_eval tuning basis. `M3_13` is
retained as a guarded diagnostic around a frozen M3_1 path: it can only shrink
`pred_M3_1 - source_base`, and eta=0 is an identity diagnostic.

The active physics design handoff is
`M3_16_source_only_phys_m3trust_lite`. The runnable source-stage wrapper trains
a clean physics-informed HyperDA-TRUST path from the
`source_pooled_global_backbone` source-only checkpoint, forbids M2_1/M3_1
warm-start, reuses the M3_1 architecture route, and injects physical
information only as bounded coefficient-logit modulation. M3_14 is rejected,
and M3_15 is an M3_1-warm-start diagnostic that cannot replace this mainline.
The current physics-module ablation boundary is US-R1 seed0 K=0 only,
comparing M3_1 against M3_16. US-R2 through US-R6 must not be run for current
physics ablation selection; they are locked confirmation regions after method
freeze. See
`docs/hydroda_physics_formula_knowledge_base.md` and
`docs/m3_14_source_trained_physics_hypertrust_plan.md`.

SAFE K=4/K=12 may update only lightweight target-specific variables on the fixed K
labeled target_support cycles. The source backbone, prompt encoder,
hypernetwork, and basis bank remain frozen. Diagnostic runs must use a
source-side calibrated SAFE policy:

```text
safe_policy.json
policy_source=source_side_episode_calibration
target_val_usage=unused_in_main_protocol
target_eval_usage=final_eval_only_no_selection
theta_SAFE = theta_prior + alpha_K * (theta_adapt - theta_prior)
```

Runs without this policy are diagnostic only and must be marked as such or
fallback to the K0 anchor state. Current rejected K-shot rows are K0-equivalent
fallback and do not support a few-shot improvement claim.

## Active Entry Points

```text
run/phase4_hyperda_staged.sh
run/phase5_hyperda_zero_few_shot.sh
run/phase5_hyperda_zero_few_shot_eval.sh
run/hyperda_safe_us_r1_seed0.sh  # SAFE diagnostic convenience wrapper
```

`run/phase4_hyperda.sh` is a compatibility wrapper for the staged source prior.
Diagnostic ablation wrappers may exist, but they are not paper-main entrypoints
until promoted by source-side and target_eval evidence.

## Acceptance Gates

1. Split metadata points to `US_loro_zero_few_shot_splits.json`.
2. `target_context` and `target_support` are 2015-2021 and disjoint from
   `target_eval`.
3. K=0 has no target labels and no target adaptation.
4. K=4/K=12 use exactly the fixed support budget for the run seed and region.
5. K-shot diagnostic metadata includes SAFE policy path/hash, support manifest
   hash, `anchor_alpha`, `adapt_mix_rho`, trainable parameter count, adaptation
   steps, learning rate, and support loss summary.
6. K-shot result tables expose `stage3_posterior_decision`; rows with
   `rejected_to_k0_anchor` are not accepted adaptation.
7. `target_val` is unused for target-side selection, early stopping, residual
   gain calibration, or policy selection.
8. Final reported metrics come only from `target_eval`.
