# M3_14 Source-Trained Physics-Informed HyperDA-TRUST Plan

This document freezes the design intent for
`M3_14_source_trained_phys_formula_gain_hypertrust`. It is the local design
plan, knowledge-base handoff, and implementation contract for the runnable
source-trained candidate.

## Decision

M3_14 is not a post-processing guard around M3_13 and not a warm-started
M3_1/M2_1 checkpoint edit. It is a clean source-stage retraining of the
HyperDA-TRUST design plus a bounded physical coefficient-modulation module.

Training start:

```text
checkpoint_start = source_pooled_global_backbone
allowed init     = --init_from_source_base_checkpoint only
forbidden init   = RESUME_FROM_M3_1_BEST
forbidden init   = --init_from_prompt_checkpoint
forbidden init   = M2_1 or M3_1 checkpoint warm-start
```

Design anchor:

```text
M3_1_hyperda_trust_medium architecture route
context_encoder = current_mean_std
hyper_coeff_generator = shared_layer_aware_rank_gated_stable
hyper_adapter_param_style = dora_like_gain_bounded
trust_strength = 0.50
source_neighbor_top_m = 4
trust_routing_geometry = prompt_embedding
source_trust_query_mode = prompt_embedding
```

M3_14 copies the M3_1 architecture decisions, not the M3_1 learned checkpoint.

## Current Experiment Boundary

Current physics-module development is restricted to:

```text
target_region = US-R1
seed = 0
K = 0
comparison = M3_1_hyperda_trust_medium
             vs M3_14_source_trained_phys_formula_gain_hypertrust
```

Do not run `US-R2` to `US-R6` for physics-module ablations at this stage.
Those regions are deferred confirmation regions after the method is frozen.
When the method is frozen, expand once to `US-R2` through `US-R6` without
changing formulas, hyperparameters, selection rules, or acceptance gates.

Current US-R1 `target_eval` is development-ablation evidence only. It is not a
paper-eligible screen for choosing more physics variants.

## Physical Module

The physical module builds `z_phys` from raw input-side formula features:

```text
d_H, d_V
m_H, m_V
gamma
rho_H, rho_V
B_pol
B_temp
B_vert
source_fit_gain_prior_summaries
finite_input_coverage
base_valid_mask_fraction_diagnostic_only
```

Allowed inputs:

```text
x_raw or raw x
month
region_mask from frozen artifact
source_fit/source_val metadata
```

Forbidden inputs:

```text
target labels
target_context labels
target_support labels for K=0
target_val
target_eval statistics
target_full_train records
channel_11_hard_mask_usage
```

The sign convention and source formulas are documented in
`docs/hydroda_physics_formula_knowledge_base.md`.

## Injection Contract

M3_14 can modify only HyperDA coefficient logits and source-fit regularization:

```text
logits_l = logits_l_M3design
           + sigmoid(g_l) * 0.05 * DeltaLogits_l(z_prompt, z_phys)
```

Required properties:

- `DeltaLogits_l` is bounded;
- the scale is fixed at `0.05` unless a future source-only preregistered plan
  changes it before any target_eval run;
- the branch must be zero-initialized or otherwise metadata-proven to preserve
  the M3_1 design path at step 0;
- no physical `q_surface` or `q_rootzone` is added directly to the final
  increment output;
- any consistency guard is diagnostic by default and, if enabled, can only
  shrink variable gates.

Disallowed final form:

```text
pred = pred_M3 + eta * q_phys
pred = pred_M3 + eta * clip(q_phys - pred_M3)
pred = source_base + guard * (pred_M3 - source_base)   # M3_13-style guard
```

Allowed M3_14 final form:

```text
pred_increment = f_{theta0, zeta_star(z_prompt, z_phys)}(x)
```

where `zeta_star` is produced by the M3_1-style trust-routed coefficient path
with bounded physical coefficient-logit modulation.

## Source-Fit Regularization

During source_fit training, M3_14 uses a weak physical consistency regularizer
with default wrapper weight:

```text
--hyper_phys_consistency_regularization_weight 0.01
```

Purpose:

- encourage high-confidence TB innovation signs to avoid contradicting predicted
  increment direction;
- keep the constraint weak enough that source labels remain the primary
  training signal;
- log physical agreement, conflict coverage, and channel-11 diagnostic coverage.

Selection remains source-side:

```text
checkpoint_selection = source_val only
selection_metric = source_val_dual_variable_cvar_safe_score
target_eval_usage = final_eval_only_no_selection
```

## Acceptance Gate

The US-R1 development ablation can freeze M3_14 only if:

```text
source_val not weaker than M3_1
target_eval improves at least one variable RMSE
other variable RMSE degrades by <= 0.2%
```

If RMSE does not improve but diagnostics are coherent, M3_14 remains an
interpretability ablation and must not replace M3_1.

If any target_eval observation leads to changing a formula, hyperparameter,
selection rule, or acceptance rule, the changed variant is exploratory and must
restart source-side selection before any confirmation-region claim.

## Deferred Confirmation Policy

After freeze, run a single locked confirmation sweep:

```text
regions = US-R2, US-R3, US-R4, US-R5, US-R6
seed = 0 unless a later preregistered multi-seed confirmation plan says otherwise
K = 0
method = frozen M3_14
baseline = same-split M3_1_hyperda_trust_medium
```

Forbidden during confirmation:

- formula edits;
- coefficient scale edits;
- new physical features;
- different selection metric;
- target_eval-tuned thresholds;
- region-specific acceptance gates.

## Implemented Entry Points

Current source-stage wrapper:

```text
ABLATION_ID=M3_14_source_trained_phys_formula_gain_hypertrust \
  bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 0
```

The wrapper:

- starts from `--init_from_source_base_checkpoint`;
- refuses `RESUME_FROM_M3_1_BEST=1`;
- refuses `HYPER_PHYS_GAIN_BASIS_RESIDUAL=1`;
- refuses current `US-R2` through `US-R6` runs unless an explicit frozen
  confirmation flag is set after method freeze;
- logs `checkpoint_start=source_pooled_global_backbone`,
  `current_ablation_policy=US-R1_seed0_K0_only`, and
  `final_output_residual_allowed=false`.

Implemented formula source:

```text
phys_context_source = raw_input_side_formula_gain
phys_formula_source = raw_input_side_formula_gain
phys_formula_schema = m3_14_raw_input_side_formula_gain_v1
```

## Implemented Tests

Wrapper dry-run:

- `tests/test_m3_14_source_trained_phys_formula_gain_hypertrust.py`
  verifies warm-start refusal, deferred-region refusal, final-output residual
  refusal, and clean-source dry-run flags.

Formula unit tests:

- `tests/test_phys_trust_diagnostics.py` verifies `d_p > 0` dry-direction
  encoding, `m_p=-tanh(d_p)` wet-support encoding, bounded features, source-fit
  gain-prior summaries, and channel-11 diagnostic-only behavior.

Structure tests:

- `tests/test_hyperda_model.py` verifies zero-initialized formula-gain
  coefficient-logit modulation preserves the M3_1 design path at step 0 and
  does not instantiate the final-output physics residual branch.

Leakage tests:

- gain bank accepts `source_fit` only;
- selection accepts `source_val` only;
- target-side roles are rejected:
  `target_context`, `target_val`, `target_eval`, and `target_full_train`.

Documentation tests:

- experiment cards report formula schema, source bank hash, source split roles,
  coefficient modulation scale, guard mode, target_eval no-selection status, and
  US-R1-only development boundary.
