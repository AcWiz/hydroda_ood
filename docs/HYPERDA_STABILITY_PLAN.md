# HyperDA Stability Plan

> **For future Codex sessions:** read this after the core project context and
> `docs/CODEX_RESEARCH_OPERATING_PROTOCOL.md` when auditing or changing Phase 5
> HyperDA zero/few-shot evaluation.

## Purpose

The immediate goal is to make the current HyperDA K=0/K=4/K=12 target
evaluation credible and diagnosable before changing the method. The observed
red flag is that K=12 can perform worse than K=0. Treat this first as a possible
pipeline, protocol, or unstable-adaptation issue, not as a scientific
conclusion.

This document records the current path and a minimal P0/P1 implementation plan.
Do not implement P2/P3/P4/P5 from this document until P0/P1 are planned,
reviewed, and preferably merged.

## Current Phase5 Path

Top-level entrypoint:

```bash
bash run/phase5_hyperda_zero_few_shot_eval.sh [source_checkpoint] [target_region] [seed] [cuda_device] [output_base]
```

The wrapper loops over `K_LIST`, adapts once per K, then evaluates the resulting
checkpoint on `target_eval`.

- K=0: `adaptation_setting=zero_shot_context`, `adaptation_steps=0`,
  `anchor_alpha=0.0`.
- K=4: `adaptation_setting=few_shot_k4`, default `steps=100`, `lr=1e-3`,
  `anchor_alpha=0.75`.
- K=12: `adaptation_setting=few_shot_k12`, default `steps=80`, `lr=3e-4`,
  `anchor_alpha=0.25`.

`ANCHOR_ALPHA` is source-anchor interpolation on target-specific tensors:

```text
theta_final = theta_init + alpha * (theta_adapt - theta_init)
```

The wrapper then calls `scripts/eval/evaluate_checkpoint.py` with
`--split_type target_eval` and `--predictor_type hyperda_target_adapt`.

## Files Controlling Evaluation

- `run/phase5_hyperda_zero_few_shot_eval.sh`: K loop, adaptation call,
  target-eval call, overview table.
- `run/phase5_hyperda_zero_few_shot.sh`: one-K adaptation wrapper.
- `scripts/train/train_hyperda_few_shot_adapt.py`: V4.4 zero/few-shot
  adaptation, checkpoint saving, adaptation metadata.
- `scripts/eval/evaluate_checkpoint.py`: target-eval dataset construction,
  checkpoint predictor loading, metrics output.
- `hydroda/baselines/prompt_conditioned.py`: HyperDA target-adapt predictor;
  paper-facing checkpoints must contain `target_context_prompt_state`, so eval
  inputs do not update target prompts.
- `hydroda/models/hyper_conditional_unet.py`: HyperDA model and target-specific
  modules.
- `hydroda/models/target_adaptation.py`: target prompt, adapter coefficient
  residuals, monthly gain, optional spatial refine modules.
- `hydroda/data/dataset.py`: split entry lookup, target_context input-only
  samples, target_support labeled samples, target_eval samples and masks.
- `hydroda/data/protocol.py` and `hydroda/data/leakage_guard.py`: V4.4 date and
  leakage checks.
- `hydroda/splits/manifest.py`, `hydroda/splits/kdate.py`,
  `scripts/data/build_zero_few_shot_splits.py`: split generation and date hashes.

## Target-Specific Variables

Created in `HyperAdapterConditionalResUNet` when `enable_target_adaptation=True`:

- `target_prompt`
- `target_adapter_coefficient_residual_b`
- `target_adapter_coefficient_residual_d2`
- `target_adapter_coefficient_residual_d1`
- `residual_gain`
- optional `target_spatial_refine`

Current target trainable names observed in the V4.4 path:

```text
target_prompt.latent
target_prompt.proj.weight
target_prompt.proj.bias
target_adapter_coefficient_residual_b.logit_delta
target_adapter_coefficient_residual_d2.logit_delta
target_adapter_coefficient_residual_d1.logit_delta
residual_gain.gain_delta
residual_gain.bias
```

Current update path:

1. `scripts/train/train_hyperda_few_shot_adapt.py` loads a source HyperDA
   checkpoint.
2. `model.freeze_source_prior_for_target_adaptation()` freezes the source prior
   and enables target modules.
3. `apply_target_adaptation_stage(..., stage1_epochs=0)` currently makes all
   stage-1 target modules trainable.
4. For K>0, AdamW trains target modules on `target_support` only.
5. Source-anchor interpolation is applied after training.
6. The final checkpoint stores the adapted target tensors and the frozen
   target-context prompt state.

## Metadata And Hashes

Adaptation outputs:

- `K*/adapt/checkpoints/checkpoint_final_preregistered.pt`
- `K*/adapt/metadata.json`

Evaluation outputs:

- `K*/eval/<target_region>/summary.json`
- `K*/eval/<target_region>/diagnostics.json`
- `K*/eval/<target_region>/metrics_long.csv`
- `K*/eval/<target_region>/metrics_by_region.csv`
- `K*/eval/<target_region>/metrics_by_season.csv`
- top-level `overview.csv`, `overview.json`, `overview.md`

Currently recorded or derivable:

- `protocol_freeze_id`
- `split_manifest_path`
- `split_manifest_sha256`
- `target_context_dates_hash`
- `target_support_dates_hash`
- `target_eval_dates_hash`
- `target_support_dates`
- `target_context_prompt_state` summary
- `normalization_source=source_fit_only_from_source_checkpoint`
- `model_selection_source=source_val_preregistered`
- `target_eval_usage=final_eval_only_no_training_no_selection`
- `trainable_parameter_names`
- `trainable_parameter_count`
- `target_parameter_l2_drift`

Current gaps to close in P0/P1:

- Base/source checkpoint SHA256 is not recorded.
- `target_labels_loaded` and `target_labels_used` are not explicit.
- Adaptation scope is not explicit.
- Group-wise parameter counts are not logged.
- Drift groups are too coarse: all adapter coefficient residuals are combined.
- Evaluation command/config is not saved as structured metadata.

## Current Red-Flag Evidence

Existing US-R1 seed 0 artifacts show K=12 can underperform K=0 while using the
same context/eval hashes:

```text
artifacts/runs/phase5_hyperda_zero_few_shot_eval/US-R1_s0_20260611T041317Z
```

In that run:

- K0 surface skill primary is about `0.16797`.
- K12 surface skill primary is about `0.17413`, slightly better on surface.
- K0 rootzone skill primary is about `0.23377`.
- K12 rootzone skill primary is about `0.23582`, slightly better on rootzone.

An older same-day run showed a stronger K12 degradation:

```text
artifacts/runs/phase5_hyperda_zero_few_shot_eval/US-R1_s0_20260611T021545Z
```

In that run, K12 surface/rootzone skills were substantially below K0. The
metadata format was weaker in that older run, so it is useful for symptom
triage but not enough to conclude method behavior.

## P0: Identity And Protocol Audit

Goal: prove that K=12 can be forced to behave like K=0 when no target update is
allowed. This separates pipeline/split/eval mismatch from adaptation instability.

Desired command:

```bash
AUDIT_IDENTITY=1 K_LIST="0 12" bash run/phase5_hyperda_zero_few_shot_eval.sh
```

or an equivalent new script:

```bash
bash run/audit_hyperda_k_identity.sh
```

Identity mode must ensure:

- same base checkpoint as K0;
- same target context as K0;
- same target eval split as K0;
- no trainable target update;
- no support-induced parameter change;
- final target-specific state equals zero-shot initialization;
- target-eval prediction path uses the frozen target-context prompt state, not
  target_eval prompt updates.

Implementation outline:

1. Add an audit flag to the wrapper and train script, for example
   `AUDIT_IDENTITY=1` / `--audit_identity`.
2. In audit identity for K>0, force:
   - `adaptation_steps=0`;
   - no support loader iteration;
   - `target_labels_loaded=false`;
   - `target_labels_used=false`;
   - `anchor_alpha=0.0`;
   - `adapt_scope=none`.
3. Record base checkpoint path and SHA256.
4. Record split manifest SHA256, context/support/eval date hashes, support dates,
   support cycle count, normalizer provenance, trainable names/count, and drift.
5. Add a structured audit summary under the run directory.
6. Compare K0 and K12 identity metrics and, if practical, prediction hashes.
7. Emit an explicit warning or failure when identity differs beyond a small
   tolerance.

Important expected result:

```text
K12 identity/no-update ~= K0 zero-shot
```

Any material difference means the K axis changes something outside adaptation
and must be audited before method tuning.

## P1: Adaptation Scope Controls And Drift Logging

Goal: make target adaptation diagnosable by controlling which target-specific
parameter groups can update.

User-facing option:

```bash
ADAPT_SCOPE=none
ADAPT_SCOPE=prompt_only
ADAPT_SCOPE=coeff_only
ADAPT_SCOPE=gain_only
ADAPT_SCOPE=coeff_gain
ADAPT_SCOPE=all
```

Wrapper examples:

```bash
ADAPT_SCOPE=prompt_only K_LIST="12" bash run/phase5_hyperda_zero_few_shot_eval.sh
ADAPT_SCOPE=coeff_only  K_LIST="12" bash run/phase5_hyperda_zero_few_shot_eval.sh
ADAPT_SCOPE=gain_only   K_LIST="12" bash run/phase5_hyperda_zero_few_shot_eval.sh
```

Proposed mapping:

- `none`: no target-specific trainable parameters.
- `prompt_only`: only `target_prompt.*`.
- `coeff_only`: only
  `target_adapter_coefficient_residual_{b,d2,d1}.logit_delta`.
- `gain_only`: only `residual_gain.*`.
- `coeff_gain`: coefficient residuals plus `residual_gain.*`.
- `all`: current behavior.

For `none`, avoid constructing AdamW with an empty parameter list. If
`adaptation_steps>0`, either skip training with an explicit audit/protocol note
or reject the configuration unless `AUDIT_IDENTITY=1`.

Required logs:

- `adapt_scope`
- trainable parameter names
- total trainable parameter count
- group-wise parameter counts
- group-wise L2 drift:
  - `target_prompt`
  - `adapter_coeff_bottleneck`
  - `adapter_coeff_dec2`
  - `adapter_coeff_dec1`
  - `monthly_gain`
  - `spatial_refine`
  - `other_target_specific`
- support loss before/after adaptation when labels are used
- explicit `target_labels_loaded` and `target_labels_used`

Update outputs:

- `metadata.json`
- checkpoint `config`
- top-level `overview.csv`
- top-level `overview.json`
- top-level `overview.md`

## Tests To Add For P0/P1

Focused tests are preferred over long experiments.

Suggested tests:

- Identity/scope helper test: `adapt_scope=none` leaves no trainable target
  parameters.
- Scope mapping test: each scope enables only intended parameter names.
- Drift grouping test: coefficient residual groups split into bottleneck, dec2,
  and dec1.
- Metadata test: base checkpoint hash, support dates, context hash, label flags,
  and scope are present in the JSON sidecar.
- Leakage-oriented test: identity mode does not instantiate or iterate a
  `target_support` labeled loader.

Avoid long target-eval experiments in tests. Use model/module-level tests and
small synthetic checkpoint metadata where possible.

## Deferred Work

Do not implement these until P0/P1/P2 diagnostics are complete:

- P3 monthly gain freezing or hierarchical monthly gain.
- P4 source-episodic adapter prior bank.
- P5 fallback or non-degradation gate.

These are method changes. They should only be implemented after identity and
coefficient-path diagnostics prove the current K axis and evaluation path are
protocol-consistent.

## P2: Ridge Coefficient Diagnostic Probe

`ADAPT_SOLVER=ridge_coeff` is a closed-form, support-only diagnostic solver for
the low-dimensional adapter coefficient residual path. It is intentionally not a
full-parameter adaptation method and must be run with `ADAPT_SCOPE=coeff_only`.

Current controls:

```bash
ADAPT_SOLVER=ridge_coeff
ADAPT_SCOPE=coeff_only
RIDGE_LAMBDA=1.0
RIDGE_CLIP_COEFF_NORM=1.0
RIDGE_TRUST_REGION_RADIUS=1.0
RIDGE_MAX_FEATURE_PIXELS=20000
RIDGE_STANDARDIZE_FEATURES=0
```

The implementation freezes backbone, prompt encoder, hypernetwork, basis bank,
target prompt, monthly gain, spatial refine, and all non-coefficient target
variables. It solves only
`target_adapter_coefficient_residual_{b,d2,d1}.logit_delta` using
finite-difference local design columns on `target_support`. `target_eval` is not
loaded or used for the solve, feature subsampling, regularization, or selection.

Ridge metadata records lambda, coefficient norm, delta norm, clipping flags,
support count, masked pixel/observation counts, feature pixel/observation
counts, condition number/rank, support loss before/after, group-wise drift, and
`actual_optimizer_steps=0`.

## Open Risks And Ambiguities

- Metric-only identity comparison may miss prediction-level differences that
  cancel in aggregate. Prediction hashes are stronger, but require adding a
  lightweight eval artifact.
- K12 identity will still have a different `target_support_dates_hash` by split
  definition. That must not affect predictions when `AUDIT_IDENTITY=1`.
- Current K0 metadata reports target modules as trainable even though no steps
  run. Future metadata should distinguish `trainable_by_scope` from
  `updated_by_optimizer`.
- The existing evaluator should rely on stored target-context prompt state for
  HyperDA target-adapt checkpoints. P0 should assert that path explicitly.
- Existing old artifacts have incomplete metadata; do not use them as final
  evidence without rerunning after P0/P1 instrumentation.
