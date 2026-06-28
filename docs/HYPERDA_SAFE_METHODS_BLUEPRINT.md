# HyperDA-TRUST Methods Blueprint

This document supersedes the earlier HyperDA-SAFE main-method draft. The
current paper narrative is leakage-controlled zero-shot target-specific DA
increment operator generation. SAFE remains a diagnostic / frozen future
extension until K-shot runs produce accepted updates rather than
`rejected_to_k0_anchor` fallbacks.

## Recommended Narrative

Write the Methods section as a protocol-method pair:

1. Define neural land data-assimilation increment emulation.
2. Define the HydroDA-OOD information boundary.
3. Build target-context monthly prompt prototypes from input-side target
   history only.
4. Generate lightweight target-specific operator parameters with the HyperDA
   Operator Generator (`M2_1`).
5. Regularize target coefficients with Source-Manifold Trust Routing
   (`M3_1`, HyperDA-TRUST).
6. State training, source-side selection, normalization, and final target_eval
   rules.
7. Describe SAFE K=4/K=12 only as diagnostic/future-extension material.

## Task Definition

The task is neural land DA analysis-increment emulation:

```text
Delta s_t^R = s_{t,analysis}^R - s_{t,forecast}^R
hat{s}_{t,analysis}^R = s_{t,forecast}^R + hat{Delta s}_t^R
```

The model emulates the correction operator induced by a reference land DA
system. Do not describe the task as true soil-moisture prediction from scratch.

## Information Boundary

Use the V4.4 protocol:

```text
source_fit:     2015-2021 source domains
source_val:     2022 source domains only
target_context: 2015-2021 target input-side fields only
target_support: K labeled target DA cycles, K in {0, 4, 12}
target_val:     unused in the main protocol
target_eval:    2023-2025 final offline evaluation only
```

Current paper-facing claims are K=0. `target_eval` labels are never used for
prompt construction, normalization, adaptation-sample selection, model
selection, hyperparameter selection, policy calibration, or threshold
calibration. `month` is a deployment-known month-of-year seasonal phase, not a
target_eval selection signal.

## HyperDA Operator Generator

For each target region `R`, build monthly prompt prototypes from
`target_context` input fields only:

```text
C_R = {x_t^R : t in target_context}
P_{R,m} = q_phi(summary({x_t^R in C_R : month(t) = m}))
zeta_{R,m} = H_psi(P_{R,m})
hat{Delta s}_t^R = f_{theta_0, zeta_{R,m}}(x_t^R)
```

`theta_0` is the source-trained base operator. `H_psi` generates lightweight
operator parameters, not full backbone weights:

- adapter parameters;
- output-head residual terms;
- optional FiLM parameters.

The paper-facing M2.1 source prior should be described as stable rank-gated
bounded-DoRA lightweight operator generation:

```text
zeta_{R,l} = zeta_{0,l} + sum_j alpha_{R,l,j} B_{l,j}
```

Mention `M2_1_rank_gated_dora_stable` only as an implementation identifier.

## HyperDA-TRUST

HyperDA-TRUST is Source-Manifold Trust-Routed Operator Generation. It uses only
source_fit/source_val artifacts to build a trust bank:

- source prompt embeddings;
- nearest-source distance quantiles;
- source-neighborhood coefficient consensus.

At target deployment, TRUST blends target-prompt coefficients with nearest
source-neighborhood consensus:

```text
coeff_star = tau_layer * coeff_target_prompt
           + (1 - tau_layer) * coeff_source_neighbor_consensus
```

The module records surface/rootzone trust gates and uses
`source_val_dual_variable_cvar_safe_score` for source-side selection. It does
not use target labels, target_val, target_eval statistics, or target_eval
performance to tune trust strength. As of the 2026-06-21 US-R1 seed0
preregistered matrix, `M3_1_hyperda_trust_medium` is the current K=0
HyperDA-TRUST candidate; M3_1a/b/c are negative or neutral ablations and M3_1d
is exploratory unless explicitly downgraded.

## SAFE Diagnostic

SAFE stands for Source-Anchored Few-Shot Operator Refinement. It may be
described in an appendix or diagnostic section, not as a current main
contribution. For `K in {4,12}`, SAFE starts from the generated prior and may
update only target-specific variables on K support cycles:

```text
theta_SAFE = theta_prior + alpha_K (theta_adapt - theta_prior)
```

K-shot runs require `safe_policy.json` with
`policy_source=source_side_episode_calibration`. Current K-shot rows that are
`rejected_to_k0_anchor` are K0-equivalent fallback and must not be reported as
accepted few-shot adaptation or few-shot improvement.

## Main Comparison Ladder

Use these paper-main method IDs:

```text
forecast_only
source_pooled_global_backbone
prompt_conditioned_shared_backbone
source_regime_specialist_bank
hyperda_zero_shot_context
hyperda_trust_zero_shot_context
```

SAFE K=4/K=12, adapter/LoRA K-shot baselines, ridge calibration, target-support
means, and full-target target-region oracles belong in diagnostic, appendix, or
legacy sections unless a later explicit paper decision promotes them.

## Reviewer Checks

Before drafting final prose, verify:

- all main methods use the same split manifest and protocol hash;
- normalization provenance is source-fit only;
- source-stage checkpoint and trust routing choices are selected on source
  validation/source-side episodes only;
- target-context prompt metadata records `label_usage=none`;
- final metrics are computed from target_eval only;
- any K-shot table exposes `stage3_posterior_decision`;
- rows marked `rejected_to_k0_anchor` are described as K0-equivalent fallback.

## Suggested Methods Sections

```text
3. Method
3.1 Neural DA Increment Emulation
3.2 Zero-Shot Target Information Boundary
3.3 Target-Context Prompt Prototypes
3.4 HyperDA Operator Generator
3.5 Source-Manifold Trust Routing
3.6 Training, Source-Side Selection, and Evaluation
Appendix: SAFE Diagnostic Few-Shot Refinement
```
