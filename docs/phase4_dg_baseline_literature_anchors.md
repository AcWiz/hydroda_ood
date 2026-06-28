# Phase 4 DG Baseline Literature Anchors

Last updated: 2026-06-24

This note records the paper anchors for the Phase 4 domain-generalization
baseline wrapper:

```text
run/phase4_dg_baselines_us_r1_seed0.sh
```

The default runnable methods are:

```text
swad mixstyle disam udim moment_align iu
```

`ssa_reg`, `tca`, `self_bootstrap`, and `deep_coral` are retained as explicit
diagnostic methods and are not run by default.

## Default Methods

| Wrapper method | Method id in this repo | Paper title | Venue / year | Link | Repo status |
|---|---|---|---|---|---|
| `swad` | `swad_source_pooled_global_backbone` | SWAD: Domain Generalization by Seeking Flat Minima | NeurIPS 2021 | https://proceedings.neurips.cc/paper/2021/hash/bcb41ccdc4363c6848a1d760f26c28a0-Abstract.html | paper-main candidate under the current US-only transition |
| `mixstyle` | `mixstyle_source_pooled_global_backbone` | Domain Generalization with MixStyle | ICLR 2021 | https://openreview.net/forum?id=6xHJ37MVxxp | paper-main candidate under the current US-only transition |
| `disam` | `disam_source_domain_sharpness_alignment` | Domain-Inspired Sharpness-Aware Minimization Under Domain Shifts | ICLR 2024 | https://openreview.net/forum?id=I4wB3HA3dJ | paper-main source-only DG candidate under the current US-only transition |
| `udim` | `udim_unknown_domain_inconsistency_minimization` | Unknown Domain Inconsistency Minimization for Domain Generalization | ICLR 2024 | https://openreview.net/forum?id=eNoiRal5xi | paper-main source-only DG candidate under the current US-only transition |
| `moment_align` | `moment_alignment_source_domain_invariance` | Moment Alignment: Unifying Gradient and Hessian Matching for Domain Generalization | UAI 2025 | https://openreview.net/forum?id=EzwlQDs5Ck | paper-main source-only DG candidate under the current US-only transition |
| `iu` | `identify_unlearn_source_domain_gradient_ascent` | Unlearning during Training: Domain-Specific Gradient Ascent for Out-of-Distribution Generalization | ICLR 2026 | https://openreview.net/forum?id=9ufS5Jl0O0 | paper-main source-only DG candidate under the current US-only transition |

## Retained Internal Diagnostic

| Wrapper method | Method id in this repo | Paper title | Venue / year | Link | Repo status |
|---|---|---|---|---|---|
| `ssa_reg` | `ssa_reg_target_context_subspace_alignment` | Test-time Adaptation for Regression by Subspace Alignment | ICLR 2025 | https://openreview.net/forum?id=SXtl7NRyE5 | runnable target-context diagnostic, not DG paper-main |
| `tca` | `tca_target_context_correlation_alignment` | Test-time Correlation Alignment | ICML 2025 | https://openreview.net/forum?id=0dualJz9OI | runnable target-context diagnostic, not DG paper-main |
| `self_bootstrap` | `self_bootstrap_target_context_consistency_tta` | Self-Bootstrapping for Versatile Test-Time Adaptation | ICML 2025 | https://openreview.net/forum?id=Li4rieeClO | runnable target-context diagnostic, not DG paper-main |
| `deep_coral` | `deep_coral_target_context_alignment` | Deep CORAL: Correlation Alignment for Deep Domain Adaptation | ECCV Workshops 2016 | https://arxiv.org/abs/1607.01719 | old internal diagnostic, not run by default |

## Protocol Notes

- The DG paper-main wrapper defaults to source-only DG methods: SWAD,
  MixStyle, DISAM, UDIM, Moment Alignment, and IU.
- DISAM, UDIM, Moment Alignment, and IU use pooled source samples with
  region-mask grouped source-domain losses/features. They do not expand each
  date into five source-region episodes and do not create a target-context
  loader.
- SSA-Reg, TCA, and self-bootstrap are target-context adaptation diagnostics,
  not DG paper-main baselines.
- Diagnostic target-context methods may use only unlabeled `target_context`
  2015-2021 input-side data.
- `target_eval` remains final offline evaluation only; it must not be used for
  adaptation, checkpoint selection, hyperparameter selection, or calibration.
- `assert_allowed_for_table(..., "paper_main")` must continue to reject:

```text
tca_target_context_correlation_alignment
self_bootstrap_target_context_consistency_tta
deep_coral_target_context_alignment
ssa_reg_target_context_subspace_alignment
```

First-round future extension pool, not default runnable:

```text
QT-DoG ICML 2025
Continuous Temporal DG NeurIPS 2024
Continuous DG NeurIPS 2025
```
