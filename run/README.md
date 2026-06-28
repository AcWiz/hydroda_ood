# Run Entry Points

Thin shell wrappers for launching experiments. Each script sets environment variables
and calls a Python training/evaluation entry point under `scripts/train/` or `scripts/eval/`.

## Paper-Facing Wrappers

| Script | Description | Python Entry | Phase |
|--------|-------------|-------------|-------|
| `phase4_source_only.sh` | Train `source_pooled_global_backbone` (US-only LORO transition global) | `scripts/train/train_source_only_backbone.py` | 4 |
| `phase4_source_only_inference.sh` | Evaluate a source-only checkpoint | `scripts/eval/evaluate_checkpoint.py` | 4 |
| `phase4_prompt_conditioned.sh` | Train prompt-conditioned shared backbone | `scripts/train/train_prompt_conditioned_shared.py` | 4 |
| `phase4_prompt_conditioned_inference.sh` | Evaluate prompt-conditioned checkpoint | `scripts/eval/evaluate_checkpoint.py` | 4 |
| `phase4_hyperda_staged.sh` | Train staged HyperDA source prior from a frozen source-only base | `scripts/train/train_prompt_conditioned_shared.py` | 4 |
| `phase4_hyperda.sh` | Compatibility wrapper that delegates to `phase4_hyperda_staged.sh` | `run/phase4_hyperda_staged.sh` | 4 |
| `phase4_hyperda_inference.sh` | Evaluate staged HyperDA source-stage checkpoint | `scripts/eval/evaluate_checkpoint.py` | 4 |
| `hyperda_safe_us_r1_seed0.sh` | SAFE diagnostic convenience wrapper for US-R1 seed0 K-shot policy checks | `run/phase5_hyperda_zero_few_shot_eval.sh` | diagnostic |
| `phase5_hyperda_zero_few_shot.sh` | Main HyperDA zero/few-shot target generalization | `scripts/train/train_hyperda_few_shot_adapt.py` | 5 |
| `phase5_hyperda_zero_few_shot_eval.sh` | Run HyperDA K=0/4/12 adaptation and target_eval evaluation for one region | `scripts/train/train_hyperda_few_shot_adapt.py` + `scripts/eval/evaluate_checkpoint.py` | 5 |

## Diagnostic, Ablation, And Legacy Wrappers

These scripts may be useful for reproduction or source-side diagnostics, but
they are not paper-main protocol entrypoints.

| Script | Status | Notes |
|--------|--------|-------|
| `phase4_source_only_all_regions.sh` | legacy/internal | Trains all US regions; not OOD global. |
| `phase4_source_only_all_regions_eval.sh` | legacy/internal | Evaluation for all-regions sanity checkpoints. |
| `phase4_source_only_region_specific.sh` | legacy/internal oracle | Uses target-region history; not source-trained specialist bank. |
| `phase4_source_only_region_specific_finetune.sh` | legacy/internal oracle | Full-target finetune upper bound. |
| `phase4_hyperda_staged_ablation.sh` | source-side ablation | Rank/gating/DoRA diagnostics until promoted by evidence. |
| `stage3_calibrate_safe_policy_and_eval_us_r1_seed0.sh` | source-side policy calibration + eval convenience | Produces SAFE policy then delegates to paper-facing wrapper. |
| `stage3_hyperda_posterior_eval.sh` | diagnostic/ablation | Posterior policy diagnostics; strict paper mode required for claims. |
| `stage3_hyperda_posterior_smoke.sh` | diagnostic smoke | Fast smoke defaults. |
| `stage3_hyperda_posterior_full_inference.sh` | diagnostic full inference | Not a paper-main wrapper by default. |

## Usage

Paper main method IDs:

```text
forecast_only
source_pooled_global_backbone
prompt_conditioned_shared_backbone
source_regime_specialist_bank
hyperda_zero_shot_context
hyperda_trust_zero_shot_context
```

`source_only_backbone` remains a display alias for `source_pooled_global_backbone`.
Internal heuristic calibration baselines, `legacy_all_regions_sanity`, and
`target_full_history_region_oracle` reproduction scripts are not paper-main
entries.

```bash
# Default source-stage baselines use the frozen source_fit/source_val split.
bash run/phase4_source_only.sh

# Legacy all-regions sanity; not OOD global because it includes target-region history.
bash run/phase4_source_only_all_regions.sh 0 0

# Oracle/internal region-specific upper bounds, not source-trained specialist bank.
bash run/phase4_source_only_region_specific.sh 0 0
bash run/phase4_source_only_region_specific_finetune.sh

# Explicit legacy all-regions checkpoint for oracle finetune
bash run/phase4_source_only_region_specific_finetune.sh \
  artifacts/runs/phase4_source_only_all_regions/<run>/checkpoints/checkpoint_best_source_val_safe_score.pt 0 1

# Train staged HyperDA source prior.
# Stage 1 source-only checkpoint is loaded and frozen; Stage 2 trains prompt
# encoder, FiLM, and basis-adapter generation modules only.
bash run/phase4_hyperda_staged.sh auto US-R1 0 1
bash run/hyperda_safe_us_r1_seed0.sh

# Source-stage mainline: HyperDA Operator Generator with stable rank-gated bounded-DoRA.
# M2_1 uses shared_layer_aware_rank_gated_stable, dora_like_gain_bounded,
# temperature `2.0`, `USE_AMP=0`, and `LR=2e-4`.
ABLATION_ID=M2_1_rank_gated_dora_stable bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 1

# Retired failed exploration: original M2_rank_gated_dora had AMP skip/numerical
# failure and is `retired_failed_exploration_not_paper_main`.

# Secondary diagnostic: M2.2 saliency prior remains outside the mainline.
# Its wrapper uses legacy_gate_logit_bias_before_topk only as negative/diagnostic evidence.
PYTHONPATH=. python scripts/train/build_source_basis_saliency_prior.py \
  --source_checkpoint /path/to/M2_1_source_checkpoint.pt \
  --target_region US-R1 \
  --output_path artifacts/priors/source_basis_saliency/US-R1_s0.pt \
  --source_split source_fit
ABLATION_ID=M2_2_source_saliency_prior \
  bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 1
# The M2.2 wrapper defaults to:
#   artifacts/priors/source_basis_saliency/<target_region>_s<seed>.pt
# If the default prior is missing, the wrapper auto-builds it from the latest
# matching M2.1 stable HyperDA source-stage checkpoint. Disable with
# HYPER_SOURCE_SALIENCY_AUTO_BUILD=0.
# Override with HYPER_SOURCE_SALIENCY_PRIOR_PATH=/path/to/prior.pt when needed.
# Hessian/Fisher/top-parameter selection is reserved for a future source-side
# ablation and is not part of the paper-facing mainline.

# Source-safe residual diagnostic: M2.3 starts from M2.1 settings, keeps
# saliency as soft metadata/no hard top-k routing, and uses source-val-only rho
# selection for source-base + conservative HyperDA residual. By default it
# auto-discovers the latest matching M2.1 checkpoint; set
# M2_3_INIT_FROM_M2_1_CHECKPOINT= to disable that initialization.
ABLATION_ID=M2_3_source_safe_residual_hyperda \
  bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 1

# Prompt-only DA-aware diagnostic: M2.5a keeps M2.1 router/residual settings
# and switches only the source-stage prompt encoder to robust input-side DA
# diagnostics. Channel 11 remains bounded diagnostic coverage only, not a
# loss/metric/observation/region hard mask.
ABLATION_ID=M2_5a_da_aware_prompt_only \
  bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 1

# Source-manifold guarded diagnostic: M2.6 keeps the M2.1 current_mean_std
# source prior settings, uses source_base_residual_reliability_gated with
# rho=1.0, and adds only a source_fit/source_val calibrated
# source_manifold_distance_bounded guard. It is not target_eval-tuned shrinkage.
ABLATION_ID=M2_6_source_manifold_guarded_prior \
  bash run/phase4_hyperda_staged_ablation.sh auto US-R1 0 1

# HyperDA-TRUST decision: as of 2026-06-21, M3_1 remains the US-R1 seed0 K=0
# candidate. M3_1a/b/c are negative_or_neutral_preregistered_ablation and must
# not replace M3_1; M3_1d should only run as exploratory after US-R1
# target_eval has been inspected.
# Evidence: reports/experiments/M3_1_plus_us_r1_seed0_ablation_decision_20260621.md
#
# Next strict robustness pair, when GPU memory is available:
ABLATION_ID=M3_1_hyperda_trust_medium \
  bash run/phase4_hyperda_staged_ablation.sh auto US-R2 0 0
ABLATION_ID=M2_1_rank_gated_dora_stable \
  bash run/phase4_hyperda_staged_ablation.sh auto US-R2 0 0

# Source-side SAFE policy calibration from the M2.1 source prior, then diagnostic
# K0/K4/K12 target_eval. Current K-shot rows must be interpreted through
# stage3_posterior_decision; rejected_to_k0_anchor is K0-equivalent fallback.
# The wrapper auto-discovers the latest M2_1_rank_gated_dora_stable US-R1 seed0
# checkpoint when SOURCE_CHECKPOINT is not provided.
bash run/stage3_calibrate_safe_policy_and_eval_us_r1_seed0.sh

# One-click paper-safe single-region HyperDA K=0/4/12 adaptation and target_eval.
# If SAFE_POLICY_JSON is omitted, the wrapper first reuses the cached
# source-side safe_policy.json keyed by checkpoint/split/region/seed/budget.
# Set AUTO_GENERATE_SAFE_POLICY=1 once when the cache is missing.
AUTO_GENERATE_SAFE_POLICY=1 \
  bash run/phase5_hyperda_zero_few_shot_eval.sh \
  /path/to/source_hyperda_checkpoint.pt US-R1 0 1

# The wrapper prints a copy-ready target_eval table and writes:
#   <output_base>/overview.csv
#   <output_base>/overview.md
#   <output_base>/overview.json
#   <output_base>/K*/adapt/metadata.json

# Same, using the wrapper's source HyperDA checkpoint auto-discovery and
# cached source-side policy reuse.
bash run/phase5_hyperda_zero_few_shot_eval.sh "" US-R1 0 1

# K=4/K=12 SAFE diagnostic runs require source-side SAFE policy calibration.
# The calibration export must be safe_policy.json with:
#   policy_source=source_side_episode_calibration
#   target_val_usage=unused_in_main_protocol
#   target_eval_usage=final_eval_only_no_selection
# The eval wrapper applies policy adapt_scope/lr/steps/alpha during adaptation
# and uses policy adapt_mix_rho for target_eval unless ADAPT_MIX_RHO is set.
# K=0 can run without SAFE_POLICY_JSON because it uses no target labels.
# Cache controls:
#   SAFE_POLICY_CACHE_ROOT=artifacts/runs/stage3_source_safe_policy_cache
#   AUTO_GENERATE_SAFE_POLICY=0|1
#   SAFE_POLICY_SOURCE_QUERY_MAX_SAMPLES=256

# Diagnostic-only K-shot run without source-side policy calibration. K-shot
# checkpoints are marked diagnostic and fall back to the K0 anchor state.
REQUIRE_SAFE_POLICY_JSON_FOR_KSHOT=0 \
  bash run/phase5_hyperda_zero_few_shot_eval.sh /path/to/source.pt US-R1 0 1

# Conservative source-anchor recipe knobs; defaults are preregistered and
# must not be selected from target_val or target_eval labels.
ADAPT_RECIPE=source_anchor ANCHOR_ALPHA_K4=0.75 ANCHOR_ALPHA_K12=0.25 \
LR_K12=3e-4 MAX_STEPS_K12=80 \
  bash run/phase5_hyperda_zero_few_shot_eval.sh /path/to/source.pt US-R1 0 1

# SAFE diagnostic: Source-Anchored Few-Shot Operator Refinement fixes these
# source-preregistered defaults and delegates to the stable zero/few-shot runner.
SOURCE_CHECKPOINT=/path/to/source.pt bash run/hyperda_safe_us_r1_seed0.sh

# Review a completed run before comparing against paper baselines.
PYTHONPATH=. python scripts/analysis/review_hyperda_zero_few_shot_run.py \
  --run_dir artifacts/runs/phase5_hyperda_zero_few_shot_eval/<run> \
  --baseline_overview artifacts/runs/<v4_4_source_only_overview>.json

# One-step smoke for the zero/few-shot runner
MAX_STEPS=1 bash run/phase5_hyperda_zero_few_shot.sh <source_checkpoint> US-R1 4 0 1

# Fast adapt+eval smoke; EVAL_MAX_SAMPLES limits target_eval samples
MAX_STEPS=1 EVAL_MAX_SAMPLES=2 K_LIST="0 4" \
  bash run/phase5_hyperda_zero_few_shot_eval.sh /path/to/source.pt US-R1 0 1

# Legacy/internal full-target and target_val-router code is retained for
# historical reproduction, but it is not an active run entrypoint.
```

Retired failed explorations with status
`retired_failed_exploration_not_paper_main`:

```text
phase6_surface_residual_ridge
phase6_bora_residual_adapter
phase7_hyperda_apo
```

Existing artifacts may be retained as internal evidence, but these are not a
paper-main method and should not be restored as active run wrappers or configs.

## Prerequisites

- conda environment `hydroda-ood` activated
- `PYTHONPATH=.` set (script does this automatically)
- DA.nc at `/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc`
- Region masks at `artifacts/regions/`
- Splits at `artifacts/splits/`

## Adding New Entries

When adding a new experiment phase:
1. Create the Python entry in `scripts/train/` or `scripts/eval/`
2. Create a thin shell wrapper here that calls it
3. Update this README
