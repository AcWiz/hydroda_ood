# Run Entry Points

Thin shell wrappers for launching experiments. Each script sets environment variables
and calls a Python training/evaluation entry point under `scripts/train/` or `scripts/eval/`.

## Scripts

| Script | Description | Python Entry | Phase |
|--------|-------------|-------------|-------|
| `phase4_source_only.sh` | Train `source_pooled_global_backbone` (US-only LORO transition global) | `scripts/train/train_source_only_backbone.py` | 4 |
| `phase4_source_only_all_regions.sh` | Train `legacy_all_regions_sanity` on all US regions | `scripts/train/train_source_only_all_regions.py` | 4 |
| `phase4_source_only_all_regions_eval.sh` | Evaluate all-region source-only checkpoint by region | `scripts/eval/eval_source_only_all_regions.py` | 4 |
| `phase4_source_only_region_specific.sh` | Train `target_full_history_region_oracle` scratch upper bound | `scripts/train/train_source_only_region_specific.py` | 4 |
| `phase4_source_only_region_specific_finetune.sh` | Train `target_full_history_region_oracle` finetune upper bound | `scripts/train/train_source_only_region_specific.py` | 4 |
| `phase4_source_only_inference.sh` | Evaluate a source-only checkpoint | `scripts/eval/evaluate_checkpoint.py` | 4 |
| `phase4_prompt_conditioned.sh` | Train prompt-conditioned shared backbone | `scripts/train/train_prompt_conditioned_shared.py` | 4 |
| `phase4_prompt_conditioned_inference.sh` | Evaluate prompt-conditioned checkpoint | `scripts/eval/evaluate_checkpoint.py` | 4 |
| `phase4_hyperda.sh` | Train HyperDA source-stage basis-adapter prior | `scripts/train/train_prompt_conditioned_shared.py` | 4 |
| `phase5_hyperda_zero_few_shot.sh` | Main HyperDA zero/few-shot target generalization | `scripts/train/train_hyperda_few_shot_adapt.py` | 5 |
| `phase5_hyperda_zero_few_shot_eval.sh` | Run HyperDA K=0/4/12 adaptation and target_eval evaluation for one region | `scripts/train/train_hyperda_few_shot_adapt.py` + `scripts/eval/evaluate_checkpoint.py` | 5 |
| `phase5_hyperda_target_adapt.sh` | Legacy/internal full-target historical adaptation | `scripts/train/train_hyperda_target_adapt.py` | 5 |
| `phase5_hydroda_der.sh` | Legacy/secondary target_val HydroDA-DER router | `scripts/eval/evaluate_der_router.py` | 5 |

## Usage

Paper main method IDs:

```text
forecast_only
source_pooled_global_backbone
prompt_conditioned_shared_backbone
source_regime_specialist_bank
hyperda_zero_shot_context
hyperda_few_shot_k4
hyperda_few_shot_k12
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

# HyperDA source-stage prior and paper-facing zero/few-shot adaptation
bash run/phase4_hyperda.sh US-R1 0 1
bash run/phase5_hyperda_zero_few_shot.sh <source_checkpoint> US-R1 4 0 1

# One-click single-region HyperDA K=0/4/12 adaptation and target_eval
bash run/phase5_hyperda_zero_few_shot_eval.sh \
  /path/to/source_hyperda_checkpoint.pt US-R1 0 1

# The wrapper prints a copy-ready target_eval table and writes:
#   <output_base>/overview.csv
#   <output_base>/overview.md
#   <output_base>/overview.json
#   <output_base>/K*/adapt/metadata.json

# Same, using the wrapper's source HyperDA checkpoint auto-discovery
bash run/phase5_hyperda_zero_few_shot_eval.sh "" US-R1 0 1

# Conservative source-anchor recipe knobs; defaults are preregistered and
# must not be selected from target_val or target_eval labels.
ADAPT_RECIPE=source_anchor ANCHOR_ALPHA_K4=0.75 ANCHOR_ALPHA_K12=0.25 \
LR_K12=3e-4 MAX_STEPS_K12=80 \
  bash run/phase5_hyperda_zero_few_shot_eval.sh /path/to/source.pt US-R1 0 1

# Review a completed run before comparing against paper baselines.
PYTHONPATH=. python scripts/analysis/review_hyperda_zero_few_shot_run.py \
  --run_dir artifacts/runs/phase5_hyperda_zero_few_shot_eval/<run> \
  --baseline_overview artifacts/runs/<v4_4_source_only_overview>.json

# One-step smoke for the zero/few-shot runner
MAX_STEPS=1 bash run/phase5_hyperda_zero_few_shot.sh <source_checkpoint> US-R1 4 0 1

# Fast adapt+eval smoke; EVAL_MAX_SAMPLES limits target_eval samples
MAX_STEPS=1 EVAL_MAX_SAMPLES=2 K_LIST="0 4" \
  bash run/phase5_hyperda_zero_few_shot_eval.sh /path/to/source.pt US-R1 0 1

# Legacy/internal full-target adaptation reproduction
bash run/phase5_hyperda_target_adapt.sh <source_checkpoint> US-R1 0 1

# Legacy/secondary HydroDA-DER target_val router selection then target_eval final evaluation
SURFACE_CHECKPOINT=/path/to/phase4_global.pt \
ROOTZONE_CHECKPOINT=/path/to/runA.pt \
  bash run/phase5_hydroda_der.sh US-R1 0 1
```

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
