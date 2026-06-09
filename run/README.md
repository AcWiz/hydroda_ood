# Run Entry Points

Thin shell wrappers for launching experiments. Each script sets environment variables
and calls a Python training/evaluation entry point under `scripts/train/` or `scripts/eval/`.

## Scripts

| Script | Description | Python Entry | Phase |
|--------|-------------|-------------|-------|
| `phase4_source_only.sh` | Train source-only backbone on a US region | `scripts/train/train_source_only_backbone.py` | 4 |
| `phase4_source_only_all_regions.sh` | Train pooled global backbone on all US regions | `scripts/train/train_source_only_all_regions.py` | 4 |
| `phase4_source_only_all_regions_eval.sh` | Evaluate all-region source-only checkpoint by region | `scripts/eval/eval_source_only_all_regions.py` | 4 |
| `phase4_source_only_region_specific.sh` | Train region-specific scratch backbones | `scripts/train/train_source_only_region_specific.py` | 4 |
| `phase4_source_only_region_specific_finetune.sh` | Train region-specific backbones initialized from a pooled global checkpoint | `scripts/train/train_source_only_region_specific.py` | 4 |
| `phase4_source_only_inference.sh` | Evaluate a source-only checkpoint | `scripts/eval/evaluate_checkpoint.py` | 4 |
| `phase4_prompt_conditioned.sh` | Train prompt-conditioned shared backbone | `scripts/train/train_prompt_conditioned_shared.py` | 4 |
| `phase4_prompt_conditioned_inference.sh` | Evaluate prompt-conditioned checkpoint | `scripts/eval/evaluate_checkpoint.py` | 4 |
| `phase4_hyperda.sh` | Train HyperDA source-stage basis-adapter prior | `scripts/train/train_prompt_conditioned_shared.py` | 4 |
| `phase5_hyperda_target_adapt.sh` | Preregister HyperDA target historical adaptation protocol | `scripts/train/train_hyperda_target_adapt.py` planned | 5 |

## Usage

```bash
# Default: US-R1, adaptation_setting=target_full_train, seed=0
bash run/phase4_source_only.sh

# Paper-facing strong source baselines
bash run/phase4_source_only_all_regions.sh 0 0
bash run/phase4_source_only_region_specific.sh 0 0
bash run/phase4_source_only_region_specific_finetune.sh

# Explicit pooled global checkpoint for region-specific finetune
bash run/phase4_source_only_region_specific_finetune.sh \
  artifacts/runs/phase4_source_only_all_regions/<run>/checkpoints/checkpoint_best_source_val_safe_score.pt 0 1

# HyperDA source-stage prior and target adaptation protocol skeleton
bash run/phase4_hyperda.sh US-R1 0 1
bash run/phase5_hyperda_target_adapt.sh

# One-batch smoke for the target adaptation runner
MAX_EPOCHS=1 MAX_TRAIN_BATCHES=1 MAX_VAL_BATCHES=1 \
  bash run/phase5_hyperda_target_adapt.sh

# Explicit source checkpoint / region / seed / GPU
bash run/phase5_hyperda_target_adapt.sh <source_checkpoint> US-R1 0 1
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
