# Run Entry Points

Thin shell wrappers for launching experiments. Each script sets environment variables
and calls a Python training/evaluation entry point under `scripts/train/` or `scripts/eval/`.

## Scripts

| Script | Description | Python Entry | Phase |
|--------|-------------|-------------|-------|
| `phase4_source_only.sh` | Train source-only backbone on a US region | `scripts/train/train_source_only_backbone.py` | 4 |
| `phase5_prompt.sh` | Train prompt-conditioned shared backbone | `scripts/train/train_prompt_conditioned_shared.py` | 5 |

## Usage

```bash
# Default: US-R1, K=0, seed=0
bash run/phase4_source_only.sh

# Custom region, K, seed
bash run/phase4_source_only.sh US-R2 0 1
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
