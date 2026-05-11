# Environment Setup

## Conda Environment

Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate hydroda-ood
```

## Core Dependencies

- **Python 3.10**
- **PyTorch 2.0+** (CPU or CUDA depending on hardware)
- **Scientific stack**: numpy, pandas, scipy, xarray, netcdf4, h5netcdf, dask
- **ML stats**: scikit-learn
- **Visualization**: matplotlib, cartopy
- **Utilities**: pyyaml, tqdm, rich
- **Logging**: wandb (Weights & Biases), tensorboard
- **Testing**: pytest, pytest-cov

## GPU Check

Run the environment check script to verify GPU detection:

```bash
PYTHONPATH=. python scripts/check_environment.py
```

Expected output:
```
==============================================================
HydroDA-OOD Environment Check
==============================================================
Python: 3.10...
Git hash: abc1234
...
  torch: 2.x.x
  cuda available: True
  gpu count: N
  GPU 0: NVIDIA RTX / A100 / ...
==============================================================
Environment check PASSED.
```

## Require GPU Mode

To enforce GPU requirement in scripts:

```bash
PYTHONPATH=. python scripts/check_environment.py --require_gpu
```

This will exit with code 1 if CUDA is unavailable.

## Lightweight Runtime Info

For quick diagnostics without full environment check:

```bash
PYTHONPATH=. python scripts/print_runtime_info.py
```

## Environment Variables

Key environment variables:
- `CUDA_VISIBLE_DEVICES`: GPU device IDs (e.g., "0,1,2")
- `CONDA_DEFAULT_ENV`: Current conda environment name

## Notes

- `num_workers=0` is the default for DataLoader due to netCDF threading issues
- Mixed precision (AMP) is enabled via `--amp` flag and auto-detected based on CUDA availability
- Wandb is disabled by default (`--wandb_mode disabled`); set to `online` or `offline` to enable