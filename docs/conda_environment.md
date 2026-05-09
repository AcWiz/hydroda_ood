# Conda Environment Guide

## Quick Start

```bash
conda env create -f environment.yml
conda activate hydroda-ood
```

## Updating the Environment

```bash
conda env update -f environment.yml --prune
```

## Verification

Run the environment check script:

```bash
python scripts/check_environment.py
```

Expected output lists all core dependencies as `ok` and verifies key artifact paths.

## CPU vs GPU PyTorch

The default `environment.yml` installs **CPU-only PyTorch** to avoid CUDA version mismatches on non-GPU machines.

### GPU Users

After creating the environment, install GPU-enabled PyTorch:

```bash
# Identify your CUDA version first
nvcc --version

# Install GPU PyTorch (example for CUDA 12.8)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Or for CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Phase 3B Smoke Check (CPU-safe)

Phase 3B minimal runs CPU-only with streaming NetCDF reads:

```bash
python scripts/run_phase3B_minimal.py
```

No GPU required for Phase 3B.

## Do NOT Put Data Files in the Environment

The `environment.yml` specifies dependencies only. **Do not** include artifact or data files:

- `DA.nc` and other NetCDF data — stays in `/fastersharefiles2/fenglonghan/dataset/SMAP/`
- Region masks and tensors — stays in `artifacts/regions/`
- Geolocation files — stays in `artifacts/geolocation/`

This keeps the environment spec portable and reproducible across machines.

## Troubleshooting

### ImportError: libGL.so not found

Install system graphics libraries:

```bash
# Debian/Ubuntu
apt-get install libgl1-mesa-glx libglib2.0-0

# macOS
brew install glib
```

### Cartopy background maps missing

```bash
pip install shapely --no-binary shapely
```

### slowdask diagnostics

If xarray/dask chunking is slow, check:

```bash
python -c "import dask; dask.config.get('distributed.workers', None)"
```