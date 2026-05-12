#!/usr/bin/env python3
"""Check and print environment information for HydroDA-OOD / HyperDA V4.

Usage:
    PYTHONPATH=. python scripts/utils/check_environment.py
    PYTHONPATH=. python scripts/utils/check_environment.py --require_gpu  # exit if no CUDA

This script checks:
- python version, torch version, torch cuda version
- cuda available, GPU count, GPU name, GPU total memory
- current device, cudnn enabled, conda env name
- CUDA_VISIBLE_DEVICES, hostname, git hash
- All core dependencies
- Artifact paths
"""

import argparse
import socket
import sys
from pathlib import Path

from hydroda.utils.runtime import get_git_hash


def main():
    parser = argparse.ArgumentParser(description="Check environment for HydroDA-OOD")
    parser.add_argument("--require_gpu", action="store_true",
        help="Exit with error if CUDA unavailable")
    args = parser.parse_args()

    print("=" * 60)
    print("HydroDA-OOD Environment Check")
    print("=" * 60)

    # Python version
    print(f"Python: {sys.version}")
    print(f"Hostname: {socket.gethostname()}")

    # Git info
    git_hash = get_git_hash()
    print(f"Git hash: {git_hash}")

    # Core scientific packages
    deps = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("xarray", "xarray"),
        ("netCDF4", "netCDF4"),
        ("scipy", "scipy"),
        ("sklearn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("cartopy", "cartopy"),
        ("pyyaml", "yaml"),
        ("tqdm", "tqdm"),
        ("pytest", "pytest"),
        ("torch", "torch"),
    ]

    missing = []
    print("\nDependencies:")
    for label, module in deps:
        try:
            __import__(module)
            print(f"  {label}: ok")
        except ImportError:
            print(f"  {label}: MISSING")
            missing.append(label)

    # torch specifics
    print("\nPyTorch:")
    try:
        import torch
        print(f"  torch: {torch.__version__}")
        print(f"  cuda available: {torch.cuda.is_available()}")
        print(f"  cudnn enabled: {torch.backends.cudnn.enabled}")
        if hasattr(torch.version, 'cuda'):
            print(f"  cuda version: {torch.version.cuda}")

        if torch.cuda.is_available():
            print(f"  gpu count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_total = props.total_memory / 1e9
                mem_alloc = torch.cuda.memory_allocated(i) / 1e9
                mem_res = torch.cuda.memory_reserved(i) / 1e9
                print(f"  GPU {i}: {props.name}")
                print(f"    total_memory: {mem_total:.1f}GB")
                print(f"    allocated: {mem_alloc:.1f}GB")
                print(f"    reserved: {mem_res:.1f}GB")
        else:
            print("  No GPU — running on CPU")
            if args.require_gpu:
                print("\nERROR: --require_gpu set but no CUDA device available.")
                sys.exit(1)
    except ImportError:
        print("  torch not installed")
        if args.require_gpu:
            print("\nERROR: --require_gpu set but torch not installed.")
            sys.exit(1)

    # Environment variables
    print("\nEnvironment:")
    import os
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "(not set)")
    print(f"  CUDA_VISIBLE_DEVICES: {cuda_vis}")
    print(f"  CONDA_DEFAULT_ENV: {conda_env}")

    # Artifact paths (check existence, don't load data)
    print("\nArtifact paths:")
    artifact_paths = [
        "/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc",
        "artifacts/geolocation/US_latlon.nc",
        "artifacts/regions/US_region_masks.nc",
        "artifacts/regions/US_region_mask_tensor.pt",
        "artifacts/regions/US_region_masks_manifest.json",
        "artifacts/splits/US_loro_kdate_splits.json",
    ]
    for p in artifact_paths:
        exists = Path(p).exists()
        print(f"  {p}: {'exists' if exists else 'NOT FOUND'}")

    print("\n" + "=" * 60)

    if missing:
        print(f"ERRORS: missing dependencies: {missing}")
        sys.exit(1)

    print("Environment check PASSED.")


if __name__ == "__main__":
    main()