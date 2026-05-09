"""Lightweight environment sanity check."""
import sys
from pathlib import Path


def main():
    # Python version
    print(f"Python: {sys.version}")

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
    for label, module in deps:
        try:
            __import__(module)
            print(f"  {label}: ok")
        except ImportError:
            print(f"  {label}: MISSING")
            missing.append(label)

    # torch specifics
    try:
        import torch
        print(f"  torch: {torch.__version__}")
        print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    except ImportError:
        pass

    # Artifact paths (check existence, don't load data)
    artifact_paths = [
        "/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc",
        "artifacts/geolocation/US_latlon.nc",
        "artifacts/regions/US_region_masks.nc",
        "artifacts/regions/US_region_mask_tensor.pt",
        "artifacts/regions/US_region_masks_manifest.json",
        "artifacts/splits/US_lloro_kdate_splits.json",
    ]
    print("\nArtifact paths:")
    for p in artifact_paths:
        exists = Path(p).exists()
        print(f"  {p}: {'exists' if exists else 'NOT FOUND'}")

    if missing:
        print(f"\nERRORS: missing {missing}")
        sys.exit(1)
    print("\nAll core deps OK.")


if __name__ == "__main__":
    main()