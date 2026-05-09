"""Tests: no runtime region mask regeneration in dataset or baselines."""

from pathlib import Path


def test_dataset_does_not_call_build_region_masks():
    """HydroDADataset must load from frozen artifacts, not regenerate masks."""
    dataset_py = Path("hydroda/data/dataset.py")
    if not dataset_py.exists():
        return  # skip if dataset.py not yet created

    content = dataset_py.read_text()
    forbidden = [
        "build_region_masks",
        "generate_region_masks",
        "create_region_masks",
        "make_region_mask",
    ]
    for term in forbidden:
        assert term not in content, \
            f"dataset.py contains '{term}' — masks must be loaded from artifacts, not regenerated"


def test_baselines_do_not_regenerate_masks():
    """
    Baseline runner scripts must not contain mask generation logic.
    The build_us_region_masks.py is the one-time generation script (excluded).
    """
    baseline_dirs = [
        Path("hydroda/baselines"),
    ]
    # Exclude one-time build scripts — they are artifacts generation, not runtime
    exclude_patterns = {"build_us_region_masks", "audit_label_threshold_sensitivity", "export_region_mask_tensor", "write_region_masks_manifest"}
    forbidden = [
        "build_region_masks",
        "generate_region_masks",
        "create_region_masks",
        "make_region_mask",
    ]

    found = []
    for d in baseline_dirs:
        if not d.exists():
            continue
        for py_file in d.rglob("*.py"):
            if py_file.stem in exclude_patterns:
                continue
            content = py_file.read_text()
            for term in forbidden:
                if term in content:
                    found.append(f"{py_file}: '{term}'")

    assert not found, f"Mask regeneration found in: {found}"