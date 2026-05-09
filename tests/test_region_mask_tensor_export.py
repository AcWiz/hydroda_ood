"""Tests for US region mask tensor export."""

import torch
from pathlib import Path


def test_pt_file_exists():
    pt_path = Path("artifacts/regions/US_region_mask_tensor.pt")
    assert pt_path.exists(), f".pt not found: {pt_path}"


def test_pt_is_tensor():
    pt_path = Path("artifacts/regions/US_region_mask_tensor.pt")
    tensor = torch.load(pt_path)
    assert isinstance(tensor, torch.Tensor), f"Expected torch.Tensor, got {type(tensor)}"


def test_pt_shape():
    pt_path = Path("artifacts/regions/US_region_mask_tensor.pt")
    tensor = torch.load(pt_path)
    assert tensor.shape == (256, 640), f"Expected (256, 640), got {tensor.shape}"


def test_pt_dtype_is_uint8():
    """PT fast mirror uses torch.uint8 (NC canonical is float32)."""
    pt_path = Path("artifacts/regions/US_region_mask_tensor.pt")
    tensor = torch.load(pt_path)
    assert tensor.dtype == torch.uint8, f"Expected torch.uint8, got {tensor.dtype}"


def test_pt_unique_ids_subset_of_0_to_6():
    pt_path = Path("artifacts/regions/US_region_mask_tensor.pt")
    tensor = torch.load(pt_path)
    unique = sorted(tensor.unique().tolist())
    assert all(int(u) in range(7) for u in unique), \
        f"Unexpected ids found: {unique}. Expected 0..6"