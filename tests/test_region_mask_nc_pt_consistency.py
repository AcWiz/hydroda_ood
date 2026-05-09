"""Tests for NC and .pt consistency."""

import numpy as np
import torch
import xarray as xr


def load_nc_mask():
    ds = xr.open_dataset("artifacts/regions/US_region_masks.nc")
    arr = ds["region_mask_integer"].values
    ds.close()
    return arr


def load_pt_mask():
    tensor = torch.load("artifacts/regions/US_region_mask_tensor.pt")
    return tensor.numpy()


def test_nc_pt_shape_match():
    nc = load_nc_mask()
    pt = load_pt_mask()
    assert nc.shape == pt.shape, f"Shape mismatch: NC={nc.shape}, PT={pt.shape}"


def test_nc_pt_value_exact_match():
    nc = load_nc_mask()
    pt = load_pt_mask()
    # NC stores float32 but values are integers 0..6; cast to uint8 for comparison
    assert np.array_equal(nc.astype(np.uint8), pt), \
        "NC and PT values do not match after uint8 cast"


def test_nc_unique_ids_0_to_6():
    nc = load_nc_mask()
    unique = sorted(np.unique(nc))
    assert unique == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], f"Unexpected ids: {unique}"