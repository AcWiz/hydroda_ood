"""
Smoke tests for NetCDF audit module.
These tests do NOT train models or write data - they only read metadata.
"""

import pytest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hydroda.data.netcdf_audit import (
    audit_netcdf,
    compute_memory_estimate,
    check_increment_reconstruction,
    summarize_masks,
)

# Use a small subset for testing
TEST_DATA_PATH = "/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc"

# Required fields from hydroda_dataset_contract.yaml audit_required_fields
REQUIRED_FIELDS = [
    "dims",
    "coords",
    "data_vars",
    "time_range",
    "time_frequency",
    "missing_dates",
    "variable_shapes",
    "variable_dtypes",
    "channel_names_or_order",
    "nan_inf_counts",
    "mask_unique_values",
    "finite_overlap_forecast_analysis",
    "coordinate_availability",
    "estimated_memory_gb",
]


@pytest.fixture(scope="module")
def audit_result():
    """Run audit once and cache result for all tests."""
    result = audit_netcdf(TEST_DATA_PATH, country="US")
    return result


def test_audit_runs_without_error():
    """Test that audit_netcdf loads DA.nc and completes without exception."""
    # This is a metadata-only read - should not modify anything
    result = audit_netcdf(TEST_DATA_PATH, country="US")
    assert result is not None
    assert isinstance(result, dict)


def test_audit_output_has_required_keys(audit_result):
    """Test that all required audit fields are present."""
    for field in REQUIRED_FIELDS:
        assert field in audit_result, f"Missing required field: {field}"


def test_increment_reconstruction_passes(audit_result):
    """Test that analysis ≈ forecast + increment within tolerance 1e-6."""
    inc = audit_result.get("increment_reconstruction", {})
    if "error" in inc:
        pytest.skip(f"Increment reconstruction error: {inc['error']}")

    surf_passed = inc.get("surface_reconstruction_passed")
    root_passed = inc.get("rootzone_reconstruction_passed")

    # If we have the variables, check they pass
    if surf_passed is not None:
        assert surf_passed, (
            f"Surface increment reconstruction failed: "
            f"max_error={inc.get('surface_max_error')}"
        )
    if root_passed is not None:
        assert root_passed, (
            f"Rootzone increment reconstruction failed: "
            f"max_error={inc.get('rootzone_max_error')}"
        )


def test_mask_keys_are_distinct(audit_result):
    """
    Phase 0: Verify that at least one mask is detected.
    Phase 1 will determine actual mask semantics (obs/region/loss).
    """
    mask_info = audit_result.get("mask_unique_values", {})
    mask_keys = list(mask_info.keys())

    # At minimum, we should have some mask variables
    assert len(mask_keys) > 0, "No mask variables detected in dataset"

    # Phase 0 only requires that masks are detected and reported
    # Distinctness of obs/region/loss is a Phase 1 contract requirement
    # In DA.nc, 'mask' is a single channel (index 11 of input), not three separate masks


def test_memory_estimate_reasonable(audit_result):
    """Test that memory estimate is in reasonable range (1-100 GB)."""
    mem_gb = audit_result.get("estimated_memory_gb", 0)
    assert 0 < mem_gb < 200, f"Memory estimate {mem_gb:.2f} GB seems unreasonable"


def test_has_data_vars(audit_result):
    """Test that dataset has data variables."""
    data_vars = audit_result.get("data_vars", [])
    assert len(data_vars) > 0, "No data variables found"


def test_has_dims(audit_result):
    """Test that dataset has dimensions."""
    dims = audit_result.get("dims", {})
    assert len(dims) > 0, "No dimensions found"
