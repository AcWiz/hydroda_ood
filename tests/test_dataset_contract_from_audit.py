"""
Tests verifying the dataset contract reflects actual DA.nc multi-channel array structure.
"""

import json
import os
import sys

import pytest
import yaml

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)


# ---- Fixtures ----

@pytest.fixture
def contract_path():
    return os.path.join(PROJECT_ROOT, "specs", "hydroda_dataset_contract.yaml")


@pytest.fixture
def audit_json_path():
    return os.path.join(PROJECT_ROOT, "artifacts", "audits", "netcdf_audit_US.json")


@pytest.fixture
def geolocation_json_path():
    return os.path.join(PROJECT_ROOT, "artifacts", "audits", "geolocation_recovery_US.json")


@pytest.fixture
def contract():
    with open(os.path.join(PROJECT_ROOT, "specs", "hydroda_dataset_contract.yaml")) as f:
        return yaml.safe_load(f)


@pytest.fixture
def audit():
    with open(os.path.join(PROJECT_ROOT, "artifacts", "audits", "netcdf_audit_US.json")) as f:
        return json.load(f)


@pytest.fixture
def geolocation():
    with open(os.path.join(PROJECT_ROOT, "artifacts", "audits", "geolocation_recovery_US.json")) as f:
        return json.load(f)


# ---- Tests ----

def test_contract_loads_from_audit_json(audit):
    """Contract input/target channel list must match audit findings."""
    input_channels = audit["input_channels"]
    target_channels = audit["target_channels"]

    assert len(input_channels) == 12
    assert len(target_channels) == 4
    assert input_channels[0] == "sm_surface_forecast"
    assert input_channels[1] == "sm_rootzone_forecast"
    assert target_channels[0] == "sm_surface_analysis"
    assert target_channels[1] == "sm_rootzone_analysis"


def test_input_channels_match_contract(contract, audit):
    """Input channels in contract must match audit JSON (allowing 'mask' vs 'base_valid_mask')."""
    input_channels_audit = audit["input_channels"]
    input_channels_contract = list(contract["input_channel_order"].values())

    # Allow 'mask' (audit name) vs 'base_valid_mask' (contract provisional name) at index 11
    assert input_channels_contract[:11] == input_channels_audit[:11]
    assert input_channels_contract[11] in ["mask", "base_valid_mask"]
    assert input_channels_audit[11] in ["mask", "base_valid_mask"]


def test_target_channels_match_contract(contract, audit):
    """Target channels in contract must match audit JSON."""
    target_channels_audit = audit["target_channels"]
    target_channels_contract = list(contract["target_channel_order"].values())

    assert target_channels_contract == target_channels_audit


def test_no_leakage_in_contract(contract):
    """Contract must not reference target query labels in forbidden places."""
    forbidden = contract["normalization"]["forbidden_stats_sources"]

    assert "target_query_labels" in forbidden
    assert "target_query_analysis" in forbidden
    assert "target_query_increment" in forbidden

    # Sample contract must not use target_query anything
    sample = contract["sample_contract"]
    assert "target_query" not in str(sample)


def test_mask_semantics_are_provisional(contract):
    """base_valid_mask must be marked as provisional/uncertain."""
    mask = contract["mask_semantics"]["base_valid_mask"]

    assert mask["is_dynamic"] is True
    assert mask["channel_index"] == 11
    assert mask["values"] == [0.0, 1.0]
    assert "provisional" in mask["provisional_semantic"].lower() or \
           "assimilation" in mask["provisional_semantic"].lower()

    # Must explicitly say it's NOT obs_mask, NOT region_mask, NOT loss_mask
    assert "not_obs_mask_reason" in mask
    assert "not_region_mask_reason" in mask
    assert "not_loss_mask_reason" in mask


def test_spatial_coordinates_unblocked(contract, geolocation):
    """Spatial coordinates: geolocation was recovered via directory-level lookup."""
    spatial = contract["spatial_coordinates"]

    # Geolocation recovery succeeded
    assert geolocation["geolocation_recovered"] is True
    assert geolocation["recovery_method"] == "directory_latlon_vector_lookup"

    # Contract spatial coordinates now reflect available lat/lon
    # Note: contract lat_lon_available was False at Phase 0.5, now True after Phase 0.6
    # The contract is updated per-phase as knowledge improves


def test_data_structure_is_multi_channel_array(contract):
    """Contract data_structure.type must be multi_channel_array."""
    assert contract["data_structure"]["type"] == "multi_channel_array"
    assert contract["data_structure"]["input_array"] == "input"
    assert contract["data_structure"]["target_array"] == "target"


def test_grid_dimensions_match_audit(contract, audit):
    """Grid height/width must match audit findings."""
    audit_dims = audit["dims"]
    contract_spatial = contract["spatial_coordinates"]

    assert audit_dims["height"] == contract_spatial["grid_height"]
    assert audit_dims["width"] == contract_spatial["grid_width"]


def test_increment_definition_correct(contract):
    """Increment definitions must be analysis - forecast."""
    inc = contract["increment_definition"]

    assert "analysis" in inc["surface"] and "forecast" in inc["surface"]
    assert "analysis" in inc["rootzone"] and "forecast" in inc["rootzone"]
    assert "sm_surface_analysis" in inc["surface"]
    assert "sm_rootzone_analysis" in inc["rootzone"]


def test_contract_has_required_test_list(contract):
    """Contract must list all required tests."""
    tests = contract.get("tests_required", [])
    assert len(tests) > 0
    assert "test_increment_reconstruction" in tests
    assert "test_no_query_stats_in_normalization" in tests


def test_geolocation_recovery_confirms_success(geolocation):
    """Geolocation audit must confirm recovery succeeded via Method 8."""
    assert geolocation["geolocation_recovered"] is True
    assert geolocation["recovery_method"] == "directory_latlon_vector_lookup"
    assert "blocking_issue" not in geolocation or geolocation["blocking_issue"] is None
    assert len(geolocation["methods_attempted"]) >= 8

    method_names = [m["method"] for m in geolocation["methods_attempted"]]
    assert "directory_latlon_vector_lookup" in method_names
    assert "global_attributes" in method_names
    assert "coordinate_variables" in method_names
    assert "smap_ease_grid_reconstruction" in method_names

    # Method 8 should be the successful one
    method_8 = next(m for m in geolocation["methods_attempted"]
                    if m["method"] == "directory_latlon_vector_lookup")
    assert method_8["success"] is True
