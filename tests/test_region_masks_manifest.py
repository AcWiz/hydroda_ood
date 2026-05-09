"""Tests for region masks manifest."""

import json
from pathlib import Path

from hydroda.data.region_artifacts import load_region_manifest
from hydroda.data.file_hash import compute_sha256


def test_manifest_exists():
    path = Path("artifacts/regions/US_region_masks_manifest.json")
    assert path.exists(), f"Manifest not found: {path}"


def test_manifest_has_required_fields():
    m = load_region_manifest()
    required = ["canonical_nc", "fast_tensor_pt", "region_id_mapping",
                "manifest_version", "protocol_version", "country"]
    for field in required:
        assert field in m, f"Missing required field: {field}"


def test_manifest_paths_exist():
    m = load_region_manifest()
    for key in ["canonical_nc", "fast_tensor_pt"]:
        p = Path(m[key]["path"])
        assert p.exists(), f"Manifest references missing file: {p}"


def test_manifest_sha256_matches_actual_file():
    m = load_region_manifest()
    for key in ["canonical_nc", "fast_tensor_pt"]:
        stored_sha = m[key]["sha256"]
        actual_sha = compute_sha256(Path(m[key]["path"]))
        assert stored_sha == actual_sha, \
            f"{key} SHA256 mismatch: manifest={stored_sha}, actual={actual_sha}"


def test_manifest_region_id_mapping_complete():
    m = load_region_manifest()
    mapping = m["region_id_mapping"]
    expected_ids = {"0", "1", "2", "3", "4", "5", "6"}
    assert set(mapping.keys()) == expected_ids, \
        f"Expected keys {expected_ids}, got {set(mapping.keys())}"


def test_manifest_canonical_nc_has_all_fields():
    m = load_region_manifest()
    nc = m["canonical_nc"]
    required = ["path", "variable_name", "shape", "dtype", "sha256", "unique_region_ids"]
    for field in required:
        assert field in nc, f"canonical_nc missing field: {field}"


def test_manifest_fast_tensor_pt_has_all_fields():
    m = load_region_manifest()
    pt = m["fast_tensor_pt"]
    required = ["path", "shape", "dtype", "sha256"]
    for field in required:
        assert field in pt, f"fast_tensor_pt missing field: {field}"