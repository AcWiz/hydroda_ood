"""Tests for freeze manifest region artifact references."""

import json
from pathlib import Path


def load_freeze_manifest():
    with open("artifacts/protocol/US_region_split_freeze_manifest.json") as f:
        return json.load(f)


def test_freeze_manifest_has_region_masks_nc():
    m = load_freeze_manifest()
    assert "region_masks" in m["artifacts"], "Missing region_masks in artifacts"
    assert m["artifacts"]["region_masks"].endswith(".nc"), \
        f"Expected .nc path, got: {m['artifacts']['region_masks']}"


def test_freeze_manifest_has_region_mask_tensor_pt():
    m = load_freeze_manifest()
    assert "region_mask_tensor_pt" in m["artifacts"], \
        "Missing region_mask_tensor_pt in artifacts"
    assert m["artifacts"]["region_mask_tensor_pt"].endswith(".pt"), \
        f"Expected .pt path, got: {m['artifacts']['region_mask_tensor_pt']}"


def test_freeze_manifest_has_region_masks_manifest():
    m = load_freeze_manifest()
    assert "region_masks_manifest" in m["artifacts"], \
        "Missing region_masks_manifest in artifacts"
    assert m["artifacts"]["region_masks_manifest"].endswith(".json"), \
        f"Expected .json path, got: {m['artifacts']['region_masks_manifest']}"


def test_freeze_manifest_references_match_region_manifest():
    freeze = load_freeze_manifest()
    from hydroda.data.region_artifacts import load_region_manifest
    region = load_region_manifest()

    # The paths in freeze manifest should match those in region manifest
    assert freeze["artifacts"]["region_masks"] == region["canonical_nc"]["path"], \
        f"region_masks mismatch: freeze={freeze['artifacts']['region_masks']}, region={region['canonical_nc']['path']}"
    assert freeze["artifacts"]["region_mask_tensor_pt"] == region["fast_tensor_pt"]["path"], \
        f"region_mask_tensor_pt mismatch"


def test_freeze_manifest_new_fields_point_to_existing_files():
    freeze = load_freeze_manifest()
    new_fields = ["region_mask_tensor_pt", "region_masks_manifest"]
    for key in new_fields:
        path = Path(freeze["artifacts"][key])
        assert path.exists(), f"Freeze manifest references missing file: {path}"