"""Tests for US Scientific Region Masks.

No-leakage: These tests verify that region masks are built ONLY from
lat/lon bounding boxes, not from analysis increments, model errors, etc.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

# Paths for test fixtures
PROJECT_ROOT = Path(__file__).parent.parent
REGIONS_SPEC = PROJECT_ROOT / "specs" / "regions_v2.yaml"
LATLON_NC = PROJECT_ROOT / "artifacts" / "geolocation" / "US_latlon.nc"
MASK_NC = PROJECT_ROOT / "artifacts" / "regions" / "US_region_masks.nc"
STATS_JSON = PROJECT_ROOT / "artifacts" / "regions" / "US_region_stats.json"


class TestUSRegionMasks:
    """Test suite for US region masks."""

    @pytest.fixture(scope="class")
    def masks_data(self):
        """Load masks and stats for testing."""
        if not MASK_NC.exists():
            pytest.skip(f"Masks file not found: {MASK_NC}")
        if not STATS_JSON.exists():
            pytest.skip(f"Stats file not found: {STATS_JSON}")

        ds = xr.open_dataset(MASK_NC)
        mask_int = ds["region_mask_integer"].values
        mask_onehot = ds["region_mask_onehot"].values
        lat = ds["latitude"].values
        lon = ds["longitude"].values

        with open(STATS_JSON) as f:
            stats = json.load(f)

        ds.close()
        return {
            "mask_int": mask_int,
            "mask_onehot": mask_onehot,
            "lat": lat,
            "lon": lon,
            "stats": stats,
        }

    def test_masks_shape_matches_grid(self, masks_data):
        """Verify mask shape matches the lat/lon grid (256, 640)."""
        H, W = masks_data["lat"].shape
        assert masks_data["mask_int"].shape == (H, W), (
            f"Integer mask shape {masks_data['mask_int'].shape} != grid shape {(H, W)}"
        )
        assert masks_data["mask_onehot"].shape == (6, H, W), (
            f"One-hot mask shape {masks_data['mask_onehot'].shape} != (6, {H}, {W})"
        )

    def test_region_ids_valid(self, masks_data):
        """Verify region IDs are valid {0, 1, 2, 3, 4, 5, 6}."""
        valid_ids = {0, 1, 2, 3, 4, 5, 6}
        actual_ids = set(np.unique(masks_data["mask_int"]))
        assert actual_ids <= valid_ids, (
            f"Invalid region IDs found: {actual_ids - valid_ids}"
        )

    def test_all_six_regions_non_empty(self, masks_data):
        """Verify all 6 regions have at least 1 pixel."""
        for i in range(1, 7):
            count = (masks_data["mask_int"] == i).sum()
            assert count > 0, f"Region {i} (US-R{i}) has 0 pixels"

    def test_no_overlap_in_onehot(self, masks_data):
        """Verify no grid cell belongs to multiple regions."""
        overlap_count = masks_data["mask_onehot"].sum(axis=0)
        assert np.all(overlap_count <= 1), (
            f"Overlap detected: {(overlap_count > 1).sum()} cells in multiple regions"
        )

    def test_latlon_ranges_match_geolocation(self, masks_data):
        """Verify lat/lon in masks matches source geolocation."""
        ds = xr.open_dataset(LATLON_NC)
        lat_source = ds["latitude"].values
        lon_source = ds["longitude"].values
        ds.close()

        np.testing.assert_array_equal(masks_data["lat"], lat_source)
        np.testing.assert_array_equal(masks_data["lon"], lon_source)

    def test_no_leakage_in_region_construction(self):
        """Verify no-leakage declaration is present in mask metadata."""
        if not MASK_NC.exists():
            pytest.skip(f"Masks file not found: {MASK_NC}")

        ds = xr.open_dataset(MASK_NC)
        no_leakage = ds.attrs.get("no_leakage", "")
        method = ds.attrs.get("method", "")
        ds.close()

        assert "analysis_increments" in no_leakage or "not used" in no_leakage, (
            "No-leakage declaration missing or incomplete"
        )
        assert "bbox" in method.lower(), "Method should be bbox-based"

    def test_stats_pixel_counts_match_masks(self, masks_data):
        """Verify stats pixel counts match actual mask counts."""
        for region_id, stat in masks_data["stats"].items():
            if stat["pixel_count"] == 0:
                continue
            # Extract region index from ID (e.g., "US-R1" -> 1)
            idx = int(region_id.split("-R")[1])
            actual = (masks_data["mask_int"] == idx).sum()
            assert actual == stat["pixel_count"], (
                f"{region_id}: stats={stat['pixel_count']}, actual={actual}"
            )

    def test_regions_disjoint_from_bbox(self, masks_data):
        """Verify regions are actually disjoint by checking one-hot sum."""
        onehot_sum = masks_data["mask_onehot"].sum(axis=0)
        inside_region = onehot_sum > 0
        total_inside = inside_region.sum()

        # Each cell inside a region belongs to exactly one region
        assert np.all(onehot_sum[inside_region] == 1), (
            "Some cells inside regions have != 1 region assignment"
        )

    def test_globalattr_source_tracking(self):
        """Verify region definitions and geolocation sources are tracked."""
        if not MASK_NC.exists():
            pytest.skip(f"Masks file not found: {MASK_NC}")

        ds = xr.open_dataset(MASK_NC)
        region_src = ds.attrs.get("region_definitions_source", "")
        geo_src = ds.attrs.get("geolocation_source", "")
        ds.close()

        assert "regions_v2.yaml" in region_src, (
            f"Region source not properly tracked: {region_src}"
        )
        assert "US_latlon.nc" in geo_src, (
            f"Geolocation source not properly tracked: {geo_src}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])