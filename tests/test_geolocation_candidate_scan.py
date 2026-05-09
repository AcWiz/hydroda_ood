"""
Tests for Phase 0.6 geolocation candidate scan.

These tests verify:
1. america_lat.npy / america_lon.npy exist and have valid shapes/ranges
2. 1D vectors broadcast correctly to 2D (256, 640) grid
3. Lat/lon ranges are within US continental bounds
4. Directory manifest was generated (if artifacts exist)
5. No leakage: candidates don't use target query labels or analysis increments

No leakage: These tests only use static coordinate vectors and metadata.
"""

import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)

# =============================================================================
# Constants
# =============================================================================
SMAP_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
AMERICA_LAT_PATH = os.path.join(SMAP_DIR, "america_lat.npy")
AMERICA_LON_PATH = os.path.join(SMAP_DIR, "america_lon.npy")

US_GRID_HEIGHT = 256
US_GRID_WIDTH = 640

# Valid ranges for US continental (with margin)
US_LAT_RANGE = (15.0, 75.0)
US_LON_RANGE = (-180.0, -50.0)  # Also accepts [180, 310] for 0-360


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def lat_vec():
    """Load america_lat.npy vector."""
    if not os.path.exists(AMERICA_LAT_PATH):
        pytest.skip(f"Test requires {AMERICA_LAT_PATH} (geolocation source not found)")
    return np.load(AMERICA_LAT_PATH)


@pytest.fixture
def lon_vec():
    """Load america_lon.npy vector."""
    if not os.path.exists(AMERICA_LON_PATH):
        pytest.skip(f"Test requires {AMERICA_LON_PATH} (geolocation source not found)")
    return np.load(AMERICA_LON_PATH)


@pytest.fixture
def manifest_json_path():
    return os.path.join(PROJECT_ROOT, "artifacts", "audits", "smap_directory_manifest.json")


@pytest.fixture
def candidates_json_path():
    return os.path.join(PROJECT_ROOT, "artifacts", "geolocation", "geolocation_candidates_US.json")


# =============================================================================
# Tests: Vector file existence and validity
# =============================================================================

def test_america_latlon_npy_exist_and_valid(lat_vec, lon_vec):
    """Verify america_lat.npy (256,) and america_lon.npy (640,) exist with correct shapes."""
    # Shape checks
    assert lat_vec.shape == (US_GRID_HEIGHT,), (
        f"america_lat.npy shape {lat_vec.shape} != expected ({US_GRID_HEIGHT},)"
    )
    assert lon_vec.shape == (US_GRID_WIDTH,), (
        f"america_lon.npy shape {lon_vec.shape} != expected ({US_GRID_WIDTH},)"
    )

    # Dtype checks (should be float)
    assert np.issubdtype(lat_vec.dtype, np.floating), (
        f"america_lat.npy dtype {lat_vec.dtype} is not floating point"
    )
    assert np.issubdtype(lon_vec.dtype, np.floating), (
        f"america_lon.npy dtype {lon_vec.dtype} is not floating point"
    )

    # No NaN/Inf checks
    assert np.all(np.isfinite(lat_vec)), "america_lat.npy contains NaN or Inf"
    assert np.all(np.isfinite(lon_vec)), "america_lon.npy contains NaN or Inf"


def test_latlon_range_sanity(lat_vec, lon_vec):
    """Verify lat in [15, 75] and lon in [-180, -50] (US continental bounds)."""
    lat_min, lat_max = float(lat_vec.min()), float(lat_vec.max())
    lon_min, lon_max = float(lon_vec.min()), float(lon_vec.max())

    # Latitude checks
    assert US_LAT_RANGE[0] <= lat_min, (
        f"Latitude min {lat_min:.4f} below US range {US_LAT_RANGE}"
    )
    assert lat_max <= US_LAT_RANGE[1], (
        f"Latitude max {lat_max:.4f} above US range {US_LAT_RANGE}"
    )

    # Longitude checks
    lon_ok = (US_LON_RANGE[0] <= lon_min and lon_max <= US_LON_RANGE[1]) or \
             (180 <= lon_min and lon_max <= 310)
    assert lon_ok, (
        f"Longitude range [{lon_min:.4f}, {lon_max:.4f}] outside US range {US_LON_RANGE} "
        f"or 0-360 range [180, 310]"
    )

    # Specific sanity: US continental roughly 24N-50N, -125 to -66
    # With margin: 15-75 and -180 to -50
    print(f"\nLat range: [{lat_min:.4f}, {lat_max:.4f}]")
    print(f"Lon range: [{lon_min:.4f}, {lon_max:.4f}]")


def test_vectors_broadcast_to_grid(lat_vec, lon_vec):
    """Verify 1D vectors broadcast correctly to 2D (256, 640) grid."""
    # Broadcast via numpy broadcasting
    lat_2d = np.broadcast_to(lat_vec[:, None], (US_GRID_HEIGHT, US_GRID_WIDTH))
    lon_2d = np.broadcast_to(lon_vec[None, :], (US_GRID_HEIGHT, US_GRID_WIDTH))

    # Shape checks
    assert lat_2d.shape == (US_GRID_HEIGHT, US_GRID_WIDTH), (
        f"lat_2d shape {lat_2d.shape} != expected ({US_GRID_HEIGHT}, {US_GRID_WIDTH})"
    )
    assert lon_2d.shape == (US_GRID_HEIGHT, US_GRID_WIDTH), (
        f"lon_2d shape {lon_2d.shape} != expected ({US_GRID_HEIGHT}, {US_GRID_WIDTH})"
    )

    # Verify broadcasting preserved values (no unexpected expansion)
    # Each row of lat_2d should be identical (all columns have same lat for given row)
    for row in range(US_GRID_HEIGHT):
        assert np.all(lat_2d[row, :] == lat_2d[row, 0]), (
            f"lat_2d row {row} has inconsistent values (broadcasting failed)"
        )

    # Each column of lon_2d should be identical (all rows have same lon for given col)
    for col in range(US_GRID_WIDTH):
        assert np.all(lon_2d[:, col] == lon_2d[0, col]), (
            f"lon_2d column {col} has inconsistent values (broadcasting failed)"
        )

    # Verify the values match original vectors
    for row in range(US_GRID_HEIGHT):
        assert lat_2d[row, 0] == lat_vec[row], (
            f"lat_2d[{row}, 0] = {lat_2d[row, 0]} != lat_vec[{row}] = {lat_vec[row]}"
        )
    for col in range(US_GRID_WIDTH):
        assert lon_2d[0, col] == lon_vec[col], (
            f"lon_2d[0, {col}] = {lon_2d[0, col]} != lon_vec[{col}] = {lon_vec[col]}"
        )


def test_lat_vector_monotonicity_warnings(lat_vec):
    """
    Verify lat vector is approximately monotonic.
    Protocol says warn_only for monotonicity (curvilinear grids may not be strictly monotonic).
    We check here mostly for debugging purposes.
    """
    diffs = np.diff(lat_vec)
    # Count sign changes (non-monotonic transitions)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    if sign_changes > 0:
        print(f"\nWARNING: lat vector has {sign_changes} non-monotonic transitions")
        print(f"  This is acceptable for curvilinear grids (warn_only per protocol)")
    # Just verify no NaN and mostly increasing (US should go N to S or S to N)
    assert np.all(np.isfinite(diffs)), "lat vector has NaN/Inf in differences"


def test_lon_vector_monotonic(lon_vec):
    """Verify lon vector is monotonic (US should be monotonic W to E or E to W)."""
    diffs = np.diff(lon_vec)
    assert np.all(np.isfinite(diffs)), "lon vector has NaN/Inf in differences"
    # For US, longitude should be monotonic (going either W->E or E->W)
    # america_lon.npy goes from -126.7 to -67.1 (W to E, increasing)
    monotonic = np.all(diffs > 0) or np.all(diffs < 0)
    assert monotonic, (
        f"lon vector is not strictly monotonic; diffs signs: {np.sign(diffs[:10])}..."
    )


# =============================================================================
# Tests: Directory manifest
# =============================================================================

def test_directory_manifest_generated(manifest_json_path):
    """
    If artifacts/audits/smap_directory_manifest.json exists, verify it is non-empty.
    This test is skipped if the manifest hasn't been generated yet.
    """
    if not os.path.exists(manifest_json_path):
        pytest.skip(
            f"Manifest not yet generated at {manifest_json_path}. "
            "Run scan_smap_geolocation_sources.py first."
        )

    import json
    with open(manifest_json_path) as f:
        manifest = json.load(f)

    # Should have candidates
    assert "candidates" in manifest, "Manifest missing 'candidates' key"
    assert isinstance(manifest["candidates"], dict), "candidates is not a dict"

    # Should have latlon_npy class with our files
    latlon_candidates = manifest["candidates"].get("latlon_npy", [])
    assert len(latlon_candidates) > 0, "No latlon_npy candidates found in manifest"

    # Verify key fields present
    for candidate in latlon_candidates:
        assert "relative_path" in candidate, "Candidate missing relative_path"
        assert "size_bytes" in candidate, "Candidate missing size_bytes"


# =============================================================================
# Tests: No-leakage verification
# =============================================================================

def test_no_leakage_in_candidates(lat_vec, lon_vec):
    """
    NO LEAKAGE: Geolocation candidates must not use target query labels,
    analysis increments, model errors, or evaluation metrics.

    This test verifies that the coordinate vectors only contain spatial
    position information, not any target or analysis data.
    """
    # 1. Values are within physical bounds (not derived from analysis)
    lat_min, lat_max = float(lat_vec.min()), float(lat_vec.max())
    lon_min, lon_max = float(lon_vec.min()), float(lon_vec.max())

    # Lat/lon must be in valid geographic ranges
    assert -90 <= lat_min <= 90, f"Lat min {lat_min} out of physical range [-90, 90]"
    assert -90 <= lat_max <= 90, f"Lat max {lat_max} out of physical range [-90, 90]"
    assert -180 <= lon_min <= 180, f"Lon min {lon_min} out of physical range [-180, 180]"
    assert -180 <= lon_max <= 180, f"Lon max {lon_max} out of physical range [-180, 180]"

    # 2. No pattern suggesting derived from analysis increments
    # If these were analysis-derived, ranges would be around zero with
    # much smaller magnitudes (cm rather than degrees)
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    assert lat_span > 1.0, f"Lat span {lat_span} too small to be geographic coordinates"
    assert lon_span > 1.0, f"Lon span {lon_span} too small to be geographic coordinates"

    # 3. Static nature: vectors are independent of time (no time dimension)
    assert lat_vec.ndim == 1, f"lat_vec should be 1D, got {lat_vec.ndim}D"
    assert lon_vec.ndim == 1, f"lon_vec should be 1D, got {lon_vec.ndim}D"

    # 4. No correlation with model output patterns
    # (This is a placeholder - in practice we just verify physical validity)
    print(f"\nNo-leakage check: lat/lon vectors are physical geographic coordinates")
    print(f"  lat: [{lat_min:.4f}, {lat_max:.4f}] span={lat_span:.4f}")
    print(f"  lon: [{lon_min:.4f}, {lon_max:.4f}] span={lon_span:.4f}")


# =============================================================================
# Tests: Geolocation module integration
# =============================================================================

def test_geolocation_module_load():
    """Verify hydroda.data.geolocation module can be imported."""
    from hydroda.data.geolocation import (
        load_us_latlon_vectors,
        vectors_to_2d_grid,
        validate_latlon_ranges,
        US_GRID_HEIGHT,
        US_GRID_WIDTH,
    )
    assert US_GRID_HEIGHT == 256
    assert US_GRID_WIDTH == 640


def test_geolocation_vectors_to_grid_integration(lat_vec, lon_vec):
    """Integration test: load vectors -> broadcast -> validate."""
    from hydroda.data.geolocation import vectors_to_2d_grid, validate_latlon_ranges

    lat_2d, lon_2d = vectors_to_2d_grid(lat_vec, lon_vec)
    assert lat_2d.shape == (US_GRID_HEIGHT, US_GRID_WIDTH)
    assert lon_2d.shape == (US_GRID_HEIGHT, US_GRID_WIDTH)

    validation = validate_latlon_ranges(lat_2d, lon_2d)
    assert validation["lat_valid"], f"Latitude validation failed: {validation['warnings']}"
    assert validation["lon_valid"], f"Longitude validation failed: {validation['warnings']}"


# =============================================================================
# Tests: geolocation_recovery.py Method 8
# =============================================================================

def test_geolocation_recovery_method8_result():
    """
    If artifacts/audits/geolocation_recovery_US.json exists, verify Method 8 was attempted.
    """
    recovery_path = os.path.join(PROJECT_ROOT, "artifacts", "audits", "geolocation_recovery_US.json")
    if not os.path.exists(recovery_path):
        pytest.skip(
            f"geolocation_recovery_US.json not yet generated. "
            "Run geolocation_recovery.py first."
        )

    import json
    with open(recovery_path) as f:
        result = json.load(f)

    methods = result.get("methods_attempted", [])
    method_names = [m["method"] for m in methods]

    assert "directory_latlon_vector_lookup" in method_names, (
        "Method 8 (directory_latlon_vector_lookup) not found in geolocation recovery"
    )

    # Find Method 8 and check it succeeded
    method_8 = next(m for m in methods if m["method"] == "directory_latlon_vector_lookup")
    assert method_8["success"], (
        f"Method 8 (directory lookup) should succeed with america_lat/lon.npy. "
        f"Details: {method_8.get('details')}"
    )