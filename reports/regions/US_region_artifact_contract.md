# US Region Artifact Contract

**Status:** Frozen (Phase 2A complete)
**Artifact Version:** 1.0
**Date:** 2026-05-08

---

## 1. Canonical vs Fast Mirror

This project maintains two representations of US region masks:

| Artifact | Path | Type | Role |
|----------|------|------|------|
| Canonical NC | `artifacts/regions/US_region_masks.nc` | NetCDF | Geospatial map, cartography, paper figures, audit |
| Fast Mirror PT | `artifacts/regions/US_region_mask_tensor.pt` | PyTorch tensor | Training loop fast loading |

**.nc is canonical. .pt is a training mirror — not canonical.**

The NC file contains:
- `region_mask_integer`: shape (256, 640), float32, values 0..6
- `region_mask_onehot`: shape (6, 256, 640), float32 one-hot encoding
- `latitude` / `longitude`: (256, 640) coordinate arrays

The .pt file contains only the single `region_mask_integer` tensor as `torch.Tensor` (uint8).

---

## 2. Split JSON Is Canonical for Split Protocol

`artifacts/splits/US_loro_kdate_splits.json` is the canonical split artifact — it is NOT converted to .pt.

Reasons:
- Split protocol defines temporal support/query boundaries, not spatial masks.
- JSON is human-auditable and ensures split integrity is verifiable.
- Converting to .pt would add a non-standard artifact type with no training benefit.

---

## 3. Region ID Convention

| ID | Region | Regime |
|----|--------|--------|
| 0 | Outside / invalid / unlabeled | — |
| 1 | US-R1 | dryland_sparse_vegetation |
| 2 | US-R2 | semi_arid_transition |
| 3 | US-R3 | irrigated_managed_agriculture |
| 4 | US-R4 | rainfed_agriculture |
| 5 | US-R5 | humid_high_vegetation |
| 6 | US-R6 | mountain_cold_terrain_stress |

---

## 4. Manifest Fields

The manifest (`US_region_masks_manifest.json`) records:
- `canonical_nc`: path, variable, dtype, SHA256, region ids
- `fast_tensor_pt`: path, shape, dtype, SHA256
- `geolocation_source`: reference to US_latlon.nc
- `data_source`: reference to DA.nc
- `region_id_mapping`: human-readable id→name mapping
- `consistency_checks`: sanity flags (shape match, value match, range)
- `notes`: canonical vs mirror distinction

SHA256 values are computed at manifest generation time and verified on load.

---

## 5. Loading Interface

```python
from hydroda.data.region_artifacts import (
    load_region_mask_nc,      # Returns np.ndarray (256, 640) from .nc
    load_region_mask_tensor,   # Returns torch.Tensor from .pt
    load_region_mask_fast,     # Tries .pt first, falls back to .nc, returns np.ndarray
    load_region_manifest,      # Returns dict
    validate_region_mask_tensor,  # Validates shape and id range
)
```

**`load_region_mask_fast(prefer_pt=True)`** is the recommended entry point for dataset code.

---

## 6. Re-generating the .pt Mirror

If the .nc changes, re-run:

```bash
python scripts/export_region_mask_tensor.py
```

This exports `region_mask_integer` from the NC to a `.pt` tensor, computes SHA256, and writes the manifest.

---

## 7. Consistency Verification

Run the consistency tests:

```bash
python -m pytest tests/test_region_mask_nc_pt_consistency.py tests/test_region_masks_manifest.py -v
```

Tests verify:
- NC and .pt shapes match
- NC and .pt values are exactly equal
- Manifest SHA256s match actual files
- All required manifest fields are present

---

## 8. CN/AU Extension

When China and Australia region masks are generated, follow the same contract:
- Canonical NC with `region_mask_integer` + `region_mask_onehot` + lat/lon
- Fast .pt mirror tensor
- Manifest JSON with SHA256
- Update freeze manifest `artifacts` section

The loader (`region_artifacts.py`) is designed to be country-agnostic — pass explicit paths for non-US regions.

---

## 9. Artifacts Summary

| File | Description |
|------|-------------|
| `artifacts/regions/US_region_masks.nc` | Canonical NC (frozen, immutable) |
| `artifacts/regions/US_region_mask_tensor.pt` | Fast training mirror |
| `artifacts/regions/US_region_masks_manifest.json` | Artifact contract & checksums |
| `artifacts/regions/US_region_stats.json` | Region pixel counts & quality flags |
| `artifacts/protocol/US_region_split_freeze_manifest.json` | Protocol freeze (updated with new artifact paths) |
| `scripts/export_region_mask_tensor.py` | .pt generation script |
| `scripts/write_region_masks_manifest.py` | Manifest generation script |
| `hydroda/data/region_artifacts.py` | Unified loader utility |