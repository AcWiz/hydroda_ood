# US Scientific Region Mask Summary

## Metadata
- Region definitions: `specs/regions_v2.yaml`
- Geolocation source: `artifacts/geolocation/US_latlon.nc`
- Method: Bounding box polygon grid assignment
- No-leakage: analysis_increments/model_errors/target_labels NOT used

## Overlap Check
- **Passed**: Regions do not overlap

## Region Statistics

| Region ID | Name | Regime | Pixels | Fraction | Lat Range | Lon Range | Quality |
|-----------|------|--------|--------|----------|----------|----------|--------|
| US-R1 | Southwest Desert / Four Corners | dryland_sparse_vegetation | 4,485 | 0.027374 | [31.50, 36.95] | [-115.44, -109.09] | ok |
| US-R2 | Southern Great Plains | semi_arid_transition | 6,075 | 0.037079 | [32.08, 38.45] | [-103.49, -96.02] | ok |
| US-R3 | California Central Valley | irrigated_managed_agriculture | 2,666 | 0.016272 | [35.03, 40.46] | [-122.44, -118.52] | ok |
| US-R4 | Corn Belt | rainfed_agriculture | 8,064 | 0.049219 | [39.09, 44.95] | [-95.93, -84.07] | ok |
| US-R5 | Southeast US | humid_high_vegetation | 11,620 | 0.070923 | [29.54, 36.42] | [-90.98, -78.00] | ok |
| US-R6 | Central Rockies | mountain_cold_terrain_stress | 6,000 | 0.036621 | [38.01, 44.95] | [-112.45, -105.08] | ok |

## No-Leakage Declaration

Region masks were constructed **only** from:
- Fixed lat/lon bounding boxes from `specs/regions_v2.yaml`
- Lat/lon grids from `artifacts/geolocation/US_latlon.nc`

**NOT** from:
- DA.nc analysis increments
- Model prediction errors
- target_query labels
- Any training or evaluation results