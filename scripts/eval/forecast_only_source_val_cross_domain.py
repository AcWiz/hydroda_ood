"""Forecast-only source_val cross-domain evaluation.

Evaluates forecast-only baseline for each target_region on source_val (2021),
distinguishing between:
- target_domain: samples whose sample_region_id == target_region
- source_domain: samples whose sample_region_id != target_region

Outputs: reports/source_val_cross_domain/forecast_only_source_val_cross_domain.csv
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from hydroda.data.dataset import HydroDADataset
from hydroda.evaluation.harness import evaluate_split


class ForecastBaseline:
    """Zero-increment baseline: pred_increment = 0, pred_analysis = forecast."""

    def predict(self, sample: dict) -> dict:
        forecast_surface = sample["forecast_surface"]
        forecast_rootzone = sample["forecast_rootzone"]
        return {
            "pred_increment_surface": (
                forecast_surface - forecast_surface
            ),  # zero increment
            "pred_increment_rootzone": (
                forecast_rootzone - forecast_rootzone
            ),  # zero increment
            "pred_analysis_surface": forecast_surface,
            "pred_analysis_rootzone": forecast_rootzone,
        }


def main():
    # Paths
    root = Path("/sharefiles1/fenglonghan/projects/hydroda_ood")
    da_nc_path = "/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc"
    region_masks_nc = str(root / "artifacts/regions/US_region_masks.nc")
    splits_json = str(root / "artifacts/splits/US_loro_kdate_splits.json")
    output_dir = root / "reports/source_val_cross_domain"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "forecast_only_source_val_cross_domain.csv"

    protocol_freeze_id = "v4-source-val-cross-domain"
    method = "forecast_only"

    target_regions = ["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"]

    all_rows = []

    for target_region in target_regions:
        print(f"\nEvaluating target_region={target_region}")

        dataset = HydroDADataset(
            da_nc_path=da_nc_path,
            region_masks_nc=region_masks_nc,
            splits_json=splits_json,
            target_region=target_region,
            split_type="source_val",
            K=0,
            seed=0,
            freeze_manifest=str(
                root / "artifacts/protocol/US_region_split_freeze_manifest.json"
            ),
        )

        predictor = ForecastBaseline()
        rows = evaluate_split(
            dataset=dataset,
            predictor=predictor,
            split_role="source_val",
            experiment_id=f"forecast-only-{target_region}",
            protocol_freeze_id=protocol_freeze_id,
            method=method,
            split_file=splits_json,
            mask_file=region_masks_nc,
        )
        all_rows.extend(rows)
        print(f"  Collected {len(rows)} rows for {target_region}")

    df = pd.DataFrame(all_rows)

    # Filter to rootzone only
    df_rootzone = df[df["variable"] == "rootzone"].copy()

    # Filter to specific latw metric
    metric_name = "analysis_rmse_sqrt_before_time_avg_latw"
    df_metric = df_rootzone[df_rootzone["metric"] == metric_name].copy()

    # Compute per-row domain label
    df_metric["domain"] = df_metric.apply(
        lambda r: "target_domain"
        if r.get("sample_region_id", "") == r.get("target_region_id", "")
        else "source_domain",
        axis=1,
    )

    # Aggregate: mean per target_region × domain
    agg = (
        df_metric.groupby(["target_region_id", "domain"])["value"]
        .mean()
        .reset_index()
    )
    agg.columns = ["target_region_id", "domain", "analysis_rmse_latw"]

    # Pivot to wide format
    pivot = agg.pivot(index="target_region_id", columns="domain", values="analysis_rmse_latw")
    pivot = pivot.reindex(columns=["source_domain", "target_domain"])
    pivot = pivot.reset_index()
    pivot.columns.name = None

    # Save detailed CSV (all rows)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved full rows to {output_csv}")
    print(f"Shape: {df.shape}")

    # Save summary CSV
    summary_csv = output_dir / "forecast_only_source_val_cross_domain_summary.csv"
    pivot.to_csv(summary_csv, index=False)
    print(f"Saved summary to {summary_csv}")
    print(pivot.to_string(index=False))


if __name__ == "__main__":
    main()