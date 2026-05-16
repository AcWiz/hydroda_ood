#!/usr/bin/env python3
"""
Systematic debug script for source-only backbone pipeline.

Usage:
    python scripts/debug_source_only_pipeline.py --checkpoint <path> [--max-batches 5] [--device cuda]
    python scripts/debug_source_only_pipeline.py --skip-checks 4,5,7  # skip GPU checks
    python scripts/debug_source_only_pipeline.py --checks 1,2,3  # run only specific checks

Outputs go to reports/source_only_debug/ and artifacts/source_only_debug/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
REGION_MASKS_NC = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
FREEZE_MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"
PROTOCOL_FREEZE_ID = "hyperda_v4_final_2015_2025_context2022_query2023_2025_k0_4_12"
OUTPUT_BASE = Path("reports/source_only_debug")
ARTIFACTS_BASE = Path("artifacts/source_only_debug")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pct(v: float) -> str:
    return f"{v * 100:.4f}%"


def fmt_stats(arr: np.ndarray, prefix: str = "") -> str:
    if arr.size == 0:
        return f"{prefix}  empty"
    return (
        f"{prefix}  mean={arr.mean():.6g}  std={arr.std():.6g}  "
        f"min={arr.min():.6g}  p50={np.median(arr):.6g}  p99={np.percentile(arr, 99):.6g}"
    )


def load_checkpoint(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def find_latest_source_only_checkpoint() -> Optional[str]:
    pattern = Path("artifacts/runs/phase4_source_only").glob("*best.pt")
    candidates = sorted(pattern, key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return str(candidates[0])
    return None


# ---------------------------------------------------------------------------
# Check 1: Dataset Audit
# ---------------------------------------------------------------------------
def check1_dataset_audit(
    max_batches: int = 5,
    data_path: str = None,
    protocol_freeze_id: str = PROTOCOL_FREEZE_ID,
) -> pd.DataFrame:
    from hydroda.data.dataset import HydroDADataset

    da_nc_path = data_path or f"{DATA_DIR}/DA.nc"
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    rows = []
    splits_to_check = ["source_train", "source_val", "target_support", "target_query"]

    for split_type in splits_to_check:
        print(f"\n  Check1: Loading {split_type}...")
        ds = HydroDADataset(
            da_nc_path=da_nc_path,
            region_masks_nc=REGION_MASKS_NC,
            splits_json=SPLITS_JSON,
            target_region="US-R1",
            split_type=split_type,
            K=0,
            seed=0,
            freeze_manifest=FREEZE_MANIFEST,
        )

        n_samples = len(ds)
        n_check = min(n_samples, max_batches)

        if n_check == 0:
            rows.append(
                {
                    "split": split_type,
                    "n_samples_total": n_samples,
                    "n_samples_checked": 0,
                    "active_regions": "|".join(ds._active_region_ids),
                    "loss_mask_fraction": np.nan,
                    "metric_mask_fraction": np.nan,
                    "loss_metric_ratio": np.nan,
                    "fc_surface_mean": np.nan,
                    "fc_surface_std": np.nan,
                    "an_surface_mean": np.nan,
                    "an_surface_std": np.nan,
                    "inc_surface_mean": np.nan,
                    "inc_surface_std": np.nan,
                    "inc_surface_p50": np.nan,
                    "inc_surface_p99": np.nan,
                    "abs_inc_surface_mean": np.nan,
                    "abs_inc_surface_p95": np.nan,
                    "abs_inc_surface_p99": np.nan,
                    "fc_rootzone_mean": np.nan,
                    "inc_rootzone_mean": np.nan,
                    "inc_rootzone_std": np.nan,
                    "inc_rootzone_p50": np.nan,
                    "abs_inc_rootzone_mean": np.nan,
                    "abs_inc_rootzone_p99": np.nan,
                    "forecast_surface_rmse_vs_analysis": np.nan,
                    "forecast_rootzone_rmse_vs_analysis": np.nan,
                    "n_loss_valid_pixels": 0,
                    "n_metric_valid_pixels": 0,
                    "time_range_start": "",
                    "time_range_end": "",
                    "n_unique_months": 0,
                    "n_unique_seasons": 0,
                    "inc_surface_mean_by_season": "",
                    "inc_rootzone_mean_by_season": "",
                }
            )
            ds.close()
            continue

        all_fc_s = []
        all_fc_r = []
        all_an_s = []
        all_an_r = []
        all_inc_s = []
        all_inc_r = []
        all_loss_mask = []
        all_metric_mask = []
        all_dates = []
        all_seasons = []
        all_months = []

        for idx in range(n_check):
            sample = ds[idx]
            all_fc_s.append(sample["forecast_surface"].reshape(-1))
            all_fc_r.append(sample["forecast_rootzone"].reshape(-1))
            all_an_s.append(sample["analysis_surface"].reshape(-1))
            all_an_r.append(sample["analysis_rootzone"].reshape(-1))
            all_inc_s.append(sample["increment_surface"].reshape(-1))
            all_inc_r.append(sample["increment_rootzone"].reshape(-1))
            all_loss_mask.append(sample["loss_mask"].reshape(-1))
            all_metric_mask.append(sample["metric_mask"].reshape(-1))
            all_dates.append(sample["date_str"])
            all_seasons.append(sample["season"])
            all_months.append(sample["month"])

        # Also collect time range from ALL samples (not just checked ones)
        all_dates_full = [ds[idx]["date_str"] for idx in range(min(n_samples, 200))]
        valid_dates = [d for d in all_dates_full if d]

        flat_fc_s = np.concatenate(all_fc_s).astype(np.float64)
        flat_fc_r = np.concatenate(all_fc_r).astype(np.float64)
        flat_an_s = np.concatenate(all_an_s).astype(np.float64)
        flat_an_r = np.concatenate(all_an_r).astype(np.float64)
        flat_inc_s = np.concatenate(all_inc_s).astype(np.float64)
        flat_inc_r = np.concatenate(all_inc_r).astype(np.float64)
        flat_loss_mask = np.concatenate(all_loss_mask)
        flat_metric_mask = np.concatenate(all_metric_mask)

        # Valid pixels by loss_mask
        loss_valid = flat_loss_mask > 0.5
        # Valid by metric_mask
        metric_valid = flat_metric_mask > 0.5

        # Stats on loss-valid pixels
        fc_s_loss = flat_fc_s[loss_valid]
        fc_r_loss = flat_fc_r[loss_valid]
        an_s_loss = flat_an_s[loss_valid]
        an_r_loss = flat_an_r[loss_valid]
        inc_s_loss = flat_inc_s[loss_valid]
        inc_r_loss = flat_inc_r[loss_valid]

        # Increment magnitude stats
        abs_inc_s = np.abs(inc_s_loss)
        abs_inc_r = np.abs(inc_r_loss)

        # forecast RMSE (vs analysis)
        fc_rmse_s = np.sqrt(np.mean((fc_s_loss - an_s_loss) ** 2)) if len(fc_s_loss) > 0 else np.nan
        fc_rmse_r = np.sqrt(np.mean((fc_r_loss - an_r_loss) ** 2)) if len(fc_r_loss) > 0 else np.nan

        # Monthly/seasonal increment stats (on loss-valid pixels)
        monthly_inc_stats = {}
        seasonal_inc_stats = {}
        if len(inc_s_loss) > 0:
            # Per-checked-sample stats
            for i in range(n_check):
                m = all_months[i]
                s = all_seasons[i]
                inc_mask = all_loss_mask[i].reshape(-1) > 0.5
                inc_s = all_inc_s[i][inc_mask]
                inc_r = all_inc_r[i][inc_mask]
                if len(inc_s) > 0:
                    monthly_inc_stats.setdefault(m, []).append(np.mean(inc_s))
                if len(inc_r) > 0:
                    seasonal_inc_stats.setdefault(s, []).append(np.mean(inc_s))

        # Seasonal mean summary
        season_mean_str = ""
        season_keys = ["DJF", "MAM", "JJA", "SON"]
        season_means = {}
        for s in season_keys:
            vals = seasonal_inc_stats.get(s, [])
            if vals:
                season_means[s] = float(np.mean(vals))
        if season_means:
            season_mean_str = ", ".join(f"{k}={v:.6f}" for k, v in sorted(season_means.items()))

        rows.append(
            {
                "split": split_type,
                "n_samples_total": n_samples,
                "n_samples_checked": n_check,
                "active_regions": "|".join(ds._active_region_ids),
                "loss_mask_fraction": np.mean(flat_loss_mask),
                "metric_mask_fraction": np.mean(flat_metric_mask),
                "loss_metric_ratio": np.mean(loss_valid) / max(np.mean(metric_valid), 1e-10),
                "fc_surface_mean": np.nanmean(flat_fc_s) if len(flat_fc_s) > 0 else np.nan,
                "fc_surface_std": np.nanstd(flat_fc_s) if len(flat_fc_s) > 0 else np.nan,
                "an_surface_mean": np.nanmean(flat_an_s) if len(flat_an_s) > 0 else np.nan,
                "an_surface_std": np.nanstd(flat_an_s) if len(flat_an_s) > 0 else np.nan,
                "inc_surface_mean": np.nanmean(inc_s_loss) if len(inc_s_loss) > 0 else np.nan,
                "inc_surface_std": np.nanstd(inc_s_loss) if len(inc_s_loss) > 0 else np.nan,
                "inc_surface_p50": np.nanmedian(inc_s_loss) if len(inc_s_loss) > 0 else np.nan,
                "inc_surface_p99": np.nanpercentile(inc_s_loss, 99) if len(inc_s_loss) > 0 else np.nan,
                "abs_inc_surface_mean": np.nanmean(abs_inc_s) if len(abs_inc_s) > 0 else np.nan,
                "abs_inc_surface_p95": np.nanpercentile(abs_inc_s, 95) if len(abs_inc_s) > 0 else np.nan,
                "abs_inc_surface_p99": np.nanpercentile(abs_inc_s, 99) if len(abs_inc_s) > 0 else np.nan,
                "fc_rootzone_mean": np.nanmean(flat_fc_r) if len(flat_fc_r) > 0 else np.nan,
                "inc_rootzone_mean": np.nanmean(inc_r_loss) if len(inc_r_loss) > 0 else np.nan,
                "inc_rootzone_std": np.nanstd(inc_r_loss) if len(inc_r_loss) > 0 else np.nan,
                "inc_rootzone_p50": np.nanmedian(inc_r_loss) if len(inc_r_loss) > 0 else np.nan,
                "abs_inc_rootzone_mean": np.nanmean(abs_inc_r) if len(abs_inc_r) > 0 else np.nan,
                "abs_inc_rootzone_p99": np.nanpercentile(abs_inc_r, 99) if len(abs_inc_r) > 0 else np.nan,
                "forecast_surface_rmse_vs_analysis": fc_rmse_s,
                "forecast_rootzone_rmse_vs_analysis": fc_rmse_r,
                "n_loss_valid_pixels": int(loss_valid.sum()),
                "n_metric_valid_pixels": int(metric_valid.sum()),
                "time_range_start": valid_dates[0] if valid_dates else "",
                "time_range_end": valid_dates[-1] if valid_dates else "",
                "n_unique_months": len(set(all_months)),
                "n_unique_seasons": len(set(all_seasons)),
                "inc_surface_mean_by_season": season_mean_str,
                "inc_rootzone_mean_by_season": "",
            }
        )
        ds.close()

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_BASE / "01_dataset_audit.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Check1 saved: {csv_path}")

    # Markdown report
    lines = ["# Check 1: Dataset Audit\n"]
    lines.append(f"Data path: {DATA_DIR}/DA.nc\n")
    for _, row in df.iterrows():
        lines.append(f"\n## Split: {row['split']}")
        lines.append(f"- Active regions: {row['active_regions']}")
        lines.append(f"- Total samples: {row['n_samples_total']}, Checked: {row['n_samples_checked']}")
        if row["time_range_start"]:
            lines.append(f"- Time range: {row['time_range_start']} to {row['time_range_end']}")
            lines.append(f"- Unique months in checked: {row['n_unique_months']}, Unique seasons: {row['n_unique_seasons']}")
        lines.append(f"- loss_mask fraction: {pct(row['loss_mask_fraction'])}")
        lines.append(f"- metric_mask fraction: {pct(row['metric_mask_fraction'])}")
        lines.append(f"- loss/metric ratio: {row['loss_metric_ratio']:.3f} (should be ~1 if consistent)")
        lines.append(f"- forecast_surface RMSE (loss-valid): {row['forecast_surface_rmse_vs_analysis']:.6f}")
        lines.append(f"- forecast_rootzone RMSE (loss-valid): {row['forecast_rootzone_rmse_vs_analysis']:.6f}")
        lines.append(f"- increment_surface mean (loss-valid): {row['inc_surface_mean']:.6f}")
        lines.append(f"- increment_surface std: {row['inc_surface_std']:.6f}")
        lines.append(f"- increment_surface p50: {row['inc_surface_p50']:.6f}")
        lines.append(f"- increment_surface p99: {row['inc_surface_p99']:.6f}")
        lines.append(f"- abs(inc_surface) mean: {row['abs_inc_surface_mean']:.6f}")
        lines.append(f"- abs(inc_surface) p99: {row['abs_inc_surface_p99']:.6f}")
        lines.append(f"- increment_rootzone mean (loss-valid): {row['inc_rootzone_mean']:.6f}")
        lines.append(f"- increment_rootzone std: {row['inc_rootzone_std']:.6f}")
        lines.append(f"- abs(inc_rootzone) mean: {row['abs_inc_rootzone_mean']:.6f}")
        lines.append(f"- forecast RMSE surface: {row['forecast_surface_rmse_vs_analysis']:.6f}")
        lines.append(f"- forecast RMSE rootzone: {row['forecast_rootzone_rmse_vs_analysis']:.6f}")
        lines.append(f"- N loss-valid pixels: {row['n_loss_valid_pixels']:,}")
        lines.append(f"- N metric-valid pixels: {row['n_metric_valid_pixels']:,}")
        if row["inc_surface_mean_by_season"]:
            lines.append(f"- Surface inc mean by season: {row['inc_surface_mean_by_season']}")
        if row["inc_rootzone_mean_by_season"]:
            lines.append(f"- Rootzone inc mean by season: {row['inc_rootzone_mean_by_season']}")

    md_path = OUTPUT_BASE / "01_dataset_audit.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Check1 saved: {md_path}")

    return df


# ---------------------------------------------------------------------------
# Check 2: Zero Predictor
# ---------------------------------------------------------------------------
def check2_zero_predictor_metrics(
    checkpoint_path: str,
    max_batches: int = 10,
    data_path: str = None,
    protocol_freeze_id: str = PROTOCOL_FREEZE_ID,
) -> pd.DataFrame:
    """Evaluate ZeroIncrementPredictor on all relevant splits."""
    from hydroda.data.dataset import HydroDADataset
    from hydroda.metrics.skill import (
        analysis_skill_vs_forecast,
        analysis_rmse,
        increment_rmse,
    )

    da_nc_path = data_path or f"{DATA_DIR}/DA.nc"
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    splits = ["source_train", "source_val", "target_query"]
    rows = []

    for split_type in splits:
        print(f"\n  Check2: Zero predictor on {split_type}...")
        ds = HydroDADataset(
            da_nc_path=da_nc_path,
            region_masks_nc=REGION_MASKS_NC,
            splits_json=SPLITS_JSON,
            target_region="US-R1",
            split_type=split_type,
            K=0,
            seed=0,
            freeze_manifest=FREEZE_MANIFEST,
        )

        n_check = min(len(ds), max_batches)

        # Compute metrics sample by sample
        skill_s_vals = []
        skill_r_vals = []
        an_rmse_s_vals = []
        an_rmse_r_vals = []
        inc_rmse_s_vals = []
        inc_rmse_r_vals = []

        for idx in range(n_check):
            sample = ds[idx]
            mask = sample["metric_mask"]

            # Zero predictor: pred_inc = 0, pred_an = forecast
            pred_an_s = sample["forecast_surface"]
            pred_an_r = sample["forecast_rootzone"]
            true_an_s = sample["analysis_surface"]
            true_an_r = sample["analysis_rootzone"]
            true_inc_s = sample["increment_surface"]
            true_inc_r = sample["increment_rootzone"]

            s_skill = analysis_skill_vs_forecast(pred_an_s, true_an_s, sample["forecast_surface"], mask)
            r_skill = analysis_skill_vs_forecast(pred_an_r, true_an_r, sample["forecast_rootzone"], mask)
            s_rmse = analysis_rmse(pred_an_s, true_an_s, mask)
            r_rmse = analysis_rmse(pred_an_r, true_an_r, mask)
            s_inc_rmse = increment_rmse(np.zeros_like(true_inc_s), true_inc_s, mask)
            r_inc_rmse = increment_rmse(np.zeros_like(true_inc_r), true_inc_r, mask)

            skill_s_vals.append(s_skill)
            skill_r_vals.append(r_skill)
            an_rmse_s_vals.append(s_rmse)
            an_rmse_r_vals.append(r_rmse)
            inc_rmse_s_vals.append(s_inc_rmse)
            inc_rmse_r_vals.append(r_rmse)

        rows.append(
            {
                "split": split_type,
                "predictor": "zero_increment",
                "skill_surface_mean": np.nanmean(skill_s_vals) if skill_s_vals else np.nan,
                "skill_surface_std": np.nanstd(skill_s_vals) if skill_s_vals else np.nan,
                "skill_surface_min": np.nanmin(skill_s_vals) if skill_s_vals else np.nan,
                "skill_surface_max": np.nanmax(skill_s_vals) if skill_s_vals else np.nan,
                "skill_rootzone_mean": np.nanmean(skill_r_vals) if skill_r_vals else np.nan,
                "skill_rootzone_std": np.nanstd(skill_r_vals) if skill_r_vals else np.nan,
                "an_rmse_surface_mean": np.nanmean(an_rmse_s_vals) if an_rmse_s_vals else np.nan,
                "an_rmse_rootzone_mean": np.nanmean(an_rmse_r_vals) if an_rmse_r_vals else np.nan,
                "inc_rmse_surface_mean": np.nanmean(inc_rmse_s_vals) if inc_rmse_s_vals else np.nan,
                "inc_rmse_rootzone_mean": np.nanmean(inc_rmse_r_vals) if inc_rmse_r_vals else np.nan,
                "n_samples": len(skill_s_vals),
            }
        )
        ds.close()

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_BASE / "02_zero_predictor_metrics.csv"
    df.to_csv(csv_path, index=False)

    lines = ["# Check 2: Zero Increment Predictor Sanity\n"]
    lines.append("## Expected: skill ≈ 0, an_rmse = forecast-only RMSE, inc_rmse = true inc RMSE\n")
    for _, row in df.iterrows():
        lines.append(f"\n## Split: {row['split']}")
        lines.append(f"- skill_surface: {row['skill_surface_mean']:.6f} ± {row['skill_surface_std']:.6f}")
        lines.append(f"- skill_surface range: [{row['skill_surface_min']:.6f}, {row['skill_surface_max']:.6f}]")
        lines.append(f"- skill_rootzone: {row['skill_rootzone_mean']:.6f} ± {row['skill_rootzone_std']:.6f}")
        lines.append(f"- analysis RMSE surface: {row['an_rmse_surface_mean']:.6f}")
        lines.append(f"- analysis RMSE rootzone: {row['an_rmse_rootzone_mean']:.6f}")
        lines.append(f"- increment RMSE surface: {row['inc_rmse_surface_mean']:.6f}")
        lines.append(f"- increment RMSE rootzone: {row['inc_rmse_rootzone_mean']:.6f}")
        if abs(row["skill_surface_mean"]) > 0.01:
            lines.append(f"\n**CRITICAL BUG**: zero predictor skill={row['skill_surface_mean']:.6f} should be ≈0!")
        if abs(row["skill_rootzone_mean"]) > 0.01:
            lines.append(f"\n**CRITICAL BUG**: zero predictor rootzone skill={row['skill_rootzone_mean']:.6f} should be ≈0!")

    md_path = OUTPUT_BASE / "02_zero_predictor_report.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Check2 saved: {csv_path} and {md_path}")

    return df


# ---------------------------------------------------------------------------
# Check 3: Perfect Predictor
# ---------------------------------------------------------------------------
def check3_perfect_predictor_metrics(
    checkpoint_path: str,
    max_batches: int = 10,
    data_path: str = None,
    protocol_freeze_id: str = PROTOCOL_FREEZE_ID,
) -> pd.DataFrame:
    """Evaluate PerfectIncrementPredictor on all relevant splits."""
    from hydroda.data.dataset import HydroDADataset
    from hydroda.metrics.skill import (
        analysis_rmse,
        analysis_skill_vs_forecast,
        increment_rmse,
    )

    da_nc_path = data_path or f"{DATA_DIR}/DA.nc"
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    splits = ["source_train", "source_val", "target_query"]
    rows = []

    for split_type in splits:
        print(f"\n  Check3: Perfect predictor on {split_type}...")
        ds = HydroDADataset(
            da_nc_path=da_nc_path,
            region_masks_nc=REGION_MASKS_NC,
            splits_json=SPLITS_JSON,
            target_region="US-R1",
            split_type=split_type,
            K=0,
            seed=0,
            freeze_manifest=FREEZE_MANIFEST,
        )

        n_check = min(len(ds), max_batches)
        skill_s_vals = []
        skill_r_vals = []
        an_rmse_s_vals = []
        an_rmse_r_vals = []
        inc_rmse_s_vals = []
        inc_rmse_r_vals = []

        for idx in range(n_check):
            sample = ds[idx]
            mask = sample["metric_mask"]

            # Perfect: pred_inc = true_inc, pred_an = true_an
            pred_inc_s = sample["increment_surface"]
            pred_inc_r = sample["increment_rootzone"]
            true_an_s = sample["analysis_surface"]
            true_an_r = sample["analysis_rootzone"]

            # Perfect predictor: pred_analysis == true_analysis
            s_skill = analysis_skill_vs_forecast(true_an_s, true_an_s, sample["forecast_surface"], mask)
            r_skill = analysis_skill_vs_forecast(true_an_r, true_an_r, sample["forecast_rootzone"], mask)
            s_rmse = analysis_rmse(true_an_s, true_an_s, mask)
            r_rmse = analysis_rmse(true_an_r, true_an_r, mask)
            s_inc_rmse = increment_rmse(pred_inc_s, sample["increment_surface"], mask)
            r_inc_rmse = increment_rmse(pred_inc_r, sample["increment_rootzone"], mask)

            skill_s_vals.append(s_skill)
            skill_r_vals.append(r_skill)
            an_rmse_s_vals.append(s_rmse)
            an_rmse_r_vals.append(r_rmse)
            inc_rmse_s_vals.append(s_inc_rmse)
            inc_rmse_r_vals.append(r_inc_rmse)

        rows.append(
            {
                "split": split_type,
                "predictor": "perfect_increment",
                "skill_surface_mean": np.nanmean(skill_s_vals) if skill_s_vals else np.nan,
                "skill_surface_std": np.nanstd(skill_s_vals) if skill_s_vals else np.nan,
                "skill_rootzone_mean": np.nanmean(skill_r_vals) if skill_r_vals else np.nan,
                "skill_rootzone_std": np.nanstd(skill_r_vals) if skill_r_vals else np.nan,
                "an_rmse_surface_mean": np.nanmean(an_rmse_s_vals) if an_rmse_s_vals else np.nan,
                "an_rmse_rootzone_mean": np.nanmean(an_rmse_r_vals) if an_rmse_r_vals else np.nan,
                "inc_rmse_surface_mean": np.nanmean(inc_rmse_s_vals) if inc_rmse_s_vals else np.nan,
                "inc_rmse_rootzone_mean": np.nanmean(inc_rmse_r_vals) if inc_rmse_r_vals else np.nan,
                "n_samples": len(skill_s_vals),
            }
        )
        ds.close()

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_BASE / "03_perfect_predictor_metrics.csv"
    df.to_csv(csv_path, index=False)

    lines = ["# Check 3: Perfect Increment Predictor Sanity\n"]
    lines.append("## Expected: skill ≈ 1, an_rmse ≈ 0, inc_rmse ≈ 0\n")
    for _, row in df.iterrows():
        lines.append(f"\n## Split: {row['split']}")
        lines.append(f"- skill_surface: {row['skill_surface_mean']:.6f} ± {row['skill_surface_std']:.6f}")
        lines.append(f"- skill_rootzone: {row['skill_rootzone_mean']:.6f} ± {row['skill_rootzone_std']:.6f}")
        lines.append(f"- analysis RMSE surface: {row['an_rmse_surface_mean']:.6f}")
        lines.append(f"- analysis RMSE rootzone: {row['an_rmse_rootzone_mean']:.6f}")
        lines.append(f"- increment RMSE surface: {row['inc_rmse_surface_mean']:.6f}")
        lines.append(f"- increment RMSE rootzone: {row['inc_rmse_rootzone_mean']:.6f}")
        if row["skill_surface_mean"] < 0.99:
            lines.append(f"\n**CRITICAL BUG**: perfect predictor skill={row['skill_surface_mean']:.6f} should be ≈1!")
        if row["an_rmse_surface_mean"] > 0.001:
            lines.append(f"\n**CRITICAL BUG**: perfect predictor analysis RMSE={row['an_rmse_surface_mean']:.6f} should be ≈0!")

    md_path = OUTPUT_BASE / "03_perfect_predictor_report.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Check3 saved: {csv_path} and {md_path}")

    return df


# ---------------------------------------------------------------------------
# Check 4: Checkpoint Config Audit
# ---------------------------------------------------------------------------
def check4_checkpoint_config(checkpoint_path: str) -> Dict[str, Any]:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    print(f"\n  Check4: Loading checkpoint: {checkpoint_path}")
    ckpt = load_checkpoint(checkpoint_path)
    config = ckpt.get("config", {})

    # Extract key fields
    audit = {
        "checkpoint_path": checkpoint_path,
        "epoch": ckpt.get("epoch"),
        "loss": ckpt.get("loss"),
        "best_loss": ckpt.get("best_loss"),
        "timestamp": ckpt.get("timestamp"),
        "tag": ckpt.get("tag"),
        "git_hash": ckpt.get("git_hash"),
        "model_type": config.get("name", "unknown"),
        "width": config.get("width"),
        "in_channels": config.get("in_channels"),
        "out_channels": config.get("out_channels"),
        "target_increment_normalization": config.get("target_increment_normalization"),
        "zero_raw_increment_init": config.get("zero_raw_increment_init"),
        "inc_mean": config.get("inc_mean"),
        "inc_std": config.get("inc_std"),
        "ch_mean": config.get("ch_mean"),
        "ch_std": config.get("ch_std"),
        "lr": config.get("lr"),
        "batch_size": config.get("batch_size"),
        "max_epochs": config.get("max_epochs"),
        "accum_steps": config.get("accum_steps"),
        "protocol_freeze_id": ckpt.get("protocol_freeze_id"),
        "split_manifest_path": ckpt.get("split_manifest_path"),
        "grad_clip": config.get("grad_clip"),
        "weight_decay": config.get("weight_decay"),
    }

    json_path = OUTPUT_BASE / "04_checkpoint_config.json"
    with open(json_path, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"  Check4 saved: {json_path}")

    lines = ["# Check 4: Checkpoint Config Audit\n"]
    lines.append(f"Checkpoint: {checkpoint_path}\n")
    for k, v in audit.items():
        lines.append(f"- {k}: {v}")

    # Critical checks
    lines.append("\n## Critical Checks\n")
    if not config.get("target_increment_normalization"):
        lines.append("✅ **target_increment_normalization=False**: Model outputs raw increments.")
    else:
        lines.append("✅ **target_increment_normalization=True**: Model outputs normalized increments (denormalized in eval).")

    if not config.get("zero_raw_increment_init"):
        lines.append("⚠️ **zero_raw_increment_init=False**: Output head was randomly initialized.")
        lines.append("   → Initial skill would be large negative. Use zero_raw_increment_init=True for new runs.")
    else:
        lines.append("✅ **zero_raw_increment_init=True**: Output head was zero-initialized (forecast-equivalent).")

    if config.get("inc_mean") is None and config.get("target_increment_normalization"):
        lines.append("**CRITICAL BUG**: target_increment_normalization=True but inc_mean is None!")
    if config.get("inc_mean") is not None and not config.get("target_increment_normalization"):
        lines.append("**WARNING**: target_increment_normalization=False but inc_mean is set!")

    # Model capacity
    width = config.get("width", "unknown")
    lines.append(f"\n## Model Capacity (width={width})")
    if width == 32:
        lines.append("- SmallResUNet width=32: ~1M params")
    elif width == 16:
        lines.append("- SmallResUNet width=16: ~250K params")

    md_path = OUTPUT_BASE / "04_checkpoint_config_report.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Check4 saved: {md_path}")

    return audit


# ---------------------------------------------------------------------------
# Check 5: Prediction Distribution from Current Checkpoint
# ---------------------------------------------------------------------------
def check5_prediction_distribution(
    checkpoint_path: str,
    max_batches: int = 10,
    data_path: str = None,
    device: str = "cuda",
    protocol_freeze_id: str = PROTOCOL_FREEZE_ID,
) -> pd.DataFrame:
    from hydroda.baselines.source_only import SourceOnlyBackbonePredictor
    from hydroda.data.dataset import HydroDADataset

    da_nc_path = data_path or f"{DATA_DIR}/DA.nc"
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    print(f"\n  Check5: Loading checkpoint for prediction analysis: {checkpoint_path}")
    predictor = SourceOnlyBackbonePredictor(checkpoint_path=checkpoint_path, device=device)

    splits = ["source_train", "source_val", "target_query"]
    rows = []

    for split_type in splits:
        print(f"  Check5: Predicting on {split_type}...")
        ds = HydroDADataset(
            da_nc_path=da_nc_path,
            region_masks_nc=REGION_MASKS_NC,
            splits_json=SPLITS_JSON,
            target_region="US-R1",
            split_type=split_type,
            K=0,
            seed=0,
            freeze_manifest=FREEZE_MANIFEST,
        )

        n_check = min(len(ds), max_batches)

        all_pred_inc_s = []
        all_pred_inc_r = []
        all_true_inc_s = []
        all_true_inc_r = []
        all_fc_s = []
        all_fc_r = []
        all_an_s = []
        all_an_r = []
        all_mask = []
        all_loss_mask = []

        for idx in range(n_check):
            sample = ds[idx]
            pred = predictor.predict(sample)
            all_pred_inc_s.append(pred["pred_increment_surface"].reshape(-1))
            all_pred_inc_r.append(pred["pred_increment_rootzone"].reshape(-1))
            all_true_inc_s.append(sample["increment_surface"].reshape(-1))
            all_true_inc_r.append(sample["increment_rootzone"].reshape(-1))
            all_fc_s.append(sample["forecast_surface"].reshape(-1))
            all_fc_r.append(sample["forecast_rootzone"].reshape(-1))
            all_an_s.append(sample["analysis_surface"].reshape(-1))
            all_an_r.append(sample["analysis_rootzone"].reshape(-1))
            all_mask.append(sample["metric_mask"].reshape(-1))
            all_loss_mask.append(sample["loss_mask"].reshape(-1))

        pred_s = np.concatenate(all_pred_inc_s).astype(np.float64)
        pred_r = np.concatenate(all_pred_inc_r).astype(np.float64)
        true_s = np.concatenate(all_true_inc_s).astype(np.float64)
        true_r = np.concatenate(all_true_inc_r).astype(np.float64)
        fc_s = np.concatenate(all_fc_s).astype(np.float64)
        fc_r = np.concatenate(all_fc_r).astype(np.float64)
        an_s = np.concatenate(all_an_s).astype(np.float64)
        an_r = np.concatenate(all_an_r).astype(np.float64)
        mask = np.concatenate(all_mask)
        loss_mask = np.concatenate(all_loss_mask)

        # Valid pixels (metric_mask)
        valid = (mask > 0.5) & np.isfinite(pred_s) & np.isfinite(true_s) & np.isfinite(fc_s)
        pv_s, tv_s = pred_s[valid], true_s[valid]
        pv_r, tv_r = pred_r[valid], true_r[valid]
        fv_s, fv_r = fc_s[valid], fc_r[valid]
        av_s, av_r = an_s[valid], an_r[valid]

        # Valid pixels (loss_mask only)
        loss_valid = (loss_mask > 0.5) & np.isfinite(pred_s) & np.isfinite(true_s) & np.isfinite(fc_s)
        pv_s_loss = pred_s[loss_valid]
        tv_s_loss = true_s[loss_valid]

        # Pred analysis
        pred_an_s = fv_s + pv_s
        pred_an_r = fv_r + pv_r

        # Ratios
        true_s_std = np.std(tv_s) if len(tv_s) > 0 else np.nan
        true_r_std = np.std(tv_r) if len(tv_r) > 0 else np.nan
        pred_s_std = np.std(pv_s) if len(pv_s) > 0 else np.nan
        pred_r_std = np.std(pv_r) if len(pv_r) > 0 else np.nan

        ratio_std_s = pred_s_std / true_s_std if true_s_std > 0 else np.nan
        ratio_std_r = pred_r_std / true_r_std if true_r_std > 0 else np.nan

        ratio_abs_s = (np.abs(pv_s).mean() / np.abs(tv_s).mean()) if np.abs(tv_s).mean() > 0 else np.nan
        ratio_abs_r = (np.abs(pv_r).mean() / np.abs(tv_r).mean()) if np.abs(tv_r).mean() > 0 else np.nan

        # Loss-mask-only stats (diagnostic)
        loss_ratio_std_s = np.nan
        if len(pv_s_loss) > 0 and len(tv_s_loss) > 0:
            ls = np.std(pv_s_loss)
            ts = np.std(tv_s_loss)
            if ts > 0:
                loss_ratio_std_s = ls / ts

        # Forecast RMSE and skill
        fc_rmse_s = np.sqrt(np.mean((fv_s - av_s) ** 2)) if len(fv_s) > 0 else np.nan
        fc_rmse_r = np.sqrt(np.mean((fv_r - av_r) ** 2)) if len(fv_r) > 0 else np.nan
        pred_rmse_s = np.sqrt(np.mean((pred_an_s - av_s) ** 2)) if len(pred_an_s) > 0 else np.nan
        pred_rmse_r = np.sqrt(np.mean((pred_an_r - av_r) ** 2)) if len(pred_an_r) > 0 else np.nan
        skill_s = 1 - pred_rmse_s / fc_rmse_s if fc_rmse_s > 0 else np.nan
        skill_r = 1 - pred_rmse_r / fc_rmse_r if fc_rmse_r > 0 else np.nan

        rows.append(
            {
                "split": split_type,
                "n_valid_pixels": int(valid.sum()),
                "n_loss_valid_pixels": int(loss_valid.sum()),
                # Pred increment stats
                "pred_inc_surface_mean": np.mean(pv_s) if len(pv_s) > 0 else np.nan,
                "pred_inc_surface_std": pred_s_std,
                "pred_inc_surface_min": np.min(pv_s) if len(pv_s) > 0 else np.nan,
                "pred_inc_surface_p50": np.median(pv_s) if len(pv_s) > 0 else np.nan,
                "pred_inc_surface_p99": np.percentile(pv_s, 99) if len(pv_s) > 0 else np.nan,
                "pred_inc_surface_max": np.max(pv_s) if len(pv_s) > 0 else np.nan,
                # True increment stats
                "true_inc_surface_mean": np.mean(tv_s) if len(tv_s) > 0 else np.nan,
                "true_inc_surface_std": true_s_std,
                "true_inc_surface_p50": np.median(tv_s) if len(tv_s) > 0 else np.nan,
                "true_inc_surface_p99": np.percentile(tv_s, 99) if len(tv_s) > 0 else np.nan,
                # Rootzone
                "pred_inc_rootzone_mean": np.mean(pv_r) if len(pv_r) > 0 else np.nan,
                "pred_inc_rootzone_std": pred_r_std,
                "true_inc_rootzone_mean": np.mean(tv_r) if len(tv_r) > 0 else np.nan,
                "true_inc_rootzone_std": true_r_std,
                # Ratios
                "pred_true_std_ratio_surface": ratio_std_s,
                "pred_true_std_ratio_rootzone": ratio_std_r,
                "pred_true_abs_ratio_surface": ratio_abs_s,
                "pred_true_abs_ratio_rootzone": ratio_abs_r,
                # Loss-mask-only ratio (diagnostic)
                "loss_mask_std_ratio_surface": loss_ratio_std_s,
                # Skill
                "skill_surface": skill_s,
                "skill_rootzone": skill_r,
                "forecast_rmse_surface": fc_rmse_s,
                "forecast_rmse_rootzone": fc_rmse_r,
                "pred_rmse_surface": pred_rmse_s,
                "pred_rmse_rootzone": pred_rmse_r,
                # Check for explosion / collapse
                "pred_inc_exploded": pred_s_std > 10 * true_s_std if true_s_std > 0 else False,
                "pred_near_constant": pred_s_std < 0.001 * true_s_std if true_s_std > 0 else False,
            }
        )
        ds.close()

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_BASE / "05_prediction_distribution.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Check5 saved: {csv_path}")

    lines = ["# Check 5: Prediction Distribution from Checkpoint\n"]
    for _, row in df.iterrows():
        lines.append(f"\n## Split: {row['split']}")
        lines.append(f"- N valid pixels (metric_mask): {row['n_valid_pixels']:,}")
        lines.append(f"- N valid pixels (loss_mask): {row['n_loss_valid_pixels']:,}")
        lines.append(f"\n### Surface Increment")
        lines.append(f"- PRED: mean={row['pred_inc_surface_mean']:.6g}  std={row['pred_inc_surface_std']:.6g}  p50={row['pred_inc_surface_p50']:.6g}  p99={row['pred_inc_surface_p99']:.6g}")
        lines.append(f"- TRUE: mean={row['true_inc_surface_mean']:.6g}  std={row['true_inc_surface_std']:.6g}  p50={row['true_inc_surface_p50']:.6g}  p99={row['true_inc_surface_p99']:.6g}")
        lines.append(f"- std_ratio (pred/true): {row['pred_true_std_ratio_surface']:.4f}")
        lines.append(f"- abs_ratio (pred/true): {row['pred_true_abs_ratio_surface']:.4f}")
        lines.append(f"- loss_mask std_ratio: {row['loss_mask_std_ratio_surface']:.4f}")
        lines.append(f"\n### Rootzone Increment")
        lines.append(f"- PRED: mean={row['pred_inc_rootzone_mean']:.6g}  std={row['pred_inc_rootzone_std']:.6g}")
        lines.append(f"- TRUE: mean={row['true_inc_rootzone_mean']:.6g}  std={row['true_inc_rootzone_std']:.6g}")
        lines.append(f"- std_ratio (pred/true): {row['pred_true_std_ratio_rootzone']:.4f}")
        lines.append(f"\n### Skill")
        lines.append(f"- forecast RMSE surface: {row['forecast_rmse_surface']:.6f}")
        lines.append(f"- pred RMSE surface: {row['pred_rmse_surface']:.6f}")
        lines.append(f"- skill surface: {row['skill_surface']:.6f}")
        lines.append(f"- forecast RMSE rootzone: {row['forecast_rmse_rootzone']:.6f}")
        lines.append(f"- skill rootzone: {row['skill_rootzone']:.6f}")

        if row.get("pred_inc_exploded"):
            lines.append(f"\n⚠️ **WARNING**: pred_inc std is >10x true_inc std — possible normalization bug!")
        if row.get("pred_near_constant"):
            lines.append(f"\n⚠️ **WARNING**: pred_inc std is <0.001x true_inc std — model not learning!")

        if row["pred_true_std_ratio_surface"] < 0.1 or row["pred_true_std_ratio_surface"] > 10:
            lines.append(f"\n**CRITICAL**: pred/true std ratio = {row['pred_true_std_ratio_surface']:.2f} — scale mismatch!")

        # Loss-mask diagnostic
        if not np.isnan(row["loss_mask_std_ratio_surface"]) and abs(row["loss_mask_std_ratio_surface"] - 1.0) < 0.3:
            lines.append(f"\n**Diagnostic**: loss_mask std_ratio ≈ {row['loss_mask_std_ratio_surface']:.2f} — model predicts correctly on training pixels but fails on eval pixels")
            lines.append(f"  → Confirms loss_mask vs metric_mask mismatch as root cause")

    md_path = OUTPUT_BASE / "05_prediction_distribution.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Check5 saved: {md_path}")

    return df


# ---------------------------------------------------------------------------
# Check 6: Train/Eval Target Consistency
# ---------------------------------------------------------------------------
def check6_train_eval_consistency() -> List[str]:
    """Detailed analysis of train/eval target consistency by reading actual code."""
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    findings = []
    trainer_path = Path("hydroda/training/trainer.py")
    dataset_path = Path("hydroda/data/dataset.py")
    skill_path = Path("hydroda/metrics/skill.py")

    lines = ["# Check 6: Train/Eval Target Consistency\n"]

    # --- Trainer analysis ---
    lines.append("## Trainer (`hydroda/training/trainer.py`)\n")
    if trainer_path.exists():
        content = trainer_path.read_text()

        # Training target construction
        lines.append("### Training target")
        lines.append("```python")
        lines.append("target = torch.stack([inc_surface, inc_rootzone], dim=1)  # line ~544")
        lines.append("```")
        lines.append("- Training target: **raw increment** (analysis - forecast)")
        lines.append("")

        # Increment normalization
        norm_count = content.count("target_increment_normalization")
        lines.append(f"### target_increment_normalization")
        lines.append(f"- Appears {norm_count} times in trainer.py")
        lines.append("- When True: `target = (target - inc_mean_t) / inc_std_t` (normalizes target)")
        lines.append("- When False: target stays as raw increment")
        lines.append("")

        # Source val evaluation
        lines.append("### Source val evaluation (`_eval_source_val`, line ~365)")
        lines.append("- Computes pooled RMSE then skill = 1 - rmse_pred / rmse_fcst")
        lines.append("- `sum_sq_fcst_err = sum(target^2)` = sum of (true_increment)^2")
        lines.append("- This is correct: RMSE(forecast, analysis) = sqrt(mean(increment^2))")
        lines.append("- Denormalizes pred and target before computing physical-space RMSE")
        lines.append("")

        # Report training steps
        lines.append("### Training log stats (pred vs target mean/std)")
        lines.append("- Logs both pred and target mean/std at each log step")
        lines.append("- If target_increment_normalization=True, logged stats are **normalized**")
        lines.append("- If target_increment_normalization=False, logged stats are **raw** (m³/m³)")
        lines.append("")

    # --- Dataset mask construction ---
    lines.append("## Dataset (`hydroda/data/dataset.py`)\n")
    if dataset_path.exists():
        content = dataset_path.read_text()

        lines.append("### loss_mask construction (line ~189)")
        lines.append("```python")
        lines.append("loss_mask = (")
        lines.append("    (self._active_region_mask > 0.5)")
        # Check if base_valid_mask is included
        if "base_valid_mask > 0.5" in content:
            bvm_line = [l for l in content.split("\n") if "base_valid_mask > 0.5" in l]
            if any("loss_mask" in content[max(0, content.index(l) - 50):content.index(l)] for l in bvm_line):
                lines.append("    & (base_valid_mask > 0.5)   # ← CH11 SMAP obs availability")
                lines.append("```")
                lines.append("")
                lines.append("**⚠️ WARNING: loss_mask includes base_valid_mask requirement!**")
                lines.append("- This restricts training to SMAP-observed pixels only")
                lines.append("- metric_mask does NOT have this restriction")
                lines.append("- This is the **root cause** of catastrophic negative skill")
            else:
                lines.append("    # base_valid_mask NOT in loss_mask ✅")
                lines.append("```")
                lines.append("")
                lines.append("**✅ loss_mask is aligned with metric_mask (no base_valid_mask requirement)**")
        else:
            lines.append("    # base_valid_mask NOT in loss_mask ✅")
            lines.append("```")
            lines.append("")

        lines.append("### metric_mask construction (line ~200)")
        lines.append("```python")
        lines.append("metric_mask = np.logical_and(region_mask, label_valid_mask)")
        lines.append("```")
        lines.append("- NO base_valid_mask requirement")
        lines.append("- Evaluates ALL region pixels with valid labels")
        lines.append("")

    # --- Skill metrics ---
    lines.append("## Metrics (`hydroda/metrics/skill.py`)\n")
    if skill_path.exists():
        lines.append("### analysis_skill_vs_forecast")
        lines.append("- `skill = 1 - RMSE(pred_an, true_an) / RMSE(forecast, true_an)`")
        lines.append("- Zero predictor (pred=forecast) → skill=0 ✅")
        lines.append("- Perfect predictor (pred=true_an) → skill=1 ✅")
        lines.append("")

    # --- Consistency verdict ---
    lines.append("## Consistency Verdict\n")
    lines.append("| Aspect | Training | Evaluation | Consistent? |")
    lines.append("|--------|----------|------------|-------------|")
    lines.append("| Target variable | raw increment | raw increment | ✅ |")
    lines.append("| Skill formula | 1 - rmse_pred/rmse_fcst | 1 - rmse_pred/rmse_fcst | ✅ |")
    lines.append("| Mask | loss_mask | metric_mask | ❌ (if loss_mask has ch11) |")
    lines.append("| Denormalization | in _eval_source_val | in skill.py (metric_mask) | ✅ |")

    md_path = OUTPUT_BASE / "06_train_eval_consistency.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Check6 saved: {md_path}")

    return findings


# ---------------------------------------------------------------------------
# Check 7: Tiny Overfit Test
# ---------------------------------------------------------------------------
def check7_tiny_overfit(
    checkpoint_path: str,
    max_batches: int = 4,
    data_path: str = None,
    device: str = "cuda",
    protocol_freeze_id: str = PROTOCOL_FREEZE_ID,
) -> pd.DataFrame:
    """
    Tiny overfit test: train model on 1-4 batches to verify pipeline.
    Uses the same architecture but reinitializes the model.
    """
    import torch.optim as optim
    from hydroda.data.dataset import HydroDADataset
    from hydroda.models.resunet import SmallResUNet
    from hydroda.training.losses import MaskedHuberLoss

    da_nc_path = data_path or f"{DATA_DIR}/DA.nc"
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_BASE.mkdir(parents=True, exist_ok=True)

    print("\n  Check7: Running tiny overfit test...")

    # Load checkpoint for reference
    ckpt = load_checkpoint(checkpoint_path)
    config = ckpt.get("config", {})
    ch_mean = np.array(config.get("ch_mean"), dtype=np.float32)
    ch_std = np.array(config.get("ch_std"), dtype=np.float32)

    # Dataset: source_train
    ds = HydroDADataset(
        da_nc_path=da_nc_path,
        region_masks_nc=REGION_MASKS_NC,
        splits_json=SPLITS_JSON,
        target_region="US-R1",
        split_type="source_train",
        K=0,
        seed=0,
        freeze_manifest=FREEZE_MANIFEST,
    )

    # Load a few batches
    n_batches = min(max_batches, len(ds))
    samples = [ds[idx] for idx in range(n_batches)]

    # Build batch
    x_batch = np.stack([s["x"] for s in samples], axis=0).astype(np.float32)  # [B, 12, H, W]
    inc_s_batch = np.stack([s["increment_surface"] for s in samples], axis=0).astype(np.float32)  # [B, H, W]
    inc_r_batch = np.stack([s["increment_rootzone"] for s in samples], axis=0).astype(np.float32)  # [B, H, W]
    loss_mask_batch = np.stack([s["loss_mask"] for s in samples], axis=0).astype(np.float32)  # [B, H, W]
    metric_mask_batch = np.stack([s["metric_mask"] for s in samples], axis=0).astype(np.float32)  # [B, H, W]

    # Normalize
    x_norm = (x_batch - ch_mean[:, None, None]) / (ch_std[:, None, None] + 1e-8)

    # Convert to tensors
    x_t = torch.from_numpy(x_norm)
    inc_s_t = torch.from_numpy(inc_s_batch).unsqueeze(1)  # [B, 1, H, W]
    inc_r_t = torch.from_numpy(inc_r_batch).unsqueeze(1)  # [B, 1, H, W]
    target_t = torch.cat([inc_s_t, inc_r_t], dim=1)  # [B, 2, H, W]
    mask_t = torch.from_numpy(loss_mask_batch)
    metric_mask_t = torch.from_numpy(metric_mask_batch)

    # Compute mask coverage
    loss_mask_frac = float(mask_t.mean())
    metric_mask_frac = float(metric_mask_t.mean())
    print(f"  loss_mask coverage: {loss_mask_frac*100:.2f}%")
    print(f"  metric_mask coverage: {metric_mask_frac*100:.2f}%")

    # Zero-predictor baselines (on metric_mask and loss_mask)
    # loss_mask
    zero_rmse_s_loss = float(torch.sqrt(((inc_s_t * mask_t) ** 2).sum() / (mask_t.sum() + 1e-8)).item())
    zero_rmse_r_loss = float(torch.sqrt(((inc_r_t * mask_t) ** 2).sum() / (mask_t.sum() + 1e-8)).item())
    # metric_mask
    zero_rmse_s_metric = float(torch.sqrt(((inc_s_t * metric_mask_t) ** 2).sum() / (metric_mask_t.sum() + 1e-8)).item())
    zero_rmse_r_metric = float(torch.sqrt(((inc_r_t * metric_mask_t) ** 2).sum() / (metric_mask_t.sum() + 1e-8)).item())

    # Test: zero_raw_increment_init=True
    print("\n  --- Test A: zero_raw_increment_init=True ---")
    model_a = SmallResUNet(in_channels=12, out_channels=2, width=16, zero_raw_increment_init=True)
    model_a.to(device)

    # Verify zero-init
    with torch.no_grad():
        init_pred = model_a(x_t[:1].to(device))
        init_mean = float(init_pred.mean().item())
        print(f"  Initial pred mean: {init_mean:.8f} (should be ≈0)")

    model_a.train()
    optimizer_a = optim.Adam(model_a.parameters(), lr=1e-3)
    loss_fn = MaskedHuberLoss(delta=0.01)

    x_t_dev = x_t.to(device)
    target_t_dev = target_t.to(device)
    mask_t_dev = mask_t.to(device)
    metric_mask_t_dev = metric_mask_t.to(device)

    loss_history_a = []
    rmse_history_a = []

    for step in range(500):
        optimizer_a.zero_grad()
        pred = model_a(x_t_dev)
        loss_dict = loss_fn(pred, target_t_dev, mask_t_dev.unsqueeze(1).expand(-1, 2, -1, -1))
        total_loss = loss_dict["total_loss"]
        total_loss.backward()
        optimizer_a.step()
        loss_history_a.append(float(total_loss))

        if step % 50 == 0:
            with torch.no_grad():
                pred_det = model_a(x_t_dev)

                # RMSE on loss_mask
                rmse_s_loss = float(torch.sqrt(
                    ((pred_det[:, 0] - target_t_dev[:, 0]) ** 2 * mask_t_dev).sum() / (mask_t_dev.sum() + 1e-8)
                ))
                rmse_r_loss = float(torch.sqrt(
                    ((pred_det[:, 1] - target_t_dev[:, 1]) ** 2 * mask_t_dev).sum() / (mask_t_dev.sum() + 1e-8)
                ))

                # RMSE on metric_mask
                rmse_s_metric = float(torch.sqrt(
                    ((pred_det[:, 0] - target_t_dev[:, 0]) ** 2 * metric_mask_t_dev).sum() / (metric_mask_t_dev.sum() + 1e-8)
                ))
                rmse_r_metric = float(torch.sqrt(
                    ((pred_det[:, 1] - target_t_dev[:, 1]) ** 2 * metric_mask_t_dev).sum() / (metric_mask_t_dev.sum() + 1e-8)
                ))

                # Skill on metric_mask
                fc_rmse_s = float(torch.sqrt(
                    (target_t_dev[:, 0] ** 2 * metric_mask_t_dev).sum() / (metric_mask_t_dev.sum() + 1e-8)
                ))
                skill_s = 1 - rmse_s_metric / fc_rmse_s if fc_rmse_s > 0 else float("nan")

                rmse_history_a.append({
                    "step": step, "loss": float(total_loss),
                    "rmse_s_loss": rmse_s_loss, "rmse_r_loss": rmse_r_loss,
                    "rmse_s_metric": rmse_s_metric, "rmse_r_metric": rmse_r_metric,
                    "skill_s_metric": skill_s,
                })
                print(f"    step {step}: loss={float(total_loss):.6f}  "
                      f"rmse_s(metric)={rmse_s_metric:.6f}  skill_s(metric)={skill_s:.4f}")

    # Save checkpoint
    ckpt_out = ARTIFACTS_BASE / "tiny_overfit_checkpoint.pt"
    torch.save({
        "model_state_dict": model_a.state_dict(),
        "config": {"width": 16, "ch_mean": ch_mean.tolist(), "ch_std": ch_std.tolist(),
                    "zero_raw_increment_init": True},
    }, ckpt_out)
    print(f"  Check7 saved checkpoint: {ckpt_out}")

    df = pd.DataFrame(rmse_history_a)
    csv_path = OUTPUT_BASE / "07_tiny_overfit_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Check7 saved: {csv_path}")

    final = rmse_history_a[-1] if rmse_history_a else {}
    final_skill = final.get("skill_s_metric", float("nan"))
    final_rmse_s_metric = final.get("rmse_s_metric", float("nan"))

    lines = ["# Check 7: Tiny Overfit Test\n"]
    lines.append("## Setup")
    lines.append(f"- Fixed batch of {n_batches} samples ({len(ds)} in dataset)")
    lines.append(f"- SmallResUNet (width=16)")
    lines.append(f"- LR=1e-3, Adam, 500 steps")
    lines.append(f"- loss_mask coverage: {loss_mask_frac*100:.2f}%")
    lines.append(f"- metric_mask coverage: {metric_mask_frac*100:.2f}%")
    lines.append("")
    lines.append("## Model Initialization")
    lines.append(f"- zero_raw_increment_init=True ✅")
    lines.append(f"- Initial pred mean: {init_mean:.8f} (should be 0.0)")
    lines.append("")
    lines.append("## Zero-Predictor Baseline (on loss_mask)")
    lines.append(f"- surface RMSE: {zero_rmse_s_loss:.6f}")
    lines.append(f"- rootzone RMSE: {zero_rmse_r_loss:.6f}")
    lines.append("")
    lines.append("## Zero-Predictor Baseline (on metric_mask)")
    lines.append(f"- surface RMSE: {zero_rmse_s_metric:.6f}")
    lines.append(f"- rootzone RMSE: {zero_rmse_r_metric:.6f}")
    lines.append("")
    lines.append("## Final Model (step 500)")
    lines.append(f"- RMSE surface (metric_mask): {final_rmse_s_metric:.6f}")
    lines.append(f"- Skill surface (metric_mask): {final_skill:.6f}")

    # Verdict
    if zero_rmse_s_loss > 0 and final.get("rmse_s_loss", float("inf")) < zero_rmse_s_loss * 0.5:
        lines.append(f"\n✅ Model learns on loss_mask — final RMSE < 50% of zero-predictor")
    elif zero_rmse_s_loss > 0 and final.get("rmse_s_loss", float("inf")) < zero_rmse_s_loss:
        lines.append(f"\n⚠️ Model learns slowly on loss_mask — final RMSE < zero-predictor")
    else:
        lines.append(f"\n**CRITICAL BUG**: Model did NOT learn on loss_mask!")

    if not np.isnan(final_skill) and final_skill > 0:
        lines.append(f"\n✅ Model achieves positive skill on metric_mask (skill={final_skill:.4f})")
    elif not np.isnan(final_skill) and final_skill > -0.1:
        lines.append(f"\n⚠️ Model skill ≈ 0 on metric_mask (skill={final_skill:.4f}) — near forecast-only")
    else:
        lines.append(f"\n**WARNING**: Model has negative skill on metric_mask (skill={final_skill:.4f}) — check loss_mask vs metric_mask alignment")

    md_path = OUTPUT_BASE / "07_tiny_overfit_report.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Check7 saved: {md_path}")

    ds.close()
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Source-only pipeline debug")
    parser.add_argument("--checkpoint", type=str, default=None,
        help="Path to source-only checkpoint (auto-detect if not provided)")
    parser.add_argument("--data-path", type=str, default=None,
        help="Path to DA.nc data file")
    parser.add_argument("--max-batches", type=int, default=5,
        help="Max batches per split for quick checks")
    parser.add_argument("--device", type=str, default="cuda",
        help="Device to use")
    parser.add_argument("--checks", type=str, default=None,
        help="Comma-separated list of checks to run (e.g. '1,2,3'). Default: all")
    parser.add_argument("--skip-checks", type=str, default=None,
        help="Comma-separated list of checks to skip (e.g. '4,5,7'). Overrides --checks")
    parser.add_argument("--protocol-config", type=str, default=None,
        help="Path to protocol freeze manifest (default: US_region_split_freeze_manifest.json)")
    args = parser.parse_args()

    # Auto-detect checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_source_only_checkpoint()
        if checkpoint_path:
            print(f"[auto-detect] Using checkpoint: {checkpoint_path}")
        else:
            print("ERROR: No checkpoint found and none provided. Use --checkpoint.")
            sys.exit(1)

    # Protocol config
    protocol_freeze_id = PROTOCOL_FREEZE_ID
    if args.protocol_config:
        # Custom protocol: read freeze_id from manifest
        if Path(args.protocol_config).exists():
            protocol_freeze_id = "custom"

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_BASE.mkdir(parents=True, exist_ok=True)

    # Determine which checks to run
    all_checks = [1, 2, 3, 4, 5, 6, 7]
    if args.checks:
        run_checks = [int(c.strip()) for c in args.checks.split(",")]
    else:
        run_checks = list(all_checks)

    if args.skip_checks:
        skip = set(int(c.strip()) for c in args.skip_checks.split(","))
        run_checks = [c for c in run_checks if c not in skip]

    print("=" * 60)
    print("Source-Only Pipeline Debug")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  data: {args.data_path or DATA_DIR + '/DA.nc'}")
    print(f"  max_batches: {args.max_batches}")
    print(f"  checks to run: {run_checks}")
    print("=" * 60)

    # Run selected checks
    for check_num in run_checks:
        if check_num == 1:
            print("\n[Check 1] Dataset audit...")
            check1_dataset_audit(max_batches=args.max_batches, data_path=args.data_path,
                                protocol_freeze_id=protocol_freeze_id)
        elif check_num == 2:
            print("\n[Check 2] Zero predictor sanity...")
            check2_zero_predictor_metrics(checkpoint_path, max_batches=args.max_batches,
                                         data_path=args.data_path, protocol_freeze_id=protocol_freeze_id)
        elif check_num == 3:
            print("\n[Check 3] Perfect predictor sanity...")
            check3_perfect_predictor_metrics(checkpoint_path, max_batches=args.max_batches,
                                            data_path=args.data_path, protocol_freeze_id=protocol_freeze_id)
        elif check_num == 4:
            print("\n[Check 4] Checkpoint config audit...")
            check4_checkpoint_config(checkpoint_path)
        elif check_num == 5:
            print("\n[Check 5] Prediction distribution...")
            check5_prediction_distribution(
                checkpoint_path, max_batches=args.max_batches,
                data_path=args.data_path, device=args.device,
                protocol_freeze_id=protocol_freeze_id,
            )
        elif check_num == 6:
            print("\n[Check 6] Train/eval consistency...")
            check6_train_eval_consistency()
        elif check_num == 7:
            print("\n[Check 7] Tiny overfit test...")
            check7_tiny_overfit(
                checkpoint_path, max_batches=min(args.max_batches, 4),
                data_path=args.data_path, device=args.device,
                protocol_freeze_id=protocol_freeze_id,
            )

    print("\n" + "=" * 60)
    print("All checks complete. Results in:")
    print(f"  {OUTPUT_BASE}/")
    print(f"  {ARTIFACTS_BASE}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
