"""Fast forecast-only evaluation for Phase 3 pipeline verification.

No-leakage declaration:
    - ForecastBaseline: no fit(), no training, no target label access
    - Dataset: split from frozen manifest, no query label in normalization
    - Evaluator: query labels used only as evaluation labels post-prediction
    - mask coverage audit: computes obs_mask vs label_valid_mask per region

Usage:
    PYTHONPATH=. python scripts/run_forecast_only_eval.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
REGION_MASKS = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"
OUT_DIR = Path("artifacts/results/phase3_forecast_only")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Limit samples per region for fast pipeline verification
MAX_SAMPLES_PER_REGION = 100


def compute_sample_mask_coverage(sample: Dict[str, Any]) -> Dict[str, float]:
    """Compute per-sample mask coverage fractions."""
    region = (sample["active_region_mask"] > 0.5).astype(np.float32)
    obs = (sample["base_valid_mask"] > 0.5).astype(np.float32)
    label_valid = sample["label_valid_mask"]

    region_px = int(region.sum())
    if region_px == 0:
        return {"label_valid_fraction": 0.0, "obs_fraction": 0.0}
    return {
        "label_valid_fraction": float(label_valid.sum()) / float(region_px),
        "obs_fraction": float(obs.sum()) / float(region_px),
    }


def audit_mask_coverage(
    region: str,
    split_type: str = "target_query",
    K: int = 4,
    seed: int = 0,
    max_samples: int = 100,
) -> Dict[str, Any]:
    """Audit mask coverage for a region across samples."""
    from hydroda.data.dataset import HydroDADataset
    from hydroda.data.masks import compute_mask_coverage, derive_label_valid_mask, derive_obs_mask, derive_region_mask

    try:
        ds = HydroDADataset(
            da_nc_path=f"{DATA_DIR}/DA.nc",
            region_masks_nc=REGION_MASKS,
            splits_json=SPLITS_JSON,
            target_region=region,
            split_type=split_type,
            K=K, seed=seed,
            freeze_manifest=MANIFEST,
        )
    except Exception as e:
        return {"region": region, "error": str(e), "n_samples": 0}

    n = min(len(ds), max_samples)
    per_sample = []
    for i in range(n):
        sample = ds[i]
        label_valid = derive_label_valid_mask(sample)
        obs = derive_obs_mask(sample)
        region_mask = derive_region_mask(sample)
        cov = compute_mask_coverage(label_valid, obs, region_mask)
        per_sample.append(cov)

    ds.close()

    if not per_sample:
        return {"region": region, "n_samples": 0, "error": "no samples"}

    label_valid_means = [s["label_valid_fraction"] for s in per_sample]
    obs_means = [s["obs_fraction"] for s in per_sample]

    result = {
        "region": region,
        "n_samples": n,
        "label_valid_mean": float(np.mean(label_valid_means)),
        "label_valid_std": float(np.std(label_valid_means)),
        "obs_mean": float(np.mean(obs_means)),
        "obs_std": float(np.std(obs_means)),
        "has_any_label_valid": any(v > 0 for v in label_valid_means),
        "per_sample": per_sample,
    }

    # Identify invalid regions (label_valid_mask all zeros)
    if all(v == 0 for v in label_valid_means):
        result["invalid_region"] = True
        result["note"] = "All label_valid_mask pixels are zero across all samples"

    return result


def quick_diagnostic(region: str, split_type: str = "target_query", K: int = 4, seed: int = 0):
    """Quick diagnostic without running full evaluation."""
    from hydroda.data.dataset import HydroDADataset

    try:
        ds = HydroDADataset(
            da_nc_path=f"{DATA_DIR}/DA.nc",
            region_masks_nc=REGION_MASKS,
            splits_json=SPLITS_JSON,
            target_region=region,
            split_type=split_type,
            K=K, seed=seed,
            freeze_manifest=MANIFEST,
        )
    except Exception as e:
        return {"region": region, "error": str(e), "n_samples": 0, "n_valid_pixels_total": 0}

    n = min(len(ds), MAX_SAMPLES_PER_REGION)
    total_valid = 0
    sample_dates = []

    for i in range(n):
        sample = ds[i]
        n_valid = int((sample["metric_mask"] > 0.5).sum())
        total_valid += n_valid
        if i == 0:
            sample_dates.append(sample.get("date_str", ""))

    diag = {
        "region": region,
        "split_type": split_type,
        "K": K,
        "seed": seed,
        "n_samples_total": len(ds),
        "n_samples_evaluated": n,
        "n_valid_pixels_total": total_valid,
        "has_any_valid_pixels": total_valid > 0,
        "first_date": sample_dates[0] if sample_dates else "",
        "avg_valid_per_sample": total_valid / n if n > 0 else 0,
    }
    ds.close()
    return diag


def run_forecast_only_fast(region: str, split_type: str = "target_query", K: int = 4, seed: int = 0, max_samples: int = 50):
    """Run forecast-only on first max_samples for quick verification."""
    from hydroda.baselines.forecast import ForecastBaseline
    from hydroda.data.dataset import HydroDADataset
    from hydroda.evaluation.harness import evaluate_split

    try:
        ds = HydroDADataset(
            da_nc_path=f"{DATA_DIR}/DA.nc",
            region_masks_nc=REGION_MASKS,
            splits_json=SPLITS_JSON,
            target_region=region,
            split_type=split_type,
            K=K, seed=seed,
            freeze_manifest=MANIFEST,
        )
    except Exception as e:
        return [], {"region": region, "error": str(e)}

    n = min(len(ds), max_samples)
    print(f"    Loading first {n} samples...")

    predictor = ForecastBaseline()
    print(f"    Running evaluate_split on {n} samples...")

    rows = evaluate_split(
        dataset=ds,
        predictor=predictor,
        split_role=split_type,
        experiment_id=f"phase3_{region}",
        protocol_freeze_id="hyperda_v4_2015_2025_k0_4_12",
        method="forecast_only",
        split_file=SPLITS_JSON,
        mask_file=REGION_MASKS,
        preloaded=False,
    )
    ds.close()
    return rows, {"region": region, "n_rows": len(rows)}


def main():
    print("=" * 60)
    print("Phase 3: Forecast-only End-to-End Evaluation")
    print("=" * 60)

    # Phase A: Mask coverage audit (before evaluation)
    print("\n[Phase A] Mask Coverage Audit...")
    mask_audits = []
    for region in ["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"]:
        print(f"  Auditing {region}...", end=" ", flush=True)
        audit = audit_mask_coverage(region, max_samples=50)
        mask_audits.append(audit)
        lv = audit.get("label_valid_mean", 0.0)
        ov = audit.get("obs_mean", 0.0)
        invalid = audit.get("invalid_region", False)
        print(f"label_valid={lv:.3f}, obs={ov:.3f}, invalid={invalid}")

    with open(OUT_DIR / "mask_coverage_by_region.json", "w") as f:
        json.dump({a["region"]: {k: v for k, v in a.items() if k != "per_sample"} for a in mask_audits}, f, indent=2)

    # Phase B: Quick diagnostic across all regions
    print("\n[Phase B] Quick diagnostics...")
    diags = []
    for region in ["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"]:
        print(f"  {region}...", end=" ", flush=True)
        diag = quick_diagnostic(region)
        diags.append(diag)
        print(f"n={diag['n_samples_total']}, valid={diag['n_valid_pixels_total']}, has_valid={diag['has_any_valid_pixels']}")

    with open(OUT_DIR / "diagnostics.json", "w") as f:
        json.dump(diags, f, indent=2)

    any_valid = any(d.get("has_any_valid_pixels", False) for d in diags)

    if any_valid:
        print("\n[Phase C] Running full evaluation on valid regions...")
        all_rows = []
        for diag in diags:
            if not diag.get("has_any_valid_pixels", False):
                print(f"  Skipping {diag['region']} - no valid pixels")
                continue
            region = diag["region"]
            print(f"  {region}...", end=" ", flush=True)
            rows, info = run_forecast_only_fast(region, max_samples=MAX_SAMPLES_PER_REGION)
            all_rows.extend(rows)
            print(f"{len(rows)} rows")
    else:
        print("\n[Phase C] No valid pixels found in any region. Skipping full evaluation.")
        all_rows = []

    if all_rows:
        print("\n[Phase D] Saving results...")
        df = pd.DataFrame(all_rows)
        df.to_csv(OUT_DIR / "metrics_long.csv", index=False)

        by_region = df.groupby(["target_region_id", "variable", "metric"])["value"].mean().reset_index()
        by_region.to_csv(OUT_DIR / "metrics_by_region.csv", index=False)

        # Phase E: By-season mask coverage
        print("\n[Phase E] Computing by-season mask coverage...")
        mask_cov_by_season = {}
        for season in ["DJF", "MAM", "JJA", "SON"]:
            season_df = df[df.get("season", "") == season] if "season" in df.columns else pd.DataFrame()
            if season_df.empty:
                # Compute from raw samples
                season_samples = [s for s in all_rows if s.get("season") == season] if isinstance(all_rows[0], dict) else []
                if season_samples:
                    n_px = sum(1 for r in all_rows if r.get("season") == season)
                    mask_cov_by_season[season] = {"n_samples": len(season_samples), "n_pixel_records": n_px}
                else:
                    mask_cov_by_season[season] = {"n_samples": 0, "note": "No season data in results"}
            else:
                mask_cov_by_season[season] = {
                    "n_samples": len(season_df),
                    "n_pixel_records": len(season_df),
                }
        with open(OUT_DIR / "mask_coverage_by_season.json", "w") as f:
            json.dump(mask_cov_by_season, f, indent=2)

        summary = {
            "method": "forecast_only",
            "n_total_rows": len(all_rows),
            "regions_evaluated": sorted(df["target_region_id"].unique().tolist()),
            "variables": sorted(df["variable"].unique().tolist()),
            "metrics": sorted(df["metric"].unique().tolist()),
            "forecast_only_skill_mean": float(df[df["metric"] == "analysis_skill_vs_forecast"]["value"].mean()),
            "forecast_only_skill_std": float(df[df["metric"] == "analysis_skill_vs_forecast"]["value"].std()),
        }
        with open(OUT_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    else:
        summary = {
            "method": "forecast_only",
            "status": "no_valid_pixels",
            "note": "No valid pixels found for evaluation. Check mask_coverage_by_region.json",
            "diagnostics_file": str(OUT_DIR / "diagnostics.json"),
        }
        with open(OUT_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for diag in diags:
        n = diag.get("n_samples_total", 0)
        v = diag.get("n_valid_pixels_total", 0)
        pct = v / (n * 256 * 640) * 100 if n > 0 else 0
        print(f"  {diag['region']}: {n} samples, {v} valid px ({pct:.4f}%), has_valid={diag.get('has_any_valid_pixels', False)}")

    print(f"\n  Artifacts: {OUT_DIR}/")
    print(f"  - diagnostics.json")
    print(f"  - mask_coverage_by_region.json")
    print(f"  - summary.json")
    if all_rows:
        print(f"  - metrics_long.csv ({len(all_rows)} rows)")
        print(f"  - metrics_by_region.csv")


if __name__ == "__main__":
    main()