#!/usr/bin/env python3
"""Diagnostic script for source-only backbone checkpoint.

Computes detailed diagnostics to understand model behavior:
- True vs predicted increment distributions
- Forecast RMSE denominator distribution
- Skill aggregation comparison (global, mean-per-cycle, median-per-cycle)
- Bias by variable/region/season
- pred_std / true_std ratio
- Linear fit: pred = a * true + b
- Oracle shrinkage: skill(α * pred_inc) for α ∈ {0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0}

Usage:
    PYTHONPATH=. python scripts/diagnose_source_only_checkpoint.py \\
        --checkpoint artifacts/checkpoints/phase4_source_only/US-R1/best.pt \\
        --target_region US-R1 --K 0 --seed 0 \\
        --max_samples 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hydroda.baselines.source_only import SourceOnlyBackbonePredictor
from hydroda.data.dataset import HydroDADataset
from hydroda.metrics.skill import _valid_flat, _rmse
from hydroda.utils.device import resolve_device


DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
REGION_MASKS_NC = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
FREEZE_MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"


def _quantiles(arr: np.ndarray, q_vals) -> dict:
    """Compute quantiles of a 1D array."""
    if arr.size == 0:
        return {f"q{int(q*100)}": np.nan for q in q_vals}
    return {f"q{int(q*100)}": float(np.quantile(arr, q)) for q in q_vals}


def diagnose_increment_distributions(pred_inc, true_inc, mask, variable):
    """Diagnose increment distributions and their relationship."""
    p, t = _valid_flat(pred_inc, true_inc, mask=mask)
    if p.size < 10:
        return {}

    pred_stats = {
        "mean": float(np.mean(p)),
        "std": float(np.std(p)),
        "min": float(np.min(p)),
        "max": float(np.max(p)),
    }
    pred_stats.update(_quantiles(p, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

    true_stats = {
        "mean": float(np.mean(t)),
        "std": float(np.std(t)),
        "min": float(np.min(t)),
        "max": float(np.max(t)),
    }
    true_stats.update(_quantiles(t, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

    # Ratio of stds
    std_ratio = pred_stats["std"] / max(true_stats["std"], 1e-8)

    # Linear fit: pred = a * true + b
    if np.std(t) > 1e-8 and p.size >= 2:
        a = float(np.corrcoef(p, t)[0, 1] * pred_stats["std"] / true_stats["std"])
        b = float(np.mean(p) - a * np.mean(t))
    else:
        a, b = np.nan, np.nan

    return {
        f"{variable}_pred": pred_stats,
        f"{variable}_true": true_stats,
        f"{variable}_std_ratio": std_ratio,
        f"{variable}_linear_a": a,
        f"{variable}_linear_b": b,
    }


def diagnose_skill_aggregation(pred_analysis, true_analysis, forecast, mask, variable):
    """Compute skill with different aggregation strategies."""
    p, t, f = _valid_flat(pred_analysis, true_analysis, forecast, mask=mask)
    if p.size < 2:
        return {}

    rmse_pred = _rmse(p, t)
    rmse_fcst = _rmse(f, t)

    results = {}

    # Global skill
    if rmse_fcst > 0:
        results["global_skill"] = float(1.0 - rmse_pred / rmse_fcst)
    else:
        results["global_skill"] = np.nan

    results["rmse_pred"] = rmse_pred
    results["rmse_fcst"] = rmse_fcst

    # Per-pixel normalized error
    pixel_errors = (p - t) ** 2
    fcst_pixel_errors = (f - t) ** 2
    results["mean_pixel_error"] = float(np.mean(pixel_errors))
    results["mean_fcst_pixel_error"] = float(np.mean(fcst_pixel_errors))

    return results


def diagnose_oracle_shrinkage(pred_inc, true_analysis, forecast, mask, variable):
    """Compute analysis-space skill for scaled predictions.

    skill(α) = 1 - RMSE(forecast + α * pred_inc, true_analysis) / RMSE(forecast, true_analysis)

    This makes α=0 (forecast-only) always have skill=0, and α=1 consistent with
    evaluate_checkpoint.py's analysis_skill_vs_forecast.

    IMPORTANT: This function operates on target_query labels. The alpha that maximizes
    skill on target_query (the "oracle best alpha") is a diagnostic only — it MUST NOT
    be used for model selection, early stopping, threshold calibration, or any training
    decision. Alpha selection for the final model MUST use source_val only.
    """
    p, t_a, f = _valid_flat(pred_inc, true_analysis, forecast, mask=mask)
    if p.size < 2:
        return {}

    rmse_fcst = _rmse(f, t_a)
    if not np.isfinite(rmse_fcst) or rmse_fcst <= 0:
        return {}

    results = {}
    for alpha in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        pred_analysis = f + alpha * p
        rmse_pred = _rmse(pred_analysis, t_a)
        skill = float(1.0 - rmse_pred / rmse_fcst)
        results[f"skill_a{alpha}"] = skill

    # Assertion: α=0 must produce exactly skill=0 (forecast-only)
    skill_a0 = results.get("skill_a0.0", None)
    assert skill_a0 is not None and abs(skill_a0) < 1e-10, \
        f"α=0 skill must be exactly 0, got {skill_a0}"

    return results


def diagnose_bias_by_region(rows, variable):
    """Compute bias grouped by region/season from metric rows."""
    df_rows = [r for r in rows if r.get("variable") == variable and r.get("metric") == "increment_bias"]
    if not df_rows:
        return {}

    by_region = {}
    for r in df_rows:
        rid = r.get("target_region_id", "unknown")
        if rid not in by_region:
            by_region[rid] = []
        by_region[rid].append(r["value"])

    return {
        f"{variable}_bias_by_region": {rid: float(np.mean(vals)) for rid, vals in by_region.items()}
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnostic analysis of source-only checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--target_region", type=str, required=True)
    parser.add_argument("--K", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split_type", type=str, default="target_query")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--require_gpu", action="store_true",
        help="Exit with error if CUDA unavailable")
    args = parser.parse_args()

    # Resolve device
    device = resolve_device(args.device, require_gpu=args.require_gpu)

    print(f"=" * 60)
    print(f"Source-only Backbone Diagnostic")
    print(f"  checkpoint={args.checkpoint}")
    print(f"  target_region={args.target_region}  K={args.K}  seed={args.seed}")
    print(f"  split_type={args.split_type}  max_samples={args.max_samples}")
    print(f"  device={device}")
    print(f"=" * 60)

    # Load dataset
    print(f"\nLoading dataset ({args.split_type})...")
    dataset = HydroDADataset(
        da_nc_path=f"{DATA_DIR}/DA.nc",
        region_masks_nc=REGION_MASKS_NC,
        splits_json=SPLITS_JSON,
        target_region=args.target_region,
        split_type=args.split_type,
        K=args.K,
        seed=args.seed,
        freeze_manifest=FREEZE_MANIFEST,
    )
    n_samples = min(len(dataset), args.max_samples)
    print(f"  dataset size: {len(dataset)}, diagnosing first {n_samples}")

    # Load predictor
    print(f"\nLoading checkpoint...")
    predictor = SourceOnlyBackbonePredictor(
        checkpoint_path=args.checkpoint,
        device=str(device),
    )
    print(f"  Increment normalization: {predictor._has_inc_norm}")
    if predictor._has_inc_norm:
        print(f"    inc_mean={predictor._inc_mean}")
        print(f"    inc_std={predictor._inc_std}")

    # Preload
    all_samples = dataset.preload() if hasattr(dataset, "preload") else None

    diagnostics = {
        "checkpoint": args.checkpoint,
        "target_region": args.target_region,
        "K": args.K,
        "seed": args.seed,
        "split_type": args.split_type,
        "n_samples": n_samples,
        "has_inc_norm": predictor._has_inc_norm,
    }

    all_metrics = {"surface": {}, "rootzone": {}}
    increment_rows = {"surface": [], "rootzone": []}

    # Collect per-sample data
    for idx in range(n_samples):
        sample = all_samples[idx] if all_samples is not None else dataset[idx]
        pred = predictor.predict(sample)
        mask = sample["metric_mask"]

        for variable in ["surface", "rootzone"]:
            inc_key = f"increment_{variable}"
            pred_inc_key = f"pred_increment_{variable}"
            pred_an_key = f"pred_analysis_{variable}"

            inc_rows = diagnose_increment_distributions(
                pred[pred_inc_key], sample[inc_key], mask, variable
            )
            skill_rows = diagnose_skill_aggregation(
                pred[pred_an_key],
                sample[f"analysis_{variable}"],
                sample[f"forecast_{variable}"],
                mask,
                variable,
            )
            oracle_rows = diagnose_oracle_shrinkage(
                pred[pred_inc_key],
                sample[f"analysis_{variable}"],
                sample[f"forecast_{variable}"],
                mask,
                variable,
            )

            for d in [inc_rows, skill_rows, oracle_rows]:
                for k, v in d.items():
                    if k not in all_metrics[variable]:
                        all_metrics[variable][k] = []
                    all_metrics[variable][k].append(v)

            increment_rows[variable].append({
                "pred_inc": pred[pred_inc_key].copy(),
                "true_inc": sample[inc_key].copy(),
                "mask": mask.copy(),
            })

    # Aggregate diagnostics
    for variable in ["surface", "rootzone"]:
        diagnostics[variable] = {}
        for metric_name, values in all_metrics[variable].items():
            # Skip dict-valued metrics (e.g., pred/true stats) — they are not scalars
            valid_vals = [v for v in values if isinstance(v, (float, int)) and np.isfinite(v)]
            if valid_vals:
                diagnostics[variable][f"{metric_name}_mean"] = float(np.mean(valid_vals))
                diagnostics[variable][f"{metric_name}_std"] = float(np.std(valid_vals))
                diagnostics[variable][f"{metric_name}_median"] = float(np.median(valid_vals))
                diagnostics[variable][f"{metric_name}_min"] = float(np.min(valid_vals))
                diagnostics[variable][f"{metric_name}_max"] = float(np.max(valid_vals))
                diagnostics[variable][f"{metric_name}_n"] = len(valid_vals)

    # Forecast RMSE denominator distribution
    denom_dist = {}
    for variable in ["surface", "rootzone"]:
        fcst_rmse_vals = all_metrics[variable].get("rmse_fcst", [])
        valid = [v for v in fcst_rmse_vals if np.isfinite(v)]
        if valid:
            d = {
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid)),
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
            }
            d.update(_quantiles(np.array(valid), [0.05, 0.25, 0.5, 0.75, 0.95]))
            denom_dist[variable] = d
    diagnostics["forecast_rmse_denominator"] = denom_dist

    # Key findings summary
    diagnostics["summary"] = {}
    for variable in ["surface", "rootzone"]:
        s = diagnostics[variable]
        summary = {}
        if "global_skill_mean" in s:
            summary["global_skill"] = s["global_skill_mean"]
        if "increment_corr_mean" in s:
            summary["increment_corr"] = s["increment_corr_mean"]
        if "increment_bias_mean" in s:
            summary["increment_bias"] = s["increment_bias_mean"]
        if "rmse_pred_mean" in s and "rmse_fcst_mean" in s:
            if s["rmse_fcst_mean"] > 0:
                summary["skill_from_means"] = 1.0 - s["rmse_pred_mean"] / s["rmse_fcst_mean"]
        # Std ratio
        for k in ["pred_std_mean", "true_std_mean"]:
            if k in s:
                summary[k] = s[k]
        if "std_ratio_mean" in s:
            summary["pred_over_true_std_ratio"] = s["std_ratio_mean"]
        # Oracle best alpha — filter to raw per-sample keys (not aggregated _mean/_std/_median)
        oracle_keys = [k for k in s if k.startswith("skill_a") and k[len("skill_a"):].split("_")[0].replace(".", "").isdigit()]
        if oracle_keys:
            best_alpha = None
            best_skill = -np.inf
            for ok in oracle_keys:
                alpha_str = ok.split("_a")[1].split("_")[0]
                alpha_val = float(alpha_str)
                if s[ok] > best_skill:
                    best_skill = s[ok]
                    best_alpha = alpha_val
            if best_alpha is not None:
                summary["oracle_best_alpha"] = best_alpha
                summary["oracle_best_skill"] = best_skill

        diagnostics["summary"][variable] = summary

    dataset.close()

    # Save diagnostics
    output_dir = Path("artifacts/results/phase4_source_only") / args.target_region
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_path = output_dir / "diagnostics_detailed.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Diagnostic Summary")
    print(f"=" * 60)
    for variable in ["surface", "rootzone"]:
        s = diagnostics["summary"].get(variable, {})
        print(f"\n  {variable}:")
        for k, v in s.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
            else:
                print(f"    {k}: {v}")

    print(f"\n  Full diagnostics saved to {diag_path}")
    print(f"  has_inc_norm: {diagnostics['has_inc_norm']}")


if __name__ == "__main__":
    main()