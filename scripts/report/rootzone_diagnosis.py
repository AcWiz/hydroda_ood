#!/usr/bin/env python3
"""Rootzone diagnosis report for source-only backbone.

Produces a markdown report comparing forecast-only vs source-only model
performance with latitude-weighted metrics, increment distribution statistics,
and rootzone skill denominator sensitivity analysis.

Usage:
    PYTHONPATH=. python scripts/report/rootzone_diagnosis.py \\
        --checkpoint artifacts/checkpoints/phase4_source_only/best.pt \\
        --target_region US-R1 --K 0 --seed 0 \\
        --output reports/rootzone_diagnosis_US-R1.md

No-leakage declaration:
    - All model selection / checkpoint choice uses source_val.
    - target_query is evaluated ONLY for final reporting (explicitly marked).
    - No target_query labels used for early stopping, model selection,
      normalization, or threshold tuning.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from hydroda.data.dataset import HydroDADataset
from hydroda.baselines.source_only import SourceOnlyBackbonePredictor
from hydroda.metrics.skill import (
    weighted_mse,
    weighted_analysis_skill_components,
    _valid_weighted_flat,
    weighted_bias,
    weighted_corr,
)

DA_NC = "/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc"
REGION_MASKS_NC = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
FREEZE_MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"


def _aggregate_latw_skill(samples_mse: list, samples_fcst_mse: list) -> dict:
    """Aggregate per-sample MSE (model and forecast) across time, then sqrt for RMSE and skill.

    Skill_latw = 1 - sqrt(mean_model_mse) / sqrt(mean_forecast_mse)
    """
    valid_model = [v for v in samples_mse if np.isfinite(v)]
    valid_fcst = [v for v in samples_fcst_mse if np.isfinite(v)]
    if not valid_model or not valid_fcst:
        return {"rmse_model": np.nan, "rmse_forecast": np.nan, "skill": np.nan, "n_samples": 0}

    mean_model_mse = np.mean(valid_model)
    mean_forecast_mse = np.mean(valid_fcst)
    rmse_model = float(np.sqrt(mean_model_mse)) if mean_model_mse > 0 else 0.0
    rmse_forecast = float(np.sqrt(mean_forecast_mse)) if mean_forecast_mse > 0 else 0.0
    skill = float(1.0 - rmse_model / rmse_forecast) if rmse_forecast > 0 else np.nan

    return {
        "rmse_model": rmse_model,
        "rmse_forecast": rmse_forecast,
        "skill": skill,
        "n_samples": len(valid_model),
    }


def _forecast_only_metrics(dataset, variable: str) -> dict:
    """Compute forecast-only latw metrics by setting pred_inc = 0."""
    samples_mse = []
    samples_fcst_mse = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        mask = sample["metric_mask"]
        latw = sample["latitude_weight"]

        if variable == "surface":
            forecast = sample["forecast_surface"]
            analysis = sample["analysis_surface"]
        else:
            forecast = sample["forecast_rootzone"]
            analysis = sample["analysis_rootzone"]

        mse_m, mse_f = weighted_analysis_skill_components(
            pred_analysis=forecast,  # forecast-only: pred_analysis = forecast
            true_analysis=analysis,
            forecast=forecast,
            mask=mask,
            latitude_weight=latw,
        )
        samples_mse.append(mse_m)
        samples_fcst_mse.append(mse_f)

    return _aggregate_latw_skill(samples_mse, samples_fcst_mse)


def _model_metrics(dataset, predictor, variable: str) -> dict:
    """Compute model latw metrics over a dataset."""
    samples_mse = []
    samples_fcst_mse = []
    inc_mse_samples = []
    inc_bias_samples = []
    inc_abs_ratio_samples = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        pred = predictor.predict(sample)
        mask = sample["metric_mask"]
        latw = sample["latitude_weight"]

        if variable == "surface":
            fcst = sample["forecast_surface"]
            analysis = sample["analysis_surface"]
            true_inc = sample["increment_surface"]
            pred_analysis = pred["pred_analysis_surface"]
            pred_inc = pred["pred_increment_surface"]
        else:
            fcst = sample["forecast_rootzone"]
            analysis = sample["analysis_rootzone"]
            true_inc = sample["increment_rootzone"]
            pred_analysis = pred["pred_analysis_rootzone"]
            pred_inc = pred["pred_increment_rootzone"]

        mse_m, mse_f = weighted_analysis_skill_components(
            pred_analysis=pred_analysis,
            true_analysis=analysis,
            forecast=fcst,
            mask=mask,
            latitude_weight=latw,
        )
        samples_mse.append(mse_m)
        samples_fcst_mse.append(mse_f)

        inc_mse = weighted_mse(pred_inc, true_inc, mask, latw)
        inc_mse_samples.append(inc_mse)

        inc_bias = weighted_bias(pred_inc, true_inc, mask, latw)
        inc_bias_samples.append(inc_bias)

        # Amplitude ratio: |pred_inc| / |true_inc|
        try:
            p_flat2, t_flat2, w2 = _valid_weighted_flat(pred_inc, true_inc, mask=mask, latitude_weight=latw)
            if p_flat2.size > 0:
                pred_mean = np.average(np.abs(p_flat2), weights=w2)
                true_mean = np.average(np.abs(t_flat2), weights=w2)
                ratio = float(pred_mean / true_mean) if true_mean > 1e-8 else np.nan
                inc_abs_ratio_samples.append(ratio)
            else:
                inc_abs_ratio_samples.append(np.nan)
        except Exception:
            inc_abs_ratio_samples.append(np.nan)

    skill_result = _aggregate_latw_skill(samples_mse, samples_fcst_mse)

    valid_inc_mse = [v for v in inc_mse_samples if np.isfinite(v)]
    valid_bias = [v for v in inc_bias_samples if np.isfinite(v)]
    valid_ratio = [v for v in inc_abs_ratio_samples if np.isfinite(v)]

    return {
        **skill_result,
        "increment_rmse_latw": float(np.sqrt(np.mean(valid_inc_mse))) if valid_inc_mse else np.nan,
        "increment_bias_latw": float(np.mean(valid_bias)) if valid_bias else np.nan,
        "increment_bias_latw_std": float(np.std(valid_bias)) if valid_bias else np.nan,
        "increment_amp_ratio": float(np.mean(valid_ratio)) if valid_ratio else np.nan,
        "increment_amp_ratio_p95": float(np.percentile(valid_ratio, 95)) if len(valid_ratio) >= 20 else np.nan,
    }


def _count_worse_than_forecast(dataset, predictor) -> dict:
    """Count samples where model RMSE > forecast RMSE (model degrades forecast)."""
    surface_worse = 0
    rootzone_worse = 0
    n_total = len(dataset)

    for idx in range(n_total):
        sample = dataset[idx]
        pred = predictor.predict(sample)
        mask = sample["metric_mask"]
        latw = sample["latitude_weight"]

        for var in ["surface", "rootzone"]:
            if var == "surface":
                pa = pred["pred_analysis_surface"]
                ta = sample["analysis_surface"]
                fc = sample["forecast_surface"]
            else:
                pa = pred["pred_analysis_rootzone"]
                ta = sample["analysis_rootzone"]
                fc = sample["forecast_rootzone"]

            mse_m, mse_f = weighted_analysis_skill_components(
                pred_analysis=pa, true_analysis=ta, forecast=fc,
                mask=mask, latitude_weight=latw,
            )
            if np.isfinite(mse_m) and np.isfinite(mse_f) and mse_m > mse_f:
                if var == "surface":
                    surface_worse += 1
                else:
                    rootzone_worse += 1

    return {
        "surface_worse_than_forecast": surface_worse,
        "surface_worse_fraction": surface_worse / max(n_total, 1),
        "rootzone_worse_than_forecast": rootzone_worse,
        "rootzone_worse_fraction": rootzone_worse / max(n_total, 1),
        "n_total": n_total,
    }


def _rootzone_denominator_analysis(source_val_metrics: dict) -> str:
    """Produce rootzone skill denominator sensitivity analysis."""
    lines = []
    skill_r = source_val_metrics.get("skill", np.nan)
    rmse_fcst = source_val_metrics.get("rmse_forecast", np.nan)
    rmse_model = source_val_metrics.get("rmse_model", np.nan)

    lines.append("### Rootzone Skill Denominator Analysis")
    lines.append("")
    lines.append(f"- Forecast RMSE (denominator): {rmse_fcst:.6f}")
    lines.append(f"- Model RMSE: {rmse_model:.6f}")
    lines.append(f"- Skill: {skill_r:.4f}")

    if np.isfinite(rmse_fcst) and rmse_fcst < 0.001:
        lines.append("")
        lines.append("**WARNING: rootzone forecast RMSE < 0.001 — skill denominator is very small.**")
        lines.append("Small changes in numerator will produce large swings in reported skill.")
        lines.append("Rootzone skill should be interpreted with caution.")
    elif np.isfinite(rmse_fcst) and rmse_fcst < 0.005:
        lines.append("")
        lines.append("**CAUTION: rootzone forecast RMSE < 0.005 — skill denominator is small.**")
        lines.append("Skill values may be sensitive to small absolute errors.")
    else:
        lines.append("")
        lines.append("Rootzone forecast RMSE is sufficiently large — skill denominator is stable.")

    lines.append("")
    return "\n".join(lines)


def _increment_distribution_section(dataset_name: str, metrics: dict) -> str:
    """Generate increment distribution markdown section."""
    lines = []
    lines.append(f"#### Increment Distribution — {dataset_name}")
    lines.append("")
    lines.append("| Statistic | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| RMSE (latw) | {metrics.get('increment_rmse_latw', np.nan):.6f} |")
    lines.append(f"| Bias (latw) | {metrics.get('increment_bias_latw', np.nan):.6f} |")
    lines.append(f"| Bias std | {metrics.get('increment_bias_latw_std', np.nan):.6f} |")
    lines.append(f"| Amp ratio (mean) | {metrics.get('increment_amp_ratio', np.nan):.4f} |")
    lines.append(f"| Amp ratio (p95) | {metrics.get('increment_amp_ratio_p95', np.nan):.4f} |")
    lines.append("")
    return "\n".join(lines)


def generate_report(
    checkpoint_path: str,
    target_region: str,
    K: int,
    seed: int,
    output_path: str,
    device: str = "cuda",
) -> str:
    """Generate rootzone diagnosis markdown report."""
    device = device if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint: {checkpoint_path}")
    predictor = SourceOnlyBackbonePredictor(checkpoint_path=checkpoint_path, device=device)

    # Load source_val dataset
    print(f"Loading source_val dataset...")
    source_val_ds = HydroDADataset(
        da_nc_path=DA_NC,
        region_masks_nc=REGION_MASKS_NC,
        splits_json=SPLITS_JSON,
        target_region=target_region,
        split_type="source_val",
        K=K, seed=seed,
        freeze_manifest=FREEZE_MANIFEST,
    )
    print(f"  source_val samples: {len(source_val_ds)}")

    # Load target_query dataset (eval only, explicitly marked)
    print(f"Loading target_query dataset (evaluation only)...")
    target_query_ds = HydroDADataset(
        da_nc_path=DA_NC,
        region_masks_nc=REGION_MASKS_NC,
        splits_json=SPLITS_JSON,
        target_region=target_region,
        split_type="target_query",
        K=K, seed=seed,
        freeze_manifest=FREEZE_MANIFEST,
    )
    print(f"  target_query samples: {len(target_query_ds)}")

    lines = []
    lines.append(f"# Rootzone Diagnosis Report")
    lines.append("")
    lines.append(f"- **Checkpoint**: `{checkpoint_path}`")
    lines.append(f"- **Target region**: {target_region}")
    lines.append(f"- **K**: {K}, **seed**: {seed}")
    lines.append(f"- **Generated**: {datetime.now().isoformat()}")
    lines.append("")
    lines.append("> **No-leakage declaration**: target_query is used ONLY for final evaluation.")
    lines.append("> Model selection, early stopping, and checkpoint choice are based on source_val only.")
    lines.append("")

    # --- Source Val Evaluation ---
    lines.append("## Source Val Evaluation (used for model selection)")
    lines.append("")

    # Forecast-only on source_val
    lines.append("### Forecast-Only (source_val)")
    lines.append("")
    fc_surface = _forecast_only_metrics(source_val_ds, "surface")
    fc_rootzone = _forecast_only_metrics(source_val_ds, "rootzone")
    lines.append("| Variable | RMSE (latw) | Skill |")
    lines.append("|----------|-------------|-------|")
    lines.append(f"| Surface  | {fc_surface['rmse_model']:.6f} | baseline=0 |")
    lines.append(f"| Rootzone | {fc_rootzone['rmse_model']:.6f} | baseline=0 |")
    lines.append("")

    # Model on source_val
    lines.append("### Source-Only Model (source_val)")
    lines.append("")
    model_surface = _model_metrics(source_val_ds, predictor, "surface")
    model_rootzone = _model_metrics(source_val_ds, predictor, "rootzone")
    lines.append("| Variable | Model RMSE (latw) | Forecast RMSE (latw) | Skill (latw) |")
    lines.append("|----------|-------------------|----------------------|--------------|")
    lines.append(f"| Surface  | {model_surface['rmse_model']:.6f} | {model_surface['rmse_forecast']:.6f} | {model_surface['skill']:.4f} |")
    lines.append(f"| Rootzone | {model_rootzone['rmse_model']:.6f} | {model_rootzone['rmse_forecast']:.6f} | {model_rootzone['skill']:.4f} |")
    lines.append("")

    # Increment distribution on source_val
    lines.append(_increment_distribution_section("source_val (surface)", model_surface))
    lines.append(_increment_distribution_section("source_val (rootzone)", model_rootzone))

    # Rootzone denominator analysis
    lines.append(_rootzone_denominator_analysis(model_rootzone))

    # --- Target Query Evaluation (final only) ---
    lines.append("## Target Query Evaluation (FINAL EVALUATION ONLY — not used for selection)")
    lines.append("")

    tq_surface = _model_metrics(target_query_ds, predictor, "surface")
    tq_rootzone = _model_metrics(target_query_ds, predictor, "rootzone")

    lines.append("| Variable | Model RMSE (latw) | Forecast RMSE (latw) | Skill (latw) |")
    lines.append("|----------|-------------------|----------------------|--------------|")
    lines.append(f"| Surface  | {tq_surface['rmse_model']:.6f} | {tq_surface['rmse_forecast']:.6f} | {tq_surface['skill']:.4f} |")
    lines.append(f"| Rootzone | {tq_rootzone['rmse_model']:.6f} | {tq_rootzone['rmse_forecast']:.6f} | {tq_rootzone['skill']:.4f} |")
    lines.append("")

    lines.append(_increment_distribution_section("target_query (surface)", tq_surface))
    lines.append(_increment_distribution_section("target_query (rootzone)", tq_rootzone))

    # Worse-than-forecast analysis
    lines.append("### Samples Where Model Degrades Forecast")
    lines.append("")
    worse = _count_worse_than_forecast(target_query_ds, predictor)
    lines.append(f"- Surface: {worse['surface_worse_than_forecast']}/{worse['n_total']} ({worse['surface_worse_fraction']:.1%})")
    lines.append(f"- Rootzone: {worse['rootzone_worse_than_forecast']}/{worse['n_total']} ({worse['rootzone_worse_fraction']:.1%})")
    lines.append("")

    # Cleanup
    source_val_ds.close()
    target_query_ds.close()

    report_text = "\n".join(lines)

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nReport saved to {output_path}")
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Generate rootzone diagnosis report")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--target_region", type=str, default="US-R1")
    parser.add_argument("--K", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None,
        help="Output markdown path (default: reports/rootzone_diagnosis_{target_region}.md)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output = args.output or f"reports/rootzone_diagnosis_{args.target_region}.md"

    report = generate_report(
        checkpoint_path=args.checkpoint,
        target_region=args.target_region,
        K=args.K,
        seed=args.seed,
        output_path=output,
        device=args.device,
    )
    print(report)


if __name__ == "__main__":
    main()
