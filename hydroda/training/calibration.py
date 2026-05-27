"""Residual gain calibration for DA increment prediction.

Calibrates a per-variable alpha scaling factor on source_val
to shrink or expand predicted increments before adding to forecast.

No-leakage declaration:
    - Alpha calibration uses source_val only (never target_query)
    - Selection metric: max(min_skill), tie-break max(mean_skill)
    - Alpha=0 is always a candidate
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hydroda.metrics.skill import weighted_analysis_skill_components


def calibrate_residual_gain(
    samples_s: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    samples_r: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    alphas: List[float],
) -> Dict[str, Any]:
    """Calibrate residual gain alphas on pre-accumulated source_val samples.

    Each sample tuple is (pred_inc, true_inc, forecast, mask, latitude_weight).
    All arrays are 2D [H, W] numpy float32.

    Scans alpha grid, computing analysis skill for each alpha:

        analysis = forecast + alpha * pred_inc
        skill = 1 - sqrt(mean(model_mse)) / sqrt(mean(forecast_mse))

    where mean is over time samples (each weighted by latitude_weight).

    Selection: primary = max(min_skill), tie-break = max(mean_skill),
    tie-break = min(mean_rmse_ratio). alpha=0 is always in the candidate set.

    Args:
        samples_s: list of (pred_inc, true_inc, forecast, mask, latw) for surface
        samples_r: list of (pred_inc, true_inc, forecast, mask, latw) for rootzone
        alphas: list of alpha values to scan

    Returns:
        dict with best_alpha_surface, best_alpha_rootzone, skills, RMSEs,
        per_alpha_results, etc.
    """
    if not samples_s or not samples_r:
        return {}

    # ---- Per-alpha scan ----
    per_alpha_results: Dict[float, Dict[str, float]] = {}

    for alpha in alphas:
        # Surface
        mse_list_s = []
        for pred_inc, true_inc, fcst, mask, latw in samples_s:
            pred_an = fcst + alpha * pred_inc
            analysis = fcst + true_inc
            mse_m, _ = weighted_analysis_skill_components(
                pred_analysis=pred_an,
                true_analysis=analysis,
                forecast=fcst,
                mask=mask,
                latitude_weight=latw,
            )
            mse_list_s.append(mse_m)

        valid_mse_s = [v for v in mse_list_s if np.isfinite(v)]
        mean_mse_s = np.mean(valid_mse_s) if valid_mse_s else np.nan
        rmse_s = float(np.sqrt(mean_mse_s)) if np.isfinite(mean_mse_s) and mean_mse_s > 0 else float("nan")

        # Rootzone
        mse_list_r = []
        for pred_inc, true_inc, fcst, mask, latw in samples_r:
            pred_an = fcst + alpha * pred_inc
            analysis = fcst + true_inc
            mse_m, _ = weighted_analysis_skill_components(
                pred_analysis=pred_an,
                true_analysis=analysis,
                forecast=fcst,
                mask=mask,
                latitude_weight=latw,
            )
            mse_list_r.append(mse_m)

        valid_mse_r = [v for v in mse_list_r if np.isfinite(v)]
        mean_mse_r = np.mean(valid_mse_r) if valid_mse_r else np.nan
        rmse_r = float(np.sqrt(mean_mse_r)) if np.isfinite(mean_mse_r) and mean_mse_r > 0 else float("nan")

        per_alpha_results[alpha] = {
            "surface_model_rmse": rmse_s,
            "rootzone_model_rmse": rmse_r,
        }

    # Compute forecast RMSE (alpha-independent)
    forecast_mse_s_list = []
    forecast_mse_r_list = []
    for _pred_inc, true_inc, fcst, mask, latw in samples_s:
        analysis = fcst + true_inc
        _, mse_f = weighted_analysis_skill_components(
            pred_analysis=fcst, true_analysis=analysis, forecast=fcst,
            mask=mask, latitude_weight=latw,
        )
        forecast_mse_s_list.append(mse_f)
    for _pred_inc, true_inc, fcst, mask, latw in samples_r:
        analysis = fcst + true_inc
        _, mse_f = weighted_analysis_skill_components(
            pred_analysis=fcst, true_analysis=analysis, forecast=fcst,
            mask=mask, latitude_weight=latw,
        )
        forecast_mse_r_list.append(mse_f)

    valid_fcst_s = [v for v in forecast_mse_s_list if np.isfinite(v)]
    valid_fcst_r = [v for v in forecast_mse_r_list if np.isfinite(v)]
    mean_fcst_mse_s = np.mean(valid_fcst_s) if valid_fcst_s else np.nan
    mean_fcst_mse_r = np.mean(valid_fcst_r) if valid_fcst_r else np.nan
    rmse_forecast_s = float(np.sqrt(mean_fcst_mse_s)) if np.isfinite(mean_fcst_mse_s) and mean_fcst_mse_s > 0 else float("nan")
    rmse_forecast_r = float(np.sqrt(mean_fcst_mse_r)) if np.isfinite(mean_fcst_mse_r) and mean_fcst_mse_r > 0 else float("nan")

    # Compute skills for each alpha
    for alpha in alphas:
        rmse_s = per_alpha_results[alpha]["surface_model_rmse"]
        rmse_r = per_alpha_results[alpha]["rootzone_model_rmse"]
        skill_s = float(1.0 - rmse_s / rmse_forecast_s) if np.isfinite(rmse_s) and np.isfinite(rmse_forecast_s) and rmse_forecast_s > 0 else float("nan")
        skill_r = float(1.0 - rmse_r / rmse_forecast_r) if np.isfinite(rmse_r) and np.isfinite(rmse_forecast_r) and rmse_forecast_r > 0 else float("nan")
        per_alpha_results[alpha]["surface_skill"] = skill_s
        per_alpha_results[alpha]["rootzone_skill"] = skill_r
        per_alpha_results[alpha]["min_skill"] = min(skill_s, skill_r) if np.isfinite(skill_s) and np.isfinite(skill_r) else float("-inf")
        per_alpha_results[alpha]["mean_skill"] = float(np.mean([skill_s, skill_r])) if np.isfinite(skill_s) and np.isfinite(skill_r) else float("-inf")

    # Select best alpha: primary = max(min_skill), tie-break = max(mean_skill),
    # tie-break = min(mean_rmse_ratio)
    def _selection_key(alpha: float) -> Tuple[float, float, float]:
        r = per_alpha_results[alpha]
        rmse_ratio_s = r["surface_model_rmse"] / rmse_forecast_s if np.isfinite(r["surface_model_rmse"]) and rmse_forecast_s > 0 else float("inf")
        rmse_ratio_r = r["rootzone_model_rmse"] / rmse_forecast_r if np.isfinite(r["rootzone_model_rmse"]) and rmse_forecast_r > 0 else float("inf")
        mean_rmse_ratio = float(np.mean([rmse_ratio_s, rmse_ratio_r]))
        return (r["min_skill"], r["mean_skill"], -mean_rmse_ratio)

    best_alpha = max(alphas, key=_selection_key)
    best = per_alpha_results[best_alpha]

    # Skill before alpha (at alpha=1.0)
    alpha1 = per_alpha_results.get(1.0, {})
    skill_s_before = alpha1.get("surface_skill", float("nan"))
    skill_r_before = alpha1.get("rootzone_skill", float("nan"))
    rmse_s_model_before = alpha1.get("surface_model_rmse", float("nan"))
    rmse_r_model_before = alpha1.get("rootzone_model_rmse", float("nan"))

    return {
        "best_alpha_surface": float(best_alpha),
        "best_alpha_rootzone": float(best_alpha),
        "skill_surface_with_alpha": best["surface_skill"],
        "skill_rootzone_with_alpha": best["rootzone_skill"],
        "skill_surface_before_alpha": skill_s_before,
        "skill_rootzone_before_alpha": skill_r_before,
        "rmse_surface_model": best["surface_model_rmse"],
        "rmse_rootzone_model": best["rootzone_model_rmse"],
        "rmse_surface_forecast": rmse_forecast_s,
        "rmse_rootzone_forecast": rmse_forecast_r,
        "rmse_surface_model_before_alpha": rmse_s_model_before,
        "rmse_rootzone_model_before_alpha": rmse_r_model_before,
        "min_skill": best["min_skill"],
        "mean_skill": best["mean_skill"],
        "selection_score": best["min_skill"],
        "alpha_grid": alphas,
        "per_alpha_results": {str(a): r for a, r in per_alpha_results.items()},
    }


def _compute_region_skill(
    samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    alpha: float,
) -> float:
    """Compute latitude-weighted analysis skill for a single region and alpha.

    Args:
        samples: list of (pred_inc, true_inc, forecast, mask, latw) tuples
        alpha: residual gain alpha value

    Returns:
        skill = 1 - sqrt(mean_model_mse) / sqrt(mean_forecast_mse)
    """
    mse_list = []
    forecast_mse_list = []
    for pred_inc, true_inc, fcst, mask, latw in samples:
        pred_an = fcst + alpha * pred_inc
        analysis = fcst + true_inc
        model_mse, forecast_mse = weighted_analysis_skill_components(
            pred_analysis=pred_an,
            true_analysis=analysis,
            forecast=fcst,
            mask=mask,
            latitude_weight=latw,
        )
        if np.isfinite(model_mse):
            mse_list.append(model_mse)
        if np.isfinite(forecast_mse):
            forecast_mse_list.append(forecast_mse)

    if not mse_list or not forecast_mse_list:
        return float("nan")

    mean_model_mse = float(np.mean(mse_list))
    mean_forecast_mse = float(np.mean(forecast_mse_list))

    if not np.isfinite(mean_model_mse) or not np.isfinite(mean_forecast_mse) or mean_forecast_mse <= 0:
        return float("nan")

    rmse_model = float(np.sqrt(mean_model_mse))
    rmse_forecast = float(np.sqrt(mean_forecast_mse))
    return float(1.0 - rmse_model / rmse_forecast)


def calibrate_residual_gain_region_aware(
    samples_s_by_region: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
    samples_r_by_region: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
    alpha_grid: List[float],
    prompt_quality_metrics: Optional[Dict[str, float]] = None,
    trace_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """2D alpha grid search with region-aware transfer-safe scoring.

    Scans alpha_surface × alpha_rootzone pairs independently. For each pair,
    computes per-region, per-variable analysis skills, then aggregates into
    a transfer-safe score that penalizes regions with negative skill.

    Score formula:
        transfer_safe_score =
            0.50 * worst_region_balanced_skill
          + 0.25 * mean_region_balanced_skill
          + 0.15 * positive_improvement_rate
          + 0.10 * (1.0 - skill_dispersion)
          - penalty_negative_rootzone

    where:
        - balanced_skill = (surface_skill + rootzone_skill) / 2
        - worst_region = min over regions
        - positive_improvement_rate = fraction of (region, variable) pairs with skill > 0
        - skill_dispersion = std of balanced skills across regions
        - penalty_negative_rootzone = 0.2 * count of regions with rootzone_skill < 0

    Args:
        samples_s_by_region: dict mapping region_name -> list of surface samples
        samples_r_by_region: dict mapping region_name -> list of rootzone samples
        alpha_grid: list of alpha values for grid search
        prompt_quality_metrics: optional prompt quality metrics
        trace_path: optional path to save alpha_selection_trace.csv

    Returns:
        dict with best_alpha_surface, best_alpha_rootzone, per_alpha results,
        region × variable breakdown for best alpha, selection traces, etc.
    """
    if not samples_s_by_region or not samples_r_by_region:
        return {}

    region_names = sorted(samples_s_by_region.keys())
    if not region_names:
        return {}

    # ---- 2D alpha grid scan ----
    trace_rows: List[Dict[str, Any]] = []
    best_score = float("-inf")
    best_alpha_s = alpha_grid[0]
    best_alpha_r = alpha_grid[0]
    best_region_skills: Dict[str, Dict[str, float]] = {}

    for alpha_s in alpha_grid:
        for alpha_r in alpha_grid:
            region_skills: Dict[str, Dict[str, float]] = {}
            region_balanced: List[float] = []

            for region in region_names:
                skill_s = _compute_region_skill(samples_s_by_region[region], alpha_s)
                skill_r = _compute_region_skill(samples_r_by_region[region], alpha_r)

                region_skills[region] = {
                    "surface_skill": skill_s if np.isfinite(skill_s) else float("nan"),
                    "rootzone_skill": skill_r if np.isfinite(skill_r) else float("nan"),
                }

                bal = float(np.mean([skill_s, skill_r])) if np.isfinite(skill_s) and np.isfinite(skill_r) else float("nan")
                region_balanced.append(bal)

            # Compute transfer-safe score components
            valid_balanced = [b for b in region_balanced if np.isfinite(b)]
            if not valid_balanced:
                score = float("-inf")
                worst_balanced = float("nan")
                mean_balanced = float("nan")
                pos_rate = float("nan")
                dispersion = float("nan")
                neg_rootzone_count = 0
            else:
                worst_balanced = float(np.min(valid_balanced))
                mean_balanced = float(np.mean(valid_balanced))
                dispersion = float(np.std(valid_balanced)) if len(valid_balanced) >= 2 else 0.0

                # positive_improvement_rate: fraction of (region, variable) with skill > 0
                all_skills = []
                neg_rootzone_count = 0
                for region in region_names:
                    rs = region_skills[region]
                    for key in ["surface_skill", "rootzone_skill"]:
                        val = rs[key]
                        if np.isfinite(val):
                            all_skills.append(val)
                            if key == "rootzone_skill" and val < 0:
                                neg_rootzone_count += 1

                pos_rate = float(np.mean([1.0 if s > 0 else 0.0 for s in all_skills])) if all_skills else 0.0
                penalty_neg_rootzone = 0.2 * float(neg_rootzone_count)

                score = (
                    0.50 * worst_balanced
                    + 0.25 * mean_balanced
                    + 0.15 * pos_rate
                    + 0.10 * (1.0 - min(dispersion, 1.0))
                    - penalty_neg_rootzone
                )

            # Prompt collapse penalty
            if prompt_quality_metrics:
                pc_mean = prompt_quality_metrics.get("prompt_pairwise_cosine_distance_mean", 1.0)
                if pc_mean < 0.01:
                    score = float("-inf")

            trace_rows.append({
                "alpha_surface": alpha_s,
                "alpha_rootzone": alpha_r,
                "transfer_safe_score": score,
                "worst_region_balanced_skill": worst_balanced,
                "mean_region_balanced_skill": mean_balanced,
                "positive_improvement_rate": pos_rate,
                "skill_dispersion": dispersion,
                "neg_rootzone_count": neg_rootzone_count,
            })

            if score > best_score:
                best_score = score
                best_alpha_s = alpha_s
                best_alpha_r = alpha_r
                best_region_skills = region_skills

    # ---- Save trace CSV ----
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        if trace_rows:
            with open(trace_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
                writer.writeheader()
                writer.writerows(trace_rows)

    # ---- Compute aggregate metrics at best alpha ----
    # Surface aggregate
    mse_list_s, fcst_mse_list_s = [], []
    for region in region_names:
        for pred_inc, true_inc, fcst, mask, latw in samples_s_by_region[region]:
            pred_an = fcst + best_alpha_s * pred_inc
            analysis = fcst + true_inc
            model_mse, forecast_mse = weighted_analysis_skill_components(
                pred_analysis=pred_an, true_analysis=analysis,
                forecast=fcst, mask=mask, latitude_weight=latw,
            )
            if np.isfinite(model_mse):
                mse_list_s.append(model_mse)
            if np.isfinite(forecast_mse):
                fcst_mse_list_s.append(forecast_mse)

    mean_mse_s = float(np.mean(mse_list_s)) if mse_list_s else float("nan")
    mean_fcst_mse_s = float(np.mean(fcst_mse_list_s)) if fcst_mse_list_s else float("nan")
    rmse_s = float(np.sqrt(mean_mse_s)) if np.isfinite(mean_mse_s) and mean_mse_s > 0 else float("nan")
    rmse_fcst_s = float(np.sqrt(mean_fcst_mse_s)) if np.isfinite(mean_fcst_mse_s) and mean_fcst_mse_s > 0 else float("nan")
    skill_s = float(1.0 - rmse_s / rmse_fcst_s) if np.isfinite(rmse_s) and np.isfinite(rmse_fcst_s) and rmse_fcst_s > 0 else float("nan")

    # Rootzone aggregate
    mse_list_r, fcst_mse_list_r = [], []
    for region in region_names:
        for pred_inc, true_inc, fcst, mask, latw in samples_r_by_region[region]:
            pred_an = fcst + best_alpha_r * pred_inc
            analysis = fcst + true_inc
            model_mse, forecast_mse = weighted_analysis_skill_components(
                pred_analysis=pred_an, true_analysis=analysis,
                forecast=fcst, mask=mask, latitude_weight=latw,
            )
            if np.isfinite(model_mse):
                mse_list_r.append(model_mse)
            if np.isfinite(forecast_mse):
                fcst_mse_list_r.append(forecast_mse)

    mean_mse_r = float(np.mean(mse_list_r)) if mse_list_r else float("nan")
    mean_fcst_mse_r = float(np.mean(fcst_mse_list_r)) if fcst_mse_list_r else float("nan")
    rmse_r = float(np.sqrt(mean_mse_r)) if np.isfinite(mean_mse_r) and mean_mse_r > 0 else float("nan")
    rmse_fcst_r = float(np.sqrt(mean_fcst_mse_r)) if np.isfinite(mean_fcst_mse_r) and mean_fcst_mse_r > 0 else float("nan")
    skill_r = float(1.0 - rmse_r / rmse_fcst_r) if np.isfinite(rmse_r) and np.isfinite(rmse_fcst_r) and rmse_fcst_r > 0 else float("nan")

    return {
        "best_alpha_surface": best_alpha_s,
        "best_alpha_rootzone": best_alpha_r,
        "skill_surface_with_alpha": skill_s,
        "skill_rootzone_with_alpha": skill_r,
        "rmse_surface_model": rmse_s,
        "rmse_rootzone_model": rmse_r,
        "rmse_surface_forecast": rmse_fcst_s,
        "rmse_rootzone_forecast": rmse_fcst_r,
        "alpha_grid": alpha_grid,
        "selection_score": best_score,
        "min_skill": float(np.min([skill_s, skill_r])) if np.isfinite(skill_s) and np.isfinite(skill_r) else float("nan"),
        "mean_skill": float(np.mean([skill_s, skill_r])) if np.isfinite(skill_s) and np.isfinite(skill_r) else float("nan"),
        "region_variable_skills": best_region_skills,
        "region_names": region_names,
        "selection_trace": trace_rows,
        "calibration_mode": "region_aware_2d",
    }
