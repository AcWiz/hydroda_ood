"""Residual gain calibration for DA increment prediction.

Calibrates a per-variable alpha scaling factor on source_val
to shrink or expand predicted increments before adding to forecast.

No-leakage declaration:
    - Alpha calibration uses source_val only (never target_eval/query)
    - Selection metric: max(min_skill), tie-break max(mean_skill)
    - Alpha=0 is always a candidate
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from hydroda.metrics.skill import weighted_analysis_skill_components


def prediction_content_hash_for_increments(
    *,
    pred_increment_surface: np.ndarray,
    pred_increment_rootzone: np.ndarray,
) -> str:
    """Hash surface/rootzone increment arrays for lightweight K-specific audits."""
    hasher = hashlib.sha256()
    for name, array in (
        ("pred_increment_surface", pred_increment_surface),
        ("pred_increment_rootzone", pred_increment_rootzone),
    ):
        arr = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
        payload = {
            "name": name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        }
        hasher.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _weighted_affine_arrays(
    samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ws: List[np.ndarray] = []
    for pred_inc, true_inc, _fcst, mask, latw in samples:
        pred = np.asarray(pred_inc, dtype=np.float64)
        true = np.asarray(true_inc, dtype=np.float64)
        valid = np.asarray(mask, dtype=np.float64)
        weight = np.asarray(latw, dtype=np.float64)
        valid = valid * weight
        finite = np.isfinite(pred) & np.isfinite(true) & np.isfinite(valid) & (valid > 0.0)
        if not np.any(finite):
            continue
        xs.append(pred[finite].reshape(-1))
        ys.append(true[finite].reshape(-1))
        ws.append(valid[finite].reshape(-1))
    if not xs:
        return (
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)


def _fit_affine_coefficients(
    samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    ridge_lambda: float,
    prior_a: float = 1.0,
    prior_b: float = 0.0,
    shrinkage_strength: float = 0.0,
    shrinkage_target: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    x, y, w = _weighted_affine_arrays(samples)
    support_observation_count = int(x.size)
    if support_observation_count <= 0:
        return {
            "a": float(prior_a),
            "b": float(prior_b),
            "status": "empty_support_fallback_to_prior",
            "support_observation_count": 0,
            "weighted_sse_before": None,
            "weighted_sse_after": None,
            "weighted_mse_before": None,
            "weighted_mse_after": None,
            "fallback_rule": "prior_identity_affine",
        }
    design = np.stack([x, np.ones_like(x)], axis=1)
    weight = np.clip(w, 0.0, None)
    xtw = design.T * weight
    xtx = xtw @ design
    xty = xtw @ y
    prior = np.asarray([float(prior_a), float(prior_b)], dtype=np.float64)
    ridge = float(max(0.0, ridge_lambda))
    lhs = xtx + ridge * np.eye(2, dtype=np.float64)
    rhs = xty + ridge * prior
    shrink = float(max(0.0, shrinkage_strength))
    if shrink > 0.0 and shrinkage_target is not None:
        target = np.asarray([float(shrinkage_target[0]), float(shrinkage_target[1])], dtype=np.float64)
        lhs = lhs + shrink * np.eye(2, dtype=np.float64)
        rhs = rhs + shrink * target
    try:
        beta = np.linalg.solve(lhs, rhs)
        status = "solved"
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs
        status = "pinv_solved"
    beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
    before = x
    after = beta[0] * x + beta[1]
    weighted_sse_before = float(np.sum(weight * (before - y) ** 2))
    weighted_sse_after = float(np.sum(weight * (after - y) ** 2))
    weight_sum = float(np.sum(weight))
    return {
        "a": float(beta[0]),
        "b": float(beta[1]),
        "status": status,
        "support_observation_count": support_observation_count,
        "weighted_sse_before": weighted_sse_before,
        "weighted_sse_after": weighted_sse_after,
        "weighted_mse_before": weighted_sse_before / weight_sum if weight_sum > 0.0 else None,
        "weighted_mse_after": weighted_sse_after / weight_sum if weight_sum > 0.0 else None,
        "ridge_lambda": ridge,
        "shrinkage_strength": shrink,
        "fallback_rule": "global_per_variable_affine",
    }


def _season_sample_groups(
    samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    seasons: Optional[Iterable[str]],
) -> Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    if seasons is None:
        return {}
    season_values = [str(season) for season in seasons]
    if len(season_values) != len(samples):
        raise ValueError("seasons length must match sample count")
    groups: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}
    for sample, season in zip(samples, season_values):
        if not season:
            continue
        groups.setdefault(season, []).append(sample)
    return groups


def calibrate_residual_affine(
    samples_s: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    samples_r: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    seasons: Optional[Iterable[str]] = None,
    ridge_lambda: float = 0.1,
    shrinkage_strength: float = 0.0,
    season_shrinkage_strength: float = 8.0,
    K: int = 0,
    support_nesting_policy: str = "",
    nested_support_dates_hash: str = "",
) -> Dict[str, Any]:
    """Fit support-only per-variable residual affine calibration.

    The fit uses only target_support labels already admitted by the K-shot
    split. It does not inspect target_eval and returns all coefficients as
    metadata so downstream evaluation applies a frozen, auditable transform:
    ``pred_increment_calibrated = a * pred_increment + b``.
    """
    if not samples_s or not samples_r:
        return {
            "calibration_mode": (
                "target_support_residual_affine_v1_nested" if int(K) == 12 else "target_support_residual_affine_v1"
            ),
            "status": "empty_support_or_invalid_samples",
            "label_source": "target_support_only",
            "target_eval_usage": "final_eval_only_no_selection",
            "support_affine_coefficients": {
                "surface": {"a": 1.0, "b": 0.0, "status": "empty_support_fallback_to_identity"},
                "rootzone": {"a": 1.0, "b": 0.0, "status": "empty_support_fallback_to_identity"},
            },
            "seasonal_affine_coefficients": {},
            "effective_calibration_dof": 0.0,
            "ridge_lambda": float(ridge_lambda),
            "shrinkage_strength": float(shrinkage_strength),
            "season_shrinkage_strength": float(season_shrinkage_strength),
            "support_nesting_policy": str(support_nesting_policy or ""),
            "nested_support_dates_hash": str(nested_support_dates_hash or ""),
        }

    global_surface = _fit_affine_coefficients(
        samples_s,
        ridge_lambda=float(ridge_lambda),
        shrinkage_strength=float(shrinkage_strength),
        shrinkage_target=(1.0, 0.0),
    )
    global_rootzone = _fit_affine_coefficients(
        samples_r,
        ridge_lambda=float(ridge_lambda),
        shrinkage_strength=float(shrinkage_strength),
        shrinkage_target=(1.0, 0.0),
    )
    coeffs = {
        "surface": global_surface,
        "rootzone": global_rootzone,
    }
    seasonal_coeffs: Dict[str, Dict[str, Any]] = {}
    effective_dof = 4.0
    if int(K) == 12:
        grouped_s = _season_sample_groups(samples_s, seasons)
        grouped_r = _season_sample_groups(samples_r, seasons)
        for season in sorted(set(grouped_s) | set(grouped_r)):
            s_samples = grouped_s.get(season, [])
            r_samples = grouped_r.get(season, [])
            if not s_samples or not r_samples:
                continue
            season_surface = _fit_affine_coefficients(
                s_samples,
                ridge_lambda=float(ridge_lambda),
                shrinkage_strength=float(season_shrinkage_strength),
                shrinkage_target=(float(global_surface["a"]), float(global_surface["b"])),
            )
            season_rootzone = _fit_affine_coefficients(
                r_samples,
                ridge_lambda=float(ridge_lambda),
                shrinkage_strength=float(season_shrinkage_strength),
                shrinkage_target=(float(global_rootzone["a"]), float(global_rootzone["b"])),
            )
            seasonal_coeffs[season] = {
                "surface": season_surface,
                "rootzone": season_rootzone,
            }
            n_eff = min(
                int(season_surface.get("support_observation_count", 0) or 0),
                int(season_rootzone.get("support_observation_count", 0) or 0),
            )
            effective_dof += 4.0 * float(n_eff) / float(n_eff + max(1e-12, float(season_shrinkage_strength)))

    return {
        "calibration_mode": (
            "target_support_residual_affine_v1_nested" if int(K) == 12 else "target_support_residual_affine_v1"
        ),
        "status": "calibrated",
        "label_source": "target_support_only",
        "target_eval_usage": "final_eval_only_no_selection",
        "support_affine_coefficients": coeffs,
        "seasonal_affine_coefficients": seasonal_coeffs,
        "effective_calibration_dof": float(effective_dof),
        "ridge_lambda": float(ridge_lambda),
        "shrinkage_strength": float(shrinkage_strength),
        "season_shrinkage_strength": float(season_shrinkage_strength),
        "support_count": len(samples_s),
        "support_nesting_policy": str(support_nesting_policy or ""),
        "nested_support_dates_hash": str(nested_support_dates_hash or ""),
    }


def _coeff_pair(
    calibration: Dict[str, Any],
    variable: str,
    season: Optional[str] = None,
) -> Tuple[float, float]:
    seasonal = calibration.get("seasonal_affine_coefficients", {})
    if season and isinstance(seasonal, dict):
        season_block = seasonal.get(str(season), {})
        if isinstance(season_block, dict) and variable in season_block:
            block = season_block[variable]
            return float(block.get("a", 1.0)), float(block.get("b", 0.0))
    block = (calibration.get("support_affine_coefficients", {}) or {}).get(variable, {})
    if not isinstance(block, dict):
        return 1.0, 0.0
    return float(block.get("a", 1.0)), float(block.get("b", 0.0))


def apply_residual_affine_calibration_to_increment(
    *,
    pred_increment_surface: np.ndarray,
    pred_increment_rootzone: np.ndarray,
    calibration: Dict[str, Any],
    season: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply frozen support-affine calibration to predicted increments."""
    a_s, b_s = _coeff_pair(calibration, "surface", season=season)
    a_r, b_r = _coeff_pair(calibration, "rootzone", season=season)
    surface = (a_s * np.asarray(pred_increment_surface, dtype=np.float32) + b_s).astype(np.float32)
    rootzone = (a_r * np.asarray(pred_increment_rootzone, dtype=np.float32) + b_r).astype(np.float32)
    return surface, rootzone


def calibrate_residual_gain(
    samples_s: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    samples_r: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    alphas: List[float],
    selection_rule: str = "max_min_skill",
    paired_support_se_capped: float = 0.0,
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
        rmse_s = float(np.sqrt(mean_mse_s)) if np.isfinite(mean_mse_s) and mean_mse_s >= 0 else float("nan")

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
        rmse_r = float(np.sqrt(mean_mse_r)) if np.isfinite(mean_mse_r) and mean_mse_r >= 0 else float("nan")

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

    raw_best_alpha = max(alphas, key=_selection_key)
    if (
        selection_rule == "support_uncertainty_stable_high_alpha_with_dual_guard"
        and float(paired_support_se_capped) <= 0.0
    ):
        paired_support_se_capped = _estimate_paired_support_se_capped(
            samples_s=samples_s,
            samples_r=samples_r,
            alphas=alphas,
            best_alpha_raw=float(raw_best_alpha),
        )
    stable_selection: Dict[str, Any] = {}
    stable_rules = {
        "stable_high_alpha_with_mean_skill_guard",
        "support_uncertainty_stable_high_alpha_with_dual_guard",
    }
    if selection_rule in stable_rules:
        stable_selection = select_stable_residual_gain_alpha(
            alphas=alphas,
            per_alpha_results=per_alpha_results,
            best_alpha_raw=raw_best_alpha,
            selection_rule=selection_rule,
            paired_support_se_capped=paired_support_se_capped,
        )
        best_alpha = float(stable_selection["selected_alpha"])
    elif selection_rule == "max_min_skill":
        best_alpha = raw_best_alpha
    else:
        raise ValueError(
            "selection_rule must be 'max_min_skill', "
            "'stable_high_alpha_with_mean_skill_guard', or "
            "'support_uncertainty_stable_high_alpha_with_dual_guard', "
            f"got {selection_rule!r}"
        )
    best = per_alpha_results[best_alpha]

    # Skill before alpha (at alpha=1.0)
    alpha1 = per_alpha_results.get(1.0, {})
    skill_s_before = alpha1.get("surface_skill", float("nan"))
    skill_r_before = alpha1.get("rootzone_skill", float("nan"))
    rmse_s_model_before = alpha1.get("surface_model_rmse", float("nan"))
    rmse_r_model_before = alpha1.get("rootzone_model_rmse", float("nan"))

    result = {
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
    if selection_rule in stable_rules:
        result.update(
            {
                "calibration_mode": (
                    "target_support_residual_gain_v4_nested_stable_grid"
                    if selection_rule == "support_uncertainty_stable_high_alpha_with_dual_guard"
                    else "target_support_residual_gain_stable_grid"
                ),
                "label_source": "target_support_only",
                "target_eval_usage": "final_eval_only_no_selection",
                "selection_rule": selection_rule,
                "best_alpha_raw": float(raw_best_alpha),
                "stable_candidate_alphas": [
                    float(alpha) for alpha in stable_selection.get("stable_candidate_alphas", [])
                ],
                "selection_margin": float(stable_selection.get("selection_margin", 0.0)),
                "stability_tolerance": float(stable_selection.get("stability_tolerance", 0.0)),
                "mean_skill_tolerance": float(stable_selection.get("mean_skill_tolerance", 0.02)),
                "variable_skill_tolerance": float(stable_selection.get("variable_skill_tolerance", 0.03)),
                "paired_support_se_capped": float(stable_selection.get("paired_support_se_capped", 0.0)),
            }
        )
    return result


def _estimate_paired_support_se_capped(
    *,
    samples_s: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    samples_r: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    alphas: List[float],
    best_alpha_raw: float,
    cap: float = 0.05,
) -> float:
    """Estimate support-cycle uncertainty for raw-best-vs-candidate min-skill.

    This uses only paired target_support cycles already admitted for support
    calibration. It is intentionally capped so a tiny K cannot make the stable
    tie band unbounded.
    """
    pair_count = min(len(samples_s), len(samples_r))
    if pair_count < 2:
        return 0.0

    def _sample_skill(
        sample: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        alpha: float,
    ) -> float:
        pred_inc, true_inc, fcst, mask, latw = sample
        pred_an = fcst + float(alpha) * pred_inc
        analysis = fcst + true_inc
        model_mse, forecast_mse = weighted_analysis_skill_components(
            pred_analysis=pred_an,
            true_analysis=analysis,
            forecast=fcst,
            mask=mask,
            latitude_weight=latw,
        )
        if not np.isfinite(model_mse) or not np.isfinite(forecast_mse) or forecast_mse <= 0:
            return float("nan")
        return float(1.0 - np.sqrt(model_mse) / np.sqrt(forecast_mse))

    per_alpha_cycle_min: Dict[float, List[float]] = {}
    for alpha in alphas:
        cycle_values = []
        for idx in range(pair_count):
            skill_s = _sample_skill(samples_s[idx], float(alpha))
            skill_r = _sample_skill(samples_r[idx], float(alpha))
            if np.isfinite(skill_s) and np.isfinite(skill_r):
                cycle_values.append(float(min(skill_s, skill_r)))
        per_alpha_cycle_min[float(alpha)] = cycle_values

    best_values = per_alpha_cycle_min.get(float(best_alpha_raw), [])
    if len(best_values) < 2:
        return 0.0
    se_values = []
    for alpha in alphas:
        alpha = float(alpha)
        if alpha == float(best_alpha_raw):
            continue
        other_values = per_alpha_cycle_min.get(alpha, [])
        n = min(len(best_values), len(other_values))
        if n < 2:
            continue
        diffs = np.asarray(best_values[:n], dtype=np.float64) - np.asarray(other_values[:n], dtype=np.float64)
        finite = diffs[np.isfinite(diffs)]
        if finite.size < 2:
            continue
        se_values.append(float(np.std(finite, ddof=1) / np.sqrt(float(finite.size))))
    if not se_values:
        return 0.0
    return float(min(max(se_values), max(0.0, float(cap))))


def select_stable_residual_gain_alpha(
    *,
    alphas: List[float],
    per_alpha_results: Dict[float, Dict[str, float]],
    best_alpha_raw: Optional[float] = None,
    min_skill_tolerance_floor: float = 0.02,
    min_skill_gap_fraction: float = 0.25,
    mean_skill_tolerance: float = 0.02,
    variable_skill_tolerance: float = 0.03,
    paired_support_se_capped: float = 0.0,
    selection_rule: str = "stable_high_alpha_with_mean_skill_guard",
) -> Dict[str, Any]:
    """Select a stable support-gain alpha without target-eval feedback.

    The raw selector maximizes support min-skill. This diagnostic selector
    keeps that raw winner as the primary evidence, then allows larger alphas
    whose min-skill is within a stability tolerance and whose mean-skill is
    not materially below the raw winner.
    """
    if not alphas:
        raise ValueError("alphas must not be empty")
    if not per_alpha_results:
        raise ValueError("per_alpha_results must not be empty")
    if selection_rule not in {
        "stable_high_alpha_with_mean_skill_guard",
        "support_uncertainty_stable_high_alpha_with_dual_guard",
    }:
        raise ValueError(
            "selection_rule must be 'stable_high_alpha_with_mean_skill_guard' or "
            "'support_uncertainty_stable_high_alpha_with_dual_guard'"
        )
    if selection_rule == "support_uncertainty_stable_high_alpha_with_dual_guard":
        min_skill_tolerance_floor = max(float(min_skill_tolerance_floor), 0.025)

    sorted_alphas = sorted(float(alpha) for alpha in alphas)

    def _metric(alpha: float, key: str) -> float:
        value = float(per_alpha_results[alpha].get(key, float("-inf")))
        return value if np.isfinite(value) else float("-inf")

    def _raw_key(alpha: float) -> Tuple[float, float, float]:
        return (_metric(alpha, "min_skill"), _metric(alpha, "mean_skill"), -float(alpha))

    if best_alpha_raw is None:
        best_alpha_raw = max(sorted_alphas, key=_raw_key)
    else:
        best_alpha_raw = float(best_alpha_raw)
        if best_alpha_raw not in per_alpha_results:
            raise ValueError(f"best_alpha_raw={best_alpha_raw} is missing from per_alpha_results")
    best_min_skill = _metric(best_alpha_raw, "min_skill")
    best_mean_skill = _metric(best_alpha_raw, "mean_skill")
    best_surface_skill = _metric(best_alpha_raw, "surface_skill")
    best_rootzone_skill = _metric(best_alpha_raw, "rootzone_skill")

    lower_min_skills = [
        _metric(alpha, "min_skill")
        for alpha in sorted_alphas
        if alpha != best_alpha_raw and _metric(alpha, "min_skill") <= best_min_skill
    ]
    support_score_gap_to_best = (
        best_min_skill - max(lower_min_skills)
        if lower_min_skills and np.isfinite(best_min_skill)
        else 0.0
    )
    stability_tolerance = max(
        float(min_skill_tolerance_floor),
        float(min_skill_gap_fraction) * max(0.0, float(support_score_gap_to_best)),
        max(0.0, float(paired_support_se_capped)),
    )

    stable_candidate_alphas = []
    for alpha in sorted_alphas:
        min_skill = _metric(alpha, "min_skill")
        mean_skill = _metric(alpha, "mean_skill")
        if not np.isfinite(min_skill):
            continue
        if best_min_skill - min_skill > stability_tolerance + 1e-12:
            continue
        if mean_skill < best_mean_skill - float(mean_skill_tolerance) - 1e-12:
            continue
        if selection_rule == "support_uncertainty_stable_high_alpha_with_dual_guard":
            surface_skill = _metric(alpha, "surface_skill")
            rootzone_skill = _metric(alpha, "rootzone_skill")
            if (
                np.isfinite(best_surface_skill)
                and surface_skill < best_surface_skill - float(variable_skill_tolerance) - 1e-12
            ):
                continue
            if (
                np.isfinite(best_rootzone_skill)
                and rootzone_skill < best_rootzone_skill - float(variable_skill_tolerance) - 1e-12
            ):
                continue
        stable_candidate_alphas.append(alpha)
    if not stable_candidate_alphas:
        stable_candidate_alphas = [best_alpha_raw]
    selected_alpha = max(stable_candidate_alphas)
    selection_margin = best_min_skill - _metric(selected_alpha, "min_skill")

    return {
        "selection_rule": selection_rule,
        "selected_alpha": float(selected_alpha),
        "best_alpha_raw": float(best_alpha_raw),
        "stable_candidate_alphas": [float(alpha) for alpha in stable_candidate_alphas],
        "selection_margin": float(selection_margin),
        "stability_tolerance": float(stability_tolerance),
        "support_score_gap_to_best": float(support_score_gap_to_best),
        "mean_skill_tolerance": float(mean_skill_tolerance),
        "variable_skill_tolerance": float(variable_skill_tolerance),
        "paired_support_se_capped": max(0.0, float(paired_support_se_capped)),
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
    alpha_selection_objective: str = "transfer_safe_score",
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
    allowed_objectives = {"transfer_safe_score", "dual_variable_cvar_safe_score"}
    if alpha_selection_objective not in allowed_objectives:
        raise ValueError(
            "alpha_selection_objective must be one of "
            f"{sorted(allowed_objectives)}, got {alpha_selection_objective!r}"
        )

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
                dual_variable_cvar_safe_score = float("-inf")
                worst_balanced = float("nan")
                mean_balanced = float("nan")
                pos_rate = float("nan")
                dispersion = float("nan")
                neg_rootzone_count = 0
                worst_region_surface_skill = float("nan")
                worst_region_rootzone_skill = float("nan")
                dual_variable_non_degradation = False
            else:
                worst_balanced = float(np.min(valid_balanced))
                mean_balanced = float(np.mean(valid_balanced))
                dispersion = float(np.std(valid_balanced)) if len(valid_balanced) >= 2 else 0.0

                # positive_improvement_rate: fraction of (region, variable) with skill > 0
                all_skills = []
                surface_skills = []
                rootzone_skills = []
                neg_rootzone_count = 0
                for region in region_names:
                    rs = region_skills[region]
                    for key in ["surface_skill", "rootzone_skill"]:
                        val = rs[key]
                        if np.isfinite(val):
                            all_skills.append(val)
                            if key == "surface_skill":
                                surface_skills.append(val)
                            if key == "rootzone_skill":
                                rootzone_skills.append(val)
                            if key == "rootzone_skill" and val < 0:
                                neg_rootzone_count += 1

                pos_rate = float(np.mean([1.0 if s > 0 else 0.0 for s in all_skills])) if all_skills else 0.0
                penalty_neg_rootzone = 0.2 * float(neg_rootzone_count)
                worst_region_surface_skill = float(np.min(surface_skills)) if surface_skills else float("nan")
                worst_region_rootzone_skill = float(np.min(rootzone_skills)) if rootzone_skills else float("nan")
                dual_variable_non_degradation = bool(
                    np.isfinite(worst_region_surface_skill)
                    and np.isfinite(worst_region_rootzone_skill)
                    and worst_region_surface_skill >= 0.0
                    and worst_region_rootzone_skill >= 0.0
                )

                score = (
                    0.50 * worst_balanced
                    + 0.25 * mean_balanced
                    + 0.15 * pos_rate
                    + 0.10 * (1.0 - min(dispersion, 1.0))
                    - penalty_neg_rootzone
                )
                tail = (
                    float(np.mean([worst_region_surface_skill, worst_region_rootzone_skill]))
                    if np.isfinite(worst_region_surface_skill) and np.isfinite(worst_region_rootzone_skill)
                    else float("-inf")
                )
                dual_variable_cvar_safe_score = (
                    0.60 * tail
                    + 0.25 * mean_balanced
                    + 0.15 * pos_rate
                    - 0.20 * max(0.0, -worst_region_surface_skill)
                    - 0.20 * max(0.0, -worst_region_rootzone_skill)
                )
                if not dual_variable_non_degradation:
                    dual_variable_cvar_safe_score -= 1.0

            # Prompt collapse penalty
            if prompt_quality_metrics:
                pc_mean = prompt_quality_metrics.get("prompt_pairwise_cosine_distance_mean", 1.0)
                if pc_mean < 0.01:
                    score = float("-inf")
                    dual_variable_cvar_safe_score = float("-inf")

            trace_rows.append({
                "alpha_surface": alpha_s,
                "alpha_rootzone": alpha_r,
                "transfer_safe_score": score,
                "dual_variable_cvar_safe_score": dual_variable_cvar_safe_score,
                "dual_variable_non_degradation": dual_variable_non_degradation,
                "worst_region_surface_skill": worst_region_surface_skill,
                "worst_region_rootzone_skill": worst_region_rootzone_skill,
                "worst_region_balanced_skill": worst_balanced,
                "mean_region_balanced_skill": mean_balanced,
                "positive_improvement_rate": pos_rate,
                "skill_dispersion": dispersion,
                "neg_rootzone_count": neg_rootzone_count,
            })

            objective_score = (
                dual_variable_cvar_safe_score
                if alpha_selection_objective == "dual_variable_cvar_safe_score"
                else score
            )

            if objective_score > best_score:
                best_score = objective_score
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

    best_trace = next(
        (
            row
            for row in trace_rows
            if row["alpha_surface"] == best_alpha_s and row["alpha_rootzone"] == best_alpha_r
        ),
        {},
    )
    selected_surface_region_skills = {
        region: float(skills.get("surface_skill", float("nan")))
        for region, skills in best_region_skills.items()
    }
    selected_rootzone_region_skills = {
        region: float(skills.get("rootzone_skill", float("nan")))
        for region, skills in best_region_skills.items()
    }
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
        "alpha_selection_objective": alpha_selection_objective,
        "selection_score": best_score,
        "dual_variable_cvar_score": float(
            best_trace.get("dual_variable_cvar_safe_score", float("nan"))
        ),
        "dual_variable_non_degradation": bool(
            best_trace.get("dual_variable_non_degradation", False)
        ),
        "worst_region_surface_skill": float(best_trace.get("worst_region_surface_skill", float("nan"))),
        "worst_region_rootzone_skill": float(best_trace.get("worst_region_rootzone_skill", float("nan"))),
        "min_skill": float(np.min([skill_s, skill_r])) if np.isfinite(skill_s) and np.isfinite(skill_r) else float("nan"),
        "mean_skill": float(np.mean([skill_s, skill_r])) if np.isfinite(skill_s) and np.isfinite(skill_r) else float("nan"),
        "region_variable_skills": best_region_skills,
        "selected_surface_region_skills": selected_surface_region_skills,
        "selected_rootzone_region_skills": selected_rootzone_region_skills,
        "selected_trace": best_trace,
        "region_names": region_names,
        "selection_trace": trace_rows,
        "calibration_mode": "region_aware_2d",
    }
