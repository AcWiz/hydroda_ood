"""M3_15 source-safe physics coefficient-delta interpolation.

M3_15 keeps M3_1 as the anchor and treats the trained physics coefficient-delta
checkpoint as an optional source-selected branch. Eta is selected only on
source_val records. Eta zero is exact M3_1 identity and target_eval is refused
unless the source gate passes.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


VARIABLES = ("surface", "rootzone")
M3_15_METHOD_ID = "M3_15_m31_anchored_source_safe_phys_coeff_delta"
M3_15_SELECTION_SCHEMA = "m3_15_phys_coeff_delta_source_gate_v1"
M3_15_ROUTER_SCHEMA = "m3_15_phys_coeff_delta_router_v1"
M3_15_SOURCE_GATE_REPORT_SCHEMA = "m3_15_source_gate_report_v1"
M3_1_DUAL_CVAR_ANCHOR = 0.446573390549
SOURCE_ROLES_FOR_SELECTION = {"source_val", "source_val_pseudo_query"}
FORBIDDEN_TARGET_ROLES = {
    "target_context",
    "target_support",
    "target_val",
    "target_eval",
    "target_query",
    "target_train",
    "target_full_train",
}


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_role(record: Mapping[str, Any]) -> str:
    return str(record.get("split_role") or record.get("query_role") or "")


def _require_roles(records: Sequence[Mapping[str, Any]], allowed: set[str], *, purpose: str) -> None:
    bad = sorted({_record_role(record) for record in records if _record_role(record) not in allowed})
    if bad:
        raise ValueError(f"{purpose} accepts only {sorted(allowed)} records; got roles {bad}")


def _require_no_target_records(records: Sequence[Mapping[str, Any]], *, purpose: str) -> None:
    bad = []
    for record in records:
        role = _record_role(record)
        adaptation_setting = str(record.get("adaptation_setting", ""))
        if role in FORBIDDEN_TARGET_ROLES or adaptation_setting in FORBIDDEN_TARGET_ROLES:
            bad.append((role, adaptation_setting))
    if bad:
        raise ValueError(f"{purpose} refuses target-side records: {bad[:3]}")


def _as_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    return array


def _mask_from_record(record: Mapping[str, Any]) -> np.ndarray:
    for key in ("metric_mask", "loss_mask", "region_mask", "active_region_mask"):
        if key in record and record[key] is not None:
            return (np.asarray(record[key]) > 0.5).astype(bool)
    raise KeyError("record missing metric/loss/region mask")


def _weights_from_record(record: Mapping[str, Any], mask: np.ndarray) -> np.ndarray:
    if "latitude_weight" not in record or record["latitude_weight"] is None:
        return np.ones_like(mask, dtype=np.float64)
    weights = np.asarray(record["latitude_weight"], dtype=np.float64)
    count = min(weights.size, mask.size)
    out = np.ones(mask.size, dtype=np.float64)
    out[:count] = weights.reshape(-1)[:count]
    return out.reshape(mask.shape)


def _masked_weighted_rmse(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(truth, dtype=np.float64).reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    count = min(p.size, t.size, m.size, w.size)
    p = p[:count]
    t = t[:count]
    m = m[:count]
    w = w[:count]
    valid = m & np.isfinite(p) & np.isfinite(t) & np.isfinite(w) & (w > 0.0)
    if not np.any(valid):
        return float("nan")
    err2 = (p[valid] - t[valid]) ** 2
    return float(math.sqrt(float(np.sum(err2 * w[valid]) / np.sum(w[valid]))))


def _relative_delta(value: float, base: float) -> float:
    value_f = float(value)
    base_f = float(base)
    if not math.isfinite(value_f) or not math.isfinite(base_f) or abs(base_f) < 1e-12:
        return float("nan")
    return float((value_f - base_f) / abs(base_f))


def _safe_delta(value: float, base: float) -> float:
    value_f = float(value)
    base_f = float(base)
    if not math.isfinite(value_f) or not math.isfinite(base_f):
        return float("nan")
    return float(value_f - base_f)


def _source_region(record: Mapping[str, Any]) -> str:
    return str(
        record.get("sample_region_id")
        or record.get("source_region_id")
        or record.get("pseudo_target_region")
        or record.get("target_region_id")
        or "unknown"
    )


def _season(record: Mapping[str, Any]) -> str:
    season = record.get("season")
    if season:
        return str(season)
    month = int(record.get("month") or 1)
    if month in {12, 1, 2}:
        return "DJF"
    if month in {3, 4, 5}:
        return "MAM"
    if month in {6, 7, 8}:
        return "JJA"
    return "SON"


def apply_m3_15_interpolation(
    sample: Mapping[str, Any],
    m3_1_pred: Mapping[str, Any],
    phys_coeff_pred: Mapping[str, Any],
    *,
    eta_surface: float,
    eta_rootzone: float,
) -> dict[str, Any]:
    """Interpolate predictions with eta zero as exact M3_1 identity."""
    eta_s = float(eta_surface)
    eta_r = float(eta_rootzone)
    if eta_s < 0.0 or eta_s > 1.0 or eta_r < 0.0 or eta_r > 1.0:
        raise ValueError("M3_15 eta values must be in [0, 1]")
    m3_s = _as_array(m3_1_pred["pred_increment_surface"], name="m3_1 pred_increment_surface")
    m3_r = _as_array(m3_1_pred["pred_increment_rootzone"], name="m3_1 pred_increment_rootzone")
    phys_s = _as_array(phys_coeff_pred["pred_increment_surface"], name="phys pred_increment_surface")
    phys_r = _as_array(phys_coeff_pred["pred_increment_rootzone"], name="phys pred_increment_rootzone")
    if m3_s.shape != phys_s.shape or m3_r.shape != phys_r.shape:
        raise ValueError("M3_15 interpolation prediction shape mismatch")
    pred_s = ((1.0 - eta_s) * m3_s + eta_s * phys_s).astype(np.float32)
    pred_r = ((1.0 - eta_r) * m3_r + eta_r * phys_r).astype(np.float32)
    forecast_s = np.asarray(sample["forecast_surface"], dtype=np.float32)
    forecast_r = np.asarray(sample["forecast_rootzone"], dtype=np.float32)
    return {
        "pred_increment_surface": pred_s,
        "pred_increment_rootzone": pred_r,
        "pred_analysis_surface": (forecast_s + pred_s).astype(np.float32),
        "pred_analysis_rootzone": (forecast_r + pred_r).astype(np.float32),
        "m3_15_summary": {
            "schema_version": M3_15_ROUTER_SCHEMA,
            "method_id": M3_15_METHOD_ID,
            "eta_surface": eta_s,
            "eta_rootzone": eta_r,
            "eta_zero_identity": bool(eta_s == 0.0 and eta_r == 0.0),
            "formula": "pred_v=(1-eta_v)*pred_m3_1_v+eta_v*pred_phys_coeff_v",
            "target_eval_usage": "final_eval_only_no_selection",
        },
    }


class M315PhysCoeffDeltaPredictor:
    """Predictor wrapper applying fixed M3_15 source-selected interpolation."""

    method_name = M3_15_METHOD_ID

    def __init__(
        self,
        m3_1_predictor: Any,
        phys_coeff_predictor: Any,
        *,
        eta_surface: float,
        eta_rootzone: float,
        selection: Mapping[str, Any] | None = None,
    ) -> None:
        self.m3_1_predictor = m3_1_predictor
        self.phys_coeff_predictor = phys_coeff_predictor
        self.eta_surface = float(eta_surface)
        self.eta_rootzone = float(eta_rootzone)
        self.selection = dict(selection or {})
        self.metadata = {
            "schema_version": M3_15_ROUTER_SCHEMA,
            "method_id": M3_15_METHOD_ID,
            "base_anchor": "M3_1_hyperda_trust_medium",
            "phys_coeff_branch": "M3_15_phys_operator_residual",
            "selected_eta_surface": self.eta_surface,
            "selected_eta_rootzone": self.eta_rootzone,
            "selection_source": self.selection.get("selection_source", "manual_or_source_val"),
            "target_val_usage": "unused_in_main_protocol",
            "target_eval_usage": "final_eval_only_no_selection",
            "target_eval_selection_usage": "forbidden",
        }

    def predict(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        return apply_m3_15_interpolation(
            sample,
            self.m3_1_predictor.predict(sample),
            self.phys_coeff_predictor.predict(sample),
            eta_surface=self.eta_surface,
            eta_rootzone=self.eta_rootzone,
        )


def _empty_bucket() -> dict[str, list[float]]:
    return {
        "surface_sqerr": [],
        "surface_weight": [],
        "surface_forecast_sqerr": [],
        "surface_forecast_weight": [],
        "rootzone_sqerr": [],
        "rootzone_weight": [],
        "rootzone_forecast_sqerr": [],
        "rootzone_forecast_weight": [],
    }


def _accumulate_bucket(
    buckets: dict[str, dict[str, list[float]]],
    key: str,
    record: Mapping[str, Any],
    pred: Mapping[str, Any],
) -> None:
    bucket = buckets.setdefault(key, _empty_bucket())
    mask = _mask_from_record(record)
    weights = _weights_from_record(record, mask)
    for variable in VARIABLES:
        pred_arr = np.asarray(pred[f"pred_increment_{variable}"], dtype=np.float64)
        truth = np.asarray(record[f"increment_{variable}"], dtype=np.float64)
        p = pred_arr.reshape(-1)
        t = truth.reshape(-1)
        m = mask.reshape(-1)
        w = weights.reshape(-1)
        count = min(p.size, t.size, m.size, w.size)
        valid = (
            m[:count]
            & np.isfinite(p[:count])
            & np.isfinite(t[:count])
            & np.isfinite(w[:count])
            & (w[:count] > 0.0)
        )
        if np.any(valid):
            bucket[f"{variable}_sqerr"].append(float(np.sum(((p[:count][valid] - t[:count][valid]) ** 2) * w[:count][valid])))
            bucket[f"{variable}_weight"].append(float(np.sum(w[:count][valid])))
            bucket[f"{variable}_forecast_sqerr"].append(float(np.sum((t[:count][valid] ** 2) * w[:count][valid])))
            bucket[f"{variable}_forecast_weight"].append(float(np.sum(w[:count][valid])))


def _summarize_buckets(buckets: Mapping[str, Mapping[str, Sequence[float]]]) -> dict[str, Any]:
    per_key: dict[str, dict[str, float]] = {}
    global_sums = _empty_bucket()
    for key, bucket in buckets.items():
        row: dict[str, float] = {}
        for variable in VARIABLES:
            sqerr = float(sum(bucket.get(f"{variable}_sqerr", [])))
            weight = float(sum(bucket.get(f"{variable}_weight", [])))
            rmse = float(math.sqrt(sqerr / weight)) if weight > 0.0 else float("nan")
            fcst_sqerr = float(sum(bucket.get(f"{variable}_forecast_sqerr", [])))
            fcst_weight = float(sum(bucket.get(f"{variable}_forecast_weight", [])))
            forecast_rmse = float(math.sqrt(fcst_sqerr / fcst_weight)) if fcst_weight > 0.0 else float("nan")
            skill = (
                float(1.0 - rmse / forecast_rmse)
                if math.isfinite(rmse) and math.isfinite(forecast_rmse) and forecast_rmse > 0.0
                else float("nan")
            )
            row[f"{variable}_analysis_rmse_latw"] = rmse
            row[f"{variable}_forecast_rmse_latw"] = forecast_rmse
            row[f"{variable}_skill"] = skill
            global_sums[f"{variable}_sqerr"].append(sqerr)
            global_sums[f"{variable}_weight"].append(weight)
            global_sums[f"{variable}_forecast_sqerr"].append(fcst_sqerr)
            global_sums[f"{variable}_forecast_weight"].append(fcst_weight)
        per_key[key] = row
    summary = {}
    for variable in VARIABLES:
        sqerr = float(sum(global_sums[f"{variable}_sqerr"]))
        weight = float(sum(global_sums[f"{variable}_weight"]))
        fcst_sqerr = float(sum(global_sums[f"{variable}_forecast_sqerr"]))
        fcst_weight = float(sum(global_sums[f"{variable}_forecast_weight"]))
        rmse = float(math.sqrt(sqerr / weight)) if weight > 0.0 else float("nan")
        forecast_rmse = float(math.sqrt(fcst_sqerr / fcst_weight)) if fcst_weight > 0.0 else float("nan")
        summary[variable] = {
            "analysis_rmse_latw": rmse,
            "forecast_rmse_latw": forecast_rmse,
            "skill": (
                float(1.0 - rmse / forecast_rmse)
                if math.isfinite(rmse) and math.isfinite(forecast_rmse) and forecast_rmse > 0.0
                else float("nan")
            ),
            "weight": weight,
        }
    return {"summary": summary, "by_key": per_key}


def _dual_cvar(summary: Mapping[str, Any], by_region: Mapping[str, Any]) -> dict[str, float]:
    region_balanced: list[float] = []
    surface_skills: list[float] = []
    rootzone_skills: list[float] = []
    all_skills: list[float] = []
    for row in by_region.values():
        skill_s = float(row.get("surface_skill", float("nan")))
        skill_r = float(row.get("rootzone_skill", float("nan")))
        if math.isfinite(skill_s):
            surface_skills.append(skill_s)
            all_skills.append(skill_s)
        if math.isfinite(skill_r):
            rootzone_skills.append(skill_r)
            all_skills.append(skill_r)
        if math.isfinite(skill_s) and math.isfinite(skill_r):
            region_balanced.append(float(np.mean([skill_s, skill_r])))
    if not region_balanced or not surface_skills or not rootzone_skills:
        safe_score = float("-inf")
        worst_surface = float("nan")
        worst_rootzone = float("nan")
        mean_balanced = float("nan")
        pos_rate = float("nan")
        non_degradation = False
    else:
        worst_surface = float(np.min(surface_skills))
        worst_rootzone = float(np.min(rootzone_skills))
        mean_balanced = float(np.mean(region_balanced))
        pos_rate = float(np.mean([1.0 if skill > 0.0 else 0.0 for skill in all_skills]))
        non_degradation = bool(worst_surface >= 0.0 and worst_rootzone >= 0.0)
        tail = float(np.mean([worst_surface, worst_rootzone]))
        safe_score = (
            0.60 * tail
            + 0.25 * mean_balanced
            + 0.15 * pos_rate
            - 0.20 * max(0.0, -worst_surface)
            - 0.20 * max(0.0, -worst_rootzone)
        )
        if not non_degradation:
            safe_score -= 1.0
    return {
        "dual_variable_cvar_safe_score": float(safe_score),
        "dual_variable_cvar_score": float(safe_score),
        "worst_region_surface_skill": float(worst_surface),
        "worst_region_rootzone_skill": float(worst_rootzone),
        "mean_region_balanced_skill": float(mean_balanced),
        "positive_improvement_rate": float(pos_rate),
        "dual_variable_non_degradation": bool(non_degradation),
        "score_source": "source_val_region_variable_skill_formula",
    }


def _candidate_result(
    *,
    eta_surface: float,
    eta_rootzone: float,
    base_summary: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
    base_region: Mapping[str, Any],
    candidate_region: Mapping[str, Any],
    base_season: Mapping[str, Any],
    candidate_season: Mapping[str, Any],
    anchor_dual_cvar: float,
) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for variable in VARIABLES:
        base_rmse = float((base_summary.get(variable, {}) or {}).get("analysis_rmse_latw", float("nan")))
        cand_rmse = float((candidate_summary.get(variable, {}) or {}).get("analysis_rmse_latw", float("nan")))
        deltas[variable] = {
            "analysis_rmse_latw": _safe_delta(cand_rmse, base_rmse),
            "analysis_rmse_latw_relative": _relative_delta(cand_rmse, base_rmse),
        }
    region_deltas: dict[str, dict[str, float]] = {}
    for key, base_row in base_region.items():
        cand_row = candidate_region.get(key, {})
        region_deltas[key] = {}
        for variable in VARIABLES:
            metric = f"{variable}_analysis_rmse_latw"
            region_deltas[key][variable] = _relative_delta(cand_row.get(metric, float("nan")), base_row.get(metric, float("nan")))
    season_deltas: dict[str, dict[str, float]] = {}
    for key, base_row in base_season.items():
        cand_row = candidate_season.get(key, {})
        season_deltas[key] = {}
        for variable in VARIABLES:
            metric = f"{variable}_analysis_rmse_latw"
            season_deltas[key][variable] = _relative_delta(cand_row.get(metric, float("nan")), base_row.get(metric, float("nan")))
    finite_region = [v for row in region_deltas.values() for v in row.values() if math.isfinite(float(v))]
    finite_season = [v for row in season_deltas.values() for v in row.values() if math.isfinite(float(v))]
    cvar = _dual_cvar(candidate_summary, candidate_region)
    return {
        "eta_surface": float(eta_surface),
        "eta_rootzone": float(eta_rootzone),
        "base_summary": base_summary,
        "summary": candidate_summary,
        "base_region_summary": base_region,
        "region_summary": candidate_region,
        "base_season_summary": base_season,
        "season_summary": candidate_season,
        "region_rmse_relative_deltas": region_deltas,
        "season_rmse_relative_deltas": season_deltas,
        "max_source_region_rmse_relative_degrade": float(max(finite_region)) if finite_region else float("nan"),
        "max_source_season_rmse_relative_degrade": float(max(finite_season)) if finite_season else float("nan"),
        "dual_variable_cvar": cvar,
        "dual_variable_cvar_delta_vs_anchor": _safe_delta(cvar["dual_variable_cvar_safe_score"], anchor_dual_cvar),
        "deltas": deltas,
    }


def evaluate_record_stream_for_eta_pairs(
    records: Iterable[Mapping[str, Any]],
    *,
    eta_pairs: Sequence[tuple[float, float]],
    anchor_dual_cvar: float = M3_1_DUAL_CVAR_ANCHOR,
) -> tuple[list[dict[str, Any]], str]:
    pairs = [(float(a), float(b)) for a, b in eta_pairs]
    if not pairs:
        raise ValueError("M3_15 eta evaluation requires at least one eta pair")
    base_global: dict[str, dict[str, list[float]]] = {}
    base_region: dict[str, dict[str, list[float]]] = {}
    base_season: dict[str, dict[str, list[float]]] = {}
    candidate_global = [{} for _pair in pairs]
    candidate_region = [{} for _pair in pairs]
    candidate_season = [{} for _pair in pairs]
    hash_rows: list[dict[str, Any]] = []
    n_records = 0
    for record in records:
        _require_roles([record], SOURCE_ROLES_FOR_SELECTION, purpose="M3_15 eta selection")
        _require_no_target_records([record], purpose="M3_15 eta selection")
        for key in (
            "pred_m3_1_increment_surface",
            "pred_m3_1_increment_rootzone",
            "pred_phys_coeff_increment_surface",
            "pred_phys_coeff_increment_rootzone",
        ):
            if key not in record:
                raise KeyError(f"M3_15 source_val record missing {key!r}")
        n_records += 1
        hash_rows.append(
            {
                "sample_idx": record.get("sample_idx"),
                "split_role": record.get("split_role"),
                "query_time_index": record.get("query_time_index"),
                "query_date": record.get("query_date"),
                "sample_region_id": record.get("sample_region_id"),
                "target_region_id": record.get("target_region_id"),
            }
        )
        base_pred = {
            "pred_increment_surface": np.asarray(record["pred_m3_1_increment_surface"], dtype=np.float32),
            "pred_increment_rootzone": np.asarray(record["pred_m3_1_increment_rootzone"], dtype=np.float32),
        }
        phys_pred = {
            "pred_increment_surface": np.asarray(record["pred_phys_coeff_increment_surface"], dtype=np.float32),
            "pred_increment_rootzone": np.asarray(record["pred_phys_coeff_increment_rootzone"], dtype=np.float32),
        }
        region_key = _source_region(record)
        season_key = _season(record)
        _accumulate_bucket(base_global, "global", record, base_pred)
        _accumulate_bucket(base_region, region_key, record, base_pred)
        _accumulate_bucket(base_season, season_key, record, base_pred)
        for pair_idx, (eta_s, eta_r) in enumerate(pairs):
            routed = apply_m3_15_interpolation(
                record,
                base_pred,
                phys_pred,
                eta_surface=eta_s,
                eta_rootzone=eta_r,
            )
            _accumulate_bucket(candidate_global[pair_idx], "global", record, routed)
            _accumulate_bucket(candidate_region[pair_idx], region_key, record, routed)
            _accumulate_bucket(candidate_season[pair_idx], season_key, record, routed)
    if n_records == 0:
        raise ValueError("M3_15 eta evaluation received zero source_val records")
    base_global_summary = _summarize_buckets(base_global)["summary"]
    base_region_summary = _summarize_buckets(base_region)["by_key"]
    base_season_summary = _summarize_buckets(base_season)["by_key"]
    results = []
    for pair_idx, (eta_s, eta_r) in enumerate(pairs):
        cand_global = _summarize_buckets(candidate_global[pair_idx])["summary"]
        cand_region = _summarize_buckets(candidate_region[pair_idx])["by_key"]
        cand_season = _summarize_buckets(candidate_season[pair_idx])["by_key"]
        results.append(
            _candidate_result(
                eta_surface=eta_s,
                eta_rootzone=eta_r,
                base_summary=base_global_summary,
                candidate_summary=cand_global,
                base_region=base_region_summary,
                candidate_region=cand_region,
                base_season=base_season_summary,
                candidate_season=cand_season,
                anchor_dual_cvar=anchor_dual_cvar,
            )
        )
    return results, _stable_hash(hash_rows)


def _source_gate_report(
    result: Mapping[str, Any],
    *,
    anchor_dual_cvar: float,
    min_dual_cvar_delta: float,
    min_best_variable_rmse_relative_improve: float,
    max_other_variable_rmse_relative_degrade: float,
    max_region_rmse_relative_degrade: float,
    max_season_rmse_relative_degrade: float,
) -> dict[str, Any]:
    eta_positive = bool(float(result["eta_surface"]) > 0.0 or float(result["eta_rootzone"]) > 0.0)
    safe_score = float((result.get("dual_variable_cvar", {}) or {}).get("dual_variable_cvar_safe_score", float("nan")))
    cvar_ok = math.isfinite(safe_score) and safe_score >= float(anchor_dual_cvar) + float(min_dual_cvar_delta)
    rel = {
        variable: float((result.get("deltas", {}).get(variable, {}) or {}).get("analysis_rmse_latw_relative", float("nan")))
        for variable in VARIABLES
    }
    improvements = {variable: -value for variable, value in rel.items() if math.isfinite(value)}
    best_improve = max(improvements.values()) if improvements else float("nan")
    improved = [variable for variable, value in rel.items() if math.isfinite(value) and value <= -float(min_best_variable_rmse_relative_improve)]
    if improved:
        other_ok = all(
            value <= float(max_other_variable_rmse_relative_degrade)
            for variable, value in rel.items()
            if variable not in set(improved) and math.isfinite(value)
        )
    else:
        other_ok = False
    region_degrade = float(result.get("max_source_region_rmse_relative_degrade", float("nan")))
    season_degrade = float(result.get("max_source_season_rmse_relative_degrade", float("nan")))
    region_ok = math.isfinite(region_degrade) and region_degrade <= float(max_region_rmse_relative_degrade)
    season_ok = math.isfinite(season_degrade) and season_degrade <= float(max_season_rmse_relative_degrade)
    return {
        "eta_positive": eta_positive,
        "dual_variable_cvar_ok": cvar_ok,
        "variable_rmse_improve_ok": bool(improved and other_ok),
        "source_region_rmse_non_degrade_ok": region_ok,
        "source_season_rmse_non_degrade_ok": season_ok,
        "source_gate_pass": bool(eta_positive and cvar_ok and improved and other_ok and region_ok and season_ok),
        "thresholds": {
            "m3_1_dual_cvar_anchor": float(anchor_dual_cvar),
            "min_dual_cvar_delta": float(min_dual_cvar_delta),
            "min_best_variable_rmse_relative_improve": float(min_best_variable_rmse_relative_improve),
            "max_other_variable_rmse_relative_degrade": float(max_other_variable_rmse_relative_degrade),
            "max_region_rmse_relative_degrade": float(max_region_rmse_relative_degrade),
            "max_season_rmse_relative_degrade": float(max_season_rmse_relative_degrade),
            "requires_at_least_one_positive_eta": True,
        },
        "best_variable_rmse_relative_improve": float(best_improve),
        "variable_rmse_relative_deltas": rel,
        "max_source_region_rmse_relative_degrade": region_degrade,
        "max_source_season_rmse_relative_degrade": season_degrade,
        "dual_variable_cvar_safe_score": safe_score,
    }


def select_eta_from_source_val(
    records: Iterable[Mapping[str, Any]],
    *,
    eta_grid: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 1.0),
    anchor_dual_cvar: float = M3_1_DUAL_CVAR_ANCHOR,
    min_dual_cvar_delta: float = 0.001,
    min_best_variable_rmse_relative_improve: float = 0.001,
    max_other_variable_rmse_relative_degrade: float = 0.0005,
    max_region_rmse_relative_degrade: float = 0.003,
    max_season_rmse_relative_degrade: float = 0.003,
) -> dict[str, Any]:
    eta_values = [float(eta) for eta in eta_grid]
    eta_pairs = [(eta_s, eta_r) for eta_s in eta_values for eta_r in eta_values]
    evaluated, records_hash = evaluate_record_stream_for_eta_pairs(
        records,
        eta_pairs=eta_pairs,
        anchor_dual_cvar=anchor_dual_cvar,
    )
    passing = []
    for result in evaluated:
        report = _source_gate_report(
            result,
            anchor_dual_cvar=anchor_dual_cvar,
            min_dual_cvar_delta=min_dual_cvar_delta,
            min_best_variable_rmse_relative_improve=min_best_variable_rmse_relative_improve,
            max_other_variable_rmse_relative_degrade=max_other_variable_rmse_relative_degrade,
            max_region_rmse_relative_degrade=max_region_rmse_relative_degrade,
            max_season_rmse_relative_degrade=max_season_rmse_relative_degrade,
        )
        result["source_gate_report"] = report
        result["source_gate_pass"] = bool(report["source_gate_pass"])
        if result["source_gate_pass"]:
            passing.append(result)
    if passing:
        selected = max(
            passing,
            key=lambda item: (
                float(item["dual_variable_cvar"]["dual_variable_cvar_safe_score"]),
                -max(
                    float(item["max_source_region_rmse_relative_degrade"]),
                    float(item["max_source_season_rmse_relative_degrade"]),
                ),
                -float(item["eta_surface"]) - float(item["eta_rootzone"]),
            ),
        )
        identity_diagnostic = False
    else:
        selected = next(
            (
                item
                for item in evaluated
                if float(item["eta_surface"]) == 0.0 and float(item["eta_rootzone"]) == 0.0
            ),
            evaluated[0],
        )
        identity_diagnostic = True
    selection = {
        "schema_version": M3_15_SELECTION_SCHEMA,
        "method_id": M3_15_METHOD_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "eta_grid": eta_values,
        "eta_pair_grid": [[float(a), float(b)] for a in eta_values for b in eta_values],
        "selected_eta_surface": float(selected["eta_surface"]),
        "selected_eta_rootzone": float(selected["eta_rootzone"]),
        "source_gate_pass": bool(selected.get("source_gate_pass", False)),
        "identity_diagnostic": bool(identity_diagnostic),
        "source_gate_report": selected.get("source_gate_report", {}),
        "selection_source": "source_val_only",
        "target_val_usage": "unused_in_main_protocol",
        "target_eval_usage": "not_used_for_eta_selection",
        "target_eval_final_policy": "run_once_only_if_source_gate_passes",
        "source_val_records_hash": records_hash,
        "selection_rule": {
            "anchor_method": "M3_1_hyperda_trust_medium",
            "m3_1_dual_cvar_anchor": float(anchor_dual_cvar),
            "min_dual_cvar_delta": float(min_dual_cvar_delta),
            "min_best_variable_rmse_relative_improve": float(min_best_variable_rmse_relative_improve),
            "max_other_variable_rmse_relative_degrade": float(max_other_variable_rmse_relative_degrade),
            "max_region_rmse_relative_degrade": float(max_region_rmse_relative_degrade),
            "max_season_rmse_relative_degrade": float(max_season_rmse_relative_degrade),
            "requires_at_least_one_positive_eta": True,
        },
        "selected": selected,
        "grid": evaluated,
    }
    selection["selection_hash"] = _stable_hash(
        {
            "schema_version": selection["schema_version"],
            "method_id": selection["method_id"],
            "eta_grid": selection["eta_grid"],
            "selected_eta_surface": selection["selected_eta_surface"],
            "selected_eta_rootzone": selection["selected_eta_rootzone"],
            "source_gate_pass": selection["source_gate_pass"],
            "identity_diagnostic": selection["identity_diagnostic"],
            "source_val_records_hash": selection["source_val_records_hash"],
            "selection_rule": selection["selection_rule"],
        }
    )
    return selection


def validate_router_metadata_no_target_selection(metadata: Mapping[str, Any]) -> None:
    if metadata.get("target_eval_usage") not in {
        "not_used_for_eta_selection",
        "final_eval_only_no_selection",
    }:
        raise ValueError("M3_15 metadata indicates target_eval selection")
    if metadata.get("target_val_usage") not in {"unused", "unused_in_main_protocol", None}:
        raise ValueError("M3_15 metadata indicates target_val usage")


def validate_source_gate_for_target_eval(selection: Mapping[str, Any]) -> None:
    validate_router_metadata_no_target_selection(selection)
    if selection.get("schema_version") != M3_15_SELECTION_SCHEMA:
        raise ValueError(f"Unsupported M3_15 source-gate schema: {selection.get('schema_version')!r}")
    if selection.get("method_id") != M3_15_METHOD_ID:
        raise ValueError("M3_15 source-gate method_id mismatch")
    if selection.get("selection_source") != "source_val_only":
        raise ValueError("M3_15 target_eval requires source_val-only eta selection")
    eta_positive = (
        float(selection.get("selected_eta_surface", 0.0)) > 0.0
        or float(selection.get("selected_eta_rootzone", 0.0)) > 0.0
    )
    if not eta_positive:
        raise ValueError("M3_15 target_eval refused: no positive eta was selected")
    if bool(selection.get("identity_diagnostic", False)):
        raise ValueError("M3_15 target_eval refused: identity diagnostic fallback")
    if not bool(selection.get("source_gate_pass", False)):
        raise ValueError("M3_15 target_eval refused: source gate did not pass")


def source_gate_report_from_selection(
    selection: Mapping[str, Any],
    *,
    target_region: str,
    K: int,
    seed: int,
    selection_path: str = "",
) -> dict[str, Any]:
    validate_router_metadata_no_target_selection(selection)
    target_eval_allowed = False
    try:
        validate_source_gate_for_target_eval(selection)
        target_eval_allowed = True
    except ValueError:
        target_eval_allowed = False
    return {
        "schema_version": M3_15_SOURCE_GATE_REPORT_SCHEMA,
        "method_id": M3_15_METHOD_ID,
        "target_region": target_region,
        "K": int(K),
        "seed": int(seed),
        "source_gate_pass": bool(selection.get("source_gate_pass", False)),
        "selected_eta_surface": float(selection.get("selected_eta_surface", 0.0)),
        "selected_eta_rootzone": float(selection.get("selected_eta_rootzone", 0.0)),
        "identity_diagnostic": bool(selection.get("identity_diagnostic", False)),
        "selection_hash": selection.get("selection_hash", ""),
        "selection_path": str(selection_path),
        "target_eval_allowed": bool(target_eval_allowed),
        "target_eval_usage": "final_eval_only_no_selection_if_allowed",
        "source_gate_report": selection.get("source_gate_report", {}),
        "required_for_target_eval": True,
    }
