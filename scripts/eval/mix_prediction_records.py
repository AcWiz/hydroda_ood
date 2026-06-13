#!/usr/bin/env python3
"""Offline source-safe prediction-record mixing for P2.8b.

This script reads saved source-side zero-shot and adapted prediction records,
mixes fixed or rule-derived rho values without rerunning adaptation, and
recomputes source-safe metrics. Target-side split roles are rejected in
calibration mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from hydroda.evaluation.harness import (
    _array_record_payload,
    _json_hash_update,
    _make_metric_row,
    _prediction_hash_payload,
    metric_rows_content_hash,
    metric_values_content_hash,
    prediction_record_array,
    summarize_metric_rows,
)
from hydroda.metrics.skill import (
    compute_variable_metrics,
    effective_mask_fraction,
    valid_pixel_count,
    weighted_analysis_skill_components,
    weighted_bias,
    weighted_corr,
    weighted_mae,
    weighted_mse,
)
from scripts.eval.calibrate_source_safe_guard import (
    FORBIDDEN_TARGET_EVAL_ROLES,
    SOURCE_SAFE_QUERY_ROLES,
    candidate_config_hash,
    candidate_id_from_config,
    compute_rho_for_policy,
)

_VARIABLES = {
    "surface": {
        "forecast": "forecast_surface",
        "analysis": "analysis_surface",
        "increment": "increment_surface",
        "pred_increment": "pred_increment_surface",
    },
    "rootzone": {
        "forecast": "forecast_rootzone",
        "analysis": "analysis_rootzone",
        "increment": "increment_rootzone",
        "pred_increment": "pred_increment_rootzone",
    },
}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No prediction records loaded from {path}")
    return rows


def _record_identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(record.get("sample_idx", -1)),
        int(record.get("query_time_index", -1)),
        str(record.get("query_date", "")),
        str(record.get("target_region_id", "")),
        str(record.get("sample_region_id", "")),
    )


def _validate_source_safe_record(record: Mapping[str, Any]) -> None:
    split_role = str(record.get("split_role", ""))
    adaptation_setting = str(record.get("adaptation_setting", ""))
    if split_role in FORBIDDEN_TARGET_EVAL_ROLES or adaptation_setting in FORBIDDEN_TARGET_EVAL_ROLES:
        raise ValueError(
            "P2.8b prediction-record mixing refuses target-side calibration inputs: "
            f"split_role={split_role!r} adaptation_setting={adaptation_setting!r}"
        )
    if split_role not in SOURCE_SAFE_QUERY_ROLES:
        raise ValueError(f"Prediction records must be source-safe source_val records; got {split_role!r}")


def _prediction_content_hash(records: Sequence[Mapping[str, Any]]) -> str:
    hasher = hashlib.sha256()
    for idx, record in enumerate(records):
        arrays = record["arrays"]
        sample = {
            "time_index": record.get("query_time_index", -1),
            "date_str": record.get("query_date", ""),
        }
        pred = {
            "pred_increment_surface": prediction_record_array(arrays["pred_increment_surface"]),
            "pred_increment_rootzone": prediction_record_array(arrays["pred_increment_rootzone"]),
        }
        _json_hash_update(hasher, _prediction_hash_payload(idx, sample, pred))
    return hasher.hexdigest()


def _array_equal_payload(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return np.array_equal(prediction_record_array(dict(left)), prediction_record_array(dict(right)))


def _mixed_record(k0: Mapping[str, Any], adapted: Mapping[str, Any], *, rho: float, candidate_id: str) -> dict[str, Any]:
    arrays0 = k0["arrays"]
    arrays1 = adapted["arrays"]
    mixed_arrays: dict[str, Any] = {}
    for key in (
        "forecast_surface",
        "forecast_rootzone",
        "analysis_surface",
        "analysis_rootzone",
        "increment_surface",
        "increment_rootzone",
        "metric_mask",
        "latitude_weight",
    ):
        mixed_arrays[key] = dict(arrays0[key])
    for variable in ("surface", "rootzone"):
        key = f"pred_increment_{variable}"
        zero_inc = prediction_record_array(arrays0[key])
        adapted_inc = prediction_record_array(arrays1[key])
        mixed = (zero_inc + float(rho) * (adapted_inc - zero_inc)).astype(np.float32)
        mixed_arrays[key] = _array_record_payload(mixed)

    out = dict(k0)
    out["kind"] = "offline_mixed"
    out["candidate_id"] = candidate_id
    out["adapt_mix_rho"] = float(rho)
    out["source_zero_shot_prediction_content_hash"] = k0.get("prediction_content_hash", "")
    out["source_adapted_prediction_content_hash"] = adapted.get("prediction_content_hash", "")
    out["arrays"] = mixed_arrays
    return out


def _records_to_metric_rows(records: Sequence[Mapping[str, Any]], *, candidate_id: str, rho: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    global_accum = {
        variable: {
            "model_sse": 0.0,
            "forecast_sse": 0.0,
            "n_pixels": 0,
            "model_sse_latw": 0.0,
            "forecast_sse_latw": 0.0,
            "weight_sum": 0.0,
        }
        for variable in _VARIABLES
    }
    first_sample: dict[str, Any] | None = None

    for idx, record in enumerate(records):
        arrays = record["arrays"]
        sample = {
            "date_str": record.get("query_date", ""),
            "time_index": int(record.get("query_time_index", -1)),
            "month": record.get("month", None),
            "season": record.get("season", ""),
            "country_id": record.get("country_id", ""),
            "target_region_id": record.get("target_region_id", ""),
            "sample_region_id": record.get("sample_region_id", ""),
            "active_region_ids": list(record.get("active_region_ids", [])),
            "split_manifest_sha256": record.get("split_manifest_sha256", ""),
            "adaptation_setting": record.get("adaptation_setting", "few_shot_k12"),
            "K": record.get("K", 12),
            "seed": int(record.get("seed", -1)),
        }
        for key, payload in arrays.items():
            sample[key] = prediction_record_array(payload)
        if first_sample is None:
            first_sample = sample
        mask = sample["metric_mask"]
        latw = sample["latitude_weight"]
        n_valid = valid_pixel_count(mask)
        mask_frac = effective_mask_fraction(mask)

        for variable, keys in _VARIABLES.items():
            pred_increment = sample[keys["pred_increment"]]
            forecast = sample[keys["forecast"]]
            true_analysis = sample[keys["analysis"]]
            true_increment = sample[keys["increment"]]
            pred_analysis = forecast + pred_increment
            metrics = compute_variable_metrics(
                pred_analysis=pred_analysis,
                true_analysis=true_analysis,
                forecast=forecast,
                pred_increment=pred_increment,
                true_increment=true_increment,
                mask=mask,
            )
            for metric_name, value in metrics.items():
                rows.append(_make_metric_row(
                    experiment_id=f"p2_8b_offline_mix_{candidate_id}",
                    method=f"offline_mix_rho_{rho:g}",
                    sample=sample,
                    sample_idx=idx,
                    split_role=str(record.get("split_role", "")),
                    split_file=str(record.get("split_file", "")),
                    mask_file="",
                    target_context_dates_hash=str(record.get("target_context_dates_hash", "")),
                    target_support_dates_hash=str(record.get("target_support_dates_hash", "")),
                    support_dates_hash=str(record.get("support_dates_hash", "")),
                    target_train_dates_hash=str(record.get("target_train_dates_hash", "")),
                    target_eval_dates_hash=str(record.get("target_eval_dates_hash", "")),
                    split_manifest_sha256=str(record.get("split_manifest_sha256", "")),
                    protocol_freeze_id=str(record.get("protocol_freeze_id", "")),
                    n_valid=n_valid,
                    mask_frac=mask_frac,
                    variable=variable,
                    metric_name=metric_name,
                    value=value,
                ))

            mse_m, mse_f = weighted_analysis_skill_components(
                pred_analysis=pred_analysis,
                true_analysis=true_analysis,
                forecast=forecast,
                mask=mask,
                latitude_weight=latw,
            )
            rmse_model_latw = float(np.sqrt(mse_m)) if np.isfinite(mse_m) and mse_m > 0 else float("nan")
            rmse_forecast_latw = float(np.sqrt(mse_f)) if np.isfinite(mse_f) and mse_f > 0 else float("nan")
            skill_latw = (
                float(1.0 - rmse_model_latw / rmse_forecast_latw)
                if np.isfinite(rmse_model_latw) and np.isfinite(rmse_forecast_latw) and rmse_forecast_latw > 0
                else float("nan")
            )
            inc_mse_latw = weighted_mse(pred_increment, true_increment, mask, latw)
            latw_metrics = {
                "analysis_mse_latw": mse_m,
                "analysis_rmse_latw": rmse_model_latw,
                "analysis_rmse_sqrt_before_time_avg_latw": rmse_model_latw,
                "analysis_skill_vs_forecast_latw": skill_latw,
                "increment_mse_latw": inc_mse_latw,
                "increment_rmse_latw": float(np.sqrt(inc_mse_latw)) if np.isfinite(inc_mse_latw) and inc_mse_latw > 0 else float("nan"),
                "increment_rmse_sqrt_before_time_avg_latw": float(np.sqrt(inc_mse_latw)) if np.isfinite(inc_mse_latw) and inc_mse_latw > 0 else float("nan"),
                "increment_mae_latw": weighted_mae(pred_increment, true_increment, mask, latw),
                "increment_bias_latw": weighted_bias(pred_increment, true_increment, mask, latw),
                "increment_corr_latw": weighted_corr(pred_increment, true_increment, mask, latw),
            }
            for metric_name, value in latw_metrics.items():
                rows.append(_make_metric_row(
                    experiment_id=f"p2_8b_offline_mix_{candidate_id}",
                    method=f"offline_mix_rho_{rho:g}",
                    sample=sample,
                    sample_idx=idx,
                    split_role=str(record.get("split_role", "")),
                    split_file=str(record.get("split_file", "")),
                    mask_file="",
                    target_context_dates_hash=str(record.get("target_context_dates_hash", "")),
                    target_support_dates_hash=str(record.get("target_support_dates_hash", "")),
                    support_dates_hash=str(record.get("support_dates_hash", "")),
                    target_train_dates_hash=str(record.get("target_train_dates_hash", "")),
                    target_eval_dates_hash=str(record.get("target_eval_dates_hash", "")),
                    split_manifest_sha256=str(record.get("split_manifest_sha256", "")),
                    protocol_freeze_id=str(record.get("protocol_freeze_id", "")),
                    n_valid=n_valid,
                    mask_frac=mask_frac,
                    variable=variable,
                    metric_name=metric_name,
                    value=value,
                ))

            valid = (mask > 0.5) & np.isfinite(pred_analysis) & np.isfinite(true_analysis) & np.isfinite(forecast)
            if valid.sum() > 0:
                global_accum[variable]["model_sse"] += float(np.sum((pred_analysis[valid] - true_analysis[valid]) ** 2))
                global_accum[variable]["forecast_sse"] += float(np.sum((forecast[valid] - true_analysis[valid]) ** 2))
                global_accum[variable]["n_pixels"] += int(valid.sum())
            valid_latw = valid & np.isfinite(latw) & (latw >= 0)
            if valid_latw.sum() > 0:
                w = latw[valid_latw]
                global_accum[variable]["model_sse_latw"] += float(np.sum(w * (pred_analysis[valid_latw] - true_analysis[valid_latw]) ** 2))
                global_accum[variable]["forecast_sse_latw"] += float(np.sum(w * (forecast[valid_latw] - true_analysis[valid_latw]) ** 2))
                global_accum[variable]["weight_sum"] += float(np.sum(w))

    if first_sample is not None:
        for variable, acc in global_accum.items():
            if acc["n_pixels"] > 0 and acc["forecast_sse"] > 0:
                global_skill = 1.0 - np.sqrt(acc["model_sse"] / acc["n_pixels"]) / np.sqrt(acc["forecast_sse"] / acc["n_pixels"])
            else:
                global_skill = float("nan")
            if acc["weight_sum"] > 0 and acc["forecast_sse_latw"] > 0:
                global_skill_latw = 1.0 - np.sqrt(acc["model_sse_latw"] / acc["weight_sum"]) / np.sqrt(acc["forecast_sse_latw"] / acc["weight_sum"])
            else:
                global_skill_latw = float("nan")
            for metric_name, value in (
                ("analysis_skill_vs_forecast_global", global_skill),
                ("analysis_skill_vs_forecast_latw_global", global_skill_latw),
            ):
                rows.append(_make_metric_row(
                    experiment_id=f"p2_8b_offline_mix_{candidate_id}",
                    method=f"offline_mix_rho_{rho:g}",
                    sample=first_sample,
                    sample_idx=-1,
                    split_role=str(records[0].get("split_role", "")),
                    split_file=str(records[0].get("split_file", "")),
                    mask_file="",
                    target_context_dates_hash=str(records[0].get("target_context_dates_hash", "")),
                    target_support_dates_hash=str(records[0].get("target_support_dates_hash", "")),
                    support_dates_hash=str(records[0].get("support_dates_hash", "")),
                    target_train_dates_hash=str(records[0].get("target_train_dates_hash", "")),
                    target_eval_dates_hash=str(records[0].get("target_eval_dates_hash", "")),
                    split_manifest_sha256=str(records[0].get("split_manifest_sha256", "")),
                    protocol_freeze_id=str(records[0].get("protocol_freeze_id", "")),
                    n_valid=int(acc["n_pixels"]),
                    mask_frac=float("nan"),
                    variable=variable,
                    metric_name=metric_name,
                    value=float(value),
                ))
                rows[-1]["query_date"] = "global"
                rows[-1]["query_time_index"] = -1
                rows[-1]["season"] = "global"
    return rows


def mix_prediction_record_files(
    zero_shot_records: str | Path,
    adapted_records: str | Path,
    *,
    rho: float,
    candidate_id: str,
    calibration_mode: bool = True,
) -> dict[str, Any]:
    """Mix two aligned prediction-record JSONL files and recompute metrics."""
    if not 0.0 <= float(rho) <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    k0_rows = _read_jsonl(zero_shot_records)
    adapted_rows = _read_jsonl(adapted_records)
    if len(k0_rows) != len(adapted_rows):
        raise ValueError("Prediction record files have different lengths")
    if calibration_mode:
        for record in [*k0_rows, *adapted_rows]:
            _validate_source_safe_record(record)

    mixed_records = []
    for k0, adapted in zip(k0_rows, adapted_rows):
        if _record_identity(k0) != _record_identity(adapted):
            raise ValueError(f"Prediction record identity mismatch: {_record_identity(k0)} vs {_record_identity(adapted)}")
        mixed_records.append(_mixed_record(k0, adapted, rho=float(rho), candidate_id=candidate_id))

    metric_rows = _records_to_metric_rows(mixed_records, candidate_id=candidate_id, rho=float(rho))
    summary = summarize_metric_rows(pd.DataFrame(metric_rows))
    verification = {
        "rho0_equals_k0": False,
        "rho1_equals_adapted": False,
    }
    if float(rho) == 0.0:
        verification["rho0_equals_k0"] = all(
            _array_equal_payload(mixed["arrays"]["pred_increment_surface"], k0["arrays"]["pred_increment_surface"])
            and _array_equal_payload(mixed["arrays"]["pred_increment_rootzone"], k0["arrays"]["pred_increment_rootzone"])
            for mixed, k0 in zip(mixed_records, k0_rows)
        )
    if float(rho) == 1.0:
        verification["rho1_equals_adapted"] = all(
            _array_equal_payload(mixed["arrays"]["pred_increment_surface"], adapted["arrays"]["pred_increment_surface"])
            and _array_equal_payload(mixed["arrays"]["pred_increment_rootzone"], adapted["arrays"]["pred_increment_rootzone"])
            for mixed, adapted in zip(mixed_records, adapted_rows)
        )
    return {
        "candidate_id": candidate_id,
        "adapt_mix_rho": float(rho),
        "records": mixed_records,
        "metric_rows": metric_rows,
        "summary": summary,
        "zero_shot_prediction_content_hash": _prediction_content_hash(k0_rows),
        "adapted_prediction_content_hash": _prediction_content_hash(adapted_rows),
        "mixed_prediction_content_hash": _prediction_content_hash(mixed_records),
        "metric_content_hash": metric_rows_content_hash(metric_rows),
        "metric_values_content_hash": metric_values_content_hash(metric_rows),
        "source_prediction_record_hash": hashlib.sha256(
            (Path(zero_shot_records).read_bytes() + Path(adapted_records).read_bytes())
        ).hexdigest(),
        "verification": verification,
    }


def generate_conflict_rule_logical_rows(
    *,
    zero_shot_records: str | Path,
    adapted_records: str | Path,
    base_candidate: Mapping[str, Any],
    rho_policies: Sequence[str] = ("rule_a", "rule_b", "rule_c"),
) -> list[dict[str, Any]]:
    """Create logical candidate rows for conflict-aware rho policies."""
    rows = []
    diagnostics = {
        "support_gradient_negative_fraction": base_candidate.get("support_gradient_negative_fraction"),
        "support_gradient_cosine_min": base_candidate.get("support_gradient_cosine_min"),
    }
    for policy in rho_policies:
        rho = compute_rho_for_policy(policy, diagnostics)
        candidate = dict(base_candidate)
        candidate["rho_policy"] = policy
        candidate["candidate_id"] = candidate_id_from_config(candidate)
        result = mix_prediction_record_files(
            zero_shot_records,
            adapted_records,
            rho=rho,
            candidate_id=str(candidate["candidate_id"]),
        )
        surface = result["summary"].get("surface", {})
        rootzone = result["summary"].get("rootzone", {})
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "base_candidate_id": base_candidate.get("candidate_id", ""),
                "rho_policy": policy,
                "adapt_mix_rho": rho,
                "logical_offline_mix": True,
                "candidate_config_hash": candidate_config_hash(candidate),
                "source_prediction_record_hash": result["source_prediction_record_hash"],
                "prediction_content_hash": result["mixed_prediction_content_hash"],
                "metric_content_hash": result["metric_content_hash"],
                "surface_skill_primary": surface.get("skill_primary"),
                "rootzone_skill_primary": rootzone.get("skill_primary"),
                "support_gradient_negative_fraction": base_candidate.get("support_gradient_negative_fraction"),
                "support_gradient_cosine_min": base_candidate.get("support_gradient_cosine_min"),
                "schedule_label": base_candidate.get("schedule_label", ""),
                "support_loss_reduction": base_candidate.get("support_loss_reduction", "global_pixel"),
                "trust_policy": base_candidate.get("trust_policy", "none"),
                "K": 12,
            }
        )
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mix source-safe prediction records offline.")
    parser.add_argument("--zero_shot_records", required=True)
    parser.add_argument("--adapted_records", required=True)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--candidate_id", default="offline_mix")
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = mix_prediction_record_files(
        args.zero_shot_records,
        args.adapted_records,
        rho=args.rho,
        candidate_id=args.candidate_id,
    )
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_dir / "mixed_prediction_records.jsonl", result["records"])
        pd.DataFrame(result["metric_rows"]).to_csv(output_dir / "metrics_long.csv", index=False)
        summary = {key: value for key, value in result.items() if key not in {"records", "metric_rows"}}
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps({key: value for key, value in result.items() if key not in {"records", "metric_rows"}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
