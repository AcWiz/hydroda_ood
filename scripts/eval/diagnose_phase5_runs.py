#!/usr/bin/env python3
"""Compare Phase 5 target-adaptation runs.

The script reads evaluation ``metrics_long.csv`` files produced by
``scripts/eval/evaluate_checkpoint.py``. It does not train, select, or tune any
model and is intended only for post-hoc diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import torch


DEFAULT_METRICS = [
    "increment_rmse_latw",
    "increment_corr_latw",
    "analysis_skill_vs_forecast_latw",
]


def _finite_mean(values: pd.Series) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return float("nan")
    return float(series.mean())


def _metric_summary(base: pd.Series, cand: pd.Series, lower_is_better: bool) -> Dict[str, float]:
    baseline_mean = _finite_mean(base)
    candidate_mean = _finite_mean(cand)
    delta = candidate_mean - baseline_mean
    if lower_is_better:
        relative = (baseline_mean - candidate_mean) / baseline_mean if baseline_mean != 0 else float("nan")
    else:
        relative = (candidate_mean - baseline_mean) / abs(baseline_mean) if baseline_mean != 0 else float("nan")
    return {
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "delta": float(delta),
        "relative_improvement": float(relative),
        "n": int(min(base.dropna().shape[0], cand.dropna().shape[0])),
    }


def _metric_lower_is_better(metric: str) -> bool:
    metric_l = metric.lower()
    return "rmse" in metric_l or "mae" in metric_l or "mse" in metric_l


def _pivot_metrics(df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    required = {"query_date", "variable", "metric", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metrics frame missing required columns: {sorted(missing)}")
    filtered = df[df["metric"].isin(list(metrics))].copy()
    filtered = filtered[filtered["query_date"].astype(str) != "global"]
    index_cols = ["query_date", "variable", "metric"]
    optional_cols = [col for col in ["month", "season", "target_region_id"] if col in filtered.columns]
    grouped = filtered.groupby(index_cols + optional_cols, dropna=False)["value"].mean().reset_index()
    return grouped


def _aligned_metric_values(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    variable: str,
    metric: str,
) -> pd.DataFrame:
    base = baseline[(baseline["variable"] == variable) & (baseline["metric"] == metric)]
    cand = candidate[(candidate["variable"] == variable) & (candidate["metric"] == metric)]
    join_cols = [col for col in ["query_date", "variable", "metric"] if col in base.columns and col in cand.columns]
    context_cols = [col for col in ["month", "season", "target_region_id"] if col in base.columns]
    merged = base[join_cols + context_cols + ["value"]].merge(
        cand[join_cols + ["value"]],
        on=join_cols,
        how="inner",
        suffixes=("_baseline", "_candidate"),
    )
    return merged


def _summaries_by_group(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    group_col: str,
    metrics: Iterable[str],
) -> Dict[str, Any]:
    if group_col not in baseline.columns:
        return {}
    output: Dict[str, Any] = {}
    variables = sorted(set(baseline["variable"].dropna()) | set(candidate["variable"].dropna()))
    for group_value in sorted(baseline[group_col].dropna().unique().tolist(), key=lambda x: str(x)):
        group_base = baseline[baseline[group_col] == group_value]
        group_cand = candidate[candidate[group_col] == group_value]
        group_key = str(int(group_value)) if group_col == "month" else str(group_value)
        output[group_key] = {}
        for variable in variables:
            output[group_key][variable] = {}
            for metric in metrics:
                base_values = group_base[(group_base["variable"] == variable) & (group_base["metric"] == metric)]["value"]
                cand_values = group_cand[(group_cand["variable"] == variable) & (group_cand["metric"] == metric)]["value"]
                if base_values.empty and cand_values.empty:
                    continue
                output[group_key][variable][metric] = _metric_summary(
                    base_values,
                    cand_values,
                    lower_is_better=_metric_lower_is_better(metric),
                )
    return output


def compare_metric_frames(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    metrics: Iterable[str] = DEFAULT_METRICS,
) -> Dict[str, Any]:
    baseline = _pivot_metrics(baseline_df, metrics)
    candidate = _pivot_metrics(candidate_df, metrics)
    variables = sorted(set(baseline["variable"].dropna()) | set(candidate["variable"].dropna()))
    metric_list = list(metrics)

    overall: Dict[str, Any] = {}
    win_rate: Dict[str, Any] = {}
    for variable in variables:
        overall[variable] = {}
        win_rate[variable] = {}
        for metric in metric_list:
            base_values = baseline[(baseline["variable"] == variable) & (baseline["metric"] == metric)]["value"]
            cand_values = candidate[(candidate["variable"] == variable) & (candidate["metric"] == metric)]["value"]
            if base_values.empty and cand_values.empty:
                continue
            lower_is_better = _metric_lower_is_better(metric)
            overall[variable][metric] = _metric_summary(base_values, cand_values, lower_is_better)
            aligned = _aligned_metric_values(baseline, candidate, variable, metric)
            if aligned.empty:
                continue
            if lower_is_better:
                wins = aligned["value_candidate"] < aligned["value_baseline"]
            else:
                wins = aligned["value_candidate"] > aligned["value_baseline"]
            win_rate[variable][metric] = {
                "candidate_win_rate": float(wins.mean()),
                "n": int(len(wins)),
            }

    djf_base = baseline[(baseline.get("season") == "DJF") & (baseline["variable"] == "surface")]
    djf_cand = candidate[(candidate.get("season") == "DJF") & (candidate["variable"] == "surface")]
    djf_metric = "increment_rmse_latw"
    djf_summary = _metric_summary(
        djf_base[djf_base["metric"] == djf_metric]["value"],
        djf_cand[djf_cand["metric"] == djf_metric]["value"],
        lower_is_better=True,
    )
    djf_summary["regressed"] = bool(djf_summary["candidate_mean"] > djf_summary["baseline_mean"])

    return {
        "overall": overall,
        "by_season": _summaries_by_group(baseline, candidate, "season", metric_list),
        "by_month": _summaries_by_group(baseline, candidate, "month", metric_list),
        "win_rate": win_rate,
        "djf_surface_regression": djf_summary,
    }


def _parameter_group(name: str) -> str:
    if name.startswith("target_prompt"):
        return "target_prompt"
    if "target_adapter_coefficient_residual" in name:
        return "adapter_coefficient_residuals"
    if name.startswith("residual_gain"):
        return "residual_gain"
    if name.startswith("target_spatial_refine"):
        return "target_spatial_refine"
    return "source_prior_or_other"


def checkpoint_parameter_norms(checkpoint_path: str | Path) -> Dict[str, Dict[str, float]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    groups: Dict[str, Dict[str, float]] = {}
    for name, value in state.items():
        if not torch.is_tensor(value) or not torch.is_floating_point(value):
            continue
        group = _parameter_group(str(name))
        stats = groups.setdefault(group, {"squared_l2": 0.0, "parameter_count": 0})
        tensor = value.detach().float()
        stats["squared_l2"] += float(tensor.square().sum())
        stats["parameter_count"] += int(tensor.numel())
    for stats in groups.values():
        stats["l2_norm"] = float(stats["squared_l2"] ** 0.5)
        del stats["squared_l2"]
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Phase 5 metrics_long.csv outputs")
    parser.add_argument("--baseline_metrics", type=str, required=True)
    parser.add_argument("--candidate_metrics", type=str, required=True)
    parser.add_argument("--baseline_checkpoint", type=str, default=None)
    parser.add_argument("--candidate_checkpoint", type=str, default=None)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline_metrics)
    candidate = pd.read_csv(args.candidate_metrics)
    report = compare_metric_frames(baseline, candidate, metrics=args.metrics)
    if args.baseline_checkpoint:
        report["baseline_parameter_norms"] = checkpoint_parameter_norms(args.baseline_checkpoint)
    if args.candidate_checkpoint:
        report["candidate_parameter_norms"] = checkpoint_parameter_norms(args.candidate_checkpoint)

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote Phase 5 diagnostics to {out}")


if __name__ == "__main__":
    main()
