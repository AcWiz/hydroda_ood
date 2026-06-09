#!/usr/bin/env python3
"""Rebuild Phase 4 summary.json files and generate merged summary table.

Reads existing evaluation results from:
  - phase4_source_only_region_specific/  (one checkpoint per region)
  - phase4_source_only_all_regions/      (one checkpoint for all regions)
  - phase3_forecast_only_all_regions/    (forecast-only baseline)

For each region, reads metrics_long.csv (or per_region_summary.json for all-regions),
computes latw-aware summary statistics, and writes updated summary.json files.

Generates summary_table.md comparing source-only (region-specific),
source-only (all-regions), and forecast-only methods across all US regions.

Usage:
    PYTHONPATH=. python scripts/eval/rebuild_phase4_summary_table.py
    PYTHONPATH=. python scripts/eval/rebuild_phase4_summary_table.py --output-dir artifacts/results/phase4_summary
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────

REGION_SPECIFIC_BASE = Path("artifacts/runs/phase4_source_only_region_specific")
ALL_REGIONS_BASE = Path("artifacts/runs/phase4_source_only_all_regions")
FORECAST_ONLY_BASE = Path("artifacts/results/phase3_forecast_only_all_regions")

REGIONS = ["US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6"]
SPLIT_TYPES = ["target_eval", "source_test"]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_latest_run(base: Path, region: str | None = None) -> Path | None:
    """Find the latest run directory for a region (or all-regions)."""
    if region:
        pattern = f"phase4_source_only_region_specific_source_only_{region}_*"
    else:
        pattern = "phase4_source_only_all_regions_source_only_*"
    runs = sorted(base.glob(pattern))
    return runs[-1] if runs else None


def _find_best_checkpoint_name(run_dir: Path) -> str:
    """Find the best checkpoint name in a run directory."""
    ckpt_dir = run_dir / "checkpoints"
    for name in ["checkpoint_best_source_val_safe_score.pt", "best.pt"]:
        if (ckpt_dir / name).exists():
            return Path(name).stem
    return "best"


def _fmt(v) -> str:
    """Format a value for markdown table display."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.10f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _summary_wrmse(summary: dict, variable: str) -> float:
    """Extract the preferred latitude-weighted WRMSE from a summary dict."""
    metrics = summary.get(variable, {})
    value = metrics.get(
        "analysis_rmse_latw_mean",
        metrics.get("increment_rmse_latw_mean", metrics.get("rmse_latw_mean", float("nan"))),
    )
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _region_split_wrmse(results: dict, region: str, variable: str, split_type: str = "target_eval") -> float:
    return _summary_wrmse(results.get(region, {}).get(split_type, {}), variable)


def all_region_wrmse_win_gate(
    candidate_results: dict,
    all_regions_baseline: dict,
    region_specific_baseline: dict,
    *,
    regions: list[str] | None = None,
) -> dict:
    """Require every region x variable WRMSE to beat AR and RS baselines.

    The gate is strict: the candidate must have finite target_eval WRMSE lower
    than both the all-regions baseline and the region-specific baseline for
    each requested region and for both Surface and RootZone.
    """
    regions = REGIONS if regions is None else list(regions)
    failures = []
    n_checks = 0
    for region in regions:
        for variable in ["surface", "rootzone"]:
            n_checks += 1
            candidate = _region_split_wrmse(candidate_results, region, variable)
            all_regions = _region_split_wrmse(all_regions_baseline, region, variable)
            region_specific = _region_split_wrmse(region_specific_baseline, region, variable)
            if not (
                math.isfinite(candidate)
                and math.isfinite(all_regions)
                and math.isfinite(region_specific)
                and candidate < all_regions
                and candidate < region_specific
            ):
                failures.append(
                    {
                        "region": region,
                        "variable": variable,
                        "candidate_wrmse": candidate,
                        "all_regions_wrmse": all_regions,
                        "region_specific_wrmse": region_specific,
                    }
                )
    return {
        "pass": len(failures) == 0,
        "n_checks": n_checks,
        "failures": failures,
    }


# ── Summary computation from metrics_long.csv ──────────────────────────────────

# Metrics we track (from metrics_long.csv)
PER_SAMPLE_METRICS = [
    "analysis_skill_vs_forecast",
    "analysis_skill_vs_forecast_latw",
    "increment_rmse",
    "increment_rmse_latw",
    "increment_corr",
    "increment_corr_latw",
    "analysis_rmse",
    "analysis_rmse_latw",
]
GLOBAL_METRICS = [
    "analysis_skill_vs_forecast_global",
    "analysis_skill_vs_forecast_latw_global",
]


def _flatten_nested_metrics(nested: dict) -> dict:
    """Flatten nested {metric: {mean, std, n}} structure to {metric_mean, metric_std, metric_n}."""
    flat = {}
    for key, val in nested.items():
        if isinstance(val, dict) and "mean" in val:
            flat[f"{key}_mean"] = val["mean"]
            if "std" in val:
                flat[f"{key}_std"] = val["std"]
            if "n" in val:
                flat[f"{key}_n"] = val["n"]
        else:
            flat[key] = val
    return flat


def _weighted_aggregate_regions(source_entries: list[dict], source_region_ids: list[str]) -> dict:
    """Weighted aggregation of per_region_summary entries across multiple regions.

    Uses each metric's n as weight for weighted_mean and pooled_std.
    Output: flattened keys {metric}_mean, {metric}_std, {metric}_n.
    """
    result: dict = {"regions": source_region_ids}
    for variable in ["surface", "rootzone"]:
        var_metrics: dict = {}
        all_keys: set = set()
        for entry in source_entries:
            all_keys.update(entry.get(variable, {}).keys())
        for metric_name in all_keys:
            means, stds, ns = [], [], []
            for entry in source_entries:
                m = entry.get(variable, {}).get(metric_name, {})
                if isinstance(m, dict) and "mean" in m and "n" in m:
                    means.append(m["mean"])
                    stds.append(m.get("std", 0.0))
                    ns.append(m["n"])
            if not ns or sum(ns) == 0:
                continue
            total_n = sum(ns)
            weighted_mean = sum(m * n for m, n in zip(means, ns)) / total_n
            pooled_var = sum(
                n * (s**2 + (m - weighted_mean) ** 2) for m, s, n in zip(means, stds, ns)
            ) / total_n
            var_metrics[f"{metric_name}_mean"] = weighted_mean
            var_metrics[f"{metric_name}_std"] = (
                math.sqrt(pooled_var) if pooled_var > 0 else 0.0
            )
            var_metrics[f"{metric_name}_n"] = total_n
        result[variable] = var_metrics
    return result


def _compute_summary_from_df(df: pd.DataFrame, regions: list[str] | None = None) -> dict:
    """Compute summary dict from a metrics_long DataFrame (shared by CSV and cross-region paths)."""
    per_sample = df[df["query_date"] != "global"]
    n_dates = per_sample["query_date"].nunique()
    n_valid_pixels_total = int(
        per_sample.groupby("query_date")["n_valid_pixels"].first().sum()
    )

    # Regions from active_region_ids
    if regions is None:
        regions = sorted(
            set(
                rid
                for aids in per_sample["active_region_ids"].dropna().unique()
                for rid in str(aids).split("|")
            )
        )

    summary = {
        "n_metric_rows": len(df),
        "n_dates": n_dates,
        "n_valid_pixels_total": n_valid_pixels_total,
        "regions": regions,
        "target_region_id": sorted(df["target_region_id"].dropna().unique().tolist()),
    }

    for variable in ["surface", "rootzone"]:
        var_df = df[df["variable"] == variable]
        if var_df.empty:
            continue
        var_summary = {}
        for metric in PER_SAMPLE_METRICS + GLOBAL_METRICS:
            metric_df = var_df[var_df["metric"] == metric]
            if metric_df.empty:
                continue
            if metric.endswith("_global"):
                var_summary[metric] = float(metric_df["value"].iloc[0])
            else:
                var_summary[f"{metric}_mean"] = float(metric_df["value"].mean())
                var_summary[f"{metric}_std"] = float(metric_df["value"].std())
        summary[variable] = var_summary

    return summary


def compute_summary_from_csv(csv_path: Path) -> dict:
    """Compute summary dict from a metrics_long.csv file."""
    df = pd.read_csv(str(csv_path))
    return _compute_summary_from_df(df)


def _rows_for_regions(df: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    """Return per-sample rows plus matching region-labeled global rows."""
    region_set = set(regions)
    sample_region = df["sample_region_id"].astype(str)
    per_sample = df[(df["query_date"] != "global") & sample_region.isin(region_set)]
    labeled_global = df[(df["query_date"] == "global") & sample_region.isin(region_set)]
    if labeled_global.empty:
        unlabeled_global = df[
            (df["query_date"] == "global")
            & (df["sample_region_id"].isna() | (sample_region == ""))
        ]
        labeled_global = unlabeled_global
    return pd.concat([per_sample, labeled_global], ignore_index=True)


def _compute_per_region_summary_from_df(df: pd.DataFrame, regions: list[str]) -> dict[str, dict]:
    """Compute per-region summaries from a combined metrics_long DataFrame.

    This keeps all-region baselines on the same paper-facing metric contract as
    region-specific runs: lat-weighted WRMSE is computed from per-region rows
    and global skill rows are kept as aggregate-then-sqrt values, not
    per-sample skill means from per_region_summary.json.
    """
    results: dict[str, dict] = {}
    for region in regions:
        region_df = _rows_for_regions(df, [region])
        if region_df.empty:
            continue
        results[region] = _compute_summary_from_df(region_df, regions=[region])
    return results


# ── Main table builder ─────────────────────────────────────────────────────────

def _compute_cross_region_src_rs(target_region: str) -> dict:
    """Compute Src_RS for a target region by aggregating other models' target_eval results.

    For target region X, reads each Y model's (Y != X) evaluation on X's target_eval data,
    concatenates all metric rows, and computes a combined weighted summary.

    This ensures Src_RS and Tgt_RS evaluate on the SAME domain (X's test data),
    making the comparison fair.
    """
    dfs = []
    source_regions = []
    for src_region in REGIONS:
        if src_region == target_region:
            continue
        src_run = _find_latest_run(REGION_SPECIFIC_BASE, src_region)
        if not src_run:
            continue
        ckpt_name = _find_best_checkpoint_name(src_run)
        csv_path = src_run / "results" / ckpt_name / "target_eval" / target_region / "metrics_long.csv"
        if csv_path.exists():
            dfs.append(pd.read_csv(str(csv_path)))
            source_regions.append(src_region)

    if not dfs:
        print(f"  WARNING: No cross-region results found for {target_region}")
        return {}

    combined = pd.concat(dfs, ignore_index=True)
    summary = _compute_summary_from_df(combined, regions=source_regions)
    summary["source_region_ids"] = source_regions
    return summary


def _compute_in_domain_src_rs(target_region: str) -> dict:
    """Compute Src_RS_ID for a target region by aggregating other models' in-domain results.

    For target region X, reads each Y model's (Y != X) evaluation on Y's OWN target_eval data.
    Each model only does inference on the region it was trained on — no cross-region inference.
    All 5 in-domain results are concatenated and summarized.

    This ensures Src_RS_ID represents "how well models perform on their own domains"
    rather than cross-region generalization.
    """
    dfs = []
    source_regions = []
    for src_region in REGIONS:
        if src_region == target_region:
            continue
        src_run = _find_latest_run(REGION_SPECIFIC_BASE, src_region)
        if not src_run:
            continue
        ckpt_name = _find_best_checkpoint_name(src_run)
        # KEY DIFFERENCE: reads src_region's OWN data, not target_region's
        csv_path = src_run / "results" / ckpt_name / "target_eval" / src_region / "metrics_long.csv"
        if csv_path.exists():
            dfs.append(pd.read_csv(str(csv_path)))
            source_regions.append(src_region)

    if not dfs:
        print(f"  WARNING: No in-domain source results found for {target_region}")
        return {}

    combined = pd.concat(dfs, ignore_index=True)
    summary = _compute_summary_from_df(combined, regions=source_regions)
    summary["source_region_ids"] = source_regions
    return summary


def collect_region_specific_results() -> dict[str, dict]:
    """Collect region-specific model results. Returns {region: {split_type: summary}}.

    For each region X:
      - target_eval: X's own model evaluated on X's test data (Tgt_RS).
      - source_test: cross-region aggregate — other models Y (Y != X) evaluated on X's
        target_eval data, concatenated and summarized (Src_RS).
      - source_in_domain: in-domain aggregate — other models Y (Y != X) each evaluated
        on Y's OWN target_eval data, concatenated (Src_RS_ID).
    """
    results = {}
    for region in REGIONS:
        run_dir = _find_latest_run(REGION_SPECIFIC_BASE, region)
        if not run_dir:
            print(f"  WARNING: No run found for {region}")
            continue

        ckpt_name = _find_best_checkpoint_name(run_dir)
        results_dir = run_dir / "results" / ckpt_name

        region_results = {}

        # Tgt_RS: own model on own target_eval data
        for split_type in ["target_eval"]:
            summary_path = results_dir / split_type / region / "summary.json"
            csv_path = results_dir / split_type / region / "metrics_long.csv"

            if summary_path.exists():
                with open(summary_path) as f:
                    region_results[split_type] = json.load(f)
            elif csv_path.exists():
                region_results[split_type] = compute_summary_from_csv(csv_path)
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                with open(summary_path, "w") as f:
                    json.dump(region_results[split_type], f, indent=2)
            else:
                # Try legacy target_query/ fallback
                legacy_dir = run_dir / "results" / "target_query"
                legacy_csv = legacy_dir / "metrics_long.csv" if (legacy_dir / "metrics_long.csv").exists() else None
                if legacy_csv is not None:
                    region_results[split_type] = compute_summary_from_csv(legacy_csv)
                else:
                    legacy_ps = legacy_dir / "per_region_summary.json"
                    if legacy_ps.exists():
                        with open(legacy_ps) as f:
                            ps = json.load(f)
                        if region in ps:
                            region_results[split_type] = {
                                "surface": ps[region].get("surface", {}),
                                "rootzone": ps[region].get("rootzone", {}),
                                "regions": [region],
                                "n_dates": 0,
                            }

        # Src_RS: other models evaluated on this region's target_eval data
        src_rs_summary = _compute_cross_region_src_rs(region)
        if src_rs_summary:
            region_results["source_test"] = src_rs_summary

        # Src_RS_ID: other models each evaluated on their OWN in-domain data
        src_id_summary = _compute_in_domain_src_rs(region)
        if src_id_summary:
            region_results["source_in_domain"] = src_id_summary

        if region_results:
            results[region] = region_results

    return results


def collect_all_regions_results() -> dict[str, dict]:
    """Collect all-regions model results. {region: {split_type: summary}}.

    Prefer metrics_long.csv so paper-facing skill uses the aggregate global
    rows produced by the evaluation harness. per_region_summary.json remains a
    legacy fallback, but it only contains per-sample metric means.
    """
    run_dir = _find_latest_run(ALL_REGIONS_BASE)
    if not run_dir:
        return {}

    ckpt_name = _find_best_checkpoint_name(run_dir)
    results_dir = run_dir / "results" / ckpt_name

    csv_path = results_dir / "target_eval" / "metrics_long.csv"
    if not csv_path.exists():
        csv_path = results_dir / "source_test" / "metrics_long.csv"
    if not csv_path.exists():
        for legacy_split in ["target_query", "source_val"]:
            lp = results_dir / legacy_split / "metrics_long.csv"
            if lp.exists():
                csv_path = lp
                break
    if csv_path.exists():
        df = pd.read_csv(str(csv_path))
        target_by_region = _compute_per_region_summary_from_df(df, REGIONS)
        results: dict[str, dict] = {}
        for region in REGIONS:
            if region not in target_by_region:
                continue
            results.setdefault(region, {})["target_eval"] = target_by_region[region]

            src_ids = [r for r in REGIONS if r != region]
            source_df = _rows_for_regions(df, src_ids)
            if not source_df.empty:
                results[region]["source_test"] = _compute_summary_from_df(source_df, regions=src_ids)
        return results

    # Read per_region_summary.json (target_eval and source_test are identical for AR)
    ps_path = results_dir / "target_eval" / "per_region_summary.json"
    if not ps_path.exists():
        ps_path = results_dir / "source_test" / "per_region_summary.json"
    if not ps_path.exists():
        # Legacy fallback
        for legacy_split in ["target_query", "source_val"]:
            lp = results_dir / legacy_split / "per_region_summary.json"
            if lp.exists():
                ps_path = lp
                break
    if not ps_path.exists():
        return {}

    with open(ps_path) as f:
        all_data = json.load(f)

    results: dict[str, dict] = {}
    for region in REGIONS:
        if region not in all_data:
            continue
        raw = all_data[region]
        # target_eval: the region's own metrics
        results.setdefault(region, {})["target_eval"] = {
            "surface": _flatten_nested_metrics(raw.get("surface", {})),
            "rootzone": _flatten_nested_metrics(raw.get("rootzone", {})),
            "regions": [region],
        }
        # source_test: weighted aggregate of the other 5 regions
        src_ids = [r for r in REGIONS if r != region]
        src_entries = [all_data[r] for r in src_ids if r in all_data]
        results[region]["source_test"] = _weighted_aggregate_regions(src_entries, src_ids)

    return results


def collect_forecast_only_results() -> dict[str, dict]:
    """Collect forecast-only baseline results."""
    results = {}
    for split_type in SPLIT_TYPES:
        for region in REGIONS:
            csv_path = FORECAST_ONLY_BASE / split_type / region / "metrics_long.csv"
            if not csv_path.exists():
                continue
            results.setdefault(region, {})[split_type] = compute_summary_from_csv(csv_path)
    return results


def build_summary_table(
    rs_results: dict,
    ar_results: dict,
    fo_results: dict,
    output_dir: Path,
) -> str:
    """Build the merged markdown summary table."""
    output_dir.mkdir(parents=True, exist_ok=True)

    header_cols = [
        "Region",
        "Surf_WRMSE(Tgt_RS)", "Surf_WRMSE(Src_RS_ID)",
        "Surf_WRMSE(Tgt_AR)", "Surf_WRMSE(Src_AR)",
        "Surf_WRMSE(Tgt_FO)", "Surf_WRMSE(Src_FO)",
        "RZ_WRMSE(Tgt_RS)", "RZ_WRMSE(Src_RS_ID)",
        "RZ_WRMSE(Tgt_AR)", "RZ_WRMSE(Src_AR)",
        "RZ_WRMSE(Tgt_FO)", "RZ_WRMSE(Src_FO)",
    ]

    lines = [
        "# Phase 4 Source-Only — All Regions Summary (latw WRMSE)",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Methods:",
        "- **RS**: Region-Specific source-only (one backbone per region)",
        "- **AR**: All-Regions source-only (shared backbone)",
        "- **FO**: Forecast-Only baseline",
        "",
        "| " + " | ".join(header_cols) + " |",
        "|" + "|".join(["--------"] * len(header_cols)) + "|",
    ]

    for region in REGIONS:
        rs = rs_results.get(region, {})
        ar = ar_results.get(region, {})
        fo = fo_results.get(region, {})

        row = [region]

        # Surface WRMSE: for each method × split_type
        for method_results, method_name in [(rs, "RS"), (ar, "AR"), (fo, "FO")]:
            splits = ["target_eval", "source_in_domain"] if method_name == "RS" else ["target_eval", "source_test"]
            for split_type in splits:
                s = method_results.get(split_type, {})
                surf = s.get("surface", {})
                val = surf.get("analysis_rmse_latw_mean",
                      surf.get("increment_rmse_latw_mean",
                      surf.get("rmse_latw_mean", float("nan"))))
                row.append(_fmt(val))

        # Rootzone WRMSE
        for method_results, method_name in [(rs, "RS"), (ar, "AR"), (fo, "FO")]:
            splits = ["target_eval", "source_in_domain"] if method_name == "RS" else ["target_eval", "source_test"]
            for split_type in splits:
                s = method_results.get(split_type, {})
                rz = s.get("rootzone", {})
                val = rz.get("analysis_rmse_latw_mean",
                      rz.get("increment_rmse_latw_mean",
                      rz.get("rmse_latw_mean", float("nan"))))
                row.append(_fmt(val))

        lines.append("| " + " | ".join(row) + " |")

    # ── Skill table ────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("## Analysis Skill (latw) vs Forecast")
    lines.append("")

    skill_cols = [
        "Region",
        "Surf_Skill(Tgt_RS)", "Surf_Skill(Src_RS_ID)",
        "Surf_Skill(Tgt_AR)", "Surf_Skill(Src_AR)",
        "Surf_Skill(Tgt_FO)", "Surf_Skill(Src_FO)",
        "RZ_Skill(Tgt_RS)", "RZ_Skill(Src_RS_ID)",
        "RZ_Skill(Tgt_AR)", "RZ_Skill(Src_AR)",
        "RZ_Skill(Tgt_FO)", "RZ_Skill(Src_FO)",
    ]

    lines.append("| " + " | ".join(skill_cols) + " |")
    lines.append("|" + "|".join(["--------"] * len(skill_cols)) + "|")

    for region in REGIONS:
        rs = rs_results.get(region, {})
        ar = ar_results.get(region, {})
        fo = fo_results.get(region, {})

        row = [region]

        for method_results, method_name in [(rs, "RS"), (ar, "AR"), (fo, "FO")]:
            splits = ["target_eval", "source_in_domain"] if method_name == "RS" else ["target_eval", "source_test"]
            for split_type in splits:
                s = method_results.get(split_type, {})
                surf = s.get("surface", {})
                # Skill: prefer latw_global (aggregate-then-sqrt), fall back to per-sample latw mean
                val = surf.get("analysis_skill_vs_forecast_latw_global",
                      surf.get("skill_latw_primary",
                      surf.get("analysis_skill_vs_forecast_latw_mean",
                      surf.get("skill_latw_mean", float("nan")))))
                row.append(_fmt(val))

        for method_results, method_name in [(rs, "RS"), (ar, "AR"), (fo, "FO")]:
            splits = ["target_eval", "source_in_domain"] if method_name == "RS" else ["target_eval", "source_test"]
            for split_type in splits:
                s = method_results.get(split_type, {})
                rz = s.get("rootzone", {})
                val = rz.get("analysis_skill_vs_forecast_latw_global",
                      rz.get("skill_latw_primary",
                      rz.get("analysis_skill_vs_forecast_latw_mean",
                      rz.get("skill_latw_mean", float("nan")))))
                row.append(_fmt(val))

        lines.append("| " + " | ".join(row) + " |")

    # ── Corr_latw table ────────────────────────────────────────────────────────
    lines.append("")
    lines.append("## Increment Correlation (latw)")
    lines.append("")

    corr_cols = [
        "Region",
        "Surf_Corr(Tgt_RS)", "Surf_Corr(Src_RS_ID)",
        "Surf_Corr(Tgt_AR)", "Surf_Corr(Src_AR)",
        "Surf_Corr(Tgt_FO)", "Surf_Corr(Src_FO)",
        "RZ_Corr(Tgt_RS)", "RZ_Corr(Src_RS_ID)",
        "RZ_Corr(Tgt_AR)", "RZ_Corr(Src_AR)",
        "RZ_Corr(Tgt_FO)", "RZ_Corr(Src_FO)",
    ]

    lines.append("| " + " | ".join(corr_cols) + " |")
    lines.append("|" + "|".join(["--------"] * len(corr_cols)) + "|")

    for region in REGIONS:
        rs = rs_results.get(region, {})
        ar = ar_results.get(region, {})
        fo = fo_results.get(region, {})

        row = [region]

        for method_results, method_name in [(rs, "RS"), (ar, "AR"), (fo, "FO")]:
            splits = ["target_eval", "source_in_domain"] if method_name == "RS" else ["target_eval", "source_test"]
            for split_type in splits:
                s = method_results.get(split_type, {})
                surf = s.get("surface", {})
                val = surf.get("increment_corr_latw_mean",
                      surf.get("corr_latw_mean", float("nan")))
                row.append(_fmt(val))

        for method_results, method_name in [(rs, "RS"), (ar, "AR"), (fo, "FO")]:
            splits = ["target_eval", "source_in_domain"] if method_name == "RS" else ["target_eval", "source_test"]
            for split_type in splits:
                s = method_results.get(split_type, {})
                rz = s.get("rootzone", {})
                val = rz.get("increment_corr_latw_mean",
                      rz.get("corr_latw_mean", float("nan")))
                row.append(_fmt(val))

        lines.append("| " + " | ".join(row) + " |")

    md_path = output_dir / "summary_table.md"
    md_path.write_text("\n".join(lines) + "\n")

    return "\n".join(lines)


def build_combined_summary_payload(
    *,
    rs_results: dict,
    ar_results: dict,
    fo_results: dict,
) -> dict:
    """Build machine-readable Phase 4 summary with paper-facing baseline names."""
    return {
        "generated": datetime.now().isoformat(),
        "regions": REGIONS,
        "baselines": {
            "region_specific": {
                "paper_name": "RS-Scratch",
                "description": "One backbone per region trained from scratch on that region's 2015-2021 labels.",
            },
            "all_regions": {
                "paper_name": "Pooled Global",
                "description": "One shared backbone trained on all US regions' 2015-2021 labels.",
            },
            "forecast_only": {
                "paper_name": "Forecast-Only",
                "description": "No learned increment; predicted analysis equals forecast.",
            },
        },
        "region_specific": {
            r: {
                st: {
                    "surface": rs_results.get(r, {}).get(st, {}).get("surface", {}),
                    "rootzone": rs_results.get(r, {}).get(st, {}).get("rootzone", {}),
                }
                for st in SPLIT_TYPES + ["source_in_domain"]
            }
            for r in REGIONS
        },
        "all_regions": {
            r: {
                st: {
                    "surface": ar_results.get(r, {}).get(st, {}).get("surface", {}),
                    "rootzone": ar_results.get(r, {}).get(st, {}).get("rootzone", {}),
                }
                for st in SPLIT_TYPES
            }
            for r in REGIONS
        },
        "forecast_only": {
            r: {
                st: {
                    "surface": fo_results.get(r, {}).get(st, {}).get("surface", {}),
                    "rootzone": fo_results.get(r, {}).get(st, {}).get("rootzone", {}),
                }
                for st in SPLIT_TYPES
            }
            for r in REGIONS
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild Phase 4 summary.json files and merged summary table"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="artifacts/results/phase4_summary",
        help="Output directory for summary table"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Phase 4 Summary Table Rebuild")
    print("=" * 60)

    # Collect results from all sources
    print("\nCollecting region-specific results...")
    rs_results = collect_region_specific_results()
    print(f"  Found {len(rs_results)} regions: {sorted(rs_results.keys())}")

    print("\nCollecting all-regions results...")
    ar_results = collect_all_regions_results()
    print(f"  Found {len(ar_results)} regions: {sorted(ar_results.keys())}")

    print("\nCollecting forecast-only baseline...")
    fo_results = collect_forecast_only_results()
    print(f"  Found {len(fo_results)} regions: {sorted(fo_results.keys())}")

    # Build and save summary table
    print("\nBuilding summary table...")
    table_md = build_summary_table(rs_results, ar_results, fo_results, output_dir)

    print()
    print(table_md)
    print(f"\nSummary table saved to {output_dir / 'summary_table.md'}")

    # Save a JSON version too
    combined = build_combined_summary_payload(
        rs_results=rs_results,
        ar_results=ar_results,
        fo_results=fo_results,
    )
    json_path = output_dir / "combined_summary.json"
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Combined JSON saved to {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
