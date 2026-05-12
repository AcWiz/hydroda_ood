#!/usr/bin/env python3
"""
Phase 3B — Full 10-Seed Summary/Report Layer
============================================
Reads raw long-format metrics_long.csv files from Phase 3B full evaluations
(7 methods × 10 seeds) and produces aggregated tables, figures, and a report.

Input CSVs (from artifacts/metrics/):
  - phase3A_forecast_only_US/metrics_long.csv              [K=0,4,12,24 × 6 regions × 10 seeds]
  - phase3B_source_mean_increment_US/metrics_long.csv     [K=0,4,12,24 × 6 regions × 10 seeds]
  - phase3B_target_support_mean_increment_US/metrics_long.csv  [K=4,12,24 × 6 regions × 10 seeds]
  - phase3B_target_monthly_support_increment_US/metrics_long.csv  [K=12,24 × 6 regions × 10 seeds]
  - phase3B_ridge_core_US/metrics_long.csv                [K=4,12,24 × 6 regions × 10 seeds]
  - phase3B_ridge_input_full_US/metrics_long.csv          [K=4,12,24 × 6 regions × 10 seeds]
  - phase3B_ridge_input_geo_full_US/metrics_long.csv      [K=4,12,24 × 6 regions × 10 seeds]

Output (canonical):
  artifacts/experiments/phase3_simple_baselines/US/summaries/phase3B/
      tables/
          metrics_eval_wide.csv
          table_main_by_method_K.csv
          table_region_by_method_K.csv
          adaptation_gain_by_K.csv
          negative_adaptation_audit.csv
          row_count_audit.csv
      figures/
          fig_main_skill_by_method_K_surface.png
          fig_main_skill_by_method_K_rootzone.png
          fig_adaptation_curve_surface.png
          fig_adaptation_curve_rootzone.png
          fig_region_method_heatmap_surface.png
          fig_region_method_heatmap_rootzone.png
      reports/
          phase3B_report_US.md
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import argparse
from pathlib import Path

# Configure Chinese font for matplotlib
try:
    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ]
    chinese_font = None
    for fp in font_paths:
        if os.path.exists(fp):
            chinese_font = fm.FontProperties(fname=fp)
            break
    if chinese_font is None:
        for fp in fm.findSystemFonts():
            if "Noto" in fp and ("CJK" in fp or "Serif" in fp):
                chinese_font = fm.FontProperties(fname=fp)
                break
except Exception:
    chinese_font = None

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path("/sharefiles1/fenglonghan/projects/hydroda_ood")
ARTIFACTS = BASE / "artifacts/metrics"

# Canonical output root
CANONICAL_ROOT = BASE / "artifacts/experiments/phase3_simple_baselines/US/summaries/phase3B"
OUT_TABLES = CANONICAL_ROOT / "tables"
OUT_FIGURES = CANONICAL_ROOT / "figures"
OUT_REPORT_DIR = CANONICAL_ROOT / "reports"

OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)
OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Input mapping: display_name -> path
# forecast_only comes from phase3A; all others from phase3B
INPUTS = {
    "forecast_only":               ARTIFACTS / "phase3A_forecast_only_US/metrics_long.csv",
    "source_mean_increment":      ARTIFACTS / "phase3B_source_mean_increment_US/metrics_long.csv",
    "target_support_mean_increment":  ARTIFACTS / "phase3B_target_support_mean_increment_US/metrics_long.csv",
    "target_monthly_support_increment": ARTIFACTS / "phase3B_target_monthly_support_increment_US/metrics_long.csv",
    "ridge_core":                 ARTIFACTS / "phase3B_ridge_core_US/metrics_long.csv",
    "ridge_input_full":           ARTIFACTS / "phase3B_ridge_input_full_US/metrics_long.csv",
    "ridge_input_geo_full":       ARTIFACTS / "phase3B_ridge_input_geo_full_US/metrics_long.csv",
}

# Method display name map (Chinese + English)
METHOD_DISPLAY = {
    "forecast_only":                    "预报",
    "source_mean_increment":            "源均值",
    "target_support_mean_increment":    "目标均值",
    "target_monthly_support_increment": "月度",
    "ridge_core":                       "岭回归-core",
    "ridge_input_full":                 "岭回归-full",
    "ridge_input_geo_full":             "岭回归-geo",
}

METRICS = [
    "analysis_rmse", "analysis_mae", "analysis_skill_vs_forecast",
    "increment_rmse", "increment_mae", "increment_bias",
    "increment_corr", "sign_accuracy_deadzone",
]
VARIABLES = ["surface", "rootzone"]
ADAPTIVE_METHODS = [
    "source_mean_increment",
    "target_support_mean_increment",
    "target_monthly_support_increment",
    "ridge_core",
    "ridge_input_full",
    "ridge_input_geo_full",
]


# ── Step 1: Load & concat ──────────────────────────────────────────────────────
def load_all():
    dfs = []
    for name, path in INPUTS.items():
        if not path.exists():
            print(f"[WARN] Missing: {path}")
            continue
        df = pd.read_csv(path)
        df["method"] = name
        dfs.append(df)
        print(f"[LOAD] {name}: {len(df):,} rows, K={sorted(df['K'].unique())}, "
              f"regions={sorted(df['target_region_id'].unique())}, seeds={sorted(df['seed'].unique())}")
    combined = pd.concat(dfs, ignore_index=True)
    return combined


# ── Step 2: Validate n_time_steps ─────────────────────────────────────────────
def validate_time_steps(df):
    bad = df[df["n_time_steps"] != 1]
    if len(bad) > 0:
        print(f"[WARN] {len(bad)} rows with n_time_steps != 1")
    else:
        print("[OK] All rows have n_time_steps == 1 (each row = 1 time slice)")
    print(f"  Unique n_valid_pixels values (sample): {sorted(df['n_valid_pixels'].unique())[:10]}")


# ── Step 3: Build metrics_eval_wide.csv ──────────────────────────────────────
def build_metrics_eval_wide(df):
    """Wide-format table: one row per (method, country, region, K, seed, variable, metric)
    with mean/std/count, plus NaN/inf flags."""
    group_keys = ["method", "country_id", "target_region_id", "K", "seed", "variable", "metric"]

    df = df.copy()
    df["value_is_nan"] = df["value"].isna().astype(int)
    df["value_is_inf"] = np.isinf(df["value"]).astype(int)

    agg = df.groupby(group_keys, dropna=False).agg(
        value_mean=("value", "mean"),
        value_std=("value", "std"),
        value_count=("value", "count"),
        n_valid_pixels_mean=("n_valid_pixels", "mean"),
        mask_fraction_mean=("mask_fraction", "mean"),
        n_nan=("value_is_nan", "sum"),
        n_inf=("value_is_inf", "sum"),
    ).reset_index()

    agg["value_std"] = agg["value_std"].fillna(0.0)
    out = OUT_TABLES / "metrics_eval_wide.csv"
    agg.to_csv(out, index=False)
    print(f"[WROTE] {out}  ({len(agg):,} rows)")
    return agg


# ── Step 4: table_main_by_method_K.csv ───────────────────────────────────────
def build_table_main_by_method_K(wide):
    """Region+seed-balanced: first mean over regions per (K, seed), then aggregate across seeds.
    Produces one row per (method, K, variable)."""
    results = []

    for method in wide["method"].unique():
        for K in sorted(wide[wide["method"] == method]["K"].unique()):
            for variable in VARIABLES:
                subset = wide[
                    (wide["method"] == method) &
                    (wide["K"] == K) &
                    (wide["variable"] == variable)
                ]
                if len(subset) == 0:
                    continue

                row = {"method": method, "K": int(K), "variable": variable}

                for metric in METRICS:
                    ms = subset[subset["metric"] == metric]["value_mean"]
                    if len(ms) == 0:
                        row[f"{metric}_mean"] = np.nan
                        row[f"{metric}_std"] = np.nan
                        continue
                    row[f"{metric}_mean"] = ms.mean()
                    row[f"{metric}_std"] = ms.std() if len(ms) > 1 else 0.0

                # negative adaptation rate: fraction of region×seed cells where skill < 0
                skill_vals = subset[subset["metric"] == "analysis_skill_vs_forecast"]["value_mean"]
                if len(skill_vals) > 0:
                    row["negative_adaptation_rate"] = (skill_vals < 0).sum() / len(skill_vals)
                else:
                    row["negative_adaptation_rate"] = np.nan

                results.append(row)

    result_df = pd.DataFrame(results)
    out = OUT_TABLES / "table_main_by_method_K.csv"
    result_df.to_csv(out, index=False)
    print(f"[WROTE] {out}  ({len(result_df):,} rows)")
    return result_df


# ── Step 5: table_region_by_method_K.csv ─────────────────────────────────────
def build_table_region_by_method_K(wide):
    """Granularity: method × K × target_region_id × variable.
    Aggregates over seeds only."""
    results = []

    for method in wide["method"].unique():
        for K in sorted(wide[wide["method"] == method]["K"].unique()):
            for region in sorted(wide["target_region_id"].unique()):
                for variable in VARIABLES:
                    subset = wide[
                        (wide["method"] == method) &
                        (wide["K"] == K) &
                        (wide["target_region_id"] == region) &
                        (wide["variable"] == variable)
                    ]
                    if len(subset) == 0:
                        continue

                    row = {
                        "method": method, "K": int(K),
                        "target_region_id": region, "variable": variable,
                    }
                    for metric in METRICS:
                        ms = subset[subset["metric"] == metric]["value_mean"]
                        row[f"{metric}_mean"] = ms.mean() if len(ms) > 0 else np.nan
                        row[f"{metric}_std"] = ms.std() if len(ms) > 1 else 0.0
                    results.append(row)

    result_df = pd.DataFrame(results)
    out = OUT_TABLES / "table_region_by_method_K.csv"
    result_df.to_csv(out, index=False)
    print(f"[WROTE] {out}  ({len(result_df):,} rows)")
    return result_df


# ── Step 6: adaptation_gain_by_K.csv ─────────────────────────────────────────
def build_adaptation_gain(wide):
    """Compute adaptation gain vs forecast_only K=0 baseline.
    gain = method_skill(K, var) - forecast_only_skill(K=0, var)
    rmse_reduction = forecast_only_rmse(K=0, var) - method_rmse(K, var)
    Only methods with K>0 support should be included (source_mean_increment
    is included at K>0 as it uses source_train at K>0 in the full run)."""
    results = []

    for method in ADAPTIVE_METHODS:
        for K in sorted(wide[wide["method"] == method]["K"].unique()):
            for variable in VARIABLES:
                # baseline: forecast_only K=0 for this variable
                baseline_mask = (
                    (wide["method"] == "forecast_only") &
                    (wide["K"] == 0) &
                    (wide["variable"] == variable)
                )
                baseline_skill_vals = wide.loc[
                    baseline_mask & (wide["metric"] == "analysis_skill_vs_forecast"), "value_mean"
                ]
                baseline_rmse_vals = wide.loc[
                    baseline_mask & (wide["metric"] == "increment_rmse"), "value_mean"
                ]
                baseline_skill = baseline_skill_vals.mean() if len(baseline_skill_vals) > 0 else np.nan
                baseline_rmse = baseline_rmse_vals.mean() if len(baseline_rmse_vals) > 0 else np.nan

                # adapted: method, K, variable
                adapted_mask = (
                    (wide["method"] == method) &
                    (wide["K"] == K) &
                    (wide["variable"] == variable)
                )
                adapted_skill_vals = wide.loc[
                    adapted_mask & (wide["metric"] == "analysis_skill_vs_forecast"), "value_mean"
                ]
                adapted_rmse_vals = wide.loc[
                    adapted_mask & (wide["metric"] == "increment_rmse"), "value_mean"
                ]
                adapted_skill = adapted_skill_vals.mean() if len(adapted_skill_vals) > 0 else np.nan
                adapted_rmse = adapted_rmse_vals.mean() if len(adapted_rmse_vals) > 0 else np.nan

                skill_gain = adapted_skill - baseline_skill if not (np.isnan(baseline_skill) or np.isnan(adapted_skill)) else np.nan
                rmse_reduction = baseline_rmse - adapted_rmse if not (np.isnan(baseline_rmse) or np.isnan(adapted_rmse)) else np.nan

                results.append({
                    "method": method,
                    "K": int(K),
                    "variable": variable,
                    "skill_gain": skill_gain,
                    "rmse_reduction": rmse_reduction,
                })

    result_df = pd.DataFrame(results)
    out = OUT_TABLES / "adaptation_gain_by_K.csv"
    result_df.to_csv(out, index=False)
    print(f"[WROTE] {out}  ({len(result_df):,} rows)")
    return result_df


# ── Step 7: negative_adaptation_audit.csv ──────────────────────────────────────
def build_negative_audit(wide):
    """Per-method/variable/K breakdown of negative adaptation fraction."""
    results = []
    for method in wide["method"].unique():
        for variable in VARIABLES:
            for K in sorted(wide[wide["method"] == method]["K"].unique()):
                subset = wide[
                    (wide["method"] == method) &
                    (wide["variable"] == variable) &
                    (wide["K"] == K)
                ]
                skills = subset[subset["metric"] == "analysis_skill_vs_forecast"]["value_mean"]
                if len(skills) == 0:
                    continue
                neg_rate = (skills < 0).sum() / len(skills)
                results.append({
                    "method": method,
                    "variable": variable,
                    "K": int(K),
                    "n_total": len(skills),
                    "n_negative": int((skills < 0).sum()),
                    "negative_rate": neg_rate,
                    "skill_mean": skills.mean(),
                    "skill_min": skills.min(),
                })
    result_df = pd.DataFrame(results)
    out = OUT_TABLES / "negative_adaptation_audit.csv"
    result_df.to_csv(out, index=False)
    print(f"[WROTE] {out}  ({len(result_df):,} rows)")
    return result_df


# ── Step 8: row_count_audit.csv ───────────────────────────────────────────────
def build_row_audit(df):
    group_keys = ["method", "target_region_id", "K", "seed", "variable", "metric"]
    audit = df.groupby(group_keys, dropna=False).agg(
        row_count=("value", "count"),
        n_nan=("value", lambda x: x.isna().sum()),
        n_inf=("value", lambda x: np.isinf(x).sum()),
    ).reset_index()

    dup_mask = audit.duplicated(subset=group_keys, keep=False)
    audit["n_duplicate_keys"] = dup_mask.astype(int)

    out = OUT_TABLES / "row_count_audit.csv"
    audit.to_csv(out, index=False)
    print(f"[WROTE] {out}  ({len(audit):,} rows)")

    total = len(audit)
    missing = audit[audit["row_count"] == 0]
    nan_flags = audit[audit["n_nan"] > 0]
    inf_flags = audit[audit["n_inf"] > 0]
    print(f"  Total groups: {total}, missing: {len(missing)}, NaN flags: {len(nan_flags)}, Inf flags: {len(inf_flags)}")
    return audit


# ── Step 9: Figures ───────────────────────────────────────────────────────────
def _setup_fig(ax, xlabel, ylabel, title, font=None):
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontproperties=font)
    ax.grid(True, alpha=0.3)


def _get_method_order():
    return ["forecast_only", "source_mean_increment", "target_support_mean_increment",
            "target_monthly_support_increment", "ridge_core", "ridge_input_full", "ridge_input_geo_full"]


def plot_main_skill_by_method_K(main_df):
    """Line plot: skill vs K for each method, split by variable."""
    for variable in VARIABLES:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
        for ax, metric in zip(axes, ["analysis_skill_vs_forecast", "increment_rmse"]):
            for method in _get_method_order():
                sub = main_df[(main_df["method"] == method) & (main_df["variable"] == variable)]
                if len(sub) == 0:
                    continue
                ks = sorted(sub["K"].unique())
                means = [sub[sub["K"] == k][f"{metric}_mean"].mean() for k in ks]
                stds = [sub[sub["K"] == k][f"{metric}_std"].mean() for k in ks]
                label = METHOD_DISPLAY.get(method, method)
                ax.errorbar(ks, means, yerr=stds, marker="o", label=label, capsize=3)

            ylabel = "Skill" if metric == "analysis_skill_vs_forecast" else "RMSE"
            var_cn = "Surface" if variable == "surface" else "Rootzone"
            title = f"{var_cn} {ylabel} by K (10-seed mean ± std)"
            _setup_fig(ax, "K", ylabel, title, font=chinese_font)
            ax.legend(fontsize=8, prop=chinese_font)

        plt.tight_layout()
        out = OUT_FIGURES / f"fig_main_skill_by_method_K_{variable.lower()}.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[WROTE] {out}")


def plot_adaptation_curve(wide):
    """Adaptation curve: skill vs K for adaptive methods (K=4,12,24)."""
    for variable in VARIABLES:
        fig, ax = plt.subplots(figsize=(8, 5))
        adaptive_subset = [m for m in ADAPTIVE_METHODS if m != "source_mean_increment"]

        for method in adaptive_subset:
            sub = wide[(wide["method"] == method) & (wide["variable"] == variable)]
            skill_vals = sub[sub["metric"] == "analysis_skill_vs_forecast"]
            if len(skill_vals) == 0:
                continue
            by_K = skill_vals.groupby("K")["value_mean"].mean()
            ax.plot(by_K.index, by_K.values, marker="o", label=METHOD_DISPLAY.get(method, method))

        # baseline K=0 reference
        base_mask = (
            (wide["method"] == "forecast_only") &
            (wide["K"] == 0) &
            (wide["variable"] == variable) &
            (wide["metric"] == "analysis_skill_vs_forecast")
        )
        base_skill_vals = wide.loc[base_mask, "value_mean"]
        base_skill = base_skill_vals.mean() if len(base_skill_vals) > 0 else 0.0
        ax.axhline(base_skill, color="black", linestyle="--", label="预报 K=0", alpha=0.6)

        var_cn = "Surface" if variable == "surface" else "Rootzone"
        _setup_fig(ax, "K (support dates)", "Analysis Skill", f"{var_cn} 适配曲线", font=chinese_font)
        ax.legend(fontsize=9, prop=chinese_font)
        plt.tight_layout()

        out = OUT_FIGURES / f"fig_adaptation_curve_{variable.lower()}.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[WROTE] {out}")


def plot_region_method_heatmap(region_df):
    """Heatmap: regions × methods for analysis_skill_vs_forecast at K=4,12."""
    for variable in VARIABLES:
        fig, ax = plt.subplots(figsize=(14, 8))
        sub = region_df[(region_df["variable"] == variable) & (region_df["K"].isin([4, 12]))]
        if len(sub) == 0:
            print(f"[SKIP] heatmap {variable}: no data")
            continue

        # Average over seeds (region_df already seed-aggregated)
        pivot = sub.pivot_table(
            index="target_region_id",
            columns="method",
            values="analysis_skill_vs_forecast_mean",
            aggfunc="mean",
        )

        col_order = [c for c in _get_method_order() if c in pivot.columns]
        pivot = pivot[col_order]

        vmin = max(-0.3, pivot.values[~np.isnan(pivot.values)].min() - 0.05)
        vmax = min(0.5, pivot.values[~np.isnan(pivot.values)].max() + 0.05)
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([METHOD_DISPLAY.get(c, c) for c in pivot.columns], rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        var_cn = "Surface" if variable == "surface" else "Rootzone"
        ax.set_title(f"{var_cn} 分析 Skill（K=4,12 平均）：区域 × 方法", fontsize=11, fontproperties=chinese_font)
        plt.colorbar(im, ax=ax, label="Skill")
        plt.tight_layout()

        out = OUT_FIGURES / f"fig_region_method_heatmap_{variable.lower()}.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[WROTE] {out}")


# ── Step 10: Markdown report ──────────────────────────────────────────────────
def write_report(df, wide, main_df, region_df, gain_df, neg_df, row_audit):
    """Write phase3B_report_US.md answering all 8 required questions."""
    n_methods = len(wide["method"].unique())
    n_seeds = len(wide["seed"].unique())
    n_regions = len(wide["target_region_id"].unique())

    lines = [
        "# Phase 3B 摘要报告（US）— Full 10-Seed Evaluation\n",
        "**生成工具：** `scripts/analysis/summarize_phase3B.py`\n",
        f"**数据来源：** {n_methods} 个实验 CSV（phase3A + phase3B 全部7方法）\n",
        f"**覆盖范围：** {n_regions} regions × {n_seeds} seeds × all K\n",
        f"**加载的原始行数：** {len(df):,}\n",
        "\n",
        "---\n",
        "\n",
    ]

    # Q1: Strongest simple baseline overall
    lines.append("## Q1. 最强简单基线（整体）\n")
    skill_cols = main_df[main_df["metric"] == "analysis_skill_vs_forecast"][["method", "variable", "K", "analysis_skill_vs_forecast_mean"]]
    # Average across K and variables for each method
    if len(skill_cols) > 0:
        overall = skill_cols.groupby("method")["analysis_skill_vs_forecast_mean"].mean().sort_values(ascending=False)
        lines.append(f"**按 analysis_skill_vs_forecast 从高到低：**\n")
        for m, v in overall.items():
            lines.append(f"- {METHOD_DISPLAY.get(m, m)} ({m}): {v:.4f}\n")
    lines.append("\n---\n\n")

    # Q2: K stability
    lines.append("## Q2. K 稳定性（K=4→12→24 是否单调改善？）\n")
    for method in ["target_support_mean_increment", "ridge_core", "ridge_input_full", "ridge_input_geo_full"]:
        for var in VARIABLES:
            sub = main_df[(main_df["method"] == method) & (main_df["variable"] == var) & (main_df["K"].isin([4, 12, 24]))]
            if len(sub) < 3:
                continue
            vals = {int(r["K"]): r["analysis_skill_vs_forecast_mean"] for _, r in sub.iterrows()}
            k4 = vals.get(4, np.nan)
            k12 = vals.get(12, np.nan)
            k24 = vals.get(24, np.nan)
            mono = "单调改善" if (k4 <= k12 <= k24) else ("单调下降" if (k4 >= k12 >= k24) else "非单调")
            lines.append(f"- {METHOD_DISPLAY.get(method, method)} {var}: K4={k4:.4f} K12={k12:.4f} K24={k24:.4f} [{mono}]\n")
    lines.append("\n---\n\n")

    # Q3: Surface vs rootzone
    lines.append("## Q3. Surface vs Rootzone 难度对比\n")
    for method in _get_method_order():
        for K in sorted(main_df[main_df["method"] == method]["K"].unique()):
            s_val = main_df[(main_df["method"] == method) & (main_df["variable"] == "surface") & (main_df["K"] == K)]["analysis_skill_vs_forecast_mean"].mean()
            r_val = main_df[(main_df["method"] == method) & (main_df["variable"] == "rootzone") & (main_df["K"] == K)]["analysis_skill_vs_forecast_mean"].mean()
            if not np.isnan(s_val) and not np.isnan(r_val):
                harder = "rootzone" if s_val > r_val else "surface"
                lines.append(f"- {METHOD_DISPLAY.get(method, method)} K={K}: surface={s_val:.4f} rootzone={r_val:.4f} 较难={harder}\n")
    lines.append("\n---\n\n")

    # Q4: Hardest regions
    lines.append("## Q4. 最难区域（US-R1..R6 最低 skill）\n")
    hardest_rows = []
    for region in sorted(wide["target_region_id"].unique()):
        for var in VARIABLES:
            sub = wide[(wide["target_region_id"] == region) & (wide["variable"] == var) & (wide["metric"] == "analysis_skill_vs_forecast")]
            if len(sub) == 0:
                continue
            mean_skill = sub["value_mean"].mean()
            hardest_rows.append({"region": region, "variable": var, "mean_skill": mean_skill})
    hardest_df = pd.DataFrame(hardest_rows)
    if len(hardest_df) > 0:
        for var in VARIABLES:
            sub = hardest_df[hardest_df["variable"] == var].sort_values("mean_skill")
            lines.append(f"**{var}**（从低到高）：\n")
            for _, r in sub.iterrows():
                lines.append(f"- {r['region']}: {r['mean_skill']:.4f}\n")
    lines.append("\n---\n\n")

    # Q5: Negative adaptation
    lines.append("## Q5. 负适配（skill < 0 的比例）\n")
    neg_overall = neg_df.groupby("method")[["n_negative", "n_total"]].sum()
    neg_overall["rate"] = neg_overall["n_negative"] / neg_overall["n_total"]
    neg_overall = neg_overall.sort_values("rate", ascending=False)
    lines.append(f"| Method | Negative Rate | Nneg/Ntotal |\n")
    lines.append(f"|--------|-------------|------------|\n")
    for method, row in neg_overall.iterrows():
        lines.append(f"| {METHOD_DISPLAY.get(method, method)} | {row['rate']:.3f} | {int(row['n_negative'])}/{int(row['n_total'])} |\n")
    lines.append("\n---\n\n")

    # Q6: Ridge vs mean baselines
    lines.append("## Q6. Ridge vs Mean 基线对比\n")
    ridge_methods = ["ridge_core", "ridge_input_full", "ridge_input_geo_full"]
    mean_methods = ["target_support_mean_increment", "target_monthly_support_increment"]
    for K in [4, 12, 24]:
        lines.append(f"**K={K}**\n")
        for var in VARIABLES:
            ridge_vals = []
            for m in ridge_methods:
                v = main_df[(main_df["method"] == m) & (main_df["K"] == K) & (main_df["variable"] == var)]["analysis_skill_vs_forecast_mean"].mean()
                ridge_vals.append(v)
            mean_vals = []
            for m in mean_methods:
                if not (m == "target_monthly_support_increment" and K == 4):  # monthly not at K=4
                    v = main_df[(main_df["method"] == m) & (main_df["K"] == K) & (main_df["variable"] == var)]["analysis_skill_vs_forecast_mean"].mean()
                    mean_vals.append(v)
            ridge_mean = np.nanmean(ridge_vals)
            mean_mean = np.nanmean(mean_vals) if mean_vals else np.nan
            winner = "Ridge" if ridge_mean > mean_mean else "Mean"
            lines.append(f"- {var}: Ridge avg={ridge_mean:.4f} Mean avg={mean_mean:.4f} → {winner}\n")
    lines.append("\n---\n\n")

    # Q7: Phase 4 design implications
    lines.append("## Q7. Phase 4 神经基线设计启示\n")
    best_method = overall.idxmax() if len(overall) > 0 else "unknown"
    best_skill = overall.max() if len(overall) > 0 else 0.0
    lines.append(f"- 当前最强简单基线：{METHOD_DISPLAY.get(best_method, best_method)} (skill={best_skill:.4f})\n")
    # Check if ridge significantly outperforms mean
    ridge_skill = overall[overall.index.str.startswith("ridge")].mean() if len(overall) > 0 else 0.0
    mean_skill = overall[overall.index.isin(["source_mean_increment", "target_support_mean_increment", "target_monthly_support_increment"])].mean()
    lines.append(f"- Ridge 平均 skill: {ridge_skill:.4f}\n")
    lines.append(f"- Mean 平均 skill: {mean_skill:.4f}\n")
    if ridge_skill > 0.1:
        lines.append(f"- 结论：Ridge 已达 skill>0.1，神经基线需证明能超越线性模型方可 justification\n")
    elif ridge_skill > 0:
        lines.append(f"- 结论：Ridge 弱正 skill，神经方法有空间但需谨慎 justification\n")
    else:
        lines.append(f"- 结论：简单基线均接近或低于零，神经方法需大量工程努力方可超越\n")
    lines.append("\n---\n\n")

    # Q8: Anomaly audit
    lines.append("## Q8. 异常审计\n")
    nan_groups = row_audit[row_audit["n_nan"] > 0]
    inf_groups = row_audit[row_audit["n_inf"] > 0]
    dup_groups = row_audit[row_audit["n_duplicate_keys"] > 0]
    lines.append(f"- 含 NaN 的组数：{len(nan_groups):,}\n")
    lines.append(f"- 含 Inf 的组数：{len(inf_groups):,}\n")
    lines.append(f"- 重复 key 组数：{len(dup_groups):,}\n")

    # K=0 leakage compliance
    lines.append(f"\n**K=0 零泄漏合规检查：**\n")
    for method in wide["method"].unique():
        k0_rows = wide[(wide["method"] == method) & (wide["K"] == 0)]
        if len(k0_rows) > 0:
            lines.append(f"- {method}: {len(k0_rows)} K=0 rows (expected: forecast_only + source_mean_increment)\n")

    lines.append("\n---\n\n")

    # Row count audit summary
    lines.append("## 行数审计详情\n")
    if len(nan_groups) > 0:
        lines.append(f"\n**含 NaN 的组（TOP 10）：**\n")
        lines.append(nan_groups[["method", "target_region_id", "K", "seed", "variable", "metric", "n_nan"]].head(10).to_markdown(index=False))
        lines.append("\n")
    if len(inf_groups) > 0:
        lines.append(f"\n**含 Inf 的组（TOP 10）：**\n")
        lines.append(inf_groups[["method", "target_region_id", "K", "seed", "variable", "metric", "n_inf"]].head(10).to_markdown(index=False))
        lines.append("\n")

    lines.extend([
        "\n---\n\n",
        "## 输出路径\n",
        f"- **规范输出：** `{CANONICAL_ROOT}`\n",
        "\n",
        "*报告自动生成，请勿手动编辑。*\n",
    ])

    report_path = OUT_REPORT_DIR / "phase3B_report_US.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[WROTE] {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 3B Full Summary")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Override canonical output root path")
    args = parser.parse_args()

    global CANONICAL_ROOT, OUT_TABLES, OUT_FIGURES, OUT_REPORT_DIR
    if args.output_root:
        CANONICAL_ROOT = Path(args.output_root)
        OUT_TABLES = CANONICAL_ROOT / "tables"
        OUT_FIGURES = CANONICAL_ROOT / "figures"
        OUT_REPORT_DIR = CANONICAL_ROOT / "reports"
        OUT_TABLES.mkdir(parents=True, exist_ok=True)
        OUT_FIGURES.mkdir(parents=True, exist_ok=True)
        OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 3B — Full 10-Seed Summary")
    print("=" * 60)
    print(f"Output root: {CANONICAL_ROOT}")
    print("=" * 60)

    df = load_all()
    print(f"\nTotal rows: {len(df):,}\n")

    validate_time_steps(df)

    wide = build_metrics_eval_wide(df)
    main_df = build_table_main_by_method_K(wide)
    region_df = build_table_region_by_method_K(wide)
    gain_df = build_adaptation_gain(wide)
    neg_df = build_negative_audit(wide)
    row_audit = build_row_audit(df)

    print("\nGenerating figures...")
    plot_main_skill_by_method_K(main_df)
    plot_adaptation_curve(wide)
    plot_region_method_heatmap(region_df)

    print("\nWriting report...")
    write_report(df, wide, main_df, region_df, gain_df, neg_df, row_audit)

    print("\nDone.")


if __name__ == "__main__":
    main()
