#!/usr/bin/env python3
"""
Phase 3B-minimal Summary/Report Layer
======================================
Reads raw long-format metrics_long.csv files from phase3B_minimal experiments
and produces aggregated tables, figures, and a report.

Input CSVs:
  - artifacts/metrics/phase3A_forecast_only_US/metrics_long.csv
  - artifacts/metrics/phase3B_minimal_source_mean_increment_US/metrics_long.csv
  - artifacts/metrics/phase3B_minimal_target_support_mean_increment_US/metrics_long.csv
  - artifacts/metrics/phase3B_minimal_target_monthly_support_increment_US/metrics_long.csv
  - artifacts/metrics/phase3B_minimal_ridge_core_US/metrics_long.csv

Output (canonical):
  artifacts/experiments/phase3_simple_baselines/US/summaries/phase3B_minimal/
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
          phase3B_minimal_report_US.md

Legacy output (opt-in via --legacy-output):
  artifacts/metrics/phase3B_minimal_summary_US/ (same structure)
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
    # Try to find a Chinese font
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
        # Fallback: try to find any CJK font
        for fp in fm.findSystemFonts():
            if "Noto" in fp and ("CJK" in fp or "Serif" in fp):
                chinese_font = fm.FontProperties(fname=fp)
                break
except Exception:
    chinese_font = None

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path("/sharefiles1/fenglonghan/projects/hydroda_ood")
ARTIFACTS = BASE / "artifacts/metrics"

# Canonical output root (default)
CANONICAL_ROOT = BASE / "artifacts/experiments/phase3_simple_baselines/US/summaries/phase3B_minimal"
OUT_TABLES = CANONICAL_ROOT / "tables"
OUT_FIGURES = CANONICAL_ROOT / "figures"
OUT_REPORT_DIR = CANONICAL_ROOT / "reports"

# Legacy output root (opt-in)
LEGACY_ROOT = BASE / "artifacts/metrics/phase3B_minimal_summary_US"
LEGACY_TABLES = LEGACY_ROOT / "tables"
LEGACY_FIGURES = LEGACY_ROOT / "figures"

OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)
OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Input mapping: display_name -> path
INPUTS = {
    "forecast_only": ARTIFACTS / "phase3A_forecast_only_US/metrics_long.csv",
    "source_mean":   ARTIFACTS / "phase3B_minimal_source_mean_increment_US/metrics_long.csv",
    "tgt_mean":      ARTIFACTS / "phase3B_minimal_target_support_mean_increment_US/metrics_long.csv",
    "tgt_monthly":   ARTIFACTS / "phase3B_minimal_target_monthly_support_increment_US/metrics_long.csv",
    "ridge":         ARTIFACTS / "phase3B_minimal_ridge_core_US/metrics_long.csv",
}

# Legacy output paths for opt-in
LEGACY_OUT_DIR = LEGACY_ROOT
LEGACY_OUT_TABLES = LEGACY_TABLES
LEGACY_OUT_FIGURES = LEGACY_FIGURES

# Method display name map (Chinese)
METHOD_DISPLAY = {
    "forecast_only": "预报",
    "source_mean":   "源均值",
    "tgt_mean":      "目标均值",
    "tgt_monthly":   "月度",
    "ridge":         "岭回归",
}

METRICS = [
    "analysis_rmse", "analysis_mae", "analysis_skill_vs_forecast",
    "increment_rmse", "increment_mae", "increment_bias",
    "increment_corr", "sign_accuracy_deadzone",
]
VARIABLES = ["surface", "rootzone"]
REGION_BALANCED_METRICS = ["analysis_skill_vs_forecast", "increment_rmse", "increment_corr"]


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
              f"regions={sorted(df['target_region_id'].unique())}")
    combined = pd.concat(dfs, ignore_index=True)
    return combined


# ── Step 2: Validate n_time_steps ─────────────────────────────────────────────
def validate_time_steps(df):
    bad = df[df["n_time_steps"] != 1]
    if len(bad) > 0:
        print(f"[WARN] {len(bad)} rows with n_time_steps != 1")
    else:
        print("[OK] All rows have n_time_steps == 1 (each row = 1 time slice)")
    print(f"  Unique n_valid_pixels values: {sorted(df['n_valid_pixels'].unique())}")


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

    # Legacy output
    if args.legacy_output:
        LEGACY_OUT_TABLES.mkdir(parents=True, exist_ok=True)
        leg_out = LEGACY_OUT_TABLES / "metrics_eval_wide.csv"
        agg.to_csv(leg_out, index=False)
        print(f"[WROTE-LEGACY] {leg_out}")
    return agg


# ── Step 4: table_main_by_method_K.csv ───────────────────────────────────────
def build_table_main_by_method_K(wide):
    """Region-balanced: first mean over regions per K, then aggregate across K."""
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

                # negative adaptation rate: fraction of regions where skill < 0
                skill_vals = subset[subset["metric"] == "analysis_skill_vs_forecast"]["value_mean"]
                if len(skill_vals) > 0:
                    row["negative_adaptation_rate"] = (skill_vals < 0).sum() / len(skill_vals)
                else:
                    row["negative_adaptation_rate"] = np.nan

                # avg rank across metrics (lower = better), approximate using skill
                results.append(row)

    result_df = pd.DataFrame(results)
    out = OUT_TABLES / "table_main_by_method_K.csv"
    result_df.to_csv(out, index=False)
    print(f"[WROTE] {out}  ({len(result_df):,} rows)")

    if args.legacy_output:
        leg_out = LEGACY_OUT_TABLES / "table_main_by_method_K.csv"
        result_df.to_csv(leg_out, index=False)
        print(f"[WROTE-LEGACY] {leg_out}")
    return result_df


# ── Step 5: table_region_by_method_K.csv ─────────────────────────────────────
def build_table_region_by_method_K(wide):
    """Granularity: method × K × target_region_id × variable."""
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

    if args.legacy_output:
        leg_out = LEGACY_OUT_TABLES / "table_region_by_method_K.csv"
        result_df.to_csv(leg_out, index=False)
        print(f"[WROTE-LEGACY] {leg_out}")
    return result_df


# ── Step 6: adaptation_gain_by_K.csv ─────────────────────────────────────────
def build_adaptation_gain(wide):
    """K=0 baseline (forecast_only) vs K>0 methods."""
    results = []

    # The 'wide' DataFrame has columns: method, K, variable, metric, value_mean, ...
    # metric values include 'analysis_skill_vs_forecast', 'increment_rmse', etc.
    for method in ["tgt_mean", "tgt_monthly", "ridge"]:
        for K in sorted(wide[wide["method"] == method]["K"].unique()):
            for variable in VARIABLES:
                # baseline: forecast_only K=0 for this variable and metric
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

                results.append({
                    "method": method,
                    "K": int(K),
                    "variable": variable,
                    "skill_gain": adapted_skill - baseline_skill if not (np.isnan(baseline_skill) or np.isnan(adapted_skill)) else np.nan,
                    "rmse_reduction": baseline_rmse - adapted_rmse if not (np.isnan(baseline_rmse) or np.isnan(adapted_rmse)) else np.nan,
                })

    result_df = pd.DataFrame(results)
    out = OUT_TABLES / "adaptation_gain_by_K.csv"
    result_df.to_csv(out, index=False)
    print(f"[WROTE] {out}  ({len(result_df):,} rows)")

    if args.legacy_output:
        leg_out = LEGACY_OUT_TABLES / "adaptation_gain_by_K.csv"
        result_df.to_csv(leg_out, index=False)
        print(f"[WROTE-LEGACY] {leg_out}")
    return result_df


# ── Step 7: negative_adaptation_audit.csv ──────────────────────────────────────
def build_negative_audit(wide):
    results = []
    for method in wide["method"].unique():
        for variable in VARIABLES:
            subset = wide[(wide["method"] == method) & (wide["variable"] == variable)]
            skills = subset[subset["metric"] == "analysis_skill_vs_forecast"]["value_mean"]
            if len(skills) == 0:
                continue
            neg_rate = (skills < 0).sum() / len(skills)
            results.append({
                "method": method,
                "variable": variable,
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

    if args.legacy_output:
        leg_out = LEGACY_OUT_TABLES / "negative_adaptation_audit.csv"
        result_df.to_csv(leg_out, index=False)
        print(f"[WROTE-LEGACY] {leg_out}")
    return result_df


# ── Step 8: row_count_audit.csv ───────────────────────────────────────────────
def build_row_audit(df):
    group_keys = ["method", "target_region_id", "K", "seed", "variable", "metric"]
    audit = df.groupby(group_keys, dropna=False).agg(
        row_count=("value", "count"),
        n_nan=("value", lambda x: x.isna().sum()),
        n_inf=("value", lambda x: np.isinf(x).sum()),
    ).reset_index()

    # detect duplicates (same group key with different values)
    dup_mask = audit.duplicated(subset=group_keys, keep=False)
    audit["n_duplicate_keys"] = dup_mask.astype(int)

    out = OUT_TABLES / "row_count_audit.csv"
    audit.to_csv(out, index=False)
    print(f"[WROTE] {out}  ({len(audit):,} rows)")

    if args.legacy_output:
        leg_out = LEGACY_OUT_TABLES / "row_count_audit.csv"
        audit.to_csv(leg_out, index=False)
        print(f"[WROTE-LEGACY] {leg_out}")

    # summary
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
    return ["forecast_only", "source_mean", "tgt_mean", "tgt_monthly", "ridge"]


def _get_K_order(method):
    base = {"forecast_only": [0], "source_mean": [0], "tgt_mean": [0, 4, 12, 24], "tgt_monthly": [0, 4, 12, 24], "ridge": [4, 12]}
    return base.get(method, [])


def plot_main_skill_by_method_K(main_df):
    """Line plot: skill vs K for each method, split by variable."""
    for variable in VARIABLES:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
        variable_lower = variable.lower()

        for ax, metric in zip(axes, ["analysis_skill_vs_forecast", "increment_rmse"]):
            for method in _get_method_order():
                sub = main_df[(main_df["method"] == method) & (main_df["variable"] == variable)]
                if len(sub) == 0:
                    continue
                ks = sorted(sub["K"].unique())
                means = [sub[sub["K"] == k][f"{metric}_mean"].mean() for k in ks]
                stds  = [sub[sub["K"] == k][f"{metric}_std"].mean() for k in ks]
                label = METHOD_DISPLAY.get(method, method)
                ax.errorbar(ks, means, yerr=stds, marker="o", label=label, capsize=3)

            ylabel = "Skill" if metric == "analysis_skill_vs_forecast" else "RMSE"
            var_cn = "Surface" if variable == "surface" else "Rootzone"
            title = f"{var_cn} {ylabel} by K"
            _setup_fig(ax, "K", ylabel, title, font=chinese_font)
            ax.legend(fontsize=8, prop=chinese_font)

        plt.tight_layout()
        suffix = variable.lower()
        out = OUT_FIGURES / f"fig_main_skill_by_method_K_{suffix}.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[WROTE] {out}")

        if args.legacy_output:
            LEGACY_OUT_FIGURES.mkdir(parents=True, exist_ok=True)
            leg_out = LEGACY_OUT_FIGURES / f"fig_main_skill_by_method_K_{suffix}.png"
            plt.savefig(leg_out, dpi=300)
            print(f"[WROTE-LEGACY] {leg_out}")


def plot_adaptation_curve(wide):
    """Adaptation curve: skill vs K for adaptive methods."""
    for variable in VARIABLES:
        fig, ax = plt.subplots(figsize=(8, 5))
        adaptive_methods = ["tgt_mean", "tgt_monthly", "ridge"]

        for method in adaptive_methods:
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
        ax.axhline(base_skill, color="black", linestyle="--", label="预报", alpha=0.6)

        var_cn = "Surface" if variable == "surface" else "Rootzone"
        _setup_fig(ax, "K (support dates)", "Analysis Skill", f"{var_cn} 适配曲线", font=chinese_font)
        ax.legend(fontsize=9, prop=chinese_font)
        plt.tight_layout()

        suffix = variable.lower()
        out = OUT_FIGURES / f"fig_adaptation_curve_{suffix}.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[WROTE] {out}")

        if args.legacy_output:
            LEGACY_OUT_FIGURES.mkdir(parents=True, exist_ok=True)
            leg_out = LEGACY_OUT_FIGURES / f"fig_adaptation_curve_{suffix}.png"
            plt.savefig(leg_out, dpi=300)
            print(f"[WROTE-LEGACY] {leg_out}")


def plot_region_method_heatmap(region_df):
    """Heatmap: regions × methods for main skill metric."""
    for variable in VARIABLES:
        fig, ax = plt.subplots(figsize=(12, 8))
        sub = region_df[(region_df["variable"] == variable) & (region_df["K"].isin([4, 12]))]
        if len(sub) == 0:
            print(f"[SKIP] heatmap {variable}: no data")
            continue

        pivot = sub.pivot_table(
            index="target_region_id",
            columns="method",
            values="analysis_skill_vs_forecast_mean",
            aggfunc="mean",
        )

        # reorder columns
        col_order = [c for c in _get_method_order() if c in pivot.columns]
        pivot = pivot[col_order]

        im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu", vmin=-0.3, vmax=0.5)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([METHOD_DISPLAY.get(c, c) for c in pivot.columns], rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        var_cn = "Surface" if variable == "surface" else "Rootzone"
        ax.set_title(f"{var_cn} 分析 Skill：区域 × 方法", fontsize=11, fontproperties=chinese_font)
        plt.colorbar(im, ax=ax, label="Skill")
        plt.tight_layout()

        out = OUT_FIGURES / f"fig_region_method_heatmap_{variable.lower()}.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[WROTE] {out}")

        if args.legacy_output:
            LEGACY_OUT_FIGURES.mkdir(parents=True, exist_ok=True)
            leg_out = LEGACY_OUT_FIGURES / f"fig_region_method_heatmap_{variable.lower()}.png"
            plt.savefig(leg_out, dpi=300)
            print(f"[WROTE-LEGACY] {leg_out}")


# ── Step 10: Markdown report ──────────────────────────────────────────────────
def write_report(df, wide, main_df, region_df, gain_df, neg_df, row_audit):
    lines = [
        "# Phase 3B-minimal 摘要报告（US）\n",
        "**生成工具：** `scripts/analysis/summarize_phase3B_minimal.py`\n",
        f"**数据来源：** 5 个实验 CSV（phase3A + phase3B_minimal 变体）\n",
        f"**加载的原始行数：** {len(df):,}\n",
        "\n",
        "---\n",
        "\n",
        "## 摘要\n",
        "\n",
        "本报告展示 Phase 3B-minimal 实验的 first-pass 聚合结果。所有 adaptation baseline 在 first-pass 聚合中均表现为负 skill_vs_forecast，但这是**待验证现象，不能直接作为最终科学结论**。下一步必须检查 metric definition、forecast alignment、increment sign convention、NaN 分类和 raw metric schema。\n",
        "\n",
        "---\n",
        "\n",
        "## 数据结构说明\n",
        "\n",
        "原始 `metrics_long.csv` 中每一行对应**一个时间片**（`n_time_steps=1`）。`n_valid_pixels` 字段是隐式时间标识符，但**无法唯一映射到具体 query date**。因此，仅凭长格式文件无法唯一还原到具体 query date。\n",
        "\n",
        "---\n",
        "\n",
        "## 行数审计摘要\n",
        "\n",
        f"- 总聚合组数：{len(row_audit):,}\n",
        f"- 含 NaN 的组数：{(row_audit['n_nan'] > 0).sum()}\n",
        f"- 含 Inf 的组数：{(row_audit['n_inf'] > 0).sum()}\n",
        "\n",
        "---\n",
        "\n",
        "## 主结果：按方法和 K\n",
        "\n",
        "### Surface — Analysis Skill vs Forecast\n",
    ]

    # surface skill table
    surf = main_df[main_df["variable"] == "surface"][["method", "K", "analysis_skill_vs_forecast_mean", "analysis_skill_vs_forecast_std"]].dropna()
    if len(surf) > 0:
        surf = surf.sort_values(["K", "method"])
        lines.append(surf.to_markdown(index=False))
    lines.append("\n\n### Rootzone — Analysis Skill vs Forecast\n")
    root = main_df[main_df["variable"] == "rootzone"][["method", "K", "analysis_skill_vs_forecast_mean", "analysis_skill_vs_forecast_std"]].dropna()
    if len(root) > 0:
        root = root.sort_values(["K", "method"])
        lines.append(root.to_markdown(index=False))

    lines.extend([
        "\n\n---\n",
        "\n",
        "## 适配增益（相对于 Forecast K=0）\n",
    ])
    if len(gain_df) > 0:
        lines.append(gain_df.to_markdown(index=False))

    lines.extend([
        "\n\n---\n",
        "\n",
        "## 负适配审计\n",
        "\n",
        "`analysis_skill_vs_forecast < 0` 的区域-K 单元比例：\n",
    ])
    if len(neg_df) > 0:
        lines.append(neg_df.to_markdown(index=False))

    lines.extend([
        "\n\n---\n",
        "\n",
        "## 当前结论的限制\n",
        "\n",
        "1. **Metric definition 未审计**：analysis_skill_vs_forecast 的定义（是 forecast - analysis 还是 analysis - forecast）需确认。\n",
        "2. **Forecast alignment 未验证**：K=0 forecast_only 应有 skill ≈ 0，但实际表现需验证。\n",
        "3. **Increment sign convention 未验证**：增量符号约定（Analysis - Forecast 还是 Forecast - Analysis）需确认。\n",
        "4. **NaN 分类未完成**：3,840 个含 NaN 的组中，哪些是合法的空组，哪些是异常缺失，需区分。\n",
        "5. **泄漏审计未执行**：需确认 support selection 中没有使用目标查询标签。\n",
        "6. **Ridge 病态矩阵警告**：ridge_core 在运行中出现 ill-conditioned warning，结果可能不稳定。\n",
        "\n",
        "---\n",
        "\n",
        "## 输出路径\n",
        "\n",
        f"- **规范输出：** `{CANONICAL_ROOT}`\n",
        f"- **Legacy 输出：** `{LEGACY_ROOT}`\n",
        "\n",
        "*报告自动生成，请勿手动编辑。*\n",
    ])

    report_path = OUT_REPORT_DIR / "phase3B_minimal_report_US.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[WROTE] {report_path}")

    if args.legacy_output:
        LEGACY_OUT_DIR.mkdir(parents=True, exist_ok=True)
        leg_report = LEGACY_OUT_DIR / "phase3B_minimal_report_US.md"
        with open(leg_report, "w") as f:
            f.write("\n".join(lines))
        print(f"[WROTE-LEGACY] {leg_report}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global args
    parser = argparse.ArgumentParser(description="Phase 3B-minimal Summary")
    parser.add_argument("--legacy-output", action="store_true",
                        help="Also write outputs to legacy path artifacts/metrics/phase3B_minimal_summary_US/")
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
    print("Phase 3B-minimal Summary")
    print("=" * 60)
    print(f"Canonical output: {CANONICAL_ROOT}")
    if args.legacy_output:
        print(f"Legacy output:    {LEGACY_ROOT}")
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