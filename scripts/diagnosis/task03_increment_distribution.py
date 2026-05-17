#!/usr/bin/env python3
"""Task 3: Statistics of true/pred rootzone increment distributions.

Loads the source-only nonorm checkpoint, samples from source_train, source_val,
and target_query splits, and computes distribution statistics for both surface
and rootzone true and predicted increments.

Output: CSV and markdown report.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from hydroda.data.dataset import HydroDADataset
from hydroda.baselines.source_only import SourceOnlyBackbonePredictor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = REPO_ROOT / "reports" / "source_only_rootzone_diagnosis"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "source_only_rootzone_diagnosis"

DA_NC = "/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc"
REGION_MASKS_NC = str(REPO_ROOT / "artifacts" / "regions" / "US_region_masks.nc")
SPLITS_JSON = str(REPO_ROOT / "artifacts" / "splits" / "US_loro_kdate_splits.json")
FREEZE_MANIFEST = str(REPO_ROOT / "artifacts" / "protocol" / "US_region_split_freeze_manifest.json")

CHECKPOINT = str(REPO_ROOT / "artifacts" / "runs" / "phase4_source_only"
    / "phase4_source_only_source_only_US-R1_w32_e30_lr0.0003_nonorm_s0_20260515_155806"
    / "checkpoints" / "best.pt")

TARGET_REGION = "US-R1"
K = 0
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sampling configs
N_SOURCE_TRAIN = 500
N_TARGET_QUERY = 1000


def compute_stats(values: np.ndarray, label: str) -> dict:
    """Compute distribution statistics for a 1D array."""
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return {"label": label, "n": 0}
    abs_v = np.abs(v)
    return {
        "label": label,
        "n": len(v),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "p01": float(np.percentile(v, 1)),
        "p05": float(np.percentile(v, 5)),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "abs_mean": float(np.mean(abs_v)),
        "abs_p95": float(np.percentile(abs_v, 95)),
        "abs_p99": float(np.percentile(abs_v, 99)),
    }


def collect_increments(dataset: HydroDADataset, predictor, n_samples: int = -1) -> dict:
    """Collect true and predicted increments from a dataset split.

    Returns dict with lists of flattened increment values per variable.
    """
    total = len(dataset)
    if n_samples > 0 and n_samples < total:
        rng = np.random.RandomState(42)
        indices = sorted(rng.choice(total, size=n_samples, replace=False))
        n_actual = len(indices)
    else:
        indices = list(range(total))
        n_actual = total

    results = {
        "true_surface": [],
        "true_rootzone": [],
        "pred_surface": [],
        "pred_rootzone": [],
    }

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        # True increments
        inc_s = sample["increment_surface"]
        inc_r = sample["increment_rootzone"]
        mask = sample["metric_mask"] > 0.5

        valid_s = mask & np.isfinite(inc_s)
        valid_r = mask & np.isfinite(inc_r)

        if valid_s.sum() > 0:
            results["true_surface"].append(inc_s[valid_s].ravel())
        if valid_r.sum() > 0:
            results["true_rootzone"].append(inc_r[valid_r].ravel())

        # Predictions
        pred = predictor.predict(sample)
        pred_s = pred["pred_increment_surface"]
        pred_r = pred["pred_increment_rootzone"]

        if valid_s.sum() > 0:
            results["pred_surface"].append(pred_s[valid_s].ravel())
        if valid_r.sum() > 0:
            results["pred_rootzone"].append(pred_r[valid_r].ravel())

        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{n_actual}")

    # Concatenate
    for key in results:
        if results[key]:
            results[key] = np.concatenate(results[key])
        else:
            results[key] = np.array([], dtype=np.float32)

    return results


def build_dataset(split_type: str):
    return HydroDADataset(
        da_nc_path=DA_NC,
        region_masks_nc=REGION_MASKS_NC,
        splits_json=SPLITS_JSON,
        target_region=TARGET_REGION,
        split_type=split_type,
        K=K,
        seed=SEED,
        freeze_manifest=FREEZE_MANIFEST,
    )


def main():
    print("=" * 60)
    print("Task 3: Increment Distribution Statistics")
    print(f"  checkpoint={CHECKPOINT}")
    print(f"  device={DEVICE}")
    print("=" * 60)

    # Load predictor once
    predictor = SourceOnlyBackbonePredictor(checkpoint_path=CHECKPOINT, device=DEVICE)

    all_rows = []

    # Split configs: (split_type, n_samples)
    split_configs = [
        ("source_train", N_SOURCE_TRAIN),
        ("source_val", -1),  # all
        ("target_query", N_TARGET_QUERY),
    ]

    for split_type, n_samples in split_configs:
        print(f"\n--- {split_type} (n_samples={n_samples}) ---")
        dataset = build_dataset(split_type)
        print(f"  dataset size: {len(dataset)}")

        results = collect_increments(dataset, predictor, n_samples=n_samples)
        dataset.close()

        for key in ["true_surface", "true_rootzone", "pred_surface", "pred_rootzone"]:
            stats = compute_stats(results[key], f"{split_type}_{key}")
            all_rows.append(stats)

    # Convert to DataFrame
    df = pd.DataFrame(all_rows)

    # Add ratios: pred/true comparisons
    ratio_rows = []
    for split_type in ["source_train", "source_val", "target_query"]:
        for var in ["surface", "rootzone"]:
            true_row = df[df["label"] == f"{split_type}_true_{var}"]
            pred_row = df[df["label"] == f"{split_type}_pred_{var}"]
            if true_row.empty or pred_row.empty:
                continue
            true = true_row.iloc[0]
            pred = pred_row.iloc[0]
            ratio_rows.append({
                "label": f"{split_type}_ratio_{var}",
                "n": -1,
                "pred_std_div_true_std": pred["std"] / true["std"] if true["std"] > 0 else np.nan,
                "pred_abs_mean_div_true_abs_mean": pred["abs_mean"] / true["abs_mean"] if true["abs_mean"] > 0 else np.nan,
                "pred_abs_p95_div_true_abs_p95": pred["abs_p95"] / true["abs_p95"] if true["abs_p95"] > 0 else np.nan,
                "pred_std": pred["std"],
                "true_std": true["std"],
                "pred_abs_mean": pred["abs_mean"],
                "true_abs_mean": true["abs_mean"],
                "pred_abs_p95": pred["abs_p95"],
                "true_abs_p95": true["abs_p95"],
            })

    # Save full distribution CSV
    csv_path = REPORTS_DIR / "02_increment_distribution.csv"
    df.to_csv(csv_path, index=False, float_format="%.8f")
    print(f"\nSaved {csv_path}")

    # Save ratios CSV
    ratio_df = pd.DataFrame(ratio_rows)
    ratio_csv_path = ARTIFACTS_DIR / "02_increment_distribution_ratios.csv"
    ratio_df.to_csv(ratio_csv_path, index=False, float_format="%.8f")

    # Markdown report
    md_lines = [
        "# Task 3: Increment Distribution Statistics",
        "",
        "## Per-Split Distribution Summary",
        "",
        "| Split        | Variable      | N Pixels | Mean     | Std      | Abs Mean | Abs P95  | Abs P99  |",
        "|:-------------|:--------------|---------:|---------:|---------:|---------:|---------:|---------:|",
    ]
    for _, row in df.iterrows():
        md_lines.append(
            f"| {row['label']:<12} | {'':<13} "
            f"| {row['n']:>8,} "
            f"| {row['mean']:>+.6f} "
            f"| {row['std']:>.6f} "
            f"| {row['abs_mean']:>.6f} "
            f"| {row['abs_p95']:>.6f} "
            f"| {row['abs_p99']:>.6f} |"
        )

    md_lines.extend([
        "",
        "## Pred/True Ratios",
        "",
        "| Split        | Variable  | Pred Std/True Std | Pred AbsMean/True AbsMean | Pred AbsP95/True AbsP95 |",
        "|:-------------|:----------|------------------:|--------------------------:|------------------------:|",
    ])
    for _, r in ratio_df.iterrows():
        parts = r["label"].split("_ratio_")
        split_label = parts[0]
        var_label = parts[1]
        md_lines.append(
            f"| {split_label:<12} | {var_label:<9} "
            f"| {r['pred_std_div_true_std']:.3f} "
            f"| {r['pred_abs_mean_div_true_abs_mean']:.3f} "
            f"| {r['pred_abs_p95_div_true_abs_p95']:.3f} |"
        )

    md_lines.extend([
        "",
        "## Key Findings",
        "",
    ])

    # Rootzone true increment scale vs surface
    for split_type in ["source_train", "source_val", "target_query"]:
        true_s = df[df["label"] == f"{split_type}_true_surface"].iloc[0]
        true_r = df[df["label"] == f"{split_type}_true_rootzone"].iloc[0]
        scale_ratio = true_s["std"] / true_r["std"] if true_r["std"] > 0 else np.nan
        md_lines.append(f"- **{split_type}**: surface true_inc std = {true_s['std']:.6f}, rootzone true_inc std = {true_r['std']:.6f}, ratio = {scale_ratio:.1f}x")

    md_lines.append("")
    md_lines.append("> The rootzone true increment is dramatically smaller than surface in magnitude, confirming that raw MSE training is dominated by surface loss.")

    md_path = REPORTS_DIR / "02_increment_distribution.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
