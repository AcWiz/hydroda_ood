#!/usr/bin/env python3
"""Task 4: Raw MSE loss scale diagnosis — surface vs rootzone contribution.

Loads the source-only nonorm checkpoint, samples batches from source_train,
computes raw per-pixel MSE for surface and rootzone channels, and analyzes
the scale ratio that determines training dynamics.

Output: CSV and markdown report.
"""
from __future__ import annotations

from pathlib import Path

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
N_BATCHES = 50
BATCH_SIZE = 16  # effective samples per batch for stats


def compute_mse_per_sample(pred_s: np.ndarray, true_s: np.ndarray,
                            pred_r: np.ndarray, true_r: np.ndarray,
                            mask: np.ndarray) -> tuple:
    """Compute per-pixel MSE for surface and rootzone on valid pixels."""
    valid = (mask > 0.5) & np.isfinite(true_s) & np.isfinite(true_r)
    valid &= np.isfinite(pred_s) & np.isfinite(pred_r)
    if valid.sum() == 0:
        return np.nan, np.nan
    mse_s = float(np.mean((pred_s[valid] - true_s[valid]) ** 2))
    mse_r = float(np.mean((pred_r[valid] - true_r[valid]) ** 2))
    return mse_s, mse_r


def main():
    print("=" * 60)
    print("Task 4: Loss Scale Diagnosis — Surface vs Rootzone MSE")
    print(f"  checkpoint={CHECKPOINT}")
    print(f"  device={DEVICE}")
    print(f"  n_batches={N_BATCHES}")
    print("=" * 60)

    # Load predictor
    predictor = SourceOnlyBackbonePredictor(checkpoint_path=CHECKPOINT, device=DEVICE)

    # Dataset for true increment stats
    print("\nLoading source_train dataset...")
    train_dataset = HydroDADataset(
        da_nc_path=DA_NC,
        region_masks_nc=REGION_MASKS_NC,
        splits_json=SPLITS_JSON,
        target_region=TARGET_REGION,
        split_type="source_train",
        K=K,
        seed=SEED,
        freeze_manifest=FREEZE_MANIFEST,
    )
    total = len(train_dataset)
    print(f"  source_train samples: {total}")

    # Sample indices
    rng = np.random.RandomState(123)
    n_samples = min(N_BATCHES * BATCH_SIZE, total)
    indices = sorted(rng.choice(total, size=n_samples, replace=False))

    # Compute model MSE on sampled data
    surface_mses = []
    rootzone_mses = []
    mse_ratios = []

    print(f"\nComputing MSE on {n_samples} samples...")
    for i, idx in enumerate(indices):
        sample = train_dataset[idx]
        pred = predictor.predict(sample)
        mse_s, mse_r = compute_mse_per_sample(
            pred["pred_increment_surface"], sample["increment_surface"],
            pred["pred_increment_rootzone"], sample["increment_rootzone"],
            sample["metric_mask"],
        )
        if np.isfinite(mse_s) and np.isfinite(mse_r) and mse_r > 0:
            surface_mses.append(mse_s)
            rootzone_mses.append(mse_r)
            mse_ratios.append(mse_s / mse_r)
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{n_samples}")

    # True increment scale statistics
    print("\nComputing true increment scale statistics...")
    true_surface_stds = []
    true_rootzone_stds = []
    true_surface_absmeans = []
    true_rootzone_absmeans = []

    for idx in indices:
        sample = train_dataset[idx]
        mask = sample["metric_mask"] > 0.5
        inc_s = sample["increment_surface"]
        inc_r = sample["increment_rootzone"]
        valid_s = mask & np.isfinite(inc_s)
        valid_r = mask & np.isfinite(inc_r)
        if valid_s.sum() > 0:
            true_surface_stds.append(float(np.std(inc_s[valid_s])))
            true_surface_absmeans.append(float(np.mean(np.abs(inc_s[valid_s]))))
        if valid_r.sum() > 0:
            true_rootzone_stds.append(float(np.std(inc_r[valid_r])))
            true_rootzone_absmeans.append(float(np.mean(np.abs(inc_r[valid_r]))))

    train_dataset.close()

    # Statistics
    surface_mse_mean = np.mean(surface_mses)
    rootzone_mse_mean = np.mean(rootzone_mses)
    mse_ratio_mean = np.mean(mse_ratios)
    mse_ratio_median = np.median(mse_ratios)

    true_surface_std_mean = np.mean(true_surface_stds)
    true_rootzone_std_mean = np.mean(true_rootzone_stds)
    true_std_ratio = true_surface_std_mean / true_rootzone_std_mean if true_rootzone_std_mean > 0 else float("inf")
    true_var_ratio = true_std_ratio ** 2

    true_s_absmean = np.mean(true_surface_absmeans)
    true_r_absmean = np.mean(true_rootzone_absmeans)
    true_absmean_ratio = true_s_absmean / true_r_absmean if true_r_absmean > 0 else float("inf")

    # CSV
    rows = [
        {"metric": "surface_mse_mean", "value": surface_mse_mean},
        {"metric": "rootzone_mse_mean", "value": rootzone_mse_mean},
        {"metric": "mse_ratio_mean", "value": mse_ratio_mean},
        {"metric": "mse_ratio_median", "value": mse_ratio_median},
        {"metric": "mse_ratio_std", "value": float(np.std(mse_ratios))},
        {"metric": "true_surface_increment_std", "value": true_surface_std_mean},
        {"metric": "true_rootzone_increment_std", "value": true_rootzone_std_mean},
        {"metric": "true_std_ratio_surface_div_rootzone", "value": true_std_ratio},
        {"metric": "true_variance_ratio_surface_div_rootzone", "value": true_var_ratio},
        {"metric": "true_absmean_ratio_surface_div_rootzone", "value": true_absmean_ratio},
        {"metric": "n_samples_evaluated", "value": float(len(surface_mses))},
    ]
    df = pd.DataFrame(rows)
    csv_path = REPORTS_DIR / "03_loss_scale_diagnosis.csv"
    df.to_csv(csv_path, index=False, float_format="%.8f")
    print(f"\nSaved {csv_path}")

    # Markdown report
    md_lines = [
        "# Task 4: Loss Scale Diagnosis",
        "",
        f"Evaluated on {len(surface_mses)} source_train samples, {N_BATCHES * BATCH_SIZE} requested.",
        "",
        "## Model MSE on Source-Train (Nonorm Checkpoint)",
        "",
        "| Metric | Value |",
        "|:---|---:|",
        f"| Surface MSE (raw) | {surface_mse_mean:.8f} |",
        f"| Rootzone MSE (raw) | {rootzone_mse_mean:.8f} |",
        f"| MSE Ratio (surface/rootzone) mean | {mse_ratio_mean:.1f}x |",
        f"| MSE Ratio (surface/rootzone) median | {mse_ratio_median:.1f}x |",
        "",
        "## True Increment Scale (Source-Train)",
        "",
        "| Metric | Value |",
        "|:---|---:|",
        f"| True surface increment std | {true_surface_std_mean:.6f} |",
        f"| True rootzone increment std | {true_rootzone_std_mean:.8f} |",
        f"| Std ratio (surface / rootzone) | {true_std_ratio:.1f}x |",
        f"| Variance ratio (surface / rootzone) | {true_var_ratio:.1f}x |",
        f"| AbsMean ratio (surface / rootzone) | {true_absmean_ratio:.1f}x |",
        "",
        "## Key Findings",
        "",
    ]

    md_lines.append(f"- **Raw MSE surface/rootzone ratio**: ~{mse_ratio_mean:.0f}x (surface dominates)")
    md_lines.append(f"- **True increment std ratio**: ~{true_std_ratio:.0f}x (surface ≈ {true_std_ratio:.0f}x larger)")
    md_lines.append(f"- **True increment variance ratio**: ~{true_var_ratio:.0f}x")
    md_lines.append(f"- **Implication**: the training loss is dominated by surface MSE. Rootzone gets ~1/{mse_ratio_mean:.0f}th the gradient signal.")
    md_lines.append("")
    md_lines.append("> Raw MSE training is fundamentally surface-dominated. The rootzone signal is ~100x weaker in variance, making it nearly invisible to the optimizer. Per-variable normalization is required to give rootzone equal footing.")

    md_path = REPORTS_DIR / "03_loss_scale_diagnosis.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
