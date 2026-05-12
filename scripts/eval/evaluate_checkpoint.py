#!/usr/bin/env python3
"""Evaluate trained neural backbone checkpoints (source-only or prompt-conditioned).

Usage:
    # Source-only backbone
    PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \\
        --checkpoint artifacts/checkpoints/phase4_source_only/US-R1/best.pt \\
        --target_region US-R1 --K 0 --seed 0 \\
        --split_type target_query --predictor_type source_only

    # Prompt-conditioned shared backbone
    PYTHONPATH=. python scripts/eval/evaluate_checkpoint.py \\
        --checkpoint artifacts/checkpoints/phase4_prompt_conditioned/US-R1/best.pt \\
        --target_region US-R1 --K 0 --seed 0 \\
        --split_type target_query --predictor_type prompt_conditioned

No-leakage declaration:
    - Evaluation uses target_query split (post-prediction label use only)
    - No target_query labels used in training/normalization/prompt
    - Reuses evaluate_split() from harness.py
    - Metrics computed post-prediction with LeakageGuard protection
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hydroda.baselines.source_only import SourceOnlyBackbonePredictor
from hydroda.data.dataset import HydroDADataset
from hydroda.evaluation.harness import evaluate_split
from hydroda.utils.device import resolve_device


DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
REGION_MASKS_NC = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_kdate_splits.json"
FREEZE_MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"
PROTOCOL_FREEZE_ID = "hyperda_v4_final_2015_2025_context2022_query2023_2025_k0_4_12"

_PREDICTOR_OUTPUT_DIRS = {
    "source_only": Path("artifacts/results/phase4_source_only"),
    "prompt_conditioned": Path("artifacts/results/phase4_prompt_conditioned"),
}


def aggregate_results(rows):
    """Aggregate metrics by region, season, variable."""
    if not rows:
        return {}
    df = pd.DataFrame(rows)

    by_region = (
        df.groupby(["target_region_id", "variable", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    by_region.columns = ["target_region_id", "variable", "metric", "mean", "std", "count"]

    by_season = (
        df.groupby(["season", "variable", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    by_season.columns = ["season", "variable", "metric", "mean", "std", "count"]

    return {
        "by_region": by_region.to_dict(orient="records"),
        "by_season": by_season.to_dict(orient="records"),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate neural backbone checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--target_region", type=str, required=True)
    parser.add_argument("--K", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split_type", type=str, default="target_query")
    parser.add_argument("--max_samples", type=int, default=0,
        help="Max samples to evaluate (0 = no limit, evaluate all)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--require_gpu", action="store_true",
        help="Exit with error if CUDA unavailable")
    parser.add_argument("--predictor_type", type=str, default="source_only",
        choices=["source_only", "prompt_conditioned"],
        help="Type of predictor to load")
    parser.add_argument("--output_dir", type=str, default=None,
        help="Override output directory")
    args = parser.parse_args()

    # Resolve device
    device = resolve_device(args.device, require_gpu=args.require_gpu)

    ckpt_path = Path(args.checkpoint)

    # Determine output directory
    if args.output_dir:
        region_output_dir = Path(args.output_dir) / args.target_region
    else:
        base_dir = _PREDICTOR_OUTPUT_DIRS.get(args.predictor_type, Path("artifacts/results/phase4"))
        region_output_dir = base_dir / args.target_region
    region_output_dir.mkdir(parents=True, exist_ok=True)

    phase_label = "Phase 4A" if args.predictor_type == "source_only" else "Phase 4B"
    print("=" * 60)
    print(f"{phase_label}: Neural Backbone Evaluation")
    print(f"  predictor_type={args.predictor_type}")
    print(f"  checkpoint={ckpt_path}")
    print(f"  target_region={args.target_region}  K={args.K}  seed={args.seed}")
    print(f"  split_type={args.split_type}  max_samples={args.max_samples if args.max_samples > 0 else 'all'}")
    print(f"  device={device}")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading dataset ({args.split_type})...")
    dataset = HydroDADataset(
        da_nc_path=f"{DATA_DIR}/DA.nc",
        region_masks_nc=REGION_MASKS_NC,
        splits_json=SPLITS_JSON,
        target_region=args.target_region,
        split_type=args.split_type,
        K=args.K,
        seed=args.seed,
        freeze_manifest=FREEZE_MANIFEST,
    )

    total_samples = len(dataset)
    n_samples = min(total_samples, args.max_samples) if args.max_samples > 0 else total_samples
    print(f"  dataset size: {total_samples}, evaluating {n_samples} samples")

    # Load predictor
    print(f"\nLoading checkpoint...")
    if args.predictor_type == "prompt_conditioned":
        from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor

        predictor = PromptConditionedBackbonePredictor(
            checkpoint_path=str(ckpt_path),
            device=str(device),
            target_region=args.target_region,
        )
    else:
        predictor = SourceOnlyBackbonePredictor(
            checkpoint_path=str(ckpt_path),
            device=str(device),
        )
    print(f"  method: {predictor.method_name}")

    # Run evaluation
    print(f"\nRunning evaluation...")
    start_time = time.time()

    rows = evaluate_split(
        dataset=dataset,
        predictor=predictor,
        split_role=args.split_type,
        experiment_id=f"phase4_{args.predictor_type}_{args.target_region}_K{args.K}_S{args.seed}",
        protocol_freeze_id=PROTOCOL_FREEZE_ID,
        method=predictor.method_name,
        split_file=SPLITS_JSON,
        mask_file=REGION_MASKS_NC,
        preloaded=False,
        max_samples=args.max_samples if args.max_samples > 0 else None,
    )

    elapsed = time.time() - start_time
    print(f"  Evaluation done in {elapsed:.1f}s — {len(rows)} metric rows")

    dataset.close()

    if not rows:
        print("  WARNING: No rows generated. Check dataset/predictor.")
        summary = {"status": "no_rows", "checkpoint": str(ckpt_path)}
        with open(region_output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return

    # Save long-form results
    df = pd.DataFrame(rows)
    long_path = region_output_dir / "metrics_long.csv"
    df.to_csv(long_path, index=False)
    print(f"  Saved {len(rows)} rows to {long_path}")

    # Aggregate
    agg = aggregate_results(rows)

    if agg.get("by_region"):
        by_region_df = pd.DataFrame(agg["by_region"])
        by_region_path = region_output_dir / "metrics_by_region.csv"
        by_region_df.to_csv(by_region_path, index=False)

    if agg.get("by_season"):
        by_season_df = pd.DataFrame(agg["by_season"])
        by_season_path = region_output_dir / "metrics_by_season.csv"
        by_season_df.to_csv(by_season_path, index=False)

    # Summary
    skill_rows = df[df["metric"] == "analysis_skill_vs_forecast"]
    inc_rmse_rows = df[df["metric"] == "increment_rmse"]
    inc_corr_rows = df[df["metric"] == "increment_corr"]

    summary = {
        "method": predictor.method_name,
        "checkpoint": str(ckpt_path),
        "target_region": args.target_region,
        "K": args.K,
        "seed": args.seed,
        "split_type": args.split_type,
        "n_samples_evaluated": n_samples,
        "n_metric_rows": len(rows),
        "protocol_freeze_id": PROTOCOL_FREEZE_ID,
        "surface": {
            "skill_mean": float(skill_rows[skill_rows["variable"] == "surface"]["value"].mean()),
            "skill_std": float(skill_rows[skill_rows["variable"] == "surface"]["value"].std()),
            "rmse_mean": float(inc_rmse_rows[inc_rmse_rows["variable"] == "surface"]["value"].mean()),
            "corr_mean": float(inc_corr_rows[inc_corr_rows["variable"] == "surface"]["value"].mean()),
        },
        "rootzone": {
            "skill_mean": float(skill_rows[skill_rows["variable"] == "rootzone"]["value"].mean()),
            "skill_std": float(skill_rows[skill_rows["variable"] == "rootzone"]["value"].std()),
            "rmse_mean": float(inc_rmse_rows[inc_rmse_rows["variable"] == "rootzone"]["value"].mean()),
            "corr_mean": float(inc_corr_rows[inc_corr_rows["variable"] == "rootzone"]["value"].mean()),
        },
        "eval_time_s": elapsed,
    }

    summary_path = region_output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved to {summary_path}")
    print(f"\n  Surface  skill={summary['surface']['skill_mean']:.4f} \u00b1 {summary['surface']['skill_std']:.4f}")
    print(f"  Rootzone skill={summary['rootzone']['skill_mean']:.4f} \u00b1 {summary['rootzone']['skill_std']:.4f}")
    print(f"  Surface  inc_rmse={summary['surface']['rmse_mean']:.4f}")
    print(f"  Rootzone inc_rmse={summary['rootzone']['rmse_mean']:.4f}")
    print(f"  Surface  inc_corr={summary['surface']['corr_mean']:.4f}")
    print(f"  Rootzone inc_corr={summary['rootzone']['corr_mean']:.4f}")

    # Diagnostics
    diagnostics = {
        "checkpoint": str(ckpt_path),
        "target_region": args.target_region,
        "split_type": args.split_type,
        "predictor_type": args.predictor_type,
        "n_samples_total": total_samples,
        "n_samples_evaluated": n_samples,
        "n_metric_rows": len(rows),
        "metrics_computed": sorted(df["metric"].unique().tolist()),
        "variables": sorted(df["variable"].unique().tolist()),
        "seasonal_breakdown": sorted(df["season"].unique().tolist()) if "season" in df.columns else [],
    }
    diag_path = region_output_dir / "diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"{phase_label} Evaluation Complete")
    print(f"  Output: {region_output_dir}/")
    print(f"  metrics_long.csv | metrics_by_region.csv | metrics_by_season.csv")
    print(f"  summary.json | diagnostics.json")


if __name__ == "__main__":
    main()
