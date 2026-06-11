#!/usr/bin/env python3
"""Standalone per-region evaluation for a legacy all-regions checkpoint.

Evaluates each US region (R1-R6) independently: for each region, creates a dataset
where _active_region_mask is restricted to that region's pixels only, then runs
evaluate_split. Results are saved both per-region and aggregated.

Usage:
    # Evaluate on 2023-2025 target_eval (default):
    PYTHONPATH=. python scripts/eval/eval_source_only_all_regions.py \
        --checkpoint .../checkpoint_best_source_val_safe_score.pt \
        --split_type target_eval --adaptation_setting zero_shot_context --K 0 --seed 0 --device cuda

    # Evaluate on 2022 source_val:
    PYTHONPATH=. python scripts/eval/eval_source_only_all_regions.py \
        --checkpoint .../checkpoint_best_source_val_safe_score.pt \
        --split_type source_val --K 0 --seed 0 --device cuda

This only runs evaluation — no training.
"""
import argparse
from pathlib import Path

import pandas as pd

from hydroda.baselines.source_only import SourceOnlyBackbonePredictor
from hydroda.data.dataset import HydroDADataset, _ALL_US_REGIONS
from hydroda.evaluation.harness import evaluate_split, build_per_region_summary, KEY_METRICS
from hydroda.utils.device import resolve_device

DA_NC = "/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc"
REGION_MASKS_NC = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_zero_few_shot_splits.json"
FREEZE_MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"
PROTOCOL_FREEZE_ID = "hyperda_v4_4_zero_few_shot_generalization_2015_2025_context2015_2021_sourceval2022_eval2023_2025"


def _evaluate_one_region(
    predictor: SourceOnlyBackbonePredictor,
    region_id: str,
    split_type: str,
    K: int | None,
    seed: int,
    adaptation_setting: str,
) -> list:
    """Evaluate predictor on a single region's pixels.

    Creates a HydroDADataset with active region mask restricted to region_id,
    and runs evaluate_split.
    """
    print(f"    Loading dataset for {region_id}...")
    dataset = HydroDADataset(
        da_nc_path=DA_NC,
        region_masks_nc=REGION_MASKS_NC,
        splits_json=SPLITS_JSON,
        target_region=region_id,
        split_type=split_type,
        K=K,
        seed=seed,
        adaptation_setting=adaptation_setting,
        freeze_manifest=FREEZE_MANIFEST,
    )
    dataset.set_active_region(region_id)

    dates = [d.get("date_str", "") for d in dataset._date_records]
    years = sorted(set(d[:4] for d in dates if len(d) >= 4))
    print(f"      samples={len(dataset)}  years={years}")

    rows = evaluate_split(
        dataset=dataset,
        predictor=predictor,
        split_role=split_type,
        experiment_id="per_region_eval",
        protocol_freeze_id=PROTOCOL_FREEZE_ID,
        method="legacy_all_regions_sanity",
        split_file=SPLITS_JSON,
        mask_file=REGION_MASKS_NC,
    )

    # Override sample_region_id in all rows to the evaluated region
    for row in rows:
        row["sample_region_id"] = region_id

    dataset.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Per-region evaluation for source-only all-regions checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
        help="Path to checkpoint.pt")
    parser.add_argument("--split_type", type=str, default="target_eval",
        choices=["source_val", "source_test", "target_eval", "target_train"],
        help="Split to evaluate on. Default: target_eval (2023-2025).")
    parser.add_argument("--adaptation_setting", type=str, default="zero_shot_context",
        help="Split adaptation setting (default: zero_shot_context; main examples: zero_shot_context, few_shot_k4, few_shot_k12)")
    parser.add_argument("--K", type=int, default=0,
        help="Zero/few-shot K value for the main protocol.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None,
        help="Output directory for results (default: derives from checkpoint path)")
    parser.add_argument("--regions", type=str, nargs="*", default=None,
        help="Specific regions to evaluate (default: all US-R1..US-R6)")
    args = parser.parse_args()

    if args.adaptation_setting == "target_full_train":
        args.K = None
    elif args.K is None:
        args.K = 0

    device = resolve_device(args.device, require_gpu=False)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = checkpoint_path.parent.parent / "results" / checkpoint_path.stem
    results_dir = run_dir / args.split_type
    results_dir.mkdir(parents=True, exist_ok=True)

    regions = args.regions if args.regions else list(_ALL_US_REGIONS)

    print("=" * 60)
    print("Legacy All-Regions Sanity Per-Region Evaluation")
    print("  method:      legacy_all_regions_sanity")
    print("  status:      legacy_sanity_not_paper_facing_ood_global")
    print(f"  checkpoint:  {checkpoint_path}")
    print(f"  split_type:  {args.split_type}")
    print(f"  results:     {results_dir}")
    print(f"  regions:     {regions}")
    print(f"  adaptation_setting={args.adaptation_setting}  K={args.K}  seed={args.seed}  device={device}")
    print("=" * 60)

    # Load predictor once, reuse for all regions
    predictor = SourceOnlyBackbonePredictor(
        checkpoint_path=str(checkpoint_path), device=device, apply_residual_gain=True,
    )
    print(f"  alpha_s={predictor.alpha_surface:.3f}  alpha_r={predictor.alpha_rootzone:.3f}")

    # Evaluate each region independently
    all_rows = []

    for i, region_id in enumerate(regions):
        print(f"\n  [{i+1}/{len(regions)}] Evaluating {region_id}...")
        rows = _evaluate_one_region(
            predictor=predictor,
            region_id=region_id,
            split_type=args.split_type,
            K=args.K,
            seed=args.seed,
            adaptation_setting=args.adaptation_setting,
        )
        all_rows.extend(rows)
        print(f"    {region_id}: {len(rows)} metric rows")

    # ---- Save combined outputs ----
    df_all = pd.DataFrame(all_rows)
    per_region_summary = build_per_region_summary(df_all, results_dir)
    print(f"\n  metrics_long.csv: {len(all_rows)} rows -> {results_dir / 'metrics_long.csv'}")
    print(f"  per_region_summary.json -> {results_dir / 'per_region_summary.json'}")

    # Print summary table
    print(f"\n{'─' * 80}")
    print(f"Per-Region Analysis Skill (latw) — {args.split_type}")
    print(f"{'Region':<8} {'surface_skill':>14} {'rootzone_skill':>14}")
    print(f"{'─' * 80}")
    for region_id in sorted(per_region_summary.keys()):
        s = per_region_summary[region_id].get("surface", {}).get("analysis_skill_vs_forecast_latw", {})
        r = per_region_summary[region_id].get("rootzone", {}).get("analysis_skill_vs_forecast_latw", {})
        s_val = s.get("mean") if isinstance(s, dict) else s
        r_val = r.get("mean") if isinstance(r, dict) else r
        s_str = f"{s_val:.10f}" if s_val is not None else "N/A"
        r_str = f"{r_val:.10f}" if r_val is not None else "N/A"
        print(f"{region_id:<8} {s_str:>14} {r_str:>14}")
    print(f"{'─' * 80}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
