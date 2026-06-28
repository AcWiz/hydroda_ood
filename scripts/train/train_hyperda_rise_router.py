#!/usr/bin/env python3
"""Train the HyperDA-RISE source-side router prior.

The router prior is source-only: each source region is treated as a
pseudo-target, descriptors come from 2015-2021 source/input-side context, and
the supervision signal is candidate-expert WRMSE on source_val=2022.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd

from hydroda.data.dataset import HydroDADataset
from hydroda.data.file_hash import compute_sha256
from hydroda.data.protocol import ProtocolConfig
from hydroda.baselines.forecast import ForecastBaseline
from hydroda.evaluation.der_router import create_predictor as _create_checkpoint_predictor
from hydroda.evaluation.harness import evaluate_split
from hydroda.evaluation.rise_router import (
    build_context_descriptor,
    build_router_prior,
    candidate_metrics_to_episodes,
    write_json,
)


DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
REGION_MASKS_NC = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_zero_few_shot_splits.json"
FREEZE_MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"
PROTOCOL = ProtocolConfig()
ALL_US_REGIONS = ("US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6")


def create_predictor(
    *,
    checkpoint: str,
    predictor_type: str,
    device: str = "cpu",
    target_region: str | None = None,
) -> Any:
    """Create a frozen candidate expert for source_val scoring."""
    predictor_type = str(predictor_type)
    if predictor_type == "forecast_only":
        return ForecastBaseline()
    return _create_checkpoint_predictor(
        checkpoint=checkpoint,
        predictor_type=predictor_type,
        device=device,
        target_region=target_region,
    )


def _make_dataset(
    *,
    target_region: str,
    split_type: str,
    seed: int,
    da_nc_path: str,
    region_masks_nc: str,
    splits_json: str,
    freeze_manifest: str,
    active_region_id: str | None = None,
) -> Any:
    dataset = HydroDADataset(
        da_nc_path=da_nc_path,
        region_masks_nc=region_masks_nc,
        splits_json=splits_json,
        target_region=target_region,
        split_type=split_type,
        K=0,
        seed=seed,
        adaptation_setting="zero_shot_context",
        freeze_manifest=freeze_manifest,
    )
    if active_region_id is not None and hasattr(dataset, "set_active_region"):
        dataset.set_active_region(active_region_id)
    return dataset


def _dataset_date_hash(dataset: Any, key: str) -> str:
    return str(getattr(dataset, "_split_entry", {}).get(key, ""))


def _context_samples(dataset: Any, *, max_samples: int | None) -> Iterable[Mapping[str, Any]]:
    n_samples = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for idx in range(n_samples):
        yield dataset.get_input_side_sample(idx)


def _source_regions_for_target(target_region: str, source_regions: Sequence[str] | None = None) -> list[str]:
    if source_regions:
        return [str(region) for region in source_regions]
    return [region for region in ALL_US_REGIONS if region != target_region]


def _write_candidate_metrics_copy(source: str | Path, output_dir: Path) -> Path:
    out = output_dir / "candidate_metrics_source_val.csv"
    if Path(source).resolve() != out.resolve():
        shutil.copyfile(source, out)
    return out


def _evaluate_candidate_metrics(
    *,
    candidates: Sequence[Mapping[str, Any]],
    source_regions: Sequence[str],
    target_region: str,
    seed: int,
    output_dir: Path,
    da_nc_path: str,
    region_masks_nc: str,
    splits_json: str,
    freeze_manifest: str,
    max_eval_samples: int | None,
    device: str,
) -> Path:
    """Evaluate frozen candidate experts on source_val pseudo-target regions."""
    split_manifest_sha256 = compute_sha256(splits_json) if Path(splits_json).exists() else ""
    rows: list[Dict[str, Any]] = []
    for region_idx, pseudo_region in enumerate(source_regions, start=1):
        print(
            f"[RISE router] source_val candidate scoring "
            f"region {region_idx}/{len(source_regions)}: {pseudo_region}",
            flush=True,
        )
        dataset = _make_dataset(
            target_region=target_region,
            split_type="source_val",
            seed=seed,
            da_nc_path=da_nc_path,
            region_masks_nc=region_masks_nc,
            splits_json=splits_json,
            freeze_manifest=freeze_manifest,
            active_region_id=pseudo_region,
        )
        try:
            for candidate_idx, candidate in enumerate(candidates, start=1):
                expert_id = str(candidate["expert_id"])
                predictor_type = str(candidate.get("predictor_type", "forecast_only"))
                checkpoint = str(candidate.get("checkpoint", ""))
                print(
                    f"[RISE router]   candidate {candidate_idx}/{len(candidates)}: "
                    f"{expert_id} ({predictor_type})",
                    flush=True,
                )
                predictor = create_predictor(
                    checkpoint=checkpoint,
                    predictor_type=predictor_type,
                    device=device,
                    target_region=target_region,
                )
                metric_rows = evaluate_split(
                    dataset=dataset,
                    predictor=predictor,
                    split_role="source_val",
                    experiment_id=f"hyperda_rise_router_{pseudo_region}_{expert_id}_S{seed}",
                    protocol_freeze_id=PROTOCOL.protocol_freeze_id,
                    method=getattr(predictor, "method_name", predictor_type),
                    split_file=splits_json,
                    mask_file=region_masks_nc,
                    split_manifest_sha256=split_manifest_sha256,
                    preloaded=False,
                    max_samples=max_eval_samples,
                )
                for row in metric_rows:
                    out = dict(row)
                    out["pseudo_target_region_id"] = pseudo_region
                    out["candidate_id"] = expert_id
                    rows.append(out)
        finally:
            if hasattr(dataset, "close"):
                dataset.close()
    out_path = output_dir / "candidate_metrics_source_val.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def run_train(
    *,
    candidates: Iterable[Mapping[str, Any]],
    target_region: str,
    seed: int,
    output_dir: str | Path,
    candidate_metrics_source_val: str | Path | None,
    source_regions: Sequence[str] | None,
    temperature: float,
    da_nc_path: str,
    region_masks_nc: str,
    splits_json: str,
    freeze_manifest: str,
    max_context_samples: int | None = None,
    max_eval_samples: int | None = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Build and write a RISE source-side router prior."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_list = [dict(candidate) for candidate in candidates]
    expert_ids = [str(candidate["expert_id"]) for candidate in candidate_list]
    source_regions_resolved = _source_regions_for_target(target_region, source_regions)
    if not source_regions_resolved:
        raise ValueError("RISE router prior requires at least one non-target source region")

    descriptor_by_region: dict[str, list[float]] = {}
    descriptor_metadata: dict[str, Any] = {}
    context_date_hashes: dict[str, str] = {}
    print(
        f"[RISE router] building descriptors for {len(source_regions_resolved)} source regions "
        f"(max_context_samples={max_context_samples if max_context_samples is not None else 'all'})",
        flush=True,
    )
    for region_idx, region_id in enumerate(source_regions_resolved, start=1):
        print(f"[RISE router] descriptor {region_idx}/{len(source_regions_resolved)}: {region_id}", flush=True)
        dataset = _make_dataset(
            target_region=target_region,
            split_type="source_fit",
            seed=seed,
            da_nc_path=da_nc_path,
            region_masks_nc=region_masks_nc,
            splits_json=splits_json,
            freeze_manifest=freeze_manifest,
            active_region_id=region_id,
        )
        try:
            descriptor = build_context_descriptor(_context_samples(dataset, max_samples=max_context_samples))
            descriptor_by_region[region_id] = descriptor.vector.astype(float).tolist()
            descriptor_metadata[region_id] = descriptor.metadata
            context_date_hashes[region_id] = _dataset_date_hash(dataset, "target_context_dates_hash")
        finally:
            if hasattr(dataset, "close"):
                dataset.close()

    if candidate_metrics_source_val is None:
        print(
            f"[RISE router] no candidate metrics CSV supplied; generating source_val metrics "
            f"(max_eval_samples={max_eval_samples if max_eval_samples is not None else 'all'})",
            flush=True,
        )
        metrics_source_path = _evaluate_candidate_metrics(
            candidates=candidate_list,
            source_regions=source_regions_resolved,
            target_region=target_region,
            seed=seed,
            output_dir=output_dir,
            da_nc_path=da_nc_path,
            region_masks_nc=region_masks_nc,
            splits_json=splits_json,
            freeze_manifest=freeze_manifest,
            max_eval_samples=max_eval_samples,
            device=device,
        )
    else:
        print(f"[RISE router] using precomputed candidate metrics: {candidate_metrics_source_val}", flush=True)
        metrics_source_path = _write_candidate_metrics_copy(candidate_metrics_source_val, output_dir)
    metrics_df = pd.read_csv(metrics_source_path)
    episodes = candidate_metrics_to_episodes(
        metrics_df,
        descriptor_by_region=descriptor_by_region,
        expert_ids=expert_ids,
        metric_name="increment_rmse_latw",
        split_role="source_val",
    )
    prior = build_router_prior(
        episodes=episodes,
        candidates=candidate_list,
        temperature=temperature,
    )
    prior.update(
        {
            "target_region": target_region,
            "seed": int(seed),
            "protocol_freeze_id": PROTOCOL.protocol_freeze_id,
            "split_manifest_path": str(splits_json),
            "split_manifest_sha256": compute_sha256(splits_json) if Path(splits_json).exists() else "",
            "source_region_ids": source_regions_resolved,
            "descriptor_metadata": descriptor_metadata,
            "context_date_hashes": context_date_hashes,
        }
    )

    prior_path = write_json(output_dir / "router_prior.json", prior)
    summary = {
        "method": "HyperDA-RISE",
        "method_id": "hyperda_rise_source_side_router_prior",
        "router_prior": str(prior_path),
        "candidate_metrics_source_val": str(metrics_source_path),
        "target_region": target_region,
        "source_region_ids": source_regions_resolved,
        "seed": int(seed),
        "training_label_source": "source_val_2022",
        "split_manifest_sha256": prior["split_manifest_sha256"],
        "no_leakage_declaration": prior["no_leakage_declaration"],
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def _parse_candidate_spec(spec: str) -> Dict[str, str]:
    parts = spec.split(":")
    if len(parts) == 1:
        return {"expert_id": parts[0]}
    if len(parts) < 3:
        raise ValueError(
            "candidate must be expert_id:predictor_type:checkpoint or expert_id "
            f"(got {spec!r})"
        )
    expert_id, predictor_type = parts[0], parts[1]
    checkpoint = ":".join(parts[2:])
    return {"expert_id": expert_id, "predictor_type": predictor_type, "checkpoint": checkpoint}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HyperDA-RISE source-side router prior")
    parser.add_argument("--target_region", default="US-R1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--candidate_metrics_source_val",
        default=None,
        help="Optional precomputed source_val candidate metrics CSV. If omitted, candidates are evaluated.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="expert_id or expert_id:predictor_type:checkpoint. Repeat for each expert.",
    )
    parser.add_argument(
        "--source_region",
        action="append",
        help="Pseudo-target source region to include. Defaults to all non-target US regions.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_context_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--da_nc", default=f"{DATA_DIR}/DA.nc")
    parser.add_argument("--region_masks_nc", default=REGION_MASKS_NC)
    parser.add_argument("--splits_json", default=SPLITS_JSON)
    parser.add_argument("--freeze_manifest", default=FREEZE_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_train(
        candidates=[_parse_candidate_spec(spec) for spec in args.candidate],
        target_region=args.target_region,
        seed=args.seed,
        output_dir=args.output_dir,
        candidate_metrics_source_val=args.candidate_metrics_source_val,
        source_regions=args.source_region,
        temperature=args.temperature,
        da_nc_path=args.da_nc,
        region_masks_nc=args.region_masks_nc,
        splits_json=args.splits_json,
        freeze_manifest=args.freeze_manifest,
        max_context_samples=args.max_context_samples if args.max_context_samples > 0 else None,
        max_eval_samples=args.max_eval_samples if args.max_eval_samples > 0 else None,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
