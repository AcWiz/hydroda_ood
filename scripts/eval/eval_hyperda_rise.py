#!/usr/bin/env python3
"""Evaluate HyperDA-RISE on target_eval for K in {0,4,12}.

RISE target evaluation is deliberately split into two phases:

1. target_context input-only descriptor routing;
2. optional K-shot low-dimensional posterior solve on target_support labels.

The final target_eval path reads labels only through the evaluation harness
after predictions are produced.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pandas as pd

from hydroda.baselines.forecast import ForecastBaseline
from hydroda.data.dataset import HydroDADataset
from hydroda.data.file_hash import compute_sha256
from hydroda.data.protocol import ProtocolConfig
from hydroda.evaluation.harness import (
    evaluate_split,
    metric_rows_content_hash,
    metric_values_content_hash,
    summarize_metric_rows,
)
from hydroda.evaluation.rise_router import (
    ExpertMixturePredictor,
    build_context_descriptor,
    build_posterior_config,
    build_support_reliability_rows,
    load_router_prior,
    posterior_config_for_eval,
    route_weights_from_prior,
    support_predictions_from_predictors,
    write_json,
)
from hydroda.evaluation.der_router import create_predictor as _create_checkpoint_predictor
from hydroda.utils.device import resolve_device


DATA_DIR = "/fastersharefiles2/fenglonghan/dataset/SMAP"
REGION_MASKS_NC = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_zero_few_shot_splits.json"
FREEZE_MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"
PROTOCOL = ProtocolConfig()


def create_predictor(
    *,
    checkpoint: str,
    predictor_type: str,
    device: str = "cuda",
    target_region: str | None = None,
) -> Any:
    """Create an expert predictor for RISE."""
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
    K: int,
    seed: int,
    adaptation_setting: str,
    da_nc_path: str,
    region_masks_nc: str,
    splits_json: str,
    freeze_manifest: str,
) -> Any:
    return HydroDADataset(
        da_nc_path=da_nc_path,
        region_masks_nc=region_masks_nc,
        splits_json=splits_json,
        target_region=target_region,
        split_type=split_type,
        K=K,
        seed=seed,
        adaptation_setting=adaptation_setting,
        freeze_manifest=freeze_manifest,
    )


def _dataset_date_hash(dataset: Any, key: str) -> str:
    return str(getattr(dataset, "_split_entry", {}).get(key, ""))


def _input_side_samples(dataset: Any, *, max_samples: int | None) -> Iterable[Mapping[str, Any]]:
    n_samples = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for idx in range(n_samples):
        yield dataset.get_input_side_sample(idx)


def _candidate_map(prior: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    candidates = prior.get("candidates", [])
    out = {}
    for candidate in candidates:
        candidate = dict(candidate)
        expert_id = str(candidate["expert_id"])
        out[expert_id] = candidate
    return out


def _build_experts(prior: Mapping[str, Any], *, device: str, target_region: str) -> Dict[str, Any]:
    experts = {}
    for expert_id, candidate in _candidate_map(prior).items():
        predictor_type = str(candidate.get("predictor_type", "forecast_only"))
        checkpoint = str(candidate.get("checkpoint", ""))
        experts[expert_id] = create_predictor(
            checkpoint=checkpoint,
            predictor_type=predictor_type,
            device=device,
            target_region=target_region,
        )
    return experts


def _write_metrics(rows: list[Dict[str, Any]], output_dir: Path) -> Dict[str, Dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "metrics_long.csv", index=False)
    summary = summarize_metric_rows(df)
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_eval(
    *,
    router_prior_path: str | Path,
    target_region: str,
    K: int,
    seed: int,
    output_dir: str | Path,
    device: str,
    max_eval_samples: int | None,
    max_context_samples: int | None,
    max_support_samples: int | None,
    ridge_lambda: float,
    temperature: float,
    da_nc_path: str,
    region_masks_nc: str,
    splits_json: str,
    freeze_manifest: str,
) -> Dict[str, Any]:
    PROTOCOL.assert_supported_K(K)
    adaptation_setting = PROTOCOL.adaptation_setting_for_K(K)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prior = load_router_prior(router_prior_path)
    split_manifest_sha256 = compute_sha256(splits_json) if Path(splits_json).exists() else ""
    device_resolved = str(resolve_device(device, require_gpu=False))

    context_dataset = _make_dataset(
        target_region=target_region,
        split_type="target_context",
        K=K,
        seed=seed,
        adaptation_setting=adaptation_setting,
        da_nc_path=da_nc_path,
        region_masks_nc=region_masks_nc,
        splits_json=splits_json,
        freeze_manifest=freeze_manifest,
    )
    try:
        descriptor = build_context_descriptor(
            _input_side_samples(context_dataset, max_samples=max_context_samples)
        )
    finally:
        if hasattr(context_dataset, "close"):
            context_dataset.close()

    routed = route_weights_from_prior(
        prior,
        descriptor=descriptor.vector,
        retrieval_temperature=temperature,
    )
    experts = _build_experts(prior, device=device_resolved, target_region=target_region)

    support_samples = []
    reliability_rows = []
    if int(K) > 0:
        support_dataset = _make_dataset(
            target_region=target_region,
            split_type="target_support",
            K=K,
            seed=seed,
            adaptation_setting=adaptation_setting,
            da_nc_path=da_nc_path,
            region_masks_nc=region_masks_nc,
            splits_json=splits_json,
            freeze_manifest=freeze_manifest,
        )
        try:
            support_samples = support_predictions_from_predictors(
                support_dataset=support_dataset,
                experts=experts,
                max_samples=max_support_samples,
            )
            reliability_rows = build_support_reliability_rows(support_samples, K=K)
        finally:
            if hasattr(support_dataset, "close"):
                support_dataset.close()

    posterior = build_posterior_config(
        K=K,
        prior_weights=routed["weights"],
        support_samples=support_samples,
        ridge_lambda=ridge_lambda,
        temperature=temperature,
        support_reliability={
            str(row["support_index"]): float(row["reliability_weight"])
            for row in reliability_rows
        },
    )
    posterior.update(
        {
            "router_prior_path": str(router_prior_path),
            "router_prior_hash": routed["source_prior_hash"],
            "target_region": target_region,
            "adaptation_setting": adaptation_setting,
            "seed": int(seed),
            "protocol_freeze_id": PROTOCOL.protocol_freeze_id,
            "split_manifest_path": str(splits_json),
            "split_manifest_sha256": split_manifest_sha256,
            "target_context_descriptor": descriptor.metadata,
        }
    )
    posterior_path = write_json(output_dir / f"posterior_config_K{K}.json", posterior)
    pd.DataFrame(reliability_rows).to_csv(output_dir / "support_reliability.csv", index=False)

    eval_config = posterior_config_for_eval(posterior=posterior)
    predictor = ExpertMixturePredictor(
        experts=experts,
        weights=eval_config["weights"],
        gain=eval_config["gain"],
        bias=eval_config["bias"],
        monthly_gain=eval_config["monthly_gain"],
        monthly_bias=eval_config["monthly_bias"],
        method_name=eval_config["method_id"],
        metadata=posterior,
    )

    eval_dataset = _make_dataset(
        target_region=target_region,
        split_type="target_eval",
        K=K,
        seed=seed,
        adaptation_setting=adaptation_setting,
        da_nc_path=da_nc_path,
        region_masks_nc=region_masks_nc,
        splits_json=splits_json,
        freeze_manifest=freeze_manifest,
    )
    region_output_dir = output_dir / target_region
    started = time.time()
    try:
        rows, hashes = evaluate_split(
            dataset=eval_dataset,
            predictor=predictor,
            split_role="target_eval",
            experiment_id=f"hyperda_rise_{target_region}_K{K}_S{seed}",
            protocol_freeze_id=PROTOCOL.protocol_freeze_id,
            method=predictor.method_name,
            split_file=splits_json,
            mask_file=region_masks_nc,
            target_context_dates_hash=_dataset_date_hash(eval_dataset, "target_context_dates_hash"),
            target_support_dates_hash=_dataset_date_hash(eval_dataset, "target_support_dates_hash"),
            support_dates_hash=_dataset_date_hash(eval_dataset, "support_dates_hash"),
            target_train_dates_hash=_dataset_date_hash(eval_dataset, "target_train_dates_hash"),
            target_eval_dates_hash=_dataset_date_hash(eval_dataset, "target_eval_dates_hash"),
            split_manifest_sha256=split_manifest_sha256,
            preloaded=False,
            max_samples=max_eval_samples,
            return_hashes=True,
        )
    finally:
        if hasattr(eval_dataset, "close"):
            eval_dataset.close()

    metric_summary = _write_metrics(rows, region_output_dir)
    summary = {
        "method": "HyperDA-RISE",
        "method_id": posterior["method_id"],
        "router_prior": str(router_prior_path),
        "posterior_config": str(posterior_path),
        "target_region": target_region,
        "adaptation_setting": adaptation_setting,
        "K": int(K),
        "seed": int(seed),
        "split_type": "target_eval",
        "n_metric_rows": len(rows),
        "protocol_freeze_id": PROTOCOL.protocol_freeze_id,
        "split_manifest_sha256": split_manifest_sha256,
        "target_context_dates_hash": descriptor.metadata.get("context_hash", ""),
        "target_support_dates_hash": posterior.get("target_support_dates_hash", ""),
        "target_eval_dates_hash": _dataset_date_hash(eval_dataset, "target_eval_dates_hash"),
        "prediction_content_hash": hashes.get("prediction_content_hash", ""),
        "metric_content_hash": metric_rows_content_hash(rows),
        "metric_values_content_hash": metric_values_content_hash(rows),
        "surface": metric_summary.get("surface", {}),
        "rootzone": metric_summary.get("rootzone", {}),
        "no_leakage_declaration": posterior["no_leakage_declaration"],
        "eval_time_s": time.time() - started,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(region_output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HyperDA-RISE on target_eval")
    parser.add_argument("--router_prior", required=True)
    parser.add_argument("--target_region", default="US-R1")
    parser.add_argument("--K", type=int, required=True, choices=list(PROTOCOL.main_K_values))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--max_context_samples", type=int, default=0)
    parser.add_argument("--max_support_samples", type=int, default=0)
    parser.add_argument("--ridge_lambda", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--da_nc", default=f"{DATA_DIR}/DA.nc")
    parser.add_argument("--region_masks_nc", default=REGION_MASKS_NC)
    parser.add_argument("--splits_json", default=SPLITS_JSON)
    parser.add_argument("--freeze_manifest", default=FREEZE_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_eval(
        router_prior_path=args.router_prior,
        target_region=args.target_region,
        K=args.K,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
        max_eval_samples=args.max_eval_samples if args.max_eval_samples > 0 else None,
        max_context_samples=args.max_context_samples if args.max_context_samples > 0 else None,
        max_support_samples=args.max_support_samples if args.max_support_samples > 0 else None,
        ridge_lambda=args.ridge_lambda,
        temperature=args.temperature,
        da_nc_path=args.da_nc,
        region_masks_nc=args.region_masks_nc,
        splits_json=args.splits_json,
        freeze_manifest=args.freeze_manifest,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
