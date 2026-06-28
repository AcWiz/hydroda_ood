#!/usr/bin/env python3
"""Build source-side Stage 3 SAFE calibration rows.

This orchestrator runs existing HyperDA few-shot adaptation and evaluation
entrypoints on source-side pseudo-target episodes only:

* adaptation uses pseudo-region target_context/target_support from the frozen
  zero/few-shot manifest;
* query evaluation uses ``--split_type source_val`` plus
  ``--active_region_override`` for the pseudo-region;
* prediction records are persisted for offline rho expansion;
* target_val and target_eval are not read by this script.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from hydroda.data.file_hash import compute_sha256
from scripts.eval import calibrate_source_safe_guard as calib


DEFAULT_SOURCE_CHECKPOINT = (
    "artifacts/runs/phase4_hyperda_staged/US-R1/"
    "phase4_hyperda_staged_US-R1_s0_20260617_102149/"
    "checkpoints/checkpoint_best_source_val_transfer_safe_score.pt"
)
DEFAULT_SPLITS_JSON = "artifacts/splits/US_loro_zero_few_shot_splits.json"
ALL_US_REGIONS = ("US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6")


def _split_csv(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        raw_items = value.replace(";", ",").split(",")
    else:
        raw_items = list(value)
    return [str(item).strip() for item in raw_items if str(item).strip()]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_source_regions(path: str | Path) -> list[str]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        return []
    try:
        import torch

        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception:
        return []
    config = dict(checkpoint.get("config", {}) or {})
    if config.get("source_regions"):
        return [str(region) for region in config["source_regions"]]
    if config.get("source_region_global_indices"):
        return [f"US-R{int(idx) + 1}" for idx in config["source_region_global_indices"]]
    return []


def _load_source_heldout_checkpoint_map(value: str) -> dict[str, str]:
    if not value:
        return {}
    path = Path(value)
    payload: Any
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("--source_heldout_checkpoint_map must be a JSON object or path to one")
    out: dict[str, str] = {}
    for region, entry in payload.items():
        if isinstance(entry, Mapping):
            checkpoint = entry.get("checkpoint") or entry.get("source_checkpoint") or entry.get("path")
        else:
            checkpoint = entry
        if checkpoint:
            out[str(region)] = str(checkpoint)
    return out


def _adaptation_setting_for_k(k_value: int) -> str:
    if int(k_value) == 0:
        return "zero_shot_context"
    return f"few_shot_k{int(k_value)}"


def _candidate_policy(candidate: Mapping[str, Any], *, pseudo_target_region: str, seed: int) -> dict[str, Any]:
    k_value = int(candidate.get("K", 0))
    setting = _adaptation_setting_for_k(k_value)
    adapt_scope = str(candidate.get("adapt_scope", "coeff_only"))
    entry = {
        "adapt_scope": adapt_scope,
        "adapt_solver": candidate.get("adapt_solver", "adamw"),
        "lr": float(candidate.get("lr") or 0.0),
        "adaptation_steps": int(candidate.get("adaptation_steps") or 0),
        "anchor_alpha": float(candidate.get("anchor_alpha") or 0.0),
        "rho_policy": candidate.get("rho_policy", "fixed_1.0"),
        "adapt_mix_rho": 1.0,
        "support_loss_reduction": candidate.get("support_loss_reduction", "cycle_balanced"),
        "trust_region_mode": "none",
        "trust_total_radius": 0.0,
        "trust_prompt_radius": 0.0,
        "trust_gain_radius": 0.0,
        "trust_coeff_radius": 0.0,
        "trust_spatial_radius": 0.0,
        "freeze_monthly_gain": adapt_scope != "coeff_gain",
        "schedule_label": candidate.get("schedule_label", ""),
        "source_calibrated_candidate_id": candidate.get("candidate_id", ""),
        "source_calibrated_guard_config_hash": candidate.get(
            "candidate_config_hash",
            calib.candidate_config_hash(candidate),
        ),
    }
    return {
        "schema_version": "hyperda_safe_policy_v1",
        "policy_source": "source_side_episode_calibration",
        "target_val_usage": "unused_in_main_protocol",
        "target_eval_usage": "final_eval_only_no_selection",
        "source_episode_regions": [pseudo_target_region],
        "source_calibration": {
            "temporary_candidate_policy": True,
            "selection_query_role": "source_val_pseudo_query_only",
            "selection_label_usage": "source_pseudo_target_support_labels_only",
        },
        "final_target_region": pseudo_target_region,
        "seed": int(seed),
        "policies": {setting: entry},
    }


def _candidate_configs(candidate_set: str) -> list[dict[str, Any]]:
    k0 = dict(calib.baseline_gpu_row_configs()[0])
    return [k0, *calib.enumerate_guard_base_configs(candidate_set)]


def _run_command(command: Sequence[str], *, env: Mapping[str, str], log_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        if dry_run:
            log.write("[dry_run] command not executed\n")
            return
        subprocess.run(command, check=True, env=dict(env), stdout=log, stderr=subprocess.STDOUT)


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _flatten_drift(prefix: str, drift: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_total": drift.get("total"),
        f"{prefix}_target_prompt": drift.get("target_prompt"),
        f"{prefix}_monthly_gain": drift.get("monthly_gain", drift.get("monthly_residual_gain")),
        f"{prefix}_adapter_coeff_bottleneck": drift.get("adapter_coeff_bottleneck"),
        f"{prefix}_adapter_coeff_dec2": drift.get("adapter_coeff_dec2"),
        f"{prefix}_adapter_coeff_dec1": drift.get("adapter_coeff_dec1"),
        f"{prefix}_spatial_refine": drift.get("spatial_refine"),
    }


def _build_row(
    *,
    pseudo_target_region: str,
    seed: int,
    candidate: Mapping[str, Any],
    source_checkpoint: str,
    checkpoint_source_regions: Sequence[str],
    source_query_max_samples: int,
    evidence_level: str,
    adapt_metadata_path: Path,
    eval_summary_path: Path,
    prediction_record_path: Path,
) -> dict[str, Any]:
    metadata = _read_json(adapt_metadata_path) if adapt_metadata_path.exists() else {}
    summary = _read_json(eval_summary_path) if eval_summary_path.exists() else {}
    k_value = int(candidate.get("K", 0))
    row = {
        "episode_id": f"{pseudo_target_region}_s{int(seed)}",
        "pseudo_target_region": pseudo_target_region,
        "target_region": pseudo_target_region,
        "query_role": "source_val_pseudo_query",
        "split_type": "source_val",
        "adaptation_setting": _adaptation_setting_for_k(k_value),
        "K": k_value,
        "seed": int(seed),
        "candidate_id": candidate.get("candidate_id", ""),
        "base_config_id": candidate.get("base_config_id", ""),
        "candidate_config_hash": candidate.get("candidate_config_hash", calib.candidate_config_hash(candidate)),
        "adapt_scope": candidate.get("adapt_scope", metadata.get("adapt_scope", "")),
        "adapt_solver": candidate.get("adapt_solver", metadata.get("adapt_solver", "adamw")),
        "schedule_label": candidate.get("schedule_label", metadata.get("schedule_label", "")),
        "support_loss_reduction": candidate.get(
            "support_loss_reduction",
            metadata.get("support_loss_reduction", "global_pixel"),
        ),
        "rho_policy": candidate.get("rho_policy", metadata.get("rho_policy", "fixed_1.0")),
        "trust_policy": candidate.get("trust_policy", "none"),
        "lr": candidate.get("lr", metadata.get("lr")),
        "adaptation_steps": candidate.get("adaptation_steps", metadata.get("adaptation_steps")),
        "anchor_alpha": candidate.get("anchor_alpha", metadata.get("anchor_alpha")),
        "adapt_mix_rho": 1.0,
        "surface_skill_primary": _nested(summary, "surface", "skill_primary"),
        "rootzone_skill_primary": _nested(summary, "rootzone", "skill_primary"),
        "surface_rmse_latw": _nested(summary, "surface", "rmse_latw_mean"),
        "rootzone_rmse_latw": _nested(summary, "rootzone", "rmse_latw_mean"),
        "prediction_record_path": str(prediction_record_path),
        "prediction_content_hash": summary.get("prediction_content_hash", ""),
        "prediction_record_count": summary.get("prediction_record_count", 0),
        "metric_content_hash": summary.get("metric_content_hash", ""),
        "metric_values_content_hash": summary.get("metric_values_content_hash", ""),
        "source_checkpoint": source_checkpoint,
        "source_checkpoint_sha256": metadata.get("source_checkpoint_sha256")
        or (compute_sha256(source_checkpoint) if Path(source_checkpoint).exists() else ""),
        "checkpoint_source_regions": json.dumps(list(checkpoint_source_regions), sort_keys=True),
        "split_manifest_sha256": summary.get("split_manifest_sha256", metadata.get("split_manifest_sha256", "")),
        "target_context_dates_hash": summary.get("target_context_dates_hash", metadata.get("target_context_dates_hash", "")),
        "target_support_dates_hash": summary.get("target_support_dates_hash", metadata.get("target_support_dates_hash", "")),
        "target_eval_dates_hash": "not_used_source_val_query",
        "target_val_usage": "unused_in_main_protocol",
        "target_eval_usage": "final_eval_only_no_selection",
        "calib_max_query_samples": int(source_query_max_samples),
        "source_query_max_samples": int(source_query_max_samples),
        "source_safety_evidence_level": evidence_level,
        "paper_facing_run": False,
        "policy_source": metadata.get("policy_source", "source_side_episode_calibration" if k_value > 0 else "not_applicable_k0"),
        "stage3_posterior_decision": metadata.get("stage3_posterior_decision", ""),
        "support_gate_status": metadata.get("support_gate_status", ""),
        "support_gradient_cosine_mean": metadata.get("support_gradient_cosine_mean"),
        "support_gradient_cosine_min": metadata.get("support_gradient_cosine_min"),
        "support_gradient_negative_fraction": metadata.get("support_gradient_negative_fraction"),
        "support_cycle_loss_improvement_mean": metadata.get("support_cycle_loss_improvement_mean"),
        "support_cycle_loss_improvement_std": metadata.get("support_cycle_loss_improvement_std"),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    row.update(_flatten_drift("target_parameter_l2_drift_post_anchor", metadata.get("target_parameter_l2_drift_post_anchor", {}) or {}))
    row.update(_flatten_drift("target_parameter_l2_drift_pre_anchor", metadata.get("target_parameter_l2_drift_pre_anchor", {}) or {}))
    return row


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run source-side Stage 3 SAFE policy calibration rows.")
    parser.add_argument("--source_checkpoint", default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--final_target_region", default="US-R1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate_set", default="stage3_conservative_v1", choices=["stage3_conservative_v1"])
    parser.add_argument("--pseudo_target_regions", default="")
    parser.add_argument("--source_query_max_samples", type=int, default=0)
    parser.add_argument("--target_context_max_samples", type=int, default=0)
    parser.add_argument("--evidence_level", default="weaker", choices=["weaker", "strict"])
    parser.add_argument("--source_heldout_checkpoint_map", default="")
    parser.add_argument("--allow_in_checkpoint_source_episodes", action="store_true")
    parser.add_argument("--output_dir", default="artifacts/runs/stage3_source_safe_policy_calibration/source_rows")
    parser.add_argument("--splits_json", default=DEFAULT_SPLITS_JSON)
    parser.add_argument("--cuda_device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--adapt_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--weight_decay", default="1e-4")
    parser.add_argument("--grad_clip", default="1.0")
    parser.add_argument("--lambda_prior", default="1e-3")
    parser.add_argument("--lambda_latent", default="1e-3")
    parser.add_argument("--lambda_gain", default="1e-2")
    parser.add_argument("--lambda_gain_smooth", default="1e-3")
    parser.add_argument("--lambda_analysis", default="0.25")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if int(args.source_query_max_samples) < 0:
        raise SystemExit("--source_query_max_samples must be >=0, where 0 means full source_val")
    if int(args.target_context_max_samples) < 0:
        raise SystemExit("--target_context_max_samples must be >=0, where 0 means full target_context")
    if args.evidence_level == "strict" and args.allow_in_checkpoint_source_episodes:
        raise SystemExit("strict evidence cannot use --allow_in_checkpoint_source_episodes")

    heldout_map = _load_source_heldout_checkpoint_map(args.source_heldout_checkpoint_map)
    if args.evidence_level == "strict" and not heldout_map:
        raise SystemExit("EVIDENCE_LEVEL=strict requires --source_heldout_checkpoint_map")

    default_source_regions = _checkpoint_source_regions(args.source_checkpoint)
    pseudo_regions = _split_csv(args.pseudo_target_regions)
    if not pseudo_regions:
        pseudo_regions = [region for region in (default_source_regions or ALL_US_REGIONS) if region != args.final_target_region]
    if not pseudo_regions:
        raise SystemExit("No pseudo target regions resolved")

    commands_path = output_dir / "commands.jsonl"
    aggregate_rows: list[dict[str, Any]] = []
    plan_rows = []
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    for pseudo_region in pseudo_regions:
        if args.evidence_level == "strict":
            source_checkpoint = heldout_map.get(pseudo_region, "")
            if not source_checkpoint:
                raise SystemExit(f"Strict source-heldout mapping missing checkpoint for {pseudo_region}")
        else:
            source_checkpoint = args.source_checkpoint
        checkpoint_source_regions = _checkpoint_source_regions(source_checkpoint)
        validation = calib.validate_checkpoint_source_regions(
            checkpoint_source_regions,
            final_target_region=args.final_target_region,
            pseudo_target_region=pseudo_region,
            allow_in_checkpoint_source_episodes=bool(args.allow_in_checkpoint_source_episodes),
        )

        for candidate in _candidate_configs(args.candidate_set):
            candidate = dict(candidate)
            k_value = int(candidate.get("K", 0))
            setting = _adaptation_setting_for_k(k_value)
            candidate_id = str(candidate.get("candidate_id", ""))
            row_dir = output_dir / pseudo_region / candidate_id
            adapt_dir = row_dir / "adapt"
            eval_dir = row_dir / "eval"
            prediction_record_path = row_dir / "prediction_records.jsonl"
            row_csv = row_dir / "source_safe_candidate_rows.csv"
            if args.skip_existing and row_csv.exists():
                loaded = calib.load_calibration_rows([row_csv])
                aggregate_rows.extend(loaded)
                continue

            policy_args: list[str] = []
            if k_value > 0:
                policy_path = row_dir / "candidate_safe_policy.json"
                _write_json(policy_path, _candidate_policy(candidate, pseudo_target_region=pseudo_region, seed=args.seed))
                policy_args = ["--safe_policy_json", str(policy_path), "--require_safe_policy_json_for_kshot"]

            adapt_scope = str(candidate.get("adapt_scope", "none" if k_value == 0 else "coeff_only"))
            adapt_steps = int(candidate.get("adaptation_steps") or 0)
            anchor_alpha = float(candidate.get("anchor_alpha") or 0.0)
            lr = float(candidate.get("lr") or 1e-3)
            adapt_command = [
                sys.executable,
                "scripts/train/train_hyperda_few_shot_adapt.py",
                "--source_checkpoint",
                source_checkpoint,
                "--target_region",
                pseudo_region,
                "--K",
                str(k_value),
                "--adaptation_setting",
                setting,
                "--seed",
                str(args.seed),
                "--device",
                "cuda",
                "--splits_json",
                args.splits_json,
                "--adaptation_steps",
                str(adapt_steps),
                "--schedule_label",
                str(candidate.get("schedule_label", "")),
                "--adapt_recipe",
                "source_anchor",
                "--anchor_alpha",
                str(anchor_alpha),
                "--adapt_scope",
                adapt_scope,
                "--stage3_posterior_policy",
                "conservative_coeff_posterior",
                "--support_gate",
                "auto",
                "--adapt_solver",
                "adamw",
                "--support_loss_reduction",
                str(candidate.get("support_loss_reduction", "global_pixel")),
                "--batch_size",
                str(args.adapt_batch_size),
                "--lr",
                str(lr),
                "--weight_decay",
                str(args.weight_decay),
                "--grad_clip",
                str(args.grad_clip),
                "--lambda_prior",
                str(args.lambda_prior),
                "--lambda_latent",
                str(args.lambda_latent),
                "--lambda_gain",
                str(args.lambda_gain),
                "--lambda_gain_smooth",
                str(args.lambda_gain_smooth),
                "--lambda_analysis",
                str(args.lambda_analysis),
                "--num_workers",
                "0",
                "--use_lat_weighted_loss",
                "--target_context_max_samples",
                str(args.target_context_max_samples),
                "--output_dir",
                str(adapt_dir),
                *policy_args,
            ]
            if adapt_scope in {"none", "coeff_only"}:
                adapt_command.append("--freeze_monthly_gain")
            eval_checkpoint = adapt_dir / "checkpoints" / "checkpoint_final_preregistered.pt"
            eval_summary_path = eval_dir / pseudo_region / "summary.json"
            eval_command = [
                sys.executable,
                "scripts/eval/evaluate_checkpoint.py",
                "--checkpoint",
                str(eval_checkpoint),
                "--target_region",
                pseudo_region,
                "--adaptation_setting",
                setting,
                "--K",
                str(k_value),
                "--seed",
                str(args.seed),
                "--split_type",
                "source_val",
                "--active_region_override",
                pseudo_region,
                "--splits_json",
                args.splits_json,
                "--predictor_type",
                "hyperda_target_adapt",
                "--device",
                "cuda",
                "--output_dir",
                str(eval_dir),
                "--max_samples",
                str(args.source_query_max_samples),
                "--batch_size",
                str(args.eval_batch_size),
                "--output_level",
                "full",
                "--adapt_mix_rho",
                "1.0",
                "--prediction_record_path",
                str(prediction_record_path),
            ]
            plan_rows.append(
                {
                    "pseudo_target_region": pseudo_region,
                    "candidate_id": candidate_id,
                    "K": k_value,
                    "source_checkpoint": source_checkpoint,
                    "adapt_command": adapt_command,
                    "eval_command": eval_command,
                    "validation": validation,
                }
            )
            with commands_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(plan_rows[-1], sort_keys=True, default=str) + "\n")
            _run_command(adapt_command, env=env, log_path=row_dir / "adapt.log", dry_run=bool(args.dry_run))
            _run_command(eval_command, env=env, log_path=row_dir / "eval.log", dry_run=bool(args.dry_run))
            if args.dry_run:
                continue
            row = _build_row(
                pseudo_target_region=pseudo_region,
                seed=args.seed,
                candidate=candidate,
                source_checkpoint=source_checkpoint,
                checkpoint_source_regions=checkpoint_source_regions,
                source_query_max_samples=int(args.source_query_max_samples),
                evidence_level=validation.get("source_safety_evidence_level", args.evidence_level),
                adapt_metadata_path=adapt_dir / "metadata.json",
                eval_summary_path=eval_summary_path,
                prediction_record_path=prediction_record_path,
            )
            _write_csv(row_csv, [row])
            _write_json(row_dir / "source_safe_candidate_rows.json", [row])
            aggregate_rows.append(row)

    _write_json(output_dir / "calibration_plan.json", plan_rows)
    if aggregate_rows:
        _write_csv(output_dir / "calibration_rows.csv", aggregate_rows)
        _write_json(output_dir / "calibration_rows.json", aggregate_rows)
    else:
        _write_json(output_dir / "calibration_rows.json", [])


if __name__ == "__main__":
    main()
