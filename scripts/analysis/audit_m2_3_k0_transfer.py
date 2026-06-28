#!/usr/bin/env python3
"""Audit M2.3 K=0 target-transfer evidence against M2.1.

The script is intentionally artifact-driven: it hashes the checkpoint and split
files passed on the command line, extracts target_eval RMSE from saved summary
JSON files, and reports whether the two summaries share the same protocol,
target_context hash, target_eval hash, and split hash.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydroda.data.file_hash import compute_sha256


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return dict(payload)


def _nested_get(payload: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _first_float(payload: Mapping[str, Any], candidates: tuple[tuple[str, ...], ...]) -> float:
    for keys in candidates:
        value = _nested_get(payload, keys)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    raise ValueError(f"Could not extract RMSE from summary keys: {candidates}")


def _extract_rmse(payload: Mapping[str, Any]) -> dict[str, float]:
    return {
        "surface_rmse": _first_float(
            payload,
            (
                ("surface", "rmse_latw_mean"),
                ("surface", "rmse_mean"),
                ("surface_rmse_latw",),
                ("surface_rmse",),
                ("surface_rmse_mean",),
            ),
        ),
        "rootzone_rmse": _first_float(
            payload,
            (
                ("rootzone", "rmse_latw_mean"),
                ("rootzone", "rmse_mean"),
                ("rootzone_rmse_latw",),
                ("rootzone_rmse",),
                ("rootzone_rmse_mean",),
            ),
        ),
    }


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": compute_sha256(path),
    }


def build_audit(
    *,
    m2_1_checkpoint: Path,
    m2_3_checkpoint: Path,
    m2_1_target_eval_summary: Path,
    m2_3_target_eval_summary: Path,
    split_manifest: Path,
) -> dict[str, Any]:
    m2_1_summary = _read_json(m2_1_target_eval_summary)
    m2_3_summary = _read_json(m2_3_target_eval_summary)
    split_payload = _read_json(split_manifest)
    split_sha = compute_sha256(split_manifest)

    m2_1_metrics = _extract_rmse(m2_1_summary)
    m2_3_metrics = _extract_rmse(m2_3_summary)
    protocol_values = {
        "split_manifest": split_payload.get("protocol_freeze_id", ""),
        "m2_1": m2_1_summary.get("protocol_freeze_id", ""),
        "m2_3": m2_3_summary.get("protocol_freeze_id", ""),
    }
    split_hash_values = {
        "actual_split_manifest_sha256": split_sha,
        "m2_1_summary_split_manifest_sha256": m2_1_summary.get("split_manifest_sha256", ""),
        "m2_3_summary_split_manifest_sha256": m2_3_summary.get("split_manifest_sha256", ""),
    }
    context_values = {
        "split_manifest": split_payload.get("target_context_dates_hash", ""),
        "m2_1": m2_1_summary.get("target_context_dates_hash", ""),
        "m2_3": m2_3_summary.get("target_context_dates_hash", ""),
    }
    eval_values = {
        "split_manifest": split_payload.get("target_eval_dates_hash", ""),
        "m2_1": m2_1_summary.get("target_eval_dates_hash", ""),
        "m2_3": m2_3_summary.get("target_eval_dates_hash", ""),
    }

    def all_same(values: Mapping[str, Any]) -> bool:
        clean = [str(value) for value in values.values() if str(value)]
        return bool(clean) and len(set(clean)) == 1

    same_split_hash = (
        str(split_hash_values["m2_1_summary_split_manifest_sha256"]) == split_sha
        and str(split_hash_values["m2_3_summary_split_manifest_sha256"]) == split_sha
    )
    surface_delta = m2_3_metrics["surface_rmse"] - m2_1_metrics["surface_rmse"]
    rootzone_delta = m2_3_metrics["rootzone_rmse"] - m2_1_metrics["rootzone_rmse"]
    surface_pct = surface_delta / m2_1_metrics["surface_rmse"] * 100.0
    rootzone_pct = rootzone_delta / m2_1_metrics["rootzone_rmse"] * 100.0
    m2_3_better_or_equal = surface_delta <= 0.0 and rootzone_delta <= 0.0

    return {
        "audit_schema_version": "m2_3_k0_transfer_audit_v1",
        "m2_1_checkpoint": _artifact(m2_1_checkpoint),
        "m2_3_checkpoint": _artifact(m2_3_checkpoint),
        "split_manifest": _artifact(split_manifest),
        "m2_1_target_eval_summary": _artifact(m2_1_target_eval_summary),
        "m2_3_target_eval_summary": _artifact(m2_3_target_eval_summary),
        "protocol": {
            "values": protocol_values,
            "same": all_same(protocol_values),
        },
        "split_manifest_hash": {
            "values": split_hash_values,
            "same": same_split_hash,
        },
        "target_context_hash": {
            "values": context_values,
            "same": all_same(context_values),
        },
        "target_eval_hash": {
            "values": eval_values,
            "same": all_same(eval_values),
        },
        "same_protocol": all_same(protocol_values) and same_split_hash and all_same(context_values) and all_same(eval_values),
        "m2_1": m2_1_metrics,
        "m2_3": m2_3_metrics,
        "delta": {
            "surface_rmse": surface_delta,
            "rootzone_rmse": rootzone_delta,
            "surface_rmse_relative_pct": surface_pct,
            "rootzone_rmse_relative_pct": rootzone_pct,
        },
        "interpretation": {
            "m2_3_replaces_m2_1": bool(m2_3_better_or_equal),
            "recommended_status": (
                "candidate_replacement_pending_more_seeds"
                if m2_3_better_or_equal
                else "negative_diagnostic_ablation"
            ),
            "reason": (
                "M2.3 target_eval RMSE is no worse than M2.1 for both variables"
                if m2_3_better_or_equal
                else "M2.3 target_eval RMSE is worse than M2.1 for at least one variable"
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit M2.3 K=0 target transfer against M2.1.")
    parser.add_argument("--m2_1_checkpoint", type=Path, required=True)
    parser.add_argument("--m2_3_checkpoint", type=Path, required=True)
    parser.add_argument("--m2_1_target_eval_summary", type=Path, required=True)
    parser.add_argument("--m2_3_target_eval_summary", type=Path, required=True)
    parser.add_argument("--split_manifest", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_audit(
        m2_1_checkpoint=args.m2_1_checkpoint,
        m2_3_checkpoint=args.m2_3_checkpoint,
        m2_1_target_eval_summary=args.m2_1_target_eval_summary,
        m2_3_target_eval_summary=args.m2_3_target_eval_summary,
        split_manifest=args.split_manifest,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote M2.3 K=0 transfer audit: {args.output_json}")


if __name__ == "__main__":
    main()
