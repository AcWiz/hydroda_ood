#!/usr/bin/env python3
"""Review a HyperDA zero/few-shot run overview for protocol and overfit signals."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml


FROZEN_LOCKED_RECIPE: Dict[str, Any] = {
    "K": 12,
    "adapt_scope": "all",
    "schedule_label": "K4_schedule_on_K12",
    "lr": 0.001,
    "adaptation_steps": 100,
    "anchor_alpha": 0.75,
    "support_loss_reduction": "global_pixel",
    "trust_policy": "none",
    "rho_policy": "rule_a",
}

DEFAULT_RECOMMENDATION = Path("artifacts/reports/hyperda_p2_suite_US-R1_s0/selected_recipe_recommendation.yaml")
DEFAULT_LEAKAGE_AUDIT = Path("artifacts/reports/hyperda_p2_suite_US-R1_s0/leakage_audit.json")
DEFAULT_SELECTED_GUARD = Path(
    "artifacts/runs/phase5_hyperda_p2_8_source_safe_guard_calibration/"
    "US-R1_s0_p2_8b_final_20260613T135859Z/selected_guard_config.yaml"
)
DEFAULT_CALIBRATION_SUMMARY = Path(
    "artifacts/runs/phase5_hyperda_p2_8_source_safe_guard_calibration/"
    "US-R1_s0_p2_8b_final_20260613T135859Z/final_source_safe_calibration_summary.json"
)
DEFAULT_SOURCE_CHECKPOINT = Path(
    "artifacts/runs/phase4_prompt_conditioned/"
    "phase4_prompt_conditioned_hyperda_basis_adapter_US-R1_w32_e50_lr0.0003_norm_zero_s0_20260531_224425/"
    "checkpoints/checkpoint_best_source_val_transfer_safe_score.pt"
)
DEFAULT_SPLIT_MANIFEST = Path("artifacts/splits/US_loro_zero_few_shot_splits.json")


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"overview.csv not found: {path}")
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml_or_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, "", "NA"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_baseline(path: Optional[Path]) -> Any:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"baseline overview not found: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".csv":
        return _read_csv_rows(path)
    return path.read_text(encoding="utf-8")


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _summary_metric(summary_path: Path, variable: str, metric_keys: Sequence[str]) -> Optional[float]:
    if not summary_path.exists():
        return None
    payload = _read_json(summary_path)
    if not isinstance(payload, Mapping):
        return None
    block = payload.get(variable, {})
    if not isinstance(block, Mapping):
        return None
    return _float_or_none(_first_present(block, metric_keys))


def _resolve_row_summary_path(row: Mapping[str, Any], overview_path: Path) -> Optional[Path]:
    value = row.get("summary")
    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.exists():
        return path
    candidate = overview_path.parent / path
    if candidate.exists():
        return candidate
    return path


def _hydrate_row_metrics(row: Dict[str, str], overview_path: Path) -> Dict[str, str]:
    summary_path = _resolve_row_summary_path(row, overview_path)
    if summary_path is None:
        return row
    if _float_or_none(row.get("surface_rmse_latw")) is None:
        surface = _summary_metric(summary_path, "surface", ("rmse_latw_mean", "analysis_rmse_latw_mean"))
        if surface is not None:
            row["surface_rmse_latw"] = f"{surface:.12g}"
    if _float_or_none(row.get("rootzone_rmse_latw")) is None:
        rootzone = _summary_metric(summary_path, "rootzone", ("rmse_latw_mean", "analysis_rmse_latw_mean"))
        if rootzone is not None:
            row["rootzone_rmse_latw"] = f"{rootzone:.12g}"
    if not row.get("status") and row.get("surface_rmse_latw") and row.get("rootzone_rmse_latw"):
        row["status"] = "ok"
    return row


def _row_key(row: Mapping[str, str], index: int) -> str:
    k_value = str(row.get("K", ""))
    role = str(row.get("p2_8_role", ""))
    if role == "selected_guarded":
        return f"K{k_value}_selected_guarded"
    if role == "baseline":
        return f"K{k_value}_baseline"
    setting = str(row.get("adaptation_setting", ""))
    return k_value or setting or f"row_{index}"


def _recipe_value(payload: Mapping[str, Any], key: str) -> Any:
    if key == "trust_policy":
        return payload.get("trust_policy", payload.get("trust_region_mode"))
    return payload.get(key)


def _values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) < 1e-12
        except (TypeError, ValueError):
            return False
    if isinstance(expected, int):
        try:
            return int(actual) == expected
        except (TypeError, ValueError):
            return False
    return actual == expected


def _validate_frozen_recipe(payload: Mapping[str, Any], *, label: str) -> Dict[str, Any]:
    recipe: Dict[str, Any] = {}
    mismatches: List[str] = []
    for key, expected in FROZEN_LOCKED_RECIPE.items():
        actual = _recipe_value(payload, key)
        recipe[key] = actual
        if not _values_equal(actual, expected):
            mismatches.append(f"{label}.{key}: expected {expected!r}, got {actual!r}")
    if mismatches:
        raise ValueError("Frozen locked recipe mismatch: " + "; ".join(mismatches))
    return recipe


def locked_eval_preflight(
    *,
    recommendation_path: Path = DEFAULT_RECOMMENDATION,
    leakage_audit_path: Path = DEFAULT_LEAKAGE_AUDIT,
    selected_guard_config_path: Path = DEFAULT_SELECTED_GUARD,
    calibration_summary_path: Path = DEFAULT_CALIBRATION_SUMMARY,
    source_checkpoint_path: Path = DEFAULT_SOURCE_CHECKPOINT,
    split_manifest_path: Path = DEFAULT_SPLIT_MANIFEST,
    target_region: str = "US-R1",
    seed: int = 0,
) -> Dict[str, Any]:
    """Validate source-safe selection artifacts before reviewing locked target_eval."""
    recommendation = _read_yaml_or_json(recommendation_path)
    leakage = _read_json(leakage_audit_path)
    selected_guard = _read_yaml_or_json(selected_guard_config_path)
    calibration_summary = _read_json(calibration_summary_path)
    if not isinstance(recommendation, Mapping):
        raise ValueError(f"recommendation is not a mapping: {recommendation_path}")
    if not isinstance(leakage, Mapping):
        raise ValueError(f"leakage audit is not a mapping: {leakage_audit_path}")
    if not isinstance(selected_guard, Mapping):
        raise ValueError(f"selected guard config is not a mapping: {selected_guard_config_path}")
    if not isinstance(calibration_summary, Mapping):
        raise ValueError(f"calibration summary is not a mapping: {calibration_summary_path}")

    if recommendation.get("locked_eval_ready") is not True:
        raise ValueError("locked_eval_ready is not true in selected_recipe_recommendation.yaml")
    if leakage.get("verdict") != "pass":
        raise ValueError(f"leakage audit verdict is not pass: {leakage.get('verdict')}")

    recommendation_recipe = _validate_frozen_recipe(recommendation, label="recommendation")
    guard_recipe = _validate_frozen_recipe(selected_guard, label="selected_guard_config")

    recommended_guard_path = Path(str(recommendation.get("selected_guard_config_path", "")))
    if recommended_guard_path and recommended_guard_path != selected_guard_config_path:
        if recommended_guard_path.resolve() != selected_guard_config_path.resolve():
            raise ValueError(
                "selected_guard_config_path mismatch: "
                f"recommendation has {recommended_guard_path}, got {selected_guard_config_path}"
            )

    guard_hash = str(selected_guard.get("guard_config_hash", ""))
    recommended_hash = str(
        recommendation.get("selected_guard_config_hash")
        or recommendation.get("guard_config_hash")
        or selected_guard.get("selected_guard_config_hash", "")
    )
    if recommended_hash and guard_hash and recommended_hash != guard_hash:
        raise ValueError(f"selected guard hash mismatch: recommendation has {recommended_hash}, config has {guard_hash}")
    if not guard_hash:
        raise ValueError("selected guard config does not record guard_config_hash")

    source_sha = _file_sha256(source_checkpoint_path)
    split_sha = _file_sha256(split_manifest_path)
    recorded_source_sha = str(recommendation.get("source_checkpoint_sha256", ""))
    recorded_split_sha = str(recommendation.get("split_manifest_sha256", ""))
    if recorded_source_sha and recorded_source_sha != source_sha:
        raise ValueError(f"source checkpoint sha256 mismatch: recommendation has {recorded_source_sha}, file has {source_sha}")
    if recorded_split_sha and recorded_split_sha != split_sha:
        raise ValueError(f"split manifest sha256 mismatch: recommendation has {recorded_split_sha}, file has {split_sha}")

    support_dates = {
        "target_support_dates": selected_guard.get(
            "target_support_dates",
            recommendation.get("target_support_dates", recommendation.get("support_dates", "")),
        ),
        "target_support_dates_hash": selected_guard.get(
            "target_support_dates_hash",
            recommendation.get("target_support_dates_hash", recommendation.get("support_dates_hash", "")),
        ),
        "k4_support_dates": selected_guard.get("k4_support_dates", recommendation.get("k4_support_dates", "")),
        "k12_support_dates": selected_guard.get("k12_support_dates", recommendation.get("k12_support_dates", "")),
    }

    return {
        "target_region": target_region,
        "seed": seed,
        "target_eval_not_used_for_selection": True,
        "locked_eval_ready": True,
        "leakage_verdict": leakage.get("verdict"),
        "recommendation_path": str(recommendation_path),
        "leakage_audit_path": str(leakage_audit_path),
        "calibration_summary_path": str(calibration_summary_path),
        "source_checkpoint_path": str(source_checkpoint_path),
        "source_checkpoint_sha256": source_sha,
        "source_checkpoint_recorded_sha256": recorded_source_sha,
        "selected_guard_config_path": str(selected_guard_config_path),
        "selected_guard_config_sha256": _file_sha256(selected_guard_config_path),
        "selected_guard_config_hash": guard_hash,
        "selected_guard_config_recorded_hash": recommended_hash,
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_sha256": split_sha,
        "split_manifest_recorded_sha256": recorded_split_sha,
        "support_dates": support_dates,
        "source_checkpoint": {
            "path": str(source_checkpoint_path),
            "sha256": source_sha,
            "recorded_sha256": recorded_source_sha,
        },
        "selected_guard_config": {
            "path": str(selected_guard_config_path),
            "sha256": _file_sha256(selected_guard_config_path),
            "guard_config_hash": guard_hash,
            "recorded_guard_config_hash": recommended_hash,
            "recipe": guard_recipe,
        },
        "recommendation_recipe": recommendation_recipe,
        "split_manifest": {
            "path": str(split_manifest_path),
            "sha256": split_sha,
            "recorded_sha256": recorded_split_sha,
        },
    }


def _baseline_mentions_legacy(payload: Any) -> bool:
    text = json.dumps(payload, sort_keys=True) if not isinstance(payload, str) else payload
    return "target_full_train" in text or "v4.3" in text or "legacy" in text.lower()


def build_run_review(
    overview_path: Path,
    baseline_path: Optional[Path] = None,
) -> Dict[str, Any]:
    rows = [_hydrate_row_metrics(dict(row), overview_path) for row in _read_csv_rows(overview_path)]
    k_rows = {_row_key(row, index): row for index, row in enumerate(rows)}
    warnings: List[str] = []

    rows_by_k = {str(row.get("K", "")): row for row in rows if row.get("p2_8_role") != "selected_guarded"}
    k0_rmse = _float_or_none(rows_by_k.get("0", {}).get("surface_rmse_latw"))
    for k in ("4", "12"):
        row = rows_by_k.get(k)
        if not row:
            continue
        rmse = _float_or_none(row.get("surface_rmse_latw"))
        loss_delta = _float_or_none(row.get("support_loss_delta"))
        if k0_rmse is not None and rmse is not None and rmse > k0_rmse and loss_delta is not None and loss_delta < 0:
            warnings.append(
                f"K={k} has worse target_eval surface WRMSE than K=0 while support_loss_delta is negative; "
                "treat as few-shot overfit signal before making paper claims."
            )

    baseline_payload = _load_baseline(baseline_path)
    same_protocol = True
    if baseline_payload is None:
        same_protocol = False
        warnings.append("No same-protocol source-only baseline overview supplied; old artifacts are historical only.")
    elif _baseline_mentions_legacy(baseline_payload):
        same_protocol = False
        warnings.append(
            "Baseline overview mentions v4.3/target_full_train/legacy; do not mix it with V4.4 zero/few-shot results."
        )

    status = "needs_audit" if warnings else "ok"
    return {
        "summary": {
            "status": status,
            "overview_path": str(overview_path),
            "baseline_path": str(baseline_path) if baseline_path else "",
            "row_count": len(rows),
        },
        "warnings": warnings,
        "k_rows": k_rows,
        "baseline_protocol": {
            "same_protocol": same_protocol,
            "baseline_path": str(baseline_path) if baseline_path else "",
        },
    }


def write_markdown(path: Path, review: Dict[str, Any]) -> None:
    lines = [
        "# HyperDA Zero/Few-Shot Run Review",
        "",
        f"- Status: `{review['summary']['status']}`",
        f"- Overview: `{review['summary']['overview_path']}`",
        f"- Same-protocol source-only baseline: `{review['baseline_protocol']['same_protocol']}`",
        "",
        "## Warnings",
    ]
    warnings = review.get("warnings", [])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")
    lines.extend(["", "## K Settings"])
    for key, row in sorted(
        review.get("k_rows", {}).items(),
        key=lambda item: (int(str(item[1].get("K", "999"))) if str(item[1].get("K", "")).isdigit() else 999, item[0]),
    ):
        role = row.get("p2_8_role", "")
        label = f"K={row.get('K', key)}"
        if role:
            label += f" ({role})"
        lines.append(
            "- "
            f"{label}: setting={row.get('adaptation_setting', '')}, "
            f"steps={row.get('adaptation_steps', '')}, lr={row.get('lr', '')}, "
            f"alpha={row.get('anchor_alpha', '')}, surface_WRMSE={row.get('surface_rmse_latw', '')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_locked_markdown(path: Path, review: Dict[str, Any], metadata_audit: Mapping[str, Any]) -> None:
    write_markdown(path, review)
    with path.open("a", encoding="utf-8") as f:
        f.write("\n## Locked Eval Metadata Audit\n")
        f.write(f"- Target region: `{metadata_audit.get('target_region', '')}`\n")
        f.write(f"- Seed: `{metadata_audit.get('seed', '')}`\n")
        f.write("- Target eval not used for selection: `true`\n")
        f.write(f"- Source checkpoint SHA256: `{metadata_audit.get('source_checkpoint', {}).get('sha256', '')}`\n")
        f.write(
            "- Selected guard hash: "
            f"`{metadata_audit.get('selected_guard_config', {}).get('guard_config_hash', '')}`\n"
        )
        f.write(f"- Split manifest SHA256: `{metadata_audit.get('split_manifest', {}).get('sha256', '')}`\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review HyperDA zero/few-shot run artifacts.")
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--overview_csv", type=Path, default=None)
    parser.add_argument("--baseline_overview", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_md", type=Path, default=None)
    parser.add_argument("--recommendation", type=Path, default=None)
    parser.add_argument("--leakage_audit", type=Path, default=None)
    parser.add_argument("--selected_guard_config", type=Path, default=None)
    parser.add_argument("--calibration_summary", type=Path, default=None)
    parser.add_argument("--source_checkpoint", type=Path, default=None)
    parser.add_argument("--split_manifest", type=Path, default=None)
    parser.add_argument("--target_region", default="US-R1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.overview_csv is None:
        if args.run_dir is None:
            parser.error("provide --overview_csv or --run_dir")
        overview = args.run_dir / "overview.csv"
        locked_summary = args.run_dir / "p2_8_locked_guard_target_eval_summary.csv"
        args.overview_csv = overview if overview.exists() or not locked_summary.exists() else locked_summary
    if args.output_dir is not None:
        if args.output_json is None:
            args.output_json = args.output_dir / "locked_eval_summary.json"
        if args.output_md is None:
            args.output_md = args.output_dir / "locked_eval_summary.md"
    elif args.output_json is None:
        base = args.run_dir if args.run_dir is not None else args.overview_csv.parent
        args.output_json = base / "run_review.json"
    if args.output_md is None:
        base = args.run_dir if args.run_dir is not None else args.overview_csv.parent
        args.output_md = base / "run_review.md"
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    review = build_run_review(args.overview_csv, args.baseline_overview)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(review, indent=2), encoding="utf-8")
    if args.output_dir is not None:
        metadata_audit = locked_eval_preflight(
            recommendation_path=args.recommendation or DEFAULT_RECOMMENDATION,
            leakage_audit_path=args.leakage_audit or DEFAULT_LEAKAGE_AUDIT,
            selected_guard_config_path=args.selected_guard_config or DEFAULT_SELECTED_GUARD,
            calibration_summary_path=args.calibration_summary or DEFAULT_CALIBRATION_SUMMARY,
            source_checkpoint_path=args.source_checkpoint or DEFAULT_SOURCE_CHECKPOINT,
            split_manifest_path=args.split_manifest or DEFAULT_SPLIT_MANIFEST,
            target_region=args.target_region,
            seed=args.seed,
        )
        audit_path = args.output_dir / "locked_eval_metadata_audit.json"
        audit_path.write_text(json.dumps(metadata_audit, indent=2, sort_keys=True), encoding="utf-8")
        write_locked_markdown(args.output_md, review, metadata_audit)
        print(f"Wrote locked eval metadata audit: {audit_path}")
    else:
        write_markdown(args.output_md, review)
    print(f"Wrote run review JSON: {args.output_json}")
    print(f"Wrote run review MD: {args.output_md}")


if __name__ == "__main__":
    main()
