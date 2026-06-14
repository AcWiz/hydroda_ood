#!/usr/bin/env python3
"""Build the US-R1 seed=0 WRMSE-only HyperDA-SG v1 ablation table."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


FROZEN_LOCKED_RECIPE: dict[str, Any] = {
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

TARGET_REGION = "US-R1"
SEED = 0
NA = "NA"
LATW_METRIC_KEYS = ("rmse_latw_mean", "analysis_rmse_latw_mean", "increment_rmse_latw_mean")
REGION_LATW_METRICS = ("analysis_rmse_latw", "increment_rmse_latw", "rmse_latw_mean")

ROW_SPECS: tuple[dict[str, str], ...] = (
    {
        "method_id": "forecast_only",
        "display_name": "Forecast-only",
        "source": "forecast_summary",
    },
    {
        "method_id": "source_only_backbone",
        "display_name": "Source-only backbone",
        "source": "source_only_summary",
    },
    {
        "method_id": "prompt_conditioned_shared_backbone",
        "display_name": "Prompt-conditioned shared backbone",
        "source": "prompt_conditioned_summary",
    },
    {
        "method_id": "hyperda_k0_zero_shot",
        "display_name": "HyperDA K0 zero-shot",
        "source": "p2_suite",
        "phase": "P2.6",
        "run_id": "A0_k0_identity_base",
    },
    {
        "method_id": "hyperda_k4_original",
        "display_name": "HyperDA K4 original",
        "source": "p2_suite",
        "phase": "P2.6",
        "run_id": "A1_k4_all_original_schedule",
    },
    {
        "method_id": "hyperda_k12_original",
        "display_name": "HyperDA K12 original",
        "source": "p2_suite",
        "phase": "P2.6",
        "run_id": "A2_k12_all_original_schedule",
    },
    {
        "method_id": "hyperda_sg_k12_selected_guard",
        "display_name": "HyperDA-SG K12 selected guard",
        "source": "locked_eval",
    },
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml_or_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _to_float(value: Any) -> float | None:
    if value in (None, "", NA):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _format_value(value: float | None) -> str:
    if value is None:
        return NA
    return f"{value:.12g}"


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


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
            return int(float(actual)) == expected
        except (TypeError, ValueError):
            return False
    return actual == expected


def _validate_recipe(payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    recipe: dict[str, Any] = {}
    mismatches: list[str] = []
    for key, expected in FROZEN_LOCKED_RECIPE.items():
        actual = _recipe_value(payload, key)
        recipe[key] = actual
        if not _values_equal(actual, expected):
            mismatches.append(f"{label}.{key}: expected {expected!r}, got {actual!r}")
    if mismatches:
        raise ValueError("Frozen selected recipe mismatch: " + "; ".join(mismatches))
    return recipe


def _summary_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"surface": None, "rootzone": None, "artifact": ""}
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        return {"surface": None, "rootzone": None, "artifact": str(path)}
    surface_block = payload.get("surface", {})
    rootzone_block = payload.get("rootzone", {})
    if not isinstance(surface_block, Mapping):
        surface_block = {}
    if not isinstance(rootzone_block, Mapping):
        rootzone_block = {}
    surface = _to_float(_first_present(surface_block, LATW_METRIC_KEYS))
    rootzone = _to_float(_first_present(rootzone_block, LATW_METRIC_KEYS))
    metric_artifact = path
    if surface is None or rootzone is None:
        region_metrics = _metrics_by_region_latw(path.parent / "metrics_by_region.csv")
        if surface is None and region_metrics["surface"] is not None:
            surface = region_metrics["surface"]
            metric_artifact = path.parent / "metrics_by_region.csv"
        if rootzone is None and region_metrics["rootzone"] is not None:
            rootzone = region_metrics["rootzone"]
            metric_artifact = path.parent / "metrics_by_region.csv"
    return {"surface": surface, "rootzone": rootzone, "artifact": str(metric_artifact), "summary": str(path)}


def _metrics_by_region_latw(path: Path) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {"surface": None, "rootzone": None}
    if not path.exists():
        return metrics
    rows = _read_csv_rows(path)
    for metric_name in REGION_LATW_METRICS:
        for row in rows:
            if row.get("target_region_id") != TARGET_REGION:
                continue
            variable = row.get("variable")
            if variable not in metrics or metrics[variable] is not None:
                continue
            if row.get("metric") != metric_name:
                continue
            value = _to_float(row.get("mean"))
            if value is not None:
                metrics[variable] = value
    return metrics


def _latest_existing(paths: Sequence[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: (path.stat().st_mtime, str(path)))


def _find_forecast_summary(runs_root: Path) -> Path | None:
    candidates = [
        runs_root / "phase3_forecast_only_all_regions" / "target_eval" / TARGET_REGION / "summary.json",
        runs_root.parent / "results" / "phase3_forecast_only_all_regions" / "target_eval" / TARGET_REGION / "summary.json",
    ]
    return _latest_existing(candidates)


def _find_source_only_summary(runs_root: Path) -> Path | None:
    candidates = list(
        (runs_root / "phase4_source_only").glob(
            f"*source_only_{TARGET_REGION}_*s{SEED}_*/results/checkpoint_epoch_*/target_eval/{TARGET_REGION}/summary.json"
        )
    )
    candidates.extend(
        (runs_root / "phase4_source_only").glob(
            f"*source_only_{TARGET_REGION}_*s{SEED}_*/results/checkpoint_best_source_val_safe_score/target_eval/{TARGET_REGION}/summary.json"
        )
    )
    candidates.append(
        runs_root.parent
        / "results"
        / "phase4_source_only_inference"
        / "checkpoints_20260530_122716"
        / "target_eval"
        / TARGET_REGION
        / "summary.json"
    )
    return _latest_existing(candidates)


def _excluded_source_only_oracle_candidates(runs_root: Path) -> list[dict[str, str]]:
    candidates = sorted(
        (runs_root / "phase4_source_only_region_specific").glob(
            f"*source_only_{TARGET_REGION}_*s{SEED}_*/results/checkpoint_best_source_val_safe_score/target_eval/{TARGET_REGION}/summary.json"
        )
    )
    return [
        {
            "method_id": "source_only_backbone",
            "path": str(path),
            "reason": "phase4_source_only_region_specific is a target-full-history region-oracle/internal upper-bound artifact, not the main LORO/global source-only baseline.",
        }
        for path in candidates
    ]


def _find_prompt_conditioned_summary(runs_root: Path) -> Path | None:
    candidates = list(
        (runs_root / "phase4_prompt_conditioned").glob(
            f"*prompt_conditioned_{TARGET_REGION}_*s{SEED}_*/results/checkpoint_best_source_val_transfer_safe_score/target_eval/{TARGET_REGION}/summary.json"
        )
    )
    candidates.extend(
        (runs_root / "phase4_prompt_conditioned").glob(
            f"*prompt_conditioned_{TARGET_REGION}_*s{SEED}_*/results/best/target_eval/{TARGET_REGION}/summary.json"
        )
    )
    return _latest_existing(candidates)


def _p2_suite_metric(p2_rows: Sequence[Mapping[str, Any]], spec: Mapping[str, str]) -> dict[str, Any]:
    phase = spec["phase"]
    run_id = spec["run_id"]
    for row in p2_rows:
        if row.get("phase") == phase and row.get("run_id") == run_id:
            return {
                "surface": _to_float(row.get("surface_rmse_latw")),
                "rootzone": _to_float(row.get("rootzone_rmse_latw")),
                "artifact": str(row.get("source_file") or "p2_suite_summary.csv"),
            }
    return {"surface": None, "rootzone": None, "artifact": ""}


def _locked_eval_summary_path(locked_eval_dir: Path) -> Path:
    preferred = locked_eval_dir / "p2_8_locked_guard_target_eval_summary.csv"
    if preferred.exists():
        return preferred
    matches = sorted(locked_eval_dir.glob("*locked*summary.csv"))
    if matches:
        return matches[-1]
    raise FileNotFoundError(f"locked eval summary CSV not found in {locked_eval_dir}")


def _resolve_locked_summary_json(summary_value: Any, locked_eval_dir: Path) -> Path | None:
    if summary_value in (None, ""):
        return None
    path = Path(str(summary_value))
    if path.exists():
        return path
    candidate = locked_eval_dir / path
    if candidate.exists():
        return candidate
    return path


def _locked_eval_metric(locked_eval_dir: Path, expected_hash: str) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = _locked_eval_summary_path(locked_eval_dir)
    rows = _read_csv_rows(summary_path)
    selected = None
    for row in rows:
        role = str(row.get("p2_8_role", ""))
        row_hash = str(row.get("selected_guard_config_hash") or row.get("guard_config_hash") or "")
        if role == "selected_guarded" or (expected_hash and row_hash == expected_hash):
            selected = row
    if selected is None:
        raise ValueError(f"selected guarded row not found in {summary_path}")
    if expected_hash:
        row_hash = str(selected.get("selected_guard_config_hash") or selected.get("guard_config_hash") or "")
        if row_hash and row_hash != expected_hash:
            raise ValueError(f"locked eval selected guard hash mismatch: expected {expected_hash}, got {row_hash}")
    _validate_recipe(selected, label="locked_eval_summary")
    surface = _to_float(selected.get("surface_rmse_latw"))
    rootzone = _to_float(selected.get("rootzone_rmse_latw"))
    artifact = str(summary_path)
    if surface is None or rootzone is None:
        summary_json = _resolve_locked_summary_json(selected.get("summary"), locked_eval_dir)
        if summary_json is not None:
            fallback = _summary_metrics(summary_json)
            surface = surface if surface is not None else _to_float(fallback.get("surface"))
            rootzone = rootzone if rootzone is not None else _to_float(fallback.get("rootzone"))
            artifact = str(fallback.get("artifact") or summary_json)
    return {
        "surface": surface,
        "rootzone": rootzone,
        "artifact": artifact,
    }, selected


def _row(method_id: str, display_name: str, metrics: Mapping[str, Any], source: str) -> dict[str, str]:
    surface = _to_float(metrics.get("surface"))
    rootzone = _to_float(metrics.get("rootzone"))
    mean = (surface + rootzone) / 2.0 if surface is not None and rootzone is not None else None
    return {
        "method_id": method_id,
        "display_name": display_name,
        "target_region": TARGET_REGION,
        "seed": str(SEED),
        "surface_rmse_latw": _format_value(surface),
        "rootzone_rmse_latw": _format_value(rootzone),
        "mean_surface_rootzone_rmse_latw": _format_value(mean),
        "artifact": str(metrics.get("artifact") or ""),
        "source": source,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    fieldnames = [
        "method_id",
        "display_name",
        "target_region",
        "seed",
        "surface_rmse_latw",
        "rootzone_rmse_latw",
        "mean_surface_rootzone_rmse_latw",
        "artifact",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_markdown(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    lines = [
        "# US-R1 WRMSE Ablation Table",
        "",
        "Seed=0 pilot WRMSE-only table. Rows are fixed by protocol order, not sorted by WRMSE.",
        "",
        "| Method | Surface WRMSE | Rootzone WRMSE | Mean |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {row['surface_rmse_latw']} | "
            f"{row['rootzone_rmse_latw']} | {row['mean_surface_rootzone_rmse_latw']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & Surface WRMSE & Rootzone WRMSE & Mean \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['display_name']} & {row['surface_rmse_latw']} & "
            f"{row['rootzone_rmse_latw']} & {row['mean_surface_rootzone_rmse_latw']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_summary(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    lines = [
        "# US-R1 WRMSE Ablation Summary",
        "",
        "This is a seed=0 pilot WRMSE-only table for the locked HyperDA-SG v1 evaluation.",
        "The table excludes methods outside this seed=0 pilot scope.",
        "",
        f"- Target region: `{TARGET_REGION}`",
        f"- Seed: `{SEED}`",
        "- Final method: `hyperda_sg_k12_selected_guard`",
        "- Target eval was not used for method selection; selected guard comes from source-safe P2.8 calibration.",
        "",
        "## Missing Artifacts",
    ]
    missing = [row for row in rows if row["surface_rmse_latw"] == NA or row["rootzone_rmse_latw"] == NA]
    if missing:
        lines.extend(f"- `{row['method_id']}`" for row in missing)
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_us_r1_wrmse_ablation_table(
    *,
    runs_root: Path | str,
    locked_eval_dir: Path | str,
    p2_suite_report_dir: Path | str,
    output_dir: Path | str,
) -> dict[str, Any]:
    runs_root = Path(runs_root)
    locked_eval_dir = Path(locked_eval_dir)
    p2_suite_report_dir = Path(p2_suite_report_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recommendation_path = p2_suite_report_dir / "selected_recipe_recommendation.yaml"
    leakage_path = p2_suite_report_dir / "leakage_audit.json"
    p2_summary_path = p2_suite_report_dir / "p2_suite_summary.csv"
    recommendation = _read_yaml_or_json(recommendation_path)
    leakage = _read_json(leakage_path)
    if not isinstance(recommendation, Mapping):
        raise ValueError(f"selected recipe recommendation is not a mapping: {recommendation_path}")
    if not isinstance(leakage, Mapping):
        raise ValueError(f"leakage audit is not a mapping: {leakage_path}")
    if recommendation.get("locked_eval_ready") is not True:
        raise ValueError("selected recipe recommendation is not locked_eval_ready")
    if leakage.get("verdict") != "pass":
        raise ValueError(f"leakage audit verdict is not pass: {leakage.get('verdict')}")
    recommendation_recipe = _validate_recipe(recommendation, label="selected_recipe_recommendation")

    guard_path = Path(str(recommendation.get("selected_guard_config_path", "")))
    if not guard_path.exists():
        raise FileNotFoundError(f"selected guard config not found: {guard_path}")
    selected_guard = _read_yaml_or_json(guard_path)
    if not isinstance(selected_guard, Mapping):
        raise ValueError(f"selected guard config is not a mapping: {guard_path}")
    guard_recipe = _validate_recipe(selected_guard, label="selected_guard_config")
    guard_hash = str(selected_guard.get("guard_config_hash") or recommendation.get("selected_guard_config_hash") or "")
    if not guard_hash:
        raise ValueError("selected guard config hash is missing")
    recorded_hash = str(recommendation.get("selected_guard_config_hash") or guard_hash)
    if recorded_hash != guard_hash:
        raise ValueError(f"selected guard config hash mismatch: recommendation has {recorded_hash}, guard has {guard_hash}")

    p2_rows = _read_csv_rows(p2_summary_path)
    locked_metrics, locked_row = _locked_eval_metric(locked_eval_dir, guard_hash)

    metrics_by_source: dict[str, dict[str, Any]] = {
        "forecast_summary": _summary_metrics(_find_forecast_summary(runs_root) or Path("__missing__")),
        "source_only_summary": _summary_metrics(_find_source_only_summary(runs_root) or Path("__missing__")),
        "prompt_conditioned_summary": _summary_metrics(_find_prompt_conditioned_summary(runs_root) or Path("__missing__")),
        "locked_eval": locked_metrics,
    }

    rows: list[dict[str, str]] = []
    for spec in ROW_SPECS:
        if spec["source"] == "p2_suite":
            metrics = _p2_suite_metric(p2_rows, spec)
        else:
            metrics = metrics_by_source[spec["source"]]
        rows.append(_row(spec["method_id"], spec["display_name"], metrics, spec["source"]))

    csv_path = output_dir / "us_r1_wrmse_ablation.csv"
    md_path = output_dir / "us_r1_wrmse_ablation.md"
    tex_path = output_dir / "us_r1_wrmse_ablation.tex"
    metadata_path = output_dir / "metadata.json"
    summary_path = output_dir / "summary.md"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    _write_latex(tex_path, rows)
    metadata = {
        "target_region": TARGET_REGION,
        "seed": SEED,
        "row_order": [row["method_id"] for row in rows],
        "wrmse_fields": ["surface_rmse_latw", "rootzone_rmse_latw", "mean_surface_rootzone_rmse_latw"],
        "p2_suite_report_dir": str(p2_suite_report_dir),
        "locked_eval_dir": str(locked_eval_dir),
        "selected_recipe_recommendation": str(recommendation_path),
        "selected_recipe": recommendation_recipe,
        "selected_guard_config": str(guard_path),
        "selected_guard_config_hash": guard_hash,
        "selected_guard_recipe": guard_recipe,
        "locked_eval_selected_row": locked_row,
        "target_eval_not_used_for_selection": True,
        "pilot_scope": "seed=0 only",
        "excluded_methods": ["methods outside seed=0 pilot scope"],
        "excluded_baseline_artifacts": _excluded_source_only_oracle_candidates(runs_root),
        "outputs": {
            "csv": str(csv_path),
            "md": str(md_path),
            "tex": str(tex_path),
            "metadata": str(metadata_path),
            "summary": str(summary_path),
        },
    }
    _write_json(metadata_path, metadata)
    _write_summary(summary_path, rows)
    return {"output_dir": str(output_dir), "row_count": len(rows), "outputs": metadata["outputs"]}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build US-R1 seed=0 WRMSE-only ablation table.")
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--locked_eval_dir", type=Path, required=True)
    parser.add_argument("--p2_suite_report_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_us_r1_wrmse_ablation_table(
        runs_root=args.runs_root,
        locked_eval_dir=args.locked_eval_dir,
        p2_suite_report_dir=args.p2_suite_report_dir,
        output_dir=args.output_dir,
    )
    print(f"Wrote US-R1 WRMSE ablation table: {result['output_dir']}")


if __name__ == "__main__":
    main()
