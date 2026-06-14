#!/usr/bin/env python3
"""Build source-only HyperDA++ source-prior matrix reports.

The report intentionally reads per-candidate source-stage artifacts only. It
does not inspect target_val, target_eval, prediction records, or evaluation
directories.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


DEFAULT_RUNS_ROOT = Path("artifacts/runs/phase4_hyperda_plus")
DEFAULT_REPORT_DIR = Path("artifacts/reports/hyperda_plus_source_prior_matrix")
DEFAULT_MATRIX_CONFIG = Path("configs/experiments/hyperda_plus_source_prior_matrix.yaml")

PRIMARY_METRIC = "source_val_transfer_safe_score"
ROOTZONE_SKILL = "source_val_skill_rootzone"
K12_NONDEGRADATION = "source_pseudo_target_k12_non_degradation_rate"
ADAPTATION_DRIFT = "source_pseudo_target_adaptation_drift"
OPTIONAL_METRICS = {K12_NONDEGRADATION, ADAPTATION_DRIFT}


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _latest_run_dir(candidate_dir: Path) -> Optional[Path]:
    if not candidate_dir.exists():
        return None
    dirs = [p for p in candidate_dir.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return sorted(dirs)[-1]


def _metric(metrics: Dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in metrics and metrics[name] is not None:
            return metrics[name]
    return None


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _display_metric(value: Any, optional: bool = False) -> Any:
    if value is None and optional:
        return "not_available"
    if value is None:
        return ""
    return value


def _candidate_row(candidate_id: str, run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "source_val_metrics.json"
    metrics = _load_json(metrics_path) if metrics_path.exists() else {}
    config_path = run_dir / "config_resolved.yaml"
    config = _load_yaml(config_path) if config_path.exists() else {}
    checkpoint_metadata_path = run_dir / "checkpoint_metadata.json"
    checkpoint_metadata = (
        _load_json(checkpoint_metadata_path)
        if checkpoint_metadata_path.exists()
        else {}
    )

    safe_score = _metric(metrics, PRIMARY_METRIC, "selection_score", "best_safe_score")
    rootzone_skill = _metric(metrics, ROOTZONE_SKILL, "source_val_skill_rootzone")
    k12_non_degradation = _metric(metrics, K12_NONDEGRADATION)
    adaptation_drift = _metric(metrics, ADAPTATION_DRIFT)

    return {
        "candidate_id": candidate_id,
        "run_dir": str(run_dir),
        "checkpoint": checkpoint_metadata.get(
            "checkpoint",
            str(run_dir / "checkpoint_best_source_val_transfer_safe_score.pt"),
        ),
        PRIMARY_METRIC: _display_metric(safe_score),
        ROOTZONE_SKILL: _display_metric(rootzone_skill),
        K12_NONDEGRADATION: _display_metric(k12_non_degradation, optional=True),
        ADAPTATION_DRIFT: _display_metric(adaptation_drift, optional=True),
        "context_encoder": config.get("context_encoder", ""),
        "width": config.get("width", ""),
        "prompt_dim": config.get("prompt_dim", ""),
        "hyper_n_basis": config.get("hyper_n_basis", ""),
        "_safe_score_float": _as_float(safe_score, default=float("-inf")),
        "_rootzone_skill_float": _as_float(rootzone_skill, default=float("-inf")),
        "_k12_available": k12_non_degradation is not None,
        "_k12_float": _as_float(k12_non_degradation, default=float("-inf")),
        "_drift_available": adaptation_drift is not None,
        "_drift_float": _as_float(adaptation_drift, default=float("inf")),
    }


def _selection_key(row: Dict[str, Any], *, optional_available_for_all: Dict[str, bool]) -> tuple:
    key = [row["_safe_score_float"], row["_rootzone_skill_float"]]
    if optional_available_for_all[K12_NONDEGRADATION]:
        key.append(row["_k12_float"])
    else:
        key.append(float("-inf"))
    if optional_available_for_all[ADAPTATION_DRIFT]:
        key.append(-row["_drift_float"])
    else:
        key.append(float("-inf"))
    return tuple(key)


def _public_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def _write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "candidate_id",
        PRIMARY_METRIC,
        ROOTZONE_SKILL,
        K12_NONDEGRADATION,
        ADAPTATION_DRIFT,
        "context_encoder",
        "width",
        "prompt_dim",
        "hyper_n_basis",
        "checkpoint",
        "run_dir",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_summary_md(rows: List[Dict[str, Any]], best: Dict[str, Any], path: Path) -> None:
    lines = [
        "# HyperDA++ Source-Prior Matrix",
        "",
        "Selection uses source_fit training and source_val metrics only.",
        "",
        f"Best source prior: `{best['candidate_id']}`",
        "",
        "| candidate | safe_score | rootzone_skill | K12_non_degradation | adaptation_drift | context_encoder |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {candidate_id} | {safe} | {root} | {k12} | {drift} | {encoder} |".format(
                candidate_id=row["candidate_id"],
                safe=row.get(PRIMARY_METRIC, ""),
                root=row.get(ROOTZONE_SKILL, ""),
                k12=row.get(K12_NONDEGRADATION, ""),
                drift=row.get(ADAPTATION_DRIFT, ""),
                encoder=row.get("context_encoder", ""),
            )
        )
    lines.extend(
        [
            "",
            "Optional tie-breakers marked `not_available` are recorded but do not promote a candidate.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def build_report(
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    report_dir: Path = DEFAULT_REPORT_DIR,
    matrix_config: Path = DEFAULT_MATRIX_CONFIG,
    candidate_order: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Build source-prior matrix reports and return selected rows."""
    if candidate_order is None:
        cfg = _load_yaml(matrix_config)
        candidate_order = [c["candidate_id"] for c in cfg.get("candidates", []) if c.get("status") != "todo_scaffold"]

    rows: List[Dict[str, Any]] = []
    for candidate_id in candidate_order:
        run_dir = _latest_run_dir(runs_root / candidate_id)
        if run_dir is None:
            continue
        rows.append(_candidate_row(candidate_id, run_dir))

    if not rows:
        raise FileNotFoundError(f"No candidate source-prior runs found under {runs_root}")

    optional_available_for_all = {
        K12_NONDEGRADATION: all(row["_k12_available"] for row in rows),
        ADAPTATION_DRIFT: all(row["_drift_available"] for row in rows),
    }
    best_row = max(rows, key=lambda row: _selection_key(row, optional_available_for_all=optional_available_for_all))

    public_rows = [_public_row(row) for row in rows]
    public_best = _public_row(best_row)

    report_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(public_rows, report_dir / "summary.csv")
    _write_summary_md(public_rows, public_best, report_dir / "summary.md")
    best_payload = {
        "candidate_id": public_best["candidate_id"],
        "checkpoint": public_best["checkpoint"],
        "run_dir": public_best["run_dir"],
        "selection_metric": PRIMARY_METRIC,
        "selection_value": public_best[PRIMARY_METRIC],
        "source_only_policy": "source_fit_training_source_val_selection_no_target_eval",
    }
    with open(report_dir / "best_source_prior.yaml", "w") as f:
        yaml.safe_dump(best_payload, f, sort_keys=False)

    return {
        "best_candidate_id": public_best["candidate_id"],
        "best": public_best,
        "rows": public_rows,
        "report_dir": str(report_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HyperDA++ source-prior matrix report.")
    parser.add_argument("--runs_root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--matrix_config", type=Path, default=DEFAULT_MATRIX_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_report(
        runs_root=args.runs_root,
        report_dir=args.report_dir,
        matrix_config=args.matrix_config,
    )
    print(f"best_candidate_id={result['best_candidate_id']}")
    print(f"report_dir={result['report_dir']}")


if __name__ == "__main__":
    main()
