#!/usr/bin/env python3
"""Build source-stage summary tables for staged HyperDA V1 ablations."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping


ABLATION_ORDER = (
    "M0_current",
    "M1_shared_coeff",
    "M2_shared_coeff_gate",
    "M2_rank_gated_dora",
    "M2_1_rank_gated_dora_stable",
    "M2_2_source_saliency_prior",
    "M2_3_source_safe_residual_hyperda",
    "M3_film_only",
    "M4_adapter_only",
)

FIELDNAMES = [
    "rank_by_best_selection_value",
    "ablation_id",
    "target_region",
    "seed",
    "run_name",
    "best_selection_metric",
    "best_selection_value",
    "best_safe_score",
    "trainable_parameter_count",
    "hyper_coeff_generator",
    "hyper_adapter_param_style",
    "hyper_rank_gate_temperature_init",
    "hyper_reliability_gate",
    "hyper_reliability_init",
    "hyper_source_saliency_prior_beta",
    "hyper_source_saliency_prior_path",
    "hyper_source_saliency_prior_application",
    "hyper_prompt_manifold_reliability",
    "hyper_prompt_manifold_reliability_strength",
    "zero_shot_prior_form",
    "source_residual_rho",
    "zero_shot_rho_selection_source",
    "hyper_residual_magnitude_penalty",
    "hyper_coeff_entropy_floor",
    "hyper_coeff_entropy_penalty",
    "target_labels_used_for_adaptation",
    "target_eval_input_stats_used_for_update",
    "hyper_enable_film",
    "hyper_enable_adapters",
    "source_episode_prompt_policy",
    "source_anchor_blend_calibration",
    "hyper_output_head_residual",
    "normalization_source",
    "protocol_freeze_id",
    "split_manifest_sha256",
    "source_base_checkpoint_sha256",
    "best_checkpoint",
    "summary_json",
    "run_dir",
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "1" if value else "0"
    numeric = _to_float(value)
    if numeric is not None:
        return f"{numeric:.12g}"
    return str(value)


def _latest_summary_for_ablation(runs_root: Path, ablation_id: str, target_region: str, seed: int) -> Path | None:
    region_dir = runs_root / ablation_id / target_region
    if not region_dir.exists():
        return None
    candidates: list[Path] = []
    for run_dir in region_dir.iterdir():
        if not run_dir.is_dir() or f"s{seed}" not in run_dir.name:
            continue
        for relative in [Path("summary.json"), Path("reports") / "summary.json"]:
            path = run_dir / relative
            if path.exists():
                candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))


def _run_dir_for_summary(summary_path: Path) -> Path:
    if summary_path.parent.name == "reports":
        return summary_path.parent.parent
    return summary_path.parent


def _best_checkpoint_for_run(run_dir: Path) -> Path | None:
    preferred = run_dir / "checkpoints" / "checkpoint_best_source_val_transfer_safe_score.pt"
    if preferred.exists():
        return preferred
    candidates = sorted((run_dir / "checkpoints").glob("checkpoint_best*.pt"))
    return candidates[-1] if candidates else None


def _row_from_summary(
    *,
    ablation_id: str,
    summary_path: Path,
    target_region: str,
    seed: int,
) -> dict[str, Any]:
    payload = _read_json(summary_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"summary.json is not an object: {summary_path}")
    run_dir = _run_dir_for_summary(summary_path)
    checkpoint = _best_checkpoint_for_run(run_dir)
    return {
        "rank_by_best_selection_value": "",
        "ablation_id": ablation_id,
        "target_region": str(payload.get("target_region") or target_region),
        "seed": str(payload.get("seed") if payload.get("seed") is not None else seed),
        "run_name": str(payload.get("experiment_id") or run_dir.name),
        "best_selection_metric": str(payload.get("best_selection_metric") or ""),
        "best_selection_value": _format_value(payload.get("best_selection_value")),
        "best_safe_score": _format_value(payload.get("best_safe_score")),
        "trainable_parameter_count": _format_value(
            payload.get("trainable_parameter_count", payload.get("trainable_parameters"))
        ),
        "hyper_coeff_generator": str(payload.get("hyper_coeff_generator") or "per_adapter"),
        "hyper_adapter_param_style": str(payload.get("hyper_adapter_param_style") or "basis_1x1"),
        "hyper_rank_gate_temperature_init": _format_value(payload.get("hyper_rank_gate_temperature_init")),
        "hyper_reliability_gate": str(payload.get("hyper_reliability_gate") or "none"),
        "hyper_reliability_init": _format_value(payload.get("hyper_reliability_init")),
        "hyper_source_saliency_prior_beta": _format_value(payload.get("hyper_source_saliency_prior_beta")),
        "hyper_source_saliency_prior_path": str(payload.get("hyper_source_saliency_prior_path") or ""),
        "hyper_source_saliency_prior_application": str(
            payload.get("hyper_source_saliency_prior_application")
            or payload.get("hyper_source_saliency_prior_metadata", {}).get("application", "")
        ),
        "hyper_prompt_manifold_reliability": _format_value(payload.get("hyper_prompt_manifold_reliability")),
        "hyper_prompt_manifold_reliability_strength": _format_value(
            payload.get("hyper_prompt_manifold_reliability_strength")
        ),
        "zero_shot_prior_form": str(payload.get("zero_shot_prior_form") or ""),
        "source_residual_rho": _format_value(payload.get("source_residual_rho", payload.get("zero_shot_rho"))),
        "zero_shot_rho_selection_source": str(payload.get("zero_shot_rho_selection_source") or ""),
        "hyper_residual_magnitude_penalty": _format_value(payload.get("hyper_residual_magnitude_penalty")),
        "hyper_coeff_entropy_floor": _format_value(payload.get("hyper_coeff_entropy_floor")),
        "hyper_coeff_entropy_penalty": _format_value(payload.get("hyper_coeff_entropy_penalty")),
        "target_labels_used_for_adaptation": _format_value(payload.get("target_labels_used_for_adaptation")),
        "target_eval_input_stats_used_for_update": _format_value(
            payload.get("target_eval_input_stats_used_for_update")
        ),
        "hyper_enable_film": _format_value(payload.get("hyper_enable_film")),
        "hyper_enable_adapters": _format_value(payload.get("hyper_enable_adapters")),
        "source_episode_prompt_policy": str(payload.get("source_episode_prompt_policy") or ""),
        "source_anchor_blend_calibration": _format_value(payload.get("source_anchor_blend_calibration")),
        "hyper_output_head_residual": _format_value(payload.get("hyper_output_head_residual")),
        "normalization_source": str(payload.get("normalization_source") or ""),
        "protocol_freeze_id": str(payload.get("protocol_freeze_id") or ""),
        "split_manifest_sha256": str(payload.get("split_manifest_sha256") or ""),
        "source_base_checkpoint_sha256": str(payload.get("source_base_checkpoint_sha256") or ""),
        "best_checkpoint": str(checkpoint) if checkpoint is not None else "",
        "summary_json": str(summary_path),
        "run_dir": str(run_dir),
    }


def _assign_ranks(rows: list[dict[str, Any]]) -> None:
    sortable: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        value = _to_float(row.get("best_selection_value"))
        if value is not None:
            sortable.append((value, row))
    sortable.sort(key=lambda item: item[0], reverse=True)
    for rank, (_, row) in enumerate(sortable, start=1):
        row["rank_by_best_selection_value"] = str(rank)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in FIELDNAMES} for row in rows])


def _write_markdown(path: Path, rows: list[dict[str, Any]], *, target_region: str, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "rank_by_best_selection_value",
        "ablation_id",
        "best_selection_value",
        "best_safe_score",
        "trainable_parameter_count",
        "hyper_coeff_generator",
        "hyper_adapter_param_style",
        "hyper_reliability_gate",
        "hyper_source_saliency_prior_beta",
        "hyper_source_saliency_prior_application",
        "zero_shot_prior_form",
        "hyper_residual_magnitude_penalty",
        "hyper_coeff_entropy_penalty",
        "hyper_prompt_manifold_reliability",
        "hyper_enable_film",
        "hyper_enable_adapters",
    ]
    lines = [
        f"# HyperDA staged ablation source-stage table: {target_region} seed={seed}",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_hyperda_staged_ablation_table(
    *,
    runs_root: Path,
    output_dir: Path,
    target_region: str,
    seed: int,
) -> dict[str, Path]:
    rows: list[dict[str, Any]] = []
    for ablation_id in ABLATION_ORDER:
        summary_path = _latest_summary_for_ablation(runs_root, ablation_id, target_region, seed)
        if summary_path is None:
            continue
        rows.append(
            _row_from_summary(
                ablation_id=ablation_id,
                summary_path=summary_path,
                target_region=target_region,
                seed=seed,
            )
        )

    _assign_ranks(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{target_region}_s{seed}_source_stage"
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows, target_region=target_region, seed=seed)
    json_path.write_text(
        json.dumps(
            {
                "target_region": target_region,
                "seed": seed,
                "runs_root": str(runs_root),
                "row_count": len(rows),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"csv_path": csv_path, "md_path": md_path, "json_path": json_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build staged HyperDA source-stage ablation table.")
    parser.add_argument("--runs_root", type=Path, default=Path("artifacts/runs/phase4_hyperda_staged_ablation"))
    parser.add_argument("--output_dir", type=Path, default=Path("reports/ablations/hyperda_staged_v1"))
    parser.add_argument("--target_region", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_hyperda_staged_ablation_table(
        runs_root=args.runs_root,
        output_dir=args.output_dir,
        target_region=args.target_region,
        seed=args.seed,
    )
    print(f"Wrote staged HyperDA ablation table: {result['csv_path']}")
    print(f"Wrote staged HyperDA ablation markdown: {result['md_path']}")
    print(f"Wrote staged HyperDA ablation json: {result['json_path']}")


if __name__ == "__main__":
    main()
