#!/usr/bin/env python3
"""Review a HyperDA zero/few-shot run overview for protocol and overfit signals."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"overview.csv not found: {path}")
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def _baseline_mentions_legacy(payload: Any) -> bool:
    text = json.dumps(payload, sort_keys=True) if not isinstance(payload, str) else payload
    return "target_full_train" in text or "v4.3" in text or "legacy" in text.lower()


def build_run_review(
    overview_path: Path,
    baseline_path: Optional[Path] = None,
) -> Dict[str, Any]:
    rows = _read_csv_rows(overview_path)
    k_rows = {str(row.get("K", "")): dict(row) for row in rows}
    warnings: List[str] = []

    k0_rmse = _float_or_none(k_rows.get("0", {}).get("surface_rmse_latw"))
    for k in ("4", "12"):
        row = k_rows.get(k)
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
    for k, row in sorted(review.get("k_rows", {}).items(), key=lambda item: int(item[0]) if item[0].isdigit() else 999):
        lines.append(
            "- "
            f"K={k}: setting={row.get('adaptation_setting', '')}, "
            f"steps={row.get('adaptation_steps', '')}, lr={row.get('lr', '')}, "
            f"alpha={row.get('anchor_alpha', '')}, surface_WRMSE={row.get('surface_rmse_latw', '')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review HyperDA zero/few-shot run artifacts.")
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--overview_csv", type=Path, default=None)
    parser.add_argument("--baseline_overview", type=Path, default=None)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_md", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.overview_csv is None:
        if args.run_dir is None:
            parser.error("provide --overview_csv or --run_dir")
        args.overview_csv = args.run_dir / "overview.csv"
    if args.output_json is None:
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
    write_markdown(args.output_md, review)
    print(f"Wrote run review JSON: {args.output_json}")
    print(f"Wrote run review MD: {args.output_md}")


if __name__ == "__main__":
    main()
