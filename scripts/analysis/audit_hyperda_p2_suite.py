#!/usr/bin/env python3
"""Audit HyperDA P0/P1/P2.6/P2.7/P2.8 Phase 5 artifacts.

This script is a read-only artifact auditor. It discovers existing run outputs,
normalizes their metadata and metrics, checks whether any selected P2.8 recipe
used forbidden target-side evidence, and writes a compact report bundle.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


FORBIDDEN_SELECTION_ROLES = {"target_eval", "target_query", "target_val", "target_full_train"}
SOURCE_SAFE_SELECTION_SOURCE = "source_val_pseudo_query_only"
SOURCE_SAFE_QUERY_ROLES = {"source_val", "source_val_pseudo_query", SOURCE_SAFE_SELECTION_SOURCE}
P1_SCOPES = ("prompt_only", "coeff_only", "gain_only", "coeff_gain", "all")
HASH_FIELDS = (
    "source_checkpoint_sha256",
    "split_manifest_sha256",
    "target_context_dates_hash",
    "target_eval_dates_hash",
)
OPTIONAL_HASH_FIELDS = ("prediction_content_hash", "metric_values_content_hash")

SUMMARY_COLUMNS = [
    "phase",
    "run_id",
    "run_path",
    "source_file",
    "evidence_status",
    "missing_fields",
    "target_region",
    "seed",
    "K",
    "scope",
    "schedule",
    "lr",
    "steps",
    "anchor_alpha",
    "trust_policy",
    "trust_region_mode",
    "trust_total_radius",
    "trust_prompt_radius",
    "trust_gain_radius",
    "trust_coeff_radius",
    "trust_spatial_radius",
    "adapt_mix_rho",
    "rho_policy",
    "support_loss_reduction",
    "support_loss_before",
    "support_loss_after",
    "support_final_loss",
    "support_loss_delta",
    "support_gradient_negative_fraction",
    "support_gradient_cosine_min",
    "surface_skill_primary",
    "rootzone_skill_primary",
    "overall_skill",
    "surface_rmse_latw",
    "rootzone_rmse_latw",
    "surface_corr_latw",
    "rootzone_corr_latw",
    "delta_vs_K0",
    "delta_vs_K4",
    "target_parameter_l2_drift_total",
    "target_parameter_l2_drift_target_prompt",
    "target_parameter_l2_drift_monthly_gain",
    "target_parameter_l2_drift_adapter_coeff_bottleneck",
    "target_parameter_l2_drift_adapter_coeff_dec2",
    "target_parameter_l2_drift_adapter_coeff_dec1",
    "source_checkpoint_sha256",
    "adapted_checkpoint_sha256",
    "split_manifest_sha256",
    "target_context_dates_hash",
    "target_support_dates_hash",
    "target_eval_dates_hash",
    "prediction_content_hash",
    "metric_values_content_hash",
    "checkpoint",
    "adapt_metadata",
    "summary",
    "metrics_long",
    "candidate_id",
    "selection_source",
    "query_role",
    "split_type",
    "adaptation_setting",
    "status",
    "notes",
]


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _first_present(row: Mapping[str, Any], keys: Sequence[str], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def _first_nested_value(payload: Any, paths: Sequence[Sequence[str]]) -> Any:
    if isinstance(payload, (list, tuple)):
        for item in payload:
            value = _first_nested_value(item, paths)
            if value not in (None, ""):
                return value
        return ""
    for path in paths:
        current = payload
        for key in path:
            if isinstance(current, Mapping):
                current = current.get(key)
            else:
                current = None
            if current in (None, ""):
                break
        if current in (None, ""):
            continue
        if isinstance(current, list):
            for item in current:
                if item not in (None, ""):
                    return item
            continue
        return current
    return ""


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _as_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _extract_timestamp(path: Path) -> str:
    match = re.search(r"20\d{6}T\d{6}Z", str(path))
    return match.group(0) if match else ""


def _latest(paths: Iterable[Path]) -> Path | None:
    items = list(paths)
    if not items:
        return None
    return max(items, key=lambda path: (_extract_timestamp(path), path.stat().st_mtime, str(path)))


def _path_matches_region_seed(path: Path, target_region: str, seed: int) -> bool:
    text = str(path)
    return target_region in text and f"s{seed}" in text


def _row_matches_region_seed(row: Mapping[str, Any], target_region: str, seed: int) -> bool:
    row_region = _first_present(row, ("target_region", "target_region_id", "final_target_region"), "")
    row_seed = _as_int(_first_present(row, ("seed", "support_seed"), ""))
    if row_region not in ("", target_region):
        return False
    return row_seed in (None, seed)


def _find_latest_file(
    runs_root: Path,
    phase_dirs: Sequence[str],
    filename: str,
    target_region: str,
    seed: int,
) -> Path | None:
    candidates: list[Path] = []
    for phase_dir in phase_dirs:
        root = runs_root / phase_dir
        if not root.exists():
            continue
        for path in root.rglob(filename):
            if _path_matches_region_seed(path, target_region, seed):
                candidates.append(path)
    return _latest(candidates)


def _csv_or_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return _read_csv_rows(path)
    payload = _read_json(path)
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("rows", "ranked_candidates", "candidates", "episodes"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, Mapping)]
        if isinstance(payload.get("top_candidate"), Mapping):
            return [dict(payload["top_candidate"])]
    return []


def _normal_metric(row: Mapping[str, Any], *keys: str) -> Any:
    return _first_present(row, keys, "")


def _derive_rho_policy(row: Mapping[str, Any]) -> str:
    policy = _first_present(row, ("rho_policy",), "")
    if policy:
        return _as_str(policy)
    rho = _as_float(_first_present(row, ("adapt_mix_rho",), ""))
    if rho is None:
        return ""
    text = ("%g" % rho).replace(".", "_")
    return f"fixed_{text}"


def _fill_p2_8_hash_fields(combined: dict[str, Any], summary: Mapping[str, Any], selected_config: Mapping[str, Any]) -> None:
    source_hash = _first_present(
        combined,
        ("source_checkpoint_sha256",),
        _first_nested_value(
            [combined, selected_config, summary],
            (
                ("source_metadata", "checkpoint_hashes"),
                ("calibration_audit", "checkpoint_hashes"),
                ("selected_guard_config", "source_metadata", "checkpoint_hashes"),
            ),
        ),
    )
    split_hash = _first_present(
        combined,
        ("split_manifest_sha256",),
        _first_nested_value(
            [combined, selected_config, summary],
            (
                ("source_metadata", "split_manifest_hashes"),
                ("calibration_audit", "split_manifest_hashes"),
                ("selected_guard_config", "source_metadata", "split_manifest_hashes"),
            ),
        ),
    )
    target_eval_hash = _first_present(
        combined,
        ("target_eval_dates_hash",),
        "not_used_source_val_query",
    )
    if source_hash:
        combined["source_checkpoint_sha256"] = source_hash
    if split_hash:
        combined["split_manifest_sha256"] = split_hash
    if target_eval_hash:
        combined["target_eval_dates_hash"] = target_eval_hash


def _normalize_row(
    *,
    phase: str,
    row: Mapping[str, Any],
    run_dir: Path | None,
    source_file: Path | None,
    target_region: str,
    seed: int,
    evidence_status: str = "present",
    notes: str = "",
) -> dict[str, str]:
    scope = _first_present(row, ("adapt_scope", "ADAPT_SCOPE", "scope"), "")
    if scope == "K0_base":
        scope = "all"
    run_id = _first_present(row, ("run_id", "candidate_run_id"), "")
    if not run_id and run_dir is not None:
        run_id = run_dir.name
    normalized: dict[str, Any] = {
        "phase": phase,
        "run_id": run_id,
        "run_path": str(run_dir) if run_dir is not None else "",
        "source_file": str(source_file) if source_file is not None else "",
        "evidence_status": evidence_status,
        "target_region": _first_present(row, ("target_region", "target_region_id", "final_target_region"), target_region),
        "seed": _first_present(row, ("seed", "support_seed"), seed),
        "K": _first_present(row, ("K",), ""),
        "scope": scope,
        "schedule": _first_present(row, ("schedule_label", "schedule"), ""),
        "lr": _first_present(row, ("lr", "requested_lr"), ""),
        "steps": _first_present(row, ("adaptation_steps", "requested_max_steps", "max_steps_requested"), ""),
        "anchor_alpha": _first_present(row, ("anchor_alpha", "requested_anchor_alpha"), ""),
        "trust_policy": _first_present(row, ("trust_policy",), ""),
        "trust_region_mode": _first_present(row, ("trust_region_mode",), ""),
        "trust_total_radius": _first_present(row, ("trust_total_radius", "trust_radii_total"), ""),
        "trust_prompt_radius": _first_present(row, ("trust_prompt_radius", "trust_radii_prompt"), ""),
        "trust_gain_radius": _first_present(row, ("trust_gain_radius", "trust_radii_gain"), ""),
        "trust_coeff_radius": _first_present(row, ("trust_coeff_radius", "trust_radii_coeff"), ""),
        "trust_spatial_radius": _first_present(row, ("trust_spatial_radius", "trust_radii_spatial"), ""),
        "adapt_mix_rho": _first_present(row, ("adapt_mix_rho",), ""),
        "rho_policy": _first_present(row, ("rho_policy",), ""),
        "support_loss_reduction": _first_present(row, ("support_loss_reduction",), ""),
        "support_loss_before": _normal_metric(row, "support_loss_before", "standard_support_loss_before_full_support"),
        "support_loss_after": _normal_metric(row, "support_loss_after", "standard_support_loss_after_full_support"),
        "support_final_loss": _normal_metric(row, "support_final_loss", "support_loss_after", "standard_support_loss_after_full_support"),
        "support_loss_delta": _normal_metric(row, "support_loss_delta", "standard_support_loss_delta_full_support"),
        "support_gradient_negative_fraction": _first_present(row, ("support_gradient_negative_fraction",), ""),
        "support_gradient_cosine_min": _first_present(row, ("support_gradient_cosine_min",), ""),
        "surface_skill_primary": _normal_metric(row, "surface_skill_primary", "eval_skill_surface"),
        "rootzone_skill_primary": _normal_metric(row, "rootzone_skill_primary", "eval_skill_rootzone"),
        "overall_skill": _normal_metric(row, "overall_skill", "eval_skill_overall"),
        "surface_rmse_latw": _normal_metric(row, "surface_rmse_latw", "surface_rmse_latw_mean"),
        "rootzone_rmse_latw": _normal_metric(row, "rootzone_rmse_latw", "rootzone_rmse_latw_mean"),
        "surface_corr_latw": _normal_metric(row, "surface_corr_latw", "surface_corr_latw_mean"),
        "rootzone_corr_latw": _normal_metric(row, "rootzone_corr_latw", "rootzone_corr_latw_mean"),
        "delta_vs_K0": _normal_metric(row, "delta_vs_K0", "skill_delta_vs_K0"),
        "delta_vs_K4": _normal_metric(row, "delta_vs_K4", "delta_vs_K4_original"),
        "target_parameter_l2_drift_total": _normal_metric(
            row,
            "target_parameter_l2_drift_total",
            "target_parameter_l2_drift_post_anchor_total",
            "drift_total",
        ),
        "target_parameter_l2_drift_target_prompt": _normal_metric(
            row,
            "target_parameter_l2_drift_target_prompt",
            "target_parameter_l2_drift_post_anchor_target_prompt",
            "drift_target_prompt",
        ),
        "target_parameter_l2_drift_monthly_gain": _normal_metric(
            row,
            "target_parameter_l2_drift_monthly_gain",
            "target_parameter_l2_drift_post_anchor_monthly_gain",
            "drift_monthly_gain",
        ),
        "target_parameter_l2_drift_adapter_coeff_bottleneck": _normal_metric(
            row,
            "target_parameter_l2_drift_adapter_coeff_bottleneck",
            "target_parameter_l2_drift_post_anchor_adapter_coeff_bottleneck",
            "drift_coeff_b",
        ),
        "target_parameter_l2_drift_adapter_coeff_dec2": _normal_metric(
            row,
            "target_parameter_l2_drift_adapter_coeff_dec2",
            "target_parameter_l2_drift_post_anchor_adapter_coeff_dec2",
            "drift_coeff_d2",
        ),
        "target_parameter_l2_drift_adapter_coeff_dec1": _normal_metric(
            row,
            "target_parameter_l2_drift_adapter_coeff_dec1",
            "target_parameter_l2_drift_post_anchor_adapter_coeff_dec1",
            "drift_coeff_d1",
        ),
        "source_checkpoint_sha256": _first_present(row, ("source_checkpoint_sha256",), ""),
        "adapted_checkpoint_sha256": _first_present(row, ("adapted_checkpoint_sha256",), ""),
        "split_manifest_sha256": _first_present(row, ("split_manifest_sha256",), ""),
        "target_context_dates_hash": _first_present(row, ("target_context_dates_hash",), ""),
        "target_support_dates_hash": _first_present(row, ("target_support_dates_hash", "support_dates_hash"), ""),
        "target_eval_dates_hash": _first_present(row, ("target_eval_dates_hash",), ""),
        "prediction_content_hash": _first_present(
            row,
            ("prediction_content_hash", "final_mixed_prediction_content_hash", "mixed_prediction_content_hash"),
            "",
        ),
        "metric_values_content_hash": _first_present(row, ("metric_values_content_hash", "metric_row_content_hash"), ""),
        "checkpoint": _first_present(row, ("checkpoint",), ""),
        "adapt_metadata": _first_present(row, ("adapt_metadata",), ""),
        "summary": _first_present(row, ("summary",), ""),
        "metrics_long": _first_present(row, ("metrics_long",), ""),
        "candidate_id": _first_present(row, ("candidate_id",), ""),
        "selection_source": _first_present(row, ("selection_source",), ""),
        "query_role": _first_present(row, ("query_role",), ""),
        "split_type": _first_present(row, ("split_type", "split_role"), ""),
        "adaptation_setting": _first_present(row, ("adaptation_setting",), ""),
        "status": _first_present(row, ("status",), ""),
        "notes": notes,
    }
    normalized["rho_policy"] = _derive_rho_policy(normalized)

    required_hash_fields = ("source_checkpoint_sha256", "split_manifest_sha256", "target_context_dates_hash", "target_eval_dates_hash")
    if phase == "P2.8":
        required_hash_fields = tuple(field for field in required_hash_fields if field != "target_context_dates_hash")
    missing_fields = [field for field in required_hash_fields if not normalized.get(field)]
    normalized["missing_fields"] = ";".join(missing_fields)

    return {column: _as_str(normalized.get(column, "")) for column in SUMMARY_COLUMNS}


def _metadata_from_audit(audit: Mapping[str, Any], key: str) -> dict[str, Any]:
    path_text = audit.get(key)
    if not path_text:
        return {}
    path = Path(str(path_text))
    if not path.exists():
        return {}
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _hash_match(audit: Mapping[str, Any], key: str) -> tuple[bool, str, str]:
    matches = audit.get("hash_matches")
    if not isinstance(matches, Mapping) or key not in matches:
        return False, "", ""
    item = matches.get(key)
    if not isinstance(item, Mapping):
        return False, "", ""
    return bool(item.get("match")), _as_str(item.get("K0", "")), _as_str(item.get("K12", ""))


def _max_drift_from_metadata(metadata: Mapping[str, Any]) -> float | None:
    drift = metadata.get("target_parameter_l2_drift")
    if isinstance(drift, Mapping):
        values = [_as_float(value) for value in drift.values()]
        present = [value for value in values if value is not None]
        if present:
            return max(abs(value) for value in present)
    scalar = _as_float(metadata.get("target_parameter_l2_drift_total"))
    return abs(scalar) if scalar is not None else None


def _audit_p0_identity(
    runs_root: Path,
    target_region: str,
    seed: int,
) -> tuple[list[dict[str, str]], dict[str, Any], list[str]]:
    missing: list[str] = []
    identity_path = None
    for phase_dirs in (
        ("phase5_hyperda_identity_gate",),
        ("phase5_hyperda_zero_few_shot_eval", "phase5_hyperda_broad_gate_smoke", "phase5_hyperda_broad_gate"),
    ):
        identity_path = _find_latest_file(
            runs_root,
            phase_dirs,
            "identity_audit.json",
            target_region,
            seed,
        )
        if identity_path is not None:
            break
    if identity_path is None:
        missing.append("P0 identity_audit.json not found for requested target region and seed.")
        return [], {"passed": False, "run_path": "", "reasons": ["missing identity_audit.json"]}, missing

    audit = _read_json(identity_path)
    if not isinstance(audit, Mapping):
        missing.append(f"P0 identity_audit.json is not an object: {identity_path}")
        return [], {"passed": False, "run_path": str(identity_path.parent), "reasons": ["invalid identity_audit.json"]}, missing

    k0_metadata = _metadata_from_audit(audit, "K0_metadata")
    k12_metadata = _metadata_from_audit(audit, "K12_metadata")
    checks = audit.get("k12_identity_checks")
    checks = checks if isinstance(checks, Mapping) else {}
    tolerance = _as_float(audit.get("tolerance")) or 1e-8
    max_metric_diff = _as_float(audit.get("max_abs_metric_diff"))

    reasons: list[str] = []
    if audit.get("status") != "passed":
        reasons.append(f"identity_audit.status={audit.get('status')}")
    if max_metric_diff is None or max_metric_diff > tolerance:
        reasons.append("identity metric diff missing or above tolerance")

    for key in ("source_checkpoint_sha256", "split_manifest_sha256", "target_context_dates_hash", "target_eval_dates_hash"):
        match, _k0_value, _k12_value = _hash_match(audit, key)
        if not match:
            reasons.append(f"{key} missing or mismatch")
    hash_matches = audit.get("hash_matches")
    if isinstance(hash_matches, Mapping):
        for key, item in hash_matches.items():
            if ("prediction" in str(key) or "metric" in str(key)) and isinstance(item, Mapping):
                if item.get("match") is False:
                    reasons.append(f"{key} mismatch")

    labels_loaded = _as_bool(_first_present(checks, ("target_labels_loaded_for_adaptation",), ""))
    labels_used = _as_bool(_first_present(checks, ("target_labels_used_for_adaptation",), ""))
    if labels_loaded is None:
        labels_loaded = _as_bool(k12_metadata.get("target_labels_loaded_for_adaptation"))
    if labels_used is None:
        labels_used = _as_bool(k12_metadata.get("target_labels_used_for_adaptation"))
    if labels_loaded:
        reasons.append("K12 identity loaded target labels")
    if labels_used:
        reasons.append("K12 identity used target labels")

    optimizer_steps = _as_int(_first_present(checks, ("actual_optimizer_steps", "optimizer_steps_run"), ""))
    if optimizer_steps is None:
        optimizer_steps = _as_int(_first_present(k12_metadata, ("actual_optimizer_steps", "optimizer_steps_run", "adaptation_steps"), ""))
    if optimizer_steps not in (0, None):
        reasons.append(f"K12 identity optimizer steps={optimizer_steps}")

    max_drift = _as_float(checks.get("max_target_parameter_l2_drift"))
    if max_drift is None:
        max_drift = _max_drift_from_metadata(k12_metadata)
    if max_drift is None:
        reasons.append("K12 identity drift missing")
    elif max_drift > tolerance:
        reasons.append(f"K12 identity drift={max_drift} above tolerance={tolerance}")

    source_match, source_k0, _source_k12 = _hash_match(audit, "source_checkpoint_sha256")
    split_match, split_k0, _split_k12 = _hash_match(audit, "split_manifest_sha256")
    context_match, context_k0, _context_k12 = _hash_match(audit, "target_context_dates_hash")
    eval_match, eval_k0, _eval_k12 = _hash_match(audit, "target_eval_dates_hash")
    pred_match, pred_k0, _pred_k12 = _hash_match(audit, "prediction_content_hash")
    metric_match, metric_k0, _metric_k12 = _hash_match(audit, "metric_values_content_hash")
    row = {
        "run_id": identity_path.parent.name,
        "target_region": target_region,
        "seed": seed,
        "K": "0/12",
        "adapt_scope": "identity",
        "status": "passed" if not reasons else "failed",
        "source_checkpoint_sha256": source_k0 if source_match else source_k0,
        "split_manifest_sha256": split_k0 if split_match else split_k0,
        "target_context_dates_hash": context_k0 if context_match else context_k0,
        "target_eval_dates_hash": eval_k0 if eval_match else eval_k0,
        "prediction_content_hash": pred_k0 if pred_match else pred_k0,
        "metric_values_content_hash": metric_k0 if metric_match else metric_k0,
        "checkpoint": _first_present(k12_metadata, ("checkpoint",), ""),
        "adapt_metadata": _as_str(audit.get("K12_metadata", "")),
        "summary": _as_str(audit.get("K12_summary", "")),
    }
    normalized = _normalize_row(
        phase="P0",
        row=row,
        run_dir=identity_path.parent,
        source_file=identity_path,
        target_region=target_region,
        seed=seed,
        evidence_status="passed" if not reasons else "failed",
        notes="; ".join(reasons),
    )
    p0 = {
        "passed": not reasons,
        "run_path": str(identity_path.parent),
        "identity_audit_path": str(identity_path),
        "source_checkpoint_sha256": row["source_checkpoint_sha256"],
        "split_manifest_sha256": row["split_manifest_sha256"],
        "target_context_dates_hash": row["target_context_dates_hash"],
        "target_eval_dates_hash": row["target_eval_dates_hash"],
        "prediction_content_hash": row["prediction_content_hash"],
        "metric_values_content_hash": row["metric_values_content_hash"],
        "reasons": reasons,
    }
    if normalized["missing_fields"]:
        missing.append(f"P0 missing hash fields: {normalized['missing_fields']} ({identity_path})")
    return [normalized], p0, missing


def _load_matching_csv_rows(path: Path, target_region: str, seed: int) -> list[dict[str, Any]]:
    rows = _read_csv_rows(path)
    return [row for row in rows if _row_matches_region_seed(row, target_region, seed)]


def _run_dir_for_file(path: Path) -> Path:
    return path.parent


def _audit_p1_scope(
    runs_root: Path,
    target_region: str,
    seed: int,
) -> tuple[list[dict[str, str]], dict[str, Any], list[str]]:
    missing: list[str] = []
    scope_candidates: dict[str, tuple[Path, dict[str, Any]]] = {}
    phase_roots = [runs_root / "phase5_hyperda_scope_gate", runs_root / "phase5_hyperda_broad_gate_smoke", runs_root / "phase5_hyperda_broad_gate"]
    for root in phase_roots:
        if not root.exists():
            continue
        for csv_path in root.rglob("overview.csv"):
            if not _path_matches_region_seed(csv_path, target_region, seed):
                continue
            for row in _load_matching_csv_rows(csv_path, target_region, seed):
                scope = _first_present(row, ("adapt_scope", "ADAPT_SCOPE"), "")
                if scope in P1_SCOPES:
                    if _as_bool(_first_present(row, ("freeze_monthly_gain", "FREEZE_MONTHLY_GAIN"), "")):
                        continue
                    current = scope_candidates.get(scope)
                    if current is None or (_extract_timestamp(csv_path), str(csv_path)) > (_extract_timestamp(current[0]), str(current[0])):
                        scope_candidates[scope] = (csv_path, row)
        summary_csv = root / "summary" / "post_p0_p1_gate_summary.csv"
        if summary_csv.exists():
            for row in _load_matching_csv_rows(summary_csv, target_region, seed):
                scope = _first_present(row, ("adapt_scope", "ADAPT_SCOPE"), "")
                if scope in P1_SCOPES:
                    if _as_bool(_first_present(row, ("freeze_monthly_gain", "FREEZE_MONTHLY_GAIN"), "")):
                        continue
                    current = scope_candidates.get(scope)
                    if current is None:
                        scope_candidates[scope] = (summary_csv, row)

    rows: list[dict[str, str]] = []
    for scope in P1_SCOPES:
        candidate = scope_candidates.get(scope)
        if candidate is None:
            missing.append(f"P1 scope row missing for scope={scope}.")
            continue
        source_file, row = candidate
        rows.append(
            _normalize_row(
                phase="P1",
                row=row,
                run_dir=_run_dir_for_file(source_file) if source_file.name == "overview.csv" else Path(str(row.get("artifact_dir", source_file.parent))),
                source_file=source_file,
                target_region=target_region,
                seed=seed,
            )
        )

    def rank_key(row: dict[str, str]) -> tuple[float, float, float]:
        delta = _as_float(row.get("delta_vs_K0")) or 0.0
        drift = _as_float(row.get("target_parameter_l2_drift_total")) or 0.0
        conflict = _as_float(row.get("support_gradient_negative_fraction")) or 0.0
        return (delta, -conflict, -drift)

    ranked = sorted(rows, key=rank_key, reverse=True)
    analysis = {
        "rows_found": len(rows),
        "missing_scopes": [scope for scope in P1_SCOPES if scope not in scope_candidates],
        "most_stable_scope": ranked[0]["scope"] if ranked else "",
        "scope_ranking": [row["scope"] for row in ranked],
    }
    return rows, analysis, missing


def _latest_summary_pair(runs_root: Path, phase_dir: str, stem: str, target_region: str, seed: int) -> tuple[Path | None, Path | None]:
    root = runs_root / phase_dir
    if not root.exists():
        return None, None
    dirs = [path for path in root.iterdir() if path.is_dir() and _path_matches_region_seed(path, target_region, seed)]
    latest_dir = _latest(dirs)
    if latest_dir is None:
        return None, None
    csv_path = latest_dir / f"{stem}.csv"
    json_path = latest_dir / f"{stem}.json"
    return (csv_path if csv_path.exists() else None, json_path if json_path.exists() else None)


def _audit_summary_phase(
    *,
    runs_root: Path,
    phase: str,
    phase_dir: str,
    stem: str,
    target_region: str,
    seed: int,
) -> tuple[list[dict[str, str]], Path | None, list[str]]:
    csv_path, json_path = _latest_summary_pair(runs_root, phase_dir, stem, target_region, seed)
    missing: list[str] = []
    source_path = json_path or csv_path
    if source_path is None:
        missing.append(f"{phase} {stem}.csv|json not found for requested target region and seed.")
        return [], None, missing
    rows = _csv_or_json_rows(source_path)
    normalized = [
        _normalize_row(
            phase=phase,
            row=row,
            run_dir=source_path.parent,
            source_file=source_path,
            target_region=target_region,
            seed=seed,
        )
        for row in rows
        if _row_matches_region_seed(row, target_region, seed)
    ]
    if not normalized:
        missing.append(f"{phase} summary has no matching rows: {source_path}")
    return normalized, source_path.parent, missing


def _overall(row: Mapping[str, str]) -> float | None:
    value = _as_float(row.get("overall_skill"))
    if value is not None:
        return value
    surface = _as_float(row.get("surface_skill_primary"))
    rootzone = _as_float(row.get("rootzone_skill_primary"))
    if surface is not None and rootzone is not None:
        return 0.5 * (surface + rootzone)
    return None


def _find_row(rows: Sequence[dict[str, str]], *, k: int | None = None, schedule_contains: str | None = None, scope: str | None = None) -> dict[str, str] | None:
    for row in rows:
        if k is not None and _as_int(row.get("K")) != k:
            continue
        if schedule_contains is not None and schedule_contains not in row.get("schedule", ""):
            continue
        if scope is not None and row.get("scope") != scope:
            continue
        return row
    return None


def _diagnose_p2_6(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {"k12_instability_source": "unresolved", "reason": "P2.6 evidence missing"}
    k0 = _find_row(rows, k=0)
    k4 = _find_row(rows, k=4, schedule_contains="original_K4")
    k12 = _find_row(rows, k=12, schedule_contains="original_K12", scope="all") or _find_row(rows, k=12, schedule_contains="original_K12")
    k12_k4_schedule = _find_row(rows, k=12, schedule_contains="K4_schedule_on_K12")
    if k12 is None:
        return {"k12_instability_source": "unresolved", "reason": "P2.6 missing original K12 row"}

    k12_score = _overall(k12)
    k4_score = _overall(k4) if k4 is not None else None
    k0_score = _overall(k0) if k0 is not None else None
    k12_k4_score = _overall(k12_k4_schedule) if k12_k4_schedule is not None else None
    if k12_score is not None and k12_k4_score is not None and k12_k4_score > k12_score + 1e-6:
        return {
            "k12_instability_source": "schedule",
            "reason": "K12 improved under K4 schedule/lr/step settings.",
            "k12_original_overall": k12_score,
            "k12_k4_schedule_overall": k12_k4_score,
        }

    conflict = _as_float(k12.get("support_gradient_negative_fraction")) or 0.0
    cosine_min = _as_float(k12.get("support_gradient_cosine_min")) or 0.0
    if conflict > 0.45 or cosine_min < -0.2:
        return {
            "k12_instability_source": "support conflict",
            "reason": "K12 support-gradient diagnostics indicate conflicting support cycles.",
            "support_gradient_negative_fraction": conflict,
            "support_gradient_cosine_min": cosine_min,
        }

    drift = _as_float(k12.get("target_parameter_l2_drift_total")) or _as_float(k12.get("target_parameter_l2_drift_target_prompt")) or 0.0
    support_delta = _as_float(k12.get("support_loss_delta"))
    support_delta = support_delta if support_delta is not None else _as_float(k12.get("support_loss_after"))
    worse_than_k4 = k4_score is not None and k12_score is not None and k12_score < k4_score
    worse_than_k0 = k0_score is not None and k12_score is not None and k12_score < k0_score
    if drift > 0.25 and (worse_than_k4 or worse_than_k0):
        return {
            "k12_instability_source": "over-adaptation",
            "reason": "K12 drift is nontrivial while evaluation skill degrades.",
            "drift": drift,
            "support_loss_delta": support_delta,
        }
    return {"k12_instability_source": "unresolved", "reason": "P2.6 diagnostics do not isolate schedule, drift, or support conflict."}


def _analyze_p2_7(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {"guard_reduces_negative_migration": False, "reason": "P2.7 evidence missing"}
    k0 = _find_row(rows, k=0)
    unguarded = _find_row(rows, k=12, schedule_contains="original", scope="all")
    guarded = [
        row
        for row in rows
        if _as_int(row.get("K")) == 12
        and (
            row.get("trust_region_mode") not in ("", "none")
            or row.get("trust_policy") not in ("", "none")
            or (_as_float(row.get("adapt_mix_rho")) is not None and (_as_float(row.get("adapt_mix_rho")) or 0.0) < 1.0)
        )
    ]
    if not guarded or unguarded is None:
        return {"guard_reduces_negative_migration": False, "reason": "Guarded or unguarded K12 rows missing."}
    best_guarded = max(guarded, key=lambda row: _overall(row) if _overall(row) is not None else -999.0)
    baseline = _overall(k0) if k0 is not None else None
    unguarded_score = _overall(unguarded)
    guarded_score = _overall(best_guarded)
    reduced = guarded_score is not None and unguarded_score is not None and guarded_score > unguarded_score
    return {
        "guard_reduces_negative_migration": bool(reduced),
        "reason": "Best guarded K12 improves over unguarded K12." if reduced else "Best guarded K12 does not improve over unguarded K12.",
        "k0_overall": baseline,
        "unguarded_k12_overall": unguarded_score,
        "best_guarded_k12_overall": guarded_score,
        "best_guarded_run_id": best_guarded.get("run_id", ""),
    }


def _load_p2_8_summary_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    top = summary.get("top_candidate")
    if isinstance(top, Mapping):
        rows.append(dict(top))
    selected = summary.get("selected_guard_config")
    if isinstance(selected, Mapping):
        rows.append(dict(selected))
    ranked = summary.get("ranked_candidates")
    if isinstance(ranked, list):
        rows.extend(dict(row) for row in ranked if isinstance(row, Mapping))
    rows_payload = summary.get("rows")
    if isinstance(rows_payload, list):
        rows.extend(dict(row) for row in rows_payload if isinstance(row, Mapping))
    return rows


def _find_p2_8_dirs(runs_root: Path, target_region: str, seed: int) -> list[Path]:
    root = runs_root / "phase5_hyperda_p2_8_source_safe_guard_calibration"
    if not root.exists():
        return []
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and _path_matches_region_seed(path, target_region, seed)],
        key=lambda path: (_extract_timestamp(path), path.stat().st_mtime, str(path)),
    )


def _audit_p2_8(
    runs_root: Path,
    target_region: str,
    seed: int,
) -> tuple[list[dict[str, str]], dict[str, Any], list[str]]:
    missing: list[str] = []
    dirs = _find_p2_8_dirs(runs_root, target_region, seed)
    if not dirs:
        missing.append("P2.8 source-safe calibration directory not found.")
        return [], {"status": "missing", "evidence_complete": False, "run_path": "", "selection_rows": []}, missing

    complete_dirs = [
        path
        for path in dirs
        if (path / "source_safe_calibration_summary.json").exists() and (path / "selected_guard_config.yaml").exists()
    ]
    run_dir = complete_dirs[-1] if complete_dirs else dirs[-1]
    summary_path = run_dir / "source_safe_calibration_summary.json"
    config_path = run_dir / "selected_guard_config.yaml"
    top5_csv = run_dir / "top5_stability.csv"
    top5_json = run_dir / "top5_stability.json"
    loso_csv = run_dir / "leave_one_source_region_out_stability.csv"

    if not summary_path.exists() or not config_path.exists():
        for name in ("source_safe_calibration_summary.json", "selected_guard_config.yaml"):
            if not (run_dir / name).exists():
                missing.append(f"P2.8 missing {name}: {run_dir}")
        if not ((run_dir / "top5_stability.csv").exists() or (run_dir / "top5_stability.json").exists()):
            missing.append(f"P2.8 missing top5_stability.csv/json: {run_dir}")
        if not (run_dir / "leave_one_source_region_out_stability.csv").exists():
            missing.append(f"P2.8 missing leave-one-source-region-out stability CSV: {run_dir}")
        for name in ("candidate_manifest.csv", "completed_rows.csv", "missing_rows.csv", "invalid_existing_rows.csv"):
            if (run_dir / name).exists():
                missing.append(f"P2.8 manifest-only evidence present: {run_dir / name}")
        return [], {
            "status": "evidence_incomplete",
            "evidence_complete": False,
            "run_path": str(run_dir),
            "summary_path": str(summary_path) if summary_path.exists() else "",
            "selected_guard_config_path": str(config_path) if config_path.exists() else "",
            "selection_rows": [],
            "selected_config": {},
        }, missing

    summary = _read_json(summary_path)
    if not isinstance(summary, Mapping):
        missing.append(f"P2.8 source_safe_calibration_summary.json is not an object: {summary_path}")
        return [], {"status": "invalid", "evidence_complete": False, "run_path": str(run_dir), "selection_rows": []}, missing
    selected_config = _read_yaml(config_path)
    selected_config = selected_config if isinstance(selected_config, Mapping) else {}
    rows = _load_p2_8_summary_rows(summary)
    top_candidate = dict(summary.get("top_candidate", {})) if isinstance(summary.get("top_candidate"), Mapping) else {}
    summary_selected = dict(summary.get("selected_guard_config", {})) if isinstance(summary.get("selected_guard_config"), Mapping) else {}
    combined = {**top_candidate, **selected_config, **summary_selected}
    combined.setdefault("selection_source", summary.get("selection_source", selected_config.get("selection_source", "")))
    combined.setdefault("target_region", target_region)
    combined.setdefault("seed", seed)
    _fill_p2_8_hash_fields(combined, summary, selected_config)
    rows_for_leakage = [combined] + rows

    evidence_complete = True
    if not (top5_csv.exists() or top5_json.exists()):
        evidence_complete = False
        missing.append(f"P2.8 missing top5_stability.csv/json: {run_dir}")
    if not loso_csv.exists():
        evidence_complete = False
        missing.append(f"P2.8 missing leave-one-source-region-out stability CSV: {run_dir}")

    normalized_rows = [
        _normalize_row(
            phase="P2.8",
            row=combined,
            run_dir=run_dir,
            source_file=summary_path,
            target_region=target_region,
            seed=seed,
            evidence_status="complete" if evidence_complete else "evidence_incomplete",
            notes="source-safe selected candidate",
        )
    ]
    audit = {
        "status": "complete" if evidence_complete else "evidence_incomplete",
        "evidence_complete": evidence_complete,
        "run_path": str(run_dir),
        "summary_path": str(summary_path),
        "selected_guard_config_path": str(config_path),
        "top5_stability_path": str(top5_csv if top5_csv.exists() else top5_json) if (top5_csv.exists() or top5_json.exists()) else "",
        "leave_one_source_region_out_stability_path": str(loso_csv) if loso_csv.exists() else "",
        "summary": summary,
        "selected_config": dict(selected_config),
        "top_candidate": combined,
        "selection_rows": rows_for_leakage,
    }
    return normalized_rows, audit, missing


def _new_leakage_checks() -> dict[str, dict[str, Any]]:
    return {
        "target_eval_metrics_not_used": {"status": "pass", "evidence": []},
        "target_eval_labels_not_used": {"status": "pass", "evidence": []},
        "target_eval_prediction_records_not_used": {"status": "pass", "evidence": []},
        "target_val_not_used": {"status": "pass", "evidence": []},
        "target_full_train_not_used": {"status": "pass", "evidence": []},
    }


def _row_forbidden_hits(row: Mapping[str, Any]) -> list[str]:
    hits: list[str] = []
    for key in ("split_type", "split_role", "query_role", "adaptation_setting"):
        value = str(row.get(key, "")).strip()
        if value in FORBIDDEN_SELECTION_ROLES:
            hits.append(value)
    return hits


def _build_leakage_audit(p2_8: Mapping[str, Any]) -> dict[str, Any]:
    checks = _new_leakage_checks()
    offending_rows: list[dict[str, Any]] = []
    for row in p2_8.get("selection_rows", []):
        if not isinstance(row, Mapping):
            continue
        hits = _row_forbidden_hits(row)
        if not hits:
            continue
        offender = {
            "candidate_id": _as_str(row.get("candidate_id", "")),
            "split_type": _as_str(_first_present(row, ("split_type", "split_role"), "")),
            "query_role": _as_str(row.get("query_role", "")),
            "adaptation_setting": _as_str(row.get("adaptation_setting", "")),
            "hits": hits,
        }
        offending_rows.append(offender)
        if "target_eval" in hits or "target_query" in hits:
            for key in (
                "target_eval_metrics_not_used",
                "target_eval_labels_not_used",
                "target_eval_prediction_records_not_used",
            ):
                checks[key]["status"] = "fail"
                checks[key]["evidence"].append(offender)
        if "target_val" in hits:
            checks["target_val_not_used"]["status"] = "fail"
            checks["target_val_not_used"]["evidence"].append(offender)
        if "target_full_train" in hits:
            checks["target_full_train_not_used"]["status"] = "fail"
            checks["target_full_train_not_used"]["evidence"].append(offender)

    selected_source = _as_str(_first_present(p2_8.get("top_candidate", {}), ("selection_source",), ""))
    if p2_8.get("status") == "complete" and selected_source not in ("", SOURCE_SAFE_SELECTION_SOURCE):
        offender = {"selection_source": selected_source, "expected": SOURCE_SAFE_SELECTION_SOURCE}
        for key in (
            "target_eval_metrics_not_used",
            "target_eval_labels_not_used",
            "target_eval_prediction_records_not_used",
        ):
            checks[key]["status"] = "fail"
            checks[key]["evidence"].append(offender)
        offending_rows.append(offender)

    leakage_fail = any(check["status"] == "fail" for check in checks.values())
    evidence_complete = bool(p2_8.get("evidence_complete"))
    if leakage_fail:
        verdict = "leakage_fail"
    elif not evidence_complete:
        verdict = "evidence_incomplete"
    else:
        verdict = "pass"
    return {
        "verdict": verdict,
        "evidence_status": "complete" if evidence_complete else "evidence_incomplete",
        "selection_source_required": SOURCE_SAFE_SELECTION_SOURCE,
        "p2_8_run_path": p2_8.get("run_path", ""),
        "checks": checks,
        "offending_rows": offending_rows,
    }


def _missing_hash_notes(rows: Sequence[dict[str, str]]) -> list[str]:
    notes: list[str] = []
    for row in rows:
        required = tuple(field for field in HASH_FIELDS if not (row.get("phase") == "P2.8" and field == "target_context_dates_hash"))
        missing = [field for field in required if not row.get(field)]
        if missing:
            notes.append(
                f"{row.get('phase')} {row.get('run_id')} missing hashes: {', '.join(missing)}"
            )
    return notes


def _selected_seed_scope(p2_8: Mapping[str, Any]) -> str:
    summary = p2_8.get("summary")
    if isinstance(summary, Mapping):
        audit = summary.get("calibration_audit")
        if isinstance(audit, Mapping):
            seeds = audit.get("seeds")
            if isinstance(seeds, list) and set(_as_int(seed) for seed in seeds) >= {0, 1, 2}:
                return "seeds 0,1,2"
    return "seed 0 only"


def _build_recommendation(
    *,
    target_region: str,
    seed: int,
    p0: Mapping[str, Any],
    p2_8: Mapping[str, Any],
    leakage: Mapping[str, Any],
    final_conclusion: str,
) -> dict[str, Any]:
    top = p2_8.get("top_candidate")
    top = top if isinstance(top, Mapping) else {}
    blocked_reasons: list[str] = []
    if not p0.get("passed"):
        blocked_reasons.append("fix pipeline before target_eval")
        blocked_reasons.extend(_as_str(reason) for reason in p0.get("reasons", []))
    if leakage.get("verdict") == "leakage_fail":
        blocked_reasons.append("leakage_fail: selected P2.8 rows/config reference forbidden target-side roles")
    if not p2_8.get("evidence_complete"):
        blocked_reasons.append("evidence_incomplete: P2.8 source-safe calibration/stability evidence is incomplete")

    locked_eval_ready = bool(p0.get("passed")) and leakage.get("verdict") == "pass" and bool(p2_8.get("evidence_complete"))
    forbidden_checklist = {
        key: {"status": value.get("status", "fail"), "evidence_count": len(value.get("evidence", []))}
        for key, value in leakage.get("checks", {}).items()
        if isinstance(value, Mapping)
    }
    return {
        "target_region": target_region,
        "seed": seed,
        "locked_eval_ready": locked_eval_ready,
        "final_conclusion": final_conclusion,
        "blocked_reasons": blocked_reasons,
        "selection_source": SOURCE_SAFE_SELECTION_SOURCE,
        "selected_guard_config_path": _as_str(p2_8.get("selected_guard_config_path", "")) if locked_eval_ready else _as_str(p2_8.get("selected_guard_config_path", "")),
        "seed_scope": _selected_seed_scope(p2_8),
        "candidate_id": _as_str(top.get("candidate_id", "")),
        "K": _as_int(top.get("K")),
        "adapt_scope": _as_str(_first_present(top, ("adapt_scope", "scope"), "")),
        "schedule_label": _as_str(_first_present(top, ("schedule_label", "schedule"), "")),
        "lr": _as_float(top.get("lr")),
        "adaptation_steps": _as_int(_first_present(top, ("adaptation_steps", "steps"), "")),
        "anchor_alpha": _as_float(top.get("anchor_alpha")),
        "support_loss_reduction": _as_str(top.get("support_loss_reduction", "")),
        "trust_policy": _as_str(_first_present(top, ("trust_policy", "trust_region_mode"), "")),
        "rho_policy": _derive_rho_policy(top),
        "adapt_mix_rho": _as_float(top.get("adapt_mix_rho")),
        "source_checkpoint_sha256": _as_str(top.get("source_checkpoint_sha256", "")),
        "split_manifest_sha256": _as_str(top.get("split_manifest_sha256", "")),
        "target_context_dates_hash": _as_str(top.get("target_context_dates_hash", "")),
        "target_eval_dates_hash": _as_str(top.get("target_eval_dates_hash", "")),
        "forbidden_input_checklist": forbidden_checklist,
    }


def _final_conclusion(p0: Mapping[str, Any], p2_8: Mapping[str, Any], leakage: Mapping[str, Any]) -> str:
    if not p0.get("passed") or leakage.get("verdict") == "leakage_fail":
        return "pipeline failed"
    if not p2_8.get("evidence_complete"):
        return "suspicious"
    return "pipeline clean"


def _write_summary_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def _write_missing_report(path: Path, missing_items: Sequence[str]) -> None:
    lines = ["# Missing Or Invalid Artifacts", ""]
    if missing_items:
        lines.extend(f"- {item}" for item in missing_items)
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown_summary(
    path: Path,
    *,
    target_region: str,
    seed: int,
    p0: Mapping[str, Any],
    p1: Mapping[str, Any],
    p2_6: Mapping[str, Any],
    p2_7: Mapping[str, Any],
    p2_8: Mapping[str, Any],
    leakage: Mapping[str, Any],
    recommendation: Mapping[str, Any],
    final_conclusion: str,
) -> None:
    instability = "pipeline" if final_conclusion == "pipeline failed" else _as_str(p2_6.get("k12_instability_source", "unresolved"))
    lines = [
        "# HyperDA P2 Suite Audit",
        "",
        f"- Target region: `{target_region}`",
        f"- Seed: `{seed}`",
        f"- Final conclusion A: `{final_conclusion}`",
        f"- K12 instability source B: `{instability}`",
        f"- Locked eval ready C: `{str(recommendation.get('locked_eval_ready')).lower()}`",
        f"- Selected guard config path: `{recommendation.get('selected_guard_config_path', '')}`",
        f"- Seed recommendation: `{recommendation.get('seed_scope', 'seed 0 only')}`",
        "",
        "## Evidence",
        f"- P0 identity run: `{p0.get('run_path', '')}`",
        f"- P0 status: `{'passed' if p0.get('passed') else 'failed'}`",
        f"- P1 most stable scope: `{p1.get('most_stable_scope', '')}`",
        f"- P2.6 diagnosis: `{p2_6.get('k12_instability_source', 'unresolved')}` ({p2_6.get('reason', '')})",
        f"- P2.7 guard result: `{p2_7.get('guard_reduces_negative_migration', False)}` ({p2_7.get('reason', '')})",
        f"- P2.8 run: `{p2_8.get('run_path', '')}`",
        f"- P2.8 evidence complete: `{str(p2_8.get('evidence_complete', False)).lower()}`",
        f"- Leakage verdict: `{leakage.get('verdict', '')}`",
        "",
        "## Hashes",
        f"- Source checkpoint SHA256: `{p0.get('source_checkpoint_sha256', '') or recommendation.get('source_checkpoint_sha256', '')}`",
        f"- Split manifest SHA256: `{p0.get('split_manifest_sha256', '') or recommendation.get('split_manifest_sha256', '')}`",
        f"- Target context dates hash: `{p0.get('target_context_dates_hash', '') or recommendation.get('target_context_dates_hash', '')}`",
        f"- Target eval dates hash: `{p0.get('target_eval_dates_hash', '') or recommendation.get('target_eval_dates_hash', '')}`",
        f"- Prediction content hash: `{p0.get('prediction_content_hash', '')}`",
        f"- Metric values content hash: `{p0.get('metric_values_content_hash', '')}`",
        "",
        "## Recommendation",
    ]
    if recommendation.get("locked_eval_ready"):
        lines.append("- Source-safe P2.8 selected recipe is ready for locked target-eval execution.")
    else:
        lines.append("- Do not run locked target eval from this audit state.")
        for reason in recommendation.get("blocked_reasons", []):
            lines.append(f"- Blocked reason: {reason}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_hyperda_p2_suite(
    *,
    target_region: str,
    seed: int,
    runs_root: Path | str,
    output_dir: Path | str,
) -> dict[str, Any]:
    runs_root = Path(runs_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    missing_items: list[str] = []

    p0_rows, p0, missing = _audit_p0_identity(runs_root, target_region, seed)
    all_rows.extend(p0_rows)
    missing_items.extend(missing)

    p1_rows, p1, missing = _audit_p1_scope(runs_root, target_region, seed)
    all_rows.extend(p1_rows)
    missing_items.extend(missing)

    p2_6_rows, _p2_6_dir, missing = _audit_summary_phase(
        runs_root=runs_root,
        phase="P2.6",
        phase_dir="phase5_hyperda_p2_6_schedule_drift",
        stem="p2_6_schedule_drift_summary",
        target_region=target_region,
        seed=seed,
    )
    all_rows.extend(p2_6_rows)
    missing_items.extend(missing)
    p2_6 = _diagnose_p2_6(p2_6_rows)

    p2_7_rows, _p2_7_dir, missing = _audit_summary_phase(
        runs_root=runs_root,
        phase="P2.7",
        phase_dir="phase5_hyperda_p2_7_conservative_guard",
        stem="p2_7_conservative_guard_summary",
        target_region=target_region,
        seed=seed,
    )
    all_rows.extend(p2_7_rows)
    missing_items.extend(missing)
    p2_7 = _analyze_p2_7(p2_7_rows)

    p2_8_rows, p2_8, missing = _audit_p2_8(runs_root, target_region, seed)
    all_rows.extend(p2_8_rows)
    missing_items.extend(missing)

    missing_items.extend(_missing_hash_notes(all_rows))
    leakage = _build_leakage_audit(p2_8)
    final_conclusion = _final_conclusion(p0, p2_8, leakage)
    recommendation = _build_recommendation(
        target_region=target_region,
        seed=seed,
        p0=p0,
        p2_8=p2_8,
        leakage=leakage,
        final_conclusion=final_conclusion,
    )

    _write_summary_csv(output_dir / "p2_suite_summary.csv", all_rows)
    _write_json(output_dir / "leakage_audit.json", leakage)
    _write_yaml(output_dir / "selected_recipe_recommendation.yaml", recommendation)
    _write_missing_report(output_dir / "missing_or_invalid_artifacts.md", missing_items)
    _write_markdown_summary(
        output_dir / "p2_suite_summary.md",
        target_region=target_region,
        seed=seed,
        p0=p0,
        p1=p1,
        p2_6=p2_6,
        p2_7=p2_7,
        p2_8=p2_8,
        leakage=leakage,
        recommendation=recommendation,
        final_conclusion=final_conclusion,
    )

    return {
        "output_dir": str(output_dir),
        "final_conclusion": final_conclusion,
        "locked_eval_ready": recommendation["locked_eval_ready"],
        "leakage_verdict": leakage["verdict"],
        "row_count": len(all_rows),
        "missing_count": len(missing_items),
        "p0": p0,
        "p1": p1,
        "p2_6": p2_6,
        "p2_7": p2_7,
        "p2_8": p2_8,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit HyperDA P2 suite artifacts.")
    parser.add_argument("--target_region", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = audit_hyperda_p2_suite(
        target_region=args.target_region,
        seed=args.seed,
        runs_root=args.runs_root,
        output_dir=args.output_dir,
    )
    print(f"Wrote HyperDA P2 suite audit: {result['output_dir']}")
    print(f"Final conclusion: {result['final_conclusion']}")
    print(f"Locked eval ready: {result['locked_eval_ready']}")
    print(f"Leakage verdict: {result['leakage_verdict']}")


if __name__ == "__main__":
    main()
