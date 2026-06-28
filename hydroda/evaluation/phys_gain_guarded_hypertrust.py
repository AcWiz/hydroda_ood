"""M3_13 physics-gain guarded HyperDA-TRUST router.

This module keeps the M3_1 neural predictor frozen. Source-fit labels are used
only to calibrate a physics-gain prior bank; source-val records select the
shrink strength. The forward action is constrained to shrink the M3_1 residual
relative to the frozen source-base increment, never to add a new output
residual or amplify an existing one.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from hydroda.models.phys_trust import (
    PHYS_GAIN_BASIS_BANK_SCHEMA_VERSION,
    PHYS_GAIN_BASIS_NAMES,
    PHYS_GAIN_BASIS_SCHEMA_VERSION,
    phys_gain_basis_formula_schema,
    phys_gain_basis_from_raw_tensor,
)


VARIABLES = ("surface", "rootzone")
SOURCE_ROLES_FOR_BANK = {"source_fit"}
SOURCE_ROLES_FOR_SELECTION = {"source_val", "source_val_pseudo_query"}
FORBIDDEN_TARGET_ROLES = {
    "target_context",
    "target_support",
    "target_val",
    "target_eval",
    "target_query",
    "target_train",
    "target_full_train",
}
PHYS_GAIN_GUARD_METHOD_ID = "M3_13_phys_gain_guarded_hypertrust"
PHYS_GAIN_GUARD_BANK_SCHEMA = "m3_13_phys_gain_guard_source_bank_v1"
PHYS_GAIN_GUARD_SELECTION_SCHEMA = "m3_13_phys_gain_guard_source_gate_v1"
PHYS_GAIN_GUARD_ROUTER_SCHEMA = "m3_13_phys_gain_guarded_hypertrust_router_v1"
SOURCE_REGIONS = ("US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6")


@dataclass
class GainMoments:
    n: int = 0
    sum_x: float = 0.0
    sum_x2: float = 0.0
    sum_y: float = 0.0
    sum_xy: float = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        xv = np.asarray(x, dtype=np.float64).reshape(-1)
        yv = np.asarray(y, dtype=np.float64).reshape(-1)
        count = min(xv.size, yv.size)
        if count <= 0:
            return
        xv = xv[:count]
        yv = yv[:count]
        valid = np.isfinite(xv) & np.isfinite(yv)
        if not np.any(valid):
            return
        xv = xv[valid]
        yv = yv[valid]
        self.n += int(xv.size)
        self.sum_x += float(np.sum(xv))
        self.sum_x2 += float(np.sum(xv * xv))
        self.sum_y += float(np.sum(yv))
        self.sum_xy += float(np.sum(xv * yv))

    def gain(self, *, ridge_lambda: float) -> dict[str, float]:
        if self.n <= 1:
            return {"gain": 0.0, "cov": 0.0, "var": 0.0, "n": int(self.n)}
        n = float(self.n)
        mean_x = self.sum_x / n
        mean_y = self.sum_y / n
        var = max(float(self.sum_x2 / n - mean_x * mean_x), 0.0)
        cov = float(self.sum_xy / n - mean_x * mean_y)
        denom = var + max(float(ridge_lambda), 0.0)
        gain = float(cov / denom) if denom > 0.0 else 0.0
        if not math.isfinite(gain):
            gain = 0.0
        return {"gain": gain, "cov": cov, "var": var, "n": int(self.n)}


@dataclass
class CouplingMoments:
    n: int = 0
    sum_surface: float = 0.0
    sum_surface2: float = 0.0
    sum_rootzone: float = 0.0
    sum_surface_rootzone: float = 0.0

    def update(self, surface: np.ndarray, rootzone: np.ndarray) -> None:
        sv = np.asarray(surface, dtype=np.float64).reshape(-1)
        rv = np.asarray(rootzone, dtype=np.float64).reshape(-1)
        count = min(sv.size, rv.size)
        if count <= 0:
            return
        sv = sv[:count]
        rv = rv[:count]
        valid = np.isfinite(sv) & np.isfinite(rv)
        if not np.any(valid):
            return
        sv = sv[valid]
        rv = rv[valid]
        self.n += int(sv.size)
        self.sum_surface += float(np.sum(sv))
        self.sum_surface2 += float(np.sum(sv * sv))
        self.sum_rootzone += float(np.sum(rv))
        self.sum_surface_rootzone += float(np.sum(sv * rv))

    def gain(self, *, ridge_lambda: float) -> dict[str, float]:
        if self.n <= 1:
            return {"gain": 0.0, "cov": 0.0, "var": 0.0, "n": int(self.n)}
        n = float(self.n)
        mean_s = self.sum_surface / n
        mean_r = self.sum_rootzone / n
        var = max(float(self.sum_surface2 / n - mean_s * mean_s), 0.0)
        cov = float(self.sum_surface_rootzone / n - mean_s * mean_r)
        denom = var + max(float(ridge_lambda), 0.0)
        gain = float(cov / denom) if denom > 0.0 else 0.0
        if not math.isfinite(gain):
            gain = 0.0
        return {"gain": gain, "cov": cov, "var": var, "n": int(self.n)}


@dataclass
class GuardBucket:
    basis_surface: dict[str, GainMoments] = field(
        default_factory=lambda: {name: GainMoments() for name in PHYS_GAIN_BASIS_NAMES}
    )
    basis_rootzone: dict[str, GainMoments] = field(
        default_factory=lambda: {name: GainMoments() for name in PHYS_GAIN_BASIS_NAMES}
    )
    coupling: CouplingMoments = field(default_factory=CouplingMoments)
    n_records: int = 0


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    return array


def _input_array(record: Mapping[str, Any]) -> np.ndarray:
    if record.get("x_raw") is not None:
        return _as_array(record["x_raw"], name="x_raw")
    return _as_array(record["x"], name="x")


def _record_role(record: Mapping[str, Any]) -> str:
    return str(record.get("split_role") or record.get("query_role") or "")


def _require_roles(records: Sequence[Mapping[str, Any]], allowed: set[str], *, purpose: str) -> None:
    bad = sorted({_record_role(record) for record in records if _record_role(record) not in allowed})
    if bad:
        raise ValueError(f"{purpose} accepts only {sorted(allowed)} records; got roles {bad}")


def _require_no_target_records(records: Sequence[Mapping[str, Any]], *, purpose: str) -> None:
    bad = []
    for record in records:
        role = _record_role(record)
        adaptation_setting = str(record.get("adaptation_setting", ""))
        if role in FORBIDDEN_TARGET_ROLES or adaptation_setting in FORBIDDEN_TARGET_ROLES:
            bad.append((role, adaptation_setting))
    if bad:
        raise ValueError(f"{purpose} refuses target-side records: {bad[:3]}")


def source_region_from_record(record: Mapping[str, Any]) -> str:
    value = (
        record.get("sample_region_id")
        or record.get("source_region_id")
        or record.get("pseudo_target_region")
        or record.get("target_region_id")
        or ""
    )
    region = str(value)
    if not region:
        active = record.get("active_region_ids") or []
        if isinstance(active, str):
            parts = [part for part in active.split("|") if part]
            region = parts[0] if len(parts) == 1 else ""
        elif isinstance(active, Sequence) and not isinstance(active, (bytes, bytearray)) and len(active) == 1:
            region = str(active[0])
    if not region:
        raise ValueError("record lacks sample_region_id/source_region_id/target_region_id for grouping")
    return region


def month_from_record(record: Mapping[str, Any]) -> int:
    month = record.get("month")
    if month is None:
        date = str(record.get("query_date") or record.get("date_str") or "")
        month = int(date[5:7]) if len(date) >= 7 else 1
    value = int(month)
    if value < 1 or value > 12:
        raise ValueError(f"month must be in 1..12, got {month!r}")
    return value


def _mask_from_record(record: Mapping[str, Any], *, prefer_metric: bool = True) -> np.ndarray:
    keys = ("metric_mask", "loss_mask", "region_mask", "active_region_mask") if prefer_metric else (
        "region_mask",
        "active_region_mask",
        "metric_mask",
        "loss_mask",
    )
    for key in keys:
        if key in record and record[key] is not None:
            return (np.asarray(record[key]) > 0.5).astype(bool)
    raise KeyError("record missing metric/loss/region mask")


def _masked_values(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    count = min(arr.size, m.size)
    arr = arr.reshape(-1)[:count]
    m = m.reshape(-1)[:count]
    valid = m & np.isfinite(arr)
    return arr[valid]


def _masked_pair(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    count = min(a.size, b.size, m.size)
    a = a[:count]
    b = b[:count]
    m = m[:count]
    valid = m & np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        empty = np.empty((0,), dtype=np.float64)
        return empty, empty
    return a[valid], b[valid]


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    pairs = [
        (float(value), max(0.0, float(weight)))
        for value, weight in zip(values, weights)
        if math.isfinite(float(value)) and max(0.0, float(weight)) > 0.0
    ]
    total = float(sum(weight for _value, weight in pairs))
    if total <= 0.0:
        finite = [float(value) for value in values if math.isfinite(float(value))]
        return float(np.mean(finite)) if finite else 0.0
    return float(sum(value * weight for value, weight in pairs) / total)


def _basis_maps_from_record(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if "phys_gain_basis" in record and record["phys_gain_basis"] is not None:
        basis = np.asarray(record["phys_gain_basis"], dtype=np.float32)
        if basis.ndim == 4 and basis.shape[0] == 1:
            basis = basis[0]
    else:
        basis_tensor, _summary = phys_gain_basis_from_raw_tensor(_input_array(record), return_summary=False)
        basis = basis_tensor.detach().cpu().numpy().astype(np.float32)
    if basis.ndim != 3 or basis.shape[0] != len(PHYS_GAIN_BASIS_NAMES):
        raise ValueError(f"phys_gain_basis must have shape [5,H,W], got {basis.shape}")
    return {name: np.asarray(basis[idx], dtype=np.float32) for idx, name in enumerate(PHYS_GAIN_BASIS_NAMES)}


def _entry_from_bucket(region: str, month: int, bucket: GuardBucket, *, ridge_lambda: float) -> dict[str, Any]:
    surface_stats = {
        name: bucket.basis_surface[name].gain(ridge_lambda=ridge_lambda)
        for name in PHYS_GAIN_BASIS_NAMES
    }
    rootzone_stats = {
        name: bucket.basis_rootzone[name].gain(ridge_lambda=ridge_lambda)
        for name in PHYS_GAIN_BASIS_NAMES
    }
    coupling = bucket.coupling.gain(ridge_lambda=ridge_lambda)
    return {
        "source_region": region,
        "month": int(month),
        "n_records": int(bucket.n_records),
        "n_pixels": int(
            min(
                [surface_stats[name]["n"] for name in PHYS_GAIN_BASIS_NAMES]
                + [rootzone_stats[name]["n"] for name in PHYS_GAIN_BASIS_NAMES]
                + [coupling["n"]]
            )
        ),
        "G0": {
            "surface": {name: surface_stats[name]["gain"] for name in PHYS_GAIN_BASIS_NAMES},
            "rootzone": {name: rootzone_stats[name]["gain"] for name in PHYS_GAIN_BASIS_NAMES},
        },
        "stats": {
            "surface": surface_stats,
            "rootzone": rootzone_stats,
            "rootzone_from_surface": coupling,
        },
        "C0_rootzone_from_surface": coupling["gain"],
    }


def _consensus_entry(
    entries: Sequence[Mapping[str, Any]],
    weights: Sequence[float] | None = None,
    *,
    region: str | None,
    month: int | None,
    fallback_level: str,
) -> dict[str, Any]:
    resolved_weights = list(weights) if weights is not None else [float(entry.get("n_pixels", 0) or 0) for entry in entries]
    if len(resolved_weights) != len(entries):
        raise ValueError("consensus weights and entries have different lengths")

    def mean_gain(variable: str, basis_name: str) -> float:
        return _weighted_mean(
            [
                float(((entry.get("G0", {}) or {}).get(variable, {}) or {}).get(basis_name, 0.0))
                for entry in entries
            ],
            resolved_weights,
        )

    return {
        "source_region": region or "source_consensus",
        "month": int(month) if month is not None else None,
        "n_records": int(sum(int(entry.get("n_records", 0) or 0) for entry in entries)),
        "n_pixels": int(sum(int(entry.get("n_pixels", 0) or 0) for entry in entries)),
        "G0": {
            variable: {name: mean_gain(variable, name) for name in PHYS_GAIN_BASIS_NAMES}
            for variable in VARIABLES
        },
        "C0_rootzone_from_surface": _weighted_mean(
            [float(entry.get("C0_rootzone_from_surface", 0.0) or 0.0) for entry in entries],
            resolved_weights,
        ),
        "fallback_level": fallback_level,
    }


class PhysGainGuardBankAccumulator:
    """Streaming source_fit gain-bank accumulator for M3_13."""

    def __init__(
        self,
        *,
        ridge_lambda: float = 1e-4,
        source_checkpoint: str = "",
        split_manifest: str = "",
        source_neighbor_top_m: int = 4,
    ) -> None:
        self.ridge_lambda = max(float(ridge_lambda), 0.0)
        self.source_checkpoint = str(source_checkpoint or "")
        self.split_manifest = str(split_manifest or "")
        self.source_neighbor_top_m = int(source_neighbor_top_m)
        if self.source_neighbor_top_m < 1:
            raise ValueError("source_neighbor_top_m must be >= 1")
        self.grouped: dict[tuple[str, int], GuardBucket] = {}
        self.scale_abs = {"surface": [], "rootzone": []}
        self.n_records_seen = 0
        self.n_records_used = 0

    def update(self, record: Mapping[str, Any]) -> None:
        _require_roles([record], SOURCE_ROLES_FOR_BANK, purpose="M3_13 source gain bank construction")
        _require_no_target_records([record], purpose="M3_13 source gain bank construction")
        self.n_records_seen += 1
        basis = _basis_maps_from_record(record)
        inc_s = _as_array(record["increment_surface"], name="increment_surface")
        inc_r = _as_array(record["increment_rootzone"], name="increment_rootzone")
        mask = _mask_from_record(record)
        key = (source_region_from_record(record), month_from_record(record))
        bucket = self.grouped.setdefault(key, GuardBucket())
        used_any = False
        for name in PHYS_GAIN_BASIS_NAMES:
            b_vals, s_vals = _masked_pair(basis[name], inc_s, mask)
            _, r_vals = _masked_pair(basis[name], inc_r, mask)
            if b_vals.size > 0 and s_vals.size > 0:
                bucket.basis_surface[name].update(b_vals, s_vals)
                used_any = True
            if b_vals.size > 0 and r_vals.size > 0:
                bucket.basis_rootzone[name].update(b_vals, r_vals)
                used_any = True
        s_c, r_c = _masked_pair(inc_s, inc_r, mask)
        if s_c.size > 0:
            bucket.coupling.update(s_c, r_c)
            self.scale_abs["surface"].append(np.abs(s_c).astype(np.float64))
            self.scale_abs["rootzone"].append(np.abs(r_c).astype(np.float64))
            used_any = True
        if used_any:
            bucket.n_records += 1
            self.n_records_used += 1

    def finalize(self) -> dict[str, Any]:
        if not self.grouped:
            raise ValueError("No valid source_fit pixels available for M3_13 source gain bank")
        entries: dict[str, Any] = {}
        monthly_values: dict[int, list[dict[str, Any]]] = {month: [] for month in range(1, 13)}
        for (region, month), bucket in sorted(self.grouped.items()):
            entry = _entry_from_bucket(region, month, bucket, ridge_lambda=self.ridge_lambda)
            entries[f"{region}|{month:02d}"] = entry
            monthly_values[month].append(entry)
        month_consensus = {
            f"{month:02d}": _consensus_entry(
                entries_for_month,
                region=None,
                month=month,
                fallback_level="source_month_consensus",
            )
            for month, entries_for_month in monthly_values.items()
            if entries_for_month
        }
        global_consensus = _consensus_entry(
            list(entries.values()),
            region=None,
            month=None,
            fallback_level="source_global_consensus",
        )

        def mean_abs(variable: str) -> float:
            rows = self.scale_abs[variable]
            if not rows:
                return 0.0
            arr = np.concatenate(rows)
            finite = arr[np.isfinite(arr)]
            return float(np.mean(finite)) if finite.size else 0.0

        bank = {
            "schema_version": PHYS_GAIN_GUARD_BANK_SCHEMA,
            "method_id": PHYS_GAIN_GUARD_METHOD_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_gain_bank_schema": PHYS_GAIN_BASIS_BANK_SCHEMA_VERSION,
            "phys_gain_basis_schema_version": PHYS_GAIN_BASIS_SCHEMA_VERSION,
            "formula_schema": phys_gain_basis_formula_schema(),
            "source_split_roles": {
                "bank": sorted(SOURCE_ROLES_FOR_BANK),
                "eta_selection": sorted(SOURCE_ROLES_FOR_SELECTION),
                "forbidden": sorted(FORBIDDEN_TARGET_ROLES),
            },
            "source_label_usage": "source_fit_increments_for_gain_bank_only",
            "bank_split_roles": sorted(SOURCE_ROLES_FOR_BANK),
            "target_val_usage": "unused_in_main_protocol",
            "target_eval_usage": "not_used_for_bank_or_eta_selection",
            "target_eval_selection_usage": "forbidden",
            "neural_training_epochs": 0,
            "neural_parameter_updates": 0,
            "ridge_lambda": float(self.ridge_lambda),
            "source_neighbor_top_m": int(self.source_neighbor_top_m),
            "gain_source_priority": [
                "source_trust_top_m_neighbor_weighted_consensus",
                "source_month_consensus",
                "source_global_consensus",
            ],
            "basis_names": list(PHYS_GAIN_BASIS_NAMES),
            "variables": list(VARIABLES),
            "formula": {
                "q_surface": "sum_b G0_surface_b * B_b",
                "q_rootzone": "sum_b G0_rootzone_b * B_b + C0_rz * q_surface",
                "risk_v": "confidence_v * sign_conflict(pred_M3_1_v - source_base_v, q_v)",
                "guard_v": "clamp(1 - eta_v * risk_v, 0.90, 1.00)",
                "final": "source_base_v + guard_v * (pred_M3_1_v - source_base_v)",
            },
            "confidence_formula": "finite_gain_basis_and_region_mask_only",
            "channel_11_usage": "diagnostic_coverage_only_not_obs_loss_metric_region_mask",
            "hard_mask_channels": ["region_mask"],
            "source_checkpoint": str(self.source_checkpoint or ""),
            "source_checkpoint_sha256": (
                _file_sha256(self.source_checkpoint)
                if self.source_checkpoint and Path(self.source_checkpoint).exists()
                else ""
            ),
            "split_manifest": str(self.split_manifest or ""),
            "split_manifest_sha256": (
                _file_sha256(self.split_manifest)
                if self.split_manifest and Path(self.split_manifest).exists()
                else ""
            ),
            "source_scale": {
                "surface": mean_abs("surface"),
                "rootzone": mean_abs("rootzone"),
            },
            "entries": entries,
            "month_consensus": month_consensus,
            "global_consensus": global_consensus,
            "n_source_records_seen": int(self.n_records_seen),
            "n_source_records_used": int(self.n_records_used),
            "eta_grid": [],
            "selection_hash": "",
        }
        bank["source_gain_bank_hash"] = _stable_hash(
            {
                "schema_version": bank["schema_version"],
                "method_id": bank["method_id"],
                "ridge_lambda": bank["ridge_lambda"],
                "entries": entries,
                "month_consensus": month_consensus,
                "global_consensus": global_consensus,
                "formula": bank["formula"],
            }
        )
        bank["bank_content_hash"] = bank["source_gain_bank_hash"]
        return bank


def build_source_phys_gain_guard_bank_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    ridge_lambda: float = 1e-4,
    source_checkpoint: str = "",
    split_manifest: str = "",
    source_neighbor_top_m: int = 4,
) -> dict[str, Any]:
    accumulator = PhysGainGuardBankAccumulator(
        ridge_lambda=ridge_lambda,
        source_checkpoint=source_checkpoint,
        split_manifest=split_manifest,
        source_neighbor_top_m=source_neighbor_top_m,
    )
    for record in records:
        accumulator.update(record)
    return accumulator.finalize()


def save_gain_bank(bank: Mapping[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bank, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_gain_bank(path: str | Path) -> dict[str, Any]:
    bank = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_gain_bank_metadata(bank)
    return bank


def validate_gain_bank_metadata(bank: Mapping[str, Any]) -> None:
    if bank.get("schema_version") != PHYS_GAIN_GUARD_BANK_SCHEMA:
        raise ValueError(f"Unsupported M3_13 gain bank schema: {bank.get('schema_version')!r}")
    if bank.get("method_id") != PHYS_GAIN_GUARD_METHOD_ID:
        raise ValueError("M3_13 gain bank method_id mismatch")
    if bank.get("source_label_usage") != "source_fit_increments_for_gain_bank_only":
        raise ValueError("M3_13 gain bank must declare source_fit label usage only")
    if bank.get("target_eval_usage") != "not_used_for_bank_or_eta_selection":
        raise ValueError("M3_13 gain bank must not use target_eval for bank or eta selection")
    roles = set((bank.get("source_split_roles", {}) or {}).get("bank", []))
    if roles != SOURCE_ROLES_FOR_BANK:
        raise ValueError("M3_13 gain bank must be built from source_fit only")
    if not isinstance(bank.get("entries"), Mapping):
        raise ValueError("M3_13 gain bank missing entries")


def _neighbor_region_month_weight(item: Any, *, default_month: int, rank: int) -> tuple[str, int, float] | None:
    if isinstance(item, str):
        if "|" in item:
            region, month_text = item.split("|", 1)
            return region, int(month_text), 1.0 / float(rank + 1)
        return item, int(default_month), 1.0 / float(rank + 1)
    if not isinstance(item, Mapping):
        return None
    key = item.get("entry_key") or item.get("source_region_month_key")
    if key is not None and "|" in str(key):
        region, month_text = str(key).split("|", 1)
        month = int(month_text)
    else:
        region = str(
            item.get("source_region")
            or item.get("source_region_id")
            or item.get("region")
            or item.get("sample_region_id")
            or ""
        )
        month = int(item.get("month", default_month))
    if not region or month < 1 or month > 12:
        return None
    weight = item.get("weight", item.get("trust_weight", item.get("neighbor_weight")))
    if weight is None:
        distance = item.get("distance", item.get("neighbor_distance"))
        weight = 1.0 / max(float(distance), 1e-6) if distance is not None and math.isfinite(float(distance)) else 1.0 / float(rank + 1)
    weight_f = float(weight)
    if not math.isfinite(weight_f) or weight_f <= 0.0:
        return None
    return region, month, weight_f


def _entry_from_neighbors(bank: Mapping[str, Any], sample: Mapping[str, Any], *, month: int) -> tuple[Mapping[str, Any] | None, str]:
    neighbors = sample.get("source_trust_neighbors") or sample.get("nearest_source_neighbors") or []
    if not isinstance(neighbors, Sequence) or isinstance(neighbors, (str, bytes, bytearray)):
        return None, ""
    entries = bank.get("entries", {})
    top_m = int(bank.get("source_neighbor_top_m", 4) or 4)
    pairs: list[tuple[Mapping[str, Any], float]] = []
    for rank, item in enumerate(list(neighbors)[:top_m]):
        parsed = _neighbor_region_month_weight(item, default_month=month, rank=rank)
        if parsed is None:
            continue
        region, neighbor_month, weight = parsed
        key = f"{region}|{neighbor_month:02d}"
        if key in entries:
            pairs.append((entries[key], weight))
    if not pairs:
        return None, ""
    entry = _consensus_entry(
        [entry for entry, _weight in pairs],
        [weight for _entry, weight in pairs],
        region=None,
        month=month,
        fallback_level="source_trust_top_m_neighbor_weighted_consensus",
    )
    entry["neighbor_count"] = len(pairs)
    return entry, "source_trust_top_m_neighbor_weighted_consensus"


def select_gain_entry(bank: Mapping[str, Any], sample: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    validate_gain_bank_metadata(bank)
    month = month_from_record(sample)
    neighbor_entry, neighbor_level = _entry_from_neighbors(bank, sample, month=month)
    if neighbor_entry is not None:
        return neighbor_entry, neighbor_level
    month_key = f"{month:02d}"
    month_consensus = bank.get("month_consensus", {})
    if month_key in month_consensus:
        return month_consensus[month_key], "source_month_consensus"
    global_entry = bank.get("global_consensus", {})
    if not global_entry:
        raise ValueError("M3_13 gain bank lacks global_consensus fallback")
    return global_entry, "source_global_consensus"


def physical_gain_query_from_sample(
    sample: Mapping[str, Any],
    bank: Mapping[str, Any],
    *,
    proposal_clip_scale: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    basis = _basis_maps_from_record(sample)
    entry, fallback_level = select_gain_entry(bank, sample)
    g0 = entry.get("G0", {})

    def q_for(variable: str) -> np.ndarray:
        gains = (g0.get(variable, {}) or {})
        q = np.zeros_like(next(iter(basis.values())), dtype=np.float32)
        for name in PHYS_GAIN_BASIS_NAMES:
            q = q + float(gains.get(name, 0.0) or 0.0) * basis[name].astype(np.float32)
        return q.astype(np.float32)

    q_surface = q_for("surface")
    q_rootzone = (q_for("rootzone") + float(entry.get("C0_rootzone_from_surface", 0.0) or 0.0) * q_surface).astype(np.float32)
    scale = bank.get("source_scale", {})
    clip_s = max(float(scale.get("surface", 0.0) or 0.0), 0.0) * float(proposal_clip_scale)
    clip_r = max(float(scale.get("rootzone", 0.0) or 0.0), 0.0) * float(proposal_clip_scale)
    if clip_s > 0.0:
        q_surface = np.clip(q_surface, -clip_s, clip_s).astype(np.float32)
    if clip_r > 0.0:
        q_rootzone = np.clip(q_rootzone, -clip_r, clip_r).astype(np.float32)
    summary = {
        "fallback_level": fallback_level,
        "bank_region": entry.get("source_region", ""),
        "bank_month": entry.get("month"),
        "neighbor_count": int(entry.get("neighbor_count", 0) or 0),
        "n_pixels": int(entry.get("n_pixels", 0) or 0),
        "clip_surface": float(clip_s),
        "clip_rootzone": float(clip_r),
        "G0": entry.get("G0", {}),
        "C0_rootzone_from_surface": float(entry.get("C0_rootzone_from_surface", 0.0) or 0.0),
    }
    return {"surface": q_surface, "rootzone": q_rootzone}, summary


def confidence_from_sample(sample: Mapping[str, Any]) -> dict[str, np.ndarray]:
    basis = _basis_maps_from_record(sample)
    shape = next(iter(basis.values())).shape
    finite = np.ones(shape, dtype=bool)
    for value in basis.values():
        finite &= np.isfinite(value)
    region_mask = sample.get("region_mask", sample.get("active_region_mask"))
    if region_mask is not None:
        finite &= np.asarray(region_mask, dtype=np.float32) > 0.5
    conf = finite.astype(np.float32)
    return {"surface": conf.copy(), "rootzone": conf.copy()}


def sign_conflict_risk(residual: np.ndarray, q: np.ndarray) -> np.ndarray:
    res = np.asarray(residual, dtype=np.float32)
    query = np.asarray(q, dtype=np.float32)
    conflict = (np.sign(res) * np.sign(query)) < 0.0
    return (conflict & np.isfinite(res) & np.isfinite(query)).astype(np.float32)


def apply_phys_gain_guard(
    sample: Mapping[str, Any],
    base_pred: Mapping[str, Any],
    source_base_pred: Mapping[str, Any],
    bank: Mapping[str, Any],
    *,
    eta_surface: float = 0.0,
    eta_rootzone: float | None = None,
    guard_min: float = 0.90,
    proposal_clip_scale: float = 1.0,
) -> dict[str, Any]:
    eta_s = float(eta_surface)
    eta_r = eta_s if eta_rootzone is None else float(eta_rootzone)
    if not 0.0 <= eta_s <= 1.0 or not 0.0 <= eta_r <= 1.0:
        raise ValueError("eta values must be in [0, 1]")
    if not 0.0 <= float(guard_min) <= 1.0:
        raise ValueError("guard_min must be in [0, 1]")

    pred_s = np.asarray(base_pred["pred_increment_surface"], dtype=np.float32)
    pred_r = np.asarray(base_pred["pred_increment_rootzone"], dtype=np.float32)
    source_s = np.asarray(source_base_pred["pred_increment_surface"], dtype=np.float32)
    source_r = np.asarray(source_base_pred["pred_increment_rootzone"], dtype=np.float32)
    residual_s = (pred_s - source_s).astype(np.float32)
    residual_r = (pred_r - source_r).astype(np.float32)
    if eta_s == 0.0 and eta_r == 0.0:
        guard_s = np.ones_like(residual_s, dtype=np.float32)
        guard_r = np.ones_like(residual_r, dtype=np.float32)
        q_summary = {"fallback_level": "eta_zero_no_query", "coverage": 0.0, "conflict_surface": 0.0, "conflict_rootzone": 0.0}
    else:
        query, q_summary = physical_gain_query_from_sample(sample, bank, proposal_clip_scale=proposal_clip_scale)
        conf = confidence_from_sample(sample)
        risk_s = conf["surface"] * sign_conflict_risk(residual_s, query["surface"])
        risk_r = conf["rootzone"] * sign_conflict_risk(residual_r, query["rootzone"])
        guard_s = np.clip(1.0 - eta_s * risk_s, float(guard_min), 1.0).astype(np.float32)
        guard_r = np.clip(1.0 - eta_r * risk_r, float(guard_min), 1.0).astype(np.float32)
        q_summary = dict(q_summary)
        q_summary.update(
            {
                "coverage": float(np.mean(conf["surface"] > 0.0)) if conf["surface"].size else 0.0,
                "conflict_surface": float(np.mean(risk_s > 0.0)) if risk_s.size else 0.0,
                "conflict_rootzone": float(np.mean(risk_r > 0.0)) if risk_r.size else 0.0,
            }
        )
    final_s = (source_s + guard_s * residual_s).astype(np.float32)
    final_r = (source_r + guard_r * residual_r).astype(np.float32)
    forecast_s = np.asarray(sample["forecast_surface"], dtype=np.float32)
    forecast_r = np.asarray(sample["forecast_rootzone"], dtype=np.float32)
    summary = {
        "schema_version": PHYS_GAIN_GUARD_ROUTER_SCHEMA,
        "eta_surface": eta_s,
        "eta_rootzone": eta_r,
        "guard_min": float(guard_min),
        "guard_surface_mean": float(np.mean(guard_s)) if guard_s.size else 1.0,
        "guard_rootzone_mean": float(np.mean(guard_r)) if guard_r.size else 1.0,
        "guard_surface_min": float(np.min(guard_s)) if guard_s.size else 1.0,
        "guard_rootzone_min": float(np.min(guard_r)) if guard_r.size else 1.0,
        "guard_coverage_surface": float(np.mean(guard_s < 1.0)) if guard_s.size else 0.0,
        "guard_coverage_rootzone": float(np.mean(guard_r < 1.0)) if guard_r.size else 0.0,
        "residual_abs_mean_surface_before": float(np.mean(np.abs(residual_s))) if residual_s.size else 0.0,
        "residual_abs_mean_surface_after": float(np.mean(np.abs(final_s - source_s))) if residual_s.size else 0.0,
        "residual_abs_mean_rootzone_before": float(np.mean(np.abs(residual_r))) if residual_r.size else 0.0,
        "residual_abs_mean_rootzone_after": float(np.mean(np.abs(final_r - source_r))) if residual_r.size else 0.0,
        "query": q_summary,
        "action": "shrink_only_no_new_residual_no_amplification",
    }
    return {
        "pred_increment_surface": final_s,
        "pred_increment_rootzone": final_r,
        "pred_analysis_surface": (forecast_s + final_s).astype(np.float32),
        "pred_analysis_rootzone": (forecast_r + final_r).astype(np.float32),
        "m3_13_eta_surface": eta_s,
        "m3_13_eta_rootzone": eta_r,
        "m3_13_guard_summary": summary,
    }


class PhysGainGuardedHyperTrustPredictor:
    """Predictor wrapper applying the M3_13 shrink-only guard."""

    def __init__(
        self,
        base_predictor: Any,
        source_base_predictor: Any,
        bank: Mapping[str, Any],
        *,
        eta_surface: float = 0.0,
        eta_rootzone: float | None = None,
        guard_min: float = 0.90,
        proposal_clip_scale: float = 1.0,
        method_name: str = PHYS_GAIN_GUARD_METHOD_ID,
    ) -> None:
        validate_gain_bank_metadata(bank)
        self.base_predictor = base_predictor
        self.source_base_predictor = source_base_predictor
        self.bank = dict(bank)
        self.eta_surface = float(eta_surface)
        self.eta_rootzone = self.eta_surface if eta_rootzone is None else float(eta_rootzone)
        self.guard_min = float(guard_min)
        self.proposal_clip_scale = float(proposal_clip_scale)
        self.method_name = method_name
        self.metadata = {
            "schema_version": PHYS_GAIN_GUARD_ROUTER_SCHEMA,
            "method_id": PHYS_GAIN_GUARD_METHOD_ID,
            "base_method": getattr(base_predictor, "method_name", "unknown"),
            "source_base_method": getattr(source_base_predictor, "method_name", "unknown"),
            "base_anchor": "M3_1_hyperda_trust_medium",
            "eta_surface": self.eta_surface,
            "eta_rootzone": self.eta_rootzone,
            "guard_min": self.guard_min,
            "source_gain_bank_hash": self.bank.get("source_gain_bank_hash", self.bank.get("bank_content_hash", "")),
            "formula_schema": phys_gain_basis_formula_schema(),
            "neural_training_epochs": 0,
            "neural_parameter_updates": 0,
            "target_val_usage": "unused_in_main_protocol",
            "target_eval_usage": "final_eval_only_no_selection",
            "target_eval_selection_usage": "forbidden",
            "channel_11_usage": "diagnostic_coverage_only_not_obs_loss_metric_region_mask",
            "action": "pred = source_base + guard * (pred_M3_1 - source_base)",
        }

    def predict(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        base_pred = self.base_predictor.predict(sample)
        source_base_pred = self.source_base_predictor.predict(sample)
        return apply_phys_gain_guard(
            sample,
            base_pred,
            source_base_pred,
            self.bank,
            eta_surface=self.eta_surface,
            eta_rootzone=self.eta_rootzone,
            guard_min=self.guard_min,
            proposal_clip_scale=self.proposal_clip_scale,
        )


def build_source_records_from_predictors(
    *,
    dataset: Any,
    base_predictor: Any | None = None,
    source_base_predictor: Any | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    n = len(dataset) if max_samples is None or int(max_samples) <= 0 else min(len(dataset), int(max_samples))
    records: list[dict[str, Any]] = []
    for idx in range(n):
        sample = dataset[idx]
        record = {
            "sample_idx": idx,
            "split_role": sample.get("split_role", ""),
            "query_time_index": int(sample.get("time_index", -1)),
            "query_date": sample.get("date_str", ""),
            "month": sample.get("month"),
            "season": sample.get("season", ""),
            "country_id": sample.get("country_id", ""),
            "target_region_id": sample.get("target_region_id", ""),
            "sample_region_id": sample.get("sample_region_id", ""),
            "active_region_ids": list(sample.get("active_region_ids", [])),
            "adaptation_setting": sample.get("adaptation_setting", "zero_shot_context"),
            "K": sample.get("K", 0),
            "seed": int(sample.get("seed", -1)),
            "x": np.asarray(sample["x"], dtype=np.float32),
            "forecast_surface": np.asarray(sample["forecast_surface"], dtype=np.float32),
            "forecast_rootzone": np.asarray(sample["forecast_rootzone"], dtype=np.float32),
            "analysis_surface": np.asarray(sample["analysis_surface"], dtype=np.float32),
            "analysis_rootzone": np.asarray(sample["analysis_rootzone"], dtype=np.float32),
            "increment_surface": np.asarray(sample["increment_surface"], dtype=np.float32),
            "increment_rootzone": np.asarray(sample["increment_rootzone"], dtype=np.float32),
            "metric_mask": np.asarray(sample["metric_mask"], dtype=np.float32),
            "region_mask": np.asarray(sample.get("region_mask", sample["metric_mask"]), dtype=np.float32),
            "latitude_weight": np.asarray(sample["latitude_weight"], dtype=np.float32),
        }
        if "source_trust_neighbors" in sample:
            record["source_trust_neighbors"] = sample["source_trust_neighbors"]
        if base_predictor is not None:
            pred = base_predictor.predict(sample)
            record["pred_increment_surface"] = np.asarray(pred["pred_increment_surface"], dtype=np.float32)
            record["pred_increment_rootzone"] = np.asarray(pred["pred_increment_rootzone"], dtype=np.float32)
        if source_base_predictor is not None:
            source_pred = source_base_predictor.predict(sample)
            record["source_base_increment_surface"] = np.asarray(source_pred["pred_increment_surface"], dtype=np.float32)
            record["source_base_increment_rootzone"] = np.asarray(source_pred["pred_increment_rootzone"], dtype=np.float32)
        records.append(record)
    return records


def _empty_region_metric_block() -> dict[str, dict[str, list[float]]]:
    return {var: {"mse": [], "fcst_mse": [], "rmse": [], "corr": [], "sign": []} for var in VARIABLES}


def _accumulate_prediction_metrics(
    block: dict[str, dict[str, dict[str, list[float]]]],
    record: Mapping[str, Any],
    pred: Mapping[str, Any],
) -> None:
    from hydroda.metrics.skill import increment_corr, increment_rmse, sign_accuracy_deadzone, weighted_analysis_skill_components

    region = source_region_from_record(record)
    block.setdefault(region, _empty_region_metric_block())
    mask = np.asarray(record["metric_mask"], dtype=np.float32)
    latw = np.asarray(record["latitude_weight"], dtype=np.float32)
    for variable in VARIABLES:
        forecast = np.asarray(record[f"forecast_{variable}"], dtype=np.float32)
        true_inc = np.asarray(record[f"increment_{variable}"], dtype=np.float32)
        true_analysis = np.asarray(record[f"analysis_{variable}"], dtype=np.float32)
        pred_inc = np.asarray(pred[f"pred_increment_{variable}"], dtype=np.float32)
        pred_analysis = np.asarray(pred[f"pred_analysis_{variable}"], dtype=np.float32)
        target = block[region][variable]
        mse_m, mse_f = weighted_analysis_skill_components(
            pred_analysis=pred_analysis,
            true_analysis=true_analysis,
            forecast=forecast,
            mask=mask,
            latitude_weight=latw,
        )
        target["mse"].append(float(mse_m))
        target["fcst_mse"].append(float(mse_f))
        target["rmse"].append(float(increment_rmse(pred_inc, true_inc, mask)))
        target["corr"].append(float(increment_corr(pred_inc, true_inc, mask)))
        target["sign"].append(float(sign_accuracy_deadzone(pred_inc, true_inc, mask, epsilon=0.005)))


def _summarize_metric_block(
    block: Mapping[str, Mapping[str, Mapping[str, list[float]]]],
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    variable_summary: dict[str, Any] = {}
    region_summary: dict[str, dict[str, float]] = {}
    for region, region_block in sorted(block.items()):
        region_summary[region] = {}
        for variable in VARIABLES:
            vals = region_block[variable]
            mse = [v for v in vals["mse"] if math.isfinite(v)]
            fcst = [v for v in vals["fcst_mse"] if math.isfinite(v)]
            model_rmse = float(np.sqrt(np.mean(mse))) if mse else float("nan")
            fcst_rmse = float(np.sqrt(np.mean(fcst))) if fcst else float("nan")
            skill = (
                float(1.0 - model_rmse / fcst_rmse)
                if math.isfinite(model_rmse) and math.isfinite(fcst_rmse) and fcst_rmse > 0.0
                else float("nan")
            )
            region_summary[region][f"{variable}_skill"] = skill
            region_summary[region][f"{variable}_analysis_rmse_latw"] = model_rmse
    for variable in VARIABLES:
        mse_all: list[float] = []
        fcst_all: list[float] = []
        rmse_all: list[float] = []
        corr_all: list[float] = []
        sign_all: list[float] = []
        for region_block in block.values():
            vals = region_block[variable]
            mse_all.extend(v for v in vals["mse"] if math.isfinite(v))
            fcst_all.extend(v for v in vals["fcst_mse"] if math.isfinite(v))
            rmse_all.extend(v for v in vals["rmse"] if math.isfinite(v))
            corr_all.extend(v for v in vals["corr"] if math.isfinite(v))
            sign_all.extend(v for v in vals["sign"] if math.isfinite(v))
        model_rmse = float(np.sqrt(np.mean(mse_all))) if mse_all else float("nan")
        fcst_rmse = float(np.sqrt(np.mean(fcst_all))) if fcst_all else float("nan")
        variable_summary[variable] = {
            "skill_primary": (
                float(1.0 - model_rmse / fcst_rmse)
                if math.isfinite(model_rmse) and math.isfinite(fcst_rmse) and fcst_rmse > 0.0
                else float("nan")
            ),
            "analysis_rmse_latw": model_rmse,
            "increment_rmse_mean": float(np.mean(rmse_all)) if rmse_all else float("nan"),
            "increment_corr_mean": float(np.mean(corr_all)) if corr_all else float("nan"),
            "sign_accuracy_deadzone_mean": float(np.mean(sign_all)) if sign_all else float("nan"),
        }
    return variable_summary, region_summary


def _safe_delta(value: float, baseline: float) -> float:
    if not math.isfinite(float(value)) or not math.isfinite(float(baseline)):
        return float("nan")
    return float(value) - float(baseline)


def _relative_delta(value: float, baseline: float) -> float:
    if not math.isfinite(float(value)) or not math.isfinite(float(baseline)) or float(baseline) == 0.0:
        return float("nan")
    return float(value / baseline - 1.0)


def dual_variable_cvar_score_from_region_skills(region_skills: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    if not region_skills:
        return {
            "dual_variable_cvar_safe_score": float("-inf"),
            "dual_variable_non_degradation": False,
            "worst_region_surface_skill": float("nan"),
            "worst_region_rootzone_skill": float("nan"),
            "mean_region_balanced_skill": float("nan"),
            "positive_improvement_rate": 0.0,
        }
    surface = []
    rootzone = []
    balanced = []
    all_skills = []
    for block in region_skills.values():
        s = float(block.get("surface_skill", float("nan")))
        r = float(block.get("rootzone_skill", float("nan")))
        if math.isfinite(s):
            surface.append(s)
            all_skills.append(s)
        if math.isfinite(r):
            rootzone.append(r)
            all_skills.append(r)
        if math.isfinite(s) and math.isfinite(r):
            balanced.append(float((s + r) / 2.0))
    if not surface or not rootzone or not balanced:
        return {
            "dual_variable_cvar_safe_score": float("-inf"),
            "dual_variable_non_degradation": False,
            "worst_region_surface_skill": float("nan"),
            "worst_region_rootzone_skill": float("nan"),
            "mean_region_balanced_skill": float("nan"),
            "positive_improvement_rate": 0.0,
        }
    worst_s = float(np.min(surface))
    worst_r = float(np.min(rootzone))
    mean_bal = float(np.mean(balanced))
    pos_rate = float(np.mean([1.0 if value > 0.0 else 0.0 for value in all_skills]))
    tail = float((worst_s + worst_r) / 2.0)
    score = (
        0.60 * tail
        + 0.25 * mean_bal
        + 0.15 * pos_rate
        - 0.20 * max(0.0, -worst_s)
        - 0.20 * max(0.0, -worst_r)
    )
    non_degrade = bool(worst_s >= 0.0 and worst_r >= 0.0)
    if not non_degrade:
        score -= 1.0
    return {
        "dual_variable_cvar_safe_score": float(score),
        "dual_variable_non_degradation": bool(non_degrade),
        "worst_region_surface_skill": float(worst_s),
        "worst_region_rootzone_skill": float(worst_r),
        "mean_region_balanced_skill": float(mean_bal),
        "positive_improvement_rate": float(pos_rate),
    }


def _result_from_summaries(
    *,
    eta_surface: float,
    eta_rootzone: float,
    base_summary: Mapping[str, Any],
    base_region_summary: Mapping[str, Mapping[str, float]],
    routed_summary: Mapping[str, Any],
    routed_region_summary: Mapping[str, Mapping[str, float]],
    guard_diagnostics: Mapping[str, float],
) -> dict[str, Any]:
    base_cvar = dual_variable_cvar_score_from_region_skills(base_region_summary)
    routed_cvar = dual_variable_cvar_score_from_region_skills(routed_region_summary)
    deltas = {}
    for variable in VARIABLES:
        deltas[variable] = {
            "analysis_rmse_latw_relative": _relative_delta(
                routed_summary[variable]["analysis_rmse_latw"],
                base_summary[variable]["analysis_rmse_latw"],
            ),
            "increment_corr_delta": _safe_delta(
                routed_summary[variable]["increment_corr_mean"],
                base_summary[variable]["increment_corr_mean"],
            ),
            "sign_accuracy_deadzone_delta": _safe_delta(
                routed_summary[variable]["sign_accuracy_deadzone_mean"],
                base_summary[variable]["sign_accuracy_deadzone_mean"],
            ),
        }
    region_rmse_relative_deltas: dict[str, dict[str, float]] = {}
    for region, routed_block in routed_region_summary.items():
        base_block = base_region_summary.get(region, {})
        region_rmse_relative_deltas[region] = {}
        for variable in VARIABLES:
            key = f"{variable}_analysis_rmse_latw"
            region_rmse_relative_deltas[region][variable] = _relative_delta(
                routed_block.get(key, float("nan")),
                base_block.get(key, float("nan")),
            )
    finite_region_degrades = [
        value
        for block in region_rmse_relative_deltas.values()
        for value in block.values()
        if math.isfinite(float(value))
    ]
    return {
        "eta_surface": float(eta_surface),
        "eta_rootzone": float(eta_rootzone),
        "base_summary": base_summary,
        "summary": routed_summary,
        "base_region_summary": base_region_summary,
        "region_summary": routed_region_summary,
        "region_rmse_relative_deltas": region_rmse_relative_deltas,
        "max_source_region_rmse_relative_degrade": (
            float(max(finite_region_degrades)) if finite_region_degrades else float("nan")
        ),
        "base_dual_variable_cvar": base_cvar,
        "dual_variable_cvar": routed_cvar,
        "dual_variable_cvar_delta": _safe_delta(
            routed_cvar["dual_variable_cvar_safe_score"],
            base_cvar["dual_variable_cvar_safe_score"],
        ),
        "deltas": deltas,
        "guard_diagnostics": dict(guard_diagnostics),
    }


def evaluate_record_stream_for_eta_pairs(
    records: Iterable[Mapping[str, Any]],
    bank: Mapping[str, Any],
    *,
    eta_pairs: Sequence[tuple[float, float]],
    guard_min: float = 0.90,
    proposal_clip_scale: float = 1.0,
) -> tuple[list[dict[str, Any]], str]:
    pairs = [(float(a), float(b)) for a, b in eta_pairs]
    if not pairs:
        raise ValueError("M3_13 eta evaluation requires at least one eta pair")
    base_by_region: dict[str, dict[str, dict[str, list[float]]]] = {}
    routed_by_pair: list[dict[str, dict[str, dict[str, list[float]]]]] = [{} for _pair in pairs]
    guard_stats = [
        {
            "guard_coverage_surface": [],
            "guard_coverage_rootzone": [],
            "conflict_surface": [],
            "conflict_rootzone": [],
        }
        for _pair in pairs
    ]
    hash_rows: list[dict[str, Any]] = []
    n_records = 0
    for record in records:
        _require_roles([record], SOURCE_ROLES_FOR_SELECTION, purpose="M3_13 eta selection")
        _require_no_target_records([record], purpose="M3_13 eta selection")
        for key in ("pred_increment_surface", "pred_increment_rootzone", "source_base_increment_surface", "source_base_increment_rootzone"):
            if key not in record:
                raise KeyError(f"M3_13 source_val record missing {key!r}")
        n_records += 1
        hash_rows.append(
            {
                "sample_idx": record.get("sample_idx"),
                "split_role": record.get("split_role"),
                "query_time_index": record.get("query_time_index"),
                "query_date": record.get("query_date"),
                "sample_region_id": record.get("sample_region_id"),
                "target_region_id": record.get("target_region_id"),
            }
        )
        base_pred = {
            "pred_increment_surface": np.asarray(record["pred_increment_surface"], dtype=np.float32),
            "pred_increment_rootzone": np.asarray(record["pred_increment_rootzone"], dtype=np.float32),
            "pred_analysis_surface": np.asarray(record["forecast_surface"], dtype=np.float32)
            + np.asarray(record["pred_increment_surface"], dtype=np.float32),
            "pred_analysis_rootzone": np.asarray(record["forecast_rootzone"], dtype=np.float32)
            + np.asarray(record["pred_increment_rootzone"], dtype=np.float32),
        }
        source_base_pred = {
            "pred_increment_surface": np.asarray(record["source_base_increment_surface"], dtype=np.float32),
            "pred_increment_rootzone": np.asarray(record["source_base_increment_rootzone"], dtype=np.float32),
            "pred_analysis_surface": np.asarray(record["forecast_surface"], dtype=np.float32)
            + np.asarray(record["source_base_increment_surface"], dtype=np.float32),
            "pred_analysis_rootzone": np.asarray(record["forecast_rootzone"], dtype=np.float32)
            + np.asarray(record["source_base_increment_rootzone"], dtype=np.float32),
        }
        _accumulate_prediction_metrics(base_by_region, record, base_pred)
        for pair_idx, (eta_s, eta_r) in enumerate(pairs):
            routed = apply_phys_gain_guard(
                record,
                base_pred,
                source_base_pred,
                bank,
                eta_surface=eta_s,
                eta_rootzone=eta_r,
                guard_min=guard_min,
                proposal_clip_scale=proposal_clip_scale,
            )
            summary = routed.get("m3_13_guard_summary", {})
            query = summary.get("query", {}) if isinstance(summary, Mapping) else {}
            guard_stats[pair_idx]["guard_coverage_surface"].append(float(summary.get("guard_coverage_surface", 0.0)))
            guard_stats[pair_idx]["guard_coverage_rootzone"].append(float(summary.get("guard_coverage_rootzone", 0.0)))
            guard_stats[pair_idx]["conflict_surface"].append(float(query.get("conflict_surface", 0.0) if isinstance(query, Mapping) else 0.0))
            guard_stats[pair_idx]["conflict_rootzone"].append(float(query.get("conflict_rootzone", 0.0) if isinstance(query, Mapping) else 0.0))
            _accumulate_prediction_metrics(routed_by_pair[pair_idx], record, routed)
    if n_records == 0:
        raise ValueError("M3_13 eta evaluation received zero source_val records")
    base_summary, base_region_summary = _summarize_metric_block(base_by_region)
    results = []
    for pair_idx, ((eta_s, eta_r), routed_block) in enumerate(zip(pairs, routed_by_pair)):
        routed_summary, routed_region_summary = _summarize_metric_block(routed_block)

        def mean_stat(name: str) -> float:
            vals = [v for v in guard_stats[pair_idx][name] if math.isfinite(float(v))]
            return float(np.mean(vals)) if vals else 0.0

        results.append(
            _result_from_summaries(
                eta_surface=eta_s,
                eta_rootzone=eta_r,
                base_summary=base_summary,
                base_region_summary=base_region_summary,
                routed_summary=routed_summary,
                routed_region_summary=routed_region_summary,
                guard_diagnostics={
                    "guard_coverage_surface": mean_stat("guard_coverage_surface"),
                    "guard_coverage_rootzone": mean_stat("guard_coverage_rootzone"),
                    "conflict_coverage_surface": mean_stat("conflict_surface"),
                    "conflict_coverage_rootzone": mean_stat("conflict_rootzone"),
                },
            )
        )
    return results, _stable_hash(hash_rows)


def _source_gate_report(
    result: Mapping[str, Any],
    *,
    min_dual_cvar_delta: float,
    max_variable_rmse_relative_degrade: float,
    max_region_rmse_relative_degrade: float,
) -> dict[str, Any]:
    eta_positive = bool(float(result["eta_surface"]) > 0.0 or float(result["eta_rootzone"]) > 0.0)
    cvar_delta = float(result.get("dual_variable_cvar_delta", float("nan")))
    cvar_ok = math.isfinite(cvar_delta) and cvar_delta >= float(min_dual_cvar_delta)
    rmse_ok = all(
        (
            math.isfinite(float(result["deltas"][variable]["analysis_rmse_latw_relative"]))
            and float(result["deltas"][variable]["analysis_rmse_latw_relative"])
            <= float(max_variable_rmse_relative_degrade)
        )
        for variable in VARIABLES
    )
    region_degrade = float(result.get("max_source_region_rmse_relative_degrade", float("nan")))
    region_ok = math.isfinite(region_degrade) and region_degrade <= float(max_region_rmse_relative_degrade)
    return {
        "eta_positive": eta_positive,
        "dual_variable_cvar_delta_ok": cvar_ok,
        "variable_rmse_non_degrade_ok": rmse_ok,
        "source_region_rmse_non_degrade_ok": region_ok,
        "source_gate_pass": bool(eta_positive and cvar_ok and rmse_ok and region_ok),
        "thresholds": {
            "min_dual_cvar_delta": float(min_dual_cvar_delta),
            "max_variable_rmse_relative_degrade": float(max_variable_rmse_relative_degrade),
            "max_region_rmse_relative_degrade": float(max_region_rmse_relative_degrade),
            "requires_at_least_one_positive_eta": True,
        },
    }


def select_eta_from_source_val(
    records: Iterable[Mapping[str, Any]],
    bank: Mapping[str, Any],
    *,
    eta_grid: Sequence[float] = (0.0, 0.02, 0.05, 0.10),
    guard_min: float = 0.90,
    proposal_clip_scale: float = 1.0,
    min_dual_cvar_delta: float = -0.001,
    max_variable_rmse_relative_degrade: float = 0.001,
    max_region_rmse_relative_degrade: float = 0.003,
) -> dict[str, Any]:
    eta_values = [float(eta) for eta in eta_grid]
    eta_pairs = [(eta_s, eta_r) for eta_s in eta_values for eta_r in eta_values]
    evaluated, records_hash = evaluate_record_stream_for_eta_pairs(
        records,
        bank,
        eta_pairs=eta_pairs,
        guard_min=guard_min,
        proposal_clip_scale=proposal_clip_scale,
    )
    passing = []
    for result in evaluated:
        report = _source_gate_report(
            result,
            min_dual_cvar_delta=min_dual_cvar_delta,
            max_variable_rmse_relative_degrade=max_variable_rmse_relative_degrade,
            max_region_rmse_relative_degrade=max_region_rmse_relative_degrade,
        )
        result["source_gate_report"] = report
        result["source_gate_pass"] = bool(report["source_gate_pass"])
        if result["source_gate_pass"]:
            passing.append(result)
    if passing:
        selected = max(
            passing,
            key=lambda item: (
                float(item["dual_variable_cvar"]["dual_variable_cvar_safe_score"]),
                -max(
                    float(item["deltas"]["surface"]["analysis_rmse_latw_relative"]),
                    float(item["deltas"]["rootzone"]["analysis_rmse_latw_relative"]),
                ),
                -float(item["eta_surface"]) - float(item["eta_rootzone"]),
            ),
        )
        identity_diagnostic = False
    else:
        selected = next(
            (
                item
                for item in evaluated
                if float(item["eta_surface"]) == 0.0 and float(item["eta_rootzone"]) == 0.0
            ),
            evaluated[0],
        )
        identity_diagnostic = True
    selection = {
        "schema_version": PHYS_GAIN_GUARD_SELECTION_SCHEMA,
        "method_id": PHYS_GAIN_GUARD_METHOD_ID,
        "eta_grid": eta_values,
        "eta_pair_grid": [[float(a), float(b)] for a in eta_values for b in eta_values],
        "selected_eta_surface": float(selected["eta_surface"]),
        "selected_eta_rootzone": float(selected["eta_rootzone"]),
        "guard_min": float(guard_min),
        "source_gate_pass": bool(selected.get("source_gate_pass", False)),
        "identity_diagnostic": bool(identity_diagnostic),
        "source_gate_report": selected.get("source_gate_report", {}),
        "selection_source": "source_val_only",
        "target_val_usage": "unused_in_main_protocol",
        "target_eval_usage": "not_used_for_eta_selection",
        "target_eval_final_policy": "run_once_only_if_source_gate_passes",
        "source_val_records_hash": records_hash,
        "source_gain_bank_hash": bank.get("source_gain_bank_hash", bank.get("bank_content_hash", "")),
        "formula_schema": phys_gain_basis_formula_schema(),
        "selection_rule": {
            "min_dual_cvar_delta": float(min_dual_cvar_delta),
            "max_variable_rmse_relative_degrade": float(max_variable_rmse_relative_degrade),
            "max_region_rmse_relative_degrade": float(max_region_rmse_relative_degrade),
            "requires_at_least_one_positive_eta": True,
        },
        "selected": selected,
        "grid": evaluated,
    }
    selection["selection_hash"] = _stable_hash(
        {
            "schema_version": selection["schema_version"],
            "method_id": selection["method_id"],
            "eta_grid": selection["eta_grid"],
            "selected_eta_surface": selection["selected_eta_surface"],
            "selected_eta_rootzone": selection["selected_eta_rootzone"],
            "source_gate_pass": selection["source_gate_pass"],
            "identity_diagnostic": selection["identity_diagnostic"],
            "source_val_records_hash": selection["source_val_records_hash"],
            "source_gain_bank_hash": selection["source_gain_bank_hash"],
            "selection_rule": selection["selection_rule"],
        }
    )
    return selection


def validate_router_metadata_no_target_selection(metadata: Mapping[str, Any]) -> None:
    if metadata.get("target_eval_usage") not in {
        "not_used_for_eta_selection",
        "final_eval_only_no_selection",
    }:
        raise ValueError("M3_13 metadata indicates target_eval selection")
    if metadata.get("target_val_usage") not in {"unused", "unused_in_main_protocol", None}:
        raise ValueError("M3_13 metadata indicates target_val usage")


def validate_source_gate_for_target_eval(selection: Mapping[str, Any]) -> None:
    validate_router_metadata_no_target_selection(selection)
    if selection.get("schema_version") != PHYS_GAIN_GUARD_SELECTION_SCHEMA:
        raise ValueError(f"Unsupported M3_13 source-gate schema: {selection.get('schema_version')!r}")
    if selection.get("method_id") != PHYS_GAIN_GUARD_METHOD_ID:
        raise ValueError("M3_13 source-gate method_id mismatch")
    if selection.get("selection_source") != "source_val_only":
        raise ValueError("M3_13 target_eval requires source_val-only eta selection")
    eta_positive = (
        float(selection.get("selected_eta_surface", 0.0)) > 0.0
        or float(selection.get("selected_eta_rootzone", 0.0)) > 0.0
    )
    if not eta_positive:
        raise ValueError("M3_13 target_eval refused: no positive eta was selected")
    if bool(selection.get("identity_diagnostic", False)):
        raise ValueError("M3_13 target_eval refused: identity diagnostic fallback")
    if not bool(selection.get("source_gate_pass", False)):
        raise ValueError("M3_13 target_eval refused: source gate did not pass")
