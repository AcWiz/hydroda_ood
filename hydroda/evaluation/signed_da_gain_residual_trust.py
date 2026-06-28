"""M3_11 signed DA-gain residual trust router.

This module keeps the M3_1 predictor frozen. It builds only source_fit
covariance statistics, proposes signed DA increments from raw TB innovations,
and applies a source_val-selected bounded residual blend.
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


VARIABLES = ("surface", "rootzone")
SOURCE_ROLES_FOR_BANK = {"source_fit"}
SOURCE_ROLES_FOR_SELECTION = {"source_val", "source_val_pseudo_query"}
FORBIDDEN_TARGET_ROLES = {"target_val", "target_eval", "target_query", "target_full_train"}
SIGNED_DA_GAIN_BANK_SCHEMA = "m3_11_source_signed_da_gain_bank_v1"
SIGNED_DA_GAIN_SELECTION_SCHEMA = "m3_11_signed_da_gain_residual_trust_selection_v1"
SIGNED_DA_GAIN_ROUTER_SCHEMA = "m3_11_signed_da_gain_residual_trust_router_v1"
SIGNED_DA_GAIN_METHOD_ID = "M3_11_signed_da_gain_residual_trust"
SOURCE_REGIONS = ("US-R1", "US-R2", "US-R3", "US-R4", "US-R5", "US-R6")


@dataclass(frozen=True)
class GainEstimate:
    cov: float
    var: float
    gain: float
    n: int


@dataclass
class GainMoments:
    """Online moments for Cov(y, d) / Var(d)."""

    n: int = 0
    sum_y: float = 0.0
    sum_d: float = 0.0
    sum_yd: float = 0.0
    sum_dd: float = 0.0

    def update(self, y: np.ndarray, d: np.ndarray) -> None:
        yv = np.asarray(y, dtype=np.float64).reshape(-1)
        dv = np.asarray(d, dtype=np.float64).reshape(-1)
        valid = np.isfinite(yv) & np.isfinite(dv)
        if not np.any(valid):
            return
        yv = yv[valid]
        dv = dv[valid]
        self.n += int(yv.size)
        self.sum_y += float(np.sum(yv))
        self.sum_d += float(np.sum(dv))
        self.sum_yd += float(np.sum(yv * dv))
        self.sum_dd += float(np.sum(dv * dv))

    def estimate(self, *, ridge_lambda: float) -> GainEstimate:
        if self.n <= 1:
            return GainEstimate(cov=0.0, var=0.0, gain=0.0, n=int(self.n))
        n = float(self.n)
        mean_y = self.sum_y / n
        mean_d = self.sum_d / n
        cov = float(self.sum_yd / n - mean_y * mean_d)
        var = float(self.sum_dd / n - mean_d * mean_d)
        if not math.isfinite(var) or var < 0.0:
            var = 0.0
        denom = var + float(ridge_lambda)
        gain = float(cov / denom) if denom > 0.0 else 0.0
        if not math.isfinite(gain):
            gain = 0.0
        return GainEstimate(cov=cov, var=var, gain=gain, n=int(self.n))


@dataclass
class AbsMoments:
    n: int = 0
    sum_abs: float = 0.0

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return
        self.n += int(valid.size)
        self.sum_abs += float(np.sum(np.abs(valid)))

    def mean_abs(self) -> float:
        return float(self.sum_abs / self.n) if self.n > 0 else 0.0


@dataclass
class GainBucket:
    surface_h: GainMoments = field(default_factory=GainMoments)
    surface_v: GainMoments = field(default_factory=GainMoments)
    rootzone_h: GainMoments = field(default_factory=GainMoments)
    rootzone_v: GainMoments = field(default_factory=GainMoments)
    coupling: GainMoments = field(default_factory=GainMoments)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _as_float_array(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    return arr


def _input_array_from_record(record: Mapping[str, Any]) -> np.ndarray:
    if "x_raw" in record and record["x_raw"] is not None:
        return _as_float_array(record["x_raw"], name="x_raw")
    return _as_float_array(record["x"], name="x")


def _masked_pair(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    m = np.asarray(mask) > 0.5
    valid = m & np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        empty = np.empty((0,), dtype=np.float64)
        return empty, empty
    return a[valid].reshape(-1), b[valid].reshape(-1)


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


def da_innovations_from_x(
    x: Any,
    *,
    eps: float = 1e-6,
    tb_h_obs_channel: int = 5,
    tb_v_obs_channel: int = 6,
    tb_h_obs_err_channel: int = 7,
    tb_v_obs_err_channel: int = 8,
    tb_h_assim_channel: int = 9,
    tb_v_assim_channel: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return signed normalized TB observation-minus-assimilated innovations."""
    arr = _as_float_array(x, name="x")
    if arr.ndim != 3:
        raise ValueError(f"x must have shape [C,H,W], got {arr.shape}")
    min_channels = max(
        tb_h_obs_channel,
        tb_v_obs_channel,
        tb_h_obs_err_channel,
        tb_v_obs_err_channel,
        tb_h_assim_channel,
        tb_v_assim_channel,
    ) + 1
    if arr.shape[0] < min_channels:
        raise ValueError(f"x has {arr.shape[0]} channels, expected at least {min_channels}")
    obs_h = arr[tb_h_obs_channel]
    obs_v = arr[tb_v_obs_channel]
    err_h = np.maximum(np.abs(arr[tb_h_obs_err_channel]), float(eps))
    err_v = np.maximum(np.abs(arr[tb_v_obs_err_channel]), float(eps))
    assim_h = arr[tb_h_assim_channel]
    assim_v = arr[tb_v_assim_channel]
    d_h = ((obs_h - assim_h) / (err_h + float(eps))).astype(np.float32)
    d_v = ((obs_v - assim_v) / (err_v + float(eps))).astype(np.float32)
    return d_h, d_v


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
    month_int = int(month)
    if month_int < 1 or month_int > 12:
        raise ValueError(f"month must be in 1..12, got {month!r}")
    return month_int


def sample_arrays_for_gain(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    required = ("increment_surface", "increment_rootzone", "metric_mask")
    missing = [key for key in required if key not in record]
    if "x" not in record and "x_raw" not in record:
        missing.append("x_or_x_raw")
    if missing:
        raise KeyError(f"sample record missing required arrays for signed DA gain bank: {missing}")
    return {
        "x": _input_array_from_record(record),
        "increment_surface": np.asarray(record["increment_surface"], dtype=np.float32),
        "increment_rootzone": np.asarray(record["increment_rootzone"], dtype=np.float32),
        "metric_mask": np.asarray(record["metric_mask"], dtype=np.float32),
    }


def _entry_from_bucket(region: str, month: int, bucket: GainBucket, *, ridge_lambda: float) -> dict[str, Any]:
    gain_s_h = bucket.surface_h.estimate(ridge_lambda=ridge_lambda)
    gain_s_v = bucket.surface_v.estimate(ridge_lambda=ridge_lambda)
    gain_r_h = bucket.rootzone_h.estimate(ridge_lambda=ridge_lambda)
    gain_r_v = bucket.rootzone_v.estimate(ridge_lambda=ridge_lambda)
    coupling = bucket.coupling.estimate(ridge_lambda=ridge_lambda)
    return {
        "source_region": region,
        "month": int(month),
        "n_pixels": int(min(gain_s_h.n, gain_s_v.n, gain_r_h.n, gain_r_v.n)),
        "gains": {
            "surface": {"H": gain_s_h.gain, "V": gain_s_v.gain},
            "rootzone": {"H": gain_r_h.gain, "V": gain_r_v.gain},
        },
        "stats": {
            "surface": {"H": gain_s_h.__dict__, "V": gain_s_v.__dict__},
            "rootzone": {"H": gain_r_h.__dict__, "V": gain_r_v.__dict__},
            "rootzone_surface_coupling": coupling.__dict__,
        },
        "C_rz": coupling.gain,
    }


def _consensus_entry(
    entry_list: Sequence[Mapping[str, Any]],
    weights: Sequence[float] | None = None,
    *,
    region: str | None,
    month: int | None,
    fallback_level: str,
) -> dict[str, Any]:
    resolved_weights = list(weights) if weights is not None else [int(entry.get("n_pixels", 0)) for entry in entry_list]
    if len(resolved_weights) != len(entry_list):
        raise ValueError("consensus weights and entries have different lengths")

    def mean_gain(variable: str, pol: str) -> float:
        values = [
            float(((entry.get("gains", {}) or {}).get(variable, {}) or {}).get(pol, 0.0))
            for entry in entry_list
        ]
        return _weighted_mean(values, resolved_weights)

    return {
        "source_region": region or "source_consensus",
        "month": int(month) if month is not None else None,
        "n_pixels": int(sum(int(entry.get("n_pixels", 0) or 0) for entry in entry_list)),
        "gains": {
            "surface": {"H": mean_gain("surface", "H"), "V": mean_gain("surface", "V")},
            "rootzone": {"H": mean_gain("rootzone", "H"), "V": mean_gain("rootzone", "V")},
        },
        "C_rz": _weighted_mean([float(entry.get("C_rz", 0.0) or 0.0) for entry in entry_list], resolved_weights),
        "fallback_level": fallback_level,
    }


class SignedDAGainBankAccumulator:
    """Streaming source_fit signed DA-gain bank accumulator."""

    def __init__(
        self,
        *,
        ridge_lambda: float = 1e-3,
        eps: float = 1e-6,
        source_checkpoint: str = "",
        split_manifest: str = "",
        source_neighbor_top_m: int = 4,
    ) -> None:
        self.ridge_lambda = float(ridge_lambda)
        self.eps = float(eps)
        self.source_checkpoint = str(source_checkpoint or "")
        self.split_manifest = str(split_manifest or "")
        self.source_neighbor_top_m = int(source_neighbor_top_m)
        if self.source_neighbor_top_m < 1:
            raise ValueError("source_neighbor_top_m must be >= 1")
        self.grouped: dict[tuple[str, int], GainBucket] = {}
        self.source_scale = {
            "surface": AbsMoments(),
            "rootzone": AbsMoments(),
            "d_H": AbsMoments(),
            "d_V": AbsMoments(),
        }
        self.n_records_seen = 0
        self.n_records_used = 0

    def update(self, record: Mapping[str, Any]) -> None:
        _require_roles([record], SOURCE_ROLES_FOR_BANK, purpose="M3_11 gain bank construction")
        _require_no_target_records([record], purpose="M3_11 gain bank construction")
        self.n_records_seen += 1
        arrays = sample_arrays_for_gain(record)
        mask = np.asarray(arrays["metric_mask"], dtype=np.float32)
        d_h, d_v = da_innovations_from_x(arrays["x"], eps=self.eps)
        inc_s = np.asarray(arrays["increment_surface"], dtype=np.float32)
        inc_r = np.asarray(arrays["increment_rootzone"], dtype=np.float32)
        s_h_values, h_values = _masked_pair(inc_s, d_h, mask)
        r_h_values, h_r_values = _masked_pair(inc_r, d_h, mask)
        s_v_values, v_values = _masked_pair(inc_s, d_v, mask)
        r_v_values, v_r_values = _masked_pair(inc_r, d_v, mask)
        r_c_values, s_c_values = _masked_pair(inc_r, inc_s, mask)
        if s_h_values.size == 0 or r_h_values.size == 0:
            return
        key = (source_region_from_record(record), month_from_record(record))
        bucket = self.grouped.setdefault(key, GainBucket())
        bucket.surface_h.update(s_h_values, h_values)
        bucket.surface_v.update(s_v_values, v_values)
        bucket.rootzone_h.update(r_h_values, h_r_values)
        bucket.rootzone_v.update(r_v_values, v_r_values)
        bucket.coupling.update(r_c_values, s_c_values)
        self.source_scale["surface"].update(s_h_values)
        self.source_scale["rootzone"].update(r_h_values)
        self.source_scale["d_H"].update(h_values)
        self.source_scale["d_V"].update(v_values)
        self.n_records_used += 1

    def finalize(self) -> dict[str, Any]:
        if not self.grouped:
            raise ValueError("No valid source_fit pixels available for M3_11 signed DA gain bank")

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
        global_entry = _consensus_entry(
            list(entries.values()),
            region=None,
            month=None,
            fallback_level="source_global_consensus",
        )
        source_scale = {key: moments.mean_abs() for key, moments in self.source_scale.items()}
        bank = {
            "schema_version": SIGNED_DA_GAIN_BANK_SCHEMA,
            "method_id": SIGNED_DA_GAIN_METHOD_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_split_roles": {
                "bank": sorted(SOURCE_ROLES_FOR_BANK),
                "eta_selection": sorted(SOURCE_ROLES_FOR_SELECTION),
            },
            "source_label_usage": "source_fit_labels_only",
            "bank_split_roles": sorted(SOURCE_ROLES_FOR_BANK),
            "target_val_usage": "unused",
            "target_eval_usage": "not_used_for_bank_or_eta_selection",
            "target_eval_selection_usage": "forbidden",
            "neural_training_epochs": 0,
            "neural_parameter_updates": 0,
            "ridge_lambda": float(self.ridge_lambda),
            "eps": float(self.eps),
            "source_neighbor_top_m": int(self.source_neighbor_top_m),
            "gain_source_priority": [
                "source_trust_top_m_neighbor_weighted_consensus",
                "source_month_consensus",
                "source_global_consensus",
            ],
            "formula": {
                "d_H": "(tb_h_obs - tb_h_obs_assim) / (tb_h_obs_errstd + eps)",
                "d_V": "(tb_v_obs - tb_v_obs_assim) / (tb_v_obs_errstd + eps)",
                "G_v_p": "Cov_source(DeltaSM_v, d_p) / (Var_source(d_p) + lambda)",
                "C_rz": "Cov_source(DeltaSM_rootzone, DeltaSM_surface) / (Var_source(DeltaSM_surface) + lambda)",
                "q_surface": "clip(G_s_H*d_H + G_s_V*d_V)",
                "q_rootzone": "clip(G_r_H*d_H + G_r_V*d_V + C_rz*q_surface)",
                "final": "pred_M3_1_v + eta_v * conf_v(x) * clip(q_v - pred_M3_1_v)",
            },
            "confidence_formula": "finite_signed_tb_innovation_and_region_mask_only",
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
            "source_scale": source_scale,
            "entries": entries,
            "month_consensus": month_consensus,
            "global_consensus": global_entry,
            "n_source_records_seen": int(self.n_records_seen),
            "n_source_records_used": int(self.n_records_used),
            "accumulator": "streaming_covariance_moments_v1",
            "eta_grid": [],
            "selection_hash": "",
        }
        bank["bank_content_hash"] = _stable_hash(
            {
                "schema_version": bank["schema_version"],
                "method_id": bank["method_id"],
                "ridge_lambda": bank["ridge_lambda"],
                "eps": bank["eps"],
                "source_neighbor_top_m": bank["source_neighbor_top_m"],
                "entries": bank["entries"],
                "month_consensus": bank["month_consensus"],
                "global_consensus": bank["global_consensus"],
                "formula": bank["formula"],
            }
        )
        return bank


def build_source_signed_da_gain_bank_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    ridge_lambda: float = 1e-3,
    eps: float = 1e-6,
    source_checkpoint: str = "",
    split_manifest: str = "",
    source_neighbor_top_m: int = 4,
) -> dict[str, Any]:
    accumulator = SignedDAGainBankAccumulator(
        ridge_lambda=ridge_lambda,
        eps=eps,
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
    if bank.get("schema_version") != SIGNED_DA_GAIN_BANK_SCHEMA:
        raise ValueError(f"Unsupported M3_11 signed DA gain bank schema: {bank.get('schema_version')!r}")
    if bank.get("method_id") != SIGNED_DA_GAIN_METHOD_ID:
        raise ValueError("M3_11 gain bank method_id mismatch")
    if bank.get("source_label_usage") != "source_fit_labels_only":
        raise ValueError("M3_11 gain bank must declare source_fit_labels_only label usage")
    if bank.get("target_eval_usage") != "not_used_for_bank_or_eta_selection":
        raise ValueError("M3_11 gain bank must not use target_eval for bank or eta selection")
    if "entries" not in bank or not isinstance(bank["entries"], Mapping):
        raise ValueError("M3_11 gain bank missing entries")


def _neighbor_region_month_weight(
    item: Any,
    *,
    default_month: int,
    rank: int,
) -> tuple[str, int, float] | None:
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
    if not region:
        return None
    if month < 1 or month > 12:
        return None
    weight = item.get("weight", item.get("trust_weight", item.get("neighbor_weight")))
    if weight is None:
        distance = item.get("distance", item.get("neighbor_distance"))
        if distance is not None and math.isfinite(float(distance)):
            weight = 1.0 / max(float(distance), 1e-6)
        else:
            weight = 1.0 / float(rank + 1)
    weight_f = float(weight)
    if not math.isfinite(weight_f) or weight_f <= 0.0:
        return None
    return region, month, weight_f


def _entry_from_neighbors(
    bank: Mapping[str, Any],
    sample: Mapping[str, Any],
    *,
    month: int,
) -> tuple[Mapping[str, Any] | None, str]:
    neighbors = sample.get("source_trust_neighbors") or sample.get("nearest_source_neighbors") or []
    if not isinstance(neighbors, Sequence) or isinstance(neighbors, (str, bytes, bytearray)):
        return None, ""
    top_m = int(bank.get("source_neighbor_top_m", 4) or 4)
    entry_weight_pairs: list[tuple[Mapping[str, Any], float]] = []
    entries = bank.get("entries", {})
    for rank, item in enumerate(list(neighbors)[:top_m]):
        parsed = _neighbor_region_month_weight(item, default_month=month, rank=rank)
        if parsed is None:
            continue
        region, neighbor_month, weight = parsed
        key = f"{region}|{neighbor_month:02d}"
        if key in entries:
            entry_weight_pairs.append((entries[key], weight))
    if not entry_weight_pairs:
        return None, ""
    entry = _consensus_entry(
        [entry for entry, _weight in entry_weight_pairs],
        [weight for _entry, weight in entry_weight_pairs],
        region=None,
        month=month,
        fallback_level="source_trust_top_m_neighbor_weighted_consensus",
    )
    entry["neighbor_count"] = len(entry_weight_pairs)
    return entry, "source_trust_top_m_neighbor_weighted_consensus"


def select_gain_entry(bank: Mapping[str, Any], sample: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    validate_gain_bank_metadata(bank)
    month = month_from_record(sample)
    neighbor_entry, neighbor_level = _entry_from_neighbors(bank, sample, month=month)
    if neighbor_entry is not None:
        return neighbor_entry, neighbor_level
    month_key = f"{int(month):02d}"
    month_consensus = bank.get("month_consensus", {})
    if month_key in month_consensus:
        return month_consensus[month_key], "source_month_consensus"
    return bank.get("global_consensus", {}), "source_global_consensus"


def confidence_from_sample(sample: Mapping[str, Any], *, eps: float = 1e-6) -> dict[str, np.ndarray]:
    x = _input_array_from_record(sample)
    d_h, d_v = da_innovations_from_x(x, eps=eps)
    conf = (np.isfinite(d_h) & np.isfinite(d_v)).astype(np.float32)
    region_mask = sample.get("region_mask", sample.get("active_region_mask"))
    if region_mask is not None:
        conf = (conf * (np.asarray(region_mask, dtype=np.float32) > 0.5).astype(np.float32)).astype(np.float32)
    return {"surface": conf.copy(), "rootzone": conf.copy()}


def physical_proposal_from_sample(
    sample: Mapping[str, Any],
    bank: Mapping[str, Any],
    *,
    proposal_clip_scale: float = 1.0,
    eps: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build bounded signed DA increment proposals for one sample."""
    x = _input_array_from_record(sample)
    entry, fallback_level = select_gain_entry(bank, sample)
    d_h, d_v = da_innovations_from_x(x, eps=float(eps if eps is not None else bank.get("eps", 1e-6)))
    gains = entry.get("gains", {})
    g_s = gains.get("surface", {})
    g_r = gains.get("rootzone", {})
    raw_surface = (
        float(g_s.get("H", 0.0)) * d_h
        + float(g_s.get("V", 0.0)) * d_v
    ).astype(np.float32)
    raw_rootzone_direct = (
        float(g_r.get("H", 0.0)) * d_h
        + float(g_r.get("V", 0.0)) * d_v
    ).astype(np.float32)
    source_scale = bank.get("source_scale", {})
    clip_s = max(float(source_scale.get("surface", 0.0)), 0.0) * float(proposal_clip_scale)
    clip_r = max(float(source_scale.get("rootzone", 0.0)), 0.0) * float(proposal_clip_scale)
    surface = np.clip(raw_surface, -clip_s, clip_s).astype(np.float32) if clip_s > 0.0 else raw_surface
    raw_rootzone = (raw_rootzone_direct + float(entry.get("C_rz", 0.0)) * surface).astype(np.float32)
    rootzone = np.clip(raw_rootzone, -clip_r, clip_r).astype(np.float32) if clip_r > 0.0 else raw_rootzone
    summary = {
        "fallback_level": fallback_level,
        "bank_region": entry.get("source_region", ""),
        "bank_month": entry.get("month"),
        "neighbor_count": int(entry.get("neighbor_count", 0) or 0),
        "n_pixels": int(entry.get("n_pixels", 0) or 0),
        "clip_surface": float(clip_s),
        "clip_rootzone": float(clip_r),
        "gains": entry.get("gains", {}),
        "C_rz": float(entry.get("C_rz", 0.0) or 0.0),
    }
    return {"surface": surface, "rootzone": rootzone}, summary


def blend_prediction_with_signed_da_gain(
    sample: Mapping[str, Any],
    base_pred: Mapping[str, Any],
    bank: Mapping[str, Any],
    *,
    eta_surface: float = 0.0,
    eta_rootzone: float | None = None,
    proposal_clip_scale: float = 1.0,
    residual_clip_scale: float | None = None,
) -> dict[str, Any]:
    """Return base prediction plus bounded signed DA-gain residual."""
    eta_s = float(eta_surface)
    eta_r = eta_s if eta_rootzone is None else float(eta_rootzone)
    if not 0.0 <= eta_s <= 1.0 or not 0.0 <= eta_r <= 1.0:
        raise ValueError("eta values must be in [0, 1]")
    if eta_s == 0.0 and eta_r == 0.0:
        return {
            "pred_increment_surface": np.asarray(base_pred["pred_increment_surface"], dtype=np.float32).copy(),
            "pred_increment_rootzone": np.asarray(base_pred["pred_increment_rootzone"], dtype=np.float32).copy(),
            "pred_analysis_surface": np.asarray(base_pred["pred_analysis_surface"], dtype=np.float32).copy(),
            "pred_analysis_rootzone": np.asarray(base_pred["pred_analysis_rootzone"], dtype=np.float32).copy(),
            "m3_11_eta_surface": eta_s,
            "m3_11_eta_rootzone": eta_r,
            "m3_11_fallback_level": "eta_zero_no_proposal",
        }
    proposal, summary = physical_proposal_from_sample(
        sample,
        bank,
        proposal_clip_scale=proposal_clip_scale,
    )
    conf = confidence_from_sample(sample, eps=float(bank.get("eps", 1e-6)))
    residual_scale = float(proposal_clip_scale if residual_clip_scale is None else residual_clip_scale)
    source_scale = bank.get("source_scale", {})
    base_s = np.asarray(base_pred["pred_increment_surface"], dtype=np.float32)
    base_r = np.asarray(base_pred["pred_increment_rootzone"], dtype=np.float32)
    diff_s = (proposal["surface"] - base_s).astype(np.float32)
    diff_r = (proposal["rootzone"] - base_r).astype(np.float32)
    clip_s = max(float(source_scale.get("surface", 0.0)), 0.0) * residual_scale
    clip_r = max(float(source_scale.get("rootzone", 0.0)), 0.0) * residual_scale
    if clip_s > 0.0:
        diff_s = np.clip(diff_s, -clip_s, clip_s).astype(np.float32)
    if clip_r > 0.0:
        diff_r = np.clip(diff_r, -clip_r, clip_r).astype(np.float32)
    final_s = (base_s + eta_s * conf["surface"] * diff_s).astype(np.float32)
    final_r = (base_r + eta_r * conf["rootzone"] * diff_r).astype(np.float32)
    forecast_s = np.asarray(sample["forecast_surface"], dtype=np.float32)
    forecast_r = np.asarray(sample["forecast_rootzone"], dtype=np.float32)
    return {
        "pred_increment_surface": final_s,
        "pred_increment_rootzone": final_r,
        "pred_analysis_surface": (forecast_s + final_s).astype(np.float32),
        "pred_analysis_rootzone": (forecast_r + final_r).astype(np.float32),
        "m3_11_eta_surface": eta_s,
        "m3_11_eta_rootzone": eta_r,
        "m3_11_fallback_level": summary["fallback_level"],
        "m3_11_summary": summary,
    }


class SignedDAGainResidualTrustPredictor:
    """Predictor wrapper applying the M3_11 post-hoc residual route."""

    def __init__(
        self,
        base_predictor: Any,
        bank: Mapping[str, Any],
        *,
        eta_surface: float = 0.0,
        eta_rootzone: float | None = None,
        proposal_clip_scale: float = 1.0,
        residual_clip_scale: float | None = None,
        method_name: str = SIGNED_DA_GAIN_METHOD_ID,
    ) -> None:
        validate_gain_bank_metadata(bank)
        self.base_predictor = base_predictor
        self.bank = dict(bank)
        self.eta_surface = float(eta_surface)
        self.eta_rootzone = self.eta_surface if eta_rootzone is None else float(eta_rootzone)
        self.proposal_clip_scale = float(proposal_clip_scale)
        self.residual_clip_scale = float(
            self.proposal_clip_scale if residual_clip_scale is None else residual_clip_scale
        )
        self.method_name = method_name
        self.metadata = {
            "schema_version": SIGNED_DA_GAIN_ROUTER_SCHEMA,
            "method_id": SIGNED_DA_GAIN_METHOD_ID,
            "base_method": getattr(base_predictor, "method_name", "unknown"),
            "base_anchor": "M3_1_hyperda_trust_medium",
            "eta_surface": self.eta_surface,
            "eta_rootzone": self.eta_rootzone,
            "proposal_clip_scale": self.proposal_clip_scale,
            "residual_clip_scale": self.residual_clip_scale,
            "bank_content_hash": self.bank.get("bank_content_hash", ""),
            "neural_training_epochs": 0,
            "neural_parameter_updates": 0,
            "target_val_usage": "unused",
            "target_eval_usage": "final_eval_only_no_selection",
            "channel_11_usage": "diagnostic_coverage_only_not_obs_loss_metric_region_mask",
        }

    def predict(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        base_pred = self.base_predictor.predict(sample)
        return blend_prediction_with_signed_da_gain(
            sample,
            base_pred,
            self.bank,
            eta_surface=self.eta_surface,
            eta_rootzone=self.eta_rootzone,
            proposal_clip_scale=self.proposal_clip_scale,
            residual_clip_scale=self.residual_clip_scale,
        )


def build_source_records_from_predictor(
    *,
    dataset: Any,
    predictor: Any | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Materialize source records with labels, masks, and optional base predictions."""
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
        if predictor is not None:
            pred = predictor.predict(sample)
            record["pred_increment_surface"] = np.asarray(pred["pred_increment_surface"], dtype=np.float32)
            record["pred_increment_rootzone"] = np.asarray(pred["pred_increment_rootzone"], dtype=np.float32)
        records.append(record)
    return records


def _arrayless_record_hash(records: Sequence[Mapping[str, Any]]) -> str:
    stripped = []
    for record in records:
        stripped.append(
            {
                "sample_idx": record.get("sample_idx"),
                "split_role": record.get("split_role"),
                "query_time_index": record.get("query_time_index"),
                "query_date": record.get("query_date"),
                "sample_region_id": record.get("sample_region_id"),
                "target_region_id": record.get("target_region_id"),
            }
        )
    return _stable_hash(stripped)


def _arrayless_record_hash_rows_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return _stable_hash(list(rows))


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
        score = float("-inf")
        non_degrade = False
        worst_s = float("nan")
        worst_r = float("nan")
        mean_bal = float("nan")
        pos_rate = 0.0
    else:
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


def evaluate_records_for_eta_pair(
    records: Iterable[Mapping[str, Any]],
    bank: Mapping[str, Any],
    *,
    eta_surface: float,
    eta_rootzone: float,
    proposal_clip_scale: float = 1.0,
    residual_clip_scale: float | None = None,
) -> dict[str, Any]:
    """Evaluate one (eta_surface, eta_rootzone) pair on source_val records."""
    evaluated, _records_hash = evaluate_record_stream_for_eta_pairs(
        records,
        bank,
        eta_pairs=[(float(eta_surface), float(eta_rootzone))],
        proposal_clip_scale=proposal_clip_scale,
        residual_clip_scale=residual_clip_scale,
    )
    if not evaluated:
        raise ValueError("M3_11 eta evaluation received zero source_val records")
    return evaluated[0]


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


def _result_from_summaries(
    *,
    eta_surface: float,
    eta_rootzone: float,
    base_summary: Mapping[str, Any],
    base_region_summary: Mapping[str, Mapping[str, float]],
    routed_summary: Mapping[str, Any],
    routed_region_summary: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    base_cvar = dual_variable_cvar_score_from_region_skills(base_region_summary)
    routed_cvar = dual_variable_cvar_score_from_region_skills(routed_region_summary)
    deltas: dict[str, Any] = {}
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
    }


def evaluate_record_stream_for_eta_pairs(
    records: Iterable[Mapping[str, Any]],
    bank: Mapping[str, Any],
    *,
    eta_pairs: Sequence[tuple[float, float]],
    proposal_clip_scale: float = 1.0,
    residual_clip_scale: float | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Evaluate eta pairs over a source_val record stream without retaining tensors."""
    pairs = [(float(eta_s), float(eta_r)) for eta_s, eta_r in eta_pairs]
    if not pairs:
        raise ValueError("M3_11 eta evaluation requires at least one eta pair")
    base_by_region: dict[str, dict[str, dict[str, list[float]]]] = {}
    routed_by_pair: list[dict[str, dict[str, dict[str, list[float]]]]] = [
        {} for _pair in pairs
    ]
    hash_rows: list[dict[str, Any]] = []
    n_records = 0
    for record in records:
        _require_roles([record], SOURCE_ROLES_FOR_SELECTION, purpose="M3_11 eta selection")
        _require_no_target_records([record], purpose="M3_11 eta selection")
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
        sample = dict(record)
        base_pred = {
            "pred_increment_surface": np.asarray(record["pred_increment_surface"], dtype=np.float32),
            "pred_increment_rootzone": np.asarray(record["pred_increment_rootzone"], dtype=np.float32),
            "pred_analysis_surface": np.asarray(record["forecast_surface"], dtype=np.float32)
            + np.asarray(record["pred_increment_surface"], dtype=np.float32),
            "pred_analysis_rootzone": np.asarray(record["forecast_rootzone"], dtype=np.float32)
            + np.asarray(record["pred_increment_rootzone"], dtype=np.float32),
        }
        _accumulate_prediction_metrics(base_by_region, record, base_pred)
        for pair_idx, (eta_s, eta_r) in enumerate(pairs):
            routed = blend_prediction_with_signed_da_gain(
                sample,
                base_pred,
                bank,
                eta_surface=eta_s,
                eta_rootzone=eta_r,
                proposal_clip_scale=proposal_clip_scale,
                residual_clip_scale=residual_clip_scale,
            )
            _accumulate_prediction_metrics(routed_by_pair[pair_idx], record, routed)
    if n_records == 0:
        raise ValueError("M3_11 eta evaluation received zero source_val records")
    base_summary, base_region_summary = _summarize_metric_block(base_by_region)
    results = []
    for (eta_s, eta_r), routed_block in zip(pairs, routed_by_pair):
        routed_summary, routed_region_summary = _summarize_metric_block(routed_block)
        results.append(
            _result_from_summaries(
                eta_surface=eta_s,
                eta_rootzone=eta_r,
                base_summary=base_summary,
                base_region_summary=base_region_summary,
                routed_summary=routed_summary,
                routed_region_summary=routed_region_summary,
            )
        )
    return results, _arrayless_record_hash_rows_hash(hash_rows)


def _corr_or_sign_nondecline(result: Mapping[str, Any]) -> bool:
    deltas = result.get("deltas", {})
    ok_by_variable = []
    for variable in VARIABLES:
        block = deltas.get(variable, {})
        corr = float(block.get("increment_corr_delta", float("nan")))
        sign = float(block.get("sign_accuracy_deadzone_delta", float("nan")))
        candidates = [value for value in (corr, sign) if math.isfinite(value)]
        ok_by_variable.append(bool(candidates and max(candidates) >= 0.0))
    return bool(all(ok_by_variable))


def _source_gate_report(
    result: Mapping[str, Any],
    *,
    min_dual_cvar_delta: float,
    max_variable_rmse_relative_degrade: float,
    max_region_rmse_relative_degrade: float,
    require_corr_or_sign_nondecline: bool,
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
    corr_sign_ok = _corr_or_sign_nondecline(result)
    if not require_corr_or_sign_nondecline:
        corr_sign_ok = True
    return {
        "eta_positive": eta_positive,
        "dual_variable_cvar_delta_ok": cvar_ok,
        "variable_rmse_non_degrade_ok": rmse_ok,
        "source_region_rmse_non_degrade_ok": region_ok,
        "corr_or_sign_nondecline_ok": corr_sign_ok,
        "source_gate_pass": bool(eta_positive and cvar_ok and rmse_ok and region_ok and corr_sign_ok),
        "thresholds": {
            "min_dual_cvar_delta": float(min_dual_cvar_delta),
            "max_variable_rmse_relative_degrade": float(max_variable_rmse_relative_degrade),
            "max_region_rmse_relative_degrade": float(max_region_rmse_relative_degrade),
            "require_corr_or_sign_nondecline": bool(require_corr_or_sign_nondecline),
        },
    }


def select_eta_from_source_val(
    records: Iterable[Mapping[str, Any]],
    bank: Mapping[str, Any],
    *,
    eta_grid: Sequence[float] = (0.0, 0.025, 0.05, 0.10),
    proposal_clip_scale: float = 1.0,
    residual_clip_scale: float | None = None,
    min_dual_cvar_delta: float = 0.002,
    max_variable_rmse_relative_degrade: float = 0.001,
    max_region_rmse_relative_degrade: float = 0.005,
    require_corr_or_sign_nondecline: bool = True,
) -> dict[str, Any]:
    """Grid-search separate surface/rootzone eta values using source_val only."""
    eta_values = [float(eta) for eta in eta_grid]
    eta_pairs = [(eta_s, eta_r) for eta_s in eta_values for eta_r in eta_values]
    evaluated, records_hash = evaluate_record_stream_for_eta_pairs(
        records,
        bank,
        eta_pairs=eta_pairs,
        proposal_clip_scale=proposal_clip_scale,
        residual_clip_scale=residual_clip_scale,
    )
    passing = []
    for result in evaluated:
        report = _source_gate_report(
            result,
            min_dual_cvar_delta=min_dual_cvar_delta,
            max_variable_rmse_relative_degrade=max_variable_rmse_relative_degrade,
            max_region_rmse_relative_degrade=max_region_rmse_relative_degrade,
            require_corr_or_sign_nondecline=require_corr_or_sign_nondecline,
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
                float(item["summary"]["surface"]["increment_corr_mean"])
                + float(item["summary"]["rootzone"]["increment_corr_mean"]),
                -(float(item["eta_surface"]) + float(item["eta_rootzone"])),
            ),
        )
    else:
        selected = next(
            (
                item
                for item in evaluated
                if float(item["eta_surface"]) == 0.0 and float(item["eta_rootzone"]) == 0.0
            ),
            evaluated[0],
        )
    selection = {
        "schema_version": SIGNED_DA_GAIN_SELECTION_SCHEMA,
        "method_id": SIGNED_DA_GAIN_METHOD_ID,
        "eta_grid": eta_values,
        "eta_pair_grid": [[float(a), float(b)] for a in eta_values for b in eta_values],
        "selected_eta_surface": float(selected["eta_surface"]),
        "selected_eta_rootzone": float(selected["eta_rootzone"]),
        "source_gate_pass": bool(selected.get("source_gate_pass", False)),
        "source_gate_report": selected.get("source_gate_report", {}),
        "selection_source": "source_val_only",
        "target_val_usage": "unused",
        "target_eval_usage": "not_used_for_eta_selection",
        "target_eval_final_policy": "run_once_only_if_source_gate_passes",
        "source_val_records_hash": records_hash,
        "bank_content_hash": bank.get("bank_content_hash", ""),
        "selection_rule": {
            "min_dual_cvar_delta": float(min_dual_cvar_delta),
            "max_variable_rmse_relative_degrade": float(max_variable_rmse_relative_degrade),
            "max_region_rmse_relative_degrade": float(max_region_rmse_relative_degrade),
            "require_corr_or_sign_nondecline": bool(require_corr_or_sign_nondecline),
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
            "source_val_records_hash": selection["source_val_records_hash"],
            "bank_content_hash": selection["bank_content_hash"],
            "selection_rule": selection["selection_rule"],
        }
    )
    return selection


def validate_router_metadata_no_target_selection(metadata: Mapping[str, Any]) -> None:
    if metadata.get("target_eval_usage") not in {
        "not_used_for_eta_selection",
        "final_eval_only_no_selection",
    }:
        raise ValueError("M3_11 router metadata indicates target_eval selection")
    if metadata.get("target_val_usage") not in {"unused", "unused_in_main_protocol", None}:
        raise ValueError("M3_11 router metadata indicates target_val usage")


def validate_source_gate_for_target_eval(selection: Mapping[str, Any]) -> None:
    validate_router_metadata_no_target_selection(selection)
    if selection.get("schema_version") != SIGNED_DA_GAIN_SELECTION_SCHEMA:
        raise ValueError(f"Unsupported M3_11 eta selection schema: {selection.get('schema_version')!r}")
    if selection.get("method_id") != SIGNED_DA_GAIN_METHOD_ID:
        raise ValueError("M3_11 eta selection method_id mismatch")
    if selection.get("selection_source") != "source_val_only":
        raise ValueError("M3_11 target_eval requires source_val-only eta selection")
    eta_positive = (
        float(selection.get("selected_eta_surface", 0.0)) > 0.0
        or float(selection.get("selected_eta_rootzone", 0.0)) > 0.0
    )
    if not eta_positive:
        raise ValueError("M3_11 target_eval refused: no positive eta was selected")
    if not bool(selection.get("source_gate_pass", False)):
        raise ValueError("M3_11 target_eval refused: source gate did not pass")
