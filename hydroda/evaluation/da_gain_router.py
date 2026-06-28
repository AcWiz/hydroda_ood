"""Post-hoc DA gain consistency router for HyperDA-TRUST.

The router is intentionally lightweight: it learns only source-side covariance
statistics and blends a bounded physical DA proposal with an already frozen
predictor output. It never updates neural model parameters.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


VARIABLES = ("surface", "rootzone")
SOURCE_ROLES_FOR_BANK = {"source_fit"}
SOURCE_ROLES_FOR_SELECTION = {"source_val", "source_val_pseudo_query"}
FORBIDDEN_FOR_BANK = {"target_val", "target_eval", "target_query", "target_full_train"}
DA_GAIN_BANK_SCHEMA = "m3_10a_source_da_gain_bank_v1"
DA_GAIN_ROUTER_SCHEMA = "m3_10a_da_gain_router_lite_v1"
DA_GAIN_METHOD_ID = "M3_10a_da_gain_router_lite"
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


def _masked_pair(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    m = np.asarray(mask) > 0.5
    valid = m & np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        empty = np.empty((0,), dtype=np.float64)
        return empty, empty
    return a[valid].reshape(-1), b[valid].reshape(-1)


def _weighted_mean(values: Sequence[float], weights: Sequence[int]) -> float:
    pairs = [
        (float(value), max(0, int(weight)))
        for value, weight in zip(values, weights)
        if math.isfinite(float(value)) and max(0, int(weight)) > 0
    ]
    total = float(sum(weight for _value, weight in pairs))
    if total <= 0.0:
        finite = [float(value) for value in values if math.isfinite(float(value))]
        return float(np.mean(finite)) if finite else 0.0
    return float(sum(value * weight for value, weight in pairs) / total)


def _mean_abs_nonzero(values: list[np.ndarray]) -> float:
    if not values:
        return 0.0
    arr = np.concatenate([v.reshape(-1) for v in values if v.size > 0])
    if arr.size == 0:
        return 0.0
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(np.abs(finite)))


def _estimate_gain(y: np.ndarray, d: np.ndarray, *, ridge_lambda: float) -> GainEstimate:
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    dv = np.asarray(d, dtype=np.float64).reshape(-1)
    valid = np.isfinite(yv) & np.isfinite(dv)
    yv = yv[valid]
    dv = dv[valid]
    n = int(yv.size)
    if n <= 1:
        return GainEstimate(cov=0.0, var=0.0, gain=0.0, n=n)
    yc = yv - float(np.mean(yv))
    dc = dv - float(np.mean(dv))
    cov = float(np.mean(yc * dc))
    var = float(np.mean(dc * dc))
    denom = var + float(ridge_lambda)
    gain = float(cov / denom) if math.isfinite(var) and denom > 0.0 else 0.0
    if not math.isfinite(gain):
        gain = 0.0
    return GainEstimate(cov=cov, var=var, gain=gain, n=n)


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
        if role in FORBIDDEN_FOR_BANK or adaptation_setting in FORBIDDEN_FOR_BANK:
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
    """Return normalized H/V observation-minus-assimilated TB innovations."""
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
        elif isinstance(active, Sequence) and len(active) == 1:
            region = str(active[0])
    if not region:
        raise ValueError("record lacks sample_region_id/target_region_id for DA gain bank grouping")
    return region


def month_from_record(record: Mapping[str, Any]) -> int:
    month = record.get("month")
    if month is None:
        date = str(record.get("query_date") or record.get("date_str") or "")
        if len(date) >= 7:
            month = int(date[5:7])
        else:
            month = 1
    month_int = int(month)
    if month_int < 1 or month_int > 12:
        raise ValueError(f"month must be in 1..12, got {month!r}")
    return month_int


def sample_arrays_for_gain(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Extract the arrays needed for source gain estimation from a sample record."""
    arrays = record.get("arrays")
    if isinstance(arrays, Mapping):
        from hydroda.evaluation.harness import prediction_record_array

        out = {str(key): prediction_record_array(dict(payload)) for key, payload in arrays.items()}
        if "x" in out:
            return out
        raise ValueError("prediction record arrays do not include x; use live dataset records for DA gain bank")
    required = (
        "x",
        "increment_surface",
        "increment_rootzone",
        "metric_mask",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise KeyError(f"sample record missing required arrays for DA gain bank: {missing}")
    return {key: np.asarray(record[key], dtype=np.float32) for key in required}


class DAGainBankAccumulator:
    """Streaming source-side DA gain bank accumulator.

    The bank only needs covariance moments, so full sample tensors should not be
    retained after each update.
    """

    def __init__(
        self,
        *,
        ridge_lambda: float = 1e-3,
        eps: float = 1e-6,
        source_checkpoint: str = "",
        split_manifest: str = "",
        exploratory_after_us_r1_target_eval_seen: bool = True,
    ) -> None:
        self.ridge_lambda = float(ridge_lambda)
        self.eps = float(eps)
        self.source_checkpoint = str(source_checkpoint or "")
        self.split_manifest = str(split_manifest or "")
        self.exploratory_after_us_r1_target_eval_seen = bool(exploratory_after_us_r1_target_eval_seen)
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
        _require_roles([record], SOURCE_ROLES_FOR_BANK, purpose="DA gain bank construction")
        _require_no_target_records([record], purpose="DA gain bank construction")
        self.n_records_seen += 1
        arrays = sample_arrays_for_gain(record)
        mask = np.asarray(arrays["metric_mask"], dtype=np.float32)
        d_h, d_v = da_innovations_from_x(arrays["x"], eps=self.eps)
        inc_s = np.asarray(arrays["increment_surface"], dtype=np.float32)
        inc_r = np.asarray(arrays["increment_rootzone"], dtype=np.float32)
        s_values, h_values = _masked_pair(inc_s, d_h, mask)
        r_values, h_r_values = _masked_pair(inc_r, d_h, mask)
        s_v_values, v_values = _masked_pair(inc_s, d_v, mask)
        r_v_values, v_r_values = _masked_pair(inc_r, d_v, mask)
        s_c, r_c = _masked_pair(inc_s, inc_r, mask)
        if s_values.size == 0 or r_values.size == 0:
            return
        key = (source_region_from_record(record), month_from_record(record))
        bucket = self.grouped.setdefault(key, GainBucket())
        bucket.surface_h.update(s_values, h_values)
        bucket.surface_v.update(s_v_values, v_values)
        bucket.rootzone_h.update(r_values, h_r_values)
        bucket.rootzone_v.update(r_v_values, v_r_values)
        bucket.coupling.update(r_c, s_c)
        self.source_scale["surface"].update(s_values)
        self.source_scale["rootzone"].update(r_values)
        self.source_scale["d_H"].update(h_values)
        self.source_scale["d_V"].update(v_values)
        self.n_records_used += 1

    def finalize(self) -> dict[str, Any]:
        if not self.grouped:
            raise ValueError("No valid source_fit pixels available for DA gain bank")

        entries: dict[str, Any] = {}
        monthly_values: dict[int, list[dict[str, Any]]] = {month: [] for month in range(1, 13)}
        region_values: dict[str, list[dict[str, Any]]] = {}

        for (region, month), bucket in sorted(self.grouped.items()):
            entry = _entry_from_bucket(region, month, bucket, ridge_lambda=self.ridge_lambda)
            entries[f"{region}|{month:02d}"] = entry
            monthly_values[month].append(entry)
            region_values.setdefault(region, []).append(entry)

        def consensus(entry_list: Sequence[Mapping[str, Any]], *, region: str | None, month: int | None) -> dict[str, Any]:
            weights = [int(entry.get("n_pixels", 0)) for entry in entry_list]

            def mean_gain(variable: str, pol: str) -> float:
                values = [
                    float(((entry.get("gains", {}) or {}).get(variable, {}) or {}).get(pol, 0.0))
                    for entry in entry_list
                ]
                return _weighted_mean(values, weights)

            return {
                "source_region": region or "source_consensus",
                "month": int(month) if month is not None else None,
                "n_pixels": int(sum(weights)),
                "gains": {
                    "surface": {"H": mean_gain("surface", "H"), "V": mean_gain("surface", "V")},
                    "rootzone": {"H": mean_gain("rootzone", "H"), "V": mean_gain("rootzone", "V")},
                },
                "C_rz": _weighted_mean([float(entry.get("C_rz", 0.0)) for entry in entry_list], weights),
                "fallback_level": "source_consensus" if region is None and month is None else (
                    "month_consensus" if region is None else "region_consensus"
                ),
            }

        month_consensus = {
            f"{month:02d}": consensus(entries_for_month, region=None, month=month)
            for month, entries_for_month in monthly_values.items()
            if entries_for_month
        }
        region_consensus = {
            region: consensus(entries_for_region, region=region, month=None)
            for region, entries_for_region in sorted(region_values.items())
        }
        global_entry = consensus(list(entries.values()), region=None, month=None)

        source_scale = {
            key: moments.mean_abs()
            for key, moments in self.source_scale.items()
        }
        bank = {
            "schema_version": DA_GAIN_BANK_SCHEMA,
            "method_id": DA_GAIN_METHOD_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_label_usage": "source_fit_labels_only",
            "bank_split_roles": sorted(SOURCE_ROLES_FOR_BANK),
            "target_val_usage": "unused",
            "target_eval_usage": "not_used_for_bank_or_eta_selection",
            "exploratory_after_us_r1_target_eval_seen": bool(self.exploratory_after_us_r1_target_eval_seen),
            "neural_training_epochs": 0,
            "neural_parameter_updates": 0,
            "ridge_lambda": float(self.ridge_lambda),
            "eps": float(self.eps),
            "formula": {
                "d_H": "(TB_H_obs - TB_H_assim) / (obs_err_H + eps)",
                "d_V": "(TB_V_obs - TB_V_assim) / (obs_err_V + eps)",
                "G": "Cov_source(DeltaSM_v, d_p) / (Var_source(d_p) + lambda)",
                "C_rz": "Cov_source(DeltaSM_rootzone, DeltaSM_surface) / (Var_source(DeltaSM_surface) + lambda)",
            },
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
            "region_consensus": region_consensus,
            "global_consensus": global_entry,
            "n_source_records_seen": int(self.n_records_seen),
            "n_source_records_used": int(self.n_records_used),
            "accumulator": "streaming_covariance_moments_v1",
        }
        bank["bank_content_hash"] = _stable_hash(
            {
                "schema_version": bank["schema_version"],
                "ridge_lambda": bank["ridge_lambda"],
                "eps": bank["eps"],
                "entries": bank["entries"],
                "month_consensus": bank["month_consensus"],
                "global_consensus": bank["global_consensus"],
            }
        )
        return bank


def build_source_da_gain_bank_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    ridge_lambda: float = 1e-3,
    eps: float = 1e-6,
    source_checkpoint: str = "",
    split_manifest: str = "",
    exploratory_after_us_r1_target_eval_seen: bool = True,
) -> dict[str, Any]:
    """Build source-side DA gain bank grouped by source region and month."""
    accumulator = DAGainBankAccumulator(
        ridge_lambda=ridge_lambda,
        eps=eps,
        source_checkpoint=source_checkpoint,
        split_manifest=split_manifest,
        exploratory_after_us_r1_target_eval_seen=exploratory_after_us_r1_target_eval_seen,
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
    if bank.get("schema_version") != DA_GAIN_BANK_SCHEMA:
        raise ValueError(f"Unsupported DA gain bank schema: {bank.get('schema_version')!r}")
    if bank.get("source_label_usage") != "source_fit_labels_only":
        raise ValueError("DA gain bank must declare source_fit_labels_only label usage")
    if bank.get("target_eval_usage") != "not_used_for_bank_or_eta_selection":
        raise ValueError("DA gain bank must not use target_eval for bank or eta selection")
    if "entries" not in bank or not isinstance(bank["entries"], Mapping):
        raise ValueError("DA gain bank missing entries")


def select_gain_entry(bank: Mapping[str, Any], *, region: str, month: int) -> tuple[Mapping[str, Any], str]:
    validate_gain_bank_metadata(bank)
    key = f"{region}|{int(month):02d}"
    entries = bank.get("entries", {})
    if key in entries:
        return entries[key], "exact_region_month"
    month_key = f"{int(month):02d}"
    month_consensus = bank.get("month_consensus", {})
    if month_key in month_consensus:
        return month_consensus[month_key], "source_month_consensus"
    region_consensus = bank.get("region_consensus", {})
    if region in region_consensus:
        return region_consensus[region], "source_region_consensus"
    return bank.get("global_consensus", {}), "source_global_consensus"


def physical_proposal_from_sample(
    sample: Mapping[str, Any],
    bank: Mapping[str, Any],
    *,
    proposal_clip_scale: float = 1.0,
    eps: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build bounded source-gain physical increment proposal for one sample."""
    x = _as_float_array(sample["x"], name="sample.x")
    month = int(sample.get("month", 1))
    region = str(sample.get("sample_region_id") or sample.get("target_region_id") or "")
    entry, fallback_level = select_gain_entry(bank, region=region, month=month)
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
    raw_rootzone = (raw_rootzone_direct + float(entry.get("C_rz", 0.0)) * raw_surface).astype(np.float32)
    source_scale = bank.get("source_scale", {})
    clip_s = max(float(source_scale.get("surface", 0.0)), 0.0) * float(proposal_clip_scale)
    clip_r = max(float(source_scale.get("rootzone", 0.0)), 0.0) * float(proposal_clip_scale)
    if clip_s > 0.0:
        surface = np.clip(raw_surface, -clip_s, clip_s).astype(np.float32)
    else:
        surface = raw_surface.astype(np.float32)
    if clip_r > 0.0:
        rootzone = np.clip(raw_rootzone, -clip_r, clip_r).astype(np.float32)
    else:
        rootzone = raw_rootzone.astype(np.float32)
    summary = {
        "fallback_level": fallback_level,
        "bank_region": entry.get("source_region", ""),
        "bank_month": entry.get("month"),
        "n_pixels": int(entry.get("n_pixels", 0) or 0),
        "clip_surface": float(clip_s),
        "clip_rootzone": float(clip_r),
        "gains": entry.get("gains", {}),
        "C_rz": float(entry.get("C_rz", 0.0) or 0.0),
    }
    return {"surface": surface, "rootzone": rootzone}, summary


def blend_prediction_with_da_gain(
    sample: Mapping[str, Any],
    base_pred: Mapping[str, Any],
    bank: Mapping[str, Any],
    *,
    eta_surface: float = 0.0,
    eta_rootzone: float | None = None,
    proposal_clip_scale: float = 1.0,
) -> dict[str, Any]:
    """Return base prediction blended with DA-gain physical proposals."""
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
            "da_gain_eta_surface": eta_s,
            "da_gain_eta_rootzone": eta_r,
            "da_gain_fallback_level": "eta_zero_no_proposal",
        }
    proposal, summary = physical_proposal_from_sample(
        sample,
        bank,
        proposal_clip_scale=proposal_clip_scale,
    )
    base_s = np.asarray(base_pred["pred_increment_surface"], dtype=np.float32)
    base_r = np.asarray(base_pred["pred_increment_rootzone"], dtype=np.float32)
    final_s = ((1.0 - eta_s) * base_s + eta_s * proposal["surface"]).astype(np.float32)
    final_r = ((1.0 - eta_r) * base_r + eta_r * proposal["rootzone"]).astype(np.float32)
    forecast_s = np.asarray(sample["forecast_surface"], dtype=np.float32)
    forecast_r = np.asarray(sample["forecast_rootzone"], dtype=np.float32)
    return {
        "pred_increment_surface": final_s,
        "pred_increment_rootzone": final_r,
        "pred_analysis_surface": (forecast_s + final_s).astype(np.float32),
        "pred_analysis_rootzone": (forecast_r + final_r).astype(np.float32),
        "da_gain_eta_surface": eta_s,
        "da_gain_eta_rootzone": eta_r,
        "da_gain_fallback_level": summary["fallback_level"],
        "da_gain_summary": summary,
    }


class DAGainRouterPredictor:
    """Predictor wrapper that applies post-hoc bounded DA-gain blending."""

    def __init__(
        self,
        base_predictor: Any,
        bank: Mapping[str, Any],
        *,
        eta_surface: float = 0.0,
        eta_rootzone: float | None = None,
        proposal_clip_scale: float = 1.0,
        method_name: str = DA_GAIN_METHOD_ID,
    ) -> None:
        validate_gain_bank_metadata(bank)
        self.base_predictor = base_predictor
        self.bank = dict(bank)
        self.eta_surface = float(eta_surface)
        self.eta_rootzone = self.eta_surface if eta_rootzone is None else float(eta_rootzone)
        self.proposal_clip_scale = float(proposal_clip_scale)
        self.method_name = method_name
        self.metadata = {
            "schema_version": DA_GAIN_ROUTER_SCHEMA,
            "method_id": DA_GAIN_METHOD_ID,
            "base_method": getattr(base_predictor, "method_name", "unknown"),
            "eta_surface": self.eta_surface,
            "eta_rootzone": self.eta_rootzone,
            "proposal_clip_scale": self.proposal_clip_scale,
            "bank_content_hash": self.bank.get("bank_content_hash", ""),
            "neural_training_epochs": 0,
            "neural_parameter_updates": 0,
            "target_eval_usage": "final_eval_only_no_selection",
            "exploratory_after_us_r1_target_eval_seen": bool(
                self.bank.get("exploratory_after_us_r1_target_eval_seen", True)
            ),
        }

    def predict(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        base_pred = self.base_predictor.predict(sample)
        return blend_prediction_with_da_gain(
            sample,
            base_pred,
            self.bank,
            eta_surface=self.eta_surface,
            eta_rootzone=self.eta_rootzone,
            proposal_clip_scale=self.proposal_clip_scale,
        )


def build_source_records_from_predictor(
    *,
    dataset: Any,
    predictor: Any | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Materialize source records with x, labels, masks, and optional predictions."""
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
            "latitude_weight": np.asarray(sample["latitude_weight"], dtype=np.float32),
        }
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


def evaluate_records_for_eta(
    records: Sequence[Mapping[str, Any]],
    bank: Mapping[str, Any],
    *,
    eta: float,
    proposal_clip_scale: float = 1.0,
) -> dict[str, Any]:
    """Evaluate one eta on materialized source records."""
    _require_roles(records, SOURCE_ROLES_FOR_SELECTION, purpose="DA gain eta selection")
    from hydroda.metrics.skill import increment_corr, increment_rmse, sign_accuracy_deadzone, weighted_analysis_skill_components

    by_region: dict[str, dict[str, dict[str, list[float]]]] = {}
    base_by_region: dict[str, dict[str, dict[str, list[float]]]] = {}
    for record in records:
        sample = dict(record)
        base_pred = {
            "pred_increment_surface": np.asarray(record["pred_increment_surface"], dtype=np.float32),
            "pred_increment_rootzone": np.asarray(record["pred_increment_rootzone"], dtype=np.float32),
            "pred_analysis_surface": np.asarray(record["forecast_surface"], dtype=np.float32)
            + np.asarray(record["pred_increment_surface"], dtype=np.float32),
            "pred_analysis_rootzone": np.asarray(record["forecast_rootzone"], dtype=np.float32)
            + np.asarray(record["pred_increment_rootzone"], dtype=np.float32),
        }
        routed = blend_prediction_with_da_gain(
            sample,
            base_pred,
            bank,
            eta_surface=float(eta),
            eta_rootzone=float(eta),
            proposal_clip_scale=proposal_clip_scale,
        )
        region = source_region_from_record(record)
        by_region.setdefault(region, {var: {"mse": [], "fcst_mse": [], "rmse": [], "corr": [], "sign": []} for var in VARIABLES})
        base_by_region.setdefault(region, {var: {"mse": [], "fcst_mse": [], "rmse": [], "corr": [], "sign": []} for var in VARIABLES})
        mask = np.asarray(record["metric_mask"], dtype=np.float32)
        latw = np.asarray(record["latitude_weight"], dtype=np.float32)
        for variable in VARIABLES:
            forecast = np.asarray(record[f"forecast_{variable}"], dtype=np.float32)
            true_inc = np.asarray(record[f"increment_{variable}"], dtype=np.float32)
            true_analysis = np.asarray(record[f"analysis_{variable}"], dtype=np.float32)
            base_inc = np.asarray(base_pred[f"pred_increment_{variable}"], dtype=np.float32)
            routed_inc = np.asarray(routed[f"pred_increment_{variable}"], dtype=np.float32)
            base_analysis = forecast + base_inc
            routed_analysis = forecast + routed_inc
            for target, pred_inc, pred_analysis in (
                (base_by_region[region][variable], base_inc, base_analysis),
                (by_region[region][variable], routed_inc, routed_analysis),
            ):
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

    def summarize(block: Mapping[str, Mapping[str, Mapping[str, list[float]]]]) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
        variable_summary: dict[str, Any] = {}
        region_skills: dict[str, dict[str, float]] = {}
        for region, region_block in sorted(block.items()):
            region_skills[region] = {}
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
                region_skills[region][f"{variable}_skill"] = skill
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
                "increment_rmse_latw_mean": float(np.mean(rmse_all)) if rmse_all else float("nan"),
                "increment_corr_latw_mean": float(np.mean(corr_all)) if corr_all else float("nan"),
                "sign_accuracy_deadzone_mean": float(np.mean(sign_all)) if sign_all else float("nan"),
            }
        return variable_summary, region_skills

    base_summary, base_region_skills = summarize(base_by_region)
    routed_summary, routed_region_skills = summarize(by_region)
    base_cvar = dual_variable_cvar_score_from_region_skills(base_region_skills)
    routed_cvar = dual_variable_cvar_score_from_region_skills(routed_region_skills)
    deltas = {}
    for variable in VARIABLES:
        deltas[variable] = {
            "analysis_rmse_latw_relative": _relative_delta(
                routed_summary[variable]["analysis_rmse_latw"],
                base_summary[variable]["analysis_rmse_latw"],
            ),
            "increment_corr_latw_delta": _safe_delta(
                routed_summary[variable]["increment_corr_latw_mean"],
                base_summary[variable]["increment_corr_latw_mean"],
            ),
            "sign_accuracy_deadzone_delta": _safe_delta(
                routed_summary[variable]["sign_accuracy_deadzone_mean"],
                base_summary[variable]["sign_accuracy_deadzone_mean"],
            ),
        }
    return {
        "eta": float(eta),
        "base_summary": base_summary,
        "summary": routed_summary,
        "base_region_skills": base_region_skills,
        "region_skills": routed_region_skills,
        "base_dual_variable_cvar": base_cvar,
        "dual_variable_cvar": routed_cvar,
        "dual_variable_cvar_delta": _safe_delta(
            routed_cvar["dual_variable_cvar_safe_score"],
            base_cvar["dual_variable_cvar_safe_score"],
        ),
        "deltas": deltas,
    }


def _safe_delta(value: float, baseline: float) -> float:
    if not math.isfinite(float(value)) or not math.isfinite(float(baseline)):
        return float("nan")
    return float(value) - float(baseline)


def _relative_delta(value: float, baseline: float) -> float:
    if not math.isfinite(float(value)) or not math.isfinite(float(baseline)) or float(baseline) == 0.0:
        return float("nan")
    return float(value / baseline - 1.0)


def select_eta_from_source_val(
    records: Sequence[Mapping[str, Any]],
    bank: Mapping[str, Any],
    *,
    eta_grid: Sequence[float] = (0.0, 0.025, 0.05, 0.10),
    proposal_clip_scale: float = 1.0,
    min_dual_cvar_delta: float = -0.005,
    max_rmse_relative_degrade: float = 0.002,
    require_corr_or_sign_gain: bool = True,
) -> dict[str, Any]:
    """Grid-search eta using source_val only."""
    _require_roles(records, SOURCE_ROLES_FOR_SELECTION, purpose="DA gain eta selection")
    evaluated = [
        evaluate_records_for_eta(records, bank, eta=float(eta), proposal_clip_scale=proposal_clip_scale)
        for eta in eta_grid
    ]
    passing = []
    for result in evaluated:
        deltas = result["deltas"]
        rmse_ok = all(
            (
                math.isfinite(float(deltas[variable]["analysis_rmse_latw_relative"]))
                and float(deltas[variable]["analysis_rmse_latw_relative"]) <= float(max_rmse_relative_degrade)
            )
            for variable in VARIABLES
        )
        cvar_ok = (
            math.isfinite(float(result["dual_variable_cvar_delta"]))
            and float(result["dual_variable_cvar_delta"]) >= float(min_dual_cvar_delta)
        )
        gain_ok = any(
            (
                float(deltas[variable]["increment_corr_latw_delta"]) > 0.0
                or float(deltas[variable]["sign_accuracy_deadzone_delta"]) > 0.0
            )
            for variable in VARIABLES
            if math.isfinite(float(deltas[variable]["increment_corr_latw_delta"]))
            or math.isfinite(float(deltas[variable]["sign_accuracy_deadzone_delta"]))
        )
        if cvar_ok and rmse_ok and (gain_ok or not require_corr_or_sign_gain):
            result["source_gate_pass"] = True
            passing.append(result)
        else:
            result["source_gate_pass"] = False
    if passing:
        selected = max(
            passing,
            key=lambda item: (
                float(item["dual_variable_cvar"]["dual_variable_cvar_safe_score"]),
                float(item["summary"]["surface"]["increment_corr_latw_mean"])
                + float(item["summary"]["rootzone"]["increment_corr_latw_mean"]),
                -float(item["eta"]),
            ),
        )
    else:
        selected = next((item for item in evaluated if float(item["eta"]) == 0.0), evaluated[0])
    return {
        "schema_version": DA_GAIN_ROUTER_SCHEMA,
        "method_id": DA_GAIN_METHOD_ID,
        "eta_grid": [float(eta) for eta in eta_grid],
        "selected_eta_surface": float(selected["eta"]),
        "selected_eta_rootzone": float(selected["eta"]),
        "source_gate_pass": bool(selected.get("source_gate_pass", False)),
        "selection_source": "source_val_only",
        "target_val_usage": "unused",
        "target_eval_usage": "not_used_for_eta_selection",
        "exploratory_after_us_r1_target_eval_seen": bool(
            bank.get("exploratory_after_us_r1_target_eval_seen", True)
        ),
        "source_val_records_hash": _arrayless_record_hash(records),
        "bank_content_hash": bank.get("bank_content_hash", ""),
        "selection_rule": {
            "min_dual_cvar_delta": float(min_dual_cvar_delta),
            "max_rmse_relative_degrade": float(max_rmse_relative_degrade),
            "require_corr_or_sign_gain": bool(require_corr_or_sign_gain),
        },
        "selected": selected,
        "grid": evaluated,
    }


def validate_router_metadata_no_target_selection(metadata: Mapping[str, Any]) -> None:
    if metadata.get("target_eval_usage") not in {
        "not_used_for_eta_selection",
        "final_eval_only_no_selection",
    }:
        raise ValueError("DA gain router metadata indicates target_eval selection")
    if metadata.get("target_val_usage") not in {"unused", None}:
        raise ValueError("DA gain router metadata indicates target_val usage")
