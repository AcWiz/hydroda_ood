#!/usr/bin/env python3
"""Source-safe conservative guard calibration for HyperDA P2.8.

The calibration contract is deliberately narrower than the target-eval
diagnostic wrappers:

* candidate selection reads source-side pseudo-query rows only;
* pseudo-query role must be source_val/source_val_pseudo_query;
* target_eval rows are rejected even when marked diagnostic_only;
* final target_eval is never read by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "p2_8_source_safe_guard_config_v1"
SUMMARY_SCHEMA_VERSION = "p2_8_source_safe_guard_calibration_v1"
SOURCE_SAFE_QUERY_ROLES = {"source_val", "source_val_pseudo_query"}
FORBIDDEN_TARGET_EVAL_ROLES = {"target_eval", "target_query", "target_val", "target_full_train"}

SCHEDULES = (
    {
        "schedule_label": "original_K12",
        "lr": 3e-4,
        "adaptation_steps": 80,
        "anchor_alpha": 0.25,
    },
    {
        "schedule_label": "K4_schedule_on_K12",
        "lr": 1e-3,
        "adaptation_steps": 100,
        "anchor_alpha": 0.75,
    },
)
SUPPORT_LOSS_REDUCTIONS = ("global_pixel", "cycle_balanced")
RHO_POLICIES = ("fixed_1.0", "fixed_0.75", "fixed_0.5", "fixed_0.25", "rule_a", "rule_b", "rule_c")
TRUST_POLICIES = ("none", "mild_groupwise", "strong_groupwise")
COMPACT_BASE_CONFIGS = (
    {
        "base_config_id": "C0",
        "schedule_label": "original_K12",
        "lr": 3e-4,
        "adaptation_steps": 80,
        "anchor_alpha": 0.25,
        "support_loss_reduction": "global_pixel",
        "trust_policy": "none",
    },
    {
        "base_config_id": "C1",
        "schedule_label": "original_K12",
        "lr": 3e-4,
        "adaptation_steps": 80,
        "anchor_alpha": 0.25,
        "support_loss_reduction": "cycle_balanced",
        "trust_policy": "none",
    },
    {
        "base_config_id": "C2",
        "schedule_label": "K4_schedule_on_K12",
        "lr": 1e-3,
        "adaptation_steps": 100,
        "anchor_alpha": 0.75,
        "support_loss_reduction": "global_pixel",
        "trust_policy": "none",
    },
    {
        "base_config_id": "C3",
        "schedule_label": "K4_schedule_on_K12",
        "lr": 1e-3,
        "adaptation_steps": 100,
        "anchor_alpha": 0.75,
        "support_loss_reduction": "cycle_balanced",
        "trust_policy": "none",
    },
    {
        "base_config_id": "C4",
        "schedule_label": "K4_schedule_on_K12",
        "lr": 1e-3,
        "adaptation_steps": 100,
        "anchor_alpha": 0.75,
        "support_loss_reduction": "cycle_balanced",
        "trust_policy": "mild_groupwise",
    },
    {
        "base_config_id": "C5",
        "schedule_label": "K4_schedule_on_K12",
        "lr": 1e-3,
        "adaptation_steps": 100,
        "anchor_alpha": 0.75,
        "support_loss_reduction": "cycle_balanced",
        "trust_policy": "strong_groupwise",
    },
)
_CANDIDATE_HASH_FIELDS = (
    "K",
    "adapt_scope",
    "adapt_solver",
    "schedule_label",
    "support_loss_reduction",
    "rho_policy",
    "trust_policy",
    "lr",
    "adaptation_steps",
    "anchor_alpha",
)
_STRICT_EXISTING_ROW_REQUIRED_FIELDS = (
    "episode_id",
    "pseudo_target_region",
    "candidate_id",
    "K",
    "seed",
    "source_checkpoint_sha256",
    "split_manifest_sha256",
)
_SAMPLE_BUDGET_FIELDS = (
    "calib_max_query_samples",
    "source_query_max_samples",
    "max_samples",
    "eval_max_samples",
)

_FLOAT_FIELDS = {
    "score",
    "mean_delta_vs_K0",
    "mean_regret_vs_K0",
    "negative_transfer_rate_vs_K0",
    "surface_skill_primary",
    "rootzone_skill_primary",
    "overall_skill",
    "support_gradient_negative_fraction",
    "support_gradient_cosine_min",
    "support_gradient_cosine_mean",
    "target_parameter_l2_drift_post_anchor_total",
    "target_parameter_l2_drift_post_anchor_target_prompt",
    "target_parameter_l2_drift_post_anchor_monthly_gain",
    "target_parameter_l2_drift_post_anchor_adapter_coeff_bottleneck",
    "target_parameter_l2_drift_post_anchor_adapter_coeff_dec2",
    "target_parameter_l2_drift_post_anchor_adapter_coeff_dec1",
    "target_parameter_l2_drift_post_anchor_spatial_refine",
}


def _clean_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _clean_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _json_safe(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return {str(key): _json_safe(value) for key, value in payload.items() if key is not None}
    if isinstance(payload, (list, tuple)):
        return [_json_safe(value) for value in payload]
    return payload


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        if isinstance(payload.get("rows"), list):
            return [dict(row) for row in payload["rows"] if isinstance(row, Mapping)]
        if isinstance(payload.get("episodes"), list):
            return [dict(row) for row in payload["episodes"] if isinstance(row, Mapping)]
    raise ValueError(f"Cannot load calibration rows from JSON shape: {path}")


def _canonicalize_row(row: Mapping[str, Any], source_path: Path) -> dict[str, Any]:
    out = {str(key): value for key, value in row.items() if key is not None}
    out["_source_path"] = str(source_path)
    for key in list(out):
        if key in _FLOAT_FIELDS:
            value = _clean_float(out.get(key))
            if value is not None:
                out[key] = value
    for key in ("K", "seed"):
        value = _clean_int(out.get(key))
        if value is not None:
            out[key] = value
    if not out.get("query_role"):
        split_type = str(out.get("split_type", "")).strip()
        out["query_role"] = "source_val_pseudo_query" if split_type == "source_val" else split_type
    if not out.get("split_type") and str(out.get("query_role")) in SOURCE_SAFE_QUERY_ROLES:
        out["split_type"] = "source_val"
    if not out.get("episode_id"):
        region = out.get("pseudo_target_region") or out.get("target_region") or out.get("target_region_id")
        seed = out.get("seed", "")
        out["episode_id"] = f"{region}_S{seed}" if region else f"episode_{_json_hash(out)[:12]}"
    if not out.get("pseudo_target_region"):
        out["pseudo_target_region"] = out.get("target_region") or out.get("target_region_id") or ""
    return out


def _validate_source_safe_row(row: Mapping[str, Any]) -> None:
    split_type = str(row.get("split_type", "")).strip()
    query_role = str(row.get("query_role", "")).strip()
    adaptation_setting = str(row.get("adaptation_setting", "")).strip()
    if (
        split_type in FORBIDDEN_TARGET_EVAL_ROLES
        or query_role in FORBIDDEN_TARGET_EVAL_ROLES
        or adaptation_setting in FORBIDDEN_TARGET_EVAL_ROLES
    ):
        raise ValueError(
            "P2.8 calibration refuses target_eval/target_query/target_val/target_full_train rows. "
            f"row={row.get('episode_id')} split_type={split_type!r} query_role={query_role!r} "
            f"adaptation_setting={adaptation_setting!r}"
        )
    if _as_bool(row.get("diagnostic_only")) and (
        "target_eval" in split_type.lower() or "target_eval" in query_role.lower()
    ):
        raise ValueError("diagnostic_only target_eval rows cannot be read by source-safe calibration")
    if query_role not in SOURCE_SAFE_QUERY_ROLES:
        raise ValueError(
            "P2.8 calibration rows must be source_val/source_val_pseudo_query only; "
            f"got query_role={query_role!r} for row={row.get('episode_id')}"
        )


def load_calibration_rows(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Load and validate source-safe calibration rows from CSV or JSON files."""
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        loaded = _json_rows(path) if path.suffix.lower() == ".json" else _csv_rows(path)
        for row in loaded:
            clean = _canonicalize_row(row, path)
            _validate_source_safe_row(clean)
            rows.append(clean)
    if not rows:
        raise ValueError("No source-safe calibration rows loaded")
    return rows


def discover_calibration_rows(roots: Sequence[str | Path]) -> list[Path]:
    """Find likely source-safe row files under roots."""
    names = {
        "source_safe_candidate_rows.csv",
        "source_safe_candidate_rows.json",
        "candidate_rows.csv",
        "candidate_rows.json",
        "overview.csv",
        "overview.json",
    }
    out: list[Path] = []
    for root in roots:
        path = Path(root)
        if path.is_file():
            out.append(path)
            continue
        if not path.exists():
            continue
        for candidate in sorted(path.rglob("*")):
            if candidate.is_file() and candidate.name in names:
                out.append(candidate)
    return out


def candidate_id_from_config(config: Mapping[str, Any]) -> str:
    schedule = str(config.get("schedule_label", ""))
    loss = str(config.get("support_loss_reduction", "global_pixel"))
    rho = str(config.get("rho_policy") or rho_policy_from_row(config))
    trust = str(config.get("trust_policy") or trust_policy_from_row(config))
    return f"schedule_{schedule}__loss_{loss}__rho_{rho}__trust_{trust}"


def candidate_config_hash(config: Mapping[str, Any]) -> str:
    """Hash the fields that define one candidate row's runnable semantics."""
    payload = {key: config.get(key) for key in _CANDIDATE_HASH_FIELDS if key in config}
    if "base_config_id" in config:
        payload["base_config_id"] = config.get("base_config_id")
    return _json_hash(payload)


def enumerate_guard_base_configs(candidate_set: str = "compact_v1") -> list[dict[str, Any]]:
    """Enumerate GPU-heavy base adaptations before offline rho expansion."""
    if candidate_set == "compact_v1":
        base_configs = [dict(config) for config in COMPACT_BASE_CONFIGS]
    elif candidate_set == "full_v1":
        base_configs = []
        idx = 0
        for schedule in SCHEDULES:
            for loss_reduction in SUPPORT_LOSS_REDUCTIONS:
                for trust_policy in TRUST_POLICIES:
                    base_configs.append(
                        {
                            "base_config_id": f"F{idx}",
                            "support_loss_reduction": loss_reduction,
                            "trust_policy": trust_policy,
                            **schedule,
                        }
                    )
                    idx += 1
    else:
        raise ValueError(f"Unsupported candidate_set={candidate_set!r}")

    out = []
    for base in base_configs:
        candidate = {
            "K": 12,
            "adapt_scope": "all",
            "adapt_solver": "adamw",
            "rho_policy": "fixed_1.0",
            **base,
        }
        candidate["candidate_id"] = candidate_id_from_config(candidate)
        candidate["candidate_config_hash"] = candidate_config_hash(candidate)
        out.append(candidate)
    return out


def enumerate_guard_candidates(candidate_set: str = "compact_v1") -> list[dict[str, Any]]:
    """Enumerate logical P2.8 K12 guard candidates for the requested set."""
    candidates: list[dict[str, Any]] = []
    for base in enumerate_guard_base_configs(candidate_set):
        for rho_policy in RHO_POLICIES:
            candidate = dict(base)
            candidate["rho_policy"] = rho_policy
            candidate["candidate_id"] = candidate_id_from_config(candidate)
            candidate["candidate_config_hash"] = candidate_config_hash(candidate)
            candidates.append(candidate)
    return candidates


def base_configs_for_logical_candidate_ids(
    *,
    candidate_set: str = "compact_v1",
    candidate_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Map logical top-candidate IDs to the unique base adaptations they need."""
    wanted = {str(candidate_id).strip() for candidate_id in candidate_ids if str(candidate_id).strip()}
    if not wanted:
        return enumerate_guard_base_configs(candidate_set)
    logical = enumerate_guard_candidates(candidate_set)
    base_by_id = {str(base["base_config_id"]): dict(base) for base in enumerate_guard_base_configs(candidate_set)}
    selected_base_ids: list[str] = []
    seen: set[str] = set()
    for candidate in logical:
        if str(candidate.get("candidate_id")) not in wanted:
            continue
        base_id = str(candidate.get("base_config_id", ""))
        if base_id and base_id not in seen:
            seen.add(base_id)
            selected_base_ids.append(base_id)
    missing = sorted(wanted - {str(candidate.get("candidate_id")) for candidate in logical})
    if missing:
        raise ValueError(f"Unknown TOP_CANDIDATE_IDS for {candidate_set}: {missing}")
    return [base_by_id[base_id] for base_id in selected_base_ids]


def baseline_gpu_row_configs() -> list[dict[str, Any]]:
    """Required baseline source rows for each pseudo-target region."""
    return [
        {
            "candidate_id": "K0_identity",
            "base_config_id": "K0",
            "candidate_config_hash": "K0_identity_static",
            "K": 0,
            "adapt_scope": "all",
            "adapt_solver": "adamw",
            "schedule_label": "identity_base",
            "support_loss_reduction": "global_pixel",
            "rho_policy": "fixed_1.0",
            "trust_policy": "none",
        },
        {
            "candidate_id": "K4_original",
            "base_config_id": "K4",
            "candidate_config_hash": "K4_original_static",
            "K": 4,
            "adapt_scope": "all",
            "adapt_solver": "adamw",
            "schedule_label": "original_K4",
            "support_loss_reduction": "global_pixel",
            "rho_policy": "fixed_1.0",
            "trust_policy": "none",
        },
    ]


def required_gpu_row_configs(
    *,
    candidate_set: str = "compact_v1",
    top_candidate_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Return baseline rows plus K12 base adaptations needed for this stage."""
    return [
        *baseline_gpu_row_configs(),
        *base_configs_for_logical_candidate_ids(
            candidate_set=candidate_set,
            candidate_ids=top_candidate_ids,
        ),
    ]


def rho_policy_from_row(row: Mapping[str, Any]) -> str:
    policy = row.get("rho_policy")
    if policy:
        return str(policy)
    rho = _clean_float(row.get("adapt_mix_rho"))
    if rho is None:
        return "fixed_1.0"
    return f"fixed_{rho:g}"


def trust_policy_from_row(row: Mapping[str, Any]) -> str:
    policy = row.get("trust_policy")
    if policy:
        return str(policy)
    mode = str(row.get("trust_region_mode", "none") or "none")
    if mode == "none":
        return "none"
    label = str(row.get("trust_policy_label", "") or "")
    if label in TRUST_POLICIES:
        return label
    return "mild_groupwise" if mode == "groupwise" else mode


def compute_rho_for_policy(policy: str, diagnostics: Mapping[str, Any]) -> float:
    """Compute fixed or conflict-aware output mixture rho from support diagnostics."""
    policy = str(policy)
    if policy.startswith("fixed_"):
        return float(policy.split("_", 1)[1])
    neg_frac = _clean_float(diagnostics.get("support_gradient_negative_fraction"))
    min_cos = _clean_float(diagnostics.get("support_gradient_cosine_min"))
    neg_frac = 0.0 if neg_frac is None else neg_frac
    min_cos = 1.0 if min_cos is None else min_cos
    if policy == "rule_a":
        return 0.5 if neg_frac > 0.45 else 1.0
    if policy == "rule_b":
        return 0.5 if neg_frac > 0.45 or min_cos < -0.2 else 1.0
    if policy == "rule_c":
        return 0.25 if neg_frac > 0.50 or min_cos < -0.5 else 1.0
    raise ValueError(f"unsupported rho_policy={policy!r}")


def _overall_skill(row: Mapping[str, Any]) -> float | None:
    overall = _clean_float(row.get("overall_skill"))
    if overall is not None:
        return overall
    surface = _clean_float(row.get("surface_skill_primary"))
    rootzone = _clean_float(row.get("rootzone_skill_primary"))
    if surface is None or rootzone is None:
        return None
    return (surface + rootzone) / 2.0


def _candidate_id_for_row(row: Mapping[str, Any]) -> str:
    if row.get("candidate_id"):
        return str(row["candidate_id"])
    if _clean_int(row.get("K")) == 0:
        return "K0_identity"
    return candidate_id_from_config(row)


def _is_k0_baseline(row: Mapping[str, Any]) -> bool:
    candidate_id = _candidate_id_for_row(row).lower()
    return _clean_int(row.get("K")) == 0 or "k0" in candidate_id or "identity" in candidate_id


def score_candidates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Score K12 guard candidates against per-episode K0 source-safe baselines."""
    k0_by_episode: dict[str, float] = {}
    for row in rows:
        if not _is_k0_baseline(row):
            continue
        skill = _overall_skill(row)
        if skill is not None:
            k0_by_episode[str(row.get("episode_id"))] = skill
    if not k0_by_episode:
        raise ValueError("Cannot score candidates without K0 source-safe baseline rows")

    grouped: dict[str, list[dict[str, Any]]] = {}
    exemplars: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if _is_k0_baseline(row):
            continue
        if _clean_int(row.get("K")) != 12:
            continue
        episode_id = str(row.get("episode_id"))
        if episode_id not in k0_by_episode:
            continue
        skill = _overall_skill(row)
        if skill is None:
            continue
        candidate_id = _candidate_id_for_row(row)
        delta = skill - k0_by_episode[episode_id]
        grouped.setdefault(candidate_id, []).append(
            {
                "episode_id": episode_id,
                "pseudo_target_region": row.get("pseudo_target_region", ""),
                "overall_skill": skill,
                "k0_overall_skill": k0_by_episode[episode_id],
                "delta_vs_K0": delta,
                "regret_vs_K0": max(0.0, -delta),
                "negative_transfer_vs_K0": delta < 0.0,
            }
        )
        exemplars.setdefault(candidate_id, row)

    summaries: list[dict[str, Any]] = []
    for candidate_id, episode_rows in grouped.items():
        deltas = [float(row["delta_vs_K0"]) for row in episode_rows]
        regrets = [float(row["regret_vs_K0"]) for row in episode_rows]
        neg_rate = sum(1 for row in episode_rows if row["negative_transfer_vs_K0"]) / len(episode_rows)
        mean_delta = statistics.fmean(deltas)
        mean_regret = statistics.fmean(regrets)
        score = mean_delta - 0.5 * mean_regret - 0.05 * neg_rate
        exemplar = exemplars[candidate_id]
        summary = {
            "candidate_id": candidate_id,
            "score": score,
            "mean_delta_vs_K0": mean_delta,
            "mean_regret_vs_K0": mean_regret,
            "negative_transfer_rate_vs_K0": neg_rate,
            "episode_count": len(episode_rows),
            "episode_results": episode_rows,
            "schedule_label": exemplar.get("schedule_label", ""),
            "support_loss_reduction": exemplar.get("support_loss_reduction", "global_pixel"),
            "rho_policy": rho_policy_from_row(exemplar),
            "trust_policy": trust_policy_from_row(exemplar),
            "adapt_scope": exemplar.get("adapt_scope", "all"),
            "adapt_solver": exemplar.get("adapt_solver", "adamw"),
            "lr": _clean_float(exemplar.get("lr") or exemplar.get("requested_lr")),
            "adaptation_steps": _clean_int(exemplar.get("adaptation_steps") or exemplar.get("requested_max_steps")),
            "anchor_alpha": _clean_float(exemplar.get("anchor_alpha") or exemplar.get("requested_anchor_alpha")),
        }
        summaries.append(summary)
    if not summaries:
        raise ValueError("No K12 guard candidate rows could be scored")
    return summaries


def expand_logical_candidate_rows_from_prediction_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_set: str = "compact_v1",
) -> list[dict[str, Any]]:
    """Expand base K12 rows into fixed/rule rho logical rows using saved records."""
    if candidate_set != "compact_v1":
        return [dict(row) for row in rows]
    k0_records_by_episode = {
        str(row.get("episode_id")): Path(str(row.get("prediction_record_path")))
        for row in rows
        if _is_k0_baseline(row) and row.get("prediction_record_path")
    }
    out: list[dict[str, Any]] = []
    seen_k12: set[tuple[str, str]] = set()
    for row in rows:
        row_dict = dict(row)
        if _is_k0_baseline(row) or _clean_int(row.get("K")) != 12:
            out.append(row_dict)
            continue
        episode_id = str(row.get("episode_id"))
        adapted_record_path = Path(str(row.get("prediction_record_path", "")))
        k0_record_path = k0_records_by_episode.get(episode_id)
        if not k0_record_path or not k0_record_path.exists() or not adapted_record_path.exists():
            out.append(row_dict)
            continue
        for rho_policy in RHO_POLICIES:
            candidate = dict(row_dict)
            candidate["rho_policy"] = rho_policy
            candidate["candidate_id"] = candidate_id_from_config(candidate)
            candidate["candidate_config_hash"] = candidate_config_hash(candidate)
            key = (episode_id, str(candidate["candidate_id"]))
            if key in seen_k12:
                continue
            seen_k12.add(key)
            rho = compute_rho_for_policy(rho_policy, row_dict)
            if rho_policy == rho_policy_from_row(row_dict) and float(rho) == _clean_float(row_dict.get("adapt_mix_rho") or 1.0):
                candidate["logical_offline_mix"] = False
                out.append(candidate)
                continue
            try:
                from scripts.eval import mix_prediction_records

                mixed = mix_prediction_records.mix_prediction_record_files(
                    k0_record_path,
                    adapted_record_path,
                    rho=rho,
                    candidate_id=str(candidate["candidate_id"]),
                )
            except Exception as exc:
                fallback = dict(candidate)
                fallback["logical_offline_mix"] = "failed"
                fallback["offline_mix_error"] = str(exc)
                out.append(fallback)
                continue
            surface = mixed["summary"].get("surface", {})
            rootzone = mixed["summary"].get("rootzone", {})
            candidate.update(
                {
                    "adapt_mix_rho": rho,
                    "logical_offline_mix": True,
                    "surface_skill_primary": surface.get("skill_primary"),
                    "rootzone_skill_primary": rootzone.get("skill_primary"),
                    "prediction_content_hash": mixed.get("mixed_prediction_content_hash", ""),
                    "source_prediction_record_hash": mixed.get("source_prediction_record_hash", ""),
                    "metric_content_hash": mixed.get("metric_content_hash", ""),
                    "metric_values_content_hash": mixed.get("metric_values_content_hash", ""),
                }
            )
            out.append(candidate)
    return out


def _rho_value(policy: str) -> float:
    if policy.startswith("fixed_"):
        return float(policy.split("_", 1)[1])
    return {"rule_a": 0.5, "rule_b": 0.5, "rule_c": 0.25}.get(policy, 1.0)


def _simplicity_key(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    schedule = 0 if row.get("schedule_label") == "original_K12" else 1
    loss = 0 if row.get("support_loss_reduction") == "global_pixel" else 1
    rho_order = {
        "fixed_1.0": 0,
        "fixed_0.75": 1,
        "fixed_0.5": 2,
        "fixed_0.25": 3,
        "rule_a": 4,
        "rule_b": 5,
        "rule_c": 6,
    }.get(str(row.get("rho_policy")), 9)
    trust_order = {"none": 0, "mild_groupwise": 1, "strong_groupwise": 2}.get(str(row.get("trust_policy")), 9)
    return schedule, loss, rho_order, trust_order


def rank_candidates(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rank candidates with the preregistered score and deterministic tie-breakers."""
    ranked = [dict(row) for row in summaries]

    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        score_bucket = round(float(row.get("score", float("-inf"))), 12)
        trust_strength = {"none": 0, "mild_groupwise": 1, "strong_groupwise": 2}.get(str(row.get("trust_policy")), 0)
        return (
            -score_bucket,
            float(row.get("negative_transfer_rate_vs_K0", 1.0)),
            _simplicity_key(row),
            _rho_value(str(row.get("rho_policy", "fixed_1.0"))),
            -trust_strength,
            str(row.get("candidate_id", "")),
        )

    ranked.sort(key=key)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked


def build_calibration_audit_metadata(
    rows: Sequence[Mapping[str, Any]],
    row_paths: Sequence[str | Path],
    candidate_summaries: Sequence[Mapping[str, Any]],
    *,
    checkpoint_source_regions: Sequence[str] = (),
    candidate_set: str = "compact_v1",
) -> dict[str, Any]:
    """Summarize source-safe row provenance and candidate coverage."""
    row_file_hashes = {str(path): _file_sha256(path) for path in row_paths if Path(path).exists()}
    episode_ids = sorted({str(row.get("episode_id")) for row in rows if row.get("episode_id")})
    pseudo_regions = sorted({str(row.get("pseudo_target_region")) for row in rows if row.get("pseudo_target_region")})
    seeds = sorted(
        {
            int(seed)
            for seed in (_clean_int(row.get("seed")) for row in rows)
            if seed is not None
        }
    )
    expected_candidates = {candidate["candidate_id"] for candidate in enumerate_guard_candidates(candidate_set)}
    observed_candidate_ids = sorted({str(row.get("candidate_id")) for row in candidate_summaries if row.get("candidate_id")})
    observed_candidate_set = set(observed_candidate_ids)
    source_regions = sorted({str(region) for region in checkpoint_source_regions if region} or set(pseudo_regions))
    episode_count = len(episode_ids)
    return {
        "row_files": sorted(str(path) for path in row_paths),
        "row_file_hashes": row_file_hashes,
        "row_count": len(rows),
        "row_content_hash": _json_hash(list(rows)),
        "source_pseudo_target_episode_count": episode_count,
        "source_regions": source_regions,
        "pseudo_target_regions": pseudo_regions,
        "seeds": seeds,
        "candidate_count_observed": len(observed_candidate_ids),
        "candidate_count_expected": len(expected_candidates),
        "candidate_ids_observed": observed_candidate_ids,
        "missing_candidate_ids": sorted(expected_candidates - observed_candidate_set),
        "unexpected_candidate_ids": sorted(observed_candidate_set - expected_candidates),
        "candidate_episode_ratio": (len(observed_candidate_ids) / episode_count) if episode_count else None,
    }


def deterministic_source_subset_hash(
    *,
    final_target_region: str,
    seed: int,
    pseudo_target_regions: Sequence[str],
    source_query_max_samples: int,
) -> str:
    """Stable identifier for the coarse source-val pseudo-query subset."""
    return _json_hash(
        {
            "final_target_region": final_target_region,
            "seed": int(seed),
            "pseudo_target_regions": sorted(str(region) for region in pseudo_target_regions),
            "source_query_max_samples": int(source_query_max_samples),
            "subset_policy": "first_n_per_source_val_active_region",
        }
    )


def stage_top_candidate_ids(summaries: Sequence[Mapping[str, Any]], *, top_k: int = 5) -> list[str]:
    """Return the top candidate IDs from ranked coarse-stage summaries."""
    return [str(row.get("candidate_id")) for row in rank_candidates(summaries)[: int(top_k)]]


def filter_summaries_to_candidate_ids(
    summaries: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Keep only summaries whose candidate ID was handed off from stage 1."""
    allowed = {str(candidate_id) for candidate_id in candidate_ids}
    return [dict(row) for row in summaries if str(row.get("candidate_id")) in allowed]


def _candidate_row_dirs(source_rows_root: Path, pseudo_target_region: str, candidate: Mapping[str, Any]) -> list[Path]:
    ids = [
        str(candidate.get("candidate_id", "")),
        str(candidate.get("base_config_id", "")),
    ]
    if str(candidate.get("candidate_id", "")) != str(candidate.get("base_config_id", "")):
        ids.append(candidate_id_from_config({**candidate, "rho_policy": "fixed_1.0"}))
    return [source_rows_root / pseudo_target_region / item for item in ids if item]


def _candidate_row_files(source_rows_root: Path, pseudo_target_region: str, candidate: Mapping[str, Any]) -> list[Path]:
    files: list[Path] = []
    for row_dir in _candidate_row_dirs(source_rows_root, pseudo_target_region, candidate):
        files.extend(
            [
                row_dir / "source_safe_candidate_rows.csv",
                row_dir / "source_safe_candidate_rows.json",
            ]
        )
    return files


def _row_sample_budget(row: Mapping[str, Any]) -> int | None:
    for field in _SAMPLE_BUDGET_FIELDS:
        value = _clean_int(row.get(field))
        if value is not None:
            return value
    return None


def _strict_existing_row_invalid_reasons(
    row: Mapping[str, Any],
    *,
    expected_hash: str,
    expected_sample_budget: int | None = None,
) -> list[str]:
    reasons: list[str] = []
    missing_required = [
        field
        for field in _STRICT_EXISTING_ROW_REQUIRED_FIELDS
        if row.get(field) in (None, "")
    ]
    if missing_required:
        reasons.append("missing_required_metadata:" + ",".join(missing_required))
    observed_hash = str(row.get("candidate_config_hash", ""))
    if not observed_hash:
        reasons.append("missing_candidate_config_hash")
    elif observed_hash != expected_hash:
        reasons.append("config_hash_mismatch")
    prediction_record_path = str(row.get("prediction_record_path", "")).strip()
    if not prediction_record_path:
        reasons.append("missing_prediction_record_path")
    else:
        record_path = Path(prediction_record_path)
        if not record_path.exists():
            if not record_path.is_absolute() and row.get("_source_path"):
                source_relative_path = Path(str(row["_source_path"])).parent / record_path
            else:
                source_relative_path = record_path
            if not source_relative_path.exists():
                reasons.append("prediction_record_path_missing")
    if not str(row.get("prediction_content_hash", "")).strip():
        reasons.append("missing_prediction_content_hash")
    split_type = str(row.get("split_type", "")).strip()
    query_role = str(row.get("query_role", "")).strip()
    if split_type != "source_val" or query_role not in SOURCE_SAFE_QUERY_ROLES:
        reasons.append(f"non_source_safe_role:{split_type}/{query_role}")
    if expected_sample_budget is not None:
        observed_budget = _row_sample_budget(row)
        if observed_budget != int(expected_sample_budget):
            reasons.append(f"sample_budget_mismatch:{observed_budget}!={int(expected_sample_budget)}")
    return reasons


def _artifact_status_for_candidate(
    source_rows_root: Path,
    pseudo_target_region: str,
    candidate: Mapping[str, Any],
    *,
    expected_sample_budget: int | None = None,
) -> tuple[str, Path | None, str]:
    expected_hash = str(candidate.get("candidate_config_hash") or candidate_config_hash(candidate))
    for row_file in _candidate_row_files(source_rows_root, pseudo_target_region, candidate):
        if not row_file.exists():
            continue
        try:
            rows = load_calibration_rows([row_file])
        except Exception as exc:
            return "unreadable_artifact", row_file, str(exc)
        matching = [
            row
            for row in rows
            if str(row.get("candidate_id")) in {str(candidate.get("candidate_id")), candidate_id_from_config({**candidate, "rho_policy": "fixed_1.0"})}
            or str(row.get("base_config_id", "")) == str(candidate.get("base_config_id", ""))
        ]
        if not matching:
            return "candidate_id_mismatch", row_file, "row does not match requested candidate/base_config"
        observed_hashes = {str(row.get("candidate_config_hash", "")) for row in matching}
        strict_reasons = [
            reason
            for row in matching
            for reason in _strict_existing_row_invalid_reasons(
                row,
                expected_hash=expected_hash,
                expected_sample_budget=expected_sample_budget,
            )
        ]
        if not strict_reasons:
            return "valid", row_file, ""
        if "" in observed_hashes:
            return "missing_config_hash", row_file, ";".join(sorted(set(strict_reasons)))
        if expected_hash not in observed_hashes:
            return "config_hash_mismatch", row_file, f"expected {expected_hash}, observed {sorted(observed_hashes)}"
        return "invalid_existing_row", row_file, ";".join(sorted(set(strict_reasons)))
    return "missing_artifact", None, "source_safe_candidate_rows.csv/json not found"


def build_resume_manifest(
    *,
    base_candidates: Sequence[Mapping[str, Any]],
    pseudo_target_regions: Sequence[str],
    source_rows_root: str | Path,
    base_command_prefix: str,
    expected_sample_budget: int | None = None,
) -> dict[str, Any]:
    """Classify candidate rows as completed or missing for resumable wrappers."""
    root = Path(source_rows_root)
    manifest_rows: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    invalid_existing: list[dict[str, Any]] = []
    for region in [str(item).strip() for item in pseudo_target_regions if str(item).strip()]:
        for raw_candidate in base_candidates:
            candidate = dict(raw_candidate)
            candidate.setdefault("candidate_config_hash", candidate_config_hash(candidate))
            status, row_file, reason = _artifact_status_for_candidate(
                root,
                region,
                candidate,
                expected_sample_budget=expected_sample_budget,
            )
            row = {
                "pseudo_target_region": region,
                "candidate_id": candidate.get("candidate_id", ""),
                "base_config_id": candidate.get("base_config_id", ""),
                "candidate_config_hash": candidate.get("candidate_config_hash", ""),
                "expected_sample_budget": expected_sample_budget if expected_sample_budget is not None else "",
                "artifact_status": status,
                "artifact_path": str(row_file or ""),
                "reason": reason,
                "resume_command": (
                    f"{base_command_prefix} # rerun pseudo_target_region={region} "
                    f"candidate_id={candidate.get('candidate_id', '')}"
                ),
            }
            manifest_rows.append(row)
            if status == "valid":
                completed.append(row)
            else:
                missing.append(row)
                if row_file is not None:
                    invalid_existing.append(row)

    command_lines = [
        "# P2.8b Resume Commands",
        "",
        f"estimated_remaining_rows: {len(missing)}",
        "",
    ]
    command_lines.extend(f"- `{row['resume_command']}`  # {row['artifact_status']}" for row in missing)
    return {
        "candidate_manifest": manifest_rows,
        "completed_rows": completed,
        "missing_rows": missing,
        "invalid_existing_rows": invalid_existing,
        "estimated_remaining_rows": len(missing),
        "resume_commands_md": "\n".join(command_lines) + "\n",
    }


def _summarize_candidate_from_episode_results(candidate: Mapping[str, Any], episode_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [
        float(value)
        for value in (_clean_float(row.get("delta_vs_K0")) for row in episode_results)
        if value is not None
    ]
    if not deltas:
        score = float("-inf")
        mean_delta = None
        mean_regret = None
        neg_rate = None
    else:
        mean_delta = statistics.fmean(deltas)
        regrets = [max(0.0, -delta) for delta in deltas]
        mean_regret = statistics.fmean(regrets)
        neg_rate = sum(1 for delta in deltas if delta < 0.0) / len(deltas)
        score = mean_delta - 0.5 * mean_regret - 0.05 * neg_rate
    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "score": score,
        "mean_delta_vs_K0": mean_delta,
        "mean_regret_vs_K0": mean_regret,
        "negative_transfer_rate_vs_K0": neg_rate,
        "episode_count": len(deltas),
        "episode_results": list(episode_results),
        "schedule_label": candidate.get("schedule_label", ""),
        "support_loss_reduction": candidate.get("support_loss_reduction", "global_pixel"),
        "rho_policy": candidate.get("rho_policy", "fixed_1.0"),
        "trust_policy": candidate.get("trust_policy", "none"),
        "adapt_scope": candidate.get("adapt_scope", "all"),
        "adapt_solver": candidate.get("adapt_solver", "adamw"),
        "lr": _clean_float(candidate.get("lr")),
        "adaptation_steps": _clean_int(candidate.get("adaptation_steps")),
        "anchor_alpha": _clean_float(candidate.get("anchor_alpha")),
    }


def compute_stability_diagnostics(candidate_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute deterministic top-rank and leave-one-source-region-out diagnostics."""
    ranked = rank_candidates(candidate_summaries)
    top5 = [dict(row) for row in ranked[:5]]
    top_score = _clean_float(top5[0].get("score")) if top5 else None
    second_score = _clean_float(top5[1].get("score")) if len(top5) >= 2 else None
    fifth_or_last_score = _clean_float(top5[-1].get("score")) if top5 else None
    regions = sorted(
        {
            str(result.get("pseudo_target_region"))
            for summary in candidate_summaries
            for result in summary.get("episode_results", [])
            if result.get("pseudo_target_region")
        }
    )
    loso_rows = []
    selection_counts: dict[str, int] = {}
    if len(regions) >= 2:
        for heldout in regions:
            subset_summaries = []
            for summary in candidate_summaries:
                kept_results = [
                    result
                    for result in summary.get("episode_results", [])
                    if str(result.get("pseudo_target_region")) != heldout
                ]
                if kept_results:
                    subset_summaries.append(_summarize_candidate_from_episode_results(summary, kept_results))
            if not subset_summaries:
                continue
            selected = rank_candidates(subset_summaries)[0]
            selected_id = str(selected.get("candidate_id", ""))
            selection_counts[selected_id] = selection_counts.get(selected_id, 0) + 1
            loso_rows.append(
                {
                    "heldout_source_region": heldout,
                    "selected_candidate_id": selected_id,
                    "selected_score": selected.get("score"),
                    "candidate_count": len(subset_summaries),
                    "episode_count": selected.get("episode_count"),
                }
            )
    return {
        "top5_candidates": top5,
        "score_gap_top1_top2": (top_score - second_score) if top_score is not None and second_score is not None else None,
        "score_gap_top1_top5": (
            top_score - fifth_or_last_score if top_score is not None and fifth_or_last_score is not None else None
        ),
        "leave_one_source_region_out_enabled": len(regions) >= 2,
        "leave_one_source_region_out_regions": regions,
        "leave_one_source_region_out": loso_rows,
        "leave_one_source_region_out_selection_counts": dict(sorted(selection_counts.items())),
    }


def validate_checkpoint_source_regions(
    source_regions: Sequence[str],
    *,
    final_target_region: str,
    pseudo_target_region: str,
    allow_in_checkpoint_source_episodes: bool,
) -> dict[str, Any]:
    """Validate strict source-heldout eligibility or record explicit weaker fallback."""
    source_set = {str(region) for region in source_regions}
    overlaps = []
    if final_target_region in source_set:
        overlaps.append(f"final target {final_target_region}")
    if pseudo_target_region in source_set:
        overlaps.append(f"pseudo-target {pseudo_target_region}")
    if overlaps and not allow_in_checkpoint_source_episodes:
        joined = ", ".join(overlaps)
        raise ValueError(
            "Strict source-safe calibration requires checkpoint source_regions to exclude "
            f"both final target and pseudo-target; found {joined} in {sorted(source_set)}"
        )
    if overlaps:
        return {
            "source_safety_evidence_level": "source_safe_in_checkpoint_weaker",
            "allow_in_checkpoint_source_episodes": True,
            "weaker_evidence_reason": (
                "Explicit fallback: checkpoint source_regions contain "
                f"{', '.join(overlaps)}. Calibration query remains source_val and target_eval is not read."
            ),
            "checkpoint_source_regions": sorted(source_set),
            "final_target_region": final_target_region,
            "pseudo_target_region": pseudo_target_region,
        }
    return {
        "source_safety_evidence_level": "source_heldout_strict",
        "allow_in_checkpoint_source_episodes": bool(allow_in_checkpoint_source_episodes),
        "weaker_evidence_reason": "",
        "checkpoint_source_regions": sorted(source_set),
        "final_target_region": final_target_region,
        "pseudo_target_region": pseudo_target_region,
    }


def _source_k4_original_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    out = []
    for row in rows:
        split_type = str(row.get("split_type", ""))
        query_role = str(row.get("query_role", ""))
        if split_type in FORBIDDEN_TARGET_EVAL_ROLES or query_role in FORBIDDEN_TARGET_EVAL_ROLES:
            continue
        if _clean_int(row.get("K")) == 4 and str(row.get("schedule_label", "")) == "original_K4":
            out.append(row)
    return out


def _median(values: Iterable[float | None], default: float = 0.0) -> float:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(statistics.median(clean)) if clean else float(default)


def derive_trust_radii(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Derive mild/strong groupwise trust radii from source K4-original drift."""
    source_rows = _source_k4_original_rows(rows)
    coeff_values = []
    for row in source_rows:
        parts = [
            _clean_float(row.get("target_parameter_l2_drift_post_anchor_adapter_coeff_bottleneck")),
            _clean_float(row.get("target_parameter_l2_drift_post_anchor_adapter_coeff_dec2")),
            _clean_float(row.get("target_parameter_l2_drift_post_anchor_adapter_coeff_dec1")),
        ]
        clean = [part for part in parts if part is not None]
        coeff_values.append(math.sqrt(sum(float(part) ** 2 for part in clean)) if clean else None)
    mild = {
        "total": _median(row.get("target_parameter_l2_drift_post_anchor_total") for row in source_rows),
        "prompt": _median(row.get("target_parameter_l2_drift_post_anchor_target_prompt") for row in source_rows),
        "gain": _median(row.get("target_parameter_l2_drift_post_anchor_monthly_gain") for row in source_rows),
        "coeff": _median(coeff_values),
        "spatial": _median(row.get("target_parameter_l2_drift_post_anchor_spatial_refine") for row in source_rows),
    }
    strong = {key: 0.5 * value for key, value in mild.items()}
    none = {key: 0.0 for key in mild}
    return {
        "none": none,
        "mild_groupwise": mild,
        "strong_groupwise": strong,
    }


def compute_guard_config_hash(config: Mapping[str, Any]) -> str:
    payload = dict(config)
    payload.pop("guard_config_hash", None)
    return _json_hash(payload)


def _fixed_rho_from_policy(policy: str) -> float | None:
    if policy.startswith("fixed_"):
        return float(policy.split("_", 1)[1])
    return None


def build_selected_guard_config(
    *,
    candidate: Mapping[str, Any],
    ranking_summary: Mapping[str, Any],
    trust_radii: Mapping[str, Mapping[str, float]],
    final_target_region: str,
    seed: int,
    evidence_level: str,
    source_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the locked target-eval guard config and stable hash."""
    trust_policy = str(candidate.get("trust_policy", "none"))
    radii = dict(trust_radii.get(trust_policy, trust_radii.get("none", {})))
    rho_policy = str(candidate.get("rho_policy", "fixed_1.0"))
    fixed_rho = _fixed_rho_from_policy(rho_policy)
    strict_source_heldout = evidence_level == "source_heldout_strict"
    config = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate.get("candidate_id", ""),
        "final_target_region": final_target_region,
        "seed": int(seed),
        "K": 12,
        "adapt_scope": candidate.get("adapt_scope", "all"),
        "adapt_solver": candidate.get("adapt_solver", "adamw"),
        "schedule_label": candidate.get("schedule_label", ""),
        "support_loss_reduction": candidate.get("support_loss_reduction", "global_pixel"),
        "lr": _clean_float(candidate.get("lr")),
        "adaptation_steps": _clean_int(candidate.get("adaptation_steps")),
        "anchor_alpha": _clean_float(candidate.get("anchor_alpha")),
        "rho_policy": rho_policy,
        "adapt_mix_rho": fixed_rho,
        "rho_rule_source": "target_support_gradient_diagnostics_after_adaptation_before_eval"
        if fixed_rho is None
        else "fixed_source_safe_calibration",
        "trust_policy": trust_policy,
        "trust_region_mode": "none" if trust_policy == "none" else "groupwise",
        "trust_total_radius": float(radii.get("total", 0.0)),
        "trust_prompt_radius": float(radii.get("prompt", 0.0)),
        "trust_gain_radius": float(radii.get("gain", 0.0)),
        "trust_coeff_radius": float(radii.get("coeff", 0.0)),
        "trust_spatial_radius": float(radii.get("spatial", 0.0)),
        "selection_score": _clean_float(ranking_summary.get("score")),
        "selection_mean_delta_vs_K0": _clean_float(ranking_summary.get("mean_delta_vs_K0")),
        "selection_mean_regret_vs_K0": _clean_float(ranking_summary.get("mean_regret_vs_K0")),
        "selection_negative_transfer_rate_vs_K0": _clean_float(
            ranking_summary.get("negative_transfer_rate_vs_K0")
        ),
        "selection_query_role": "source_val_pseudo_query_only",
        "selection_label_usage": "source_pseudo_target_support_labels_only",
        "target_eval_usage": "never_read_by_calibration",
        "target_val_usage": "unused_in_main_protocol",
        "source_safety_evidence_level": evidence_level,
        "calibration_mode": "source_heldout_pseudo_target" if strict_source_heldout else "in_checkpoint_source_dev",
        "paper_grade_source_heldout": bool(strict_source_heldout),
        "source_metadata": dict(source_metadata),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    config["guard_config_hash"] = compute_guard_config_hash(config)
    return config


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row if key != "episode_results"})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_rankings_md(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    headers = [
        "rank",
        "candidate_id",
        "score",
        "mean_delta_vs_K0",
        "mean_regret_vs_K0",
        "negative_transfer_rate_vs_K0",
        "episode_count",
    ]
    lines = [
        "# P2.8 Source-Safe Guard Candidate Rankings",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join([":--"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(header, "")) for header in headers) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_md(path: Path, summary: Mapping[str, Any]) -> None:
    selected = summary.get("selected_guard_config", {})
    lines = [
        "# P2.8 Source-Safe Guard Calibration Summary",
        "",
        f"- schema_version: `{summary.get('schema_version')}`",
        f"- source_safety_evidence_level: `{summary.get('source_safety_evidence_level')}`",
        f"- selected_candidate: `{selected.get('candidate_id')}`",
        f"- guard_config_hash: `{selected.get('guard_config_hash')}`",
        f"- calibration_query_role: `source_val_pseudo_query_only`",
        f"- target_eval_usage: `never_read_by_calibration`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _checkpoint_source_regions(path: str | None) -> list[str]:
    if not path:
        return []
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        return []
    try:
        import torch

        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception:
        return []
    config = dict(checkpoint.get("config", {}))
    if config.get("source_regions"):
        return [str(region) for region in config["source_regions"]]
    if config.get("source_region_global_indices"):
        return [f"US-R{int(idx) + 1}" for idx in config["source_region_global_indices"]]
    return []


def _split_regions(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate P2.8 source-safe conservative guard from source rows.")
    parser.add_argument("--calibration_rows", nargs="*", default=[])
    parser.add_argument("--input_roots", nargs="*", default=[])
    parser.add_argument("--output_dir", default="artifacts/runs/phase5_hyperda_p2_8_source_safe_guard_calibration")
    parser.add_argument("--final_target_region", default="US-R1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source_checkpoint", default="")
    parser.add_argument("--checkpoint_source_regions", default="")
    parser.add_argument("--allow_in_checkpoint_source_episodes", action="store_true")
    parser.add_argument("--write_candidate_plan", default="")
    parser.add_argument("--candidate_set", default="compact_v1", choices=["compact_v1", "full_v1"])
    parser.add_argument("--calibration_stage", default="coarse", choices=["coarse", "final"])
    parser.add_argument("--source_query_max_samples", type=int, default=256)
    parser.add_argument("--top_candidate_ids", default="")
    parser.add_argument("--pseudo_target_regions", default="")
    parser.add_argument("--source_rows_root", default="")
    parser.add_argument("--resume_command_prefix", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.write_candidate_plan:
        _write_json(Path(args.write_candidate_plan), enumerate_guard_candidates(args.candidate_set))

    paths = [Path(path) for path in args.calibration_rows]
    paths.extend(discover_calibration_rows(args.input_roots))
    if not paths:
        raise SystemExit("No calibration rows provided. Pass --calibration_rows or --input_roots.")

    rows = load_calibration_rows(paths)
    rows = expand_logical_candidate_rows_from_prediction_records(rows, candidate_set=args.candidate_set)
    checkpoint_source_regions = _split_regions(args.checkpoint_source_regions) or _checkpoint_source_regions(
        args.source_checkpoint
    )
    pseudo_regions = sorted({str(row.get("pseudo_target_region")) for row in rows if row.get("pseudo_target_region")})
    validation_metadata = []
    if checkpoint_source_regions:
        for pseudo_region in pseudo_regions:
            validation_metadata.append(
                validate_checkpoint_source_regions(
                    checkpoint_source_regions,
                    final_target_region=args.final_target_region,
                    pseudo_target_region=pseudo_region,
                    allow_in_checkpoint_source_episodes=bool(args.allow_in_checkpoint_source_episodes),
                )
            )
    elif not args.allow_in_checkpoint_source_episodes:
        raise SystemExit(
            "Strict P2.8 calibration requires checkpoint source_regions. "
            "Pass --source_checkpoint/--checkpoint_source_regions, or explicitly opt in with "
            "--allow_in_checkpoint_source_episodes for weaker local evidence."
        )
    evidence_level = (
        "source_safe_in_checkpoint_weaker"
        if args.allow_in_checkpoint_source_episodes
        or any(item.get("source_safety_evidence_level") == "source_safe_in_checkpoint_weaker" for item in validation_metadata)
        else "source_heldout_strict"
    )

    trust_radii = derive_trust_radii(rows)
    summaries = score_candidates(rows)
    top_candidate_ids = _split_regions(args.top_candidate_ids)
    if args.calibration_stage == "final" and top_candidate_ids:
        summaries = filter_summaries_to_candidate_ids(summaries, top_candidate_ids)
        if not summaries:
            raise SystemExit("Final calibration had no rows matching --top_candidate_ids")
    rankings = rank_candidates(summaries)
    selected = rankings[0]
    audit_metadata = build_calibration_audit_metadata(
        rows,
        paths,
        summaries,
        checkpoint_source_regions=checkpoint_source_regions,
        candidate_set=args.candidate_set,
    )
    pseudo_target_regions = _split_regions(args.pseudo_target_regions) or pseudo_regions
    subset_hash = deterministic_source_subset_hash(
        final_target_region=args.final_target_region,
        seed=args.seed,
        pseudo_target_regions=pseudo_target_regions,
        source_query_max_samples=args.source_query_max_samples,
    )
    source_metadata = {
        **audit_metadata,
        "candidate_set": args.candidate_set,
        "calibration_stage": args.calibration_stage,
        "source_query_max_samples": int(args.source_query_max_samples),
        "deterministic_source_subset_hash": subset_hash,
        "top_candidate_ids_input": top_candidate_ids,
        "checkpoint_source_regions": checkpoint_source_regions,
        "checkpoint_hashes": sorted(
            {str(row.get("source_checkpoint_sha256")) for row in rows if row.get("source_checkpoint_sha256")}
        ),
        "split_manifest_hashes": sorted(
            {str(row.get("split_manifest_sha256")) for row in rows if row.get("split_manifest_sha256")}
        ),
        "validation_metadata": validation_metadata,
    }
    selected_config = build_selected_guard_config(
        candidate=selected,
        ranking_summary=selected,
        trust_radii=trust_radii,
        final_target_region=args.final_target_region,
        seed=args.seed,
        evidence_level=evidence_level,
        source_metadata=source_metadata,
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "source_safety_evidence_level": evidence_level,
        "score_formula": "mean_delta_vs_K0 - 0.5 * mean_regret_vs_K0 - 0.05 * negative_transfer_rate_vs_K0",
        "tie_breakers": [
            "lower_negative_transfer_rate",
            "simpler_config",
            "smaller_rho_or_stronger_guard_when_effectively_tied",
        ],
        "calibration_query_role": "source_val_pseudo_query_only",
        "target_eval_usage": "never_read_by_calibration",
        "trust_radii": trust_radii,
        "calibration_audit": source_metadata,
        "stability_diagnostics": compute_stability_diagnostics(summaries),
        "selected_guard_config": selected_config,
    }
    top5_ids = stage_top_candidate_ids(summaries, top_k=5)
    summary["top5_candidate_ids"] = top5_ids
    summary["deterministic_source_subset_hash"] = subset_hash
    per_episode = {
        str(row.get("episode_id")): {
            "pseudo_target_region": row.get("pseudo_target_region", ""),
            "query_role": row.get("query_role", ""),
            "split_type": row.get("split_type", ""),
            "row_hash": _json_hash(row),
        }
        for row in rows
    }
    leakage_metadata = {
        "target_eval_loaded": False,
        "target_eval_labels_loaded": False,
        "target_eval_features_loaded": False,
        "calibration_query_split": "source_val",
        "support_label_source": "source_pseudo_target_K12_support_only",
        "final_target_region": args.final_target_region,
        "evidence_level": evidence_level,
    }

    _write_json(output_dir / "selected_guard_config.yaml", selected_config)
    _write_json(output_dir / "selected_guard_config.json", selected_config)
    _write_json(output_dir / "source_safe_calibration_summary.json", summary)
    _write_json(output_dir / "candidate_rankings.json", rankings)
    _write_json(output_dir / "top5_stability.json", summary["stability_diagnostics"])
    _write_json(output_dir / "trust_radii.json", trust_radii)
    _write_json(output_dir / "per_episode_metadata.json", per_episode)
    _write_json(output_dir / "leakage_protocol_metadata.json", leakage_metadata)
    _write_json(output_dir / "calibration_audit_metadata.json", source_metadata)
    _write_json(output_dir / "stability_diagnostics.json", summary["stability_diagnostics"])
    _write_csv(output_dir / "candidate_rankings.csv", rankings)
    _write_csv(output_dir / "source_safe_calibration_summary.csv", [selected_config])
    _write_csv(output_dir / "top5_stability.csv", summary["stability_diagnostics"].get("top5_candidates", []))
    _write_csv(output_dir / "top5_candidates.csv", summary["stability_diagnostics"].get("top5_candidates", []))
    _write_csv(
        output_dir / "leave_one_source_region_out_stability.csv",
        summary["stability_diagnostics"].get("leave_one_source_region_out", []),
    )
    _write_rankings_md(output_dir / "candidate_rankings.md", rankings)
    _write_summary_md(output_dir / "source_safe_calibration_summary.md", summary)
    stage_prefix = "final" if args.calibration_stage == "final" else "coarse"
    _write_json(output_dir / f"{stage_prefix}_source_safe_calibration_summary.json", summary)
    _write_csv(output_dir / f"{stage_prefix}_source_safe_calibration_summary.csv", [selected_config])
    _write_summary_md(output_dir / f"{stage_prefix}_source_safe_calibration_summary.md", summary)
    if args.calibration_stage == "final":
        _write_json(output_dir / "final_candidate_rankings.json", rankings)
        _write_csv(output_dir / "final_candidate_rankings.csv", rankings)
    if args.source_rows_root or args.pseudo_target_regions:
        resume_manifest = build_resume_manifest(
            base_candidates=required_gpu_row_configs(
                candidate_set=args.candidate_set,
                top_candidate_ids=top_candidate_ids if args.calibration_stage == "final" else (),
            ),
            pseudo_target_regions=pseudo_target_regions,
            source_rows_root=args.source_rows_root or output_dir / "source_val_candidate_rows",
            base_command_prefix=args.resume_command_prefix
            or "RESUME=1 SKIP_EXISTING=1 bash run/phase5_hyperda_p2_8_source_safe_guard_calibration.sh",
            expected_sample_budget=int(args.source_query_max_samples),
        )
        _write_csv(output_dir / "candidate_manifest.csv", resume_manifest["candidate_manifest"])
        _write_csv(output_dir / "completed_rows.csv", resume_manifest["completed_rows"])
        _write_csv(output_dir / "missing_rows.csv", resume_manifest["missing_rows"])
        _write_csv(output_dir / "invalid_existing_rows.csv", resume_manifest["invalid_existing_rows"])
        (output_dir / "resume_commands.md").write_text(resume_manifest["resume_commands_md"], encoding="utf-8")
        (output_dir / "estimated_remaining_rows").write_text(
            str(resume_manifest["estimated_remaining_rows"]) + "\n",
            encoding="utf-8",
        )
    print(f"Selected P2.8 guard config: {selected_config['candidate_id']}")
    print(f"guard_config_hash={selected_config['guard_config_hash']}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
