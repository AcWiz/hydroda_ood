"""Source-side adapter basis saliency utilities for HyperDA."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


ADAPTER_LAYER_NAMES = ("bottleneck", "dec2", "dec1")
SOURCE_SALIENCY_SCHEMA_VERSION = "hyperda_source_basis_saliency_prior_v1"
SOURCE_SALIENCY_PRIOR_FORM = "adapter_layer_basis_normalized_logit_bias"
ALLOWED_SALIENCY_SOURCE_SPLITS = frozenset({"source_fit", "source_episode", "source_side_episode"})
FORBIDDEN_SALIENCY_SOURCE_SPLITS = frozenset({"target_val", "target_eval", "target_query", "target_full_train"})


def tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    cpu = tensor.detach().cpu().contiguous()
    digest.update(str(tuple(cpu.shape)).encode("utf-8"))
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def normalize_saliency_scores(scores: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return finite per-layer z-scored saliency for gate-logit biasing."""
    tensor = torch.as_tensor(scores, dtype=torch.float32).detach().clone()
    if tensor.ndim != 2:
        raise ValueError("saliency scores must have shape [n_layers, n_basis]")
    if tensor.numel() == 0:
        raise ValueError("saliency scores must be non-empty")
    if not torch.isfinite(tensor).all():
        raise ValueError("saliency scores must be finite")
    mean = tensor.mean(dim=1, keepdim=True)
    std = tensor.std(dim=1, unbiased=False, keepdim=True)
    normalized = (tensor - mean) / std.clamp_min(float(eps))
    normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    if not torch.isfinite(normalized).all():
        raise ValueError("normalized saliency prior must be finite")
    return normalized


def _as_split_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    try:
        return {str(item) for item in value}
    except TypeError:
        return {str(value)}


def validate_saliency_metadata(metadata: Mapping[str, Any]) -> None:
    """Reject saliency artifacts that are not source-side only."""
    source_splits = _as_split_set(metadata.get("source_splits") or metadata.get("split_roles"))
    if not source_splits:
        source_split = metadata.get("source_split")
        if source_split is not None:
            source_splits = {str(source_split)}
    forbidden = source_splits & FORBIDDEN_SALIENCY_SOURCE_SPLITS
    if forbidden:
        raise ValueError(
            "source saliency prior cannot use target-side split roles: "
            f"{sorted(forbidden)}"
        )
    if source_splits and not source_splits <= ALLOWED_SALIENCY_SOURCE_SPLITS:
        raise ValueError(
            "source saliency prior split roles must be source-side only; "
            f"got {sorted(source_splits)}"
        )
    for key in ("target_val_usage", "target_eval_usage"):
        value = str(metadata.get(key, "")).lower()
        if (
            ("selection" in value and "no_selection" not in value)
            or ("training" in value and "no_training" not in value)
            or "calibration" in value
        ):
            raise ValueError(f"source saliency prior metadata has unsafe {key}={metadata.get(key)!r}")
    payload = json.dumps(dict(metadata), sort_keys=True, default=str).lower()
    for token in ("target_val_grid_search", "target_eval_grid_search", "target_query_labels"):
        if token in payload:
            raise ValueError(f"source saliency prior metadata references forbidden token {token!r}")


def make_saliency_artifact(
    scores: torch.Tensor,
    *,
    score_type: str,
    source_split: str = "source_fit",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a validated source-side saliency artifact payload."""
    scores_t = torch.as_tensor(scores, dtype=torch.float32).detach().cpu()
    prior = normalize_saliency_scores(scores_t)
    if scores_t.shape[0] != len(ADAPTER_LAYER_NAMES):
        raise ValueError(
            "saliency scores must have one row per HyperDA adapter layer; "
            f"expected {len(ADAPTER_LAYER_NAMES)}, got {scores_t.shape[0]}"
        )
    meta = dict(metadata or {})
    meta.update(
        {
            "schema_version": SOURCE_SALIENCY_SCHEMA_VERSION,
            "prior_form": SOURCE_SALIENCY_PRIOR_FORM,
            "score_type": str(score_type),
            "source_split": str(source_split),
            "source_splits": [str(source_split)],
            "layer_names": list(ADAPTER_LAYER_NAMES),
            "shape": list(scores_t.shape),
            "target_val_usage": meta.get("target_val_usage", "unused_in_main_protocol"),
            "target_eval_usage": meta.get("target_eval_usage", "final_eval_only_no_selection"),
        }
    )
    validate_saliency_metadata(meta)
    return {
        "schema_version": SOURCE_SALIENCY_SCHEMA_VERSION,
        "prior_form": SOURCE_SALIENCY_PRIOR_FORM,
        "scores": scores_t,
        "prior": prior,
        "metadata": meta,
        "prior_sha256": tensor_sha256(prior),
        "scores_sha256": tensor_sha256(scores_t),
    }


def load_source_saliency_prior(
    path: str | Path,
    *,
    expected_n_layers: int,
    expected_n_basis: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load and validate a source-side saliency prior artifact."""
    artifact_path = Path(path).expanduser()
    if not artifact_path.exists():
        raise FileNotFoundError(f"source saliency prior not found: {artifact_path}")
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("source saliency prior artifact must be a mapping")
    schema = payload.get("schema_version") or payload.get("metadata", {}).get("schema_version")
    if schema != SOURCE_SALIENCY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported source saliency prior schema_version={schema!r}; "
            f"expected {SOURCE_SALIENCY_SCHEMA_VERSION!r}"
        )
    metadata = dict(payload.get("metadata", {}))
    validate_saliency_metadata(metadata)
    if "prior" in payload:
        prior = torch.as_tensor(payload["prior"], dtype=torch.float32).detach().cpu()
    elif "scores" in payload:
        prior = normalize_saliency_scores(torch.as_tensor(payload["scores"], dtype=torch.float32))
    else:
        raise ValueError("source saliency prior artifact must contain 'prior' or 'scores'")
    expected_shape = (int(expected_n_layers), int(expected_n_basis))
    if tuple(prior.shape) != expected_shape:
        raise ValueError(
            "source saliency prior shape mismatch: "
            f"expected {expected_shape}, got {tuple(prior.shape)}"
        )
    if not torch.isfinite(prior).all():
        raise ValueError("source saliency prior must be finite")
    metadata.setdefault("schema_version", SOURCE_SALIENCY_SCHEMA_VERSION)
    metadata.setdefault("prior_form", SOURCE_SALIENCY_PRIOR_FORM)
    metadata.setdefault("prior_sha256", payload.get("prior_sha256") or tensor_sha256(prior))
    metadata.setdefault("path", str(artifact_path))
    return prior, metadata
