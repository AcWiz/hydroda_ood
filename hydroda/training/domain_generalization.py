"""Domain-generalization helpers for source-side neural baselines."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import Dataset


FORBIDDEN_TARGET_CONTEXT_KEYS = {
    "increment_surface",
    "increment_rootzone",
    "analysis_surface",
    "analysis_rootzone",
    "target",
    "y",
    "loss_mask",
    "metric_mask",
}

DEFAULT_IU_SCORE_CAP = 10.0


def _as_feature_matrix(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 2:
        return features.float()
    if features.ndim == 4:
        return features.permute(0, 2, 3, 1).reshape(-1, features.size(1)).float()
    if features.ndim > 2:
        return features.reshape(features.size(0), -1).float()
    raise ValueError(f"features must have at least 2 dimensions, got shape={tuple(features.shape)}")


def coral_loss(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    """Deep CORAL feature-alignment loss for dense feature maps."""
    source = _as_feature_matrix(source_features)
    target = _as_feature_matrix(target_features)
    if source.size(1) != target.size(1):
        raise ValueError(
            f"Feature dimensions must match for CORAL: source={source.size(1)}, target={target.size(1)}"
        )

    mean_loss = (source.mean(dim=0) - target.mean(dim=0)).pow(2).mean()
    if source.size(0) < 2 or target.size(0) < 2:
        return mean_loss

    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    source_cov = source_centered.t().matmul(source_centered) / float(source.size(0) - 1)
    target_cov = target_centered.t().matmul(target_centered) / float(target.size(0) - 1)
    cov_loss = (source_cov - target_cov).pow(2).sum() / (4.0 * source.size(1) * source.size(1))
    return mean_loss + cov_loss


def _correlation_matrix(features: torch.Tensor) -> torch.Tensor:
    centered = features - features.mean(dim=0, keepdim=True)
    if centered.size(0) < 2:
        return torch.zeros(
            (features.size(1), features.size(1)),
            device=features.device,
            dtype=features.dtype,
        )
    std = centered.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    standardized = centered / std
    return standardized.t().matmul(standardized) / float(centered.size(0))


def tca_correlation_alignment_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
) -> torch.Tensor:
    """Align source/target feature correlation matrices using target inputs only."""
    source = _as_feature_matrix(source_features)
    target = _as_feature_matrix(target_features)
    if source.size(1) != target.size(1):
        raise ValueError(
            f"Feature dimensions must match for TCA: source={source.size(1)}, target={target.size(1)}"
        )
    source_corr = _correlation_matrix(source)
    target_corr = _correlation_matrix(target)
    return (source_corr - target_corr).pow(2).mean()


def prediction_consistency_loss(clean_predictions: torch.Tensor, augmented_predictions: torch.Tensor) -> torch.Tensor:
    """Mean-squared consistency loss between clean and augmented predictions."""
    if clean_predictions.shape != augmented_predictions.shape:
        raise ValueError(
            "Prediction shapes must match for self-bootstrap consistency: "
            f"clean={tuple(clean_predictions.shape)}, augmented={tuple(augmented_predictions.shape)}"
        )
    return (clean_predictions.float() - augmented_predictions.float()).pow(2).mean()


def domain_loss_variance(domain_losses: Mapping[Any, torch.Tensor] | torch.Tensor | Sequence[float]) -> torch.Tensor:
    """Variance across source-domain losses.

    DISAM uses this as a source-only regularizer: source regions with identical
    loss have zero penalty; uneven per-region losses receive a positive penalty.
    """
    if isinstance(domain_losses, Mapping):
        values = list(domain_losses.values())
        if not values:
            return torch.tensor(0.0)
        tensor_values = torch.stack([torch.as_tensor(v).float() for v in values])
    else:
        tensor_values = torch.as_tensor(domain_losses).float()
    tensor_values = tensor_values.reshape(-1)
    if tensor_values.numel() < 2:
        return tensor_values.sum() * 0.0
    return tensor_values.var(unbiased=False)


def unknown_domain_inconsistency_loss(
    clean_domain_losses: Mapping[Any, torch.Tensor],
    perturbed_domain_losses: Mapping[Any, torch.Tensor],
) -> torch.Tensor:
    """Mean-squared source-region loss change under an unknown-domain perturbation.

    This is the HydroDA dense-regression adaptation of UDIM: the SAM-perturbed
    loss landscape is treated as a source-only proxy for an unknown domain, and
    the model is penalized when source-region losses move inconsistently across
    that perturbation.
    """
    common_keys = sorted(set(clean_domain_losses).intersection(perturbed_domain_losses))
    if common_keys:
        reference = clean_domain_losses[common_keys[0]]
    elif clean_domain_losses:
        reference = next(iter(clean_domain_losses.values()))
    elif perturbed_domain_losses:
        reference = next(iter(perturbed_domain_losses.values()))
    else:
        return torch.tensor(0.0)
    if len(common_keys) < 2:
        return torch.as_tensor(reference).float().sum() * 0.0
    clean = torch.stack([torch.as_tensor(clean_domain_losses[key]).float() for key in common_keys])
    perturbed = torch.stack([torch.as_tensor(perturbed_domain_losses[key]).float() for key in common_keys])
    return (clean - perturbed).pow(2).mean()


def _domain_index_groups(
    domain_ids: Sequence[Any] | torch.Tensor,
    *,
    device: torch.device,
) -> Dict[Any, torch.Tensor]:
    if isinstance(domain_ids, torch.Tensor):
        ids_cpu = domain_ids.detach().cpu().reshape(-1).tolist()
    else:
        ids_cpu = list(domain_ids)
    groups: Dict[Any, list[int]] = {}
    for idx, domain_id in enumerate(ids_cpu):
        if domain_id is None:
            continue
        groups.setdefault(str(domain_id), []).append(idx)
    return {
        domain_id: torch.as_tensor(indices, dtype=torch.long, device=device)
        for domain_id, indices in groups.items()
        if indices
    }


def domain_masked_huber_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    domain_ids: Sequence[Any] | torch.Tensor,
    *,
    latitude_weight: Optional[torch.Tensor] = None,
    delta: float = 0.01,
) -> Dict[Any, torch.Tensor]:
    """Compute differentiable masked Huber loss for each source domain."""
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes must match, got {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.ndim != 4:
        raise ValueError(f"pred/target must be [B, C, H, W], got shape={tuple(pred.shape)}")
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError(f"mask must be [B, H, W] or [B, 1, H, W], got shape={tuple(mask.shape)}")
    groups = _domain_index_groups(domain_ids, device=pred.device)
    if not groups:
        return {}

    abs_diff = torch.abs(pred - target)
    loss_raw = torch.where(
        abs_diff < delta,
        0.5 * (pred - target).pow(2) / delta,
        abs_diff - 0.5 * delta,
    )
    weight = mask.to(dtype=pred.dtype, device=pred.device)
    if latitude_weight is not None:
        lat_w = latitude_weight.to(dtype=pred.dtype, device=pred.device)
        if lat_w.ndim == 2:
            lat_w = lat_w.unsqueeze(0).unsqueeze(0)
        elif lat_w.ndim == 3:
            lat_w = lat_w.unsqueeze(1)
        weight = weight * lat_w
    weight = weight.expand_as(loss_raw)

    per_domain: Dict[Any, torch.Tensor] = {}
    for domain_id, indices in groups.items():
        domain_loss = loss_raw.index_select(0, indices)
        domain_weight = weight.index_select(0, indices)
        denom = domain_weight.sum().clamp_min(1e-6)
        per_domain[domain_id] = (domain_loss * domain_weight).sum() / denom
    return per_domain


def _active_region_numbers(active_region_ids: Optional[Sequence[Any]]) -> list[int]:
    numbers: list[int] = []
    if not active_region_ids:
        return numbers
    for item in active_region_ids:
        if item is None:
            continue
        if isinstance(item, str):
            items = [item]
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            items = list(item)
        else:
            items = [item]
        for region_id in items:
            if region_id is None:
                continue
            text = str(region_id)
            try:
                number = int(text.split("-R", 1)[1]) if "-R" in text else int(text)
            except ValueError:
                continue
            if number > 0 and number not in numbers:
                numbers.append(number)
    return numbers


def region_masked_huber_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    region_mask_integer: torch.Tensor,
    *,
    active_region_ids: Optional[Sequence[Any]] = None,
    latitude_weight: Optional[torch.Tensor] = None,
    delta: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """Compute per-source-region losses from pooled spatial samples."""
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes must match, got {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.ndim != 4:
        raise ValueError(f"pred/target must be [B, C, H, W], got shape={tuple(pred.shape)}")
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError(f"mask must be [B, H, W] or [B, 1, H, W], got shape={tuple(mask.shape)}")
    if region_mask_integer.ndim == 2:
        region_mask_integer = region_mask_integer.unsqueeze(0)
    if region_mask_integer.ndim == 3:
        region_mask_integer = region_mask_integer.unsqueeze(1)
    if region_mask_integer.ndim != 4:
        raise ValueError(
            "region_mask_integer must be [H, W], [B, H, W], or [B, 1, H, W], "
            f"got shape={tuple(region_mask_integer.shape)}"
        )

    region_numbers = _active_region_numbers(active_region_ids)
    if not region_numbers:
        unique = torch.unique(region_mask_integer.detach())
        region_numbers = sorted(int(v.item()) for v in unique if int(v.item()) > 0)
    if len(region_numbers) < 2:
        return {}

    region_mask = region_mask_integer.to(device=pred.device)
    if region_mask.size(0) == 1 and pred.size(0) > 1:
        region_mask = region_mask.expand(pred.size(0), -1, -1, -1)
    abs_diff = torch.abs(pred - target)
    loss_raw = torch.where(
        abs_diff < delta,
        0.5 * (pred - target).pow(2) / delta,
        abs_diff - 0.5 * delta,
    )
    base_weight = mask.to(dtype=pred.dtype, device=pred.device)
    if latitude_weight is not None:
        lat_w = latitude_weight.to(dtype=pred.dtype, device=pred.device)
        if lat_w.ndim == 2:
            lat_w = lat_w.unsqueeze(0).unsqueeze(0)
        elif lat_w.ndim == 3:
            lat_w = lat_w.unsqueeze(1)
        base_weight = base_weight * lat_w

    per_domain: Dict[str, torch.Tensor] = {}
    for region_num in region_numbers:
        domain_weight = base_weight * (region_mask == region_num).to(dtype=pred.dtype)
        domain_weight = domain_weight.expand_as(loss_raw)
        denom = domain_weight.sum()
        if denom.item() <= 0.0:
            continue
        per_domain[f"US-R{region_num}"] = (loss_raw * domain_weight).sum() / denom.clamp_min(1e-6)
    return per_domain


def _feature_matrix_for_indices(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    selected = features.index_select(0, indices)
    if selected.ndim == 2:
        return selected.float()
    if selected.ndim == 4:
        return selected.permute(0, 2, 3, 1).reshape(-1, selected.size(1)).float()
    if selected.ndim > 2:
        return selected.reshape(selected.size(0), -1).float()
    raise ValueError(f"features must have at least 2 dimensions, got shape={tuple(features.shape)}")


def moment_alignment_loss(
    features: torch.Tensor,
    domain_ids: Sequence[Any] | torch.Tensor,
    *,
    order: int = 2,
) -> torch.Tensor:
    """Align source-region feature moments to pooled source moments.

    This is a dense-regression adaptation of recent moment-alignment DG ideas:
    it uses only source-region IDs and aligns each source domain's feature mean,
    and optionally variance, to the pooled source feature moments.
    """
    if order not in {1, 2}:
        raise ValueError(f"order must be 1 or 2, got {order}")
    if features.ndim < 2:
        raise ValueError(f"features must have at least 2 dimensions, got shape={tuple(features.shape)}")
    groups = _domain_index_groups(domain_ids, device=features.device)
    if len(groups) < 2:
        return features.sum() * 0.0

    all_indices = torch.arange(features.size(0), dtype=torch.long, device=features.device)
    pooled = _feature_matrix_for_indices(features, all_indices)
    pooled_mean = pooled.mean(dim=0)
    pooled_var = pooled.var(dim=0, unbiased=False) if pooled.size(0) > 1 else torch.zeros_like(pooled_mean)

    losses = []
    for indices in groups.values():
        domain = _feature_matrix_for_indices(features, indices)
        domain_mean = domain.mean(dim=0)
        losses.append((domain_mean - pooled_mean).pow(2).mean())
        if order >= 2:
            domain_var = (
                domain.var(dim=0, unbiased=False)
                if domain.size(0) > 1
                else torch.zeros_like(pooled_var)
            )
            losses.append((domain_var - pooled_var).pow(2).mean())
    return torch.stack(losses).mean()


def _region_feature_matrix(
    features: torch.Tensor,
    region_mask_integer: torch.Tensor,
    region_num: int,
) -> torch.Tensor:
    if features.ndim == 2:
        raise ValueError("region-based feature grouping requires spatial feature maps")
    if features.ndim != 4:
        if features.ndim > 2:
            features = features.reshape(features.size(0), features.size(1), -1, 1)
        else:
            raise ValueError(f"features must have at least 3 dimensions, got shape={tuple(features.shape)}")
    if region_mask_integer.ndim == 2:
        region_mask_integer = region_mask_integer.unsqueeze(0)
    if region_mask_integer.ndim == 4 and region_mask_integer.size(1) == 1:
        region_mask_integer = region_mask_integer[:, 0]
    if region_mask_integer.ndim != 3:
        raise ValueError(
            "region_mask_integer must be [H, W], [B, H, W], or [B, 1, H, W], "
            f"got shape={tuple(region_mask_integer.shape)}"
        )
    if region_mask_integer.shape[-2:] != features.shape[-2:]:
        nearest = torch.nn.functional.interpolate(
            region_mask_integer.unsqueeze(1).float(),
            size=features.shape[-2:],
            mode="nearest",
        )
        region_mask_integer = nearest[:, 0].to(dtype=torch.long)
    if region_mask_integer.size(0) == 1 and features.size(0) > 1:
        region_mask_integer = region_mask_integer.expand(features.size(0), -1, -1)
    region_mask = region_mask_integer.to(device=features.device) == region_num
    selected = features.permute(0, 2, 3, 1)[region_mask]
    return selected.float()


def region_moment_alignment_loss(
    features: torch.Tensor,
    region_mask_integer: torch.Tensor,
    *,
    active_region_ids: Optional[Sequence[Any]] = None,
    order: int = 2,
) -> torch.Tensor:
    """Align spatial source-region feature moments without expanding samples."""
    if order not in {1, 2}:
        raise ValueError(f"order must be 1 or 2, got {order}")
    region_numbers = _active_region_numbers(active_region_ids)
    if not region_numbers:
        unique = torch.unique(region_mask_integer.detach())
        region_numbers = sorted(int(v.item()) for v in unique if int(v.item()) > 0)
    matrices = []
    for region_num in region_numbers:
        matrix = _region_feature_matrix(features, region_mask_integer, region_num)
        if matrix.numel() > 0:
            matrices.append(matrix)
    if len(matrices) < 2:
        return features.sum() * 0.0
    pooled = torch.cat(matrices, dim=0)
    pooled_mean = pooled.mean(dim=0)
    pooled_var = pooled.var(dim=0, unbiased=False) if pooled.size(0) > 1 else torch.zeros_like(pooled_mean)
    losses = []
    for matrix in matrices:
        domain_mean = matrix.mean(dim=0)
        losses.append((domain_mean - pooled_mean).pow(2).mean())
        if order >= 2:
            domain_var = (
                matrix.var(dim=0, unbiased=False)
                if matrix.size(0) > 1
                else torch.zeros_like(pooled_var)
            )
            losses.append((domain_var - pooled_var).pow(2).mean())
    return torch.stack(losses).mean()


def inter_domain_variance_channel_scores(
    features: torch.Tensor,
    domain_ids: Sequence[Any] | torch.Tensor,
) -> torch.Tensor:
    """Score feature channels by source-domain mean variance.

    This is the Identify step in the HydroDA adaptation of IU: channels whose
    source-region means vary strongly are treated as domain-specific candidates.
    """
    if features.ndim == 2:
        pooled = features.float()
    elif features.ndim == 4:
        pooled = features.mean(dim=(2, 3)).float()
    elif features.ndim > 2:
        pooled = features.reshape(features.size(0), features.size(1), -1).mean(dim=2).float()
    else:
        raise ValueError(f"features must have at least 2 dimensions, got shape={tuple(features.shape)}")
    groups = _domain_index_groups(domain_ids, device=features.device)
    if len(groups) < 2:
        return pooled.new_zeros(pooled.size(1))
    domain_means = []
    for indices in groups.values():
        domain_means.append(pooled.index_select(0, indices).mean(dim=0))
    return torch.stack(domain_means, dim=0).var(dim=0, unbiased=False)


def region_inter_domain_variance_channel_scores(
    features: torch.Tensor,
    region_mask_integer: torch.Tensor,
    *,
    active_region_ids: Optional[Sequence[Any]] = None,
) -> torch.Tensor:
    """Score channels by spatial source-region mean variance."""
    if features.ndim < 3:
        raise ValueError("region-based feature scoring requires spatial feature maps")
    region_numbers = _active_region_numbers(active_region_ids)
    if not region_numbers:
        unique = torch.unique(region_mask_integer.detach())
        region_numbers = sorted(int(v.item()) for v in unique if int(v.item()) > 0)
    domain_means = []
    for region_num in region_numbers:
        matrix = _region_feature_matrix(features, region_mask_integer, region_num)
        if matrix.numel() > 0:
            domain_means.append(matrix.mean(dim=0))
    if len(domain_means) < 2:
        return features.new_zeros(features.size(1))
    return torch.stack(domain_means, dim=0).var(dim=0, unbiased=False)


def identify_unlearn_loss(
    features: torch.Tensor,
    domain_ids: Sequence[Any] | torch.Tensor,
    *,
    top_fraction: float = 0.25,
    sample_top_fraction: float = 0.5,
    score_cap: float = DEFAULT_IU_SCORE_CAP,
) -> torch.Tensor:
    """Source-only bounded IU score for domain-specific feature reliance.

    The training loop adds this positive score as a penalty, suppressing
    high-energy samples/channels whose means vary across source regions.
    """
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")
    if not 0.0 < sample_top_fraction <= 1.0:
        raise ValueError(f"sample_top_fraction must be in (0, 1], got {sample_top_fraction}")
    groups = _domain_index_groups(domain_ids, device=features.device)
    if len(groups) < 2:
        return features.sum() * 0.0

    if features.ndim == 2:
        sample_channels = features.float()
    elif features.ndim == 4:
        sample_channels = features.mean(dim=(2, 3)).float()
    elif features.ndim > 2:
        sample_channels = features.reshape(features.size(0), features.size(1), -1).mean(dim=2).float()
    else:
        raise ValueError(f"features must have at least 2 dimensions, got shape={tuple(features.shape)}")

    scores = inter_domain_variance_channel_scores(features, domain_ids)
    if scores.numel() == 0 or torch.count_nonzero(scores).item() == 0:
        return features.sum() * 0.0
    k_channels = max(1, int(torch.ceil(torch.tensor(float(scores.numel()) * top_fraction)).item()))
    channel_indices = torch.topk(scores, k=k_channels).indices
    selected = sample_channels.index_select(1, channel_indices)

    raw_scores = sample_channels.pow(2).mean(dim=1)
    selected_scores = selected.pow(2).mean(dim=1)
    k_samples = max(1, int(torch.ceil(torch.tensor(float(selected_scores.numel()) * sample_top_fraction)).item()))
    sample_indices = torch.topk(selected_scores.detach(), k=k_samples).indices
    normalizer = raw_scores.mean().detach().clamp_min(1e-6)
    bounded_score = selected_scores.index_select(0, sample_indices) / normalizer
    return bounded_score.clamp(max=float(score_cap)).mean()


def region_identify_unlearn_loss(
    features: torch.Tensor,
    region_mask_integer: torch.Tensor,
    *,
    active_region_ids: Optional[Sequence[Any]] = None,
    top_fraction: float = 0.25,
    sample_top_fraction: float = 0.5,
    score_cap: float = DEFAULT_IU_SCORE_CAP,
) -> torch.Tensor:
    """Source-only bounded IU score from spatial source-region groups."""
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")
    if not 0.0 < sample_top_fraction <= 1.0:
        raise ValueError(f"sample_top_fraction must be in (0, 1], got {sample_top_fraction}")
    scores = region_inter_domain_variance_channel_scores(
        features,
        region_mask_integer,
        active_region_ids=active_region_ids,
    )
    if scores.numel() == 0 or torch.count_nonzero(scores).item() == 0:
        return features.sum() * 0.0
    k_channels = max(1, int(torch.ceil(torch.tensor(float(scores.numel()) * top_fraction)).item()))
    channel_indices = torch.topk(scores, k=k_channels).indices
    if features.ndim != 4:
        raise ValueError("region_identify_unlearn_loss requires [B, C, H, W] features")
    selected = features.index_select(1, channel_indices).float()
    raw_scores = features.float().pow(2).mean(dim=1).reshape(-1)
    selected_scores = selected.pow(2).mean(dim=1).reshape(-1)
    region_mask = region_mask_integer
    if region_mask.ndim == 2:
        region_mask = region_mask.unsqueeze(0)
    if region_mask.ndim == 4 and region_mask.size(1) == 1:
        region_mask = region_mask[:, 0]
    if region_mask.shape[-2:] != features.shape[-2:]:
        region_mask = torch.nn.functional.interpolate(
            region_mask.unsqueeze(1).float(),
            size=features.shape[-2:],
            mode="nearest",
        )[:, 0].to(dtype=torch.long)
    if region_mask.size(0) == 1 and features.size(0) > 1:
        region_mask = region_mask.expand(features.size(0), -1, -1)
    region_numbers = _active_region_numbers(active_region_ids)
    if not region_numbers:
        unique = torch.unique(region_mask.detach())
        region_numbers = sorted(int(v.item()) for v in unique if int(v.item()) > 0)
    active_mask = torch.zeros_like(region_mask, dtype=torch.bool, device=features.device)
    region_mask = region_mask.to(device=features.device)
    for region_num in region_numbers:
        active_mask = active_mask | (region_mask == region_num)
    flat_active_mask = active_mask.reshape(-1)
    active_scores = selected_scores[flat_active_mask]
    if active_scores.numel() == 0:
        return features.sum() * 0.0
    k_samples = max(1, int(torch.ceil(torch.tensor(float(active_scores.numel()) * sample_top_fraction)).item()))
    sample_indices = torch.topk(active_scores.detach(), k=k_samples).indices
    normalizer = raw_scores[flat_active_mask].mean().detach().clamp_min(1e-6)
    bounded_score = active_scores.index_select(0, sample_indices) / normalizer
    return bounded_score.clamp(max=float(score_cap)).mean()


class SAMSharpnessPerturbation:
    """Reversible SAM-style perturbation helper for source-only sharpness DG."""

    def __init__(self, model: nn.Module, *, rho: float = 0.05, eps: float = 1e-12) -> None:
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        self.model = model
        self.rho = float(rho)
        self.eps = float(eps)
        self._perturbations: Dict[nn.Parameter, torch.Tensor] = {}

    def grad_norm(self) -> torch.Tensor:
        grads = [
            p.grad.detach().norm(p=2)
            for p in self.model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        if not grads:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        return torch.norm(torch.stack(grads), p=2)

    @torch.no_grad()
    def perturb(self) -> torch.Tensor:
        self.restore()
        norm = self.grad_norm()
        if not torch.isfinite(norm) or norm.item() <= 0.0 or self.rho == 0.0:
            return norm
        scale = self.rho / (norm + self.eps)
        for param in self.model.parameters():
            if not param.requires_grad or param.grad is None:
                continue
            delta = param.grad * scale.to(param.device)
            param.add_(delta)
            self._perturbations[param] = delta.detach().clone()
        return norm

    @torch.no_grad()
    def restore(self) -> None:
        if not self._perturbations:
            return
        for param, delta in self._perturbations.items():
            param.sub_(delta.to(param.device))
        self._perturbations.clear()


class SelfBootstrapAugmentation(nn.Module):
    """Lightweight input deterioration for self-bootstrap consistency TTA."""

    def __init__(self, *, noise_std: float = 0.01, channel_dropout_p: float = 0.05) -> None:
        super().__init__()
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}")
        if not 0.0 <= channel_dropout_p < 1.0:
            raise ValueError(f"channel_dropout_p must be in [0, 1), got {channel_dropout_p}")
        self.noise_std = float(noise_std)
        self.channel_dropout_p = float(channel_dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        augmented = x
        if self.noise_std > 0.0:
            augmented = augmented + torch.randn_like(augmented) * self.noise_std
        if self.channel_dropout_p > 0.0:
            keep = (
                torch.rand(
                    (augmented.size(0), augmented.size(1), 1, 1),
                    device=augmented.device,
                    dtype=augmented.dtype,
                )
                >= self.channel_dropout_p
            ).to(augmented.dtype)
            augmented = augmented * keep
        return augmented


def _validate_input_only_target_context(
    sample_or_batch: Mapping[str, Any],
    *,
    split_type: str,
    method_name: str,
) -> None:
    if split_type != "target_context":
        if split_type == "target_eval":
            raise ValueError(f"{method_name} target_context loader must not read target_eval")
        raise ValueError(f"{method_name} requires split_type='target_context', got {split_type!r}")
    present = FORBIDDEN_TARGET_CONTEXT_KEYS.intersection(sample_or_batch.keys())
    if present:
        raise ValueError(f"{method_name} target_context batch contains label-bearing keys: {sorted(present)}")


def validate_deep_coral_target_context(sample_or_batch: Mapping[str, Any], *, split_type: str) -> None:
    """Validate that a retained Deep CORAL diagnostic target-context batch is input-only."""
    _validate_input_only_target_context(
        sample_or_batch,
        split_type=split_type,
        method_name="Deep CORAL",
    )


def validate_ssa_reg_target_context(sample_or_batch: Mapping[str, Any], *, split_type: str) -> None:
    """Validate that an SSA-Reg target-context batch is input-only."""
    _validate_input_only_target_context(
        sample_or_batch,
        split_type=split_type,
        method_name="SSA-Reg",
    )


def validate_tca_target_context(sample_or_batch: Mapping[str, Any], *, split_type: str) -> None:
    """Validate that a TCA target-context batch is input-only."""
    _validate_input_only_target_context(
        sample_or_batch,
        split_type=split_type,
        method_name="TCA",
    )


def validate_self_bootstrap_target_context(sample_or_batch: Mapping[str, Any], *, split_type: str) -> None:
    """Validate that a self-bootstrap target-context batch is input-only."""
    _validate_input_only_target_context(
        sample_or_batch,
        split_type=split_type,
        method_name="Self-Bootstrap",
    )


class InputOnlyTargetContextDataset(Dataset):
    """Dataset wrapper that exposes only input-side target_context fields."""

    def __init__(self, dataset: Any, *, method_name: str = "Target-context alignment") -> None:
        split_type = str(getattr(dataset, "split_type", ""))
        self.method_name = method_name
        _validate_input_only_target_context({}, split_type=split_type, method_name=method_name)
        if not hasattr(dataset, "get_input_side_sample"):
            raise TypeError("target_context dataset must implement get_input_side_sample(idx)")
        self.dataset = dataset
        self.split_type = split_type

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = dict(self.dataset.get_input_side_sample(idx))
        _validate_input_only_target_context(
            sample,
            split_type=self.split_type,
            method_name=self.method_name,
        )
        return sample


def collate_input_only_samples(batch: list[Mapping[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty target_context batch")
    for sample in batch:
        _validate_input_only_target_context(
            sample,
            split_type="target_context",
            method_name="Target-context alignment",
        )
    return {
        "x": torch.stack([torch.as_tensor(sample["x"], dtype=torch.float32) for sample in batch], dim=0),
        "date_str": [sample.get("date_str", "") for sample in batch],
        "month": torch.as_tensor([int(sample.get("month", 0)) for sample in batch], dtype=torch.long),
    }


def _projection_from_features(features: torch.Tensor, *, rank: int) -> torch.Tensor:
    matrix = _as_feature_matrix(features)
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    effective_rank = min(int(rank), matrix.size(1))
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    if centered.size(0) < 2:
        return torch.eye(matrix.size(1), device=matrix.device, dtype=matrix.dtype)
    cov = centered.t().matmul(centered) / float(centered.size(0) - 1)
    _, eigvecs = torch.linalg.eigh(cov)
    basis = eigvecs[:, -effective_rank:]
    return basis.matmul(basis.t())


def subspace_alignment_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    *,
    rank: int = 8,
) -> torch.Tensor:
    """Significant-subspace alignment loss for unlabeled regression TTA.

    The loss aligns source and target feature means plus the top feature-space
    eigenspace projection. It uses only input-side target-context features.
    """
    source = _as_feature_matrix(source_features)
    target = _as_feature_matrix(target_features)
    if source.size(1) != target.size(1):
        raise ValueError(
            f"Feature dimensions must match for SSA-Reg: source={source.size(1)}, target={target.size(1)}"
        )
    mean_loss = (source.mean(dim=0) - target.mean(dim=0)).pow(2).mean()
    source_projection = _projection_from_features(source, rank=rank)
    target_projection = _projection_from_features(target, rank=rank)
    projection_loss = (source_projection - target_projection).pow(2).mean()
    return mean_loss + projection_loss


class SSARegState:
    """Metadata/checkpoint helper for SSA-Reg target-context subspace alignment."""

    def __init__(
        self,
        *,
        rank: int = 8,
        lambda_align: float = 0.01,
        feature_layer: str = "bottleneck",
    ) -> None:
        self.rank = int(rank)
        self.lambda_align = float(lambda_align)
        self.feature_layer = str(feature_layer)

    def metadata(self) -> Dict[str, Any]:
        return {
            "ssa_reg_rank": self.rank,
            "ssa_reg_lambda": self.lambda_align,
            "ssa_reg_feature_layer": self.feature_layer,
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        model_state_dict: Optional[Mapping[str, torch.Tensor]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {"tag": "ssa_reg"}
        payload.update(self.metadata())
        if model_state_dict is not None:
            payload["model_state_dict"] = {k: v.detach().cpu().clone() for k, v in model_state_dict.items()}
        if metadata:
            payload.update(dict(metadata))
        torch.save(payload, path)
        return path


class SWADState:
    """Source-val-controlled dense stochastic weight averaging state."""

    def __init__(
        self,
        *,
        start_epoch: int = 10,
        tolerance: float = 0.02,
        patience: int = 3,
        mode: str = "min",
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("SWAD mode must be 'min' or 'max'")
        self.start_epoch = int(start_epoch)
        self.tolerance = float(tolerance)
        self.patience = int(patience)
        self.mode = mode
        self.best_metric: Optional[float] = None
        self.bad_epochs = 0
        self.n_averaged = 0
        self.window: list[Dict[str, Any]] = []
        self._avg_state: Optional[Dict[str, torch.Tensor]] = None

    def _is_improved(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        return metric < self.best_metric if self.mode == "min" else metric > self.best_metric

    def _within_tolerance(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        scale = max(abs(self.best_metric), 1e-12)
        if self.mode == "min":
            return metric <= self.best_metric + self.tolerance * scale
        return metric >= self.best_metric - self.tolerance * scale

    def update(self, *, epoch: int, source_val_metric: float, model: nn.Module) -> bool:
        """Maybe add model weights to the SWAD window."""
        metric = float(source_val_metric)
        if epoch < self.start_epoch:
            return False

        if self._is_improved(metric):
            self.best_metric = metric
            self.bad_epochs = 0
        elif self._within_tolerance(metric):
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs > self.patience:
                return False

        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if self._avg_state is None:
            self._avg_state = state
        else:
            for key, value in state.items():
                avg_value = self._avg_state[key]
                if torch.is_floating_point(avg_value):
                    avg_value.mul_(self.n_averaged / (self.n_averaged + 1.0)).add_(
                        value / (self.n_averaged + 1.0)
                    )
                else:
                    self._avg_state[key] = value
        self.n_averaged += 1
        self.window.append({"epoch": int(epoch), "source_val_metric": metric})
        return True

    def averaged_state_dict(self) -> Dict[str, torch.Tensor]:
        if self._avg_state is None:
            raise RuntimeError("SWADState has no averaged weights")
        return {k: v.clone() for k, v in self._avg_state.items()}

    def metadata(self) -> Dict[str, Any]:
        return {
            "swad_start_epoch": self.start_epoch,
            "swad_tolerance": self.tolerance,
            "swad_patience": self.patience,
            "swad_mode": self.mode,
            "swad_n_averaged": self.n_averaged,
            "swad_window": list(self.window),
            "swad_best_metric": self.best_metric,
        }

    def save_checkpoint(self, path: str | Path, metadata: Optional[Mapping[str, Any]] = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "tag": "swad",
            "model_state_dict": self.averaged_state_dict(),
        }
        payload.update(self.metadata())
        if metadata:
            payload.update(dict(metadata))
        torch.save(payload, path)
        return path
