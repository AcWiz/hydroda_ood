"""Loss functions for DA increment prediction.

No-leakage declaration:
    - All losses operate only on loss_mask pixels from source_train split
    - No target_query labels used in loss computation
    - No normalization inside loss functions
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedHuberLoss(nn.Module):
    """Masked Huber loss for DA increment prediction.

    Args:
        delta: Huber delta threshold (default 0.01). Smooth L1 loss is used,
            which corresponds to Huber loss with delta=1.0 in smooth L1 formulation.
        surface_weight: weight for surface channel loss (default 1.0)
        rootzone_weight: weight for rootzone channel loss (default 1.0)
    """

    def __init__(
        self,
        delta: float = 0.01,
        surface_weight: float = 1.0,
        rootzone_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.delta = float(delta)
        self.surface_weight = float(surface_weight)
        self.rootzone_weight = float(rootzone_weight)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute masked Huber loss.

        Args:
            pred: predicted increments, shape [B, 2, H, W]
            target: true increments, shape [B, 2, H, W]
            mask: binary mask, shape [B, 1, H, W] or [B, H, W]

        Returns:
            dict with surface_loss, rootzone_loss, total_loss,
            valid_pixel_count (as float), valid_pixel_fraction
        """
        # Ensure mask has channel dimension
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)  # [B, 1, H, W]

        # Expand mask to match pred/target shape
        mask = mask.expand_as(pred)  # [B, 2, H, W]

        # Apply mask: set invalid pixels to 0 loss contribution
        # We compute loss everywhere then multiply by mask
        diff = pred - target  # [B, 2, H, W]

        # Smooth L1 (Huber) loss: for |x| < delta use 0.5*x^2, else delta*(|x|-0.5*delta)
        # PyTorch smooth_l1_loss uses beta=1.0 by default (Huber formulation)
        # We scale by delta to get delta-scaled Huber
        abs_diff = torch.abs(diff)
        loss_raw = torch.where(
            abs_diff < self.delta,
            0.5 * diff ** 2 / self.delta,
            abs_diff - 0.5 * self.delta,
        )  # [B, 2, H, W]

        # Apply mask
        loss_masked = loss_raw * mask

        # Sum over spatial dims, keep channel dim
        loss_per_channel = loss_masked.sum(dim=(2, 3))  # [B, 2]

        # Count valid pixels per channel
        valid_count_per_channel = mask.sum(dim=(2, 3))  # [B, 2]

        # Average over batch and valid pixels for each channel
        surface_loss = (
            (loss_per_channel[:, 0] / torch.clamp(valid_count_per_channel[:, 0], min=1.0))
            .mean()
        )
        rootzone_loss = (
            (loss_per_channel[:, 1] / torch.clamp(valid_count_per_channel[:, 1], min=1.0))
            .mean()
        )

        total_loss = self.surface_weight * surface_loss + self.rootzone_weight * rootzone_loss

        total_valid = valid_count_per_channel.sum()
        valid_fraction = total_valid / max(1.0, mask.numel())

        return {
            "surface_loss": surface_loss,
            "rootzone_loss": rootzone_loss,
            "total_loss": total_loss,
            "valid_pixel_count": total_valid.detach(),
            "valid_pixel_fraction": valid_fraction.detach(),
        }


class MaskedMSELoss(nn.Module):
    """Masked MSE loss as fallback if Huber diverges.

    Args:
        surface_weight: weight for surface channel loss (default 1.0)
        rootzone_weight: weight for rootzone channel loss (default 1.0)
    """

    def __init__(
        self,
        surface_weight: float = 1.0,
        rootzone_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.surface_weight = float(surface_weight)
        self.rootzone_weight = float(rootzone_weight)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute masked MSE loss.

        Args:
            pred: predicted increments, shape [B, 2, H, W]
            target: true increments, shape [B, 2, H, W]
            mask: binary mask, shape [B, 1, H, W] or [B, H, W]

        Returns:
            dict with surface_loss, rootzone_loss, total_loss,
            valid_pixel_count, valid_pixel_fraction
        """
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(pred)

        diff = pred - target
        sq_error = diff ** 2
        loss_masked = sq_error * mask

        loss_per_channel = loss_masked.sum(dim=(2, 3))
        valid_count_per_channel = mask.sum(dim=(2, 3))

        surface_loss = (
            (loss_per_channel[:, 0] / torch.clamp(valid_count_per_channel[:, 0], min=1.0))
            .mean()
        )
        rootzone_loss = (
            (loss_per_channel[:, 1] / torch.clamp(valid_count_per_channel[:, 1], min=1.0))
            .mean()
        )

        total_loss = self.surface_weight * surface_loss + self.rootzone_weight * rootzone_loss

        total_valid = valid_count_per_channel.sum()
        valid_fraction = total_valid / max(1.0, mask.numel())

        return {
            "surface_loss": surface_loss,
            "rootzone_loss": rootzone_loss,
            "total_loss": total_loss,
            "valid_pixel_count": total_valid.detach(),
            "valid_pixel_fraction": valid_fraction.detach(),
        }


def compute_da_increment_loss(
    pred_increment: torch.Tensor,
    true_increment: torch.Tensor,
    loss_mask: torch.Tensor,
    delta: float = 0.01,
    surface_weight: float = 1.0,
    rootzone_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Convenience function for computing DA increment loss.

    Args:
        pred_increment: predicted increments [B, 2, H, W]
        true_increment: true increments [B, 2, H, W]
        loss_mask: binary mask [B, 1, H, W] or [B, H, W]
        delta: Huber delta (default 0.01)
        surface_weight: surface channel weight (default 1.0)
        rootzone_weight: rootzone channel weight (default 1.0)

    Returns:
        dict with surface_loss, rootzone_loss, total_loss,
        valid_pixel_count, valid_pixel_fraction
    """
    loss_fn = MaskedHuberLoss(delta=delta, surface_weight=surface_weight, rootzone_weight=rootzone_weight)
    return loss_fn(pred_increment, true_increment, loss_mask)