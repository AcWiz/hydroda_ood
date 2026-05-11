"""Test zero_raw_increment_init: forecast-equivalent initialization semantics.

Coverage:
1. pred_increment_raw ≈ 0 at initialization (with mean/std normalization)
2. bias_norm = -inc_mean / inc_std ensures pred_inc_raw = 0 at init

No-leakage declaration:
    - Unit-test only, no data loading
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hydroda.models.resunet import SmallResUNet


class DummyTrainer:
    """Minimal stand-in for Trainer to test zero_raw_increment_init logic."""

    def __init__(self, model, inc_mean, inc_std, zero_raw_increment_init):
        self.model = model
        self._inc_mean = inc_mean
        self._inc_std = inc_std
        self.zero_raw_increment_init = zero_raw_increment_init

        if self.zero_raw_increment_init:
            if self._inc_mean is not None and self._inc_std is not None:
                bias_surface = -self._inc_mean[0] / self._inc_std[0]
                bias_rootzone = -self._inc_mean[1] / self._inc_std[1]
                with torch.no_grad():
                    self.model.head.bias[0] = torch.tensor(bias_surface, device=self.model.head.bias.device)
                    self.model.head.bias[1] = torch.tensor(bias_rootzone, device=self.model.head.bias.device)


def test_zero_raw_bias_formula():
    """Verify bias = -inc_mean / inc_std gives pred_inc_raw ≈ 0."""
    inc_mean = np.array([0.05, -0.03], dtype=np.float32)
    inc_std = np.array([0.15, 0.10], dtype=np.float32)

    model = SmallResUNet(in_channels=12, out_channels=2, width=8, zero_raw_increment_init=False)
    # Manually set weights to zero
    nn = torch.nn
    nn.init.zeros_(model.head.weight)
    nn.init.zeros_(model.head.bias)

    # Apply zero_raw_increment_init logic
    trainer = DummyTrainer(model, inc_mean, inc_std, zero_raw_increment_init=True)

    # With inc_norm=True, model outputs pred_inc_norm = bias (since weights=0)
    # pred_inc_raw = bias * inc_std + inc_mean
    bias_s = trainer.model.head.bias[0].item()
    bias_r = trainer.model.head.bias[1].item()

    expected_bias_s = -inc_mean[0] / inc_std[0]  # -0.05 / 0.15 ≈ -0.333
    expected_bias_r = -inc_mean[1] / inc_std[1]  # -(-0.03) / 0.10 = 0.3

    assert abs(bias_s - expected_bias_s) < 1e-6, f"surface bias: {bias_s} vs expected {expected_bias_s}"
    assert abs(bias_r - expected_bias_r) < 1e-6, f"rootzone bias: {bias_r} vs expected {expected_bias_r}"

    # Denormalized prediction should be ≈ 0
    # (Simulating what predictor does after model forward)
    pred_inc_norm_s = bias_s
    pred_inc_raw_s = pred_inc_norm_s * inc_std[0] + inc_mean[0]
    pred_inc_norm_r = bias_r
    pred_inc_raw_r = pred_inc_norm_r * inc_std[1] + inc_mean[1]

    assert abs(pred_inc_raw_s) < 1e-6, f"surface pred_inc_raw: {pred_inc_raw_s}"
    assert abs(pred_inc_raw_r) < 1e-6, f"rootzone pred_inc_raw: {pred_inc_raw_r}"


def test_without_inc_normalization_zero_bias():
    """When zero_raw_increment_init=True but no inc normalization, bias stays at 0."""
    model = SmallResUNet(in_channels=12, out_channels=2, width=8, zero_raw_increment_init=True)

    trainer = DummyTrainer(model, inc_mean=None, inc_std=None, zero_raw_increment_init=True)

    # Without inc_norm, no bias adjustment applied
    assert trainer.model.head.bias[0].item() == 0.0
    assert trainer.model.head.bias[1].item() == 0.0


def test_pred_inc_norm_zero_not_forecast_equivalent():
    """Demonstrate that pred_inc_norm=0 → pred_inc_raw=inc_mean ≠ forecast-only."""
    inc_mean = np.array([0.05, -0.03], dtype=np.float32)
    inc_std = np.array([0.15, 0.10], dtype=np.float32)

    # If we had simple zero-init (bias=0), pred_inc_norm=0 → pred_inc_raw = inc_mean
    wrong_bias = 0.0
    wrong_pred_raw_s = wrong_bias * inc_std[0] + inc_mean[0]  # = 0.05
    wrong_pred_raw_r = wrong_bias * inc_std[1] + inc_mean[1]  # = -0.03

    assert abs(wrong_pred_raw_s - inc_mean[0]) < 1e-6
    assert abs(wrong_pred_raw_r - inc_mean[1]) < 1e-6

    # This is NOT forecast-only: forecast-equivalent requires pred_inc_raw = 0
    # So bias must be -inc_mean/inc_std, not 0
    correct_bias_s = -inc_mean[0] / inc_std[0]
    correct_bias_r = -inc_mean[1] / inc_std[1]

    correct_pred_raw_s = correct_bias_s * inc_std[0] + inc_mean[0]
    correct_pred_raw_r = correct_bias_r * inc_std[1] + inc_mean[1]

    assert abs(correct_pred_raw_s) < 1e-6
    assert abs(correct_pred_raw_r) < 1e-6


def test_zero_raw_init_with_real_forward():
    """Smoke test: model with zero_raw_increment_init outputs near-zero increment."""
    torch.manual_seed(0)
    inc_mean = np.array([0.05, -0.03], dtype=np.float32)
    inc_std = np.array([0.15, 0.10], dtype=np.float32)

    model = SmallResUNet(in_channels=12, out_channels=2, width=8, zero_raw_increment_init=False)
    nn = torch.nn
    nn.init.zeros_(model.head.weight)

    # Apply zero_raw_increment_init logic
    trainer = DummyTrainer(model, inc_mean, inc_std, zero_raw_increment_init=True)

    model.eval()
    x = torch.randn(2, 12, 32, 32)

    with torch.no_grad():
        pred_norm = model(x)  # [B, 2, H, W]

    # Bias should be set such that denormalized output ≈ 0
    pred_inc_raw_s = pred_norm[0, 0].mean().item() * inc_std[0] + inc_mean[0]
    pred_inc_raw_r = pred_norm[0, 1].mean().item() * inc_std[1] + inc_mean[1]

    # Should be near zero (bias cancels inc_mean)
    assert abs(pred_inc_raw_s) < 0.1, f"surface pred_inc_raw should be near 0: {pred_inc_raw_s}"
    assert abs(pred_inc_raw_r) < 0.1, f"rootzone pred_inc_raw should be near 0: {pred_inc_raw_r}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))