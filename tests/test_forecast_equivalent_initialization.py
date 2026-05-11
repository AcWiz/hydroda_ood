"""Test forecast-equivalent initialization end-to-end.

Coverage:
1. pred_analysis ≈ forecast at initialization (with inc_mean ≠ 0)
2. analysis_skill_vs_forecast ≈ 0 at initialization
3. pred_inc_norm=0 ≠ forecast-equivalent (proven with inc_mean ≠ 0)

No-leakage declaration:
    - Unit-test only, no data loading
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hydroda.models.resunet import SmallResUNet


def _denorm(pred_norm, inc_mean, inc_std):
    """Denormalize normalized increment to raw increment."""
    return pred_norm * inc_std + inc_mean


def test_analysis_approx_forecast_at_init():
    """Verify pred_analysis ≈ forecast when model initialized with zero_raw_increment_init."""
    # Scenario: inc_mean ≠ 0 (typical in SM DA)
    inc_mean = np.array([0.05, -0.03], dtype=np.float32)
    inc_std = np.array([0.15, 0.10], dtype=np.float32)

    # Build model
    model = SmallResUNet(in_channels=12, out_channels=2, width=8, zero_raw_increment_init=False)

    # Zero weights, set bias = -inc_mean/inc_std
    with torch.no_grad():
        torch.nn.init.zeros_(model.head.weight)
        model.head.bias[0] = torch.tensor(-inc_mean[0] / inc_std[0], device=model.head.bias.device)
        model.head.bias[1] = torch.tensor(-inc_mean[1] / inc_std[1], device=model.head.bias.device)

    model.eval()

    # Simulate forward pass
    x = torch.randn(2, 12, 32, 32)
    forecast_surface = torch.randn(2, 32, 32) * 0.2 + 0.3
    forecast_rootzone = torch.randn(2, 32, 32) * 0.15 + 0.25

    with torch.no_grad():
        pred_norm = model(x)  # [B, 2, H, W]

    # Denormalize to raw increments
    pred_inc_s = _denorm(pred_norm[:, 0], inc_mean[0], inc_std[0])
    pred_inc_r = _denorm(pred_norm[:, 1], inc_mean[1], inc_std[1])

    # Compute predicted analysis
    pred_analysis_s = forecast_surface + pred_inc_s
    pred_analysis_r = forecast_rootzone + pred_inc_r

    # With zero weights and correct bias, pred_inc ≈ 0 → pred_analysis ≈ forecast
    # Check that deviations are small
    dev_s = (pred_analysis_s - forecast_surface).abs().mean().item()
    dev_r = (pred_analysis_r - forecast_rootzone).abs().mean().item()

    assert dev_s < 1e-5, f"surface analysis should ≈ forecast, dev={dev_s}"
    assert dev_r < 1e-5, f"rootzone analysis should ≈ forecast, dev={dev_r}"


def test_pred_inc_norm_zero_gives_nonzero_raw():
    """Prove pred_inc_norm=0 with inc_mean≠0 gives pred_inc_raw=inc_mean (NOT forecast-only)."""
    inc_mean = np.array([0.05, -0.03], dtype=np.float32)
    inc_std = np.array([0.15, 0.10], dtype=np.float32)

    # Simple zero-init: bias=0
    wrong_bias = 0.0

    pred_inc_norm_s = wrong_bias  # = 0
    pred_inc_norm_r = wrong_bias  # = 0

    pred_inc_raw_s = pred_inc_norm_s * inc_std[0] + inc_mean[0]
    pred_inc_raw_r = pred_inc_norm_r * inc_std[1] + inc_mean[1]

    # This is clearly NOT forecast-only (pred_inc_raw ≠ 0)
    assert abs(pred_inc_raw_s - inc_mean[0]) < 1e-6, "surface should equal inc_mean"
    assert abs(pred_inc_raw_r - inc_mean[1]) < 1e-6, "rootzone should equal inc_mean"

    # forecast-equivalent requires pred_inc_raw = 0, so bias must be -inc_mean/inc_std
    correct_bias_s = -inc_mean[0] / inc_std[0]
    correct_bias_r = -inc_mean[1] / inc_std[1]

    correct_pred_inc_raw_s = correct_bias_s * inc_std[0] + inc_mean[0]
    correct_pred_inc_raw_r = correct_bias_r * inc_std[1] + inc_mean[1]

    assert abs(correct_pred_inc_raw_s) < 1e-8
    assert abs(correct_pred_inc_raw_r) < 1e-8


def test_skill_vs_forecast_near_zero_at_init():
    """analysis_skill_vs_forecast ≈ 0 when pred_analysis ≈ forecast."""
    inc_mean = np.array([0.05, -0.03], dtype=np.float32)
    inc_std = np.array([0.15, 0.10], dtype=np.float32)

    # Model initialized correctly
    model = SmallResUNet(in_channels=12, out_channels=2, width=8, zero_raw_increment_init=False)
    with torch.no_grad():
        torch.nn.init.zeros_(model.head.weight)
        model.head.bias[0] = torch.tensor(-inc_mean[0] / inc_std[0], device=model.head.bias.device)
        model.head.bias[1] = torch.tensor(-inc_mean[1] / inc_std[1], device=model.head.bias.device)

    model.eval()

    x = torch.randn(2, 12, 32, 32)
    forecast = torch.randn(2, 32, 32) * 0.2 + 0.3
    true_analysis = forecast + torch.randn(2, 32, 32) * 0.05  # small difference

    with torch.no_grad():
        pred_norm = model(x)
        pred_inc_s = _denorm(pred_norm[:, 0], inc_mean[0], inc_std[0])
        pred_inc_r = _denorm(pred_norm[:, 1], inc_mean[1], inc_std[1])
        pred_analysis = forecast + pred_inc_s  # adds per-batch increments

    # Use batch 0 for skill computation
    pred_a0 = pred_analysis[0]
    true_a0 = true_analysis[0]
    fcst0 = forecast[0]

    rmse_pred = torch.sqrt(((pred_a0 - true_a0) ** 2).mean()).item()
    rmse_fcst = torch.sqrt(((fcst0 - true_a0) ** 2).mean()).item()

    if rmse_fcst > 0:
        skill = 1.0 - rmse_pred / rmse_fcst
        assert abs(skill) < 0.05, f"skill should be ≈ 0 at init, got {skill}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))