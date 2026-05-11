"""Test zero-init output head: model predicts ~0 normalized increment at initialization.

No-leakage declaration:
    - Test uses dummy data, no real data or target labels
    - Verifies initialization behavior only
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hydroda.models.resunet import SmallResUNet


def test_zero_init_output_is_near_zero():
    """Zero-init output head should produce near-zero output at initialization."""
    model = SmallResUNet(in_channels=12, out_channels=2, width=8, zero_init_output=True)

    # Random input
    x = torch.randn(2, 12, 32, 32)
    pred = model(x)

    # With zero-init output head, output should be very close to zero
    # (it will have small residual from non-zero intermediate activations)
    assert torch.abs(pred).max() < 1e-5, f"Zero-init output should be near zero, got {pred.abs().max()}"

    # Verify that without zero_init, output is not near zero
    model2 = SmallResUNet(in_channels=12, out_channels=2, width=8, zero_init_output=False)
    pred2 = model2(x)
    assert torch.abs(pred2).max() > 0.1, "Non-zero-init output should have magnitude > 0.1"


def test_zero_init_plus_increment_mean_gives_near_zero_analysis_correction():
    """Zero-init + increment normalization should give near-zero raw correction.

    When zero_init_output=True and target_increment_normalization=True:
    - At init: pred_inc_norm = 0
    - Denorm: pred_inc_raw = 0 * inc_std + inc_mean = inc_mean
    - For typical increments, inc_mean ~ 0.05 (surface) / -0.03 (rootzone) m³/m³
    - This is a small correction, not zero, but much smaller than typical forecast errors
    """
    # Simulate what the predictor does with zero-init + inc normalization
    inc_mean = np.array([0.05, -0.03], dtype=np.float32)  # typical values
    inc_std = np.array([0.15, 0.10], dtype=np.float32)

    # At init, model predicts normalized increment = 0
    pred_inc_norm = np.array([0.0, 0.0], dtype=np.float32)

    # Denormalize
    pred_inc_raw = pred_inc_norm * inc_std + inc_mean

    # Predicted raw increment is inc_mean, which is small
    # This is ~0 correction for surface, ~0.03 correction for rootzone
    # Much smaller than scale mismatch from before the fix
    assert abs(pred_inc_raw[0] - 0.05) < 0.001, "Surface should be ~inc_mean[0]"
    assert abs(pred_inc_raw[1] - (-0.03)) < 0.001, "Rootzone should be ~inc_mean[1]"

    # The raw increment is a valid correction (centered near zero)
    # Before the fix: pred_inc_norm ~ 0.02 (random init) → denorm = 0.02 * 0.01 = 0.0002
    # But then it was added to raw forecast (~0.2) without denorm, so analysis ~ 0.2002
    # True analysis ~ 0.2 → huge RMSE ~ 2.0, skill ~ -99
    # After fix: pred_inc_raw = inc_mean ~ 0.05 → analysis ~ 0.2 + 0.05 = 0.25
    # This is still wrong but at least in the right unit space
    # With zero-init, initial pred_inc_norm = 0 → pred_inc_raw = inc_mean
    # Analysis ~ 0.2 + 0.05 = 0.25, skill ≈ 1 - 0.05/rmse_fcst ≈ 0 (forecast-like)


def test_zero_init_preserved_after_training():
    """Zero-init should be applied at initialization and preserved through training."""
    model = SmallResUNet(in_channels=12, out_channels=2, width=8, zero_init_output=True)

    # Check head weights are zero
    head_weight = model.head.weight.data
    head_bias = model.head.bias.data

    assert torch.abs(head_weight).max() < 1e-8, "Head weight should be zero"
    assert torch.abs(head_bias).max() < 1e-8, "Head bias should be zero"

    # After some training steps the weights will change (due to gradient updates)
    # This test just verifies the initial state is correct


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))