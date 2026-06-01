from __future__ import annotations

import torch

from hydroda.models.hyper_adapters import BasisHyperAdapter
from hydroda.models.hyper_conditional_unet import HyperAdapterConditionalResUNet
from hydroda.models.target_adaptation import (
    AdapterCoefficientResidual,
    MonthlyResidualGain,
    TargetLatentPrompt,
)


def test_target_latent_prompt_adds_trainable_shift():
    module = TargetLatentPrompt(prompt_dim=4, latent_dim=2)
    z = torch.zeros(3, 4)

    out = module(z)

    assert out.shape == z.shape
    assert module.latent.requires_grad
    assert module.proj.weight.requires_grad


def test_monthly_residual_gain_is_identity_at_initialization():
    gain = MonthlyResidualGain(out_channels=2, n_months=12)
    y = torch.randn(3, 2, 4, 5)
    months = torch.tensor([1, 6, 12])

    out = gain(y, months)

    assert torch.allclose(out, y)


def test_adapter_coefficient_residual_changes_coefficients():
    adapter = BasisHyperAdapter(channels=4, prompt_dim=6, n_basis=3, adapter_bottleneck=2)
    residual = AdapterCoefficientResidual(n_basis=3)
    z = torch.randn(2, 6)

    base = adapter.coefficients(z)
    residual.logit_delta.data[0] = 2.0
    shifted = adapter.coefficients(z, logit_residual=residual())

    assert shifted.shape == base.shape
    assert not torch.allclose(shifted, base)


def test_hyperda_freeze_source_prior_leaves_only_target_adaptation_trainable():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        enable_target_adaptation=True,
        target_latent_dim=4,
    )

    model.freeze_source_prior_for_target_adaptation()

    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all(
        name.startswith("target_")
        or name.startswith("residual_gain")
        or "coefficient_residual" in name
        for name in trainable
    )


def test_hyperda_target_adaptation_forward_accepts_month():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        enable_target_adaptation=True,
        target_latent_dim=4,
    )
    x = torch.randn(2, 12, 16, 16)
    z = torch.randn(2, 8)
    month = torch.tensor([1, 12])

    y = model(x, z, month=month)

    assert y.shape == (2, 2, 16, 16)
