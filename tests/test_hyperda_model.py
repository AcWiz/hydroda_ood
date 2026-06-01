from __future__ import annotations

import torch

from hydroda.models.hyper_adapters import BasisHyperAdapter
from hydroda.models.hyper_conditional_unet import HyperAdapterConditionalResUNet


def test_basis_hyper_adapter_returns_coefficients_and_feature_shape():
    adapter = BasisHyperAdapter(
        channels=16,
        prompt_dim=8,
        n_basis=4,
        adapter_bottleneck=6,
        adapter_scale=0.5,
    )
    h = torch.randn(3, 16, 8, 12)
    z = torch.randn(3, 8)

    out = adapter(h, z)
    coeffs = adapter.coefficients(z)

    assert out.shape == h.shape
    assert coeffs.shape == (3, 4)
    assert torch.allclose(coeffs.sum(dim=1), torch.ones(3), atol=1e-6)


def test_hyper_adapter_conditional_resunet_output_shape_and_gradients():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
    )
    x = torch.randn(2, 12, 32, 48)
    z = torch.randn(2, 16)

    pred = model(x, z)
    loss = pred.square().mean()
    loss.backward()

    assert pred.shape == (2, 2, 32, 48)
    assert model.hyper_adapter.coeff_head.weight.grad is not None
    assert torch.isfinite(model.hyper_adapter.coeff_head.weight.grad).all()


def test_hyper_adapter_conditional_resunet_keeps_film_and_decoder_adapters():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
    )

    assert hasattr(model, "film1")
    assert hasattr(model, "film2")
    assert hasattr(model, "film3")
    assert hasattr(model, "film_b")
    assert hasattr(model, "hyper_adapter_b")
    assert hasattr(model, "hyper_adapter_d2")
    assert hasattr(model, "hyper_adapter_d1")

    up_abs_sum = 0.0
    for adapter in [model.hyper_adapter_b, model.hyper_adapter_d2, model.hyper_adapter_d1]:
        for basis in adapter.bases:
            up_abs_sum += float(basis.up.weight.detach().abs().sum())
    assert up_abs_sum > 0.0


def test_hyper_adapter_conditional_resunet_requires_prompt():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
    )
    x = torch.randn(1, 12, 32, 48)

    try:
        model(x, None)
    except ValueError as exc:
        assert "prompt" in str(exc).lower()
    else:
        raise AssertionError("HyperAdapterConditionalResUNet should require a prompt tensor")
