from __future__ import annotations

import torch

from hydroda.models.hyper_adapters import BasisHyperAdapter
from hydroda.models.hyper_conditional_unet import HyperAdapterConditionalResUNet
from hydroda.models.target_adaptation import (
    AdapterCoefficientResidual,
    DARegimeGainMixer,
    BoundedDARegimeGainMixer,
    HydroMSRGainOutputAdapter,
    HydroMSRGainLiteOutputAdapter,
    HydroMSROutputAdapter,
    HydroMSRROSEOutputAdapter,
    MonthlyResidualGain,
    RobustObservationSpaceEncoder,
    TargetLatentPrompt,
    TargetSpatialResidualHead,
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


def test_target_spatial_residual_head_is_zero_at_initialization():
    head = TargetSpatialResidualHead(
        input_channels=12,
        out_channels=2,
        hidden_channels=4,
        refine_rootzone=False,
    )
    x = torch.randn(2, 12, 8, 8)
    y = torch.randn(2, 2, 8, 8)

    residual = head(x, y)

    assert residual.shape == y.shape
    assert torch.allclose(residual, torch.zeros_like(y))


def test_target_spatial_residual_head_can_protect_rootzone_channel():
    head = TargetSpatialResidualHead(
        input_channels=12,
        out_channels=2,
        hidden_channels=4,
        refine_rootzone=False,
    )
    head.final.bias.data.fill_(1.0)
    x = torch.randn(1, 12, 4, 4)
    y = torch.zeros(1, 2, 4, 4)

    residual = head(x, y)

    assert torch.all(residual[:, 0] != 0)
    assert torch.allclose(residual[:, 1], torch.zeros_like(residual[:, 1]))


def test_hydro_msr_output_adapter_is_zero_at_initialization():
    adapter = HydroMSROutputAdapter(
        input_channels=12,
        out_channels=2,
        hidden_channels=4,
        refine_rootzone=True,
    )
    x = torch.randn(2, 12, 8, 8)
    y = torch.randn(2, 2, 8, 8)

    residual = adapter(x, y)

    assert residual.shape == y.shape
    assert torch.allclose(residual, torch.zeros_like(y))


def test_hydro_msr_output_adapter_can_protect_rootzone_channel():
    adapter = HydroMSROutputAdapter(
        input_channels=12,
        out_channels=2,
        hidden_channels=4,
        refine_rootzone=False,
    )
    adapter.surface_head.bias.data.fill_(1.0)
    adapter.rootzone_head.bias.data.fill_(1.0)
    adapter.surface_to_rootzone_scale.data.fill_(1.0)
    x = torch.randn(1, 12, 4, 4)
    y = torch.zeros(1, 2, 4, 4)

    residual = adapter(x, y)

    assert torch.all(residual[:, 0] != 0)
    assert torch.allclose(residual[:, 1], torch.zeros_like(residual[:, 1]))


def test_hydro_msr_surface_to_rootzone_coupling_is_identity_at_initialization():
    adapter = HydroMSROutputAdapter(
        input_channels=12,
        out_channels=2,
        hidden_channels=4,
        refine_rootzone=True,
    )

    assert torch.equal(adapter.surface_to_rootzone_scale.detach(), torch.zeros_like(adapter.surface_to_rootzone_scale))


def test_hydro_msr_da_film_is_identity_at_initialization():
    torch.manual_seed(7)
    base = HydroMSROutputAdapter(
        input_channels=12,
        out_channels=2,
        hidden_channels=4,
        refine_rootzone=True,
        enable_da_film=False,
    )
    film = HydroMSROutputAdapter(
        input_channels=12,
        out_channels=2,
        hidden_channels=4,
        refine_rootzone=True,
        enable_da_film=True,
    )
    film.load_state_dict(base.state_dict(), strict=False)
    base.surface_head.bias.data.fill_(0.5)
    base.rootzone_head.bias.data.fill_(0.25)
    film.surface_head.bias.data.copy_(base.surface_head.bias.data)
    film.rootzone_head.bias.data.copy_(base.rootzone_head.bias.data)
    x = torch.randn(2, 12, 8, 8)
    y = torch.randn(2, 2, 8, 8)
    x_raw = torch.randn(2, 12, 8, 8)

    without_film = base(x, y, x_raw=x_raw)
    with_film = film(x, y, x_raw=x_raw)

    assert torch.allclose(with_film, without_film)


def test_da_regime_gain_mixer_alpha_is_identity_and_bounded_at_initialization():
    mixer = DARegimeGainMixer(out_channels=2, hidden_channels=4)
    candidate = torch.randn(2, 2, 5, 6)
    y = torch.randn(2, 2, 5, 6)
    x_raw = torch.randn(2, 12, 5, 6)
    x_raw[:, 7:9] = x_raw[:, 7:9].abs() + 0.1
    months = torch.tensor([1, 12])

    alpha = mixer(candidate, y, x_raw, months)

    assert alpha.shape == candidate.shape
    assert torch.all(alpha >= 0.0)
    assert torch.all(alpha <= 2.0)
    assert torch.allclose(alpha, torch.ones_like(alpha))


def test_da_regime_gain_mixer_month_encoding_shape():
    mixer = DARegimeGainMixer(out_channels=2, hidden_channels=4)
    months = torch.tensor([1, 7, 12])

    encoding = mixer.month_encoding(
        months,
        spatial_shape=(3, 5),
        device=months.device,
        dtype=torch.float32,
    )

    assert encoding.shape == (3, 2, 3, 5)
    assert torch.allclose(encoding[:, :, 0, 0], encoding[:, :, -1, -1])


def test_hydro_msr_gain_adapter_preserves_hydro_msr_candidate_at_initialization():
    adapter = HydroMSRGainOutputAdapter(
        input_channels=12,
        out_channels=2,
        hidden_channels=4,
        refine_rootzone=True,
    )
    adapter.candidate.surface_head.bias.data.fill_(0.5)
    adapter.candidate.rootzone_head.bias.data.fill_(0.25)
    x = torch.randn(2, 12, 8, 8)
    y = torch.randn(2, 2, 8, 8)
    x_raw = torch.randn(2, 12, 8, 8)
    x_raw[:, 7:9] = x_raw[:, 7:9].abs() + 0.1
    months = torch.tensor([2, 8])

    residual = adapter(x, y, x_raw=x_raw, month=months)
    candidate = adapter.candidate(x, y, x_raw=x_raw)

    assert torch.allclose(residual, candidate)


def test_hydro_msr_gain_adapter_does_not_hard_gate_base_valid_mask_zero():
    adapter = HydroMSRGainOutputAdapter(
        input_channels=12,
        out_channels=2,
        hidden_channels=4,
        refine_rootzone=True,
    )
    adapter.candidate.surface_head.bias.data.fill_(0.5)
    adapter.candidate.rootzone_head.bias.data.fill_(0.25)
    x = torch.randn(1, 12, 4, 4)
    y = torch.zeros(1, 2, 4, 4)
    x_raw = torch.zeros(1, 12, 4, 4)
    x_raw[:, 7:9] = 0.5
    x_raw[:, 11:12] = 0.0

    residual = adapter(x, y, x_raw=x_raw, month=torch.tensor([1]))

    assert torch.all(residual[:, 0] != 0)
    assert torch.all(residual[:, 1] != 0)


def test_bounded_da_regime_gain_mixer_alpha_is_identity_and_span_bounded():
    mixer = BoundedDARegimeGainMixer(out_channels=2, hidden_channels=4, gain_span=0.25)
    candidate = torch.randn(2, 2, 5, 6)
    y = torch.randn(2, 2, 5, 6)
    x_raw = torch.randn(2, 12, 5, 6)
    x_raw[:, 7:9] = x_raw[:, 7:9].abs() + 0.1
    months = torch.tensor([1, 12])

    alpha = mixer(candidate, y, x_raw, months)

    assert alpha.shape == candidate.shape
    assert torch.all(alpha >= 0.75)
    assert torch.all(alpha <= 1.25)
    assert torch.allclose(alpha, torch.ones_like(alpha))


def test_hydro_msr_gain_lite_adapter_preserves_hydro_msr_candidate_at_initialization():
    adapter = HydroMSRGainLiteOutputAdapter(
        input_channels=12,
        out_channels=2,
        hidden_channels=4,
        refine_rootzone=True,
        gain_span=0.25,
    )
    adapter.candidate.surface_head.bias.data.fill_(0.5)
    adapter.candidate.rootzone_head.bias.data.fill_(0.25)
    x = torch.randn(2, 12, 8, 8)
    y = torch.randn(2, 2, 8, 8)
    x_raw = torch.randn(2, 12, 8, 8)
    x_raw[:, 7:9] = x_raw[:, 7:9].abs() + 0.1
    months = torch.tensor([2, 8])

    residual = adapter(x, y, x_raw=x_raw, month=months)
    candidate = adapter.candidate(x, y, x_raw=x_raw)
    alpha = adapter.gain_mixer(candidate, y, x_raw, months)

    assert torch.allclose(residual, candidate)
    assert torch.allclose(alpha, torch.ones_like(alpha))
    assert torch.all(alpha >= 0.75)
    assert torch.all(alpha <= 1.25)


def test_hydro_msr_gain_lite_learns_surface_gain_only_by_default():
    adapter = HydroMSRGainLiteOutputAdapter(
        input_channels=12,
        out_channels=2,
        hidden_channels=4,
        refine_rootzone=True,
        gain_span=0.25,
    )
    adapter.candidate.surface_head.bias.data.fill_(0.5)
    adapter.candidate.rootzone_head.bias.data.fill_(0.25)
    final = adapter.gain_mixer.net[-1]
    final.bias.data[0] = 10.0
    final.bias.data[1] = -10.0
    x = torch.randn(1, 12, 4, 4)
    y = torch.zeros(1, 2, 4, 4)
    x_raw = torch.zeros(1, 12, 4, 4)
    x_raw[:, 7:9] = 0.5
    months = torch.tensor([1])

    residual = adapter(x, y, x_raw=x_raw, month=months)
    candidate = adapter.candidate(x, y, x_raw=x_raw)

    assert torch.all(residual[:, 0] > candidate[:, 0])
    assert torch.allclose(residual[:, 1], candidate[:, 1])


def test_hydro_msr_gain_lite_does_not_hard_gate_base_valid_mask_zero():
    adapter = HydroMSRGainLiteOutputAdapter(
        input_channels=12,
        out_channels=2,
        hidden_channels=4,
        refine_rootzone=True,
        gain_span=0.25,
    )
    adapter.candidate.surface_head.bias.data.fill_(0.5)
    adapter.candidate.rootzone_head.bias.data.fill_(0.25)
    x = torch.randn(1, 12, 4, 4)
    y = torch.zeros(1, 2, 4, 4)
    x_raw = torch.zeros(1, 12, 4, 4)
    x_raw[:, 7:9] = 0.5
    x_raw[:, 11:12] = 0.0

    residual = adapter(x, y, x_raw=x_raw, month=torch.tensor([1]))

    assert torch.all(residual[:, 0] != 0)
    assert torch.all(residual[:, 1] != 0)


def test_robust_observation_space_encoder_outputs_bounded_finite_features():
    encoder = RobustObservationSpaceEncoder()
    x_raw = torch.zeros(2, 12, 5, 6)
    x_raw[:, 0:1] = torch.linspace(0.05, 0.45, 30).view(1, 1, 5, 6)
    x_raw[:, 1:2] = torch.linspace(0.10, 0.50, 30).view(1, 1, 5, 6)
    x_raw[:, 2:3] = 295.0
    x_raw[:, 3:4] = 305.0
    x_raw[:, 4:5] = 1.5
    x_raw[:, 5:6] = 330.0
    x_raw[:, 6:7] = 210.0
    x_raw[:, 7:8] = 0.001
    x_raw[:, 8:9] = 0.001
    x_raw[:, 9:10] = 220.0
    x_raw[:, 10:11] = 320.0
    x_raw[:, 11:12] = 1.0
    x_raw[0, 5, 0, 0] = float("nan")

    features = encoder(x_raw)

    assert features.shape == (2, encoder.feature_channels, 5, 6)
    assert torch.isfinite(features).all()
    assert torch.all(features >= -1.0)
    assert torch.all(features <= 1.0)


def test_hydro_msr_rose_adapter_is_zero_at_initialization():
    adapter = HydroMSRROSEOutputAdapter(
        input_channels=12,
        out_channels=2,
        hidden_channels=4,
        refine_rootzone=True,
    )
    x = torch.randn(2, 12, 8, 8)
    y = torch.randn(2, 2, 8, 8)
    x_raw = torch.randn(2, 12, 8, 8)
    x_raw[:, 7:9] = x_raw[:, 7:9].abs() + 0.1

    residual = adapter(x, y, x_raw=x_raw)

    assert residual.shape == y.shape
    assert torch.allclose(residual, torch.zeros_like(y))


def test_hydro_msr_rose_adapter_does_not_hard_gate_base_valid_mask_zero():
    adapter = HydroMSRROSEOutputAdapter(
        input_channels=12,
        out_channels=2,
        hidden_channels=4,
        refine_rootzone=True,
    )
    adapter.surface_head.bias.data.fill_(0.5)
    adapter.rootzone_head.bias.data.fill_(0.25)
    x = torch.randn(1, 12, 4, 4)
    y = torch.zeros(1, 2, 4, 4)
    x_raw = torch.zeros(1, 12, 4, 4)
    x_raw[:, 7:9] = 0.5
    x_raw[:, 11:12] = 0.0

    residual = adapter(x, y, x_raw=x_raw)

    assert torch.all(residual[:, 0] != 0)
    assert torch.all(residual[:, 1] != 0)


def test_pigo_modules_are_not_part_of_target_adaptation_api():
    import hydroda.models.target_adaptation as target_adaptation

    assert not hasattr(target_adaptation, "PhysicsInformedInnovationGate")
    assert not hasattr(target_adaptation, "PIGOResidualOperator")


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
        enable_target_spatial_refine=True,
        target_spatial_refine_hidden=4,
    )

    model.freeze_source_prior_for_target_adaptation()

    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert any(name.startswith("target_spatial_refine") for name in trainable)
    assert all(
        name.startswith("target_")
        or name.startswith("residual_gain")
        or "coefficient_residual" in name
        for name in trainable
    )


def test_hyperda_can_use_hydro_msr_target_spatial_refine():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        enable_target_adaptation=True,
        target_latent_dim=4,
        enable_target_spatial_refine=True,
        target_spatial_refine_hidden=4,
        target_spatial_refine_rootzone=True,
        target_spatial_refine_type="hydro_msr",
        enable_hydro_msr_da_film=True,
    )
    x = torch.randn(2, 12, 16, 16)
    z = torch.randn(2, 8)
    month = torch.tensor([1, 12])

    y = model(x, z, month=month, x_raw=x)

    assert isinstance(model.target_spatial_refine, HydroMSROutputAdapter)
    assert model.target_spatial_refine_type == "hydro_msr"
    assert y.shape == (2, 2, 16, 16)


def test_hyperda_can_use_hydro_msr_gain_target_spatial_refine():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        enable_target_adaptation=True,
        target_latent_dim=4,
        enable_target_spatial_refine=True,
        target_spatial_refine_hidden=4,
        target_spatial_refine_rootzone=True,
        target_spatial_refine_type="hydro_msr_gain",
    )
    x = torch.randn(2, 12, 16, 16)
    z = torch.randn(2, 8)
    month = torch.tensor([1, 12])

    y = model(x, z, month=month, x_raw=x)

    assert isinstance(model.target_spatial_refine, HydroMSRGainOutputAdapter)
    assert model.target_spatial_refine_type == "hydro_msr_gain"
    assert y.shape == (2, 2, 16, 16)


def test_hyperda_can_use_hydro_msr_gain_lite_target_spatial_refine():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        enable_target_adaptation=True,
        target_latent_dim=4,
        enable_target_spatial_refine=True,
        target_spatial_refine_hidden=4,
        target_spatial_refine_rootzone=True,
        target_spatial_refine_type="hydro_msr_gain_lite",
        target_spatial_refine_gain_span=0.25,
    )
    x = torch.randn(2, 12, 16, 16)
    z = torch.randn(2, 8)
    month = torch.tensor([1, 12])

    y = model(x, z, month=month, x_raw=x)

    assert isinstance(model.target_spatial_refine, HydroMSRGainLiteOutputAdapter)
    assert model.target_spatial_refine_type == "hydro_msr_gain_lite"
    assert model.target_spatial_refine_gain_span == 0.25
    assert y.shape == (2, 2, 16, 16)


def test_hyperda_can_use_hydro_msr_rose_target_spatial_refine():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        enable_target_adaptation=True,
        target_latent_dim=4,
        enable_target_spatial_refine=True,
        target_spatial_refine_hidden=4,
        target_spatial_refine_rootzone=True,
        target_spatial_refine_type="hydro_msr_rose",
    )
    x = torch.randn(2, 12, 16, 16)
    x[:, 7:9] = x[:, 7:9].abs() + 0.1
    z = torch.randn(2, 8)
    month = torch.tensor([1, 12])

    y = model(x, z, month=month, x_raw=x)

    assert isinstance(model.target_spatial_refine, HydroMSRROSEOutputAdapter)
    assert model.target_spatial_refine_type == "hydro_msr_rose"
    assert y.shape == (2, 2, 16, 16)


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
        enable_target_spatial_refine=True,
        target_spatial_refine_hidden=4,
    )
    x = torch.randn(2, 12, 16, 16)
    z = torch.randn(2, 8)
    month = torch.tensor([1, 12])

    y = model(x, z, month=month)

    assert y.shape == (2, 2, 16, 16)


def test_hyperda_target_spatial_refine_can_use_raw_input():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        enable_target_adaptation=True,
        target_latent_dim=4,
        enable_target_spatial_refine=True,
        target_spatial_refine_hidden=4,
        target_spatial_refine_input="raw",
    )
    seen = {}

    def capture_forward(x, y, x_raw=None):
        seen["x"] = x.detach().clone()
        seen["x_raw"] = x_raw.detach().clone() if x_raw is not None else None
        return torch.zeros_like(y)

    model.target_spatial_refine.forward = capture_forward
    x_norm = torch.zeros(1, 12, 16, 16)
    x_raw = torch.ones(1, 12, 16, 16)
    z = torch.zeros(1, 8)
    month = torch.tensor([1])

    model(x_norm, z, month=month, x_raw=x_raw)

    assert torch.equal(seen["x"], x_raw)
    assert torch.equal(seen["x_raw"], x_raw)
