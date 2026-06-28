from __future__ import annotations

import pytest
import torch

from hydroda.models.hyper_adapters import BasisHyperAdapter
from hydroda.models.hyper_conditional_unet import HyperAdapterConditionalResUNet
from hydroda.models.source_saliency import make_saliency_artifact, load_source_saliency_prior
from hydroda.models.prompt_encoder import RegionPromptEncoder, RobustInputSideDAPromptEncoder
from hydroda.models.resunet import SmallResUNet
from scripts.train.train_prompt_conditioned_shared import (
    apply_trainable_scope,
    load_source_base_checkpoint_into_hyperda,
    trainable_parameter_names,
    validate_source_base_checkpoint_for_staged_init,
)


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


def test_basis_hyper_adapter_dora_like_gain_initializes_as_existing_adapter():
    torch.manual_seed(7)
    baseline = BasisHyperAdapter(
        channels=8,
        prompt_dim=6,
        n_basis=3,
        adapter_bottleneck=4,
        adapter_scale=0.5,
    )
    torch.manual_seed(7)
    dora_like = BasisHyperAdapter(
        channels=8,
        prompt_dim=6,
        n_basis=3,
        adapter_bottleneck=4,
        adapter_scale=0.5,
        adapter_param_style="dora_like_gain",
    )
    h = torch.randn(2, 8, 5, 7)
    z = torch.randn(2, 6)

    assert dora_like.adapter_param_style == "dora_like_gain"
    assert dora_like.basis_gain_delta is not None
    assert torch.allclose(dora_like.basis_gain_delta, torch.zeros_like(dora_like.basis_gain_delta))
    assert torch.allclose(dora_like(h, z), baseline(h, z), atol=1e-7)


def test_basis_hyper_adapter_bounded_dora_gain_is_identity_centered_and_trainable():
    torch.manual_seed(11)
    baseline = BasisHyperAdapter(
        channels=8,
        prompt_dim=6,
        n_basis=3,
        adapter_bottleneck=4,
        adapter_scale=0.5,
    )
    torch.manual_seed(11)
    bounded = BasisHyperAdapter(
        channels=8,
        prompt_dim=6,
        n_basis=3,
        adapter_bottleneck=4,
        adapter_scale=0.5,
        adapter_param_style="dora_like_gain_bounded",
    )
    h = torch.randn(2, 8, 5, 7)
    z = torch.randn(2, 6)

    out = bounded(h, z)
    out.square().mean().backward()

    assert bounded.adapter_param_style == "dora_like_gain_bounded"
    assert bounded.basis_gain_delta is not None
    assert torch.allclose(bounded.basis_gain_delta, torch.zeros_like(bounded.basis_gain_delta))
    assert torch.allclose(out, baseline(h, z), atol=1e-7)
    assert bounded.basis_gain_delta.grad is not None
    assert torch.isfinite(bounded.basis_gain_delta.grad).all()


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
    assert model.hyper_coeff_generator == "per_adapter"
    assert model.hyper_reliability_gate == "none"
    assert model.hyper_enable_film is True
    assert model.hyper_enable_adapters is True
    assert model.hyper_adapter.coeff_head.weight.grad is not None
    assert torch.isfinite(model.hyper_adapter.coeff_head.weight.grad).all()


def test_rank_gated_shared_coeff_generator_masks_to_top_k_and_trains_gates():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=6,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated",
        hyper_rank_gate_top_k=3,
        hyper_adapter_param_style="dora_like_gain",
    )
    x = torch.randn(2, 12, 32, 48)
    z = torch.randn(2, 16)

    logits = model.adapter_coefficient_logits(z, "bottleneck")
    coeffs = model.hyper_adapter_b.coefficients(z, coeff_logits=logits)
    pred = model(x, z)
    pred.square().mean().backward()

    assert model.hyper_coeff_generator == "shared_layer_aware_rank_gated"
    assert logits.shape == (2, 6)
    assert coeffs.shape == (2, 6)
    assert torch.allclose(coeffs.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.all((coeffs > 0).sum(dim=1) <= 3)
    assert model.shared_coeff_generator is not None
    assert model.shared_coeff_generator.coeff_head.weight.grad is not None
    assert model.shared_coeff_generator.gate_head.weight.grad is not None
    assert torch.isfinite(model.shared_coeff_generator.coeff_head.weight.grad).all()
    assert model.hyper_adapter_b.basis_gain_delta is not None
    assert model.hyper_adapter_b.basis_gain_delta.grad is not None


def test_stable_rank_gated_shared_coeff_generator_uses_finite_floor_and_trains_bounded_gain():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=6,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=3,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
    )
    x = torch.randn(2, 12, 32, 48)
    z = torch.randn(2, 16)

    logits = model.adapter_coefficient_logits(z, "bottleneck")
    coeffs = model.hyper_adapter_b.coefficients(z, coeff_logits=logits)
    topk = torch.topk(logits, k=3, dim=-1).indices
    topk_mask = torch.zeros_like(coeffs, dtype=torch.bool)
    topk_mask.scatter_(1, topk, True)
    pred = model(x, z)
    pred.square().mean().backward()

    assert model.hyper_coeff_generator == "shared_layer_aware_rank_gated_stable"
    assert model.hyper_adapter_b.adapter_param_style == "dora_like_gain_bounded"
    assert logits.shape == (2, 6)
    assert torch.isfinite(logits).all()
    assert coeffs.shape == (2, 6)
    assert torch.allclose(coeffs.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.all(coeffs[~topk_mask] < 1e-10)
    assert model.shared_coeff_generator is not None
    assert torch.isfinite(model.shared_coeff_generator.coeff_head.weight.grad).all()
    assert torch.isfinite(model.shared_coeff_generator.gate_head.weight.grad).all()
    assert model.shared_coeff_generator.log_temperature.grad is not None
    assert torch.isfinite(model.shared_coeff_generator.log_temperature.grad).all()
    assert model.hyper_adapter_b.basis_gain_delta is not None
    assert model.hyper_adapter_b.basis_gain_delta.grad is not None
    assert torch.isfinite(model.hyper_adapter_b.basis_gain_delta.grad).all()


def test_source_saliency_prior_artifact_is_finite_and_rejects_target_roles(tmp_path):
    scores = torch.tensor(
        [
            [0.1, 0.2, 0.4],
            [0.0, 0.3, 0.6],
            [1.0, 0.5, 0.25],
        ],
        dtype=torch.float32,
    )
    artifact = make_saliency_artifact(
        scores,
        score_type="unit_test_snip",
        source_split="source_fit",
    )
    path = tmp_path / "prior.pt"
    torch.save(artifact, path)

    prior, metadata = load_source_saliency_prior(
        path,
        expected_n_layers=3,
        expected_n_basis=3,
    )

    assert prior.shape == (3, 3)
    assert torch.isfinite(prior).all()
    assert metadata["source_split"] == "source_fit"
    assert metadata["target_eval_usage"] == "final_eval_only_no_selection"

    bad = dict(artifact)
    bad["metadata"] = dict(artifact["metadata"], source_splits=["target_eval"])
    bad_path = tmp_path / "bad.pt"
    torch.save(bad, bad_path)
    try:
        load_source_saliency_prior(bad_path, expected_n_layers=3, expected_n_basis=3)
    except ValueError as exc:
        assert "target-side" in str(exc)
    else:
        raise AssertionError("target_eval saliency artifact should be rejected")


def test_stable_rank_gated_saliency_beta_zero_preserves_logits():
    torch.manual_seed(123)
    baseline = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=5,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
    )
    torch.manual_seed(123)
    prior_model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=5,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_source_saliency_prior=torch.zeros(3, 5),
        hyper_source_saliency_prior_beta=0.0,
    )
    z = torch.randn(3, 16)

    assert torch.allclose(
        baseline.adapter_coefficient_logits(z, "bottleneck"),
        prior_model.adapter_coefficient_logits(z, "bottleneck"),
        atol=0.0,
        rtol=0.0,
    )


def test_stable_rank_gated_saliency_prior_beta_does_not_change_topk_by_default():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=5,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=1,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_source_saliency_prior=torch.tensor(
            [
                [-10.0, 10.0, -10.0, -10.0, -10.0],
                [-5.0, -5.0, 10.0, -5.0, -5.0],
                [-5.0, -5.0, -5.0, 10.0, -5.0],
            ],
            dtype=torch.float32,
        ),
        hyper_source_saliency_prior_beta=20.0,
    )
    with torch.no_grad():
        model.shared_coeff_generator.gate_head.weight.zero_()
        model.shared_coeff_generator.gate_head.bias.copy_(
            torch.tensor([2.0, 0.0, -1.0, -2.0, -3.0])
        )
        model.shared_coeff_generator.coeff_head.weight.zero_()
        model.shared_coeff_generator.coeff_head.bias.zero_()
    z = torch.zeros(2, 16)
    logits = model.adapter_coefficient_logits(z, "bottleneck")
    coeffs = model.hyper_adapter_b.coefficients(z, coeff_logits=logits)

    assert torch.isfinite(logits).all()
    assert torch.argmax(coeffs, dim=1).tolist() == [0, 0]


def test_stable_rank_gated_saliency_prior_can_change_topk_in_legacy_diagnostic_mode():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=5,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=1,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_source_saliency_prior=torch.tensor(
            [
                [-10.0, 10.0, -10.0, -10.0, -10.0],
                [-5.0, -5.0, 10.0, -5.0, -5.0],
                [-5.0, -5.0, -5.0, 10.0, -5.0],
            ],
            dtype=torch.float32,
        ),
        hyper_source_saliency_prior_beta=20.0,
        hyper_source_saliency_prior_application="legacy_gate_logit_bias_before_topk",
    )
    with torch.no_grad():
        model.shared_coeff_generator.gate_head.weight.zero_()
        model.shared_coeff_generator.gate_head.bias.copy_(
            torch.tensor([2.0, 0.0, -1.0, -2.0, -3.0])
        )
        model.shared_coeff_generator.coeff_head.weight.zero_()
        model.shared_coeff_generator.coeff_head.bias.zero_()
    z = torch.zeros(2, 16)
    logits = model.adapter_coefficient_logits(z, "bottleneck")
    coeffs = model.hyper_adapter_b.coefficients(z, coeff_logits=logits)

    assert torch.isfinite(logits).all()
    assert torch.argmax(coeffs, dim=1).tolist() == [1, 1]


def test_phys_token_operator_zero_init_preserves_m3_1_logits_and_forward():
    torch.manual_seed(2026)
    baseline = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=4,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=0.5,
        hyper_source_trust_top_m=2,
        hyper_source_trust_variable_gate=True,
        zero_shot_prior_form="source_base_residual_reliability_gated",
    )
    torch.manual_seed(2026)
    phys = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=4,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=0.5,
        hyper_source_trust_top_m=2,
        hyper_source_trust_variable_gate=True,
        zero_shot_prior_form="source_base_residual_reliability_gated",
        hyper_phys_context_modulation=True,
        hyper_phys_delta_scale=0.25,
        hyper_phys_gate_init=0.90,
        hyper_operator_droppath_p=0.10,
    )
    baseline.eval()
    phys.eval()
    x = torch.randn(2, 12, 16, 16)
    z = torch.randn(2, 16)
    z_phys = torch.randn(2, 16)
    trust_bank = {
        "schema_version": "hyperda_source_trust_bank_v1",
        "source": "source_fit_source_val_only",
        "label_usage": "none",
        "target_eval_usage": "final_eval_only_no_selection",
        "source_prompt_embeddings": torch.randn(5, 16),
        "layer_consensus_logits": {
            "bottleneck": torch.randn(5, 4),
            "dec2": torch.randn(5, 4),
            "dec1": torch.randn(5, 4),
        },
        "prompt_distance_quantiles": {"q75": 1.0},
    }
    reliability = torch.zeros(2, 5)

    with torch.no_grad():
        for layer_name in ("bottleneck", "dec2", "dec1"):
            assert torch.allclose(
                baseline.adapter_coefficient_logits(
                    z,
                    layer_name,
                    source_trust_bank=trust_bank,
                ),
                phys.adapter_coefficient_logits(
                    z,
                    layer_name,
                    source_trust_bank=trust_bank,
                    z_phys=z_phys,
                ),
                atol=0.0,
                rtol=0.0,
            )
        assert torch.allclose(
            baseline(
                x,
                z,
                reliability_features=reliability,
                source_trust_bank=trust_bank,
            ),
            phys(
                x,
                z,
                reliability_features=reliability,
                source_trust_bank=trust_bank,
                z_phys=z_phys,
            ),
            atol=0.0,
            rtol=0.0,
        )


def test_formula_phys_context_encoder_zero_init_preserves_m3_1_logits_and_forward():
    torch.manual_seed(2027)
    baseline = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=4,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=0.5,
        hyper_source_trust_top_m=2,
        hyper_source_trust_variable_gate=True,
        zero_shot_prior_form="source_base_residual_reliability_gated",
    )
    torch.manual_seed(2027)
    phys = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=4,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=0.5,
        hyper_source_trust_top_m=2,
        hyper_source_trust_variable_gate=True,
        zero_shot_prior_form="source_base_residual_reliability_gated",
        hyper_phys_context_modulation=True,
        hyper_phys_delta_scale=0.10,
        hyper_phys_gate_init=0.50,
        hyper_operator_droppath_p=0.10,
        phys_context_source="raw_input_side_formula_v2",
    )
    baseline.eval()
    phys.eval()
    assert phys.formula_phys_context_encoder is not None

    x = torch.randn(2, 12, 16, 16)
    z = torch.randn(2, 16)
    formula_features = torch.rand(2, 10)
    z_phys = phys.formula_phys_context_encoder(formula_features)
    trust_bank = {
        "schema_version": "hyperda_source_trust_bank_v1",
        "source": "source_fit_source_val_only",
        "label_usage": "none",
        "target_eval_usage": "final_eval_only_no_selection",
        "source_prompt_embeddings": torch.randn(5, 16),
        "layer_consensus_logits": {
            "bottleneck": torch.randn(5, 4),
            "dec2": torch.randn(5, 4),
            "dec1": torch.randn(5, 4),
        },
        "prompt_distance_quantiles": {"q75": 1.0},
    }
    reliability = torch.zeros(2, 5)
    variable_gate = torch.ones(2, 2)

    with torch.no_grad():
        for layer_name in ("bottleneck", "dec2", "dec1"):
            assert torch.allclose(
                baseline.adapter_coefficient_logits(
                    z,
                    layer_name,
                    source_trust_bank=trust_bank,
                ),
                phys.adapter_coefficient_logits(
                    z,
                    layer_name,
                    source_trust_bank=trust_bank,
                    z_phys=z_phys,
                ),
                atol=0.0,
                rtol=0.0,
            )
        assert torch.allclose(
            baseline(
                x,
                z,
                reliability_features=reliability,
                source_trust_bank=trust_bank,
                variable_trust_gate=variable_gate,
            ),
            phys(
                x,
                z,
                reliability_features=reliability,
                source_trust_bank=trust_bank,
                z_phys=z_phys,
                variable_trust_gate=variable_gate,
            ),
            atol=0.0,
            rtol=0.0,
        )
    assert phys.last_phys_operator_summary["enabled"] is True
    assert phys.last_phys_operator_summary["phys_delta_scale"] == pytest.approx(0.10)


def test_formula_phys_context_encoder_enhanced_zero_init_preserves_m3_1_logits_and_forward():
    from hydroda.models.phys_trust import PHYS_FORMULA_ENHANCED_FEATURE_SCHEMA

    torch.manual_seed(2028)
    baseline = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=4,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=0.5,
        hyper_source_trust_top_m=2,
        hyper_source_trust_variable_gate=True,
        zero_shot_prior_form="source_base_residual_reliability_gated",
    )
    torch.manual_seed(2028)
    phys = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=4,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=0.5,
        hyper_source_trust_top_m=2,
        hyper_source_trust_variable_gate=True,
        zero_shot_prior_form="source_base_residual_reliability_gated",
        hyper_phys_context_modulation=True,
        hyper_phys_delta_scale=0.05,
        hyper_phys_gate_init=0.35,
        hyper_operator_droppath_p=0.10,
        phys_context_source="raw_input_side_formula_v3_enhanced",
    )
    baseline.eval()
    phys.eval()
    assert phys.formula_phys_context_encoder is not None
    assert phys.formula_phys_context_encoder.feature_dim == len(PHYS_FORMULA_ENHANCED_FEATURE_SCHEMA)

    x = torch.randn(2, 12, 16, 16)
    z = torch.randn(2, 16)
    formula_features = torch.rand(2, len(PHYS_FORMULA_ENHANCED_FEATURE_SCHEMA))
    z_phys = phys.formula_phys_context_encoder(formula_features)
    trust_bank = {
        "schema_version": "hyperda_source_trust_bank_v1",
        "source": "source_fit_source_val_only",
        "label_usage": "none",
        "target_eval_usage": "final_eval_only_no_selection",
        "source_prompt_embeddings": torch.randn(5, 16),
        "layer_consensus_logits": {
            "bottleneck": torch.randn(5, 4),
            "dec2": torch.randn(5, 4),
            "dec1": torch.randn(5, 4),
        },
        "prompt_distance_quantiles": {"q75": 1.0},
    }
    reliability = torch.zeros(2, 5)
    variable_gate = torch.ones(2, 2)

    with torch.no_grad():
        for layer_name in ("bottleneck", "dec2", "dec1"):
            assert torch.allclose(
                baseline.adapter_coefficient_logits(
                    z,
                    layer_name,
                    source_trust_bank=trust_bank,
                ),
                phys.adapter_coefficient_logits(
                    z,
                    layer_name,
                    source_trust_bank=trust_bank,
                    z_phys=z_phys,
                ),
                atol=0.0,
                rtol=0.0,
            )
        assert torch.allclose(
            baseline(
                x,
                z,
                reliability_features=reliability,
                source_trust_bank=trust_bank,
                variable_trust_gate=variable_gate,
            ),
            phys(
                x,
                z,
                reliability_features=reliability,
                source_trust_bank=trust_bank,
                z_phys=z_phys,
                variable_trust_gate=variable_gate,
            ),
            atol=0.0,
            rtol=0.0,
        )
    assert phys.last_phys_operator_summary["enabled"] is True
    assert phys.last_phys_operator_summary["phys_context_source"] == "raw_input_side_formula_v3_enhanced"
    assert phys.last_phys_operator_summary["phys_delta_scale"] == pytest.approx(0.05)
    assert phys.last_phys_operator_summary["phys_gate_init"] == pytest.approx(0.35)
    assert phys.last_phys_operator_summary["target_eval_usage"] == "final_eval_only_no_selection"


def test_m3_14_formula_gain_operator_zero_init_preserves_logits_and_has_no_output_residual_branch():
    from hydroda.models.phys_trust import PHYS_FORMULA_GAIN_FEATURE_SCHEMA

    torch.manual_seed(2029)
    baseline = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=4,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=0.5,
        hyper_source_trust_top_m=2,
        hyper_source_trust_variable_gate=True,
        zero_shot_prior_form="source_base_residual_reliability_gated",
    )
    torch.manual_seed(2029)
    phys = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=4,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=0.5,
        hyper_source_trust_top_m=2,
        hyper_source_trust_variable_gate=True,
        zero_shot_prior_form="source_base_residual_reliability_gated",
        hyper_phys_context_modulation=True,
        hyper_phys_delta_scale=0.05,
        hyper_phys_gate_init=0.50,
        hyper_operator_droppath_p=0.10,
        phys_context_source="raw_input_side_formula_gain",
    )
    baseline.eval()
    phys.eval()
    assert phys.formula_phys_context_encoder is not None
    assert phys.formula_phys_context_encoder.feature_dim == len(PHYS_FORMULA_GAIN_FEATURE_SCHEMA)
    assert phys.phys_gain_basis_residual is None
    assert phys.hyper_phys_gain_basis_residual is False

    x = torch.randn(2, 12, 16, 16)
    z = torch.randn(2, 16)
    formula_features = torch.rand(2, len(PHYS_FORMULA_GAIN_FEATURE_SCHEMA))
    z_phys = phys.formula_phys_context_encoder(formula_features)
    trust_bank = {
        "schema_version": "hyperda_source_trust_bank_v1",
        "source": "source_fit_source_val_only",
        "label_usage": "none",
        "target_eval_usage": "final_eval_only_no_selection",
        "source_prompt_embeddings": torch.randn(5, 16),
        "layer_consensus_logits": {
            "bottleneck": torch.randn(5, 4),
            "dec2": torch.randn(5, 4),
            "dec1": torch.randn(5, 4),
        },
        "prompt_distance_quantiles": {"q75": 1.0},
    }
    reliability = torch.zeros(2, 5)
    variable_gate = torch.ones(2, 2)

    with torch.no_grad():
        for layer_name in ("bottleneck", "dec2", "dec1"):
            assert torch.allclose(
                baseline.adapter_coefficient_logits(
                    z,
                    layer_name,
                    source_trust_bank=trust_bank,
                ),
                phys.adapter_coefficient_logits(
                    z,
                    layer_name,
                    source_trust_bank=trust_bank,
                    z_phys=z_phys,
                ),
                atol=0.0,
                rtol=0.0,
            )
        assert torch.allclose(
            baseline(
                x,
                z,
                reliability_features=reliability,
                source_trust_bank=trust_bank,
                variable_trust_gate=variable_gate,
            ),
            phys(
                x,
                z,
                reliability_features=reliability,
                source_trust_bank=trust_bank,
                z_phys=z_phys,
                variable_trust_gate=variable_gate,
            ),
            atol=0.0,
            rtol=0.0,
        )
    summary = phys.last_phys_operator_summary
    assert summary["enabled"] is True
    assert summary["method_id"] == "M3_14_source_trained_phys_formula_gain_hypertrust"
    assert summary["phys_context_source"] == "raw_input_side_formula_gain"
    assert summary["coefficient_injection_role"] == "bounded_operator_coefficient_logit_delta_only"
    assert summary["final_output_residual_allowed"] is False
    assert summary["phys_delta_scale"] == pytest.approx(0.05)
    assert summary["phys_gate_init"] == pytest.approx(0.50)
    assert summary["checkpoint_start"] == "source_pooled_global_backbone"


def test_m3_16_lite_formula_gain_operator_zero_init_preserves_m3_1_path():
    from hydroda.models.phys_trust import PHYS_FORMULA_GAIN_FEATURE_SCHEMA

    torch.manual_seed(2030)
    baseline = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=4,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=0.5,
        hyper_source_trust_top_m=2,
        hyper_source_trust_variable_gate=True,
        zero_shot_prior_form="source_base_residual_reliability_gated",
    )
    torch.manual_seed(2030)
    phys = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=4,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=0.5,
        hyper_source_trust_top_m=2,
        hyper_source_trust_variable_gate=True,
        zero_shot_prior_form="source_base_residual_reliability_gated",
        hyper_phys_context_modulation=True,
        hyper_phys_delta_scale=0.03,
        hyper_phys_gate_init=0.25,
        hyper_operator_droppath_p=0.10,
        phys_context_source="raw_input_side_formula_gain",
    )
    baseline.eval()
    phys.eval()
    assert phys.formula_phys_context_encoder is not None
    assert phys.formula_phys_context_encoder.feature_dim == len(PHYS_FORMULA_GAIN_FEATURE_SCHEMA)
    assert phys.phys_gain_basis_residual is None
    assert phys.hyper_phys_gain_basis_residual is False

    x = torch.randn(2, 12, 16, 16)
    z = torch.randn(2, 16)
    z_phys = phys.formula_phys_context_encoder(torch.rand(2, len(PHYS_FORMULA_GAIN_FEATURE_SCHEMA)))
    trust_bank = {
        "schema_version": "hyperda_source_trust_bank_v1",
        "source": "source_fit_source_val_only",
        "label_usage": "none",
        "target_eval_usage": "final_eval_only_no_selection",
        "source_prompt_embeddings": torch.randn(5, 16),
        "layer_consensus_logits": {
            "bottleneck": torch.randn(5, 4),
            "dec2": torch.randn(5, 4),
            "dec1": torch.randn(5, 4),
        },
        "prompt_distance_quantiles": {"q75": 1.0},
    }
    reliability = torch.zeros(2, 5)
    variable_gate = torch.ones(2, 2)

    with torch.no_grad():
        for layer_name in ("bottleneck", "dec2", "dec1"):
            assert torch.allclose(
                baseline.adapter_coefficient_logits(
                    z,
                    layer_name,
                    source_trust_bank=trust_bank,
                ),
                phys.adapter_coefficient_logits(
                    z,
                    layer_name,
                    source_trust_bank=trust_bank,
                    z_phys=z_phys,
                ),
                atol=0.0,
                rtol=0.0,
            )
        assert torch.allclose(
            baseline(
                x,
                z,
                reliability_features=reliability,
                source_trust_bank=trust_bank,
                variable_trust_gate=variable_gate,
            ),
            phys(
                x,
                z,
                reliability_features=reliability,
                source_trust_bank=trust_bank,
                z_phys=z_phys,
                variable_trust_gate=variable_gate,
            ),
            atol=0.0,
            rtol=0.0,
        )
    summary = phys.last_phys_operator_summary
    assert summary["enabled"] is True
    assert summary["method_id"] == "M3_16_source_only_phys_m3trust_lite"
    assert summary["coefficient_injection_role"] == "bounded_operator_coefficient_logit_delta_only"
    assert summary["final_output_residual_allowed"] is False
    assert summary["second_model_forward_allowed"] is False
    assert summary["stage2_source_only_invariant"] is True
    assert summary["phys_delta_scale"] == pytest.approx(0.03)
    assert summary["phys_gate_init"] == pytest.approx(0.25)
    assert summary["checkpoint_start"] == "source_pooled_global_backbone"
    assert summary["source_fit_regularization_lambda_default"] == pytest.approx(0.0)


def test_m3_15_phys_coeff_delta_scope_trains_only_operator_residual():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=4,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=0.5,
        hyper_source_trust_top_m=2,
        hyper_source_trust_variable_gate=True,
        zero_shot_prior_form="source_base_residual_reliability_gated",
        hyper_phys_context_modulation=True,
        phys_context_source="raw_input_side_formula_gain",
    )
    prompt_encoder = RegionPromptEncoder(num_regions=5, input_channels=12, hidden_dim=16)

    metadata = apply_trainable_scope(
        model=model,
        prompt_encoder=prompt_encoder,
        trainable_scope="phys_coeff_delta_only",
    )

    trainable = metadata["trainable_parameter_names"]
    assert trainable
    assert all(name.startswith("model.phys_operator_residual.") for name in trainable)
    assert not any(name.startswith("model.formula_phys_context_encoder.") for name in trainable)
    assert not any(name.startswith("prompt_encoder.") for name in trainable)
    assert "formula_phys_context_encoder" in metadata["frozen_source_base_modules"]
    assert model.phys_gain_basis_residual is None


def test_phys_token_does_not_replace_prompt_space_trust_neighbor_query():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=4,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware",
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=1.0,
        hyper_source_trust_top_m=1,
        hyper_phys_context_modulation=True,
        hyper_operator_droppath_p=0.0,
    )
    model.eval()
    z = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    z_phys_a = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    z_phys_b = torch.tensor([[0.0, -1.0, 0.0, 0.0]])
    trust_bank = {
        "schema_version": "hyperda_source_trust_bank_v1",
        "source": "source_fit_source_val_only",
        "label_usage": "none",
        "target_eval_usage": "final_eval_only_no_selection",
        "source_prompt_embeddings": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "source_trust_query_embeddings": torch.tensor(
            [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "layer_consensus_logits": {
            "bottleneck": torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]),
        },
        "prompt_distance_quantiles": {"q75": 1.0},
        "source_trust_query_mode": "prompt_embedding",
    }

    logits_a = model.adapter_coefficient_logits(
        z,
        "bottleneck",
        source_trust_bank=trust_bank,
        z_phys=z_phys_a,
    )
    logits_b = model.adapter_coefficient_logits(
        z,
        "bottleneck",
        source_trust_bank=trust_bank,
        z_phys=z_phys_b,
    )

    assert torch.allclose(logits_a, logits_b)
    summary = model.last_trust_routing_summary["bottleneck"]
    assert summary["source_bank_embedding_key"] == "source_prompt_embeddings"
    assert summary["trust_query_source"] == "target_prompt"
    assert summary["nearest_neighbor_indices"] == [[0]]
    assert summary["trust_routing_geometry"] == "prompt_embedding"


def test_operator_droppath_only_changes_phys_delta_in_train_mode():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=4,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware",
        hyper_phys_context_modulation=True,
        hyper_operator_droppath_p=0.5,
    )
    with torch.no_grad():
        model.phys_operator_residual.delta_head.weight.fill_(0.25)
        model.phys_operator_residual.delta_head.bias.fill_(0.1)
    z = torch.ones(8, 4)
    z_phys = torch.full((8, 4), 0.5)
    base = torch.zeros(8, 3)

    model.eval()
    eval_a = model.phys_modulated_coefficient_logits(z, z_phys, "bottleneck", base)
    eval_b = model.phys_modulated_coefficient_logits(z, z_phys, "bottleneck", base)
    assert torch.allclose(eval_a, eval_b)
    assert model.last_phys_operator_summary["operator_droppath_active"] is False

    model.train()
    torch.manual_seed(1)
    train_a = model.phys_modulated_coefficient_logits(z, z_phys, "bottleneck", base)
    torch.manual_seed(2)
    train_b = model.phys_modulated_coefficient_logits(z, z_phys, "bottleneck", base)
    assert not torch.allclose(train_a, train_b)
    assert model.last_phys_operator_summary["operator_droppath_active"] is True
    assert model.last_phys_operator_summary["operator_droppath_p"] == pytest.approx(0.5)


def test_prompt_manifold_reliability_scales_adapter_residual_gate_only_with_features():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_reliability_gate="prompt_scalar",
        hyper_prompt_manifold_reliability=True,
        hyper_prompt_manifold_reliability_strength=0.5,
    )
    z = torch.randn(2, 16)
    features = torch.zeros(2, 5)
    features[:, 4] = torch.tensor([0.0, 1.0])

    multiplier = model.prompt_manifold_reliability_multiplier(z, features)
    gate = model.adapter_reliability_gate(z, "bottleneck") * multiplier

    assert torch.allclose(multiplier[:, 0], torch.tensor([1.0, 0.5]))
    assert (gate[:, 0] <= model.adapter_reliability_gate(z, "bottleneck")[:, 0]).all()


def test_source_manifold_guard_shrinks_residual_without_changing_source_base_path():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        zero_shot_prior_form="source_base_residual_reliability_gated",
        source_residual_rho=1.0,
        hyper_source_manifold_guard=True,
        hyper_source_manifold_guard_strength=0.5,
    )
    x = torch.randn(1, 12, 16, 16)
    z = torch.randn(1, 16)
    low_distance = torch.zeros(1, 5)
    high_distance = torch.zeros(1, 5)
    high_distance[:, 4] = 1.0

    with torch.no_grad():
        source_before = model.source_base_forward(x)
        rho0 = model(x, z, reliability_features=high_distance, rho=0.0)
        rho1_low = model(x, z, reliability_features=low_distance, rho=1.0)
        rho1_high = model(x, z, reliability_features=high_distance, rho=1.0)
        source_after = model.source_base_forward(x)

    assert torch.allclose(source_before, source_after)
    assert torch.allclose(rho0, source_before)
    assert torch.linalg.vector_norm(rho1_high - source_before) <= torch.linalg.vector_norm(rho1_low - source_before)


def test_trust_routed_coefficients_blend_target_logits_with_source_neighbor_consensus():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=4,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware",
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=0.5,
        hyper_source_trust_top_m=2,
    )
    z = torch.tensor([[0.8, 0.2, 0.0, 0.0]], dtype=torch.float32)
    target_logits = torch.tensor([[4.0, 0.0, -4.0]], dtype=torch.float32)
    trust_bank = {
        "schema_version": "hyperda_source_trust_bank_v1",
        "source": "source_fit_source_val_only",
        "label_usage": "none",
        "target_eval_usage": "final_eval_only_no_selection",
        "source_prompt_embeddings": torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "layer_consensus_logits": {
            "bottleneck": torch.tensor(
                [
                    [0.0, 4.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [4.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        },
        "distance_quantiles": {"q75": 1.0},
    }

    routed = model.trust_routed_coefficient_logits(
        z,
        "bottleneck",
        target_logits,
        source_trust_bank=trust_bank,
    )

    distances = torch.tensor([[((0.8 - 1.0) ** 2 + 0.2**2) ** 0.5, (0.8**2 + (0.2 - 1.0) ** 2) ** 0.5]])
    weights = torch.softmax(-distances / 1.0, dim=1)
    expected_consensus = weights[:, :1] * torch.tensor([[0.0, 4.0, 0.0]]) + weights[:, 1:] * torch.tensor([[0.0, 2.0, 0.0]])
    expected = 0.5 * target_logits + 0.5 * expected_consensus
    assert torch.allclose(routed, expected, atol=1e-6)
    assert model.last_trust_routing_summary["bottleneck"]["source_neighbor_top_m"] == 2
    assert model.last_trust_routing_summary["bottleneck"]["trust_strength"] == 0.5
    assert model.last_trust_routing_summary["bottleneck"]["nearest_neighbor_indices"] == [[0, 1]]


def test_trust_routing_can_use_separate_query_without_replacing_target_prompt_logits():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=4,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware",
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=1.0,
        hyper_source_trust_top_m=1,
    )
    target_prompt = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    raw_trust_query = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    target_logits = torch.tensor([[9.0, -9.0, 1.0]], dtype=torch.float32)
    trust_bank = {
        "schema_version": "hyperda_source_trust_bank_v1",
        "source": "source_fit_source_val_only",
        "label_usage": "none",
        "target_eval_usage": "final_eval_only_no_selection",
        "source_prompt_embeddings": torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "source_trust_query_embeddings": torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "layer_consensus_logits": {
            "bottleneck": torch.tensor(
                [
                    [3.0, 0.0, 0.0],
                    [0.0, 5.0, 0.0],
                ],
                dtype=torch.float32,
            )
        },
        "distance_quantiles": {"q75": 1.0},
    }

    routed = model.trust_routed_coefficient_logits(
        target_prompt,
        "bottleneck",
        target_logits,
        source_trust_bank=trust_bank,
        source_trust_query=raw_trust_query,
    )

    assert torch.allclose(routed, torch.tensor([[0.0, 5.0, 0.0]]), atol=1e-6)
    assert model.last_trust_routing_summary["bottleneck"]["nearest_neighbor_indices"] == [[1]]
    assert model.last_trust_routing_summary["bottleneck"]["trust_query_source"] == "source_trust_query"


def test_raw_trust_query_requires_matching_source_query_bank():
    from hydroda.baselines.prompt_conditioned import normalize_hyperda_source_trust_bank_state

    stale_bank = {
        "schema_version": "hyperda_source_trust_bank_v1",
        "source": "source_fit_source_val_only",
        "label_usage": "none",
        "target_eval_usage": "final_eval_only_no_selection",
        "source_trust_query_mode": "raw_input_side_da_diagnostics",
        "source_prompt_embeddings": torch.zeros(2, 4),
        "layer_consensus_logits": {
            "bottleneck": torch.zeros(2, 3),
        },
        "distance_quantiles": {"q75": 1.0},
    }

    with pytest.raises(ValueError, match="raw_input_side_da_diagnostics.*source_trust_query_embeddings"):
        normalize_hyperda_source_trust_bank_state(stale_bank)


def test_blended_trust_query_uses_separate_blended_bank_without_changing_prompt_logits():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=4,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware",
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=1.0,
        hyper_source_trust_top_m=1,
    )
    target_prompt = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    blended_query = torch.tensor([[0.75, 0.25, 0.0, 0.0]], dtype=torch.float32)
    target_logits = torch.tensor([[9.0, -9.0, 1.0]], dtype=torch.float32)
    trust_bank = {
        "schema_version": "hyperda_source_trust_bank_v1",
        "source": "source_fit_source_val_only",
        "source_trust_query_mode": "blended_prompt_raw_da_0p25",
        "source_prompt_embeddings": torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "source_trust_query_embeddings": torch.tensor(
            [
                [0.75, 0.25, 0.0, 0.0],
                [0.25, 0.75, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "layer_consensus_logits": {
            "bottleneck": torch.tensor(
                [
                    [3.0, 0.0, 0.0],
                    [0.0, 5.0, 0.0],
                ],
                dtype=torch.float32,
            )
        },
        "distance_quantiles": {"q75": 1.0},
    }

    routed = model.trust_routed_coefficient_logits(
        target_prompt,
        "bottleneck",
        target_logits,
        source_trust_bank=trust_bank,
        source_trust_query=blended_query,
    )

    assert torch.allclose(routed, torch.tensor([[3.0, 0.0, 0.0]]), atol=1e-6)
    summary = model.last_trust_routing_summary["bottleneck"]
    assert summary["nearest_neighbor_indices"] == [[0]]
    assert summary["source_bank_embedding_key"] == "source_trust_query_embeddings"
    assert summary["trust_query_source"] == "source_trust_query"


def test_phys_agreement_guard_keeps_prompt_space_m3_1_when_neighbors_agree():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=4,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware",
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=0.5,
        hyper_source_trust_top_m=1,
        hyper_phys_agreement_guard=True,
        hyper_phys_agreement_guard_strength=1.0,
    )
    target_prompt = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    raw_phys_query = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    target_logits = torch.tensor([[4.0, 0.0, -4.0]], dtype=torch.float32)
    trust_bank = {
        "schema_version": "hyperda_source_trust_bank_v1",
        "source": "source_fit_source_val_only",
        "source_trust_query_mode": "raw_input_side_da_diagnostics",
        "source_prompt_embeddings": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "source_trust_query_embeddings": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "layer_consensus_logits": {
            "bottleneck": torch.tensor(
                [[0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
                dtype=torch.float32,
            )
        },
        "distance_quantiles": {"q75": 1.0},
    }

    routed = model.trust_routed_coefficient_logits(
        target_prompt,
        "bottleneck",
        target_logits,
        source_trust_bank=trust_bank,
        source_trust_query=raw_phys_query,
    )

    expected = 0.5 * target_logits + 0.5 * torch.tensor([[0.0, 4.0, 0.0]])
    summary = model.last_trust_routing_summary["bottleneck"]
    assert torch.allclose(routed, expected, atol=1e-6)
    assert summary["source_bank_embedding_key"] == "source_prompt_embeddings"
    assert summary["trust_query_source"] == "target_prompt"
    assert summary["phys_agreement_guard"]["enabled"] is True
    assert summary["phys_agreement_guard"]["soft_neighbor_agreement_mean"] == pytest.approx(1.0)
    assert summary["effective_trust_strength_mean"] == pytest.approx(0.5)


def test_phys_agreement_guard_uses_prompt_temperature_and_phys_ood_temperature():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=2,
        hyper_n_basis=2,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware",
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=1.0,
        hyper_source_trust_top_m=2,
        hyper_phys_agreement_guard=True,
        hyper_phys_agreement_guard_strength=1.0,
    )
    target_prompt = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    raw_phys_query = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    target_logits = torch.zeros(1, 2, dtype=torch.float32)
    consensus_logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]], dtype=torch.float32)
    trust_bank = {
        "schema_version": "hyperda_source_trust_bank_v1",
        "source": "source_fit_source_val_only",
        "source_trust_query_mode": "raw_input_side_da_diagnostics",
        "source_prompt_embeddings": torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
        "source_trust_query_embeddings": torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "layer_consensus_logits": {"bottleneck": consensus_logits},
        "distance_quantiles": {"q75": 10.0},
        "prompt_distance_quantiles": {"q75": 0.1},
        "source_trust_query_distance_quantiles": {"q75": 10.0},
    }

    routed = model.trust_routed_coefficient_logits(
        target_prompt,
        "bottleneck",
        target_logits,
        source_trust_bank=trust_bank,
        source_trust_query=raw_phys_query,
    )

    weights = torch.softmax(torch.tensor([[0.0, -10.0]], dtype=torch.float32), dim=1)
    expected = weights @ consensus_logits
    summary = model.last_trust_routing_summary["bottleneck"]
    assert torch.allclose(routed, expected, atol=1e-5)
    assert summary["distance_temperature"] == pytest.approx(0.1)
    assert summary["phys_agreement_guard"]["distance_temperature"] == pytest.approx(10.0)
    assert summary["phys_agreement_guard"]["multiplier_mean"] == pytest.approx(1.0)


def test_phys_agreement_guard_only_shrinks_when_phys_neighbors_disagree():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=4,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware",
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=0.5,
        hyper_source_trust_top_m=1,
        hyper_phys_agreement_guard=True,
        hyper_phys_agreement_guard_strength=1.0,
    )
    target_prompt = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    raw_phys_query = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    target_logits = torch.tensor([[4.0, 0.0, -4.0]], dtype=torch.float32)
    trust_bank = {
        "schema_version": "hyperda_source_trust_bank_v1",
        "source": "source_fit_source_val_only",
        "source_trust_query_mode": "raw_input_side_da_diagnostics",
        "source_prompt_embeddings": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "source_trust_query_embeddings": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "layer_consensus_logits": {
            "bottleneck": torch.tensor(
                [[0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
                dtype=torch.float32,
            )
        },
        "distance_quantiles": {"q75": 1.0},
    }

    routed = model.trust_routed_coefficient_logits(
        target_prompt,
        "bottleneck",
        target_logits,
        source_trust_bank=trust_bank,
        source_trust_query=raw_phys_query,
    )

    summary = model.last_trust_routing_summary["bottleneck"]
    assert torch.allclose(routed, target_logits, atol=1e-6)
    assert summary["source_bank_embedding_key"] == "source_prompt_embeddings"
    assert summary["phys_agreement_guard"]["prompt_neighbor_indices"] == [[0]]
    assert summary["phys_agreement_guard"]["phys_neighbor_indices"] == [[1]]
    assert summary["phys_agreement_guard"]["soft_neighbor_agreement_mean"] == pytest.approx(0.0)
    assert summary["effective_trust_strength_mean"] == pytest.approx(0.0)


def test_phys_agreement_guard_conservative_and_rule_requires_disagreement_and_ood():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=4,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware",
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=0.5,
        hyper_source_trust_top_m=1,
        hyper_phys_agreement_guard=True,
        hyper_phys_agreement_guard_strength=1.0,
        hyper_phys_agreement_guard_min_multiplier=0.8,
        hyper_phys_agreement_guard_risk_rule="and",
    )
    target_prompt = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    raw_phys_query = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    target_logits = torch.tensor([[4.0, 0.0, -4.0]], dtype=torch.float32)
    trust_bank = {
        "schema_version": "hyperda_source_trust_bank_v1",
        "source": "source_fit_source_val_only",
        "source_trust_query_mode": "raw_input_side_da_diagnostics",
        "source_prompt_embeddings": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "source_trust_query_embeddings": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "layer_consensus_logits": {
            "bottleneck": torch.tensor(
                [[0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
                dtype=torch.float32,
            )
        },
        "source_trust_query_distance_quantiles": {"q75": 10.0},
        "prompt_distance_quantiles": {"q75": 1.0},
        "distance_quantiles": {"q75": 1.0},
    }

    routed = model.trust_routed_coefficient_logits(
        target_prompt,
        "bottleneck",
        target_logits,
        source_trust_bank=trust_bank,
        source_trust_query=raw_phys_query,
    )

    expected = 0.5 * target_logits + 0.5 * torch.tensor([[0.0, 4.0, 0.0]])
    summary = model.last_trust_routing_summary["bottleneck"]
    assert torch.allclose(routed, expected, atol=1e-6)
    assert summary["phys_agreement_guard"]["soft_neighbor_agreement_mean"] == pytest.approx(0.0)
    assert summary["phys_agreement_guard"]["phys_ood_distance_bounded_mean"] < 0.2
    assert summary["phys_agreement_guard"]["risk_rule"] == "and"
    assert summary["phys_agreement_guard"]["multiplier_mean"] == pytest.approx(1.0)
    assert summary["effective_trust_strength_mean"] == pytest.approx(0.5)


def test_phys_agreement_guard_min_multiplier_prevents_global_off_switch():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=4,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware",
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=0.5,
        hyper_source_trust_top_m=1,
        hyper_phys_agreement_guard=True,
        hyper_phys_agreement_guard_strength=1.0,
        hyper_phys_agreement_guard_min_multiplier=0.8,
        hyper_phys_agreement_guard_risk_rule="and",
    )
    target_prompt = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    raw_phys_query = torch.tensor([[0.0, 3.0, 0.0, 0.0]], dtype=torch.float32)
    target_logits = torch.tensor([[4.0, 0.0, -4.0]], dtype=torch.float32)
    trust_bank = {
        "schema_version": "hyperda_source_trust_bank_v1",
        "source": "source_fit_source_val_only",
        "source_trust_query_mode": "raw_input_side_da_diagnostics",
        "source_prompt_embeddings": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "source_trust_query_embeddings": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "layer_consensus_logits": {
            "bottleneck": torch.tensor(
                [[0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
                dtype=torch.float32,
            )
        },
        "source_trust_query_distance_quantiles": {"q75": 1.0},
        "prompt_distance_quantiles": {"q75": 1.0},
        "distance_quantiles": {"q75": 1.0},
    }

    routed = model.trust_routed_coefficient_logits(
        target_prompt,
        "bottleneck",
        target_logits,
        source_trust_bank=trust_bank,
        source_trust_query=raw_phys_query,
    )

    expected = 0.6 * target_logits + 0.4 * torch.tensor([[0.0, 4.0, 0.0]])
    summary = model.last_trust_routing_summary["bottleneck"]
    assert torch.allclose(routed, expected, atol=1e-6)
    assert summary["phys_agreement_guard"]["phys_ood_distance_bounded_mean"] == pytest.approx(1.0)
    assert summary["phys_agreement_guard"]["multiplier_min"] == pytest.approx(0.8)
    assert summary["phys_agreement_guard"]["min_multiplier"] == pytest.approx(0.8)
    assert summary["effective_trust_strength_mean"] == pytest.approx(0.4)


def test_phys_agreement_guard_shrinks_source_residual_without_amplifying_it():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=4,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware",
        zero_shot_prior_form="source_base_residual_reliability_gated",
        source_residual_rho=1.0,
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=0.5,
        hyper_source_trust_top_m=1,
        hyper_phys_agreement_guard=True,
        hyper_phys_agreement_guard_strength=1.0,
    )
    x = torch.randn(1, 12, 16, 16)
    target_prompt = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    raw_agree = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    raw_disagree = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    reliability = torch.zeros(1, 5)
    trust_bank = {
        "schema_version": "hyperda_source_trust_bank_v1",
        "source": "source_fit_source_val_only",
        "source_trust_query_mode": "raw_input_side_da_diagnostics",
        "source_prompt_embeddings": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "source_trust_query_embeddings": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "layer_consensus_logits": {
            "bottleneck": torch.zeros(2, 3),
            "dec2": torch.zeros(2, 3),
            "dec1": torch.zeros(2, 3),
        },
        "distance_quantiles": {"q75": 1.0},
    }
    with torch.no_grad():
        model.residual_head.weight.zero_()
        model.residual_head.bias.copy_(torch.tensor([0.1, -0.2]))
        source_base = model.source_base_forward(x)
        agree = model(
            x,
            target_prompt,
            reliability_features=reliability,
            source_trust_bank=trust_bank,
            source_trust_query=raw_agree,
        )
        disagree = model(
            x,
            target_prompt,
            reliability_features=reliability,
            source_trust_bank=trust_bank,
            source_trust_query=raw_disagree,
        )

    agree_norm = torch.linalg.vector_norm(agree - source_base)
    disagree_norm = torch.linalg.vector_norm(disagree - source_base)
    assert disagree_norm <= agree_norm
    assert disagree_norm == pytest.approx(0.0, abs=1e-6)


def test_trust_strength_zero_preserves_m2_1_coefficient_logits_exactly():
    torch.manual_seed(314)
    baseline = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=8,
        hyper_n_basis=4,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
    )
    torch.manual_seed(314)
    trust_zero = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=8,
        hyper_n_basis=4,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_source_trust_routing=True,
        hyper_source_trust_strength=0.0,
        hyper_source_trust_top_m=4,
    )
    z = torch.randn(3, 8)
    trust_bank = {
        "schema_version": "hyperda_source_trust_bank_v1",
        "source_prompt_embeddings": torch.randn(5, 8),
        "layer_consensus_logits": {
            "bottleneck": torch.randn(5, 4),
            "dec2": torch.randn(5, 4),
            "dec1": torch.randn(5, 4),
        },
        "distance_quantiles": {"q75": 1.0},
    }

    for layer_name in ("bottleneck", "dec2", "dec1"):
        assert torch.allclose(
            baseline.adapter_coefficient_logits(z, layer_name),
            trust_zero.adapter_coefficient_logits(
                z,
                layer_name,
                source_trust_bank=trust_bank,
            ),
            atol=0.0,
            rtol=0.0,
        )


def test_variable_trust_gate_can_shrink_surface_without_shrinking_rootzone():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        zero_shot_prior_form="source_base_residual_reliability_gated",
        source_residual_rho=1.0,
        hyper_source_trust_variable_gate=True,
    )
    x = torch.randn(1, 12, 16, 16)
    z = torch.randn(1, 16)
    reliability = torch.zeros(1, 5)
    gate = torch.tensor([[0.25, 1.0]], dtype=torch.float32)

    with torch.no_grad():
        model.residual_head.weight.zero_()
        model.residual_head.bias.copy_(torch.tensor([0.1, -0.2]))
        source_base = model.source_base_forward(x)
        no_gate = model(x, z, reliability_features=reliability, rho=1.0)
        variable_gated = model(
            x,
            z,
            reliability_features=reliability,
            rho=1.0,
            variable_trust_gate=gate,
        )

    residual_no_gate = no_gate - source_base
    residual_gated = variable_gated - source_base
    assert torch.allclose(residual_gated[:, 0], residual_no_gate[:, 0] * 0.25, atol=1e-6)
    assert torch.allclose(residual_gated[:, 1], residual_no_gate[:, 1], atol=1e-6)
    assert model.last_variable_trust_gate_summary["surface"]["mean"] == 0.25
    assert model.last_variable_trust_gate_summary["rootzone"]["mean"] == 1.0


def test_source_residual_prior_zero_init_is_strict_source_base_identity():
    source = SmallResUNet(in_channels=12, out_channels=2, width=8)
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        zero_shot_prior_form="source_residual_prior",
        source_residual_rho=1.0,
    )
    shared = {
        name: value
        for name, value in source.state_dict().items()
        if name in model.state_dict()
    }
    model.load_state_dict({**model.state_dict(), **shared})
    x = torch.randn(2, 12, 32, 48)
    z = torch.randn(2, 16)
    reliability = torch.tensor(
        [
            [4.0, 1.0, 20.0, 0.8, 0.1],
            [0.0, 0.0, 20.0, 0.6, 0.4],
        ],
        dtype=torch.float32,
    )

    expected = source(x)
    pred = model(x, z, reliability_features=reliability)

    assert model.zero_shot_prior_form == "source_residual_prior"
    assert torch.allclose(pred, expected, atol=1e-6)


def test_source_residual_prior_rho_zero_is_source_base_and_rho_one_adds_delta():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        zero_shot_prior_form="source_residual_prior",
        source_residual_rho=1.0,
    )
    x = torch.randn(1, 12, 32, 48)
    z = torch.randn(1, 16)
    reliability = torch.ones(1, 5)
    with torch.no_grad():
        model.residual_head.weight.fill_(0.02)
        model.residual_head.bias.fill_(0.1)

    source_base = model.source_base_forward(x)
    rho0 = model(x, z, rho=0.0, reliability_features=reliability)
    rho1 = model(x, z, rho=1.0, reliability_features=reliability)
    rho_half = model(x, z, rho=0.5, reliability_features=reliability)

    assert torch.allclose(rho0, source_base, atol=1e-6)
    assert not torch.allclose(rho1, source_base)
    assert torch.allclose(rho_half, source_base + 0.5 * (rho1 - source_base), atol=1e-6)


def test_source_residual_reliability_gate_is_bounded():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        zero_shot_prior_form="source_residual_prior",
        source_residual_gate="prompt_reliability_scalar",
        source_residual_reliability_dim=5,
    )
    z = torch.randn(4, 16)
    reliability = torch.randn(4, 5)

    gate = model.source_residual_reliability_gate(z, reliability)

    assert gate.shape == (4, 1)
    assert torch.all(gate >= 0.0)
    assert torch.all(gate <= 1.0)


def test_shared_layer_aware_coeff_generator_uses_layer_embedding_and_gets_gradients():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=4,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware",
    )
    x = torch.randn(2, 12, 32, 48)
    z = torch.randn(2, 16)

    logits_b = model.adapter_coefficient_logits(z, "bottleneck")
    logits_d2 = model.adapter_coefficient_logits(z, "dec2")
    pred = model(x, z)
    pred.square().mean().backward()

    assert logits_b.shape == (2, 4)
    assert logits_d2.shape == (2, 4)
    assert not torch.allclose(logits_b, logits_d2)
    assert model.shared_coeff_generator is not None
    assert model.shared_coeff_generator.coeff_head.weight.grad is not None
    assert torch.isfinite(model.shared_coeff_generator.coeff_head.weight.grad).all()


def test_prompt_scalar_reliability_gate_is_bounded_and_init_near_configured_value():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware",
        hyper_reliability_gate="prompt_scalar",
        hyper_reliability_init=0.95,
    )
    z = torch.randn(5, 16)

    gate_b = model.adapter_reliability_gate(z, "bottleneck")
    gate_d1 = model.adapter_reliability_gate(z, "dec1")

    assert gate_b.shape == (5, 1)
    assert torch.all(gate_b >= 0.0)
    assert torch.all(gate_b <= 1.0)
    assert torch.allclose(gate_b, torch.full_like(gate_b, 0.95), atol=1e-4)
    assert torch.allclose(gate_d1, torch.full_like(gate_d1, 0.95), atol=1e-4)


def test_hyperda_can_skip_film_and_exclude_film_from_staged_optimizer():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_enable_film=False,
        hyper_enable_adapters=False,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16)
    x = torch.randn(1, 12, 32, 48)
    z1 = torch.randn(1, 16)
    z2 = torch.randn(1, 16)

    out1 = model(x, z1)
    out2 = model(x, z2)
    metadata = apply_trainable_scope(
        model=model,
        prompt_encoder=prompt_encoder,
        trainable_scope="source_base_frozen_adapter_film",
    )
    trainable = metadata["trainable_parameter_names"]

    assert torch.allclose(out1, out2, atol=1e-6)
    assert not any(name.startswith("model.film") for name in trainable)
    assert not any(name.startswith("model.hyper_adapter") for name in trainable)
    assert any(name.startswith("prompt_encoder.") for name in trainable)


def test_hyperda_can_skip_adapter_residual_and_exclude_adapters_from_staged_optimizer():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_enable_film=False,
        hyper_enable_adapters=False,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16)
    x = torch.randn(1, 12, 32, 48)
    z = torch.randn(1, 16)

    before = model(x, z)
    with torch.no_grad():
        for adapter in [model.hyper_adapter_b, model.hyper_adapter_d2, model.hyper_adapter_d1]:
            adapter.coeff_head.weight.fill_(100.0)
            adapter.coeff_head.bias.fill_(100.0)
            for basis in adapter.bases:
                basis.down.weight.fill_(10.0)
                basis.down.bias.fill_(10.0)
                basis.up.weight.fill_(10.0)
                basis.up.bias.fill_(10.0)
    after = model(x, z)
    metadata = apply_trainable_scope(
        model=model,
        prompt_encoder=prompt_encoder,
        trainable_scope="source_base_frozen_adapter_film",
    )
    trainable = metadata["trainable_parameter_names"]

    assert torch.allclose(before, after, atol=1e-6)
    assert not any(name.startswith("model.hyper_adapter") for name in trainable)
    assert not any(name.startswith("model.shared_coeff_generator") for name in trainable)
    assert not any(name.startswith("model.reliability_gate") for name in trainable)


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


def test_prompt_encoders_return_finite_prompt_vectors():
    """Both source-stage context encoders must preserve the prompt interface."""
    x = torch.randn(2, 12, 8, 10)
    x[0, 0, 0, 0] = float("nan")
    x[1, 3, 1, 1] = float("inf")
    region_ids = torch.tensor([0, 1], dtype=torch.long)
    month = torch.tensor([1, 12], dtype=torch.long)

    for encoder_cls in [RegionPromptEncoder, RobustInputSideDAPromptEncoder]:
        encoder = encoder_cls(num_regions=2, input_channels=12, hidden_dim=16)
        z = encoder(x, region_ids, month)

        assert z.shape == (2, 16)
        assert torch.isfinite(z).all()


def test_robust_prompt_encoder_does_not_use_channel_11_as_mask_semantics():
    """Channel 11 may vary, but robust diagnostics must not use it as a hard mask."""
    encoder = RobustInputSideDAPromptEncoder(num_regions=1, input_channels=12, hidden_dim=8)
    x = torch.ones(1, 12, 4, 4)
    x[:, 11] = 0.0
    region_ids = torch.tensor([0], dtype=torch.long)
    month = torch.tensor([6], dtype=torch.long)

    z_zero_mask_like_channel = encoder(x, region_ids, month)
    stats_zero = encoder._compute_input_stats(x)
    x[:, 11] = 1.0
    z_one_mask_like_channel = encoder(x, region_ids, month)
    stats_one = encoder._compute_input_stats(x)
    diagnostic_idx = encoder.diagnostic_schema.index("base_valid_mask_fraction_diagnostic_only")
    stable_indices = [idx for idx in range(stats_zero.shape[1]) if idx != diagnostic_idx]

    assert torch.isfinite(z_zero_mask_like_channel).all()
    assert torch.isfinite(z_one_mask_like_channel).all()
    assert torch.allclose(stats_zero[:, stable_indices], stats_one[:, stable_indices], atol=1e-6)
    assert stats_zero[0, diagnostic_idx].item() == 0.0
    assert stats_one[0, diagnostic_idx].item() == 1.0


def test_robust_prompt_encoder_uses_da_aware_input_side_diagnostic_schema():
    encoder = RobustInputSideDAPromptEncoder(num_regions=1, input_channels=12, hidden_dim=8)

    assert encoder.diagnostic_schema == [
        "sm_surface_forecast_median",
        "sm_surface_forecast_iqr",
        "sm_rootzone_forecast_median",
        "sm_rootzone_forecast_iqr",
        "soil_temp_layer1_forecast_median",
        "soil_temp_layer1_forecast_iqr",
        "surface_temp_forecast_median",
        "surface_temp_forecast_iqr",
        "mwrtm_vegopacity_median",
        "mwrtm_vegopacity_iqr",
        "tb_h_innovation_median",
        "tb_h_innovation_iqr",
        "tb_v_innovation_median",
        "tb_v_innovation_iqr",
        "tb_obs_hv_contrast_median",
        "tb_obs_hv_contrast_iqr",
        "tb_assim_hv_contrast_median",
        "tb_assim_hv_contrast_iqr",
        "tb_h_obs_error_confidence",
        "tb_v_obs_error_confidence",
        "tb_h_innovation_normalized_abs_median",
        "tb_v_innovation_normalized_abs_median",
        "finite_input_coverage",
        "base_valid_mask_fraction_diagnostic_only",
    ]
    assert len(encoder.diagnostic_schema) == 24


def test_robust_prompt_encoder_computes_da_innovation_and_coverage_features():
    encoder = RobustInputSideDAPromptEncoder(num_regions=1, input_channels=12, hidden_dim=8)
    x = torch.zeros(1, 12, 2, 2)
    x[:, 0] = 0.20
    x[:, 1] = 0.35
    x[:, 2] = 280.0
    x[:, 3] = 295.0
    x[:, 4] = 0.40
    x[:, 5] = torch.tensor([[[10.0, 12.0], [14.0, 16.0]]])
    x[:, 6] = torch.tensor([[[20.0, 22.0], [24.0, 26.0]]])
    x[:, 7] = 2.0
    x[:, 8] = 4.0
    x[:, 9] = torch.tensor([[[8.0, 10.0], [12.0, 14.0]]])
    x[:, 10] = torch.tensor([[[18.0, 20.0], [22.0, 24.0]]])
    x[:, 11] = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    x[:, 0, 0, 0] = float("nan")

    stats = encoder._compute_input_stats(x)
    by_name = dict(zip(encoder.diagnostic_schema, stats[0].tolist()))

    assert stats.shape == (1, 24)
    assert torch.isfinite(stats).all()
    assert by_name["tb_h_innovation_median"] == torch.tensor(2.0).item()
    assert by_name["tb_v_innovation_median"] == torch.tensor(2.0).item()
    assert by_name["tb_obs_hv_contrast_median"] == torch.tensor(10.0).item()
    assert by_name["tb_assim_hv_contrast_median"] == torch.tensor(10.0).item()
    assert by_name["tb_h_obs_error_confidence"] == torch.tensor(1.0 / 3.0).item()
    assert by_name["tb_v_obs_error_confidence"] == torch.tensor(1.0 / 5.0).item()
    assert by_name["tb_h_innovation_normalized_abs_median"] == torch.tensor(2.0 / 3.0).item()
    assert by_name["tb_v_innovation_normalized_abs_median"] == torch.tensor(2.0 / 5.0).item()
    assert by_name["finite_input_coverage"] == torch.tensor(47.0 / 48.0).item()
    assert by_name["base_valid_mask_fraction_diagnostic_only"] == torch.tensor(0.5).item()


def test_robust_prompt_encoder_zero_confidence_when_obs_error_missing():
    encoder = RobustInputSideDAPromptEncoder(num_regions=1, input_channels=12, hidden_dim=8)
    x = torch.ones(1, 12, 2, 2)
    x[:, 7] = float("nan")
    x[:, 8] = float("inf")

    stats = encoder._compute_input_stats(x)
    by_name = dict(zip(encoder.diagnostic_schema, stats[0].tolist()))

    assert torch.isfinite(stats).all()
    assert by_name["tb_h_obs_error_confidence"] == 0.0
    assert by_name["tb_v_obs_error_confidence"] == 0.0


def test_source_base_checkpoint_loads_shared_backbone_into_hyperda(tmp_path):
    source = SmallResUNet(in_channels=12, out_channels=2, width=8)
    with torch.no_grad():
        for idx, param in enumerate(source.parameters()):
            param.fill_(float(idx + 1) / 100.0)
    ckpt_path = tmp_path / "source.pt"
    torch.save(
        {
            "model_state_dict": source.state_dict(),
            "config": {
                "width": 8,
                "target_increment_normalization": True,
                "ch_mean": [float(i) for i in range(12)],
                "ch_std": [float(i + 1) for i in range(12)],
                "inc_mean": [0.1, 0.2],
                "inc_std": [0.01, 0.02],
            },
        },
        ckpt_path,
    )

    target = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
    )

    result = load_source_base_checkpoint_into_hyperda(
        model=target,
        checkpoint_path=str(ckpt_path),
        expected_width=8,
        device=torch.device("cpu"),
    )

    for name in [
        "enc1.net.0.weight",
        "enc2.net.0.weight",
        "enc3.net.0.weight",
        "bottleneck.net.0.weight",
        "dec2.net.0.weight",
        "dec1.net.0.weight",
        "head.weight",
    ]:
        assert torch.allclose(target.state_dict()[name], source.state_dict()[name])

    assert result["checkpoint_path"] == str(ckpt_path.resolve())
    assert result["checkpoint_sha256"]
    assert result["ch_mean"][0] == 0.0
    assert result["ch_std"][0] == 1.0
    assert result["inc_mean"] == [0.1, 0.2]
    assert result["inc_std"] == [0.01, 0.02]


def test_source_base_checkpoint_preflight_rejects_bad_stage1_checkpoint(tmp_path):
    missing = tmp_path / "missing.pt"
    try:
        validate_source_base_checkpoint_for_staged_init(
            checkpoint_path=str(missing),
            expected_width=8,
            require_increment_stats=True,
        )
    except FileNotFoundError as exc:
        assert "init_from_source_base_checkpoint" in str(exc)
    else:
        raise AssertionError("missing staged source checkpoint should fail before run setup")

    bad = tmp_path / "prompt_conditioned.pt"
    torch.save(
        {
            "model_state_dict": SmallResUNet(width=8).state_dict(),
            "prompt_encoder_state_dict": {},
            "config": {
                "model_type": "hyperda_basis_adapter",
                "width": 16,
                "ch_mean": [0.0] * 12,
                "ch_std": [1.0] * 12,
                "inc_mean": [0.0, 0.0],
                "inc_std": [1.0, 1.0],
            },
        },
        bad,
    )

    try:
        validate_source_base_checkpoint_for_staged_init(
            checkpoint_path=str(bad),
            expected_width=8,
            require_increment_stats=True,
        )
    except ValueError as exc:
        message = str(exc)
        assert "source-only" in message or "width mismatch" in message
    else:
        raise AssertionError("non source-only staged source checkpoint should fail before run setup")


def test_source_base_frozen_adapter_film_scope_freezes_base_and_keeps_generation_trainable():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16)

    metadata = apply_trainable_scope(
        model=model,
        prompt_encoder=prompt_encoder,
        trainable_scope="source_base_frozen_adapter_film",
    )
    trainable = trainable_parameter_names(model, prompt_encoder)

    for prefix in [
        "model.enc1.",
        "model.enc2.",
        "model.enc3.",
        "model.bottleneck.",
        "model.dec2.",
        "model.dec1.",
        "model.head.",
    ]:
        assert all(not name.startswith(prefix) for name in trainable)

    for prefix in [
        "prompt_encoder.",
        "model.film1.",
        "model.film2.",
        "model.film3.",
        "model.film_b.",
        "model.hyper_adapter_b.",
        "model.hyper_adapter_d2.",
        "model.hyper_adapter_d1.",
    ]:
        assert any(name.startswith(prefix) for name in trainable)

    optimizer_params = [
        p
        for group in torch.optim.AdamW(
            [p for p in list(model.parameters()) + list(prompt_encoder.parameters()) if p.requires_grad],
            lr=1e-3,
        ).param_groups
        for p in group["params"]
    ]
    frozen_params = [p for p in list(model.parameters()) + list(prompt_encoder.parameters()) if not p.requires_grad]

    assert metadata["trainable_scope"] == "source_base_frozen_adapter_film"
    assert "enc1" in metadata["frozen_source_base_modules"]
    assert metadata["trainable_parameter_count"] == sum(p.numel() for p in optimizer_params)
    assert all(all(p is not frozen for frozen in frozen_params) for p in optimizer_params)


def test_staged_scope_for_shared_coeff_trains_shared_generator_not_per_adapter_heads():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware",
        hyper_reliability_gate="prompt_scalar",
    )
    prompt_encoder = RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16)

    metadata = apply_trainable_scope(
        model=model,
        prompt_encoder=prompt_encoder,
        trainable_scope="source_base_frozen_adapter_film",
    )
    trainable = metadata["trainable_parameter_names"]

    assert any(name.startswith("model.shared_coeff_generator.") for name in trainable)
    assert any(name.startswith("model.reliability_gate.") for name in trainable)
    assert not any(name.startswith("model.hyper_adapter_b.coeff_head.") for name in trainable)
    assert any(name.startswith("model.hyper_adapter_b.bases.") for name in trainable)


def test_staged_scope_for_rank_gated_generator_trains_shared_generator_and_dora_gain():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=5,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated",
        hyper_rank_gate_top_k=2,
        hyper_adapter_param_style="dora_like_gain",
    )
    prompt_encoder = RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16)

    metadata = apply_trainable_scope(
        model=model,
        prompt_encoder=prompt_encoder,
        trainable_scope="source_base_frozen_adapter_film",
    )
    trainable = metadata["trainable_parameter_names"]

    assert any(name.startswith("model.shared_coeff_generator.") for name in trainable)
    assert any(name.startswith("model.shared_coeff_generator.gate_head.") for name in trainable)
    assert not any(name.startswith("model.hyper_adapter_b.coeff_head.") for name in trainable)
    assert any(name == "model.hyper_adapter_b.basis_gain_delta" for name in trainable)


def test_staged_scope_for_stable_rank_gated_generator_trains_shared_generator_and_bounded_gain():
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=5,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
    )
    prompt_encoder = RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16)

    metadata = apply_trainable_scope(
        model=model,
        prompt_encoder=prompt_encoder,
        trainable_scope="source_base_frozen_adapter_film",
    )
    trainable = metadata["trainable_parameter_names"]

    assert any(name.startswith("model.shared_coeff_generator.") for name in trainable)
    assert any(name.startswith("model.shared_coeff_generator.gate_head.") for name in trainable)
    assert not any(name.startswith("model.hyper_adapter_b.coeff_head.") for name in trainable)
    assert any(name == "model.hyper_adapter_b.basis_gain_delta" for name in trainable)
