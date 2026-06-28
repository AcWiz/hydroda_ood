from __future__ import annotations

import numpy as np
import pytest

from hydroda.evaluation.rise_router import (
    ExpertMixturePredictor,
    build_context_descriptor,
    build_posterior_config,
    build_router_prior,
    route_weights_from_prior,
    solve_support_posterior,
    validate_rise_metadata_no_target_eval_selection,
    validate_support_budget,
)


class ConstantPredictor:
    def __init__(self, *, surface: float, rootzone: float):
        self.surface = float(surface)
        self.rootzone = float(rootzone)
        self.method_name = f"constant_{surface}_{rootzone}"

    def predict(self, sample):
        shape = sample["forecast_surface"].shape
        inc_s = np.full(shape, self.surface, dtype=np.float32)
        inc_r = np.full(shape, self.rootzone, dtype=np.float32)
        return {
            "pred_increment_surface": inc_s,
            "pred_increment_rootzone": inc_r,
            "pred_analysis_surface": sample["forecast_surface"] + inc_s,
            "pred_analysis_rootzone": sample["forecast_rootzone"] + inc_r,
        }


def _input_side_sample(month: int = 1, value: float = 1.0):
    x = np.full((12, 2, 2), value, dtype=np.float32)
    return {
        "x": x,
        "forecast_surface": x[0],
        "forecast_rootzone": x[1],
        "region_mask": np.ones((2, 2), dtype=np.float32),
        "base_valid_mask": np.ones((2, 2), dtype=np.float32),
        "latitude_weight": np.ones((2, 2), dtype=np.float32),
        "date_str": f"2015-{month:02d}-01",
        "month": month,
        "time_index": month,
    }


def _eval_sample():
    return {
        "forecast_surface": np.full((2, 2), 10.0, dtype=np.float32),
        "forecast_rootzone": np.full((2, 2), 20.0, dtype=np.float32),
    }


def test_context_descriptor_uses_input_side_fields_only():
    descriptor = build_context_descriptor([_input_side_sample(month=4, value=2.0)])

    assert descriptor.metadata["label_usage"] == "none"
    assert descriptor.metadata["n_samples"] == 1
    assert descriptor.vector.ndim == 1
    assert np.isfinite(descriptor.vector).all()

    poisoned = dict(_input_side_sample())
    poisoned["increment_surface"] = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="input-side"):
        build_context_descriptor([poisoned])


def test_router_prior_training_refuses_target_eval_metrics():
    candidates = [{"expert_id": "A"}, {"expert_id": "B"}]
    episode = {
        "pseudo_target_region_id": "US-R2",
        "descriptor": [0.0, 0.0],
        "metric_split_role": "source_val",
        "expert_metrics": {
            "surface": {"A": 0.20, "B": 0.10},
            "rootzone": {"A": 0.30, "B": 0.40},
        },
    }

    prior = build_router_prior(episodes=[episode], candidates=candidates, temperature=0.1)
    routed = route_weights_from_prior(prior, descriptor=[0.0, 0.0])

    assert prior["method"] == "HyperDA-RISE"
    assert prior["training_label_source"] == "source_val_2022"
    assert routed["weights"]["surface"]["B"] > routed["weights"]["surface"]["A"]
    assert routed["weights"]["rootzone"]["A"] > routed["weights"]["rootzone"]["B"]
    assert routed["no_leakage_declaration"]["target_eval_used_for_router_weights"] is False

    bad_episode = dict(episode)
    bad_episode["metric_split_role"] = "target_eval"
    with pytest.raises(ValueError, match="target_eval"):
        build_router_prior(episodes=[bad_episode], candidates=candidates, temperature=0.1)


def test_k0_posterior_cannot_use_support_labels():
    prior_weights = {
        "surface": {"A": 0.6, "B": 0.4},
        "rootzone": {"A": 0.5, "B": 0.5},
    }
    config = build_posterior_config(
        K=0,
        prior_weights=prior_weights,
        support_samples=[],
        ridge_lambda=1.0,
        temperature=0.1,
    )

    assert config["method_id"] == "hyperda_rise_k0_context_router"
    assert config["support_label_usage"] == "none"
    assert config["updated_parameter_keys"] == []
    assert config["no_leakage_declaration"]["target_support_labels_used"] is False

    with pytest.raises(ValueError, match="K=0"):
        build_posterior_config(
            K=0,
            prior_weights=prior_weights,
            support_samples=[{"true_increment_surface": np.zeros((2, 2), dtype=np.float32)}],
            ridge_lambda=1.0,
            temperature=0.1,
        )


def test_support_budget_counts_da_cycles_not_pixels():
    support = []
    for idx in range(4):
        support.append(
            {
                "date_str": f"2016-01-{idx + 1:02d}",
                "metric_mask": np.ones((16, 16), dtype=np.float32),
            }
        )

    validate_support_budget(support, K=4)

    with pytest.raises(ValueError, match="support DA cycles"):
        validate_support_budget(support + [support[0]], K=4)


def test_support_posterior_updates_only_low_dim_parameters():
    prior_weights = {
        "surface": {"A": 0.5, "B": 0.5},
        "rootzone": {"A": 0.5, "B": 0.5},
    }
    support = [
        {
            "date_str": "2016-01-01",
            "month": 1,
            "metric_mask": np.ones((2, 2), dtype=np.float32),
            "true_increment_surface": np.full((2, 2), 2.0, dtype=np.float32),
            "true_increment_rootzone": np.full((2, 2), 4.0, dtype=np.float32),
            "expert_predictions": {
                "A": {
                    "pred_increment_surface": np.zeros((2, 2), dtype=np.float32),
                    "pred_increment_rootzone": np.zeros((2, 2), dtype=np.float32),
                },
                "B": {
                    "pred_increment_surface": np.full((2, 2), 2.0, dtype=np.float32),
                    "pred_increment_rootzone": np.full((2, 2), 4.0, dtype=np.float32),
                },
            },
        }
    ]

    posterior = solve_support_posterior(
        prior_weights=prior_weights,
        support_samples=support,
        K=4,
        ridge_lambda=0.01,
    )

    allowed = {
        "mixture_logits",
        "mixture_weights",
        "scalar_gain",
        "scalar_bias",
        "monthly_gain",
        "monthly_bias",
    }
    assert set(posterior["updated_parameter_keys"]).issubset(allowed)
    assert "backbone" in posterior["frozen_parameter_keys"]
    assert "hyperda_basis" in posterior["frozen_parameter_keys"]
    assert posterior["weights"]["surface"]["B"] > posterior["weights"]["surface"]["A"]
    assert posterior["weights"]["rootzone"]["B"] > posterior["weights"]["rootzone"]["A"]
    assert posterior["no_leakage_declaration"]["target_eval_used_for_posterior"] is False


def test_mixture_predictor_mixes_variables_and_recomputes_analysis():
    predictor = ExpertMixturePredictor(
        experts={
            "A": ConstantPredictor(surface=0.0, rootzone=8.0),
            "B": ConstantPredictor(surface=2.0, rootzone=4.0),
        },
        weights={
            "surface": {"A": 0.25, "B": 0.75},
            "rootzone": {"A": 0.75, "B": 0.25},
        },
        gain={"surface": 1.0, "rootzone": 1.0},
        bias={"surface": 0.0, "rootzone": 0.0},
    )

    pred = predictor.predict(_eval_sample())

    assert np.allclose(pred["pred_increment_surface"], 1.5)
    assert np.allclose(pred["pred_increment_rootzone"], 7.0)
    assert np.allclose(pred["pred_analysis_surface"], 11.5)
    assert np.allclose(pred["pred_analysis_rootzone"], 27.0)


def test_metadata_validator_rejects_target_eval_selection():
    validate_rise_metadata_no_target_eval_selection(
        {"no_leakage_declaration": {"target_eval_used_for_router_weights": False}}
    )

    with pytest.raises(ValueError, match="target_eval"):
        validate_rise_metadata_no_target_eval_selection(
            {"no_leakage_declaration": {"target_eval_used_for_router_weights": True}}
        )
