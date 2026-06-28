from __future__ import annotations

import torch
from torch import nn

import pytest


def test_mixstyle_preserves_shape_is_eval_identity_and_has_no_parameters():
    from hydroda.models.mixstyle import MixStyle2d

    layer = MixStyle2d(p=1.0, alpha=0.1)
    x = torch.randn(4, 3, 8, 8)

    layer.train()
    y = layer(x)
    assert y.shape == x.shape
    assert sum(p.numel() for p in layer.parameters()) == 0

    layer.eval()
    eval_y = layer(x)
    assert torch.allclose(eval_y, x)


def test_small_resunet_forward_features_exposes_bottleneck_without_changing_forward():
    from hydroda.models.resunet import SmallResUNet

    model = SmallResUNet(in_channels=12, out_channels=2, width=4)
    x = torch.randn(2, 12, 16, 16)

    pred = model(x)
    features = model.forward_features(x, return_layer="bottleneck")

    assert pred.shape == (2, 2, 16, 16)
    assert features.shape[:2] == (2, 16)
    assert features.shape[-2:] == (4, 4)


def test_coral_loss_is_zero_for_identical_features_and_positive_for_shifted_features():
    from hydroda.training.domain_generalization import coral_loss

    features = torch.randn(3, 5, 4, 4)
    assert coral_loss(features, features).item() < 1e-8

    shifted = features * 1.5 + 2.0
    assert coral_loss(features, shifted).item() > 0.0


def test_tca_correlation_alignment_loss_is_zero_for_identical_and_positive_for_changed_correlations():
    from hydroda.training.domain_generalization import tca_correlation_alignment_loss

    features = torch.randn(6, 4, 3, 3)
    assert tca_correlation_alignment_loss(features, features).item() < 1e-8

    correlated = features.clone()
    correlated[:, 1] = 2.0 * correlated[:, 0] + 0.25 * correlated[:, 1] + 1.5
    assert tca_correlation_alignment_loss(features, correlated).item() > 0.0


def test_prediction_consistency_loss_is_zero_for_identical_and_positive_for_shifted_predictions():
    from hydroda.training.domain_generalization import prediction_consistency_loss

    pred = torch.randn(2, 2, 4, 4)
    assert prediction_consistency_loss(pred, pred).item() < 1e-8
    assert prediction_consistency_loss(pred, pred + 0.5).item() > 0.0


def test_domain_loss_variance_is_zero_for_equal_losses_and_positive_for_unequal_losses():
    from hydroda.training.domain_generalization import domain_loss_variance

    equal = {
        "US-R1": torch.tensor(1.0),
        "US-R2": torch.tensor(1.0),
        "US-R3": torch.tensor(1.0),
    }
    assert domain_loss_variance(equal).item() < 1e-8

    unequal = {
        "US-R1": torch.tensor(1.0),
        "US-R2": torch.tensor(2.0),
        "US-R3": torch.tensor(4.0),
    }
    assert domain_loss_variance(unequal).item() > 0.0


def test_unknown_domain_inconsistency_loss_matches_common_region_losses():
    from hydroda.training.domain_generalization import unknown_domain_inconsistency_loss

    clean = {
        "US-R1": torch.tensor(1.0),
        "US-R2": torch.tensor(2.0),
        "US-R3": torch.tensor(3.0),
    }
    perturbed = {
        "US-R1": torch.tensor(2.0),
        "US-R2": torch.tensor(4.0),
        "US-R4": torch.tensor(8.0),
    }

    loss = unknown_domain_inconsistency_loss(clean, perturbed)

    assert torch.isclose(loss, torch.tensor(2.5))


def test_unknown_domain_inconsistency_loss_is_zero_without_two_common_regions():
    from hydroda.training.domain_generalization import unknown_domain_inconsistency_loss

    clean = {"US-R1": torch.tensor(1.0), "US-R2": torch.tensor(2.0)}
    perturbed = {"US-R1": torch.tensor(5.0), "US-R3": torch.tensor(9.0)}

    loss = unknown_domain_inconsistency_loss(clean, perturbed)

    assert loss.item() == pytest.approx(0.0, abs=1e-8)


def test_region_masked_huber_losses_extracts_multiple_domains_from_one_pooled_sample():
    from hydroda.training.domain_generalization import region_masked_huber_losses

    pred = torch.zeros(1, 2, 2, 3)
    target = torch.zeros_like(pred)
    target[:, :, :, 1:] = 1.0
    loss_mask = torch.ones(1, 2, 3)
    region_mask_integer = torch.tensor([[[2, 3, 3], [2, 3, 0]]])

    losses = region_masked_huber_losses(
        pred,
        target,
        loss_mask,
        region_mask_integer,
        active_region_ids=[["US-R2", "US-R3"]],
        delta=1.0,
    )

    assert set(losses) == {"US-R2", "US-R3"}
    assert torch.isclose(losses["US-R2"], torch.tensor(0.0))
    assert torch.isclose(losses["US-R3"], torch.tensor(0.5))


def test_moment_alignment_loss_is_zero_for_identical_domain_moments_and_positive_for_shifted_domain():
    from hydroda.training.domain_generalization import moment_alignment_loss

    base = torch.tensor(
        [
            [[[1.0]], [[2.0]]],
            [[[3.0]], [[4.0]]],
            [[[1.0]], [[2.0]]],
            [[[3.0]], [[4.0]]],
        ]
    )
    domains = ["US-R1", "US-R1", "US-R2", "US-R2"]
    assert moment_alignment_loss(base, domains, order=2).item() < 1e-8

    shifted = base.clone()
    shifted[2:] = shifted[2:] + 2.0
    assert moment_alignment_loss(shifted, domains, order=2).item() > 0.0


def test_region_moment_alignment_loss_uses_spatial_regions_without_expanding_samples():
    from hydroda.training.domain_generalization import region_moment_alignment_loss

    features = torch.zeros(1, 2, 2, 3)
    features[:, 0, :, 1:] = 2.0
    region_mask_integer = torch.tensor([[[2, 3, 3], [2, 3, 0]]])

    loss = region_moment_alignment_loss(
        features,
        region_mask_integer,
        active_region_ids=[["US-R2", "US-R3"]],
        order=1,
    )

    assert loss.item() > 0.0


def test_identify_unlearn_loss_is_zero_without_multiple_domains_and_positive_for_domain_specific_features():
    from hydroda.training.domain_generalization import (
        identify_unlearn_loss,
        inter_domain_variance_channel_scores,
    )

    one_domain_features = torch.randn(3, 4, 2, 2)
    assert identify_unlearn_loss(one_domain_features, ["US-R1", "US-R1", "US-R1"]).item() < 1e-8

    features = torch.zeros(4, 3, 1, 1)
    features[:2, 0] = 1.0
    features[2:, 0] = 4.0
    domains = ["US-R1", "US-R1", "US-R2", "US-R2"]
    scores = inter_domain_variance_channel_scores(features, domains)
    assert int(torch.argmax(scores).item()) == 0
    assert identify_unlearn_loss(features, domains, top_fraction=1 / 3, sample_top_fraction=0.5).item() > 0.0


def test_identify_unlearn_loss_is_zero_when_inter_domain_variance_is_zero():
    from hydroda.training.domain_generalization import identify_unlearn_loss

    features = torch.ones(4, 3, 2, 2) * 7.0
    domains = ["US-R1", "US-R1", "US-R2", "US-R2"]

    loss = identify_unlearn_loss(features, domains)

    assert loss.item() == pytest.approx(0.0, abs=1e-8)


def test_identify_unlearn_loss_is_capped_and_scale_stable_for_large_features():
    from hydroda.training.domain_generalization import identify_unlearn_loss

    features = torch.zeros(4, 3, 1, 1)
    features[:2, 0] = 2_000.0
    features[2:, 0] = 8_000.0
    domains = ["US-R1", "US-R1", "US-R2", "US-R2"]

    loss = identify_unlearn_loss(
        features,
        domains,
        top_fraction=1 / 3,
        sample_top_fraction=0.5,
        score_cap=10.0,
    )
    scaled_loss = identify_unlearn_loss(
        features * 1_000.0,
        domains,
        top_fraction=1 / 3,
        sample_top_fraction=0.5,
        score_cap=10.0,
    )

    assert loss.item() <= 10.0
    assert scaled_loss.item() <= 10.0
    assert scaled_loss.item() == pytest.approx(loss.item(), rel=1e-6)


def test_region_identify_unlearn_loss_uses_spatial_regions_without_expanding_samples():
    from hydroda.training.domain_generalization import (
        region_identify_unlearn_loss,
        region_inter_domain_variance_channel_scores,
    )

    features = torch.zeros(1, 3, 2, 3)
    features[:, 0, :, :1] = 1.0
    features[:, 0, :, 1:] = 4.0
    region_mask_integer = torch.tensor([[[2, 3, 3], [2, 3, 0]]])

    scores = region_inter_domain_variance_channel_scores(
        features,
        region_mask_integer,
        active_region_ids=[["US-R2", "US-R3"]],
    )
    assert int(torch.argmax(scores).item()) == 0
    assert region_identify_unlearn_loss(
        features,
        region_mask_integer,
        active_region_ids=[["US-R2", "US-R3"]],
        top_fraction=1 / 3,
        sample_top_fraction=0.5,
    ).item() > 0.0


def test_region_identify_unlearn_loss_is_capped_and_scale_stable():
    from hydroda.training.domain_generalization import region_identify_unlearn_loss

    features = torch.zeros(1, 3, 2, 3)
    features[:, 0, :, :1] = 2_000.0
    features[:, 0, :, 1:] = 8_000.0
    region_mask_integer = torch.tensor([[[2, 3, 3], [2, 3, 0]]])

    loss = region_identify_unlearn_loss(
        features,
        region_mask_integer,
        active_region_ids=[["US-R2", "US-R3"]],
        top_fraction=1 / 3,
        sample_top_fraction=0.5,
        score_cap=10.0,
    )
    scaled_loss = region_identify_unlearn_loss(
        features * 1_000.0,
        region_mask_integer,
        active_region_ids=[["US-R2", "US-R3"]],
        top_fraction=1 / 3,
        sample_top_fraction=0.5,
        score_cap=10.0,
    )

    assert loss.item() <= 10.0
    assert scaled_loss.item() <= 10.0
    assert scaled_loss.item() == pytest.approx(loss.item(), rel=1e-6)


def test_sam_sharpness_perturbation_restores_original_weights():
    from hydroda.training.domain_generalization import SAMSharpnessPerturbation

    model = nn.Linear(2, 1, bias=False)
    original = model.weight.detach().clone()
    x = torch.ones(3, 2)
    y = model(x).sum()
    y.backward()

    helper = SAMSharpnessPerturbation(model, rho=0.05)
    grad_norm = helper.perturb()
    assert grad_norm.item() > 0.0
    assert not torch.allclose(model.weight, original)

    helper.restore()
    assert torch.allclose(model.weight, original)


def test_collate_preserves_source_region_episode_ids():
    from hydroda.data.dataset import collate_hydroda_samples

    def sample(region_id: str) -> dict:
        return {
            "x": torch.ones(12, 2, 2).numpy(),
            "increment_surface": torch.zeros(2, 2).numpy(),
            "increment_rootzone": torch.zeros(2, 2).numpy(),
            "loss_mask": torch.ones(2, 2).numpy(),
            "latitude_weight": torch.ones(2, 2).numpy(),
            "sample_region_id": region_id,
            "active_region_ids": [region_id],
        }

    batch = collate_hydroda_samples([sample("US-R2"), sample("US-R3")])
    assert batch["sample_region_id"] == ["US-R2", "US-R3"]
    assert batch["active_region_ids"] == [["US-R2"], ["US-R3"]]


class _TinySourceDataset:
    _date_records = [{"date_str": "2015-01-01"}]

    def __len__(self):
        return 1

    def __getitem__(self, idx: int):
        return {
            "x": torch.ones(12, 2, 2).numpy(),
            "increment_surface": torch.zeros(2, 2).numpy(),
            "increment_rootzone": torch.zeros(2, 2).numpy(),
            "loss_mask": torch.ones(2, 2).numpy(),
            "latitude_weight": torch.ones(2, 2).numpy(),
            "sample_region_id": "US-R2",
        }


def test_disam_trainer_does_not_require_target_context_loader(tmp_path):
    from hydroda.models.resunet import SmallResUNet
    from hydroda.training.trainer import Trainer

    trainer = Trainer(
        model=SmallResUNet(in_channels=12, out_channels=2, width=2),
        train_dataset=_TinySourceDataset(),
        max_epochs=1,
        batch_size=1,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        dg_method="disam",
        use_lat_weighted_loss=False,
    )
    assert trainer.target_context_dataset is None
    assert trainer._build_target_context_dataloader() is None


def test_udim_trainer_does_not_require_target_context_loader_and_uses_sam(tmp_path):
    from hydroda.models.resunet import SmallResUNet
    from hydroda.training.trainer import Trainer

    trainer = Trainer(
        model=SmallResUNet(in_channels=12, out_channels=2, width=2),
        train_dataset=_TinySourceDataset(),
        max_epochs=1,
        batch_size=1,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        dg_method="udim",
        udim_rho=0.07,
        udim_lambda=0.2,
        use_lat_weighted_loss=False,
    )

    assert trainer.target_context_dataset is None
    assert trainer._build_target_context_dataloader() is None
    assert trainer.sam_perturbation is not None
    assert trainer.udim_rho == pytest.approx(0.07)
    assert trainer.udim_lambda == pytest.approx(0.2)


def test_udim_two_step_update_detaches_clean_region_losses_after_first_backward(tmp_path):
    from hydroda.training.trainer import Trainer

    class _TinyPredictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(12, 2, kernel_size=1, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    trainer = Trainer(
        model=_TinyPredictor(),
        train_dataset=_TinySourceDataset(),
        max_epochs=1,
        batch_size=1,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        dg_method="udim",
        udim_rho=0.01,
        udim_lambda=0.1,
        use_lat_weighted_loss=False,
    )
    x_norm = torch.randn(1, 12, 2, 3)
    target = torch.zeros(1, 2, 2, 3)
    loss_mask = torch.ones(1, 2, 3)
    region_mask_integer = torch.tensor([[[2, 3, 3], [2, 3, 0]]])

    result = trainer._udim_two_step_update(
        x_norm=x_norm,
        target=target,
        loss_mask=loss_mask,
        latitude_weight=None,
        sample_region_ids=None,
        region_mask_integer=region_mask_integer,
        active_region_ids=[["US-R2", "US-R3"]],
    )

    assert result is not None
    _, losses = result
    assert losses["udim_inconsistency_loss"].item() >= 0.0
    assert losses["udim_sam_grad_norm"].item() > 0.0


def test_iu_trainer_does_not_require_target_context_loader(tmp_path):
    from hydroda.models.resunet import SmallResUNet
    from hydroda.training.trainer import Trainer

    trainer = Trainer(
        model=SmallResUNet(in_channels=12, out_channels=2, width=2),
        train_dataset=_TinySourceDataset(),
        max_epochs=1,
        batch_size=1,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        dg_method="iu",
        use_lat_weighted_loss=False,
    )
    assert trainer.target_context_dataset is None
    assert trainer._build_target_context_dataloader() is None


def test_iu_trainer_adds_domain_specific_feature_penalty_to_total_loss(tmp_path):
    from hydroda.training.trainer import Trainer

    class _FeatureOnlyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.size(0), 2, x.size(2), x.size(3), device=x.device) * self.weight

        def forward_features(self, x: torch.Tensor, return_layer: str = "bottleneck") -> torch.Tensor:
            features = torch.zeros(x.size(0), 2, 2, 3, device=x.device)
            features[:, 0, :, :1] = 1.0
            features[:, 0, :, 1:] = 4.0
            return features * self.weight

    trainer = Trainer(
        model=_FeatureOnlyModel(),
        train_dataset=_TinySourceDataset(),
        max_epochs=1,
        batch_size=1,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        dg_method="iu",
        iu_lambda=0.1,
        iu_top_fraction=0.5,
        iu_sample_top_fraction=1.0,
        use_lat_weighted_loss=False,
    )
    losses = {"total_loss": torch.tensor(2.0)}

    updated = trainer._add_identify_unlearn(
        losses,
        x_norm=torch.ones(1, 12, 2, 3),
        sample_region_ids=None,
        region_mask_integer=torch.tensor([[[2, 3, 3], [2, 3, 0]]]),
        active_region_ids=[["US-R2", "US-R3"]],
    )

    assert updated["iu_unlearn_loss"].item() > 0.0
    assert updated["total_loss"].item() > 2.0


def test_trainer_source_val_loss_selection_ignores_safe_score_for_best_checkpoint(tmp_path):
    from hydroda.training.trainer import Trainer

    class _TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.size(0), 2, x.size(2), x.size(3), device=x.device) * self.weight

    trainer = Trainer(
        model=_TinyModel(),
        train_dataset=_TinySourceDataset(),
        max_epochs=1,
        batch_size=1,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        selection_metric="source_val_loss",
        use_lat_weighted_loss=False,
    )

    first = trainer._resolve_checkpoint_selection(
        avg_loss=0.8,
        source_val_metrics={"source_val_loss": 0.5},
        gain_results={"selection_score": 0.9},
    )
    if first.is_best:
        trainer.best_loss = first.best_metric
    trainer.best_safe_score = max(trainer.best_safe_score, 0.9)

    second = trainer._resolve_checkpoint_selection(
        avg_loss=0.7,
        source_val_metrics={"source_val_loss": 0.4},
        gain_results={"selection_score": 0.1},
    )

    assert first.is_best is True
    assert first.best_metric == pytest.approx(0.5)
    assert second.is_best is True
    assert second.best_metric == pytest.approx(0.4)


def test_target_context_diagnostics_still_require_target_context_loader(tmp_path):
    from hydroda.models.resunet import SmallResUNet
    from hydroda.training.trainer import Trainer

    with pytest.raises(ValueError, match="target_context_dataset"):
        Trainer(
            model=SmallResUNet(in_channels=12, out_channels=2, width=2),
            train_dataset=_TinySourceDataset(),
            max_epochs=1,
            batch_size=1,
            device="cpu",
            checkpoint_dir=str(tmp_path),
            dg_method="tca",
            use_lat_weighted_loss=False,
        )


def test_self_bootstrap_augmentation_preserves_shape_and_has_no_trainable_parameters():
    from hydroda.training.domain_generalization import SelfBootstrapAugmentation

    augment = SelfBootstrapAugmentation(noise_std=0.01, channel_dropout_p=0.25)
    x = torch.randn(4, 12, 8, 8)

    y = augment(x)
    assert y.shape == x.shape
    assert sum(p.numel() for p in augment.parameters()) == 0


class _InputOnlyDataset:
    def __init__(self, split_type: str = "target_context", sample: dict | None = None):
        self.split_type = split_type
        self.sample = sample or {
            "x": torch.ones(12, 2, 2).numpy(),
            "forecast_surface": torch.ones(2, 2).numpy(),
            "forecast_rootzone": torch.ones(2, 2).numpy(),
            "region_mask": torch.ones(2, 2).numpy(),
            "base_valid_mask": torch.ones(2, 2).numpy(),
            "latitude_weight": torch.ones(2, 2).numpy(),
            "date_str": "2015-01-01",
            "month": 1,
        }

    def __len__(self):
        return 1

    def get_input_side_sample(self, idx: int):
        return dict(self.sample)


def test_input_only_target_context_dataset_rejects_labels_and_target_eval():
    from hydroda.training.domain_generalization import InputOnlyTargetContextDataset

    safe = InputOnlyTargetContextDataset(_InputOnlyDataset())
    sample = safe[0]
    forbidden = {
        "increment_surface",
        "increment_rootzone",
        "analysis_surface",
        "analysis_rootzone",
        "target",
        "y",
        "loss_mask",
        "metric_mask",
    }
    assert forbidden.isdisjoint(sample)

    with pytest.raises(ValueError, match="target_context"):
        InputOnlyTargetContextDataset(_InputOnlyDataset(split_type="target_eval"))

    poisoned = dict(_InputOnlyDataset().sample)
    poisoned["increment_surface"] = torch.zeros(2, 2).numpy()
    with pytest.raises(ValueError, match="label-bearing"):
        InputOnlyTargetContextDataset(_InputOnlyDataset(sample=poisoned))[0]


def test_ssa_reg_protocol_validator_refuses_target_eval_or_label_batches():
    from hydroda.training.domain_generalization import validate_ssa_reg_target_context

    validate_ssa_reg_target_context({"x": torch.ones(2, 12, 4, 4)}, split_type="target_context")

    with pytest.raises(ValueError, match="target_eval"):
        validate_ssa_reg_target_context({"x": torch.ones(2, 12, 4, 4)}, split_type="target_eval")

    with pytest.raises(ValueError, match="label-bearing"):
        validate_ssa_reg_target_context(
            {
                "x": torch.ones(2, 12, 4, 4),
                "loss_mask": torch.ones(2, 4, 4),
            },
            split_type="target_context",
        )


@pytest.mark.parametrize(
    ("validator_name", "method_label"),
    [
        ("validate_tca_target_context", "TCA"),
        ("validate_self_bootstrap_target_context", "Self-Bootstrap"),
    ],
)
def test_new_target_context_validators_refuse_target_eval_or_label_batches(validator_name, method_label):
    import hydroda.training.domain_generalization as dg

    validator = getattr(dg, validator_name)
    validator({"x": torch.ones(2, 12, 4, 4)}, split_type="target_context")

    with pytest.raises(ValueError, match="target_eval"):
        validator({"x": torch.ones(2, 12, 4, 4)}, split_type="target_eval")

    with pytest.raises(ValueError, match="label-bearing"):
        validator(
            {
                "x": torch.ones(2, 12, 4, 4),
                "metric_mask": torch.ones(2, 4, 4),
            },
            split_type="target_context",
        )


def test_ssa_reg_subspace_alignment_loss_is_zero_for_identical_and_positive_for_shifted_subspace():
    from hydroda.training.domain_generalization import subspace_alignment_loss

    source = torch.eye(4).view(4, 4, 1, 1)
    assert subspace_alignment_loss(source, source, rank=2).item() < 1e-8

    target = source.clone()
    target[:, 2:] = target[:, 2:] * 3.0 + 1.5
    assert subspace_alignment_loss(source, target, rank=2).item() > 0.0


def test_ssa_reg_state_writes_distinct_checkpoint(tmp_path):
    from hydroda.training.domain_generalization import SSARegState

    state = SSARegState(rank=3, lambda_align=0.05, feature_layer="bottleneck")
    checkpoint_path = tmp_path / "ssa_reg.pt"
    state.save_checkpoint(
        checkpoint_path,
        metadata={"method": "ssa_reg_target_context_subspace_alignment"},
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["tag"] == "ssa_reg"
    assert payload["method"] == "ssa_reg_target_context_subspace_alignment"
    assert payload["ssa_reg_rank"] == 3
    assert payload["ssa_reg_lambda"] == 0.05
    assert payload["ssa_reg_feature_layer"] == "bottleneck"


def test_swad_state_averages_weights_and_writes_distinct_checkpoint(tmp_path):
    from hydroda.training.domain_generalization import SWADState

    model = nn.Linear(2, 1, bias=False)
    swad = SWADState(start_epoch=0, tolerance=0.10, patience=2)

    with torch.no_grad():
        model.weight.fill_(1.0)
    assert swad.update(epoch=0, source_val_metric=1.0, model=model)

    with torch.no_grad():
        model.weight.fill_(3.0)
    assert swad.update(epoch=1, source_val_metric=1.05, model=model)

    state = swad.averaged_state_dict()
    assert torch.allclose(state["weight"], torch.full_like(state["weight"], 2.0))

    checkpoint_path = tmp_path / "swad.pt"
    swad.save_checkpoint(checkpoint_path, metadata={"method": "swad_source_pooled_global_backbone"})
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint_path.name == "swad.pt"
    assert payload["tag"] == "swad"
    assert payload["method"] == "swad_source_pooled_global_backbone"
    assert torch.allclose(payload["model_state_dict"]["weight"], torch.full_like(model.weight, 2.0))
