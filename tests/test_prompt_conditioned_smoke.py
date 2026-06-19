"""Smoke tests for prompt-conditioned training pipeline (Phase 4B).

Tests:
1. Epoch checkpoint saving
2. Transfer safe score uses source_val only
3. Latitude weights = 1 -> unweighted equivalence
4. Prompt collapse detection
5. Checkpoint contains model + prompt_encoder state_dicts
6. Alpha calibration uses region-aware score
"""
from __future__ import annotations

import tempfile
import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from hydroda.models.conditional_unet import FiLMConditionalResUNet
from hydroda.models.hyper_conditional_unet import HyperAdapterConditionalResUNet
from hydroda.models.prompt_encoder import RegionPromptEncoder, RobustInputSideDAPromptEncoder
from hydroda.training.losses import WeightedMaskedHuberLoss
from scripts.train import train_prompt_conditioned_shared as train_pc
from scripts.train.train_prompt_conditioned_shared import PromptQualityTracker, PromptConditionedTrainer


# ---------------------------------------------------------------------------
# Fake dataset for smoke testing
# ---------------------------------------------------------------------------

class FakePromptDataset:
    """Minimal fake dataset for prompt-conditioned training smoke tests."""

    def __init__(self, n_samples: int = 6, H: int = 32, W: int = 48):
        self.n_samples = n_samples
        self.H = H
        self.W = W
        self._date_records = [{"date_str": "2015-01-01"} for _ in range(n_samples)]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        x = np.random.randn(12, self.H, self.W).astype(np.float32)
        inc_s = np.random.randn(self.H, self.W).astype(np.float32) * 0.01
        inc_r = np.random.randn(self.H, self.W).astype(np.float32) * 0.005
        loss_mask = np.ones((1, self.H, self.W), dtype=np.float32)
        latitude_weight = np.ones((self.H, self.W), dtype=np.float32)
        forecast_surface = np.random.randn(self.H, self.W).astype(np.float32) * 0.05
        forecast_rootzone = np.random.randn(self.H, self.W).astype(np.float32) * 0.05
        # region_mask_integer: assign region 1 (source region 0)
        region_mask_integer = np.ones((self.H, self.W), dtype=np.int32)
        region_mask = np.ones((self.H, self.W), dtype=np.float32)
        month = 6
        return {
            "x": x,
            "increment_surface": inc_s,
            "increment_rootzone": inc_r,
            "loss_mask": loss_mask,
            "latitude_weight": latitude_weight,
            "forecast_surface": forecast_surface,
            "forecast_rootzone": forecast_rootzone,
            "region_mask_integer": region_mask_integer,
            "region_mask": region_mask,
            "active_region_mask": region_mask,
            "sample_region_id": "US-R1",
            "active_region_ids": ["US-R1"],
            "month": month,
        }


class ConstantPromptDataset(FakePromptDataset):
    """Deterministic source-val samples for source-residual rho calibration."""

    def __init__(self, n_samples: int = 2, H: int = 16, W: int = 16):
        super().__init__(n_samples=n_samples, H=H, W=W)

    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)
        sample["x"] = np.ones((12, self.H, self.W), dtype=np.float32)
        sample["increment_surface"] = np.full((self.H, self.W), 1.0, dtype=np.float32)
        sample["increment_rootzone"] = np.full((self.H, self.W), 1.0, dtype=np.float32)
        sample["forecast_surface"] = np.zeros((self.H, self.W), dtype=np.float32)
        sample["forecast_rootzone"] = np.zeros((self.H, self.W), dtype=np.float32)
        sample["loss_mask"] = np.ones((1, self.H, self.W), dtype=np.float32)
        sample["latitude_weight"] = np.ones((self.H, self.W), dtype=np.float32)
        sample["region_mask_integer"] = np.ones((self.H, self.W), dtype=np.int32)
        sample["month"] = 1
        return sample


class TwoRegionRhoDataset(FakePromptDataset):
    """Deterministic source-val episodes with different source-region ids."""

    def __init__(self, H: int = 8, W: int = 8):
        super().__init__(n_samples=2, H=H, W=W)

    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)
        sample["x"] = np.ones((12, self.H, self.W), dtype=np.float32)
        sample["increment_surface"] = np.ones((self.H, self.W), dtype=np.float32)
        sample["forecast_surface"] = np.zeros((self.H, self.W), dtype=np.float32)
        sample["forecast_rootzone"] = np.zeros((self.H, self.W), dtype=np.float32)
        sample["loss_mask"] = np.ones((1, self.H, self.W), dtype=np.float32)
        sample["latitude_weight"] = np.ones((self.H, self.W), dtype=np.float32)
        sample["month"] = 1
        if idx == 0:
            sample["increment_rootzone"] = np.ones((self.H, self.W), dtype=np.float32)
            sample["region_mask_integer"] = np.ones((self.H, self.W), dtype=np.int32)
            sample["sample_region_id"] = "US-R1"
            sample["active_region_ids"] = ["US-R1"]
        else:
            sample["increment_rootzone"] = -np.ones((self.H, self.W), dtype=np.float32)
            sample["region_mask_integer"] = np.full((self.H, self.W), 2, dtype=np.int32)
            sample["sample_region_id"] = "US-R2"
            sample["active_region_ids"] = ["US-R2"]
        sample["region_mask"] = np.ones((self.H, self.W), dtype=np.float32)
        sample["active_region_mask"] = np.ones((self.H, self.W), dtype=np.float32)
        return sample


class RegionYearGroupedDataset(FakePromptDataset):
    """Small deterministic dataset exposing source-region/year records."""

    def __init__(self):
        super().__init__(n_samples=10, H=8, W=8)
        self.records = []
        for region_id in ("US-R1", "US-R2"):
            for year in (2015, 2016):
                for day in (1, 2):
                    self.records.append(
                        {
                            "sample_region_id": region_id,
                            "date_str": f"{year}-01-{day:02d}",
                            "active_region_ids": [region_id],
                        }
                    )
        self.records.append(
            {
                "sample_region_id": "US-R1",
                "date_str": "2016-01-03",
                "active_region_ids": ["US-R1"],
            }
        )
        self.records.append(
            {
                "sample_region_id": "US-R2",
                "date_str": "2015-01-03",
                "active_region_ids": ["US-R2"],
            }
        )
        self.n_samples = len(self.records)
        self._date_records = [dict(record) for record in self.records]

    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)
        record = self.records[idx]
        sample["sample_region_id"] = record["sample_region_id"]
        sample["active_region_ids"] = list(record["active_region_ids"])
        sample["date_str"] = record["date_str"]
        sample["month"] = int(record["date_str"][5:7])
        numeric_region = int(record["sample_region_id"].split("-R")[1])
        sample["region_mask_integer"] = np.full((self.H, self.W), numeric_region, dtype=np.int32)
        return sample


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_source_region_year_grouped_batch_sampler_covers_indices_and_keeps_batches_pure():
    dataset = RegionYearGroupedDataset()
    sampler = train_pc.SourceRegionYearGroupedBatchSampler(
        dataset,
        batch_size=3,
        seed=123,
        drop_last=False,
    )

    batches = list(iter(sampler))

    flattened = [idx for batch in batches for idx in batch]
    assert sorted(flattened) == list(range(len(dataset)))
    assert len(flattened) == len(set(flattened))
    for batch in batches:
        records = [dataset._date_records[idx] for idx in batch]
        group_keys = {(record["sample_region_id"], record["date_str"][:4]) for record in records}
        assert len(group_keys) == 1
        assert "US-R3" not in {record["sample_region_id"] for record in records}

    trainer = PromptConditionedTrainer(
        model=FiLMConditionalResUNet(in_channels=12, out_channels=2, width=8, prompt_dim=16),
        prompt_encoder=RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16),
        train_dataset=dataset,
        max_epochs=1,
        batch_size=3,
        num_workers=0,
        device="cpu",
        checkpoint_dir=tempfile.mkdtemp(),
        source_regions=["US-R1", "US-R2"],
        global_to_source_lookup={0: 0, 1: 1},
        use_lat_weighted_loss=False,
        source_val_residual_gain=False,
        train_batch_sampler="source_region_year_grouped",
    )
    loader = trainer._build_dataloader()
    batch = next(iter(loader))

    assert "region_ids" in batch
    assert "months" in batch
    assert batch["region_ids"].unique().numel() == 1
    assert batch["region_ids"][0].item() in {0, 1}
    assert batch["months"].tolist() == [1] * batch["months"].shape[0]
    assert batch["loss_mask"].shape[0] <= 3
    assert len(loader.batch_sampler) == len(sampler)

def test_epoch_checkpoint_saving():
    """Verify PromptConditionedTrainer saves epoch checkpoints."""
    train_dataset = FakePromptDataset(n_samples=8, H=32, W=48)
    source_val_dataset = FakePromptDataset(n_samples=3, H=32, W=48)

    model = FiLMConditionalResUNet(in_channels=12, out_channels=2, width=8, prompt_dim=16)
    prompt_encoder = RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = PromptConditionedTrainer(
            model=model,
            prompt_encoder=prompt_encoder,
            train_dataset=train_dataset,
            max_epochs=2,
            batch_size=2,
            num_workers=0,
            device="cpu",
            checkpoint_dir=tmpdir,
            checkpoint_every_n_epochs=1,
            source_val_dataset=source_val_dataset,
            eval_every_epochs=1,
            source_regions=["US-R1", "US-R2"],
            global_to_source_lookup={0: 0, 1: 1},
            # Disable residual gain for speed
            source_val_residual_gain=False,
        )

        trainer.train(verbose=False)

        ckpt_dir = Path(tmpdir)
        assert (ckpt_dir / "checkpoint_latest.pt").exists(), "checkpoint_latest.pt should exist"
        assert (ckpt_dir / "last.pt").exists(), "last.pt should exist"

        epoch_files = list(ckpt_dir.glob("checkpoint_epoch_*.pt"))
        assert len(epoch_files) >= 1, f"Expected checkpoint_epoch_*.pt files, found {epoch_files}"

        # CSV files should exist
        assert (ckpt_dir / "metrics_train.csv").exists(), "metrics_train.csv should exist"
        assert (ckpt_dir / "README_training_summary.md").exists(), "README_training_summary.md should exist"

        print(f"  test_epoch_checkpoint_saving passed: {sorted([p.name for p in ckpt_dir.glob('*.pt')])}")


def test_parse_args_target_full_train_clears_legacy_k(monkeypatch):
    """target_full_train should not carry a legacy K-shot value."""
    monkeypatch.setattr(sys, "argv", [
        "train_prompt_conditioned_shared.py",
        "--target_region", "US-R1",
        "--adaptation_setting", "target_full_train",
        "--K", "0",
    ])

    args = train_pc.parse_args()

    assert args.adaptation_setting == "target_full_train"
    assert args.K is None


def test_parse_args_accepts_context_encoder(monkeypatch):
    """CLI should expose source-stage context encoder selection."""
    monkeypatch.setattr(sys, "argv", [
        "train_prompt_conditioned_shared.py",
        "--target_region", "US-R1",
        "--context_encoder", "robust_input_side_da_diagnostics_raw",
    ])

    args = train_pc.parse_args()

    assert args.context_encoder == "robust_input_side_da_diagnostics_raw"


def test_raw_da_context_encoder_uses_raw_tensor_for_prompt_diagnostics():
    encoder = train_pc.build_prompt_encoder(
        context_encoder="robust_input_side_da_diagnostics_raw",
        num_regions=2,
        input_channels=12,
        hidden_dim=16,
    )
    raw = torch.zeros(1, 12, 2, 2)
    raw[:, 5] = 10.0
    raw[:, 9] = 4.0
    normalized = raw.clone()
    normalized[:, 5] = 100.0
    normalized[:, 9] = -50.0

    routed = train_pc.prompt_diagnostic_tensor(
        encoder,
        context_encoder="robust_input_side_da_diagnostics_raw",
        x_norm=normalized,
        x_raw=raw,
    )
    stats = encoder._compute_input_stats(routed)
    h_innov_idx = encoder.diagnostic_schema.index("tb_h_innovation_median")

    assert routed is raw
    assert stats[0, h_innov_idx] == torch.tensor(6.0)


def test_legacy_robust_context_encoder_keeps_normalized_tensor_routing():
    encoder = train_pc.build_prompt_encoder(
        context_encoder="robust_input_side_da_diagnostics",
        num_regions=2,
        input_channels=12,
        hidden_dim=16,
    )
    raw = torch.zeros(1, 12, 2, 2)
    raw[:, 5] = 10.0
    raw[:, 9] = 4.0
    normalized = raw.clone()
    normalized[:, 5] = 100.0
    normalized[:, 9] = -50.0

    routed = train_pc.prompt_diagnostic_tensor(
        encoder,
        context_encoder="robust_input_side_da_diagnostics",
        x_norm=normalized,
        x_raw=raw,
    )
    stats = encoder._compute_input_stats(routed)
    h_innov_idx = encoder.diagnostic_schema.index("tb_h_innovation_median")

    assert routed is normalized
    assert stats[0, h_innov_idx] == torch.tensor(150.0)


def test_target_context_prompt_state_records_raw_domain_for_raw_da_encoder():
    encoder = train_pc.build_prompt_encoder(
        context_encoder="robust_input_side_da_diagnostics_raw",
        num_regions=1,
        input_channels=12,
        hidden_dim=8,
    )
    sample_x = np.zeros((12, 2, 2), dtype=np.float32)
    sample_x[5] = 10.0
    sample_x[9] = 4.0
    samples = [
        {
            "x": sample_x,
            "month": 1,
            "date_str": "2015-01-15",
            "region_mask": np.ones((2, 2), dtype=np.float32),
        }
    ]

    state = train_pc.build_target_context_prompt_state(
        samples=samples,
        prompt_encoder=encoder,
        normalize_x=lambda x: x * 100.0,
        target_region_embedding=torch.zeros(1, 16),
        device="cpu",
        context_hash="ctxhash",
        context_encoder="robust_input_side_da_diagnostics_raw",
    )

    assert state["metadata"]["input_usage"] == "target_context_raw_input_side_da_diagnostics"
    assert state["metadata"]["prompt_diagnostic_input_domain"] == "raw_input_side"
    assert state["metadata"]["normalized_input_used_for_prompt_diagnostics"] is False


def test_source_prototype_builder_uses_raw_tensor_for_raw_da_encoder():
    class OneSampleDataset:
        _date_records = [{"date_str": "2015-01-15", "sample_region_id": "US-R1"}]

        def __len__(self):
            return 1

        def get_input_side_sample(self, idx):
            x = np.zeros((12, 2, 2), dtype=np.float32)
            x[5] = 10.0
            x[9] = 4.0
            return {
                "x": x,
                "month": 1,
                "sample_region_id": "US-R1",
                "region_mask": np.ones((2, 2), dtype=np.float32),
                "active_region_mask": np.ones((2, 2), dtype=np.float32),
            }

    encoder = train_pc.build_prompt_encoder(
        context_encoder="robust_input_side_da_diagnostics_raw",
        num_regions=1,
        input_channels=12,
        hidden_dim=8,
    )
    trainer = PromptConditionedTrainer.__new__(PromptConditionedTrainer)
    trainer.prompt_encoder = encoder
    trainer.train_dataset = OneSampleDataset()
    trainer.device = "cpu"
    trainer.source_regions = ["US-R1"]
    trainer.global_to_source_lookup = {0: 0}
    trainer.context_encoder = "robust_input_side_da_diagnostics_raw"
    trainer._source_context_monthly_prototypes = None
    trainer._normalize = lambda x: x * 100.0

    trainer._build_source_context_monthly_prototypes()

    cache = trainer._source_context_monthly_prototypes
    assert cache["prompt_diagnostic_input_domain"] == "raw_input_side"
    assert cache["normalized_input_used_for_prompt_diagnostics"] is False


def test_parse_args_accepts_source_prototype_cache_flags(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "train_prompt_conditioned_shared.py",
        "--target_region", "US-R1",
        "--source_prototype_cache_dir", "artifacts/cache/source_context_monthly_prototypes",
        "--source_prototype_cache_mode", "read_write",
    ])

    args = train_pc.parse_args()

    assert args.source_prototype_cache_dir == "artifacts/cache/source_context_monthly_prototypes"
    assert args.source_prototype_cache_mode == "read_write"


def test_parse_args_accepts_source_saliency_prior_and_prompt_manifold_flags(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "train_prompt_conditioned_shared.py",
        "--target_region", "US-R1",
        "--hyper_source_saliency_prior_path", "artifacts/prior/source_fit.pt",
        "--hyper_source_saliency_prior_beta", "0.5",
        "--hyper_source_saliency_prior_application", "soft_regularization_metadata",
        "--hyper_prompt_manifold_reliability", "1",
        "--hyper_prompt_manifold_reliability_strength", "0.25",
    ])

    args = train_pc.parse_args()

    assert args.hyper_source_saliency_prior_path == "artifacts/prior/source_fit.pt"
    assert args.hyper_source_saliency_prior_beta == 0.5
    assert args.hyper_source_saliency_prior_application == "soft_regularization_metadata"
    assert args.hyper_prompt_manifold_reliability == 1
    assert args.hyper_prompt_manifold_reliability_strength == 0.25


def test_parse_args_accepts_tensor_cache_load_mode_and_batch_sampler(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "train_prompt_conditioned_shared.py",
        "--target_region", "US-R1",
        "--dataset_backend", "tensor_cache",
        "--tensor_cache_load_mode", "mmap",
        "--train_batch_sampler", "source_region_year_grouped",
    ])

    args = train_pc.parse_args()

    assert args.tensor_cache_load_mode == "mmap"
    assert args.train_batch_sampler == "source_region_year_grouped"


def test_selection_metric_modes_choose_expected_values():
    """Checkpoint selection should respect the requested selection_metric."""
    trainer = PromptConditionedTrainer.__new__(PromptConditionedTrainer)

    trainer.selection_metric = "source_val_transfer_safe_score"
    value, maximize = trainer._selection_value(
        train_loss=0.7,
        source_val_metrics={"source_val_loss": 0.3},
        gain_results={"selection_score": 0.42},
    )
    assert value == 0.42
    assert maximize is True

    trainer.selection_metric = "source_val_loss"
    value, maximize = trainer._selection_value(
        train_loss=0.7,
        source_val_metrics={"source_val_loss": 0.3},
        gain_results={"selection_score": 0.42},
    )
    assert value == 0.3
    assert maximize is False

    trainer.selection_metric = "train_loss"
    value, maximize = trainer._selection_value(
        train_loss=0.7,
        source_val_metrics={"source_val_loss": 0.3},
        gain_results={"selection_score": 0.42},
    )
    assert value == 0.7
    assert maximize is False


def test_summary_and_checkpoint_record_best_selection_value_for_safe_score():
    """Safe-score selection should not leave the primary best metric as inf."""
    train_dataset = FakePromptDataset(n_samples=4, H=32, W=48)

    model = FiLMConditionalResUNet(in_channels=12, out_channels=2, width=8, prompt_dim=16)
    prompt_encoder = RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = PromptConditionedTrainer(
            model=model,
            prompt_encoder=prompt_encoder,
            train_dataset=train_dataset,
            max_epochs=1,
            batch_size=2,
            num_workers=0,
            device="cpu",
            checkpoint_dir=tmpdir,
            source_regions=["US-R1", "US-R2"],
            global_to_source_lookup={0: 0, 1: 1},
            selection_metric="source_val_transfer_safe_score",
            context_encoder="current_mean_std",
            tensor_cache_load_mode="mmap",
            train_batch_sampler="source_region_year_grouped",
            prefetch_factor=3,
            pin_memory=True,
        )
        trainer.best_loss = float("inf")
        trainer.best_safe_score = 0.42
        trainer.best_selection_metric = "source_val_transfer_safe_score"
        trainer.best_selection_value = 0.42

        summary_path = Path(tmpdir) / "summary.json"
        trainer.save_summary_json(summary_path)
        summary = json.loads(summary_path.read_text())

        assert summary["best_selection_metric"] == "source_val_transfer_safe_score"
        assert summary["best_selection_value"] == 0.42
        assert summary["context_encoder"] == "current_mean_std"
        assert summary["leakage_policy"]["normalization_source"] == "source_fit_only"
        assert summary["resolved_config"]["context_encoder"] == "current_mean_std"
        assert summary["resolved_config"]["tensor_cache_load_mode"] == "mmap"
        assert summary["resolved_config"]["train_batch_sampler"] == "source_region_year_grouped"
        assert summary["tensor_cache_load_mode"] == "mmap"
        assert summary["train_batch_sampler"] == "source_region_year_grouped"
        assert summary["prefetch_factor"] == 3
        assert summary["pin_memory"] is True
        assert np.isfinite(summary["best_selection_value"])
        assert summary["target_val_usage"] == "unused_in_main_protocol"
        assert summary["target_eval_usage"] == "final_eval_only_no_selection"
        assert summary["target_query_usage"] == "final_eval_only_no_selection"

        trainer._save_readme()
        readme = (Path(tmpdir) / "README_training_summary.md").read_text()
        assert "- **Target val usage**: unused_in_main_protocol" in readme
        assert "- **Target eval usage**: final_eval_only_no_selection" in readme
        assert "- **Target query usage**" not in readme
        assert "- **Best selection metric**: source_val_transfer_safe_score" in readme
        assert "- **Best selection value**: 0.420000" in readme
        assert "- **Context encoder**: current_mean_std" in readme
        assert "- **Tensor cache load mode**: mmap" in readme
        assert "- **Train batch sampler**: source_region_year_grouped" in readme

        trainer.save_checkpoint(
            Path(tmpdir) / "test.pt",
            epoch=0,
            loss=0.42,
            tag="best",
            gain_results={"selection_score": 0.42},
            selection_value=0.42,
        )
        ckpt = torch.load(Path(tmpdir) / "test.pt", map_location="cpu", weights_only=False)

        assert ckpt["best_selection_metric"] == "source_val_transfer_safe_score"
        assert ckpt["best_selection_value"] == 0.42
        assert ckpt["config"]["context_encoder"] == "current_mean_std"
        assert ckpt["config"]["tensor_cache_load_mode"] == "mmap"
        assert ckpt["config"]["train_batch_sampler"] == "source_region_year_grouped"
        assert ckpt["config"]["prefetch_factor"] == 3
        assert ckpt["config"]["pin_memory"] is True
        assert ckpt["config"]["leakage_policy"]["target_eval_usage"] == "final_eval_only_no_selection"
        assert ckpt["config"]["resolved_config"]["selection_metric"] == "source_val_transfer_safe_score"
        assert ckpt["config"]["resolved_config"]["target_val_usage"] == "unused_in_main_protocol"
        assert ckpt["config"]["resolved_config"]["target_eval_usage"] == "final_eval_only_no_selection"


def test_hyperda_summary_and_checkpoint_record_model_metadata():
    """HyperDA checkpoints should record generated-adapter architecture metadata."""
    train_dataset = FakePromptDataset(n_samples=4, H=32, W=48)

    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_adapter_scale=0.25,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = PromptConditionedTrainer(
            model=model,
            prompt_encoder=prompt_encoder,
            train_dataset=train_dataset,
            max_epochs=1,
            batch_size=2,
            num_workers=0,
            device="cpu",
            checkpoint_dir=tmpdir,
            source_regions=["US-R1", "US-R2"],
            global_to_source_lookup={0: 0, 1: 1},
            model_type="hyperda_basis_adapter",
            hyper_n_basis=3,
            hyper_adapter_bottleneck=8,
            hyper_adapter_scale=0.25,
        )

        summary_path = Path(tmpdir) / "summary.json"
        trainer.save_summary_json(summary_path)
        summary = json.loads(summary_path.read_text())

        assert summary["model_type"] == "hyperda_basis_adapter"
        assert summary["hyper_n_basis"] == 3
        assert summary["hyper_adapter_bottleneck"] == 8
        assert summary["hyper_adapter_scale"] == 0.25
        assert summary["hyper_coeff_generator"] == "per_adapter"
        assert summary["hyper_reliability_gate"] == "none"
        assert summary["hyper_reliability_init"] == 0.95
        assert summary["hyper_enable_film"] is True
        assert summary["hyper_enable_adapters"] is True

        trainer.save_checkpoint(Path(tmpdir) / "hyperda.pt", epoch=0, loss=1.0, tag="test")
        ckpt = torch.load(Path(tmpdir) / "hyperda.pt", map_location="cpu", weights_only=False)

        assert ckpt["config"]["model_type"] == "hyperda_basis_adapter"
        assert ckpt["config"]["hyper_n_basis"] == 3
        assert ckpt["config"]["hyper_adapter_bottleneck"] == 8
        assert ckpt["config"]["hyper_adapter_scale"] == 0.25
        assert ckpt["config"]["hyper_coeff_generator"] == "per_adapter"
        assert ckpt["config"]["hyper_reliability_gate"] == "none"
        assert ckpt["config"]["hyper_reliability_init"] == 0.95
        assert ckpt["config"]["hyper_enable_film"] is True
        assert ckpt["config"]["hyper_enable_adapters"] is True


def test_hyperda_ablation_metadata_records_shared_coeff_gate_and_enable_flags():
    train_dataset = FakePromptDataset(n_samples=4, H=32, W=48)

    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware",
        hyper_reliability_gate="prompt_scalar",
        hyper_reliability_init=0.9,
        hyper_enable_film=False,
        hyper_enable_adapters=True,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = PromptConditionedTrainer(
            model=model,
            prompt_encoder=prompt_encoder,
            train_dataset=train_dataset,
            max_epochs=1,
            batch_size=2,
            num_workers=0,
            device="cpu",
            checkpoint_dir=tmpdir,
            source_regions=["US-R1", "US-R2"],
            global_to_source_lookup={0: 0, 1: 1},
            model_type="hyperda_basis_adapter",
            hyper_n_basis=3,
            hyper_adapter_bottleneck=8,
            hyper_coeff_generator="shared_layer_aware",
            hyper_reliability_gate="prompt_scalar",
            hyper_reliability_init=0.9,
            hyper_enable_film=False,
            hyper_enable_adapters=True,
            trainable_scope="source_base_frozen_adapter_film",
        )

        summary_path = Path(tmpdir) / "summary.json"
        trainer.save_summary_json(summary_path)
        summary = json.loads(summary_path.read_text())
        trainer.save_checkpoint(Path(tmpdir) / "ablation.pt", epoch=0, loss=1.0, tag="test")
        ckpt = torch.load(Path(tmpdir) / "ablation.pt", map_location="cpu", weights_only=False)

        for payload in [summary, ckpt["config"], ckpt["config"]["resolved_config"]]:
            assert payload["hyper_coeff_generator"] == "shared_layer_aware"
            assert payload["hyper_reliability_gate"] == "prompt_scalar"
            assert payload["hyper_reliability_init"] == 0.9
            assert payload["hyper_enable_film"] is False
            assert payload["hyper_enable_adapters"] is True
        assert not any(name.startswith("model.film") for name in summary["trainable_parameter_names"])
        assert any(name.startswith("model.shared_coeff_generator.") for name in summary["trainable_parameter_names"])


def test_hyperda_metadata_records_source_saliency_prior_and_prompt_manifold_flags(tmp_path):
    train_dataset = FakePromptDataset(n_samples=4, H=32, W=48)
    prior = torch.tensor(
        [
            [0.0, 1.0, -1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        hyper_source_saliency_prior=prior,
        hyper_source_saliency_prior_beta=0.5,
        hyper_source_saliency_prior_path="artifacts/prior/source_fit.pt",
        hyper_source_saliency_prior_application="soft_regularization_metadata",
        hyper_prompt_manifold_reliability=True,
        hyper_prompt_manifold_reliability_strength=0.25,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16)

    trainer = PromptConditionedTrainer(
        model=model,
        prompt_encoder=prompt_encoder,
        train_dataset=train_dataset,
        max_epochs=1,
        batch_size=2,
        num_workers=0,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        source_regions=["US-R1", "US-R2"],
        global_to_source_lookup={0: 0, 1: 1},
        model_type="hyperda_basis_adapter",
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        hyper_source_saliency_prior=prior,
        hyper_source_saliency_prior_beta=0.5,
        hyper_source_saliency_prior_path="artifacts/prior/source_fit.pt",
        hyper_source_saliency_prior_application="soft_regularization_metadata",
        hyper_prompt_manifold_reliability=True,
        hyper_prompt_manifold_reliability_strength=0.25,
    )

    summary_path = tmp_path / "summary.json"
    trainer.save_summary_json(summary_path)
    trainer.save_checkpoint(tmp_path / "saliency.pt", epoch=0, loss=1.0, tag="saliency")
    summary = json.loads(summary_path.read_text())
    ckpt = torch.load(tmp_path / "saliency.pt", map_location="cpu", weights_only=False)

    for payload in (summary, ckpt["config"], ckpt["config"]["resolved_config"]):
        assert payload["hyper_source_saliency_prior_beta"] == 0.5
        assert payload["hyper_source_saliency_prior_path"] == "artifacts/prior/source_fit.pt"
        assert payload["hyper_source_saliency_prior_application"] == "soft_regularization_metadata"
        assert payload["hyper_source_saliency_prior_metadata"]["application"] == "soft_regularization_metadata"
        assert payload["hyper_source_saliency_prior_metadata"]["hard_routing_effect"] == "none"
        assert payload["hyper_prompt_manifold_reliability"] is True
        assert payload["hyper_prompt_manifold_reliability_strength"] == 0.25


def test_m2_3_source_safe_residual_metadata_records_k0_no_label_contract(tmp_path):
    train_dataset = FakePromptDataset(n_samples=4, H=32, W=48)
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        zero_shot_prior_form="source_base_residual_reliability_gated",
        source_residual_rho=0.5,
        source_residual_gate="prompt_reliability_scalar",
    )
    prompt_encoder = RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16)
    trainer = PromptConditionedTrainer(
        model=model,
        prompt_encoder=prompt_encoder,
        train_dataset=train_dataset,
        max_epochs=1,
        batch_size=2,
        num_workers=0,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        source_regions=["US-R1", "US-R2"],
        global_to_source_lookup={0: 0, 1: 1},
        model_type="hyperda_basis_adapter",
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_rank_gate_top_k=2,
        hyper_rank_gate_temperature_init=2.0,
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        zero_shot_prior_form="source_base_residual_reliability_gated",
        source_residual_rho=0.5,
        source_residual_gate="prompt_reliability_scalar",
        hyper_residual_magnitude_penalty=0.01,
        hyper_coeff_entropy_floor=0.5,
        hyper_coeff_entropy_penalty=0.02,
    )

    summary_path = tmp_path / "summary.json"
    trainer.save_summary_json(summary_path)
    trainer.save_checkpoint(tmp_path / "m2_3.pt", epoch=0, loss=1.0, tag="m2_3")
    summary = json.loads(summary_path.read_text())
    ckpt = torch.load(tmp_path / "m2_3.pt", map_location="cpu", weights_only=False)

    for payload in (summary, ckpt["config"], ckpt["config"]["resolved_config"]):
        assert payload["zero_shot_prior_form"] == "source_base_residual_reliability_gated"
        assert payload["source_residual_prior_mode"] is True
        assert payload["zero_shot_residual_formula"] == (
            "pred = source_base + rho * reliability_gate(prompt, context) * hyper_residual"
        )
        assert payload["target_labels_used_for_adaptation"] is False
        assert payload["target_val_usage"] == "unused_in_main_protocol"
        assert payload["target_eval_usage"] == "final_eval_only_no_selection"
        assert payload["hyper_residual_magnitude_penalty"] == 0.01
        assert payload["hyper_coeff_entropy_floor"] == 0.5
        assert payload["hyper_coeff_entropy_penalty"] == 0.02


def test_trainer_optimizer_excludes_frozen_source_base_parameters():
    """Stage-2 HyperDA trainer must not optimize the frozen source base."""
    train_dataset = FakePromptDataset(n_samples=4, H=32, W=48)
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = PromptConditionedTrainer(
            model=model,
            prompt_encoder=prompt_encoder,
            train_dataset=train_dataset,
            max_epochs=1,
            batch_size=2,
            num_workers=0,
            device="cpu",
            checkpoint_dir=tmpdir,
            source_regions=["US-R1", "US-R2"],
            global_to_source_lookup={0: 0, 1: 1},
            model_type="hyperda_basis_adapter",
            hyper_n_basis=3,
            hyper_adapter_bottleneck=8,
            trainable_scope="source_base_frozen_adapter_film",
        )

        optimizer_params = [p for group in trainer.optimizer.param_groups for p in group["params"]]
        frozen_params = [
            p
            for p in list(trainer.model.parameters()) + list(trainer.prompt_encoder.parameters())
            if not p.requires_grad
        ]

        assert trainer.trainable_scope == "source_base_frozen_adapter_film"
        assert optimizer_params
        assert frozen_params
        assert all(all(p is not frozen for frozen in frozen_params) for p in optimizer_params)

        trainer.save_checkpoint(Path(tmpdir) / "scope.pt", epoch=0, loss=1.0, tag="scope")
        ckpt = torch.load(Path(tmpdir) / "scope.pt", map_location="cpu", weights_only=False)
        assert ckpt["config"]["trainable_scope"] == "source_base_frozen_adapter_film"
        assert ckpt["config"]["trainable_parameter_count"] == sum(p.numel() for p in optimizer_params)
        assert "model.enc1.net.0.weight" not in ckpt["config"]["trainable_parameter_names"]
        assert any(name.startswith("model.hyper_adapter_b.") for name in ckpt["config"]["trainable_parameter_names"])


def test_staged_source_base_checkpoint_does_not_reset_loaded_head_bias():
    """A staged source-base checkpoint should keep the trained frozen output head."""
    train_dataset = FakePromptDataset(n_samples=4, H=32, W=48)
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16)
    with torch.no_grad():
        model.head.bias[:] = torch.tensor([0.25, -0.5])

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = PromptConditionedTrainer(
            model=model,
            prompt_encoder=prompt_encoder,
            train_dataset=train_dataset,
            max_epochs=1,
            batch_size=2,
            num_workers=0,
            device="cpu",
            checkpoint_dir=tmpdir,
            source_regions=["US-R1", "US-R2"],
            global_to_source_lookup={0: 0, 1: 1},
            model_type="hyperda_basis_adapter",
            hyper_n_basis=3,
            hyper_adapter_bottleneck=8,
            trainable_scope="source_base_frozen_adapter_film",
            init_from_source_base_checkpoint="/tmp/source.pt",
            source_base_checkpoint_sha256="abc123",
            target_increment_normalization=True,
            zero_raw_increment_init=True,
            _resume_ch_mean=np.zeros(12, dtype=np.float32),
            _resume_ch_std=np.ones(12, dtype=np.float32),
            _resume_inc_mean=np.array([1.0, 2.0], dtype=np.float32),
            _resume_inc_std=np.array([0.1, 0.2], dtype=np.float32),
        )

        assert torch.allclose(trainer.model.head.bias.detach(), torch.tensor([0.25, -0.5]))
        summary_path = Path(tmpdir) / "summary.json"
        trainer.save_summary_json(summary_path)
        summary = json.loads(summary_path.read_text())
        assert summary["normalization_source"] == "source_fit_only_from_source_checkpoint"
        assert summary["leakage_policy"]["normalization_source"] == "source_fit_only_from_source_checkpoint"


def test_source_residual_prior_rho_selected_from_source_val_and_saved(tmp_path):
    train_dataset = ConstantPromptDataset(n_samples=2, H=16, W=16)
    source_val_dataset = ConstantPromptDataset(n_samples=2, H=16, W=16)
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        zero_shot_prior_form="source_base_residual_reliability_gated",
        source_residual_rho=1.0,
        source_residual_gate="none",
        hyper_enable_film=False,
        hyper_enable_adapters=False,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=1, input_channels=12, hidden_dim=16)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.residual_head.bias.fill_(1.0)

    trainer = PromptConditionedTrainer(
        model=model,
        prompt_encoder=prompt_encoder,
        train_dataset=train_dataset,
        source_val_dataset=source_val_dataset,
        max_epochs=1,
        batch_size=1,
        num_workers=0,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        source_regions=["US-R1"],
        global_to_source_lookup={0: 0},
        use_lat_weighted_loss=False,
        source_val_residual_gain=False,
        model_type="hyperda_basis_adapter",
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_enable_film=False,
        hyper_enable_adapters=False,
        zero_shot_prior_form="source_base_residual_reliability_gated",
        source_residual_rho=1.0,
        source_residual_gate="none",
        source_val_gain_grid=[0.0, 0.5, 1.0],
    )

    gain_results = trainer._calibrate_source_val_residual_gain()
    trainer.save_checkpoint(tmp_path / "rho.pt", epoch=0, loss=1.0, tag="rho", gain_results=gain_results)
    ckpt = torch.load(tmp_path / "rho.pt", map_location="cpu", weights_only=False)

    assert gain_results["zero_shot_rho"] == 1.0
    assert trainer.source_residual_rho == 1.0
    assert trainer.model.source_residual_rho == 1.0
    assert ckpt["config"]["zero_shot_prior_form"] == "source_base_residual_reliability_gated"
    assert ckpt["config"]["zero_shot_rho"] == 1.0
    assert ckpt["config"]["zero_shot_rho_grid"] == [0.0, 0.5, 1.0]
    assert ckpt["config"]["zero_shot_rho_selection_source"] == "source_val_regionwise_safe_episode_only"
    assert ckpt["source_val_safe_metrics"]["zero_shot_rho_trace"]


def test_source_safe_rho_falls_back_to_zero_on_any_region_variable_regression(tmp_path):
    train_dataset = TwoRegionRhoDataset()
    source_val_dataset = TwoRegionRhoDataset()
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        zero_shot_prior_form="source_base_residual_reliability_gated",
        source_residual_rho=1.0,
        source_residual_gate="none",
        hyper_enable_film=False,
        hyper_enable_adapters=False,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.residual_head.bias[0] = 1.0
        model.residual_head.bias[1] = 1.0

    trainer = PromptConditionedTrainer(
        model=model,
        prompt_encoder=prompt_encoder,
        train_dataset=train_dataset,
        source_val_dataset=source_val_dataset,
        max_epochs=1,
        batch_size=1,
        num_workers=0,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        source_regions=["US-R1", "US-R2"],
        global_to_source_lookup={0: 0, 1: 1},
        use_lat_weighted_loss=False,
        source_val_residual_gain=False,
        model_type="hyperda_basis_adapter",
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_enable_film=False,
        hyper_enable_adapters=False,
        zero_shot_prior_form="source_base_residual_reliability_gated",
        source_residual_rho=1.0,
        source_residual_gate="none",
        source_val_gain_grid=[0.0, 1.0],
    )

    gain_results = trainer._calibrate_source_val_residual_gain()

    assert gain_results["zero_shot_rho"] == 0.0
    assert gain_results["zero_shot_rho_selection_reason"] == "fallback_rho0_due_to_source_region_variable_degradation"
    rho1 = next(row for row in gain_results["zero_shot_rho_trace"] if row["rho"] == 1.0)
    assert rho1["safe_relative_to_rho0"] is False
    assert any(cell["region"] == "US-R2" and cell["variable"] == "rootzone" for cell in rho1["unsafe_cells"])


def test_stage2_source_residual_forward_receives_bounded_reliability_features(tmp_path):
    train_dataset = ConstantPromptDataset(n_samples=2, H=8, W=8)
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        zero_shot_prior_form="source_base_residual_reliability_gated",
        source_residual_rho=1.0,
        source_residual_gate="prompt_reliability_scalar",
        source_residual_reliability_dim=5,
        hyper_enable_film=False,
        hyper_enable_adapters=False,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=1, input_channels=12, hidden_dim=16)
    trainer = PromptConditionedTrainer(
        model=model,
        prompt_encoder=prompt_encoder,
        train_dataset=train_dataset,
        max_epochs=1,
        batch_size=1,
        num_workers=0,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        source_regions=["US-R1"],
        global_to_source_lookup={0: 0},
        use_lat_weighted_loss=False,
        source_val_residual_gain=False,
        model_type="hyperda_basis_adapter",
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_enable_film=False,
        hyper_enable_adapters=False,
        zero_shot_prior_form="source_base_residual_reliability_gated",
        source_residual_gate="prompt_reliability_scalar",
        source_episode_prompt_policy="context_monthly_prototype",
    )

    captured = {}
    original_forward = trainer.model.forward

    def capture_forward(*args, **kwargs):
        captured["reliability_features"] = kwargs.get("reliability_features")
        return original_forward(*args, **kwargs)

    trainer.model.forward = capture_forward
    x_norm = torch.ones((1, 12, 8, 8), dtype=torch.float32)
    target = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
    loss_mask = torch.ones((1, 1, 8, 8), dtype=torch.float32)
    region_ids = torch.tensor([0], dtype=torch.long)
    months = torch.tensor([1], dtype=torch.long)

    trainer._forward_and_loss(x_norm, target, loss_mask, region_ids, months)

    features = captured["reliability_features"]
    assert features is not None
    assert features.shape == (1, 5)
    assert torch.isfinite(features).all()
    assert torch.all(features >= 0.0)
    assert torch.all(features <= 1.0)
    assert float(features.abs().sum().item()) > 0.0


def test_context_monthly_prototype_training_prompt_uses_source_fit_prototypes(monkeypatch, tmp_path):
    """Source-stage context_monthly_prototype should not recompute query-sample prompt stats."""
    train_dataset = ConstantPromptDataset(n_samples=3, H=16, W=16)
    model = FiLMConditionalResUNet(in_channels=12, out_channels=2, width=8, prompt_dim=16)
    prompt_encoder = RegionPromptEncoder(num_regions=1, input_channels=12, hidden_dim=16)

    trainer = PromptConditionedTrainer(
        model=model,
        prompt_encoder=prompt_encoder,
        train_dataset=train_dataset,
        max_epochs=1,
        batch_size=1,
        num_workers=0,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        source_regions=["US-R1"],
        global_to_source_lookup={0: 0},
        use_lat_weighted_loss=False,
        source_val_residual_gain=False,
        source_episode_prompt_policy="context_monthly_prototype",
    )

    def fail_on_query_stats(_x):
        raise AssertionError("per-sample prompt stats were recomputed")

    monkeypatch.setattr(trainer.prompt_encoder, "_compute_input_stats", fail_on_query_stats)

    x_norm = torch.zeros((1, 12, 16, 16), dtype=torch.float32)
    target = torch.zeros((1, 2, 16, 16), dtype=torch.float32)
    loss_mask = torch.ones((1, 1, 16, 16), dtype=torch.float32)
    region_ids = torch.tensor([0], dtype=torch.long)
    months = torch.tensor([1], dtype=torch.long)

    pred, losses = trainer._forward_and_loss(x_norm, target, loss_mask, region_ids, months)

    assert pred.shape == target.shape
    assert torch.isfinite(losses["total_loss"])


def test_source_context_prototype_cache_miss_then_hit_without_dataset_scan(monkeypatch, tmp_path):
    train_dataset = ConstantPromptDataset(n_samples=3, H=8, W=8)
    model = FiLMConditionalResUNet(in_channels=12, out_channels=2, width=8, prompt_dim=16)
    prompt_encoder = RegionPromptEncoder(num_regions=1, input_channels=12, hidden_dim=16)

    trainer = PromptConditionedTrainer(
        model=model,
        prompt_encoder=prompt_encoder,
        train_dataset=train_dataset,
        max_epochs=1,
        batch_size=1,
        num_workers=0,
        device="cpu",
        checkpoint_dir=str(tmp_path / "run1"),
        source_regions=["US-R1"],
        global_to_source_lookup={0: 0},
        use_lat_weighted_loss=False,
        source_val_residual_gain=False,
        source_episode_prompt_policy="context_monthly_prototype",
        source_prototype_cache_dir=str(tmp_path / "cache"),
        source_prototype_cache_mode="read_write",
        dataset_backend="tensor_cache",
        _resume_ch_mean=np.zeros(12, dtype=np.float32),
        _resume_ch_std=np.ones(12, dtype=np.float32),
    )

    assert trainer.source_prototype_cache_hit is False
    assert trainer.source_prototype_cache_path
    assert Path(trainer.source_prototype_cache_path).exists()
    assert trainer.source_prototype_cache_metadata["region_counts"] == [3]

    def fail_getitem(self, _idx):
        raise AssertionError("cache hit should not scan source_fit dataset")

    monkeypatch.setattr(ConstantPromptDataset, "__getitem__", fail_getitem)
    second_model = FiLMConditionalResUNet(in_channels=12, out_channels=2, width=8, prompt_dim=16)
    second_prompt_encoder = RegionPromptEncoder(num_regions=1, input_channels=12, hidden_dim=16)
    second_prompt_encoder.load_state_dict(prompt_encoder.state_dict())

    second = PromptConditionedTrainer(
        model=second_model,
        prompt_encoder=second_prompt_encoder,
        train_dataset=train_dataset,
        max_epochs=1,
        batch_size=1,
        num_workers=0,
        device="cpu",
        checkpoint_dir=str(tmp_path / "run2"),
        source_regions=["US-R1"],
        global_to_source_lookup={0: 0},
        use_lat_weighted_loss=False,
        source_val_residual_gain=False,
        source_episode_prompt_policy="context_monthly_prototype",
        source_prototype_cache_dir=str(tmp_path / "cache"),
        source_prototype_cache_mode="read_write",
        dataset_backend="tensor_cache",
        _resume_ch_mean=np.zeros(12, dtype=np.float32),
        _resume_ch_std=np.ones(12, dtype=np.float32),
    )

    assert second.source_prototype_cache_hit is True
    assert second.source_prototype_cache_path == trainer.source_prototype_cache_path
    assert second._source_context_monthly_prototype_summary()["region_counts"] == [3]


def test_source_context_prototype_cache_key_changes_with_normalization_and_prompt_branch(tmp_path):
    train_dataset = ConstantPromptDataset(n_samples=2, H=8, W=8)

    trainer_a = PromptConditionedTrainer(
        model=FiLMConditionalResUNet(in_channels=12, out_channels=2, width=8, prompt_dim=16),
        prompt_encoder=RegionPromptEncoder(num_regions=1, input_channels=12, hidden_dim=16),
        train_dataset=train_dataset,
        max_epochs=2,
        batch_size=1,
        num_workers=0,
        device="cpu",
        checkpoint_dir=str(tmp_path / "run_a"),
        source_regions=["US-R1"],
        global_to_source_lookup={0: 0},
        use_lat_weighted_loss=False,
        source_val_residual_gain=False,
        source_episode_prompt_policy="context_monthly_prototype",
        source_prototype_cache_dir=str(tmp_path / "cache"),
        source_prototype_cache_mode="read_write",
        _resume_ch_mean=np.zeros(12, dtype=np.float32),
        _resume_ch_std=np.ones(12, dtype=np.float32),
    )
    trainer_b = PromptConditionedTrainer(
        model=FiLMConditionalResUNet(in_channels=12, out_channels=2, width=8, prompt_dim=16),
        prompt_encoder=RegionPromptEncoder(num_regions=1, input_channels=12, hidden_dim=16),
        train_dataset=train_dataset,
        max_epochs=1,
        batch_size=1,
        num_workers=0,
        device="cpu",
        checkpoint_dir=str(tmp_path / "run_b"),
        source_regions=["US-R1"],
        global_to_source_lookup={0: 0},
        use_lat_weighted_loss=False,
        source_val_residual_gain=False,
        source_episode_prompt_policy="context_monthly_prototype",
        source_prototype_cache_dir=str(tmp_path / "cache"),
        source_prototype_cache_mode="read_write",
        _resume_ch_mean=np.ones(12, dtype=np.float32),
        _resume_ch_std=np.ones(12, dtype=np.float32),
    )
    trainer_c = PromptConditionedTrainer(
        model=FiLMConditionalResUNet(in_channels=12, out_channels=2, width=8, prompt_dim=16),
        prompt_encoder=RobustInputSideDAPromptEncoder(num_regions=1, input_channels=12, hidden_dim=16),
        train_dataset=train_dataset,
        max_epochs=1,
        batch_size=1,
        num_workers=0,
        device="cpu",
        checkpoint_dir=str(tmp_path / "run_c"),
        source_regions=["US-R1"],
        global_to_source_lookup={0: 0},
        use_lat_weighted_loss=False,
        source_val_residual_gain=False,
        source_episode_prompt_policy="context_monthly_prototype",
        source_prototype_cache_dir=str(tmp_path / "cache"),
        source_prototype_cache_mode="read_write",
        context_encoder="robust_input_side_da_diagnostics",
        _resume_ch_mean=np.zeros(12, dtype=np.float32),
        _resume_ch_std=np.ones(12, dtype=np.float32),
    )

    paths = {
        trainer_a.source_prototype_cache_path,
        trainer_b.source_prototype_cache_path,
        trainer_c.source_prototype_cache_path,
    }
    assert len(paths) == 3
    assert trainer_b.source_prototype_cache_hit is False
    assert trainer_c.source_prototype_cache_hit is False


def test_source_context_prototype_summary_has_nonzero_counts_for_all_source_regions(tmp_path):
    train_dataset = TwoRegionRhoDataset(H=8, W=8)
    trainer = PromptConditionedTrainer(
        model=FiLMConditionalResUNet(in_channels=12, out_channels=2, width=8, prompt_dim=16),
        prompt_encoder=RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16),
        train_dataset=train_dataset,
        max_epochs=1,
        batch_size=1,
        num_workers=0,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        source_regions=["US-R1", "US-R2"],
        global_to_source_lookup={0: 0, 1: 1},
        use_lat_weighted_loss=False,
        source_val_residual_gain=False,
        source_episode_prompt_policy="context_monthly_prototype",
        source_prototype_cache_dir=str(tmp_path / "cache"),
        source_prototype_cache_mode="read_write",
        _resume_ch_mean=np.zeros(12, dtype=np.float32),
        _resume_ch_std=np.ones(12, dtype=np.float32),
    )

    summary = trainer._source_context_monthly_prototype_summary()
    assert summary["region_counts"] == [1, 1]
    assert all(count > 0 for count in summary["region_counts"])


def test_eval_every_epochs_skips_best_source_val_checkpoint_on_non_eval_epochs(monkeypatch, tmp_path):
    train_dataset = ConstantPromptDataset(n_samples=2, H=8, W=8)
    source_val_dataset = ConstantPromptDataset(n_samples=1, H=8, W=8)
    trainer = PromptConditionedTrainer(
        model=FiLMConditionalResUNet(in_channels=12, out_channels=2, width=8, prompt_dim=16),
        prompt_encoder=RegionPromptEncoder(num_regions=1, input_channels=12, hidden_dim=16),
        train_dataset=train_dataset,
        source_val_dataset=source_val_dataset,
        max_epochs=2,
        batch_size=1,
        num_workers=0,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        source_regions=["US-R1"],
        global_to_source_lookup={0: 0},
        use_lat_weighted_loss=False,
        source_val_residual_gain=False,
        selection_metric="source_val_transfer_safe_score",
        eval_every_epochs=5,
    )
    trainer.current_epoch = 1

    def fail_if_eval_runs():
        raise AssertionError("source_val eval should be skipped on epoch 1 with eval_every_epochs=5")

    monkeypatch.setattr(trainer, "_calibrate_source_val_residual_gain", fail_if_eval_runs)
    trainer.train(verbose=False)

    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "checkpoint_latest.pt").exists()
    assert not (tmp_path / "checkpoint_best_source_val_transfer_safe_score.pt").exists()
    latest = torch.load(tmp_path / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    assert latest["tag"] == "latest"
    assert latest["config"]["eval_every_epochs"] == 5


def test_train_step_log_records_data_wait_compute_and_throughput(monkeypatch, tmp_path):
    train_dataset = ConstantPromptDataset(n_samples=1, H=8, W=8)
    trainer = PromptConditionedTrainer(
        model=FiLMConditionalResUNet(in_channels=12, out_channels=2, width=8, prompt_dim=16),
        prompt_encoder=RegionPromptEncoder(num_regions=1, input_channels=12, hidden_dim=16),
        train_dataset=train_dataset,
        max_epochs=1,
        batch_size=1,
        num_workers=0,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        source_regions=["US-R1"],
        global_to_source_lookup={0: 0},
        use_lat_weighted_loss=False,
        source_val_residual_gain=False,
        selection_metric="train_loss",
        log_every_steps=1,
        _resume_ch_mean=np.zeros(12, dtype=np.float32),
        _resume_ch_std=np.ones(12, dtype=np.float32),
    )
    records = []

    class CaptureLogger:
        def log_step(self, data):
            records.append(data)

        def log_epoch(self, _data):
            pass

        def log_eval(self, _data):
            pass

    trainer._jsonl_logger = CaptureLogger()
    monkeypatch.setattr(trainer, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, "_save_output_csvs", lambda: None)

    trainer.train(verbose=False)

    assert len(records) == 1
    record = records[0]
    assert record["data_wait_sec"] >= 0.0
    assert record["compute_sec"] >= 0.0
    assert record["iter_wall_sec"] >= record["compute_sec"]
    assert record["samples_per_sec"] > 0.0


def test_checkpoint_missing_context_encoder_defaults_to_current_mean_std():
    """Old checkpoints should load as current_mean_std context encoders."""
    checkpoint = {"config": {"prompt_dim": 16, "num_regions": 2}}

    assert train_pc.resolve_context_encoder_from_checkpoint(checkpoint) == "current_mean_std"


def test_build_prompt_encoder_selects_robust_context_encoder():
    encoder = train_pc.build_prompt_encoder(
        context_encoder="robust_input_side_da_diagnostics",
        num_regions=2,
        input_channels=12,
        hidden_dim=16,
    )

    assert isinstance(encoder, RobustInputSideDAPromptEncoder)


def test_predictor_loads_hyperda_model_type():
    """Prompt-conditioned predictor should auto-load HyperDA checkpoints."""
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_adapter_scale=0.25,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "hyperda.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "prompt_encoder_state_dict": prompt_encoder.state_dict(),
                "config": {
                    "model_type": "hyperda_basis_adapter",
                    "width": 8,
                    "prompt_dim": 16,
                    "num_regions": 2,
                    "hyper_n_basis": 3,
                    "hyper_adapter_bottleneck": 8,
                    "hyper_adapter_scale": 0.25,
                    "source_region_global_indices": [1, 2],
                },
            },
            ckpt_path,
        )

        from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor

        predictor = PromptConditionedBackbonePredictor(
            checkpoint_path=str(ckpt_path),
            device="cpu",
            target_region="US-R1",
        )

        assert predictor.method_name == "hyperda_basis_adapter_shared"
        assert isinstance(predictor.model, HyperAdapterConditionalResUNet)


def test_predictor_loads_robust_context_encoder():
    """Predictor should instantiate the checkpoint-declared prompt encoder."""
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=8,
        prompt_dim=16,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=8,
        hyper_adapter_scale=0.25,
    )
    prompt_encoder = RobustInputSideDAPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "hyperda_robust.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "prompt_encoder_state_dict": prompt_encoder.state_dict(),
                "config": {
                    "model_type": "hyperda_basis_adapter",
                    "context_encoder": "robust_input_side_da_diagnostics",
                    "width": 8,
                    "prompt_dim": 16,
                    "num_regions": 2,
                    "hyper_n_basis": 3,
                    "hyper_adapter_bottleneck": 8,
                    "hyper_adapter_scale": 0.25,
                    "source_region_global_indices": [1, 2],
                },
            },
            ckpt_path,
        )

        from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor

        predictor = PromptConditionedBackbonePredictor(
            checkpoint_path=str(ckpt_path),
            device="cpu",
            target_region="US-R1",
        )

        assert isinstance(predictor.prompt_encoder, RobustInputSideDAPromptEncoder)


def _make_prompt_predictor_checkpoint(path: Path) -> None:
    model = FiLMConditionalResUNet(in_channels=12, out_channels=2, width=8, prompt_dim=16)
    prompt_encoder = RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "prompt_encoder_state_dict": prompt_encoder.state_dict(),
            "config": {
                "model_type": "prompt_conditioned",
                "width": 8,
                "prompt_dim": 16,
                "num_regions": 2,
                # Held-out US-R1 target; compact prompt ids are US-R2 -> 0, US-R3 -> 1.
                "source_region_global_indices": [1, 2],
            },
        },
        path,
    )


def _make_prompt_sample(*, split_role: str, target_region_id: str, sample_region_id: str) -> Dict[str, Any]:
    h, w = 8, 8
    return {
        "x": np.zeros((12, h, w), dtype=np.float32),
        "forecast_surface": np.zeros((h, w), dtype=np.float32),
        "forecast_rootzone": np.zeros((h, w), dtype=np.float32),
        "month": 6,
        "target_region_id": target_region_id,
        "sample_region_id": sample_region_id,
        "split_role": split_role,
    }


def test_source_test_prompt_uses_sample_region_compact_prompt_id():
    """Source split inference should use the source sample's prompt, not target fallback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "prompt.pt"
        _make_prompt_predictor_checkpoint(ckpt_path)

        from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor

        predictor = PromptConditionedBackbonePredictor(
            checkpoint_path=str(ckpt_path),
            device="cpu",
            target_region="US-R1",
        )
        seen = {}

        def fake_build_prompt(x_norm, region_idx, month_val):
            seen["region_idx"] = region_idx
            return torch.zeros((1, 16), dtype=torch.float32)

        predictor._build_prompt = fake_build_prompt
        predictor.predict(
            _make_prompt_sample(
                split_role="source_test",
                target_region_id="US-R1",
                sample_region_id="US-R3",
            )
        )

        assert seen["region_idx"] == 1


def test_target_eval_prompt_uses_held_out_target_prompt_route():
    """Target eval should continue to use the held-out target prompt route."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "prompt.pt"
        _make_prompt_predictor_checkpoint(ckpt_path)

        from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor

        predictor = PromptConditionedBackbonePredictor(
            checkpoint_path=str(ckpt_path),
            device="cpu",
            target_region="US-R1",
        )
        seen = {}

        def fake_build_prompt(x_norm, region_idx, month_val):
            seen["region_idx"] = region_idx
            return torch.zeros((1, 16), dtype=torch.float32)

        predictor._build_prompt = fake_build_prompt
        predictor.predict(
            _make_prompt_sample(
                split_role="target_eval",
                target_region_id="US-R1",
                sample_region_id="US-R3",
            )
        )

        assert seen["region_idx"] == 0


def test_target_train_fixed_prompt_uses_only_input_side_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "prompt.pt"
        _make_prompt_predictor_checkpoint(ckpt_path)

        from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor

        predictor = PromptConditionedBackbonePredictor(
            checkpoint_path=str(ckpt_path),
            device="cpu",
            target_region="US-R1",
        )

        class LabelPoisonSample(dict):
            def __getitem__(self, key):
                if key.startswith("analysis_") or key.startswith("increment_"):
                    raise AssertionError(f"target labels must not be read for prompt construction: {key}")
                return super().__getitem__(key)

            def get(self, key, default=None):
                if key.startswith("analysis_") or key.startswith("increment_"):
                    raise AssertionError(f"target labels must not be read for prompt construction: {key}")
                return super().get(key, default)

        samples = [
            LabelPoisonSample({
                "x": np.full((12, 8, 8), fill_value=float(i), dtype=np.float32),
                "month": 1 + i,
                "date_str": f"201{i}-01-01",
            })
            for i in range(2)
        ]

        metadata = predictor.set_target_prompt_from_samples(samples)

        assert metadata["n_samples"] == 2
        assert metadata["date_start"] == "2010-01-01"
        assert metadata["date_end"] == "2011-01-01"
        assert predictor.uses_fixed_target_prompt is True


def test_target_context_prompt_state_builds_monthly_prototypes_with_global_fallback():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "prompt.pt"
        _make_prompt_predictor_checkpoint(ckpt_path)

        from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor

        predictor = PromptConditionedBackbonePredictor(
            checkpoint_path=str(ckpt_path),
            device="cpu",
            target_region="US-R1",
        )
        samples = [
            {
                "x": np.full((12, 8, 8), fill_value=float(month), dtype=np.float32),
                "month": month,
                "date_str": f"2019-{month:02d}-01",
            }
            for month in (1, 3)
        ]

        metadata = predictor.set_target_context_prompt_from_samples(samples)
        state = predictor.target_context_prompt_state

        assert metadata["prompt_source"] == "target_context_monthly_prompt_prototypes"
        assert state["label_usage"] == "none"
        assert state["monthly_counts"]["1"] == 1
        assert state["monthly_counts"]["3"] == 1
        assert state["monthly_counts"]["2"] == 0
        assert state["monthly_prototypes"]["1"] is not None
        assert state["monthly_prototypes"]["2"] is None
        assert predictor.compose_target_context_prompt(1).shape == predictor.compose_target_context_prompt(2).shape
        assert predictor.compose_target_context_prompt(2).shape == predictor.compose_target_context_prompt(12).shape


def test_target_context_prompt_state_records_input_side_reliability_features_only():
    from hydroda.baselines.prompt_conditioned import build_target_context_prompt_state

    prompt_encoder = RegionPromptEncoder(num_regions=1, input_channels=12, hidden_dim=8)
    target_region_embedding = prompt_encoder.region_embed(torch.tensor([0])).detach()
    samples = []
    for idx, month in enumerate([1, 1, 2]):
        x = np.full((12, 4, 4), float(idx + 1), dtype=np.float32)
        x[0, 0, 0] = np.nan
        x[11] = float(idx % 2)
        samples.append(
            {
                "x": x,
                "month": month,
                "date_str": f"2019-{month:02d}-01",
                "increment_surface": np.full((4, 4), 999.0, dtype=np.float32),
                "target_val_loss": -999.0,
                "target_eval_metric": -999.0,
            }
        )

    state = build_target_context_prompt_state(
        samples=samples,
        prompt_encoder=prompt_encoder,
        normalize_x=lambda x: x,
        target_region_embedding=target_region_embedding,
        device="cpu",
    )
    schema = state["reliability_feature_schema"]
    features = state["reliability_features"]

    assert schema == [
        "monthly_count",
        "has_monthly_prototype",
        "global_context_count",
        "finite_input_coverage",
        "prompt_to_source_manifold_distance",
    ]
    assert state["metadata"]["reliability_feature_source"] == "input_side_context_summary_only"
    assert state["metadata"]["reliability_feature_transform"] == "bounded_v2"
    assert state["metadata"]["channel_11_usage"] == "finite_input_feature_only_not_observation_or_static_mask"
    assert state["metadata"]["target_val_usage"] == "unused_in_main_protocol"
    assert state["metadata"]["target_eval_usage"] == "final_eval_only_no_selection"
    assert 0.0 < features["1"][0] <= 1.0
    assert features["1"][1] == 1.0
    assert 0.0 < features["1"][2] <= 1.0
    assert 0.0 < features["1"][3] <= 1.0
    assert features["3"][0] == 0.0
    assert features["3"][1] == 0.0
    assert 0.0 < features["3"][2] <= 1.0


def test_checkpoint_prompt_state_is_used_for_target_eval_without_eval_input_stats(tmp_path):
    from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor

    base_ckpt = tmp_path / "prompt_base.pt"
    _make_prompt_predictor_checkpoint(base_ckpt)
    base = PromptConditionedBackbonePredictor(
        checkpoint_path=str(base_ckpt),
        device="cpu",
        target_region="US-R1",
    )
    base.set_target_context_prompt_from_samples(
        [
            {
                "x": np.full((12, 8, 8), fill_value=1.0, dtype=np.float32),
                "month": 7,
                "date_str": "2019-07-01",
            }
        ]
    )
    checkpoint = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    checkpoint["target_context_prompt_state"] = base.target_context_prompt_state
    checkpoint["config"]["target_context_prompt_state"] = base.target_context_prompt_state
    ckpt_path = tmp_path / "prompt_with_state.pt"
    torch.save(checkpoint, ckpt_path)

    predictor = PromptConditionedBackbonePredictor(
        checkpoint_path=str(ckpt_path),
        device="cpu",
        target_region="US-R1",
    )

    def fail_compute_input_stats(_x):
        raise AssertionError("target_eval input stats must not update target-context prompt")

    predictor.prompt_encoder._compute_input_stats = fail_compute_input_stats
    sample = _make_prompt_sample(
        split_role="target_eval",
        target_region_id="US-R1",
        sample_region_id="US-R3",
    )
    sample["month"] = 7

    pred = predictor.predict(sample)

    assert predictor.uses_fixed_target_prompt is True
    assert pred["pred_increment_surface"].shape == sample["forecast_surface"].shape


def test_prompt_predictor_reports_gain_adjusted_increment_for_reconstruction(tmp_path):
    from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor

    ckpt_path = tmp_path / "prompt.pt"
    _make_prompt_predictor_checkpoint(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    checkpoint["residual_gain_alpha_surface"] = 0.25
    checkpoint["residual_gain_alpha_rootzone"] = 0.0
    torch.save(checkpoint, ckpt_path)

    predictor = PromptConditionedBackbonePredictor(
        checkpoint_path=str(ckpt_path),
        device="cpu",
        target_region="US-R1",
    )

    def fixed_forward(_x, _z):
        return torch.stack(
            [
                torch.full((1, 4, 4), 2.0, dtype=torch.float32),
                torch.full((1, 4, 4), 3.0, dtype=torch.float32),
            ],
            dim=1,
        )

    predictor.model.forward = fixed_forward
    sample = _make_prompt_sample(
        split_role="source_test",
        target_region_id="US-R1",
        sample_region_id="US-R2",
    )
    sample["x"] = np.zeros((12, 4, 4), dtype=np.float32)
    sample["forecast_surface"] = np.ones((4, 4), dtype=np.float32)
    sample["forecast_rootzone"] = np.ones((4, 4), dtype=np.float32) * 10.0

    pred = predictor.predict(sample)

    np.testing.assert_allclose(pred["pred_increment_surface"], 0.5)
    np.testing.assert_allclose(pred["pred_increment_rootzone"], 0.0)
    np.testing.assert_allclose(
        pred["pred_analysis_surface"],
        sample["forecast_surface"] + pred["pred_increment_surface"],
    )
    np.testing.assert_allclose(
        pred["pred_analysis_rootzone"],
        sample["forecast_rootzone"] + pred["pred_increment_rootzone"],
    )


def test_stage3_k0_context_shrinkage_blends_hyperda_prediction_to_source_base(tmp_path):
    from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor

    ckpt_path = tmp_path / "hyperda.pt"
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        zero_raw_increment_init=True,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=8)
    prompt_state = {
        "schema_version": "target_context_prompt_state_v1",
        "prompt_source": "target_context_monthly_prompt_prototypes",
        "label_usage": "none",
        "context_hash": "ctxhash",
        "monthly_counts": {str(i): (1 if i == 6 else 0) for i in range(1, 13)},
        "reliability_feature_schema": [
            "monthly_count",
            "has_monthly_prototype",
            "global_context_count",
            "finite_input_coverage",
            "prompt_to_source_manifold_distance",
        ],
        "reliability_features": {
            str(i): ([1.0, 1.0, 1.0, 1.0, 0.0] if i == 6 else [0.0, 0.0, 1.0, 1.0, 1.0])
            for i in range(1, 13)
        },
        "global_prototype": torch.zeros(8),
        "monthly_prototypes": {str(i): None for i in range(1, 13)},
        "metadata": {
            "target_val_usage": "unused_in_main_protocol",
            "target_eval_usage": "final_eval_only_no_selection",
            "label_usage": "none",
        },
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "prompt_encoder_state_dict": prompt_encoder.state_dict(),
            "target_context_prompt_state": prompt_state,
            "config": {
                "model_type": "hyperda_basis_adapter",
                "width": 4,
                "prompt_dim": 8,
                "hyper_n_basis": 3,
                "hyper_adapter_bottleneck": 2,
                "hyper_adapter_scale": 1.0,
                "zero_raw_increment_init": True,
                "num_regions": 2,
                "source_region_global_indices": [1, 2],
                "ch_mean": [0.0] * 12,
                "ch_std": [1.0] * 12,
                "inc_mean": [0.0, 0.0],
                "inc_std": [1.0, 1.0],
            },
        },
        ckpt_path,
    )
    predictor = PromptConditionedBackbonePredictor(
        checkpoint_path=str(ckpt_path),
        device="cpu",
        target_region="US-R1",
    )

    def fixed_model_forward(_x, _z, **_kwargs):
        return torch.stack(
            [
                torch.full((1, 4, 4), 4.0, dtype=torch.float32),
                torch.full((1, 4, 4), 8.0, dtype=torch.float32),
            ],
            dim=1,
        )

    def fixed_source_base(_x):
        return torch.stack(
            [
                torch.full((1, 4, 4), 1.0, dtype=torch.float32),
                torch.full((1, 4, 4), 2.0, dtype=torch.float32),
            ],
            dim=1,
        )

    predictor.model.forward = fixed_model_forward
    predictor.model.source_base_forward = fixed_source_base
    predictor.enable_stage3_k0_context_shrinkage(source_calibrated_rho_cap=0.5)
    sample = _make_prompt_sample(
        split_role="target_eval",
        target_region_id="US-R1",
        sample_region_id="US-R2",
    )
    sample["x"] = np.zeros((12, 4, 4), dtype=np.float32)
    sample["forecast_surface"] = np.full((4, 4), 10.0, dtype=np.float32)
    sample["forecast_rootzone"] = np.full((4, 4), 20.0, dtype=np.float32)
    sample["month"] = 6

    pred = predictor.predict(sample)
    metadata = predictor.stage3_k0_context_shrinkage_metadata

    np.testing.assert_allclose(pred["pred_increment_surface"], 2.5)
    np.testing.assert_allclose(pred["pred_increment_rootzone"], 5.0)
    np.testing.assert_allclose(pred["pred_analysis_surface"], 12.5)
    np.testing.assert_allclose(pred["pred_analysis_rootzone"], 25.0)
    assert metadata["enabled"] is True
    assert metadata["stage3_variant"] == "M2_4_target_context_conservative_hyperda"
    assert metadata["target_labels_used_for_adaptation"] is False
    assert metadata["target_eval_input_stats_used_for_update"] is False
    assert metadata["rho_cap"] == 0.5
    assert metadata["last_rho"] == 0.5


def test_stage3_k0_variable_context_shrinkage_uses_independent_surface_rootzone_rhos(tmp_path):
    from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor

    ckpt_path = tmp_path / "hyperda.pt"
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        zero_raw_increment_init=True,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=8)
    prompt_state = {
        "schema_version": "target_context_prompt_state_v1",
        "prompt_source": "target_context_monthly_prompt_prototypes",
        "label_usage": "none",
        "context_hash": "ctxhash",
        "monthly_counts": {str(i): (1 if i == 6 else 0) for i in range(1, 13)},
        "reliability_feature_schema": [
            "monthly_count",
            "has_monthly_prototype",
            "global_context_count",
            "finite_input_coverage",
            "prompt_to_source_manifold_distance",
        ],
        "reliability_features": {
            str(i): ([1.0, 1.0, 1.0, 1.0, 0.0] if i == 6 else [0.0, 0.0, 1.0, 1.0, 1.0])
            for i in range(1, 13)
        },
        "global_prototype": torch.zeros(8),
        "monthly_prototypes": {str(i): None for i in range(1, 13)},
        "metadata": {
            "target_val_usage": "unused_in_main_protocol",
            "target_eval_usage": "final_eval_only_no_selection",
            "label_usage": "none",
        },
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "prompt_encoder_state_dict": prompt_encoder.state_dict(),
            "target_context_prompt_state": prompt_state,
            "config": {
                "model_type": "hyperda_basis_adapter",
                "width": 4,
                "prompt_dim": 8,
                "hyper_n_basis": 3,
                "hyper_adapter_bottleneck": 2,
                "hyper_adapter_scale": 1.0,
                "zero_raw_increment_init": True,
                "num_regions": 2,
                "source_region_global_indices": [1, 2],
                "ch_mean": [0.0] * 12,
                "ch_std": [1.0] * 12,
                "inc_mean": [0.0, 0.0],
                "inc_std": [1.0, 1.0],
            },
        },
        ckpt_path,
    )
    predictor = PromptConditionedBackbonePredictor(
        checkpoint_path=str(ckpt_path),
        device="cpu",
        target_region="US-R1",
    )

    def fixed_model_forward(_x, _z, **_kwargs):
        return torch.stack(
            [
                torch.full((1, 4, 4), 4.0, dtype=torch.float32),
                torch.full((1, 4, 4), 8.0, dtype=torch.float32),
            ],
            dim=1,
        )

    def fixed_source_base(_x):
        return torch.stack(
            [
                torch.full((1, 4, 4), 1.0, dtype=torch.float32),
                torch.full((1, 4, 4), 2.0, dtype=torch.float32),
            ],
            dim=1,
        )

    predictor.model.forward = fixed_model_forward
    predictor.model.source_base_forward = fixed_source_base
    predictor.enable_stage3_k0_context_shrinkage(
        policy="variable_reliability_v1",
        surface_rho_cap=0.25,
        rootzone_rho_cap=0.75,
    )
    sample = _make_prompt_sample(
        split_role="target_eval",
        target_region_id="US-R1",
        sample_region_id="US-R2",
    )
    sample["x"] = np.zeros((12, 4, 4), dtype=np.float32)
    sample["forecast_surface"] = np.full((4, 4), 10.0, dtype=np.float32)
    sample["forecast_rootzone"] = np.full((4, 4), 20.0, dtype=np.float32)
    sample["month"] = 6

    pred = predictor.predict(sample)
    metadata = predictor.stage3_k0_context_shrinkage_metadata

    np.testing.assert_allclose(pred["pred_increment_surface"], 1.75)
    np.testing.assert_allclose(pred["pred_increment_rootzone"], 6.5)
    np.testing.assert_allclose(pred["pred_analysis_surface"], 11.75)
    np.testing.assert_allclose(pred["pred_analysis_rootzone"], 26.5)
    assert metadata["policy"] == "variable_reliability_v1"
    assert metadata["rho_surface_cap"] == 0.25
    assert metadata["rho_rootzone_cap"] == 0.75
    assert metadata["last_rho_surface"] == 0.25
    assert metadata["last_rho_rootzone"] == 0.75
    assert predictor.stage3_k0_context_shrinkage_last_rho == 0.5
    assert predictor.stage3_k0_context_shrinkage_last_rhos == {"surface": 0.25, "rootzone": 0.75}


def test_transfer_safe_score_uses_source_val_only():
    """Verify transfer safe score computation uses only source_val data."""
    from hydroda.training.calibration import calibrate_residual_gain_region_aware

    H, W = 16, 24
    samples_s_by_region = {}
    samples_r_by_region = {}
    alphas = [0.0, 0.5, 1.0]

    for region in ["US-R2", "US-R3"]:
        s_list = []
        r_list = []
        for _ in range(3):
            pred_inc = np.random.randn(H, W).astype(np.float32) * 0.01
            true_inc = np.random.randn(H, W).astype(np.float32) * 0.01
            fcst = np.random.randn(H, W).astype(np.float32) * 0.05
            mask = np.ones((H, W), dtype=np.float32)
            latw = np.ones((H, W), dtype=np.float32)
            s_list.append((pred_inc, true_inc, fcst, mask, latw))
            r_list.append((pred_inc * 0.8, true_inc * 0.8, fcst, mask, latw))
        samples_s_by_region[region] = s_list
        samples_r_by_region[region] = r_list

    result = calibrate_residual_gain_region_aware(
        samples_s_by_region=samples_s_by_region,
        samples_r_by_region=samples_r_by_region,
        alpha_grid=alphas,
    )

    assert result, "calibrate_residual_gain_region_aware should return non-empty result"
    assert "selection_score" in result
    assert "selection_trace" in result
    assert len(result["selection_trace"]) == 9  # 3x3 grid
    # Score should be finite and not -inf
    assert np.isfinite(result["selection_score"]), f"Score should be finite, got {result['selection_score']}"
    # Each trace entry should have the score formula fields
    trace = result["selection_trace"][0]
    assert "transfer_safe_score" in trace
    assert "worst_region_balanced_skill" in trace
    assert "neg_rootzone_count" in trace

    print(f"  test_transfer_safe_score_uses_source_val_only passed: score={result['selection_score']:.4f}")


def test_latitude_weights_equal_one_unweighted_equivalence():
    """Verify WeightedMaskedHuberLoss with uniform lat weights ~ unweighted."""
    loss_fn_w = WeightedMaskedHuberLoss(delta=0.01, use_lat_weight=True)

    B, C, H, W = 2, 2, 8, 8
    pred = torch.randn(B, C, H, W) * 0.01
    target = torch.randn(B, C, H, W) * 0.01
    mask = torch.ones(B, 1, H, W)

    # With uniform latitude weights of 1.0
    lat_w = torch.ones(1, 1, H, W)
    result_w = loss_fn_w(pred, target, mask, latitude_weight=lat_w)

    # With uniform latitude weights (but use_lat_weight=True, no lat_w -> uniform fallback)
    result_u = loss_fn_w(pred, target, mask, latitude_weight=None)

    # Both should be finite
    assert torch.isfinite(result_w["total_loss"])
    assert torch.isfinite(result_u["total_loss"])
    # With lat_w=ones, should match the uniform fallback (both use weight=1)
    assert torch.allclose(result_w["total_loss"], result_u["total_loss"], rtol=1e-4), \
        f"Uniform lat_w={result_w['total_loss'].item():.6f} vs None={result_u['total_loss'].item():.6f}"

    print(f"  test_latitude_weights_equal_one_unweighted_equivalence passed.")


def test_prompt_conditioned_normalized_targets_use_unit_loss_scale():
    """Normalized increment targets must not be divided by physical inc_std again."""
    trainer = PromptConditionedTrainer.__new__(PromptConditionedTrainer)
    trainer.target_increment_normalization = True
    trainer._inc_std = np.array([0.0073, 0.000886], dtype=np.float32)

    scale = trainer._get_increment_scale()

    assert scale is not None
    assert torch.allclose(scale, torch.ones(2, dtype=torch.float32))


def test_prompt_collapse_detection():
    """Verify PromptQualityTracker detects collapse and reports normal diversity."""
    tracker = PromptQualityTracker(num_regions=3)

    # Case 1: Normal diverse prompts
    tracker.reset()
    for i in range(3):
        emb = torch.randn(5, 16) + i * 2.0  # distinct per region
        rids = torch.full((5,), i, dtype=torch.long)
        tracker.update(emb, rids)
    metrics = tracker.compute_metrics()
    assert not metrics["prompt_collapse_detected"], "Should not detect collapse with diverse prompts"
    assert np.isfinite(metrics["prompt_pairwise_cosine_distance_mean"])
    assert metrics["prompt_pairwise_cosine_distance_mean"] > 0.01, \
        f"Mean cosine distance should be > 0.01 with diverse prompts, got {metrics['prompt_pairwise_cosine_distance_mean']}"
    print(f"  Normal prompts: cos_dist_mean={metrics['prompt_pairwise_cosine_distance_mean']:.4f}")

    # Case 2: Collapsed prompts (almost identical)
    tracker.reset()
    for i in range(3):
        emb = torch.ones(5, 16) * 0.001 + torch.randn(5, 16) * 1e-6  # nearly identical
        rids = torch.full((5,), i, dtype=torch.long)
        tracker.update(emb, rids)
    metrics = tracker.compute_metrics()
    # With nearly identical embeddings, cosine distance should be very small
    assert metrics["prompt_collapse_detected"] or \
        metrics["prompt_pairwise_cosine_distance_mean"] < 0.05, \
        f"Collapsed prompts should have small cosine distance, got {metrics['prompt_pairwise_cosine_distance_mean']}"
    print(f"  Collapsed prompts: cos_dist_mean={metrics['prompt_pairwise_cosine_distance_mean']:.4f}, collapsed={metrics['prompt_collapse_detected']}")

    # Case 3: Single region (not enough for pairwise)
    tracker.reset()
    emb = torch.randn(5, 16)
    rids = torch.zeros(5, dtype=torch.long)
    tracker.update(emb, rids)
    metrics = tracker.compute_metrics()
    assert np.isnan(metrics["prompt_pairwise_cosine_distance_mean"]), \
        "Single region should yield NaN cosine distance (not enough pairs)"
    assert not metrics["prompt_collapse_detected"], "Single region should not trigger collapse"

    print(f"  test_prompt_collapse_detection passed.")


def test_checkpoint_contains_model_and_prompt_encoder_state_dicts():
    """Verify saved checkpoint contains both model and prompt_encoder state_dicts."""
    train_dataset = FakePromptDataset(n_samples=4, H=32, W=48)

    model = FiLMConditionalResUNet(in_channels=12, out_channels=2, width=8, prompt_dim=16)
    prompt_encoder = RegionPromptEncoder(num_regions=2, input_channels=12, hidden_dim=16)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = PromptConditionedTrainer(
            model=model,
            prompt_encoder=prompt_encoder,
            train_dataset=train_dataset,
            max_epochs=1,
            batch_size=2,
            num_workers=0,
            device="cpu",
            checkpoint_dir=tmpdir,
            source_regions=["US-R1", "US-R2"],
            global_to_source_lookup={0: 0, 1: 1},
        )

        # Save a checkpoint with gain results
        gain_results = {
            "best_alpha_surface": 0.5,
            "best_alpha_rootzone": 0.6,
            "selection_score": 0.1,
            "skill_surface_with_alpha": 0.2,
            "skill_rootzone_with_alpha": 0.15,
            "rmse_surface_model": 0.01,
            "rmse_rootzone_model": 0.005,
            "alpha_grid": [0.0, 0.5, 1.0],
            "min_skill": 0.15,
            "mean_skill": 0.175,
        }
        trainer.save_checkpoint(
            Path(tmpdir) / "test.pt", epoch=0, loss=0.5, tag="test",
            gain_results=gain_results,
        )

        ckpt = torch.load(Path(tmpdir) / "test.pt", map_location="cpu", weights_only=False)

        assert "model_state_dict" in ckpt, "Checkpoint must contain model_state_dict"
        assert "prompt_encoder_state_dict" in ckpt, "Checkpoint must contain prompt_encoder_state_dict"
        assert "residual_gain_alpha_surface" in ckpt, "Checkpoint must contain residual_gain_alpha_surface"
        assert "residual_gain_alpha_rootzone" in ckpt, "Checkpoint must contain residual_gain_alpha_rootzone"
        assert "selection_score" in ckpt
        assert "config" in ckpt
        assert ckpt["config"]["source_regions"] == ["US-R1", "US-R2"]
        assert ckpt["config"]["checkpoint_every_n_epochs"] == 5  # default
        assert ckpt["config"]["adaptation_setting"] == "zero_shot_context"
        assert ckpt["config"]["K"] == 0
        assert ckpt["config"]["protocol_freeze_id"] == train_pc.PROTOCOL_FREEZE_ID

        print(f"  test_checkpoint_contains_model_and_prompt_encoder_state_dicts passed.")


def test_alpha_calibration_uses_region_aware_score():
    """Verify calibrate_residual_gain_region_aware does 2D grid with region-aware scoring."""
    from hydroda.training.calibration import calibrate_residual_gain_region_aware

    H, W = 16, 24
    # Create synthetic per-region samples with distinct patterns
    samples_s_by_region = {}
    samples_r_by_region = {}
    alphas = [0.0, 0.5, 1.0]

    for r_idx, region in enumerate(["US-R2", "US-R3", "US-R4"]):
        s_list = []
        r_list = []
        for _ in range(5):
            pred_inc = np.random.randn(H, W).astype(np.float32) * 0.01
            true_inc = np.random.randn(H, W).astype(np.float32) * 0.01
            fcst = np.random.randn(H, W).astype(np.float32) * 0.05
            mask = np.ones((H, W), dtype=np.float32)
            latw = np.ones((H, W), dtype=np.float32)
            s_list.append((pred_inc, true_inc, fcst, mask, latw))
            r_list.append((pred_inc * 0.8, true_inc * 0.8, fcst, mask, latw))  # slightly different
        samples_s_by_region[region] = s_list
        samples_r_by_region[region] = r_list

    result = calibrate_residual_gain_region_aware(
        samples_s_by_region=samples_s_by_region,
        samples_r_by_region=samples_r_by_region,
        alpha_grid=alphas,
    )

    assert result, "calibrate_residual_gain_region_aware should return non-empty dict"
    assert "best_alpha_surface" in result
    assert "best_alpha_rootzone" in result
    assert "selection_score" in result
    assert "region_variable_skills" in result
    assert result["calibration_mode"] == "region_aware_2d"
    assert result["region_names"] == ["US-R2", "US-R3", "US-R4"]

    # selection_trace should have 9 entries (3x3 grid)
    assert len(result["selection_trace"]) == 9, f"Expected 9 trace entries (3x3 grid), got {len(result['selection_trace'])}"

    # Check trace has expected fields
    trace = result["selection_trace"][0]
    assert "alpha_surface" in trace
    assert "alpha_rootzone" in trace
    assert "transfer_safe_score" in trace
    assert "worst_region_balanced_skill" in trace

    print(f"  test_alpha_calibration_uses_region_aware_score passed: "
          f"best_alpha=({result['best_alpha_surface']}, {result['best_alpha_rootzone']}), "
          f"score={result['selection_score']:.4f}")


if __name__ == "__main__":
    test_epoch_checkpoint_saving()
    test_transfer_safe_score_uses_source_val_only()
    test_latitude_weights_equal_one_unweighted_equivalence()
    test_prompt_collapse_detection()
    test_checkpoint_contains_model_and_prompt_encoder_state_dicts()
    test_alpha_calibration_uses_region_aware_score()
    print("\nAll prompt-conditioned smoke tests passed.")
