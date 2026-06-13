from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from hydroda.models.hyper_conditional_unet import HyperAdapterConditionalResUNet
from hydroda.models.prompt_encoder import RegionPromptEncoder


def _write_source_checkpoint(path: Path) -> None:
    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=4,
        prompt_dim=8,
        hyper_n_basis=3,
        hyper_adapter_bottleneck=2,
        zero_raw_increment_init=True,
    )
    prompt_encoder = RegionPromptEncoder(num_regions=5, input_channels=12, hidden_dim=8)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "prompt_encoder_state_dict": prompt_encoder.state_dict(),
            "config": {
                "model_type": "hyperda_basis_adapter",
                "width": 4,
                "prompt_dim": 8,
                "hyper_n_basis": 3,
                "hyper_adapter_bottleneck": 2,
                "hyper_adapter_scale": 1.0,
                "zero_raw_increment_init": True,
                "num_regions": 5,
                "ch_mean": [0.0] * 12,
                "ch_std": [1.0] * 12,
                "inc_mean": [0.0, 0.0],
                "inc_std": [1.0, 1.0],
                "source_regions": ["US-R2", "US-R3", "US-R4", "US-R5", "US-R6"],
                "source_region_global_indices": [1, 2, 3, 4, 5],
            },
        },
        path,
    )


def test_parse_args_accepts_bank_builder_options(monkeypatch, tmp_path):
    from scripts.train import build_source_episode_adapter_bank as bank

    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_source_episode_adapter_bank.py",
            "--source_checkpoint",
            str(source_checkpoint),
            "--output_dir",
            str(tmp_path / "bank"),
            "--K_list",
            "0,4,12",
            "--pseudo_target_regions",
            "US-R2,US-R3",
            "--max_episodes",
            "2",
            "--max_query_samples",
            "16",
            "--ridge_lambda",
            "2.0",
            "--ridge_clip_coeff_norm",
            "0.5",
            "--ridge_trust_region_radius",
            "0.25",
            "--device",
            "cpu",
        ],
    )

    args = bank.parse_args()

    assert args.K_list == [0, 4, 12]
    assert args.pseudo_target_regions == ["US-R2", "US-R3"]
    assert args.max_episodes == 2
    assert args.max_query_samples == 16
    assert args.ridge_lambda == pytest.approx(2.0)
    assert args.ridge_clip_coeff_norm == pytest.approx(0.5)
    assert args.ridge_trust_region_radius == pytest.approx(0.25)


def test_episode_metadata_schema_requires_protocol_fields():
    from scripts.train import build_source_episode_adapter_bank as bank

    metadata = {
        field: "value"
        for field in bank.REQUIRED_EPISODE_METADATA_FIELDS
    }
    metadata.update(
        {
            "schema_version": bank.EPISODE_SCHEMA_VERSION,
            "K": 0,
            "seed": 0,
            "support_count": 0,
            "adapter_space": bank.ADAPTER_SPACE,
            "adapter_vector_dim": 9,
            "adapter_delta_norm": 0.0,
            "target_eval_used": False,
            "leakage_metadata": {
                "target_eval_used": False,
                "target_eval_labels_loaded": False,
                "query_role": "source_val",
            },
        }
    )

    bank.validate_episode_metadata(metadata)
    missing = dict(metadata)
    missing.pop("checkpoint_hash")
    with pytest.raises(ValueError, match="checkpoint_hash"):
        bank.validate_episode_metadata(missing)


def test_validate_pseudo_target_regions_uses_checkpoint_source_regions(tmp_path):
    from scripts.train import build_source_episode_adapter_bank as bank

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    assert bank.resolve_pseudo_target_regions(checkpoint, ["US-R2"], allow_regions_not_in_checkpoint=False) == ["US-R2"]
    with pytest.raises(ValueError, match="not present in checkpoint source_regions"):
        bank.resolve_pseudo_target_regions(checkpoint, ["US-R1"], allow_regions_not_in_checkpoint=False)


def test_build_k0_episode_metadata_records_no_support_label_use(tmp_path):
    from scripts.train import build_source_episode_adapter_bank as bank

    metadata = bank.build_episode_metadata(
        episode_id="US-R2_K0_seed0",
        pseudo_target_region="US-R2",
        source_regions_used=["US-R3", "US-R4"],
        K=0,
        seed=0,
        context_dates=["2015-01-01"],
        context_dates_hash="context_hash",
        support_dates=[],
        support_dates_hash="support_hash",
        query_dates=["2022-01-01"],
        query_dates_hash="query_hash",
        prompt_context_stats={"n_samples": 1, "label_usage": "none"},
        adapter_parameter_names=["a", "b"],
        adapter_vector_dim=2,
        adapter_delta_norm=0.0,
        query_skill={"surface": {"skill_primary": 0.1}},
        checkpoint_path=str(tmp_path / "source.pt"),
        checkpoint_hash="ckpt_hash",
        split_manifest_path="splits.json",
        split_manifest_hash="split_hash",
        normalizer_provenance={"normalization_source": "source_fit_only_from_source_checkpoint"},
        ridge_diagnostics={},
        checkpoint_source_regions=["US-R2", "US-R3", "US-R4"],
        allowed_region_override=False,
    )

    bank.validate_episode_metadata(metadata)
    assert metadata["support_count"] == 0
    assert metadata["leakage_metadata"]["support_labels_used"] is False
    assert metadata["leakage_metadata"]["target_eval_used"] is False
    assert metadata["leakage_metadata"]["target_eval_labels_loaded"] is False
    assert metadata["leakage_metadata"]["query_role"] == "source_val"


def test_save_adapter_coeff_artifact_contains_only_coefficient_vectors(tmp_path):
    from scripts.train import build_source_episode_adapter_bank as bank
    from scripts.train.train_hyperda_few_shot_adapt import (
        coefficient_residual_vector,
        load_source_checkpoint_for_few_shot,
        set_coefficient_residual_vector,
    )

    ckpt_path = tmp_path / "source.pt"
    _write_source_checkpoint(ckpt_path)
    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=str(ckpt_path),
        device=torch.device("cpu"),
        target_latent_dim=4,
    )
    base = coefficient_residual_vector(state.model)
    set_coefficient_residual_vector(state.model, base + 0.25)

    artifact_path = tmp_path / "adapter_coefficients.pt"
    payload = bank.save_adapter_coefficient_artifact(
        artifact_path,
        state.model,
        base_coeff_vector=base,
    )
    saved = torch.load(artifact_path, map_location="cpu", weights_only=False)

    assert payload["space"] == "hyperda_adapter_coefficient_residual_logits"
    assert torch.allclose(saved["delta_coeff_vector"], torch.full_like(base, 0.25))
    assert "model_state_dict" not in saved
    assert "prompt_encoder_state_dict" not in saved
    assert saved["parameter_names"] == [
        "target_adapter_coefficient_residual_b.logit_delta",
        "target_adapter_coefficient_residual_d2.logit_delta",
        "target_adapter_coefficient_residual_d1.logit_delta",
    ]


def test_write_bank_manifest_and_index(tmp_path):
    from scripts.train import build_source_episode_adapter_bank as bank

    output_dir = tmp_path / "bank"
    episode_row = {
        "episode_id": "US-R2_K0_seed0",
        "pseudo_target_region": "US-R2",
        "K": 0,
        "metadata_path": "episodes/US-R2_K0_seed0/metadata.json",
        "adapter_path": "episodes/US-R2_K0_seed0/adapter_coefficients.pt",
        "query_metrics_path": "episodes/US-R2_K0_seed0/query_metrics.json",
        "adapter_delta_norm": 0.0,
        "surface_skill_primary": 0.1,
        "rootzone_skill_primary": 0.2,
    }
    manifest = bank.write_bank_manifest(
        output_dir=output_dir,
        episodes=[episode_row],
        config={"source_checkpoint": "source.pt", "K_list": [0]},
        checkpoint_hash="ckpt_hash",
        split_manifest_hash="split_hash",
    )

    assert manifest["schema_version"] == bank.BANK_SCHEMA_VERSION
    assert manifest["n_episodes"] == 1
    assert (output_dir / "manifest.json").exists()
    rows = [json.loads(line) for line in (output_dir / "episodes_index.jsonl").read_text().splitlines()]
    assert rows == [episode_row]
    summary = (output_dir / "summary.csv").read_text()
    assert "episode_id,pseudo_target_region,K" in summary


def test_pseudo_query_dataset_overrides_target_eval_metadata():
    from scripts.train import build_source_episode_adapter_bank as bank

    class TinyDataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            assert idx == 0
            return {
                "split_role": "source_val",
                "target_eval_dates_hash": "real_target_eval_hash",
                "target_region_id": "US-R2",
                "active_region_ids": ["US-R2"],
            }

    sample = bank._PseudoQueryDataset(TinyDataset())[0]

    assert sample["split_role"] == "source_val_pseudo_query"
    assert sample["target_eval_dates_hash"] == "not_used_source_val_query"
    assert sample["target_region_id"] == "US-R2"
    assert sample["active_region_ids"] == ["US-R2"]
