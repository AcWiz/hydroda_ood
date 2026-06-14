"""Tests for the paper-facing HyperDA zero/few-shot protocol."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hydroda.data.leakage_guard import LeakageGuard
from hydroda.data.protocol import ProtocolConfig
from hydroda.splits.manifest import create_split_manifest, validate_no_leakage


def _record(idx: int, date_str: str) -> dict:
    return {
        "time_index": idx,
        "date_str": date_str,
        "datetime_str": f"{date_str}T00:00:00",
    }


def test_protocol_main_k_axis_and_legacy_full_target_train_opt_in():
    protocol = ProtocolConfig()

    assert "zero_few_shot_generalization" in protocol.protocol_freeze_id
    assert tuple(protocol.main_K_values) == (0, 4, 12)
    assert tuple(protocol.main_adaptation_settings) == (
        "zero_shot_context",
        "few_shot_k4",
        "few_shot_k12",
    )

    for k in (0, 4, 12):
        protocol.assert_supported_K(k)
    with pytest.raises(ValueError):
        protocol.assert_supported_K(24)

    with pytest.raises(ValueError, match="legacy"):
        protocol.assert_supported_adaptation_setting("target_full_train")
    protocol.assert_supported_adaptation_setting(
        "target_full_train",
        allow_legacy_full_target_train=True,
    )


def test_target_val_is_not_a_main_protocol_selection_source():
    guard = LeakageGuard(ProtocolConfig())

    guard.check_model_selection_scope(
        ["2022-06-01"],
        purpose="checkpoint_selection",
        model_selection_source="source_val_preregistered",
    )

    with pytest.raises(ValueError, match="target_val"):
        guard.check_model_selection_scope(
            ["2022-06-01"],
            purpose="checkpoint_selection",
            model_selection_source="target_val",
        )
    with pytest.raises(ValueError, match="target_val"):
        guard.check_target_side_selection_scope(["2022-06-01"], purpose="early_stopping")
    with pytest.raises(ValueError, match="gain calibration"):
        guard.check_target_residual_gain_calibration_scope(
            ["2019-06-01"],
            purpose="gain calibration",
        )


def test_zero_few_shot_manifest_schema_and_validation():
    manifest = create_split_manifest(
        target_region="US-R1",
        source_regions=["US-R2"],
        K=4,
        seed=1,
        source_train_dates=[_record(1, "2021-01-15")],
        source_val_dates=[_record(2, "2022-06-15")],
        target_context_dates=[_record(3, "2015-03-15"), _record(4, "2021-09-15")],
        support_dates=[_record(5, "2019-04-15"), _record(6, "2019-07-15")],
        query_dates=[_record(7, "2023-01-15")],
        adaptation_setting="few_shot_k4",
    )

    assert manifest["manifest_schema_version"] == "v4_4_zero_few_shot"
    assert manifest["adaptation_setting"] == "few_shot_k4"
    assert manifest["target_context_dates"]
    assert manifest["target_support_dates"] == manifest["target_train_dates"]
    assert manifest["target_context_dates_hash"]
    assert manifest["target_support_dates_hash"]
    assert manifest["model_selection_source"] == "source_val_preregistered"
    assert manifest["target_val_usage"] == "unused_in_main_protocol"
    assert manifest["target_full_train_usage"] == "legacy_internal_only"

    results = validate_no_leakage(manifest)
    assert results["target_context_in_context_year"] is True
    assert results["support_in_support_year"] is True
    assert results["query_in_query_years"] is True
    assert results["target_val_unused_in_main_protocol"] is True


def test_zero_shot_manifest_defaults_missing_support_to_empty_list():
    manifest = create_split_manifest(
        target_region="US-R1",
        source_regions=["US-R2"],
        K=0,
        seed=0,
        source_train_dates=[_record(1, "2021-01-15")],
        source_val_dates=[_record(2, "2022-06-15")],
        target_context_dates=[_record(3, "2015-03-15"), _record(4, "2021-09-15")],
        query_dates=[_record(7, "2023-01-15")],
        adaptation_setting="zero_shot_context",
    )

    assert manifest["target_support_dates"] == []
    assert manifest["target_train_dates"] == []
    assert validate_no_leakage(manifest)["k0_has_empty_support"] is True


def test_manifest_without_setting_defaults_to_main_zero_shot():
    manifest = create_split_manifest(
        target_region="US-R1",
        source_regions=["US-R2"],
        seed=0,
        source_train_dates=[_record(1, "2021-01-15")],
        source_val_dates=[_record(2, "2022-06-15")],
        target_context_dates=[_record(3, "2015-03-15")],
        query_dates=[_record(7, "2023-01-15")],
    )

    assert manifest["adaptation_protocol"] == "zero_few_shot_generalization"
    assert manifest["adaptation_setting"] == "zero_shot_context"
    assert manifest["K"] == 0
    assert manifest["target_support_dates"] == []


def test_manifest_rejects_unsupported_main_k_without_legacy_setting():
    with pytest.raises(ValueError, match="Unsupported main zero/few-shot K"):
        create_split_manifest(
            target_region="US-R1",
            source_regions=["US-R2"],
            K=1,
            seed=0,
            source_train_dates=[_record(1, "2021-01-15")],
            source_val_dates=[_record(2, "2022-06-15")],
            target_context_dates=[_record(3, "2015-03-15")],
            support_dates=[_record(4, "2019-05-15")],
            query_dates=[_record(7, "2023-01-15")],
        )


def test_manifest_rejects_k_adaptation_setting_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        create_split_manifest(
            target_region="US-R1",
            source_regions=["US-R2"],
            K=12,
            seed=0,
            source_train_dates=[_record(1, "2021-01-15")],
            source_val_dates=[_record(2, "2022-06-15")],
            target_context_dates=[_record(3, "2015-03-15")],
            support_dates=[_record(4, "2019-05-15")],
            query_dates=[_record(7, "2023-01-15")],
            adaptation_setting="few_shot_k4",
        )


def test_zero_few_shot_validation_enforces_main_k_budget():
    zero_with_support = create_split_manifest(
        target_region="US-R1",
        source_regions=["US-R2"],
        K=0,
        seed=0,
        source_train_dates=[_record(1, "2021-01-15")],
        source_val_dates=[_record(2, "2022-06-15")],
        target_context_dates=[_record(3, "2015-03-15")],
        support_dates=[_record(5, "2019-04-15")],
        query_dates=[_record(7, "2023-01-15")],
        adaptation_setting="zero_shot_context",
    )
    zero_results = validate_no_leakage(zero_with_support)
    assert zero_results["k_matches_or_less_support_count"] is False
    assert zero_results["k0_has_empty_support"] is False

    five_support = [_record(10 + i, f"2019-{i + 1:02d}-15") for i in range(5)]
    k4_with_five_support = create_split_manifest(
        target_region="US-R1",
        source_regions=["US-R2"],
        K=4,
        seed=0,
        source_train_dates=[_record(1, "2021-01-15")],
        source_val_dates=[_record(2, "2022-06-15")],
        target_context_dates=[_record(3, "2015-03-15")],
        support_dates=five_support,
        query_dates=[_record(7, "2023-01-15")],
        adaptation_setting="few_shot_k4",
    )
    assert validate_no_leakage(k4_with_five_support)["k_matches_or_less_support_count"] is False


def test_converter_support_selection_uses_base_valid_coverage_mask():
    from scripts.data.convert_kdate_to_zero_few_shot_splits import select_support_records_with_coverage

    records = [
        _record(1, "2019-01-15") | {"base_valid_coverage": 0.1},
        _record(2, "2019-02-15") | {"base_valid_coverage": 0.9},
        _record(3, "2019-04-15") | {"base_valid_coverage": 0.2},
        _record(4, "2019-05-15") | {"base_valid_coverage": 0.8},
        _record(5, "2019-07-15") | {"base_valid_coverage": 0.9},
        _record(6, "2019-10-15") | {"base_valid_coverage": 0.9},
    ]

    selected = select_support_records_with_coverage(records, K=4, seed=0, min_coverage=0.5)
    selected_dates = {record["date_str"] for record in selected}

    assert "2019-01-15" not in selected_dates
    assert "2019-04-15" not in selected_dates
    assert selected_dates <= {"2019-02-15", "2019-05-15", "2019-07-15", "2019-10-15"}


def test_evaluate_checkpoint_preserves_legacy_full_target_train_split_path():
    from scripts.eval.evaluate_checkpoint import resolve_split_protocol_for_adaptation

    split_path, protocol_freeze_id = resolve_split_protocol_for_adaptation("target_full_train")
    assert split_path == "artifacts/splits/US_loro_target_train_splits.json"
    assert "historical_target_adapt" in protocol_freeze_id

    split_path, protocol_freeze_id = resolve_split_protocol_for_adaptation("few_shot_k4")
    assert split_path == "artifacts/splits/US_loro_zero_few_shot_splits.json"
    assert "zero_few_shot_generalization" in protocol_freeze_id


def test_phase4_train_entrypoints_default_to_main_zero_shot(monkeypatch):
    from scripts.train import train_prompt_conditioned_shared as prompt_runner
    from scripts.train import train_source_only_backbone as source_runner

    monkeypatch.setattr(
        "sys.argv",
        ["train_source_only_backbone.py", "--target_region", "US-R1"],
    )
    source_args = source_runner.parse_args()
    assert source_args.adaptation_setting == "zero_shot_context"
    assert source_args.K == 0

    monkeypatch.setattr(
        "sys.argv",
        ["train_prompt_conditioned_shared.py", "--target_region", "US-R1"],
    )
    prompt_args = prompt_runner.parse_args()
    assert prompt_args.adaptation_setting == "zero_shot_context"
    assert prompt_args.K == 0


def test_dataset_default_split_selection_prefers_main_zero_shot(tmp_path):
    from hydroda.data.dataset import HydroDADataset

    split_path = tmp_path / "splits.json"
    manifest = create_split_manifest(
        target_region="US-R1",
        source_regions=["US-R2"],
        K=0,
        seed=0,
        source_train_dates=[_record(1, "2021-01-15")],
        source_val_dates=[_record(2, "2022-06-15")],
        target_context_dates=[_record(3, "2015-03-15")],
        query_dates=[_record(4, "2023-01-15")],
        adaptation_setting="zero_shot_context",
    )
    split_path.write_text(json.dumps({"splits": [manifest]}))

    dataset = HydroDADataset.__new__(HydroDADataset)
    dataset.splits_json = str(split_path)
    dataset.target_region = "US-R1"
    dataset.seed = 0
    dataset.K = None
    dataset.adaptation_setting = None

    entry = HydroDADataset._load_split_entry(dataset)
    assert entry["adaptation_setting"] == "zero_shot_context"


def test_dataset_date_records_fallback_when_primary_key_is_empty():
    from hydroda.data.dataset import HydroDADataset

    dataset = HydroDADataset.__new__(HydroDADataset)
    dataset._split_entry = {
        "source_test_dates": [],
        "target_eval_dates": [_record(4, "2023-01-15")],
    }

    records = HydroDADataset._get_date_records(dataset, "source_test_dates")

    assert records == [_record(4, "2023-01-15")]


def test_paper_main_registry_uses_zero_few_shot_methods():
    from hydroda.baselines.registry import PAPER_MAIN_BASELINES, assert_allowed_for_table

    allowed = PAPER_MAIN_BASELINES["zero_few_shot_generalization"]
    assert "source_pooled_global_backbone" in allowed
    assert "source_regime_specialist_bank" in allowed
    assert "hyperda_zero_shot_context" in allowed
    assert "hyperda_few_shot_k4" in allowed
    assert "hyperda_few_shot_k12" in allowed
    assert "legacy_all_regions_sanity" not in allowed
    assert "target_full_history_region_oracle" not in allowed
    assert "hyperda_generated_operator_full_target_train" not in allowed

    assert_allowed_for_table("source_pooled_global_backbone", "paper_main")
    assert_allowed_for_table("hyperda_few_shot_k4", "paper_main")
    with pytest.raises(ValueError, match="paper_main"):
        assert_allowed_for_table("legacy_all_regions_sanity", "paper_main")
    with pytest.raises(ValueError, match="paper_main"):
        assert_allowed_for_table("target_full_history_region_oracle", "paper_main")
    with pytest.raises(ValueError, match="paper_main"):
        assert_allowed_for_table("hyperda_generated_operator_full_target_train", "paper_main")


def test_zero_few_shot_artifact_contract_if_present():
    path = Path("artifacts/splits/US_loro_zero_few_shot_splits.json")
    if not path.exists():
        pytest.skip(f"split artifact not generated in this checkout: {path}")

    data = json.loads(path.read_text())
    splits = data["splits"]
    expected = {
        (f"US-R{i}", k, seed)
        for i in range(1, 7)
        for k in (0, 4, 12)
        for seed in (0, 1, 2)
    }
    observed = {
        (s["target_region_id"], int(s["K"]), int(s["seed"]))
        for s in splits
        if s.get("adaptation_protocol") == "zero_few_shot_generalization"
    }
    assert expected <= observed

    for split in splits:
        if split.get("adaptation_protocol") != "zero_few_shot_generalization":
            continue
        k = int(split["K"])
        context_dates = {d["date_str"] for d in split["target_context_dates"]}
        support_dates = {d["date_str"] for d in split["target_support_dates"]}
        eval_dates = {d["date_str"] for d in split["target_eval_dates"]}
        assert all(2015 <= int(d[:4]) <= 2021 for d in context_dates)
        assert len(support_dates) <= k
        if k == 0:
            assert not support_dates
        assert all(2023 <= int(d[:4]) <= 2025 for d in eval_dates)
        assert context_dates.isdisjoint(eval_dates)
        assert support_dates.isdisjoint(eval_dates)
