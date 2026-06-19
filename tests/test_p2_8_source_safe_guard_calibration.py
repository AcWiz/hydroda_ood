from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _write_rows(path: Path, rows: list[dict[str, object]]) -> Path:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(
    episode: str,
    candidate_id: str,
    overall_skill: float,
    *,
    K: int = 12,
    schedule_label: str = "original_K12",
    rho_policy: str = "fixed_1.0",
    trust_policy: str = "none",
) -> dict[str, object]:
    return {
        "episode_id": episode,
        "pseudo_target_region": episode.split("_", 1)[0],
        "query_role": "source_val",
        "split_type": "source_val",
        "K": K,
        "candidate_id": candidate_id,
        "schedule_label": schedule_label,
        "support_loss_reduction": "global_pixel",
        "rho_policy": rho_policy,
        "trust_policy": trust_policy,
        "surface_skill_primary": overall_skill,
        "rootzone_skill_primary": overall_skill,
        "source_checkpoint_sha256": "abc",
        "split_manifest_sha256": "def",
        "target_eval_dates_hash": "not_used_source_val_query",
    }


def test_guard_candidate_enumeration_defaults_to_compact_budgeted_grid():
    from scripts.eval import calibrate_source_safe_guard as calib

    candidates = calib.enumerate_guard_candidates()
    base_configs = calib.enumerate_guard_base_configs()

    assert len(base_configs) == 12
    assert [c["base_config_id"] for c in base_configs] == [
        f"C{i}_{scope}" for i in range(6) for scope in ("coeff_gain", "safe_operator")
    ]
    assert len(candidates) == 84
    assert {c["rho_policy"] for c in candidates} == {
        "fixed_1.0",
        "fixed_0.75",
        "fixed_0.5",
        "fixed_0.25",
        "rule_a",
        "rule_b",
        "rule_c",
    }
    assert all(c["K"] == 12 and c["adapt_scope"] in {"coeff_gain", "safe_operator"} for c in candidates)
    assert len({c["candidate_id"] for c in candidates}) == len(candidates)


def test_full_candidate_set_is_retained_behind_explicit_full_v1():
    from scripts.eval import calibrate_source_safe_guard as calib

    candidates = calib.enumerate_guard_candidates(candidate_set="full_v1")

    assert len(candidates) == 168
    assert {c["schedule_label"] for c in candidates} == {"original_K12", "K4_schedule_on_K12"}
    assert {c["support_loss_reduction"] for c in candidates} == {"global_pixel", "cycle_balanced"}
    assert {c["trust_policy"] for c in candidates} == {"none", "mild_groupwise", "strong_groupwise"}
    assert {c["adapt_scope"] for c in candidates} == {"coeff_gain", "safe_operator"}


def test_stage3_conservative_candidate_set_excludes_safe_operator_scope():
    from scripts.eval import calibrate_source_safe_guard as calib

    base_configs = calib.enumerate_guard_base_configs(candidate_set="stage3_conservative_v1")
    logical = calib.enumerate_guard_candidates(candidate_set="stage3_conservative_v1")

    assert base_configs
    assert logical
    assert {row["K"] for row in base_configs} == {4, 12}
    assert {row["K"] for row in logical} == {4, 12}
    assert {row["adapt_scope"] for row in logical} <= {"coeff_only", "coeff_gain", "none"}
    assert "safe_operator" not in {row["adapt_scope"] for row in logical}
    assert len({row["candidate_id"] for row in logical}) == len(logical)


@pytest.mark.parametrize(
    ("policy", "diagnostics", "expected"),
    [
        ("fixed_1.0", {}, 1.0),
        ("fixed_0.75", {}, 0.75),
        ("fixed_0.5", {}, 0.5),
        ("fixed_0.25", {}, 0.25),
        ("rule_a", {"support_gradient_negative_fraction": 0.46}, 0.5),
        ("rule_a", {"support_gradient_negative_fraction": 0.45}, 1.0),
        ("rule_b", {"support_gradient_negative_fraction": 0.2, "support_gradient_cosine_min": -0.21}, 0.5),
        ("rule_b", {"support_gradient_negative_fraction": 0.2, "support_gradient_cosine_min": -0.2}, 1.0),
        ("rule_c", {"support_gradient_negative_fraction": 0.51, "support_gradient_cosine_min": 0.0}, 0.25),
        ("rule_c", {"support_gradient_negative_fraction": 0.5, "support_gradient_cosine_min": -0.5}, 1.0),
    ],
)
def test_conflict_aware_rho_policies(policy, diagnostics, expected):
    from scripts.eval import calibrate_source_safe_guard as calib

    assert calib.compute_rho_for_policy(policy, diagnostics) == pytest.approx(expected)


def test_source_safe_score_and_ranking_are_exact_for_synthetic_rows(tmp_path):
    from scripts.eval import calibrate_source_safe_guard as calib

    rows = []
    for episode, baseline in [("US-R2_e0", 0.10), ("US-R3_e0", 0.10), ("US-R4_e0", 0.10)]:
        rows.append(_row(episode, "K0_identity", baseline, K=0, schedule_label="identity_base"))
    rows.extend(
        [
            _row("US-R2_e0", "A", 0.20),
            _row("US-R3_e0", "A", 0.00),
            _row("US-R4_e0", "A", 0.10),
            _row("US-R2_e0", "B", 0.15, rho_policy="fixed_0.5", trust_policy="mild_groupwise"),
            _row("US-R3_e0", "B", 0.12, rho_policy="fixed_0.5", trust_policy="mild_groupwise"),
            _row("US-R4_e0", "B", 0.09, rho_policy="fixed_0.5", trust_policy="mild_groupwise"),
        ]
    )
    loaded = calib.load_calibration_rows([_write_rows(tmp_path / "rows.csv", rows)])
    summaries = calib.score_candidates(loaded)

    by_id = {row["candidate_id"]: row for row in summaries}
    assert by_id["A"]["mean_delta_vs_K0"] == pytest.approx(0.0)
    assert by_id["A"]["mean_regret_vs_K0"] == pytest.approx(1.0 / 30.0)
    assert by_id["A"]["negative_transfer_rate_vs_K0"] == pytest.approx(1.0 / 3.0)
    assert by_id["A"]["score"] == pytest.approx(-1.0 / 30.0)
    assert by_id["B"]["mean_delta_vs_K0"] == pytest.approx(0.02)
    assert by_id["B"]["mean_regret_vs_K0"] == pytest.approx(0.01 / 3.0)
    assert by_id["B"]["score"] == pytest.approx(0.02 - 0.5 * (0.01 / 3.0) - 0.05 * (1.0 / 3.0))
    assert calib.rank_candidates(summaries)[0]["candidate_id"] == "B"


def test_m2_4_context_reliability_shrinkage_is_bounded_and_monotone():
    from scripts.eval import calibrate_source_safe_guard as calib

    high_reliability = [1.0, 1.0, 1.0, 0.95, 0.05]
    medium_reliability = [0.6, 1.0, 0.8, 0.75, 0.30]
    low_reliability = [0.1, 0.0, 0.5, 0.40, 0.90]

    high = calib.compute_m2_4_context_shrinkage(high_reliability, source_calibrated_rho_cap=0.8)
    medium = calib.compute_m2_4_context_shrinkage(medium_reliability, source_calibrated_rho_cap=0.8)
    low = calib.compute_m2_4_context_shrinkage(low_reliability, source_calibrated_rho_cap=0.8)

    assert 0.0 <= low <= medium <= high <= 0.8
    assert calib.compute_m2_4_context_shrinkage(high_reliability, source_calibrated_rho_cap=2.0) <= 1.0
    assert calib.compute_m2_4_context_shrinkage(high_reliability, source_calibrated_rho_cap=-1.0) == 0.0


def test_m2_4_source_episode_selection_uses_worst_case_non_degradation_vs_m2_1():
    from scripts.eval import calibrate_source_safe_guard as calib

    rows = []
    for episode in ("US-R2_e0", "US-R3_e0"):
        rows.append(
            {
                **_row(episode, "M2_1_frozen_prior", 0.50, K=0, schedule_label="M2_1_frozen_prior"),
                "surface_skill_primary": 0.50,
                "rootzone_skill_primary": 0.50,
            }
        )
    rows.extend(
        [
            {
                **_row("US-R2_e0", "rho1_mean_good_but_bad_worst_case", 0.65, K=0),
                "surface_skill_primary": 0.65,
                "rootzone_skill_primary": 0.65,
                "context_shrinkage_rho": 1.0,
            },
            {
                **_row("US-R3_e0", "rho1_mean_good_but_bad_worst_case", 0.59, K=0),
                "surface_skill_primary": 0.70,
                "rootzone_skill_primary": 0.48,
                "context_shrinkage_rho": 1.0,
            },
            {
                **_row("US-R2_e0", "rho05_safe", 0.535, K=0),
                "surface_skill_primary": 0.55,
                "rootzone_skill_primary": 0.52,
                "context_shrinkage_rho": 0.5,
            },
            {
                **_row("US-R3_e0", "rho05_safe", 0.495, K=0),
                "surface_skill_primary": 0.50,
                "rootzone_skill_primary": 0.49,
                "context_shrinkage_rho": 0.5,
            },
        ]
    )

    summaries = calib.score_m2_4_source_episode_candidates(rows, safe_score_floor_ratio=0.98)
    by_id = {row["candidate_id"]: row for row in summaries}

    assert by_id["rho1_mean_good_but_bad_worst_case"]["mean_delta_vs_m2_1"] > by_id["rho05_safe"][
        "mean_delta_vs_m2_1"
    ]
    assert by_id["rho1_mean_good_but_bad_worst_case"]["safe_against_m2_1_98pct"] is False
    assert by_id["rho1_mean_good_but_bad_worst_case"]["worst_rootzone_ratio_vs_m2_1"] == pytest.approx(0.96)
    assert by_id["rho05_safe"]["safe_against_m2_1_98pct"] is True
    assert by_id["rho05_safe"]["worst_variable_ratio_vs_m2_1"] == pytest.approx(0.98)

    selected = calib.select_m2_4_conservative_candidate(rows, safe_score_floor_ratio=0.98)
    assert selected["candidate_id"] == "rho05_safe"
    assert selected["selection_baseline_candidate_id"] == "M2_1_frozen_prior"
    assert selected["target_labels_used_for_adaptation"] is False
    assert selected["target_val_usage"] == "unused_in_main_protocol"
    assert selected["target_eval_usage"] == "final_eval_only_no_selection"
    assert selected["target_eval_input_stats_used_for_update"] is False


def test_m2_4a_variable_source_episode_selection_exports_surface_rootzone_caps():
    from scripts.eval import calibrate_source_safe_guard as calib

    rows = []
    for episode in ("US-R2_e0", "US-R3_e0"):
        rows.append(
            {
                **_row(episode, "M2_1_frozen_prior", 0.50, K=0, schedule_label="M2_1_frozen_prior"),
                "surface_skill_primary": 0.50,
                "rootzone_skill_primary": 0.50,
            }
        )
    rows.extend(
        [
            {
                **_row("US-R2_e0", "m2_4a_s05_r025", 0.53, K=0),
                "surface_skill_primary": 0.56,
                "rootzone_skill_primary": 0.50,
                "context_shrinkage_policy": "source_episode_calibrated_v1",
                "context_shrinkage_rho_surface": 0.5,
                "context_shrinkage_rho_rootzone": 0.25,
            },
            {
                **_row("US-R3_e0", "m2_4a_s05_r025", 0.52, K=0),
                "surface_skill_primary": 0.54,
                "rootzone_skill_primary": 0.50,
                "context_shrinkage_policy": "source_episode_calibrated_v1",
                "context_shrinkage_rho_surface": 0.5,
                "context_shrinkage_rho_rootzone": 0.25,
            },
            {
                **_row("US-R2_e0", "m2_4a_s075_r075", 0.56, K=0),
                "surface_skill_primary": 0.58,
                "rootzone_skill_primary": 0.54,
                "context_shrinkage_policy": "source_episode_calibrated_v1",
                "context_shrinkage_rho_surface": 0.75,
                "context_shrinkage_rho_rootzone": 0.75,
            },
            {
                **_row("US-R3_e0", "m2_4a_s075_r075", 0.51, K=0),
                "surface_skill_primary": 0.56,
                "rootzone_skill_primary": 0.46,
                "context_shrinkage_policy": "source_episode_calibrated_v1",
                "context_shrinkage_rho_surface": 0.75,
                "context_shrinkage_rho_rootzone": 0.75,
            },
        ]
    )

    selected = calib.select_m2_4a_variable_conservative_candidate(rows, safe_score_floor_ratio=0.98)
    policy = calib.build_stage3_k0_m2_4a_policy_json(
        selected,
        final_target_region="US-R1",
        seed=0,
    )

    assert selected["candidate_id"] == "m2_4a_surface_0.75__rootzone_0.25"
    assert selected["rho_surface_cap"] == pytest.approx(0.75)
    assert selected["rho_rootzone_cap"] == pytest.approx(0.25)
    assert selected["safe_against_m2_1_98pct"] is True
    assert selected["source_episode_regions"] == ["US-R2", "US-R3"]
    assert policy["schema_version"] == "stage3_k0_m2_4a_source_episode_policy_v1"
    assert policy["policy_source"] == "source_episode_calibrated_v1"
    assert policy["policy"] == "source_episode_calibrated_v1"
    assert policy["rho_surface_cap"] == pytest.approx(0.75)
    assert policy["rho_rootzone_cap"] == pytest.approx(0.25)
    assert policy["source_episode_regions"] == ["US-R2", "US-R3"]
    assert policy["target_labels_used_for_adaptation"] is False
    assert policy["target_eval_input_stats_used_for_update"] is False
    assert policy["policy_hash"] == calib.compute_stage3_k0_m2_4a_policy_hash(policy)


def test_calibration_rejects_target_eval_rows_even_when_marked_diagnostic(tmp_path):
    from scripts.eval import calibrate_source_safe_guard as calib

    rows = [
        _row("US-R2_e0", "K0_identity", 0.1, K=0, schedule_label="identity_base"),
        {
            **_row("US-R2_e0", "A", 0.2),
            "split_type": "target_eval",
            "query_role": "target_eval",
            "diagnostic_only": "true",
        },
    ]

    with pytest.raises(ValueError, match="target_eval"):
        calib.load_calibration_rows([_write_rows(tmp_path / "target_eval.csv", rows)])


@pytest.mark.parametrize(
    ("split_type", "query_role", "adaptation_setting"),
    [
        ("target_query", "source_val", ""),
        ("target_val", "source_val", ""),
        ("source_val", "target_val", ""),
        ("source_val", "source_val", "target_full_train"),
    ],
)
def test_calibration_rejects_forbidden_target_side_rows(tmp_path, split_type, query_role, adaptation_setting):
    from scripts.eval import calibrate_source_safe_guard as calib

    row = {
        **_row("US-R2_e0", "A", 0.2),
        "split_type": split_type,
        "query_role": query_role,
        "adaptation_setting": adaptation_setting,
    }

    with pytest.raises(ValueError, match=split_type if split_type != "source_val" else query_role or adaptation_setting):
        calib.load_calibration_rows([_write_rows(tmp_path / "forbidden.csv", [row])])


def test_strict_checkpoint_region_validation_rejects_final_or_pseudo_target():
    from scripts.eval import calibrate_source_safe_guard as calib

    with pytest.raises(ValueError, match="final target"):
        calib.validate_checkpoint_source_regions(
            ["US-R1", "US-R3"],
            final_target_region="US-R1",
            pseudo_target_region="US-R2",
            allow_in_checkpoint_source_episodes=False,
        )
    with pytest.raises(ValueError, match="pseudo-target"):
        calib.validate_checkpoint_source_regions(
            ["US-R2", "US-R3"],
            final_target_region="US-R1",
            pseudo_target_region="US-R2",
            allow_in_checkpoint_source_episodes=False,
        )


def test_explicit_fallback_records_weaker_evidence_metadata():
    from scripts.eval import calibrate_source_safe_guard as calib

    result = calib.validate_checkpoint_source_regions(
        ["US-R2", "US-R3"],
        final_target_region="US-R1",
        pseudo_target_region="US-R2",
        allow_in_checkpoint_source_episodes=True,
    )

    assert result["source_safety_evidence_level"] == "source_safe_in_checkpoint_weaker"
    assert result["allow_in_checkpoint_source_episodes"] is True
    assert "pseudo-target" in result["weaker_evidence_reason"]


def test_trust_radii_are_derived_from_source_k4_original_drift_only():
    from scripts.eval import calibrate_source_safe_guard as calib

    rows = [
        {
            "episode_id": "US-R2_e0",
            "query_role": "source_val",
            "split_type": "source_val",
            "K": 4,
            "schedule_label": "original_K4",
            "target_parameter_l2_drift_post_anchor_total": "4.0",
            "target_parameter_l2_drift_post_anchor_target_prompt": "2.0",
            "target_parameter_l2_drift_post_anchor_monthly_gain": "0.4",
            "target_parameter_l2_drift_post_anchor_adapter_coeff_bottleneck": "0.3",
            "target_parameter_l2_drift_post_anchor_adapter_coeff_dec2": "0.4",
            "target_parameter_l2_drift_post_anchor_adapter_coeff_dec1": "0.0",
        },
        {
            "episode_id": "US-R3_e0",
            "query_role": "source_val",
            "split_type": "source_val",
            "K": 4,
            "schedule_label": "original_K4",
            "target_parameter_l2_drift_post_anchor_total": "6.0",
            "target_parameter_l2_drift_post_anchor_target_prompt": "4.0",
            "target_parameter_l2_drift_post_anchor_monthly_gain": "0.8",
            "target_parameter_l2_drift_post_anchor_adapter_coeff_bottleneck": "0.6",
            "target_parameter_l2_drift_post_anchor_adapter_coeff_dec2": "0.8",
            "target_parameter_l2_drift_post_anchor_adapter_coeff_dec1": "0.0",
        },
        {
            "episode_id": "US-R9_target_eval_diag",
            "query_role": "target_eval",
            "split_type": "target_eval",
            "K": 4,
            "schedule_label": "original_K4",
            "target_parameter_l2_drift_post_anchor_total": "1000.0",
        },
    ]

    radii = calib.derive_trust_radii(rows)

    assert radii["mild_groupwise"]["total"] == pytest.approx(5.0)
    assert radii["mild_groupwise"]["prompt"] == pytest.approx(3.0)
    assert radii["mild_groupwise"]["gain"] == pytest.approx(0.6)
    assert radii["mild_groupwise"]["coeff"] == pytest.approx(0.75)
    assert radii["strong_groupwise"]["total"] == pytest.approx(2.5)
    assert radii["strong_groupwise"]["prompt"] == pytest.approx(1.5)


def test_calibration_audit_reports_counts_missing_candidates_and_hashes(tmp_path):
    from scripts.eval import calibrate_source_safe_guard as calib

    rows = []
    for episode, seed in [("US-R2_e0", 0), ("US-R3_e1", 1)]:
        rows.append({**_row(episode, "K0_identity", 0.10, K=0, schedule_label="identity_base"), "seed": seed})
        rows.append({**_row(episode, "K4_original", 0.12, K=4, schedule_label="original_K4"), "seed": seed})
    rows.append({**_row("US-R2_e0", "A", 0.20), "seed": 0})
    row_path = _write_rows(tmp_path / "rows.csv", rows)
    loaded = calib.load_calibration_rows([row_path])
    summaries = calib.score_candidates(loaded)
    audit = calib.build_calibration_audit_metadata(loaded, [row_path], summaries, checkpoint_source_regions=["US-R2", "US-R3"])

    assert audit["row_files"] == [str(row_path)]
    assert audit["row_file_hashes"][str(row_path)]
    assert audit["row_count"] == len(rows)
    assert audit["source_pseudo_target_episode_count"] == 2
    assert audit["pseudo_target_regions"] == ["US-R2", "US-R3"]
    assert audit["source_regions"] == ["US-R2", "US-R3"]
    assert audit["seeds"] == [0, 1]
    assert audit["candidate_count_observed"] == 1
    assert audit["candidate_count_expected"] == len(calib.enumerate_guard_candidates())
    assert "A" in audit["unexpected_candidate_ids"]
    assert "scope_coeff_gain__schedule_original_K12__loss_global_pixel__rho_fixed_1.0__trust_none" in audit["missing_candidate_ids"]
    assert audit["candidate_episode_ratio"] == pytest.approx(1 / 2)


def test_resume_manifest_splits_completed_missing_and_hash_mismatches(tmp_path):
    from scripts.eval import calibrate_source_safe_guard as calib

    candidates = calib.enumerate_guard_candidates(candidate_set="compact_v1")[:2]
    row_root = tmp_path / "rows"
    completed = candidates[0]
    mismatched = candidates[1]
    pred_path = tmp_path / "valid_prediction_records.jsonl"
    pred_path.write_text('{"ok": true}\n', encoding="utf-8")
    valid_dir = row_root / "US-R2" / completed["candidate_id"]
    valid_dir.mkdir(parents=True)
    _write_rows(
        valid_dir / "source_safe_candidate_rows.csv",
        [
            {
                **_row("US-R2_e0", completed["candidate_id"], 0.12),
                "candidate_config_hash": calib.candidate_config_hash(completed),
                "base_config_id": completed["base_config_id"],
                "prediction_record_path": str(pred_path),
                "prediction_content_hash": "pred-hash",
                "calib_max_query_samples": 256,
                "seed": 0,
            }
        ],
    )
    mismatch_dir = row_root / "US-R2" / mismatched["candidate_id"]
    mismatch_dir.mkdir(parents=True)
    _write_rows(
        mismatch_dir / "source_safe_candidate_rows.csv",
        [
            {
                **_row("US-R2_e0", mismatched["candidate_id"], 0.12),
                "candidate_config_hash": "old-hash",
                "base_config_id": mismatched["base_config_id"],
                "prediction_record_path": str(pred_path),
                "prediction_content_hash": "pred-hash",
                "calib_max_query_samples": 256,
                "seed": 0,
            }
        ],
    )

    manifest = calib.build_resume_manifest(
        base_candidates=candidates,
        pseudo_target_regions=["US-R2", "US-R3"],
        source_rows_root=row_root,
        base_command_prefix="RESUME=1 SKIP_EXISTING=1 bash run/phase5_hyperda_p2_8_source_safe_guard_calibration.sh",
        expected_sample_budget=256,
    )

    completed_ids = {(row["pseudo_target_region"], row["candidate_id"]) for row in manifest["completed_rows"]}
    missing_ids = {(row["pseudo_target_region"], row["candidate_id"]) for row in manifest["missing_rows"]}
    assert ("US-R2", completed["candidate_id"]) in completed_ids
    assert ("US-R2", mismatched["candidate_id"]) in missing_ids
    assert ("US-R3", completed["candidate_id"]) in missing_ids
    assert manifest["estimated_remaining_rows"] == 3
    assert "config_hash_mismatch" in {row["artifact_status"] for row in manifest["missing_rows"]}
    assert "config_hash_mismatch" in {row["artifact_status"] for row in manifest["invalid_existing_rows"]}
    assert "RESUME=1 SKIP_EXISTING=1" in manifest["resume_commands_md"]


def test_resume_manifest_requires_prediction_records_hashes_source_roles_and_sample_budget(tmp_path):
    from scripts.eval import calibrate_source_safe_guard as calib

    candidate = calib.enumerate_guard_base_configs(candidate_set="compact_v1")[0]
    row_root = tmp_path / "rows"
    invalid_dir = row_root / "US-R2" / candidate["candidate_id"]
    invalid_dir.mkdir(parents=True)
    _write_rows(
        invalid_dir / "source_safe_candidate_rows.csv",
        [
            {
                **_row("US-R2_e0", candidate["candidate_id"], 0.12),
                "candidate_config_hash": calib.candidate_config_hash(candidate),
                "base_config_id": candidate["base_config_id"],
                "prediction_record_path": str(tmp_path / "missing.jsonl"),
                "prediction_content_hash": "",
                "calib_max_query_samples": 128,
            }
        ],
    )
    old_baseline_dir = row_root / "US-R2" / "K0_identity"
    old_baseline_dir.mkdir(parents=True)
    _write_rows(
        old_baseline_dir / "source_safe_candidate_rows.csv",
        [_row("US-R2_e0", "K0_identity", 0.1, K=0, schedule_label="identity_base")],
    )

    manifest = calib.build_resume_manifest(
        base_candidates=[calib.baseline_gpu_row_configs()[0], candidate],
        pseudo_target_regions=["US-R2"],
        source_rows_root=row_root,
        base_command_prefix="rerun",
        expected_sample_budget=256,
    )

    assert manifest["completed_rows"] == []
    assert len(manifest["missing_rows"]) == 2
    invalid_reasons = " ".join(row["reason"] for row in manifest["invalid_existing_rows"])
    assert "prediction_record_path_missing" in invalid_reasons
    assert "missing_prediction_content_hash" in invalid_reasons
    assert "sample_budget_mismatch" in invalid_reasons
    assert "missing_candidate_config_hash" in invalid_reasons


def test_strict_existing_row_accepts_repo_relative_prediction_record_path(tmp_path):
    from scripts.eval import calibrate_source_safe_guard as calib

    record_path = tmp_path / "prediction_records.jsonl"
    record_path.write_text("{}\n", encoding="utf-8")
    row_file = tmp_path / "row_dir" / "source_safe_candidate_rows.csv"
    row_file.parent.mkdir()
    relative_record = Path(__import__("os").path.relpath(record_path, Path.cwd()))
    _write_rows(
        row_file,
        [
            {
                **_row("US-R2_e0", "K0_identity", 0.1, K=0, schedule_label="identity_base"),
                "seed": 0,
                "candidate_config_hash": "K0_identity_static",
                "prediction_record_path": str(relative_record),
                "prediction_content_hash": "pred-hash",
                "calib_max_query_samples": 256,
            }
        ],
    )

    row = calib.load_calibration_rows([row_file])[0]

    assert calib._strict_existing_row_invalid_reasons(
        row,
        expected_hash="K0_identity_static",
        expected_sample_budget=256,
    ) == []


def test_top_candidate_ids_map_to_unique_required_base_configs():
    from scripts.eval import calibrate_source_safe_guard as calib

    logical = calib.enumerate_guard_candidates(candidate_set="compact_v1")
    c0_ids = [row["candidate_id"] for row in logical if row["base_config_id"] == "C0_coeff_gain"][:3]
    c2_id = next(row["candidate_id"] for row in logical if row["base_config_id"] == "C2_coeff_gain")

    required = calib.base_configs_for_logical_candidate_ids(
        candidate_set="compact_v1",
        candidate_ids=[*c0_ids, c2_id],
    )

    assert [row["base_config_id"] for row in required] == ["C0_coeff_gain", "C2_coeff_gain"]
    assert all(row["rho_policy"] == "fixed_1.0" for row in required)


def test_calibration_writes_stage_alias_artifacts(tmp_path):
    rows = [
        _row("US-R2_e0", "K0_identity", 0.10, K=0, schedule_label="identity_base"),
        _row(
            "US-R2_e0",
            "schedule_original_K12__loss_global_pixel__rho_fixed_1.0__trust_none",
            0.20,
        ),
    ]
    row_path = _write_rows(tmp_path / "rows.csv", rows)
    output = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            "scripts/eval/calibrate_source_safe_guard.py",
            "--calibration_rows",
            str(row_path),
            "--output_dir",
            str(output),
            "--allow_in_checkpoint_source_episodes",
            "--candidate_set",
            "compact_v1",
            "--calibration_stage",
            "coarse",
            "--pseudo_target_regions",
            "US-R2",
            "--source_rows_root",
            str(tmp_path / "missing_rows_root"),
        ],
        check=True,
    )

    for name in [
        "coarse_source_safe_calibration_summary.json",
        "coarse_source_safe_calibration_summary.csv",
        "coarse_source_safe_calibration_summary.md",
        "top5_candidates.csv",
        "leave_one_source_region_out_stability.csv",
        "invalid_existing_rows.csv",
    ]:
        assert (output / name).exists()


def test_two_stage_helpers_use_deterministic_subset_top5_and_final_ranking():
    from scripts.eval import calibrate_source_safe_guard as calib

    summaries = []
    for idx in range(7):
        summaries.append(
            {
                "candidate_id": f"C{idx}",
                "score": float(7 - idx),
                "negative_transfer_rate_vs_K0": 0.0,
                "schedule_label": "original_K12",
                "support_loss_reduction": "global_pixel",
                "rho_policy": "fixed_1.0",
                "trust_policy": "none",
                "episode_results": [{"pseudo_target_region": "US-R2", "delta_vs_K0": float(7 - idx)}],
            }
        )

    subset_hash_a = calib.deterministic_source_subset_hash(
        final_target_region="US-R1",
        seed=0,
        pseudo_target_regions=["US-R4", "US-R2"],
        source_query_max_samples=256,
    )
    subset_hash_b = calib.deterministic_source_subset_hash(
        final_target_region="US-R1",
        seed=0,
        pseudo_target_regions=["US-R2", "US-R4"],
        source_query_max_samples=256,
    )
    assert subset_hash_a == subset_hash_b

    top5 = calib.stage_top_candidate_ids(summaries, top_k=5)
    assert top5 == ["C0", "C1", "C2", "C3", "C4"]

    final = calib.filter_summaries_to_candidate_ids(
        [
            {**summaries[5], "score": 100.0},
            {**summaries[2], "score": 2.0},
            {**summaries[1], "score": 3.0},
        ],
        top5,
    )
    assert [row["candidate_id"] for row in calib.rank_candidates(final)] == ["C1", "C2"]


def test_stability_diagnostics_report_top5_gaps_and_leave_one_region_out():
    from scripts.eval import calibrate_source_safe_guard as calib

    summaries = [
        {
            "candidate_id": "A",
            "score": 0.40,
            "negative_transfer_rate_vs_K0": 0.0,
            "schedule_label": "original_K12",
            "support_loss_reduction": "global_pixel",
            "rho_policy": "fixed_1.0",
            "trust_policy": "none",
            "episode_results": [
                {"episode_id": "US-R2_e0", "pseudo_target_region": "US-R2", "delta_vs_K0": 0.4},
                {"episode_id": "US-R3_e0", "pseudo_target_region": "US-R3", "delta_vs_K0": 0.4},
            ],
        },
        {
            "candidate_id": "B",
            "score": 0.39,
            "negative_transfer_rate_vs_K0": 0.0,
            "schedule_label": "original_K12",
            "support_loss_reduction": "global_pixel",
            "rho_policy": "fixed_0.5",
            "trust_policy": "mild_groupwise",
            "episode_results": [
                {"episode_id": "US-R2_e0", "pseudo_target_region": "US-R2", "delta_vs_K0": 0.9},
                {"episode_id": "US-R3_e0", "pseudo_target_region": "US-R3", "delta_vs_K0": 0.2},
            ],
        },
        {
            "candidate_id": "C",
            "score": 0.10,
            "negative_transfer_rate_vs_K0": 0.0,
            "schedule_label": "K4_schedule_on_K12",
            "support_loss_reduction": "cycle_balanced",
            "rho_policy": "rule_a",
            "trust_policy": "strong_groupwise",
            "episode_results": [
                {"episode_id": "US-R2_e0", "pseudo_target_region": "US-R2", "delta_vs_K0": 0.1},
                {"episode_id": "US-R3_e0", "pseudo_target_region": "US-R3", "delta_vs_K0": 0.1},
            ],
        },
    ]

    diagnostics = calib.compute_stability_diagnostics(summaries)

    assert [row["candidate_id"] for row in diagnostics["top5_candidates"]] == ["A", "B", "C"]
    assert diagnostics["score_gap_top1_top2"] == pytest.approx(0.01)
    assert diagnostics["score_gap_top1_top5"] == pytest.approx(0.30)
    assert diagnostics["leave_one_source_region_out_enabled"] is True
    by_region = {row["heldout_source_region"]: row for row in diagnostics["leave_one_source_region_out"]}
    assert by_region["US-R2"]["selected_candidate_id"] == "A"
    assert by_region["US-R3"]["selected_candidate_id"] == "B"
    assert diagnostics["leave_one_source_region_out_selection_counts"] == {"A": 1, "B": 1}


def test_guard_config_hash_is_stable_and_selected_schema_is_locked():
    from scripts.eval import calibrate_source_safe_guard as calib

    candidate = {
        "candidate_id": "schedule_original_K12__loss_global_pixel__rho_fixed_0.5__trust_mild_groupwise",
        "schedule_label": "original_K12",
        "adapt_scope": "all",
        "adapt_solver": "adamw",
        "support_loss_reduction": "global_pixel",
        "rho_policy": "fixed_0.5",
        "trust_policy": "mild_groupwise",
        "lr": 3e-4,
        "adaptation_steps": 80,
        "anchor_alpha": 0.25,
    }
    summary = {"score": 0.1, "negative_transfer_rate_vs_K0": 0.0}
    radii = {
        "mild_groupwise": {"total": 5.0, "prompt": 3.0, "gain": 0.6, "coeff": 0.75, "spatial": 0.0},
        "strong_groupwise": {"total": 2.5, "prompt": 1.5, "gain": 0.3, "coeff": 0.375, "spatial": 0.0},
        "none": {"total": 0.0, "prompt": 0.0, "gain": 0.0, "coeff": 0.0, "spatial": 0.0},
    }

    config = calib.build_selected_guard_config(
        candidate=candidate,
        ranking_summary=summary,
        trust_radii=radii,
        final_target_region="US-R1",
        seed=0,
        evidence_level="source_safe_in_checkpoint_weaker",
        source_metadata={"checkpoint_hashes": ["abc"]},
    )
    reordered = json.loads(json.dumps(config, sort_keys=True))

    assert config["schema_version"] == "p2_8_source_safe_guard_config_v1"
    assert config["calibration_mode"] == "in_checkpoint_source_dev"
    assert config["paper_grade_source_heldout"] is False
    assert config["selection_query_role"] == "source_val_pseudo_query_only"
    assert config["target_eval_usage"] == "never_read_by_calibration"
    assert config["trust_region_mode"] == "groupwise"
    assert config["trust_total_radius"] == pytest.approx(5.0)
    assert config["adapt_mix_rho"] == pytest.approx(0.5)
    assert config["guard_config_hash"] == calib.compute_guard_config_hash(reordered)

    changed = dict(config)
    changed["adapt_mix_rho"] = 0.25
    changed.pop("guard_config_hash")
    assert config["guard_config_hash"] != calib.compute_guard_config_hash(changed)


def test_build_safe_policy_json_from_source_calibration_config():
    from scripts.eval import calibrate_source_safe_guard as calib

    selected_config = {
        "candidate_id": "schedule_original_K12__loss_cycle_balanced__rho_fixed_1.0__trust_none",
        "guard_config_hash": "guardhash",
        "adapt_scope": "coeff_gain",
        "adapt_solver": "adamw",
        "lr": 3e-4,
        "adaptation_steps": 80,
        "anchor_alpha": 0.25,
        "support_loss_reduction": "cycle_balanced",
        "trust_region_mode": "groupwise",
        "trust_total_radius": 2.0,
        "trust_prompt_radius": 1.0,
        "trust_gain_radius": 0.25,
        "trust_coeff_radius": 0.5,
        "trust_spatial_radius": 0.0,
        "source_safety_evidence_level": "source_heldout_strict",
        "source_metadata": {"pseudo_target_regions": ["US-R2", "US-R3"]},
    }

    policy = calib.build_safe_policy_json(
        selected_config,
        final_target_region="US-R1",
        seed=0,
    )

    assert policy["schema_version"] == "hyperda_safe_policy_v1"
    assert policy["policy_source"] == "source_side_episode_calibration"
    assert policy["target_val_usage"] == "unused_in_main_protocol"
    assert policy["target_eval_usage"] == "final_eval_only_no_selection"
    assert policy["source_episode_regions"] == ["US-R2", "US-R3"]
    assert policy["source_calibration"]["guard_config_hash"] == "guardhash"
    assert policy["policies"]["few_shot_k4"]["adapt_scope"] == "coeff_gain"
    assert policy["policies"]["few_shot_k4"]["adaptation_steps"] == 80
    assert policy["policies"]["few_shot_k12"]["support_loss_reduction"] == "cycle_balanced"
    assert policy["policy_hash"] == calib.compute_safe_policy_hash(policy)

    changed = json.loads(json.dumps(policy, sort_keys=True))
    changed["policies"]["few_shot_k12"]["anchor_alpha"] = 0.5
    changed.pop("policy_hash")
    assert policy["policy_hash"] != calib.compute_safe_policy_hash(changed)


def test_per_k_source_safe_policy_can_select_different_k4_and_k12_configs():
    from scripts.eval import calibrate_source_safe_guard as calib

    rows = []
    for episode in ("US-R2_e0", "US-R3_e0"):
        rows.append(_row(episode, "K0_identity", 0.50, K=0, schedule_label="identity_base"))
        rows.append(
            {
                **_row(
                    episode,
                    "K4_conservative_coeff_only",
                    0.52,
                    K=4,
                    schedule_label="source_safe_K4_conservative",
                    rho_policy="fixed_0.5",
                    trust_policy="strong_groupwise",
                ),
                "adapt_scope": "coeff_only",
                "lr": 3e-4,
                "adaptation_steps": 20,
                "anchor_alpha": 0.25,
                "support_loss_reduction": "cycle_balanced",
                "target_parameter_l2_drift_post_anchor_total": 0.10,
            }
        )
        rows.append(
            {
                **_row(
                    episode,
                    "K4_aggressive_surface_only",
                    0.55,
                    K=4,
                    schedule_label="source_safe_K4_aggressive",
                    rho_policy="fixed_1.0",
                ),
                "adapt_scope": "safe_operator",
                "surface_skill_primary": 0.70,
                "rootzone_skill_primary": 0.40,
                "lr": 1e-3,
                "adaptation_steps": 100,
                "anchor_alpha": 0.75,
                "target_parameter_l2_drift_post_anchor_total": 5.0,
            }
        )
        rows.append(
            {
                **_row(
                    episode,
                    "K12_cycle_balanced_coeff_gain",
                    0.60,
                    K=12,
                    schedule_label="source_safe_K12_cycle_balanced",
                    rho_policy="fixed_0.75",
                    trust_policy="mild_groupwise",
                ),
                "adapt_scope": "coeff_gain",
                "lr": 2e-4,
                "adaptation_steps": 60,
                "anchor_alpha": 0.20,
                "support_loss_reduction": "cycle_balanced",
                "target_parameter_l2_drift_post_anchor_total": 0.50,
            }
        )

    selected_by_k = calib.select_per_k_safe_policy_configs(
        rows,
        trust_radii=calib.derive_trust_radii(rows),
        final_target_region="US-R1",
        seed=0,
        evidence_level="source_heldout_strict",
        source_metadata={"pseudo_target_regions": ["US-R2", "US-R3"]},
    )
    policy = calib.build_safe_policy_json(
        selected_by_k[12],
        final_target_region="US-R1",
        seed=0,
        selected_configs_by_k=selected_by_k,
    )

    k4_policy = policy["policies"]["few_shot_k4"]
    k12_policy = policy["policies"]["few_shot_k12"]
    assert k4_policy["adapt_scope"] == "coeff_only"
    assert k4_policy["adaptation_steps"] == 20
    assert k4_policy["anchor_alpha"] == pytest.approx(0.25)
    assert k4_policy["adapt_mix_rho"] == pytest.approx(0.5)
    assert k12_policy["adapt_scope"] == "coeff_gain"
    assert k12_policy["support_loss_reduction"] == "cycle_balanced"
    assert k12_policy["adaptation_steps"] == 60
    assert k12_policy["anchor_alpha"] == pytest.approx(0.20)
    assert k4_policy != k12_policy
    assert policy["source_calibration"]["selected_config_by_k"]["4"]["candidate_id"] == "K4_conservative_coeff_only"
    assert policy["source_calibration"]["selected_config_by_k"]["12"]["candidate_id"] == "K12_cycle_balanced_coeff_gain"
    assert policy["target_val_usage"] == "unused_in_main_protocol"
    assert policy["target_eval_usage"] == "final_eval_only_no_selection"
    assert policy["policy_hash"] == calib.compute_safe_policy_hash(policy)


def test_expand_prediction_records_applies_rho_candidates_to_k4_and_k12(tmp_path, monkeypatch):
    from scripts.eval import calibrate_source_safe_guard as calib

    k0_path = tmp_path / "k0.jsonl"
    k4_path = tmp_path / "k4.jsonl"
    k12_path = tmp_path / "k12.jsonl"
    for path in (k0_path, k4_path, k12_path):
        path.write_text("{}\n", encoding="utf-8")

    def fake_mix(k0_record_path, adapted_record_path, *, rho, candidate_id):
        return {
            "summary": {
                "surface": {"skill_primary": 0.1 + float(rho)},
                "rootzone": {"skill_primary": 0.2 + float(rho)},
            },
            "mixed_prediction_content_hash": f"mixed-{adapted_record_path.name}-{rho:g}",
            "source_prediction_record_hash": "source-hash",
            "metric_content_hash": "metric-hash",
            "metric_values_content_hash": "metric-values-hash",
        }

    import scripts.eval.mix_prediction_records as mixer

    monkeypatch.setattr(mixer, "mix_prediction_record_files", fake_mix)

    rows = [
        {
            **_row("US-R2_e0", "K0_identity", 0.10, K=0, schedule_label="identity_base"),
            "prediction_record_path": str(k0_path),
        },
        {
            **_row(
                "US-R2_e0",
                "K4_base",
                0.11,
                K=4,
                schedule_label="source_safe_K4_conservative",
            ),
            "adapt_scope": "coeff_only",
            "prediction_record_path": str(k4_path),
        },
        {
            **_row(
                "US-R2_e0",
                "K12_base",
                0.12,
                K=12,
                schedule_label="source_safe_K12_conservative",
            ),
            "adapt_scope": "coeff_gain",
            "prediction_record_path": str(k12_path),
        },
    ]

    expanded = calib.expand_logical_candidate_rows_from_prediction_records(
        rows,
        candidate_set="stage3_conservative_v1",
    )

    expanded_by_k = {4: [], 12: []}
    for row in expanded:
        if row.get("logical_offline_mix") is True:
            expanded_by_k[int(row["K"])].append(row)

    assert {row["rho_policy"] for row in expanded_by_k[4]} >= {"fixed_0.75", "fixed_0.5", "rule_a"}
    assert {row["rho_policy"] for row in expanded_by_k[12]} >= {"fixed_0.75", "fixed_0.5", "rule_a"}
    assert all(row["adapt_mix_rho"] is not None for row in expanded_by_k[4] + expanded_by_k[12])


def test_per_k_scoring_allows_source_calibrated_no_update_when_k4_overupdates():
    from scripts.eval import calibrate_source_safe_guard as calib

    rows = []
    for episode in ("US-R2_e0", "US-R3_e0"):
        rows.append(_row(episode, "K0_identity", 0.50, K=0, schedule_label="identity_base"))
        rows.append(
            {
                **_row(episode, "K4_overupdate", 0.55, K=4, schedule_label="source_safe_K4_aggressive"),
                "surface_skill_primary": 0.75,
                "rootzone_skill_primary": 0.35,
                "adapt_scope": "safe_operator",
                "lr": 1e-3,
                "adaptation_steps": 100,
                "anchor_alpha": 0.75,
                "target_parameter_l2_drift_post_anchor_total": 10.0,
            }
        )

    ranked = calib.rank_candidates(calib.score_candidates_for_k(rows, K=4, include_no_update=True))

    assert ranked[0]["candidate_id"] == "K4_source_calibrated_no_update"
    assert ranked[0]["adapt_scope"] == "none"
    assert ranked[0]["adaptation_steps"] == 0
    assert ranked[0]["anchor_alpha"] == pytest.approx(0.0)


def test_p2_8_probe_wrappers_are_removed_from_active_run_entries():
    retired_wrappers = [
        "run/phase5_hyperda_p2_8_cycle_balanced_diagnostics.sh",
        "run/phase5_hyperda_p2_8_source_safe_guard_calibration.sh",
        "run/phase5_hyperda_p2_8_locked_guard_target_eval.sh",
        "run/phase5_hyperda_p2_6_schedule_drift_probe.sh",
        "run/phase5_hyperda_p2_7_conservative_guard_probe.sh",
    ]

    assert not [path for path in retired_wrappers if Path(path).exists()]

    calibration = Path("scripts/eval/calibrate_source_safe_guard.py").read_text()
    assert "FORBIDDEN_TARGET_EVAL_ROLES" in calibration
    assert "target_eval" in calibration
    assert "target_query" in calibration
    assert "target_val" in calibration
    assert "target_full_train" in calibration
    assert "bash run/phase5_hyperda_p2_8_source_safe_guard_calibration.sh" not in calibration
    assert "python scripts/eval/calibrate_source_safe_guard.py" in calibration


def test_evaluate_checkpoint_exposes_source_only_active_region_override():
    text = Path("scripts/eval/evaluate_checkpoint.py").read_text()

    assert "--active_region_override" in text
    assert "valid only for source_* splits" in text
    assert 'args.split_type not in ("source_train", "source_fit", "source_val", "source_test")' in text
    assert "dataset.set_active_region(args.active_region_override)" in text
