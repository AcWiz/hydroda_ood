from __future__ import annotations

import csv
import json
from pathlib import Path


def _write_run(
    root: Path,
    *,
    ablation_id: str,
    target_region: str = "US-R1",
    seed: int = 0,
    best_selection_value: float,
    trainable_count: int,
    hyper_coeff_generator: str,
    hyper_adapter_param_style: str = "basis_1x1",
    summary_nested: bool = False,
    hyper_reliability_gate: str,
    hyper_enable_film: bool,
    hyper_enable_adapters: bool,
    zero_shot_prior_form: str = "direct_hyper",
    hyper_source_saliency_prior_application: str = "",
    hyper_residual_magnitude_penalty: float = 0.0,
    hyper_coeff_entropy_floor: float = 0.0,
    hyper_coeff_entropy_penalty: float = 0.0,
    summary_extra: dict | None = None,
) -> Path:
    run_dir = root / ablation_id / target_region / f"run_{ablation_id}_{target_region}_s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = run_dir / "reports" if summary_nested else run_dir
    summary_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "checkpoint_best_source_val_transfer_safe_score.pt").write_text(
        "stub",
        encoding="utf-8",
    )
    payload = {
        "experiment_id": run_dir.name,
        "target_region": target_region,
        "seed": seed,
        "best_selection_metric": "source_val_transfer_safe_score",
        "best_selection_value": best_selection_value,
        "best_safe_score": best_selection_value,
        "trainable_parameter_count": trainable_count,
        "trainable_parameter_names": ["prompt_encoder.foo", "model.hyper_adapter_b.bases.0.up.weight"],
        "hyper_coeff_generator": hyper_coeff_generator,
        "hyper_adapter_param_style": hyper_adapter_param_style,
        "hyper_rank_gate_temperature_init": 2.0 if "stable" in hyper_coeff_generator else 1.0,
        "hyper_reliability_gate": hyper_reliability_gate,
        "hyper_reliability_init": 0.95,
        "hyper_source_saliency_prior_application": hyper_source_saliency_prior_application,
        "zero_shot_prior_form": zero_shot_prior_form,
        "source_residual_rho": 0.5 if zero_shot_prior_form != "direct_hyper" else 1.0,
        "zero_shot_rho_selection_source": "source_val_regionwise_safe_episode_only",
        "hyper_residual_magnitude_penalty": hyper_residual_magnitude_penalty,
        "hyper_coeff_entropy_floor": hyper_coeff_entropy_floor,
        "hyper_coeff_entropy_penalty": hyper_coeff_entropy_penalty,
        "hyper_enable_film": hyper_enable_film,
        "hyper_enable_adapters": hyper_enable_adapters,
        "source_base_checkpoint_sha256": "abc123",
        "init_from_source_base_checkpoint": "/source.pt",
        "split_manifest_sha256": "split123",
        "protocol_freeze_id": "hyperda_v4_4_zero_few_shot_generalization_2015_2025_context2015_2021_sourceval2022_eval2023_2025",
        "normalization_source": "source_fit_only_from_source_checkpoint",
    }
    if summary_extra:
        payload.update(summary_extra)
    (summary_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return run_dir


def test_build_hyperda_staged_ablation_table_collects_source_stage_summaries(tmp_path):
    from scripts.analysis.build_hyperda_staged_ablation_table import build_hyperda_staged_ablation_table

    runs_root = tmp_path / "runs"
    _write_run(
        runs_root,
        ablation_id="M0_current",
        best_selection_value=0.10,
        trainable_count=100,
        hyper_coeff_generator="per_adapter",
        hyper_reliability_gate="none",
        hyper_enable_film=True,
        hyper_enable_adapters=True,
    )
    _write_run(
        runs_root,
        ablation_id="M2_shared_coeff_gate",
        best_selection_value=0.25,
        trainable_count=120,
        hyper_coeff_generator="shared_layer_aware",
        hyper_reliability_gate="prompt_scalar",
        hyper_enable_film=True,
        hyper_enable_adapters=True,
    )
    _write_run(
        runs_root,
        ablation_id="M2_rank_gated_dora",
        best_selection_value=0.05,
        trainable_count=125,
        hyper_coeff_generator="shared_layer_aware_rank_gated",
        hyper_adapter_param_style="dora_like_gain",
        hyper_reliability_gate="prompt_scalar",
        hyper_enable_film=True,
        hyper_enable_adapters=True,
    )
    _write_run(
        runs_root,
        ablation_id="M2_1_rank_gated_dora_stable",
        best_selection_value=0.35,
        trainable_count=125,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_adapter_param_style="dora_like_gain_bounded",
        summary_nested=True,
        hyper_reliability_gate="prompt_scalar",
        hyper_enable_film=True,
        hyper_enable_adapters=True,
    )
    _write_run(
        runs_root,
        ablation_id="M2_3_source_safe_residual_hyperda",
        best_selection_value=0.34,
        trainable_count=128,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        hyper_enable_film=True,
        hyper_enable_adapters=True,
        zero_shot_prior_form="source_base_residual_reliability_gated",
        hyper_source_saliency_prior_application="soft_regularization_metadata",
        hyper_residual_magnitude_penalty=0.001,
        hyper_coeff_entropy_floor=0.5,
        hyper_coeff_entropy_penalty=0.0001,
    )
    _write_run(
        runs_root,
        ablation_id="M2_4_target_context_conservative_hyperda",
        best_selection_value=0.40,
        trainable_count=128,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        hyper_enable_film=True,
        hyper_enable_adapters=True,
        zero_shot_prior_form="source_base_residual_reliability_gated",
    )

    result = build_hyperda_staged_ablation_table(
        runs_root=runs_root,
        output_dir=tmp_path / "reports",
        target_region="US-R1",
        seed=0,
    )

    csv_path = result["csv_path"]
    md_path = result["md_path"]
    json_path = result["json_path"]
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))

    assert csv_path.name == "US-R1_s0_source_stage.csv"
    assert md_path.name == "US-R1_s0_source_stage.md"
    assert json_path.name == "US-R1_s0_source_stage.json"
    assert [row["ablation_id"] for row in rows] == [
        "M0_current",
        "M2_shared_coeff_gate",
        "M2_rank_gated_dora",
        "M2_1_rank_gated_dora_stable",
        "M2_3_source_safe_residual_hyperda",
    ]
    assert rows[0]["best_selection_metric"] == "source_val_transfer_safe_score"
    assert rows[0]["hyper_coeff_generator"] == "per_adapter"
    assert rows[1]["hyper_coeff_generator"] == "shared_layer_aware"
    assert rows[1]["hyper_reliability_gate"] == "prompt_scalar"
    assert rows[2]["hyper_adapter_param_style"] == "dora_like_gain"
    assert rows[3]["hyper_coeff_generator"] == "shared_layer_aware_rank_gated_stable"
    assert rows[3]["hyper_adapter_param_style"] == "dora_like_gain_bounded"
    assert rows[3]["rank_by_best_selection_value"] == "1"
    assert "reports/summary.json" in rows[3]["summary_json"]
    assert "checkpoint_best_source_val_transfer_safe_score.pt" in rows[3]["best_checkpoint"]
    assert rows[4]["zero_shot_prior_form"] == "source_base_residual_reliability_gated"
    assert rows[4]["hyper_source_saliency_prior_application"] == "soft_regularization_metadata"
    assert rows[4]["hyper_residual_magnitude_penalty"] == "0.001"
    assert rows[4]["hyper_coeff_entropy_floor"] == "0.5"
    assert rows[4]["hyper_coeff_entropy_penalty"] == "0.0001"
    assert "M2_1_rank_gated_dora_stable" in md_path.read_text(encoding="utf-8")
    assert "M2_3_source_safe_residual_hyperda" in md_path.read_text(encoding="utf-8")
    assert "M2_4_target_context_conservative_hyperda" not in md_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["target_region"] == "US-R1"
    assert payload["seed"] == 0
    assert payload["row_count"] == 5


def test_build_hyperda_staged_ablation_table_includes_hyperda_trust_candidates(tmp_path):
    from scripts.analysis.build_hyperda_staged_ablation_table import build_hyperda_staged_ablation_table

    runs_root = tmp_path / "runs"
    _write_run(
        runs_root,
        ablation_id="M3_1_hyperda_trust_medium",
        best_selection_value=0.44657339054928213,
        trainable_count=144042,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        hyper_enable_film=True,
        hyper_enable_adapters=True,
        zero_shot_prior_form="source_base_residual_reliability_gated",
    )
    _write_run(
        runs_root,
        ablation_id="M3_1a_trust_medium_dualalpha",
        best_selection_value=0.4380682817986459,
        trainable_count=144042,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        hyper_enable_film=True,
        hyper_enable_adapters=True,
        zero_shot_prior_form="source_base_residual_reliability_gated",
    )
    _write_run(
        runs_root,
        ablation_id="M3_1d_trust_medium_broad",
        best_selection_value=0.447,
        trainable_count=144042,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        hyper_enable_film=True,
        hyper_enable_adapters=True,
        zero_shot_prior_form="source_base_residual_reliability_gated",
    )
    _write_run(
        runs_root,
        ablation_id="M3_unregistered_future",
        best_selection_value=0.99,
        trainable_count=144042,
        hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
        hyper_adapter_param_style="dora_like_gain_bounded",
        hyper_reliability_gate="prompt_scalar",
        hyper_enable_film=True,
        hyper_enable_adapters=True,
        zero_shot_prior_form="source_base_residual_reliability_gated",
    )

    result = build_hyperda_staged_ablation_table(
        runs_root=runs_root,
        output_dir=tmp_path / "reports",
        target_region="US-R1",
        seed=0,
    )

    rows = list(csv.DictReader(result["csv_path"].open(encoding="utf-8", newline="")))
    assert [row["ablation_id"] for row in rows] == [
        "M3_1_hyperda_trust_medium",
        "M3_1a_trust_medium_dualalpha",
        "M3_1d_trust_medium_broad",
    ]
    assert rows[0]["rank_by_best_selection_value"] == "2"
    assert rows[2]["rank_by_best_selection_value"] == "1"
    markdown = result["md_path"].read_text(encoding="utf-8")
    assert "M3_1_hyperda_trust_medium" in markdown
    assert "M3_unregistered_future" not in markdown


def test_build_hyperda_staged_ablation_table_includes_physics_informed_trust_candidates(tmp_path):
    from scripts.analysis.build_hyperda_staged_ablation_table import build_hyperda_staged_ablation_table

    runs_root = tmp_path / "runs"
    for ablation_id, score in [
        ("M3_1_hyperda_trust_medium", 0.44657339054928213),
        ("M3_8_phys_formula_operator_trust", 0.442),
        ("M3_8b_phys_formula_light_guarded_trust", 0.445),
        ("M3_8c_phys_formula_light_operator_only_trust", 0.444),
        ("M3_12_phys_gain_basis_hypertrust", 0.443),
        ("M3_13_phys_gain_guarded_hypertrust", 0.446),
        ("M3_14_source_trained_phys_formula_gain_hypertrust", 0.44455),
        ("M3_15_m31_anchored_source_safe_phys_coeff_delta", 0.4477),
        ("M3_16_source_only_phys_m3trust_lite", 0.4480),
    ]:
        summary_extra = {}
        if ablation_id == "M3_12_phys_gain_basis_hypertrust":
            summary_extra = {
                "hyper_phys_gain_basis_residual": True,
                "phys_gain_source_bank_summary": {"source_gain_bank_hash": "gainbank123"},
                "phys_gain_basis_summary": {"residual_abs_mean": 0.00125},
            }
        _write_run(
            runs_root,
            ablation_id=ablation_id,
            best_selection_value=score,
            trainable_count=144042,
            hyper_coeff_generator="shared_layer_aware_rank_gated_stable",
            hyper_adapter_param_style="dora_like_gain_bounded",
            hyper_reliability_gate="prompt_scalar",
            hyper_enable_film=True,
            hyper_enable_adapters=True,
            zero_shot_prior_form="source_base_residual_reliability_gated",
            summary_extra=summary_extra,
        )

    result = build_hyperda_staged_ablation_table(
        runs_root=runs_root,
        output_dir=tmp_path / "reports",
        target_region="US-R1",
        seed=0,
    )

    rows = list(csv.DictReader(result["csv_path"].open(encoding="utf-8", newline="")))
    assert [row["ablation_id"] for row in rows] == [
        "M3_1_hyperda_trust_medium",
        "M3_8_phys_formula_operator_trust",
        "M3_8b_phys_formula_light_guarded_trust",
        "M3_8c_phys_formula_light_operator_only_trust",
        "M3_12_phys_gain_basis_hypertrust",
        "M3_13_phys_gain_guarded_hypertrust",
        "M3_14_source_trained_phys_formula_gain_hypertrust",
        "M3_15_m31_anchored_source_safe_phys_coeff_delta",
        "M3_16_source_only_phys_m3trust_lite",
    ]
    m3_12 = next(row for row in rows if row["ablation_id"] == "M3_12_phys_gain_basis_hypertrust")
    assert m3_12["hyper_phys_gain_basis_residual"] == "1"
    assert m3_12["phys_gain_source_bank_hash"] == "gainbank123"
    markdown = result["md_path"].read_text(encoding="utf-8")
    assert "M3_8b_phys_formula_light_guarded_trust" in markdown
    assert "M3_8c_phys_formula_light_operator_only_trust" in markdown
    assert "M3_12_phys_gain_basis_hypertrust" in markdown
    assert "M3_13_phys_gain_guarded_hypertrust" in markdown
    assert "M3_14_source_trained_phys_formula_gain_hypertrust" in markdown
    assert "M3_15_m31_anchored_source_safe_phys_coeff_delta" in markdown
    assert "M3_16_source_only_phys_m3trust_lite" in markdown
