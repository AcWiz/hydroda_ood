from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


class TinyContextDataset:
    def __init__(self, *, split_type: str, active_region: str = "US-R2", K: int = 0):
        self.split_type = split_type
        self.active_region = active_region
        self.K = K
        self._split_entry = {
            "target_context_dates_hash": "contexthash",
            "target_support_dates_hash": "supporthash",
            "target_eval_dates_hash": "evalhash",
            "split_manifest_sha256": "splithash",
        }

    def __len__(self):
        return 1

    def set_active_region(self, region_id: str):
        self.active_region = region_id

    def get_input_side_sample(self, idx: int):
        x = np.ones((12, 2, 2), dtype=np.float32) * (1.0 + int(self.active_region[-1]))
        return {
            "x": x,
            "forecast_surface": x[0],
            "forecast_rootzone": x[1],
            "region_mask": np.ones((2, 2), dtype=np.float32),
            "base_valid_mask": np.ones((2, 2), dtype=np.float32),
            "latitude_weight": np.ones((2, 2), dtype=np.float32),
            "date_str": "2015-01-01",
            "month": 1,
            "time_index": idx,
            "sample_region_id": self.active_region,
        }

    def close(self):
        return None


class TinyEvalDataset(TinyContextDataset):
    def __init__(self, *, split_type: str, active_region: str = "US-R1", K: int = 0):
        super().__init__(split_type=split_type, active_region=active_region, K=K)
        self.sample = {
            "x": np.ones((12, 2, 2), dtype=np.float32),
            "forecast_surface": np.zeros((2, 2), dtype=np.float32),
            "forecast_rootzone": np.zeros((2, 2), dtype=np.float32),
            "analysis_surface": np.ones((2, 2), dtype=np.float32) * 2,
            "analysis_rootzone": np.ones((2, 2), dtype=np.float32) * 4,
            "increment_surface": np.ones((2, 2), dtype=np.float32) * 2,
            "increment_rootzone": np.ones((2, 2), dtype=np.float32) * 4,
            "metric_mask": np.ones((2, 2), dtype=np.float32),
            "latitude_weight": np.ones((2, 2), dtype=np.float32),
            "date_str": "2023-01-01" if split_type == "target_eval" else "2016-01-01",
            "time_index": 0,
            "month": 1,
            "season": "DJF",
            "country_id": "US",
            "target_region_id": "US-R1",
            "sample_region_id": "US-R1",
            "active_region_ids": ["US-R1"],
            "adaptation_setting": "few_shot_k4" if K == 4 else "zero_shot_context",
            "K": K,
            "seed": 0,
            "target_context_dates_hash": "contexthash",
            "target_support_dates_hash": "supporthash",
            "target_eval_dates_hash": "evalhash",
            "split_manifest_sha256": "splithash",
        }

    def __getitem__(self, idx: int):
        return dict(self.sample)


class ConstantPredictor:
    def __init__(self, value: float):
        self.value = float(value)
        self.method_name = f"const_{value:g}"

    def predict(self, sample):
        shape = sample["forecast_surface"].shape
        inc_s = np.ones(shape, dtype=np.float32) * self.value
        inc_r = np.ones(shape, dtype=np.float32) * self.value * 2
        return {
            "pred_increment_surface": inc_s,
            "pred_increment_rootzone": inc_r,
            "pred_analysis_surface": sample["forecast_surface"] + inc_s,
            "pred_analysis_rootzone": sample["forecast_rootzone"] + inc_r,
        }


def test_train_hyperda_rise_router_writes_source_only_prior(tmp_path, monkeypatch):
    from scripts.train import train_hyperda_rise_router as trainer

    def dataset_factory(**kwargs):
        return TinyContextDataset(
            split_type=kwargs["split_type"],
            active_region=kwargs.get("active_region_id") or "US-R2",
        )

    monkeypatch.setattr(trainer, "HydroDADataset", dataset_factory)
    monkeypatch.setattr(trainer, "compute_sha256", lambda path: "splithash")

    metrics_path = tmp_path / "candidate_metrics_source_val.csv"
    pd.DataFrame(
        [
            {
                "pseudo_target_region_id": "US-R2",
                "candidate_id": "A",
                "split_role": "source_val",
                "variable": "surface",
                "metric": "increment_rmse_latw",
                "value": 0.20,
            },
            {
                "pseudo_target_region_id": "US-R2",
                "candidate_id": "B",
                "split_role": "source_val",
                "variable": "surface",
                "metric": "increment_rmse_latw",
                "value": 0.10,
            },
            {
                "pseudo_target_region_id": "US-R2",
                "candidate_id": "A",
                "split_role": "source_val",
                "variable": "rootzone",
                "metric": "increment_rmse_latw",
                "value": 0.30,
            },
            {
                "pseudo_target_region_id": "US-R2",
                "candidate_id": "B",
                "split_role": "source_val",
                "variable": "rootzone",
                "metric": "increment_rmse_latw",
                "value": 0.40,
            },
        ]
    ).to_csv(metrics_path, index=False)

    summary = trainer.run_train(
        candidates=[{"expert_id": "A"}, {"expert_id": "B"}],
        target_region="US-R1",
        seed=0,
        output_dir=tmp_path,
        candidate_metrics_source_val=metrics_path,
        source_regions=["US-R2"],
        temperature=0.1,
        da_nc_path="DA.nc",
        region_masks_nc="masks.nc",
        splits_json="splits.json",
        freeze_manifest="freeze.json",
        max_context_samples=1,
    )

    prior = json.loads((tmp_path / "router_prior.json").read_text(encoding="utf-8"))
    assert summary["router_prior"] == str(tmp_path / "router_prior.json")
    assert prior["training_label_source"] == "source_val_2022"
    assert prior["no_leakage_declaration"]["target_eval_used_for_router_training"] is False
    assert (tmp_path / "candidate_metrics_source_val.csv").exists()


def test_train_hyperda_rise_router_can_build_source_val_candidate_metrics(tmp_path, monkeypatch):
    from scripts.train import train_hyperda_rise_router as trainer

    def dataset_factory(**kwargs):
        if kwargs["split_type"] == "source_fit":
            return TinyContextDataset(
                split_type=kwargs["split_type"],
                active_region=kwargs.get("active_region_id") or "US-R2",
            )
        return TinyEvalDataset(
            split_type=kwargs["split_type"],
            active_region=kwargs.get("active_region_id") or "US-R2",
            K=0,
        )

    monkeypatch.setattr(trainer, "HydroDADataset", dataset_factory)
    monkeypatch.setattr(trainer, "compute_sha256", lambda path: "splithash")
    monkeypatch.setattr(
        trainer,
        "create_predictor",
        lambda **kwargs: ConstantPredictor(0.0 if kwargs["checkpoint"] == "zero" else 2.0),
    )

    summary = trainer.run_train(
        candidates=[
            {"expert_id": "A", "predictor_type": "forecast_only", "checkpoint": "zero"},
            {"expert_id": "B", "predictor_type": "source_only", "checkpoint": "two"},
        ],
        target_region="US-R1",
        seed=0,
        output_dir=tmp_path,
        candidate_metrics_source_val=None,
        source_regions=["US-R2"],
        temperature=0.1,
        da_nc_path="DA.nc",
        region_masks_nc="masks.nc",
        splits_json="splits.json",
        freeze_manifest="freeze.json",
        max_context_samples=1,
        max_eval_samples=1,
    )

    metrics = pd.read_csv(tmp_path / "candidate_metrics_source_val.csv")
    assert summary["candidate_metrics_source_val"] == str(tmp_path / "candidate_metrics_source_val.csv")
    assert set(metrics["split_role"]) == {"source_val"}
    assert set(metrics["candidate_id"]) == {"A", "B"}
    assert set(metrics["pseudo_target_region_id"]) == {"US-R2"}
    prior = json.loads((tmp_path / "router_prior.json").read_text(encoding="utf-8"))
    routed_weights = prior["prototypes"][0]["weights"]["surface"]
    assert routed_weights["B"] > routed_weights["A"]


def test_eval_hyperda_rise_writes_posterior_metrics_and_reliability(tmp_path, monkeypatch):
    from scripts.eval import eval_hyperda_rise as evaluator

    def dataset_factory(**kwargs):
        return TinyEvalDataset(split_type=kwargs["split_type"], K=kwargs.get("K") or 0)

    monkeypatch.setattr(evaluator, "HydroDADataset", dataset_factory)
    monkeypatch.setattr(evaluator, "compute_sha256", lambda path: "splithash")
    monkeypatch.setattr(
        evaluator,
        "create_predictor",
        lambda **kwargs: ConstantPredictor(0.0 if kwargs["checkpoint"] == "zero" else 2.0),
    )

    prior = {
        "schema_version": "hyperda_rise_router_prior_v1",
        "method": "HyperDA-RISE",
        "method_id": "hyperda_rise_source_side_router_prior",
        "training_label_source": "source_val_2022",
        "temperature": 0.1,
        "expert_ids": ["A", "B"],
        "candidates": [
            {"expert_id": "A", "predictor_type": "forecast_only", "checkpoint": "zero"},
            {"expert_id": "B", "predictor_type": "source_only", "checkpoint": "two"},
        ],
        "prototypes": [
            {
                "pseudo_target_region_id": "US-R2",
                "descriptor": [1.0] * 32,
                "metric_split_role": "source_val",
                "weights": {
                    "surface": {"A": 0.5, "B": 0.5},
                    "rootzone": {"A": 0.5, "B": 0.5},
                },
                "uncertainty": {"surface": 0.1, "rootzone": 0.1},
            }
        ],
        "no_leakage_declaration": {
            "target_eval_used_for_router_weights": False,
            "target_eval_used_for_expert_selection": False,
            "dynamic_target_eval_gating": False,
        },
    }
    router_prior_path = tmp_path / "router_prior.json"
    router_prior_path.write_text(json.dumps(prior), encoding="utf-8")

    summary = evaluator.run_eval(
        router_prior_path=router_prior_path,
        target_region="US-R1",
        K=4,
        seed=0,
        output_dir=tmp_path,
        device="cpu",
        max_eval_samples=1,
        max_context_samples=1,
        max_support_samples=1,
        ridge_lambda=0.01,
        temperature=0.1,
        da_nc_path="DA.nc",
        region_masks_nc="masks.nc",
        splits_json="splits.json",
        freeze_manifest="freeze.json",
    )

    assert summary["method_id"] == "hyperda_rise_k4_support_posterior"
    assert (tmp_path / "posterior_config_K4.json").exists()
    assert (tmp_path / "support_reliability.csv").exists()
    assert (tmp_path / "US-R1" / "metrics_long.csv").exists()
    posterior = json.loads((tmp_path / "posterior_config_K4.json").read_text(encoding="utf-8"))
    assert posterior["no_leakage_declaration"]["target_eval_used_for_posterior"] is False


def test_hyperda_rise_wrapper_declares_new_method_and_not_safe_alias():
    text = Path("run/hyperda_rise_us_r1_seed0.sh").read_text(encoding="utf-8")

    assert "HyperDA-RISE" in text
    assert "Retrieval-Informed Self-supervised Expert Operator Composition" in text
    assert 'TARGET_REGION="${TARGET_REGION:-US-R1}"' in text
    assert 'SEED="${SEED:-0}"' in text
    assert 'K_LIST="${K_LIST:-0 4 12}"' in text
    assert "target_context=2015-2021 input-side only" in text
    assert "source_val 2022 candidate WRMSE" in text
    assert "target_support=K labeled DA cycles" in text
    assert "target_val=unused_in_main_protocol" in text
    assert "target_eval=2023-2025 final offline evaluation" in text
    assert "DEFAULT_SOURCE_POOLED_CHECKPOINT=" in text
    assert "DEFAULT_HYPERDA_K0_CHECKPOINT=" in text
    assert "DEFAULT_CANDIDATE_SPECS=" in text
    assert "forecast:forecast_only:zero" in text
    assert 'mkdir -p "${OUTPUT_BASE}/router"' in text
    assert "scripts/train/train_hyperda_rise_router.py" in text
    assert "scripts/eval/eval_hyperda_rise.py" in text
    assert "--target-region" in text
    assert "--candidate-specs" in text
    assert "--dry-run" in text
    assert "run/phase5_hyperda_zero_few_shot_eval.sh" not in text


def test_hyperda_rise_wrapper_help_and_dry_run_support_cli_overrides():
    script = Path("run/hyperda_rise_us_r1_seed0.sh")

    help_result = subprocess.run(
        ["bash", str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--target-region" in help_result.stdout
    assert "--candidate-specs" in help_result.stdout
    assert "defaults are embedded in this script" in help_result.stdout

    dry_run = subprocess.run(
        [
            "bash",
            str(script),
            "--dry-run",
            "--target-region",
            "US-R2",
            "--seed",
            "3",
            "--cuda-device",
            "0",
            "--k-list",
            "0 4",
            "--candidate-specs",
            "forecast:forecast_only:zero custom:source_only:/tmp/custom.pt",
            "--max-eval-samples",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "target_region=US-R2" in dry_run.stdout
    assert "seed=3" in dry_run.stdout
    assert "CUDA_VISIBLE_DEVICES=0" in dry_run.stdout
    assert "K_LIST=0 4" in dry_run.stdout
    assert "custom:source_only:/tmp/custom.pt" in dry_run.stdout
    assert "DRY_RUN=1" in dry_run.stdout
    assert "output_base=artifacts/runs/hyperda_rise/US-R2_s3_" in dry_run.stdout


def test_hyperda_rise_wrapper_defaults_to_bounded_smoke_and_full_is_explicit():
    script = Path("run/hyperda_rise_us_r1_seed0.sh")

    smoke = subprocess.run(
        ["bash", str(script), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "run_mode=smoke" in smoke.stdout
    assert "max_context_samples=2" in smoke.stdout
    assert "max_support_samples=2" in smoke.stdout
    assert "max_eval_samples=2" in smoke.stdout

    full = subprocess.run(
        ["bash", str(script), "--dry-run", "--full"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "run_mode=full" in full.stdout
    assert "max_context_samples=0" in full.stdout
    assert "max_support_samples=0" in full.stdout
    assert "max_eval_samples=0" in full.stdout
