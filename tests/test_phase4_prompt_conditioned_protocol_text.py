from pathlib import Path


def test_research_plan_protocol_block_uses_target_full_train_dates():
    text = Path("完整研究计划方案.md").read_text()

    assert "Source validation:          2021" not in text
    assert "Target train / adaptation:    2022" not in text
    assert "target 2022 input-side prompt" not in text
    assert "Source fit/train:           2015-2021" in text
    assert "Source validation:          2022" in text
    assert "Target train / adaptation:  2015-2021" in text
    assert "Target validation:          2022" in text
    assert "Target eval/test:           2023-2025" in text
    assert "HyperDA-Adapt" in text
    assert "不再把主方法表述为 zero-shot" in text


def test_project_experiment_plan_documents_use_historical_target_adaptation():
    docs = [
        Path("context/04_实验矩阵.md").read_text(),
        Path("tasks/phase4_neural_baselines.md").read_text(),
        Path("tasks/phase6_reporting_and_paper_artifacts.md").read_text(),
        Path("tasks/阶段5_HyRAO与稀疏适配.md").read_text(),
    ]
    combined = "\n".join(docs)

    assert "target 2022 input-side prompt" not in combined
    assert "HyperDA-Adapt" in combined
    assert "target_val=2022" in combined
    assert "target latent" in combined
    assert "adapter coefficient residual" in combined
    assert "residual gain" in combined
    assert "zero-shot；目标阶段训练" in combined


def test_prompt_conditioned_train_wrapper_uses_target_full_train_protocol():
    text = Path("run/phase4_prompt_conditioned.sh").read_text()

    assert "US_loro_kdate_splits.json" not in text
    assert "K=0" not in text
    assert "--K" not in text
    assert "--adaptation_setting target_full_train" in text
    assert "US_loro_target_train_splits.json" in text


def test_prompt_conditioned_inference_wrapper_uses_target_eval_protocol():
    text = Path("run/phase4_prompt_conditioned_inference.sh").read_text()

    assert "context2022_query2023_2025_k0_4_12" not in text
    assert "--K" not in text
    assert "--split_type target_eval" in text
    assert "target_query" not in text
    assert "checkpoint_best_source_val_transfer_safe_score.pt" in text
    assert 'TARGET_PROMPT_FROM_TARGET_TRAIN="${TARGET_PROMPT_FROM_TARGET_TRAIN:-1}"' in text


def test_hyperda_train_wrapper_uses_target_full_train_protocol():
    text = Path("run/phase4_hyperda.sh").read_text()

    assert "US_loro_kdate_splits.json" not in text
    assert "K=0" not in text
    assert "zero-shot" not in text.lower()
    assert "--K" not in text
    assert "--adaptation_setting target_full_train" in text
    assert "US_loro_target_train_splits.json" in text
    assert "--model_type hyperda_basis_adapter" in text
    assert "--selection_metric source_val_transfer_safe_score" in text
    assert "target_train=2015-2021" in text
    assert "target_val=2022" in text
    assert "target_eval=2023-2025" in text


def test_hyperda_inference_wrapper_uses_target_eval_protocol():
    text = Path("run/phase4_hyperda_inference.sh").read_text()

    assert "context2022_query2023_2025_k0_4_12" not in text
    assert "--K" not in text
    assert "--split_type target_eval" in text
    assert "target_query" not in text
    assert "checkpoint_best_source_val_transfer_safe_score.pt" in text
    assert "phase4_prompt_conditioned_hyperda_basis_adapter_*" in text
    assert 'TARGET_PROMPT_FROM_TARGET_TRAIN="${TARGET_PROMPT_FROM_TARGET_TRAIN:-1}"' in text
    assert 'TARGET_TRAIN_RESIDUAL_GAIN_CALIBRATION="${TARGET_TRAIN_RESIDUAL_GAIN_CALIBRATION:-1}"' in text


def test_hyperda_target_adaptation_wrapper_declares_frozen_hypernetwork_protocol():
    text = Path("run/phase5_hyperda_target_adapt.sh").read_text()

    assert "zero-shot" not in text.lower()
    assert "target_train=2015-2021" in text
    assert "target_val=2022" in text
    assert "target_eval=2023-2025" in text
    assert "freeze_hypernetwork=true" in text
    assert "trainable=target_latent,adapter_coefficient_residuals,residual_gain" in text
    assert "target_eval labels are never used for adaptation" in text
