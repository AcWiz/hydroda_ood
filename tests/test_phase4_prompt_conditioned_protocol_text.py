from pathlib import Path


def test_research_plan_protocol_block_uses_target_full_train_dates():
    text = Path("完整研究计划方案.md").read_text()

    assert "Source validation:          2021" not in text
    assert "Target train / adaptation:    2022" not in text
    assert "Source fit/train:           2015-2021" in text
    assert "Source validation:          2022" in text
    assert "Target train / adaptation:  2015-2021" in text
    assert "Target validation:          2022" in text
    assert "Target eval/test:           2023-2025" in text


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
