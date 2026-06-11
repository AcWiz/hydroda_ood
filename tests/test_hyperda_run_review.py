from __future__ import annotations

import csv
import json
from pathlib import Path


def test_hyperda_run_review_flags_overfit_and_legacy_baseline(tmp_path):
    from scripts.analysis.review_hyperda_zero_few_shot_run import build_run_review

    overview = tmp_path / "overview.csv"
    with overview.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "K",
                "adaptation_setting",
                "surface_rmse_latw",
                "support_loss_delta",
                "adaptation_steps",
                "lr",
                "anchor_alpha",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "K": "0",
                "adaptation_setting": "zero_shot_context",
                "surface_rmse_latw": "1.0",
                "support_loss_delta": "",
                "adaptation_steps": "0",
                "lr": "0.001",
                "anchor_alpha": "0.0",
            }
        )
        writer.writerow(
            {
                "K": "12",
                "adaptation_setting": "few_shot_k12",
                "surface_rmse_latw": "1.2",
                "support_loss_delta": "-0.05",
                "adaptation_steps": "80",
                "lr": "0.0003",
                "anchor_alpha": "0.25",
            }
        )

    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            [
                {
                    "method": "source_only_backbone",
                    "protocol_version": "v4.3",
                    "adaptation_setting": "target_full_train",
                }
            ]
        ),
        encoding="utf-8",
    )

    review = build_run_review(overview_path=overview, baseline_path=baseline)

    assert review["summary"]["status"] == "needs_audit"
    assert any("K=12" in warning and "support_loss_delta" in warning for warning in review["warnings"])
    assert any("target_full_train" in warning for warning in review["warnings"])
    assert review["k_rows"]["12"]["anchor_alpha"] == "0.25"
    assert review["baseline_protocol"]["same_protocol"] is False


def test_hyperda_run_review_cli_writes_json_and_markdown(tmp_path):
    from scripts.analysis.review_hyperda_zero_few_shot_run import main

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "overview.csv").write_text(
        "K,adaptation_setting,surface_rmse_latw,support_loss_delta,adaptation_steps,lr,anchor_alpha\n"
        "0,zero_shot_context,1.0,,0,0.001,0.0\n",
        encoding="utf-8",
    )

    out_json = tmp_path / "review.json"
    out_md = tmp_path / "review.md"
    main(
        [
            "--run_dir",
            str(run_dir),
            "--output_json",
            str(out_json),
            "--output_md",
            str(out_md),
        ]
    )

    assert json.loads(out_json.read_text(encoding="utf-8"))["summary"]["overview_path"].endswith("overview.csv")
    md = out_md.read_text(encoding="utf-8")
    assert "# HyperDA Zero/Few-Shot Run Review" in md
    assert "source-only" in md
