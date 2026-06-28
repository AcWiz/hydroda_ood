from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_audit_m2_3_k0_transfer_outputs_hashes_protocol_match_and_rmse_deltas(tmp_path):
    m2_1_ckpt = tmp_path / "m2_1.pt"
    m2_1_ckpt.write_bytes(b"m2.1 checkpoint")
    m2_3_ckpt = tmp_path / "m2_3.pt"
    m2_3_ckpt.write_bytes(b"m2.3 checkpoint")
    split_manifest = _write_json(
        tmp_path / "splits.json",
        {
            "protocol_freeze_id": "hyperda_v4_4_zero_few_shot_generalization_2015_2025_context2015_2021_sourceval2022_eval2023_2025",
            "target_context_dates_hash": "ctxhash",
            "target_eval_dates_hash": "evalhash",
        },
    )
    m2_1_eval = _write_json(
        tmp_path / "m2_1_eval" / "summary.json",
        {
            "protocol_freeze_id": "hyperda_v4_4_zero_few_shot_generalization_2015_2025_context2015_2021_sourceval2022_eval2023_2025",
            "split_manifest_sha256": _sha(split_manifest),
            "target_context_dates_hash": "ctxhash",
            "target_eval_dates_hash": "evalhash",
            "surface": {"rmse_latw_mean": 0.0028093959},
            "rootzone": {"rmse_latw_mean": 0.0002321174},
        },
    )
    m2_3_eval = _write_json(
        tmp_path / "m2_3_eval" / "summary.json",
        {
            "protocol_freeze_id": "hyperda_v4_4_zero_few_shot_generalization_2015_2025_context2015_2021_sourceval2022_eval2023_2025",
            "split_manifest_sha256": _sha(split_manifest),
            "target_context_dates_hash": "ctxhash",
            "target_eval_dates_hash": "evalhash",
            "surface_rmse_latw": 0.0029770673,
            "rootzone_rmse_latw": 0.0002411064,
        },
    )
    output = tmp_path / "audit.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/analysis/audit_m2_3_k0_transfer.py",
            "--m2_1_checkpoint",
            str(m2_1_ckpt),
            "--m2_3_checkpoint",
            str(m2_3_ckpt),
            "--m2_1_target_eval_summary",
            str(m2_1_eval),
            "--m2_3_target_eval_summary",
            str(m2_3_eval),
            "--split_manifest",
            str(split_manifest),
            "--output_json",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["same_protocol"] is True
    assert audit["m2_1_checkpoint"]["sha256"] == _sha(m2_1_ckpt)
    assert audit["m2_3_checkpoint"]["sha256"] == _sha(m2_3_ckpt)
    assert audit["split_manifest"]["sha256"] == _sha(split_manifest)
    assert audit["target_context_hash"]["same"] is True
    assert audit["target_eval_hash"]["same"] is True
    assert audit["m2_1"]["surface_rmse"] == 0.0028093959
    assert audit["m2_3"]["rootzone_rmse"] == 0.0002411064
    assert audit["delta"]["surface_rmse_relative_pct"] > 5.9
    assert audit["delta"]["rootzone_rmse_relative_pct"] > 3.8
    assert audit["interpretation"]["m2_3_replaces_m2_1"] is False
    assert audit["interpretation"]["recommended_status"] == "negative_diagnostic_ablation"
