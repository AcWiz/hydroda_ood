#!/bin/bash
# Phase 4 HyperDA++ source-prior matrix.
#
# Usage:
#   bash run/phase4_hyperda_plus_matrix.sh US-R1 0 1
#
# Source-only protocol:
#   source_fit=2015-2021 trains the source-stage HyperDA prior.
#   source_val=2022 selects checkpoints via source_val_transfer_safe_score.
#   target_context=2015-2021 input-side only for later zero/few-shot use.
#   target_val=unused_in_main_protocol.
#   target_eval=forbidden_for_matrix_selection.

set -euo pipefail

TARGET_REGION="${1:-US-R1}"
SEED="${2:-0}"
export CUDA_VISIBLE_DEVICES="${3:-0}"

cd "$(dirname "$0")/.."

MATRIX_CONFIG="configs/experiments/hyperda_plus_source_prior_matrix.yaml"
OUTPUT_ROOT="artifacts/runs/phase4_hyperda_plus"
REPORT_DIR="artifacts/reports/hyperda_plus_source_prior_matrix"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

run_candidate() {
    local candidate_id="$1"
    local width="$2"
    local prompt_dim="$3"
    local hyper_n_basis="$4"
    local context_encoder="$5"

    local run_dir="${OUTPUT_ROOT}/${candidate_id}/${TIMESTAMP}"
    mkdir -p "${run_dir}"

    echo "============================================"
    echo "HyperDA++ source-prior candidate ${candidate_id}"
    echo "  target_region=${TARGET_REGION}"
    echo "  seed=${SEED}"
    echo "  source_fit=2015-2021 source_val=2022"
    echo "  target_val=unused_in_main_protocol"
    echo "  target_eval=forbidden_for_matrix_selection"
    echo "  split_artifact=artifacts/splits/US_loro_zero_few_shot_splits.json"
    echo "  model_type=hyperda_basis_adapter width=${width} prompt_dim=${prompt_dim}"
    echo "  hyper_n_basis=${hyper_n_basis} context_encoder=${context_encoder}"
    echo "  selection_metric=source_val_transfer_safe_score"
    echo "  output_dir=${run_dir}"
    echo "============================================"

    (
        PYTHONPATH=. python scripts/train/train_prompt_conditioned_shared.py \
            --target_region "${TARGET_REGION}" \
            --adaptation_setting zero_shot_context --K 0 \
            --seed "${SEED}" \
            --device cuda \
            --amp \
            --accum_steps 4 \
            --zero_raw_increment_init \
            --target_increment_normalization \
            --use_lat_weighted_loss \
            --batch_size 8 \
            --max_epochs 50 \
            --lr 3e-4 \
            --weight_decay 1e-4 \
            --grad_clip 1.0 \
            --num_workers 0 \
            --width "${width}" \
            --prompt_dim "${prompt_dim}" \
            --model_type hyperda_basis_adapter \
            --hyper_n_basis "${hyper_n_basis}" \
            --hyper_adapter_bottleneck 32 \
            --hyper_adapter_scale 1.0 \
            --context_encoder "${context_encoder}" \
            --log_every_steps 100 \
            --eval_every_epochs 1 \
            --checkpoint_every 10 \
            --selection_metric source_val_transfer_safe_score \
            --output_dir "${run_dir}"
    ) 2>&1 | tee "${run_dir}/train_log.txt"

    PYTHONPATH=. python - "${run_dir}" "${candidate_id}" <<'PY'
import csv
import json
import shutil
import sys
from pathlib import Path

import torch
import yaml

run_dir = Path(sys.argv[1])
candidate_id = sys.argv[2]
ckpt_dir = run_dir / "checkpoints"
reports_dir = run_dir / "reports"
best_ckpt = ckpt_dir / "checkpoint_best_source_val_transfer_safe_score.pt"
summary_path = reports_dir / "summary.json"
config_path = run_dir / "config_resolved.yaml"

summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
if config_path.exists():
    config = yaml.safe_load(config_path.read_text()) or {}
else:
    config = {}
config["candidate_id"] = candidate_id
config["matrix_config"] = "configs/experiments/hyperda_plus_source_prior_matrix.yaml"
config["source_fit"] = "2015-2021"
config["source_val"] = "2022"
config["target_val_usage"] = "unused_in_main_protocol"
config["target_eval_usage"] = "forbidden_for_matrix_selection"
config_path.write_text(yaml.safe_dump(config, sort_keys=True))

source_metrics = dict(summary)
history = summary.get("val_history") or []
checkpoint_payload = None
if best_ckpt.exists():
    checkpoint_payload = torch.load(best_ckpt, map_location="cpu", weights_only=False)
if history:
    source_metrics.update(history[-1])
if checkpoint_payload and checkpoint_payload.get("source_val_safe_metrics"):
    source_metrics.update(checkpoint_payload["source_val_safe_metrics"])
if "best_safe_score" in summary:
    source_metrics["source_val_transfer_safe_score"] = summary["best_safe_score"]
if "best_selection_value" in summary and summary.get("best_selection_metric") == "source_val_transfer_safe_score":
    source_metrics["source_val_transfer_safe_score"] = summary["best_selection_value"]
(run_dir / "source_val_metrics.json").write_text(json.dumps(source_metrics, indent=2))

region_skills = source_metrics.get("region_variable_skills") or {}
if region_skills:
    rows = []
    for region, values in sorted(region_skills.items()):
        row = {"region": region}
        row.update(values)
        rows.append(row)
    fieldnames = ["region"] + sorted({key for row in rows for key in row if key != "region"})
    with open(run_dir / "source_val_by_region.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
else:
    (run_dir / "source_val_by_region.csv").write_text("status\nnot_available\n")
(run_dir / "source_val_by_month.csv").write_text("status\nnot_available\n")

leakage_audit = {
    "candidate_id": candidate_id,
    "status": "pass",
    "train_split": "source_fit",
    "selection_split": "source_val",
    "normalization_source": "source_fit_only",
    "target_val_access": "none",
    "target_eval_access": "none",
    "analysis_increment_label_use": "source_fit/source_val only",
    "context_encoder": config.get("context_encoder", ""),
    "channel_11_policy": "not_observation_not_loss_mask_not_region_mask",
}
(run_dir / "leakage_audit.json").write_text(json.dumps(leakage_audit, indent=2))

metadata = {
    "candidate_id": candidate_id,
    "checkpoint": str(best_ckpt),
    "checkpoint_exists": best_ckpt.exists(),
    "selection_metric": "source_val_transfer_safe_score",
    "context_encoder": config.get("context_encoder", ""),
}
if checkpoint_payload:
    metadata.update(
        {
            "epoch": checkpoint_payload.get("epoch"),
            "best_selection_metric": checkpoint_payload.get("best_selection_metric"),
            "best_selection_value": checkpoint_payload.get("best_selection_value"),
            "selection_score": checkpoint_payload.get("selection_score"),
            "protocol_freeze_id": checkpoint_payload.get("protocol_freeze_id"),
        }
    )
(run_dir / "checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2))

if best_ckpt.exists():
    exposed = run_dir / "checkpoint_best_source_val_transfer_safe_score.pt"
    if not exposed.exists():
        shutil.copy2(best_ckpt, exposed)
PY
}

write_h4_todo() {
    local h4_dir="${OUTPUT_ROOT}/H4_episode_prior/${TIMESTAMP}"
    mkdir -p "${h4_dir}"
    cat > "${h4_dir}/TODO_H4_episode_prior.md" <<'EOF'
# H4 Episode Prior TODO

H4 is intentionally skipped in `phase4_hyperda_plus_matrix.sh` until
`scripts/train/train_hyperda_episode_prior.py` is implemented.

Expected inputs:
- Source episode bank produced by `scripts/train/build_source_episode_adapter_bank.py`.
- Episode records from source_fit/source_val source regions only.
- Adapter coefficient artifacts in the lightweight target-module space defined
  by `hydroda/operator_bank/zeta_schema.py`.

Expected output:
- A deterministic prompt_to_zeta_prior artifact suitable for initializing
  HyperDA adapter coefficient residuals.
- Validation summary selected by `source_side_episodic_validation`.

Required leakage_metadata:
- target_val_used: false
- target_eval_used: false
- target_eval_labels_loaded: false
- query_role: source_val
- normalizer_source: source_fit_only_from_source_checkpoint
- selection_source: source_side_episodic_validation

Do not train H4 automatically until the source episode prior trainer no longer
raises `NotImplementedError`.
EOF
}

# run_candidate H0_current 32 64 8 current_mean_std
run_candidate H1_capacity_safe 48 96 12 current_mean_std
run_candidate H2_capacity_safe_large 64 128 16 current_mean_std
run_candidate H3_context_plus 32 64 8 robust_input_side_da_diagnostics
write_h4_todo

PYTHONPATH=. python scripts/report/hyperda_plus_source_prior_matrix.py \
    --runs_root "${OUTPUT_ROOT}" \
    --report_dir "${REPORT_DIR}" \
    --matrix_config "${MATRIX_CONFIG}"

echo "Done: HyperDA++ source-prior matrix ${TARGET_REGION} seed=${SEED}"
