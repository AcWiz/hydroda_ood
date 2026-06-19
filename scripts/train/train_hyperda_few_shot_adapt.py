#!/usr/bin/env python3
"""HyperDA zero/few-shot target adaptation with preregistered final checkpoints.

No-leakage declaration:
    - Loads a source-trained HyperDA checkpoint.
    - Freezes source backbone, prompt encoder, hypernetwork, and basis bank.
    - K=0 builds target context prompt state only; it performs no target-label
      training.
    - K=4/12 trains only lightweight target-specific variables on target_support.
    - Does not construct target_val and does not use target-side early stopping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from hydroda.data.dataset import HydroDADataset, collate_hydroda_samples
from hydroda.data.file_hash import compute_sha256
from hydroda.data.protocol import ProtocolConfig
from hydroda.baselines.prompt_conditioned import (
    build_target_context_prompt_state,
    compose_target_context_prompt_from_state,
    normalize_target_context_prompt_state,
    target_context_prompt_metadata,
    ROBUST_DA_CONTEXT_ENCODER,
    ROBUST_DA_RAW_CONTEXT_ENCODER,
)
from hydroda.models.hyper_conditional_unet import HyperAdapterConditionalResUNet
from hydroda.models.prompt_encoder import RegionPromptEncoder, RobustInputSideDAPromptEncoder
from hydroda.training.losses import MaskedHuberLoss, WeightedMaskedHuberLoss
from hydroda.utils.run_manager import RunManager
from hydroda.utils.runtime import get_git_hash, get_timestamp

from scripts.train.train_hyperda_target_adapt import (
    TargetAdaptationState,
    _analysis_loss,
    _as_list,
    _collate_target_batch,
    _denormalize_increment,
    _model_forward,
    _normalize_x,
    _target_region_embedding,
    _target_tensor,
    adaptation_regularization,
    apply_target_adaptation_stage,
    apply_target_adapter_state,
    extract_target_adapter_state,
    interpolate_target_adapter_state,
)


DA_NC = "/fastersharefiles2/fenglonghan/dataset/SMAP/DA.nc"
REGION_MASKS_NC = "artifacts/regions/US_region_masks.nc"
SPLITS_JSON = "artifacts/splits/US_loro_zero_few_shot_splits.json"
FREEZE_MANIFEST = "artifacts/protocol/US_region_split_freeze_manifest.json"
PROTOCOL = ProtocolConfig()
PROTOCOL_FREEZE_ID = PROTOCOL.protocol_freeze_id
PHASE = "phase5_hyperda_zero_few_shot"
SAFE_TRAINABLE_TARGET_GROUPS = [
    "target_prompt",
    "adapter_coefficient_residuals",
    "monthly_residual_gain",
]
FROZEN_SOURCE_GROUPS = [
    "source_backbone",
    "source_head",
    "prompt_encoder",
    "film",
    "basis_adapter_hypernetwork",
    "adapter_basis_bank",
]
ADAPT_SCOPES = (
    "none",
    "safe_operator",
    "prompt_coeff_gain",
    "prompt_only",
    "coeff_only",
    "gain_only",
    "coeff_gain",
    "all",
)
ADAPT_SOLVERS = ("adamw", "ridge_coeff")
TRUST_REGION_MODES = ("none", "global", "groupwise")
SUPPORT_LOSS_REDUCTIONS = ("global_pixel", "cycle_balanced")
STAGE3_POSTERIOR_POLICIES = (
    "safe_operator_ablation",
    "conservative_coeff_posterior",
    "source_calibrated_mix",
)
SUPPORT_GATES = ("policy_default", "auto", "off")
COEFF_RESIDUAL_PARAMETER_NAMES = (
    "target_adapter_coefficient_residual_b.logit_delta",
    "target_adapter_coefficient_residual_d2.logit_delta",
    "target_adapter_coefficient_residual_d1.logit_delta",
)


def _build_source_prompt_encoder(source_config: Dict[str, Any]) -> RegionPromptEncoder:
    context_encoder = source_config.get("context_encoder", "current_mean_std")
    kwargs = {
        "num_regions": int(source_config.get("num_regions", len(source_config.get("source_regions", [])) or 6)),
        "input_channels": 12,
        "hidden_dim": int(source_config.get("prompt_dim", 64)),
    }
    if context_encoder == "current_mean_std":
        return RegionPromptEncoder(**kwargs)
    if context_encoder in {ROBUST_DA_CONTEXT_ENCODER, ROBUST_DA_RAW_CONTEXT_ENCODER}:
        return RobustInputSideDAPromptEncoder(**kwargs)
    raise ValueError(f"Unsupported source checkpoint context_encoder: {context_encoder}")


@dataclass
class FewShotAdaptationState(TargetAdaptationState):
    """State container for the zero/few-shot runner."""


@dataclass
class RidgeSolveResult:
    """Closed-form coefficient residual solve and numerical diagnostics."""

    delta: torch.Tensor
    diagnostics: Dict[str, Any]


def default_steps_for_K(K: int) -> int:
    if int(K) == 0:
        return 0
    if int(K) == 4:
        return 100
    if int(K) == 12:
        return 80
    raise ValueError(f"unsupported K={K}; expected one of {list(PROTOCOL.main_K_values)}")


def default_lr_for_K(K: int) -> float:
    if int(K) == 12:
        return 3e-4
    return 1e-3


def default_anchor_alpha_for_K(K: int) -> float:
    if int(K) == 0:
        return 0.0
    if int(K) == 4:
        return 0.75
    if int(K) == 12:
        return 0.25
    raise ValueError(f"unsupported K={K}; expected one of {list(PROTOCOL.main_K_values)}")


def default_anchor_alpha_grid_for_K(K: int) -> List[float]:
    if int(K) == 4:
        return [0.25, 0.5, 0.75, 1.0]
    if int(K) == 12:
        return [0.1, 0.25, 0.5, 0.75, 1.0]
    return [0.0]


def _json_sha256_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_target_adapter_state_key(name: str) -> bool:
    return name.startswith("target_") or name.startswith("residual_gain.")


def hash_tensor_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    """Deterministically hash a tensor state dict by key, shape, dtype, and bytes."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def hash_source_prior_state(model: nn.Module) -> str:
    """Hash frozen HyperDA source-prior tensors, excluding target posterior variables."""
    return hash_tensor_state_dict(
        {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
            if not _is_target_adapter_state_key(name)
        }
    )


def _support_manifest_hash(
    *,
    target_region: str,
    K: int,
    seed: int,
    target_support_dates: Iterable[str],
    target_support_dates_hash: str,
    split_manifest_sha256: str,
) -> str:
    """Hash the support contract recorded by a few-shot run."""
    payload = {
        "schema_version": "hyperda_support_manifest_v1",
        "target_region": str(target_region),
        "K": int(K),
        "seed": int(seed),
        "target_support_dates": [str(date) for date in target_support_dates],
        "target_support_dates_hash": str(target_support_dates_hash or ""),
        "split_manifest_sha256": str(split_manifest_sha256 or ""),
    }
    return _json_sha256_payload(payload)


def _support_nesting_status(K: int, target_support_dates: Iterable[str]) -> str:
    if int(K) == 0:
        return "K0_no_support"
    return f"K{int(K)}_support_manifest_recorded"


def load_safe_policy_json(policy_path: str) -> Dict[str, Any]:
    """Load a source-side SAFE policy and reject target-side calibration sources."""
    path = Path(policy_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"--safe_policy_json not found: {path}")
    with open(path, encoding="utf-8") as f:
        policy = json.load(f)
    if not isinstance(policy, dict):
        raise ValueError("--safe_policy_json must contain a JSON object")
    policy_source = str(policy.get("policy_source", "")).strip()
    if policy_source != "source_side_episode_calibration":
        raise ValueError(
            "--safe_policy_json must declare policy_source='source_side_episode_calibration'; "
            f"got {policy_source!r}"
        )
    target_val_usage = str(policy.get("target_val_usage", "unused_in_main_protocol")).strip()
    target_eval_usage = str(policy.get("target_eval_usage", "final_eval_only_no_selection")).strip()
    if target_val_usage != "unused_in_main_protocol":
        raise ValueError(
            "--safe_policy_json must not use target_val; "
            f"target_val_usage={target_val_usage!r}"
        )
    if target_eval_usage not in {"final_eval_only_no_selection", "final_eval_only_no_training_no_selection"}:
        raise ValueError(
            "--safe_policy_json must not use target_eval for selection; "
            f"target_eval_usage={target_eval_usage!r}"
        )
    if any(token in json.dumps(policy, sort_keys=True).lower() for token in ("target_val_grid_search", "target_eval_grid_search")):
        raise ValueError("--safe_policy_json appears to reference target-side grid search")
    return policy


def _policy_for_setting(policy: Dict[str, Any], adaptation_setting: str, K: int) -> Dict[str, Any]:
    policies = policy.get("policies", {})
    if not isinstance(policies, dict):
        raise ValueError("--safe_policy_json field 'policies' must be an object")
    candidates = [
        str(adaptation_setting),
        f"K{int(K)}",
        f"k{int(K)}",
        "zero_shot_context" if int(K) == 0 else "",
    ]
    for key in candidates:
        if key and key in policies:
            selected = policies[key]
            if not isinstance(selected, dict):
                raise ValueError(f"policy entry {key!r} must be an object")
            return dict(selected)
    if int(K) == 0:
        return {}
    raise ValueError(
        f"--safe_policy_json does not define a policy for adaptation_setting={adaptation_setting!r} K={K}"
    )


def apply_safe_policy_to_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Apply source-side calibrated K-shot policy values after defaults are resolved."""
    args.safe_policy = {}
    args.safe_policy_selected_keys = []
    args.safe_policy_json_sha256 = ""
    args.safe_policy_hash = ""
    args.policy_source = "preregistered_default"
    args.target_eval_usage = "final_eval_only_no_selection"
    args.source_episode_regions = []
    args.source_policy_candidate_id = ""
    args.source_policy_guard_config_hash = ""
    args.rho_policy = "not_applicable_k0" if int(args.K) == 0 else "diagnostic_no_source_safe_policy_json"
    args.adapt_mix_rho = 1.0 if int(args.K) == 0 else 0.0
    if not args.safe_policy_json:
        if bool(args.require_safe_policy_json_for_kshot) and int(args.K) > 0:
            parser.error("--require_safe_policy_json_for_kshot requires --safe_policy_json for K>0")
        return

    try:
        policy = load_safe_policy_json(args.safe_policy_json)
        selected = _policy_for_setting(policy, args.adaptation_setting, args.K)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    allowed_keys = {
        "adapt_scope",
        "adapt_solver",
        "lr",
        "adaptation_steps",
        "max_steps",
        "steps",
        "anchor_alpha",
        "rho_policy",
        "adapt_mix_rho",
        "support_loss_reduction",
        "freeze_monthly_gain",
        "weight_decay",
        "grad_clip",
        "lambda_prior",
        "lambda_latent",
        "lambda_gain",
        "lambda_gain_smooth",
        "lambda_analysis",
        "schedule_label",
        "trust_region_mode",
        "trust_total_radius",
        "trust_prompt_radius",
        "trust_gain_radius",
        "trust_coeff_radius",
        "trust_spatial_radius",
        "source_calibrated_candidate_id",
        "source_calibrated_guard_config_hash",
        "source_policy_candidate_id",
        "source_policy_guard_config_hash",
        "trust_radius",
    }
    unknown = sorted(set(selected) - allowed_keys)
    if unknown:
        parser.error(f"--safe_policy_json contains unsupported policy keys: {unknown}")
    if int(args.K) > 0:
        required = {"adapt_scope", "lr", "anchor_alpha", "adapt_mix_rho"}
        if not any(key in selected for key in ("adaptation_steps", "max_steps", "steps")):
            required.add("adaptation_steps")
        missing = sorted(key for key in required if key not in selected)
        if missing:
            parser.error(f"--safe_policy_json policy for K={args.K} is missing required keys: {missing}")
    if "adapt_scope" in selected:
        if str(selected["adapt_scope"]) not in ADAPT_SCOPES:
            parser.error(f"SAFE policy adapt_scope must be one of {list(ADAPT_SCOPES)}")
        args.adapt_scope = str(selected["adapt_scope"])
    if "adapt_solver" in selected:
        if str(selected["adapt_solver"]) not in ADAPT_SOLVERS:
            parser.error(f"SAFE policy adapt_solver must be one of {list(ADAPT_SOLVERS)}")
        args.adapt_solver = str(selected["adapt_solver"])
    if "lr" in selected:
        args.lr = float(selected["lr"])
    if "adaptation_steps" in selected:
        args.adaptation_steps = int(selected["adaptation_steps"])
    elif "max_steps" in selected:
        args.adaptation_steps = int(selected["max_steps"])
    elif "steps" in selected:
        args.adaptation_steps = int(selected["steps"])
    if "anchor_alpha" in selected:
        args.anchor_alpha = float(selected["anchor_alpha"])
    if "support_loss_reduction" in selected:
        if str(selected["support_loss_reduction"]) not in SUPPORT_LOSS_REDUCTIONS:
            parser.error(
                f"SAFE policy support_loss_reduction must be one of {list(SUPPORT_LOSS_REDUCTIONS)}"
            )
        args.support_loss_reduction = str(selected["support_loss_reduction"])
    if "freeze_monthly_gain" in selected:
        args.freeze_monthly_gain = bool(selected["freeze_monthly_gain"])
    for key in (
        "weight_decay",
        "grad_clip",
        "lambda_prior",
        "lambda_latent",
        "lambda_gain",
        "lambda_gain_smooth",
        "lambda_analysis",
        "trust_total_radius",
        "trust_prompt_radius",
        "trust_gain_radius",
        "trust_coeff_radius",
        "trust_spatial_radius",
    ):
        if key in selected:
            setattr(args, key, float(selected[key]))
    if "trust_radius" in selected:
        args.trust_total_radius = float(selected["trust_radius"])
    if "trust_region_mode" in selected:
        if str(selected["trust_region_mode"]) not in TRUST_REGION_MODES:
            parser.error(f"SAFE policy trust_region_mode must be one of {list(TRUST_REGION_MODES)}")
        args.trust_region_mode = str(selected["trust_region_mode"])
    if "schedule_label" in selected:
        args.schedule_label = str(selected["schedule_label"])

    args.safe_policy = policy
    args.safe_policy_json_sha256 = compute_sha256(args.safe_policy_json)
    args.safe_policy_hash = str(policy.get("policy_hash", ""))
    args.policy_source = "source_side_episode_calibration"
    args.source_anchor_hyperparameter_source = "source_side_episode_calibration"
    args.target_eval_usage = str(policy.get("target_eval_usage", "final_eval_only_no_selection"))
    args.source_episode_regions = [
        str(region)
        for region in policy.get("source_episode_regions", [])
        if str(region)
    ]
    args.source_policy_candidate_id = str(
        selected.get("source_policy_candidate_id", selected.get("source_calibrated_candidate_id", ""))
    )
    args.source_policy_guard_config_hash = str(
        selected.get("source_policy_guard_config_hash", selected.get("source_calibrated_guard_config_hash", ""))
    )
    args.rho_policy = str(selected.get("rho_policy", f"fixed_{selected.get('adapt_mix_rho', '1.0')}"))
    args.adapt_mix_rho = float(selected.get("adapt_mix_rho", 1.0))
    args.safe_policy_selected_keys = sorted(selected)


def build_dataset_plan(K: int) -> List[str]:
    PROTOCOL.assert_supported_K(K)
    if int(K) == 0:
        return ["target_context"]
    return ["target_context", "target_support"]


def method_id_for_adaptation_setting(adaptation_setting: str, K: int) -> str:
    """Return paper-facing method IDs for the main HyperDA K-axis."""
    if adaptation_setting == "zero_shot_context" or int(K) == 0:
        return "hyperda_zero_shot_context"
    if adaptation_setting == "few_shot_k4" or int(K) == 4:
        return "hyperda_safe_few_shot_k4"
    if adaptation_setting == "few_shot_k12" or int(K) == 12:
        return "hyperda_safe_few_shot_k12"
    raise ValueError(f"unsupported HyperDA adaptation setting: {adaptation_setting!r}, K={K}")


def method_id_for_run(adaptation_setting: str, K: int, *, paper_facing_run: bool) -> str:
    """Return a method id that cannot accidentally promote diagnostic K-shot runs."""
    if adaptation_setting == "zero_shot_context" or int(K) == 0:
        return "hyperda_zero_shot_context"
    if bool(paper_facing_run):
        return method_id_for_adaptation_setting(adaptation_setting, K)
    if adaptation_setting == "few_shot_k4" or int(K) == 4:
        return "hyperda_diagnostic_few_shot_k4"
    if adaptation_setting == "few_shot_k12" or int(K) == 12:
        return "hyperda_diagnostic_few_shot_k12"
    raise ValueError(f"unsupported HyperDA adaptation setting: {adaptation_setting!r}, K={K}")


def _arg_was_provided(flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HyperDA zero/few-shot target variables")
    adapt_scope_provided = _arg_was_provided("--adapt_scope")
    freeze_monthly_gain_provided = _arg_was_provided("--freeze_monthly_gain")
    parser.add_argument("--source_checkpoint", type=str, required=True)
    parser.add_argument("--target_region", type=str, required=True)
    parser.add_argument("--K", type=int, required=True, choices=list(PROTOCOL.main_K_values))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adaptation_setting", type=str, default=None)
    parser.add_argument("--allow_legacy_full_target_train", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--da_nc", type=str, default=DA_NC)
    parser.add_argument("--region_masks_nc", type=str, default=REGION_MASKS_NC)
    parser.add_argument("--splits_json", type=str, default=SPLITS_JSON)
    parser.add_argument("--freeze_manifest", type=str, default=FREEZE_MANIFEST)
    parser.add_argument("--target_latent_dim", type=int, default=32)
    parser.add_argument("--adaptation_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_lat_weighted_loss", action="store_true", default=True)
    parser.add_argument("--no_lat_weighted_loss", action="store_false", dest="use_lat_weighted_loss")
    parser.add_argument("--lambda_prior", type=float, default=1e-3)
    parser.add_argument("--lambda_latent", type=float, default=1e-3)
    parser.add_argument("--lambda_gain", type=float, default=1e-2)
    parser.add_argument("--lambda_gain_smooth", type=float, default=1e-3)
    parser.add_argument("--lambda_analysis", type=float, default=0.25)
    parser.add_argument("--surface_weight", type=float, default=3.0)
    parser.add_argument("--rootzone_weight", type=float, default=1.0)
    parser.add_argument("--log_every_steps", type=int, default=50)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument(
        "--target_context_max_samples",
        type=int,
        default=0,
        help=(
            "Diagnostic/smoke cap for target_context prompt construction "
            "(0 = full target_context, paper/default behavior)."
        ),
    )
    parser.add_argument(
        "--schedule_label",
        type=str,
        default="",
        help="Diagnostic label for fixed adaptation schedule comparisons; never selected with target_eval.",
    )
    parser.add_argument(
        "--adapt_recipe",
        type=str,
        default="source_anchor",
        choices=["conservative", "source_anchor", "episode_prior"],
        help="Preregistered few-shot adaptation recipe; target labels never select this value.",
    )
    parser.add_argument(
        "--anchor_alpha",
        type=float,
        default=None,
        help="Fixed source-anchor interpolation alpha. Defaults by K from source-side episodic validation.",
    )
    parser.add_argument(
        "--source_anchor_hyperparameter_source",
        type=str,
        default="source_side_episodic_validation_preregistered",
        help="Metadata field naming the non-target source of alpha/lr/step choices.",
    )
    parser.add_argument(
        "--safe_policy_json",
        type=str,
        default=None,
        help=(
            "Source-side calibrated SAFE policy JSON. Required for paper-facing K4/K12 "
            "when --require_safe_policy_json_for_kshot is set."
        ),
    )
    parser.add_argument(
        "--require_safe_policy_json_for_kshot",
        action="store_true",
        help="Reject K>0 runs that do not provide --safe_policy_json.",
    )
    parser.add_argument(
        "--adapt_scope",
        type=str,
        default="safe_operator",
        choices=list(ADAPT_SCOPES),
        help=(
            "Target-specific parameter subset updated during few-shot adaptation. "
            "safe_operator is the paper-facing Prompt+Coeff+Gain scope; all/diagnostic "
            "scopes are legacy/internal."
        ),
    )
    parser.add_argument(
        "--stage3_posterior_policy",
        type=str,
        default="safe_operator_ablation",
        choices=list(STAGE3_POSTERIOR_POLICIES),
        help=(
            "Stage-3 target posterior policy. conservative_coeff_posterior freezes "
            "target_prompt and updates adapter coefficient residuals by default; "
            "source-side SAFE policy may explicitly add monthly residual gain. "
            "safe_operator_ablation preserves the older "
            "Prompt+Coeff+Gain path."
        ),
    )
    parser.add_argument(
        "--support_gate",
        type=str,
        default="policy_default",
        choices=list(SUPPORT_GATES),
        help="Support-only accept/rollback gate for target posterior candidates.",
    )
    parser.add_argument(
        "--support_gate_min_delta",
        type=float,
        default=0.0,
        help="Minimum support-objective decrease required by the support gate.",
    )
    parser.add_argument(
        "--support_gate_rootzone_tolerance",
        type=float,
        default=0.0,
        help="Allowed rootzone support-loss increase before the support gate rolls back.",
    )
    parser.add_argument(
        "--freeze_monthly_gain",
        action="store_true",
        help="Force residual monthly gain parameters to remain frozen during target adaptation.",
    )
    parser.add_argument(
        "--adapt_solver",
        type=str,
        default="adamw",
        choices=list(ADAPT_SOLVERS),
        help="Few-shot target adaptation solver. ridge_coeff solves only adapter coefficient residuals.",
    )
    parser.add_argument(
        "--ridge_lambda",
        type=float,
        default=1.0,
        help="Ridge penalty for ADAPT_SOLVER=ridge_coeff; fixed source-side/preregistered value.",
    )
    parser.add_argument(
        "--ridge_clip_coeff_norm",
        type=float,
        default=1.0,
        help="Maximum final coefficient residual vector norm for ADAPT_SOLVER=ridge_coeff.",
    )
    parser.add_argument(
        "--ridge_trust_region_radius",
        type=float,
        default=1.0,
        help="Maximum solved coefficient delta norm before source-anchor interpolation.",
    )
    parser.add_argument(
        "--ridge_max_feature_pixels",
        type=int,
        default=20000,
        help="Deterministic cap on support spatial pixels used to form the ridge design matrix; 0 means all.",
    )
    parser.add_argument(
        "--ridge_standardize_features",
        action="store_true",
        help="Scale ridge design columns by their support RMS before solving; target_eval is never used.",
    )
    parser.add_argument(
        "--trust_region_mode",
        type=str,
        default="none",
        choices=list(TRUST_REGION_MODES),
        help="Project target-specific AdamW updates to a fixed source-safe L2 trust region.",
    )
    parser.add_argument("--trust_total_radius", type=float, default=0.0)
    parser.add_argument("--trust_prompt_radius", type=float, default=0.0)
    parser.add_argument("--trust_gain_radius", type=float, default=0.0)
    parser.add_argument("--trust_coeff_radius", type=float, default=0.0)
    parser.add_argument("--trust_spatial_radius", type=float, default=0.0)
    parser.add_argument(
        "--support_loss_reduction",
        type=str,
        default="global_pixel",
        choices=list(SUPPORT_LOSS_REDUCTIONS),
        help="Support loss reduction for AdamW diagnostics/training; target_eval is never used.",
    )
    parser.add_argument(
        "--audit_identity",
        action="store_true",
        help="Protocol audit: require a no-update path suitable for K0/K12 identity comparison.",
    )
    parser.add_argument(
        "--audit_identity_tolerance",
        type=float,
        default=1e-8,
        help="Metadata tolerance used by the wrapper identity comparison.",
    )
    parser.add_argument("--enable_target_spatial_refine", action="store_true",
        help="Legacy/internal ablation; disabled by default for the main protocol.")

    args = parser.parse_args()
    PROTOCOL.assert_supported_K(args.K)
    if args.adaptation_setting is None:
        args.adaptation_setting = PROTOCOL.adaptation_setting_for_K(args.K)
    try:
        PROTOCOL.assert_supported_adaptation_setting(
            args.adaptation_setting,
            allow_legacy_full_target_train=args.allow_legacy_full_target_train,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.adaptation_setting == "target_full_train" and not args.allow_legacy_full_target_train:
        parser.error("--adaptation_setting target_full_train requires --allow_legacy_full_target_train")
    if args.adaptation_steps is None:
        args.adaptation_steps = default_steps_for_K(args.K)
    if args.lr is None:
        args.lr = default_lr_for_K(args.K)
    if args.anchor_alpha is None:
        args.anchor_alpha = default_anchor_alpha_for_K(args.K)
    apply_safe_policy_to_args(args, parser)
    if (
        args.stage3_posterior_policy in {"conservative_coeff_posterior", "source_calibrated_mix"}
        and int(args.K) > 0
        and not args.safe_policy_json
    ):
        if not adapt_scope_provided:
            args.adapt_scope = "coeff_only"
        if not freeze_monthly_gain_provided:
            args.freeze_monthly_gain = True
    if not 0.0 <= float(args.anchor_alpha) <= 1.0:
        parser.error("--anchor_alpha must be in [0, 1]")
    if float(args.ridge_lambda) < 0.0:
        parser.error("--ridge_lambda must be non-negative")
    if float(args.ridge_clip_coeff_norm) < 0.0:
        parser.error("--ridge_clip_coeff_norm must be non-negative")
    if float(args.ridge_trust_region_radius) < 0.0:
        parser.error("--ridge_trust_region_radius must be non-negative")
    if int(args.ridge_max_feature_pixels) < 0:
        parser.error("--ridge_max_feature_pixels must be non-negative")
    if int(args.target_context_max_samples) < 0:
        parser.error("--target_context_max_samples must be non-negative")
    if float(args.support_gate_min_delta) < 0.0:
        parser.error("--support_gate_min_delta must be non-negative")
    if float(args.support_gate_rootzone_tolerance) < 0.0:
        parser.error("--support_gate_rootzone_tolerance must be non-negative")
    for radius_name in (
        "trust_total_radius",
        "trust_prompt_radius",
        "trust_gain_radius",
        "trust_coeff_radius",
        "trust_spatial_radius",
    ):
        if float(getattr(args, radius_name)) < 0.0:
            parser.error(f"--{radius_name} must be non-negative")
    if int(args.K) == 0 and int(args.adaptation_steps) != 0:
        parser.error("K=0 must use adaptation_steps=0")
    if int(args.K) == 0 and abs(float(args.anchor_alpha)) > 1e-12:
        parser.error("K=0 must use anchor_alpha=0")
    if args.adapt_solver == "ridge_coeff" and int(args.K) == 0:
        parser.error("--adapt_solver ridge_coeff requires K>0 target_support labels")
    if args.adapt_solver == "ridge_coeff" and args.adapt_scope != "coeff_only":
        parser.error("--adapt_solver ridge_coeff is legacy/internal and requires --adapt_scope coeff_only")
    if args.support_gate == "policy_default":
        args.support_gate = (
            "auto"
            if args.stage3_posterior_policy in {"conservative_coeff_posterior", "source_calibrated_mix"}
            else "off"
        )
    if args.stage3_posterior_policy in {"conservative_coeff_posterior", "source_calibrated_mix"}:
        if int(args.K) == 0:
            if args.adapt_scope != "none":
                parser.error(
                    f"--stage3_posterior_policy {args.stage3_posterior_policy} "
                    "requires --adapt_scope none for K=0"
                )
            if int(args.adaptation_steps) != 0:
                parser.error(
                    f"--stage3_posterior_policy {args.stage3_posterior_policy} "
                    "requires K=0 adaptation_steps=0"
                )
            if abs(float(args.anchor_alpha)) > 1e-12:
                parser.error(
                    f"--stage3_posterior_policy {args.stage3_posterior_policy} "
                    "requires K=0 anchor_alpha=0"
                )
        else:
            audit_identity_no_update = (
                bool(args.audit_identity)
                and args.adapt_scope == "none"
                and int(args.adaptation_steps) == 0
                and abs(float(args.anchor_alpha)) <= 1e-12
            )
            policy_no_update = (
                bool(args.safe_policy_json)
                and args.policy_source == "source_side_episode_calibration"
                and "adapt_scope" in set(getattr(args, "safe_policy_selected_keys", []))
                and args.adapt_scope == "none"
                and int(args.adaptation_steps) == 0
                and abs(float(args.anchor_alpha)) <= 1e-12
            )
            policy_enabled_gain = (
                bool(args.safe_policy_json)
                and args.policy_source == "source_side_episode_calibration"
                and "adapt_scope" in set(getattr(args, "safe_policy_selected_keys", []))
                and args.adapt_scope == "coeff_gain"
            )
            if (
                args.adapt_scope != "coeff_only"
                and not policy_enabled_gain
                and not audit_identity_no_update
                and not policy_no_update
            ):
                parser.error(
                    f"--stage3_posterior_policy {args.stage3_posterior_policy} requires "
                    "--adapt_scope coeff_only for K>0 unless source-side SAFE policy "
                    "explicitly selects coeff_gain/no-update or --audit_identity requests a no-update path"
                )
    if args.audit_identity:
        if args.adapt_solver != "adamw":
            parser.error("--audit_identity requires --adapt_solver adamw")
        if args.adapt_scope != "none":
            parser.error("--audit_identity requires --adapt_scope none")
        if int(args.adaptation_steps) != 0:
            parser.error("--audit_identity requires --adaptation_steps 0")
        if abs(float(args.anchor_alpha)) > 1e-12:
            parser.error("--audit_identity requires --anchor_alpha 0")
    args.target_val_usage = "unused_in_main_protocol"
    args.model_selection_source = "source_val_preregistered"
    return args


def load_source_checkpoint_for_few_shot(
    checkpoint_path: str,
    device: torch.device,
    target_latent_dim: int = 32,
    enable_target_spatial_refine: bool = False,
) -> FewShotAdaptationState:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"source checkpoint not found: {ckpt_path}")
    source_checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    source_config = dict(source_checkpoint.get("config", {}))
    model_type = source_config.get("model_type", "prompt_conditioned")
    if model_type != "hyperda_basis_adapter":
        raise ValueError(
            "train_hyperda_few_shot_adapt.py requires config.model_type="
            f"'hyperda_basis_adapter', got {model_type!r}"
        )

    model = HyperAdapterConditionalResUNet(
        in_channels=12,
        out_channels=2,
        width=int(source_config.get("width", 32)),
        prompt_dim=int(source_config.get("prompt_dim", 64)),
        hyper_n_basis=int(source_config.get("hyper_n_basis", 8)),
        hyper_adapter_bottleneck=source_config.get("hyper_adapter_bottleneck"),
        hyper_adapter_scale=float(source_config.get("hyper_adapter_scale", 1.0)),
        hyper_coeff_generator=source_config.get("hyper_coeff_generator", "per_adapter"),
        hyper_rank_gate_top_k=int(source_config.get("hyper_rank_gate_top_k", 4)),
        hyper_rank_gate_temperature_init=float(source_config.get("hyper_rank_gate_temperature_init", 1.0)),
        hyper_adapter_param_style=source_config.get("hyper_adapter_param_style", "basis_1x1"),
        hyper_reliability_gate=source_config.get("hyper_reliability_gate", "none"),
        hyper_reliability_init=float(source_config.get("hyper_reliability_init", 0.95)),
        hyper_source_saliency_prior=source_config.get("hyper_source_saliency_prior"),
        hyper_source_saliency_prior_beta=float(source_config.get("hyper_source_saliency_prior_beta", 0.0)),
        hyper_source_saliency_prior_path=source_config.get("hyper_source_saliency_prior_path", ""),
        hyper_source_saliency_prior_application=source_config.get(
            "hyper_source_saliency_prior_application",
            "soft_regularization_metadata",
        ),
        hyper_prompt_manifold_reliability=bool(source_config.get("hyper_prompt_manifold_reliability", False)),
        hyper_prompt_manifold_reliability_strength=float(
            source_config.get("hyper_prompt_manifold_reliability_strength", 0.0)
        ),
        hyper_enable_film=bool(source_config.get("hyper_enable_film", True)),
        hyper_enable_adapters=bool(source_config.get("hyper_enable_adapters", True)),
        zero_shot_prior_form=source_config.get("zero_shot_prior_form", "direct_hyper"),
        source_residual_rho=float(source_config.get("source_residual_rho", source_config.get("zero_shot_rho", 1.0))),
        source_residual_gate=source_config.get("source_residual_gate", "prompt_reliability_scalar"),
        source_residual_gate_init=float(source_config.get("source_residual_gate_init", 0.95)),
        source_residual_reliability_dim=int(source_config.get("source_residual_reliability_dim", 5)),
        zero_raw_increment_init=bool(source_config.get("zero_raw_increment_init", False)),
        enable_target_adaptation=True,
        target_latent_dim=target_latent_dim,
        enable_target_spatial_refine=enable_target_spatial_refine,
    )
    load_result = model.load_state_dict(source_checkpoint["model_state_dict"], strict=False)
    unexpected = [k for k in load_result.unexpected_keys if not k.startswith("target_")]
    if unexpected:
        raise RuntimeError(f"unexpected source checkpoint model keys: {unexpected[:8]}")
    model.to(device)
    model.freeze_source_prior_for_target_adaptation()

    prompt_encoder = _build_source_prompt_encoder(source_config)
    prompt_state = source_checkpoint.get("prompt_encoder_state_dict")
    if prompt_state is not None:
        prompt_encoder.load_state_dict(prompt_state)
    prompt_encoder.to(device).eval()
    for param in prompt_encoder.parameters():
        param.requires_grad_(False)

    normalization = {
        "ch_mean": _as_list(source_config.get("ch_mean"), 12, 0.0),
        "ch_std": _as_list(source_config.get("ch_std"), 12, 1.0),
        "inc_mean": _as_list(source_config.get("inc_mean"), 2, 0.0) if source_config.get("inc_mean") is not None else None,
        "inc_std": _as_list(source_config.get("inc_std"), 2, 1.0) if source_config.get("inc_std") is not None else None,
    }
    return FewShotAdaptationState(
        model=model,
        prompt_encoder=prompt_encoder,
        source_checkpoint=source_checkpoint,
        source_config=source_config,
        normalization=normalization,
    )


def _loader(dataset: HydroDADataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=_collate_target_batch,
    )


def build_few_shot_target_context_prompt_state(
    state: FewShotAdaptationState,
    samples: Iterable[Dict[str, Any]],
    target_region: str,
    device: torch.device,
    context_hash: str = "",
) -> Dict[str, Any]:
    return build_target_context_prompt_state(
        samples=samples,
        prompt_encoder=state.prompt_encoder,
        normalize_x=lambda x: _normalize_x(x, state.normalization),
        target_region_embedding=_target_region_embedding(state, target_region, device),
        device=device,
        context_hash=context_hash,
        context_encoder=state.source_config.get("context_encoder", "current_mean_std"),
    )


def compose_target_context_reliability_features_from_state(
    state: Dict[str, Any],
    months: int | Iterable[int] | torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    normalized = normalize_target_context_prompt_state(state)
    schema = normalized.get("reliability_feature_schema", [])
    if isinstance(months, torch.Tensor):
        month_values = [int(v) for v in months.detach().cpu().view(-1).tolist()]
    elif isinstance(months, int):
        month_values = [int(months)]
    else:
        month_values = [int(v) for v in months]
    rows = [
        normalized["reliability_features"].get(
            str(max(1, min(12, int(month_value)))),
            [0.0] * len(schema),
        )
        for month_value in month_values
    ]
    return torch.as_tensor(rows, dtype=torch.float32, device=device)


def masked_huber_loss_components(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    reduction: str,
    delta: float,
    surface_weight: float,
    rootzone_weight: float,
    latitude_weight: Optional[torch.Tensor] = None,
    use_lat_weight: bool = False,
    increment_scale: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Masked Huber components with explicit support-cycle reduction semantics."""
    if reduction not in SUPPORT_LOSS_REDUCTIONS:
        raise ValueError(f"unsupported support loss reduction: {reduction!r}")
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    valid_weight = mask.float().to(pred.device)
    if use_lat_weight and latitude_weight is not None:
        latw = latitude_weight.to(pred.device).float()
        if latw.ndim == 2:
            latw = latw.unsqueeze(0).unsqueeze(0)
        elif latw.ndim == 3:
            latw = latw.unsqueeze(1)
        valid_weight = valid_weight * latw
    valid_weight = valid_weight.expand_as(pred)
    if increment_scale is None:
        scale = torch.ones(1, 2, 1, 1, dtype=pred.dtype, device=pred.device)
    else:
        scale = increment_scale.to(pred.device, dtype=pred.dtype).view(1, 2, 1, 1)
    diff = (pred.float() - target.float()) / scale.float()
    abs_diff = torch.abs(diff)
    delta_t = torch.as_tensor(float(delta), dtype=pred.dtype, device=pred.device)
    loss_raw = torch.where(
        abs_diff < delta_t,
        0.5 * diff.square() / delta_t.clamp_min(1e-12),
        abs_diff - 0.5 * delta_t,
    )
    weighted = loss_raw * valid_weight.float()
    if reduction == "cycle_balanced":
        loss_per_channel = weighted.sum(dim=(2, 3))
        weight_per_channel = valid_weight.sum(dim=(2, 3)).clamp_min(1.0)
        surface_loss = (loss_per_channel[:, 0] / weight_per_channel[:, 0]).mean()
        rootzone_loss = (loss_per_channel[:, 1] / weight_per_channel[:, 1]).mean()
    else:
        loss_per_channel = weighted.sum(dim=(0, 2, 3))
        weight_per_channel = valid_weight.sum(dim=(0, 2, 3)).clamp_min(1.0)
        surface_loss = loss_per_channel[0] / weight_per_channel[0]
        rootzone_loss = loss_per_channel[1] / weight_per_channel[1]
    total_loss = float(surface_weight) * surface_loss + float(rootzone_weight) * rootzone_loss
    return {
        "surface_loss": surface_loss,
        "rootzone_loss": rootzone_loss,
        "total_loss": total_loss,
        "valid_weight_sum": valid_weight.sum().detach(),
        "valid_pixel_count": mask.expand_as(pred).sum().detach(),
        "valid_pixel_fraction": (mask.expand_as(pred).sum() / max(1.0, float(mask.expand_as(pred).numel()))).detach(),
    }


def few_shot_batch_loss(
    state: FewShotAdaptationState,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    target_context_prompt_state: Dict[str, Any],
    loss_fn: nn.Module,
    normalize_increment: bool,
    lambda_prior: float,
    lambda_latent: float,
    lambda_gain: float,
    lambda_gain_smooth: float,
    lambda_analysis: float = 0.25,
    support_loss_reduction: str = "global_pixel",
) -> Dict[str, torch.Tensor]:
    """Compute few-shot loss with the frozen target-context monthly prompt state.

    Support batch inputs provide fields and supervised losses only; they do not
    update or summarize the target-context prompt state.
    """
    x = batch["x"].to(device)
    months = batch["months"].to(device)
    x_norm = _normalize_x(x, state.normalization)
    z = compose_target_context_prompt_from_state(target_context_prompt_state, months, device=device)
    reliability_features = compose_target_context_reliability_features_from_state(
        target_context_prompt_state,
        months,
        device=device,
    )
    pred = _model_forward(state.model, x_norm, z, months, x, reliability_features=reliability_features)
    target = _target_tensor(
        batch["increment_surface"].to(device),
        batch["increment_rootzone"].to(device),
        state.normalization,
        normalize_increment=normalize_increment,
    )
    loss_mask = batch["loss_mask"].to(device)
    latitude_weight = batch.get("latitude_weight")
    if latitude_weight is not None:
        latitude_weight = latitude_weight.to(device)
    if support_loss_reduction not in SUPPORT_LOSS_REDUCTIONS:
        raise ValueError(
            f"unsupported support_loss_reduction={support_loss_reduction!r}; "
            f"expected one of {list(SUPPORT_LOSS_REDUCTIONS)}"
        )
    if support_loss_reduction == "cycle_balanced":
        losses = masked_huber_loss_components(
            pred=pred,
            target=target,
            mask=loss_mask,
            latitude_weight=latitude_weight if isinstance(loss_fn, WeightedMaskedHuberLoss) else None,
            reduction="cycle_balanced",
            delta=getattr(loss_fn, "delta", 1.0),
            surface_weight=getattr(loss_fn, "surface_weight", 1.0),
            rootzone_weight=getattr(loss_fn, "rootzone_weight", 1.0),
            use_lat_weight=bool(getattr(loss_fn, "use_lat_weight", False)),
            increment_scale=torch.ones(2, dtype=torch.float32, device=device)
            if normalize_increment and isinstance(loss_fn, WeightedMaskedHuberLoss)
            else None,
        )
    elif isinstance(loss_fn, WeightedMaskedHuberLoss):
        inc_scale = torch.ones(2, dtype=torch.float32, device=device) if normalize_increment else None
        losses = loss_fn(pred, target, loss_mask, latitude_weight=latitude_weight, increment_scale=inc_scale)
    else:
        losses = loss_fn(pred, target, loss_mask)
    analysis_losses = _analysis_loss(
        pred=pred,
        target=target,
        batch=batch,
        normalization=state.normalization,
        normalize_increment=normalize_increment,
        loss_fn=loss_fn,
        loss_mask=loss_mask,
        latitude_weight=latitude_weight,
    )
    if analysis_losses is not None:
        losses["analysis_loss"] = analysis_losses["total_loss"]
        losses["analysis_surface_loss"] = analysis_losses["surface_loss"].detach()
        losses["analysis_rootzone_loss"] = analysis_losses["rootzone_loss"].detach()
    else:
        losses["analysis_loss"] = torch.zeros((), dtype=losses["total_loss"].dtype, device=losses["total_loss"].device)
    if "forecast_surface" in batch and "forecast_rootzone" in batch:
        forecast = torch.stack(
            [
                batch["forecast_surface"].to(pred.device),
                batch["forecast_rootzone"].to(pred.device),
            ],
            dim=1,
        )
        pred_analysis = forecast + _denormalize_increment(pred, state.normalization, normalize_increment)
        true_analysis = forecast + _denormalize_increment(target, state.normalization, normalize_increment)
        losses["pred_analysis_physical"] = pred_analysis.detach()
        losses["true_analysis_physical"] = true_analysis.detach()
    reg = adaptation_regularization(state.model, lambda_prior, lambda_latent, lambda_gain, lambda_gain_smooth)
    losses["regularization_loss"] = reg.detach()
    losses["objective"] = losses["total_loss"] + lambda_analysis * losses["analysis_loss"] + reg
    return losses


def train_fixed_steps(
    state: FewShotAdaptationState,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_context_prompt_state: Dict[str, Any],
    loss_fn: nn.Module,
    normalize_increment: bool,
    adaptation_steps: int,
    grad_clip: Optional[float],
    lambda_prior: float,
    lambda_latent: float,
    lambda_gain: float,
    lambda_gain_smooth: float,
    lambda_analysis: float,
    log_every_steps: int = 50,
    support_loss_reduction: str = "global_pixel",
    trust_region_anchor_state: Optional[Dict[str, torch.Tensor]] = None,
    trust_region_mode: str = "none",
    trust_total_radius: float = 0.0,
    trust_prompt_radius: float = 0.0,
    trust_gain_radius: float = 0.0,
    trust_coeff_radius: float = 0.0,
    trust_spatial_radius: float = 0.0,
) -> List[Dict[str, float]]:
    history: List[Dict[str, float]] = []
    if adaptation_steps <= 0:
        return history
    state.model.train()
    data_iter = iter(loader)
    for step in range(1, adaptation_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        optimizer.zero_grad(set_to_none=True)
        losses = few_shot_batch_loss(
            state,
            batch,
            device,
            target_context_prompt_state,
            loss_fn,
            normalize_increment,
            lambda_prior,
            lambda_latent,
            lambda_gain,
            lambda_gain_smooth,
            lambda_analysis=lambda_analysis,
            support_loss_reduction=support_loss_reduction,
        )
        losses["objective"].backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([p for p in state.model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()
        projection = project_target_state_to_trust_region(
            state.model,
            trust_region_anchor_state or {},
            mode=trust_region_mode,
            total_radius=trust_total_radius,
            prompt_radius=trust_prompt_radius,
            gain_radius=trust_gain_radius,
            coeff_radius=trust_coeff_radius,
            spatial_radius=trust_spatial_radius,
        )
        row = {
            "step": float(step),
            "objective": float(losses["objective"].detach().cpu()),
            "total_loss": float(losses["total_loss"].detach().cpu()),
            "analysis_loss": float(losses["analysis_loss"].detach().cpu()),
            "regularization_loss": float(losses["regularization_loss"].detach().cpu()),
            "trust_projection_applied": float(bool(projection.get("projection_applied", False))),
            "trust_projection_pre_total": float((projection.get("pre_projection_drift") or {}).get("total", 0.0)),
            "trust_projection_post_total": float((projection.get("post_projection_drift") or {}).get("total", 0.0)),
        }
        for group_name in (
            "target_prompt",
            "monthly_gain",
            "adapter_coeff_bottleneck",
            "adapter_coeff_dec2",
            "adapter_coeff_dec1",
            "spatial_refine",
            "other_target_specific",
        ):
            row[f"trust_projection_pre_{group_name}"] = float(
                (projection.get("pre_projection_drift") or {}).get(group_name, 0.0)
            )
            row[f"trust_projection_post_{group_name}"] = float(
                (projection.get("post_projection_drift") or {}).get(group_name, 0.0)
            )
        history.append(row)
        if log_every_steps > 0 and (step == 1 or step % log_every_steps == 0 or step == adaptation_steps):
            print(
                f"step={step}/{adaptation_steps} "
                f"objective={row['objective']:.6f} total={row['total_loss']:.6f} "
                f"analysis={row['analysis_loss']:.6f}",
                flush=True,
            )
    return history


@torch.no_grad()
def standard_support_loss_from_loader(
    state: FewShotAdaptationState,
    loader: DataLoader,
    device: torch.device,
    target_context_prompt_state: Dict[str, Any],
    loss_fn: nn.Module,
    normalize_increment: bool,
    lambda_prior: float,
    lambda_latent: float,
    lambda_gain: float,
    lambda_gain_smooth: float,
    lambda_analysis: float,
    support_loss_reduction: str = "global_pixel",
) -> Dict[str, Optional[float]]:
    """Evaluate the AdamW adaptation objective over one full support pass."""
    totals = {
        "objective": 0.0,
        "total_loss": 0.0,
        "surface_loss": 0.0,
        "rootzone_loss": 0.0,
        "analysis_loss": 0.0,
        "analysis_surface_loss": 0.0,
        "analysis_rootzone_loss": 0.0,
        "regularization_loss": 0.0,
    }
    count = 0
    was_training = state.model.training
    state.model.eval()
    try:
        for batch in loader:
            losses = few_shot_batch_loss(
                state=state,
                batch=batch,
                device=device,
                target_context_prompt_state=target_context_prompt_state,
                loss_fn=loss_fn,
                normalize_increment=normalize_increment,
                lambda_prior=lambda_prior,
                lambda_latent=lambda_latent,
                lambda_gain=lambda_gain,
                lambda_gain_smooth=lambda_gain_smooth,
                lambda_analysis=lambda_analysis,
                support_loss_reduction=support_loss_reduction,
            )
            batch_count = int(batch["x"].shape[0])
            count += batch_count
            for key in totals:
                if key in losses:
                    totals[key] += float(losses[key].detach().cpu()) * batch_count
    finally:
        state.model.train(was_training)
    if count <= 0:
        return {
            "standard_support_loss_full_support": None,
            "standard_support_surface_loss_full_support": None,
            "standard_support_rootzone_loss_full_support": None,
            "standard_support_objective_full_support": None,
            "standard_support_increment_loss_full_support": None,
            "standard_support_analysis_loss_full_support": None,
            "standard_support_analysis_surface_loss_full_support": None,
            "standard_support_analysis_rootzone_loss_full_support": None,
            "standard_support_regularization_loss_full_support": None,
        }
    return {
        "standard_support_loss_full_support": float(totals["total_loss"] / count),
        "standard_support_surface_loss_full_support": float(totals["surface_loss"] / count),
        "standard_support_rootzone_loss_full_support": float(totals["rootzone_loss"] / count),
        "standard_support_objective_full_support": float(totals["objective"] / count),
        "standard_support_increment_loss_full_support": float(totals["total_loss"] / count),
        "standard_support_analysis_loss_full_support": float(totals["analysis_loss"] / count),
        "standard_support_analysis_surface_loss_full_support": float(totals["analysis_surface_loss"] / count),
        "standard_support_analysis_rootzone_loss_full_support": float(totals["analysis_rootzone_loss"] / count),
        "standard_support_regularization_loss_full_support": float(totals["regularization_loss"] / count),
    }


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def decide_support_gate(
    *,
    before: Dict[str, Optional[float]],
    after: Dict[str, Optional[float]],
    enabled: bool,
    min_delta: float,
    rootzone_tolerance: float,
) -> Dict[str, Any]:
    """Decide whether to keep a Stage-3 target posterior using target-support losses only."""
    before_obj = _float_or_none(before.get("standard_support_objective_full_support"))
    after_obj = _float_or_none(after.get("standard_support_objective_full_support"))
    before_loss = _float_or_none(before.get("standard_support_loss_full_support"))
    after_loss = _float_or_none(after.get("standard_support_loss_full_support"))
    before_surface = _float_or_none(before.get("standard_support_surface_loss_full_support"))
    after_surface = _float_or_none(after.get("standard_support_surface_loss_full_support"))
    before_rootzone = _float_or_none(before.get("standard_support_rootzone_loss_full_support"))
    after_rootzone = _float_or_none(after.get("standard_support_rootzone_loss_full_support"))

    objective_delta = None if before_obj is None or after_obj is None else after_obj - before_obj
    loss_delta = None if before_loss is None or after_loss is None else after_loss - before_loss
    surface_delta = None if before_surface is None or after_surface is None else after_surface - before_surface
    rootzone_delta = None if before_rootzone is None or after_rootzone is None else after_rootzone - before_rootzone

    summary: Dict[str, Any] = {
        "support_gate_enabled": bool(enabled),
        "support_gate_label_source": "target_support_only",
        "support_gate_min_delta": float(min_delta),
        "support_gate_rootzone_tolerance": float(rootzone_tolerance),
        "support_objective_before": before_obj,
        "support_objective_after": after_obj,
        "support_objective_delta": objective_delta,
        "support_loss_before": before_loss,
        "support_loss_after": after_loss,
        "support_loss_delta": loss_delta,
        "support_surface_loss_before": before_surface,
        "support_surface_loss_after": after_surface,
        "support_surface_loss_delta": surface_delta,
        "support_rootzone_loss_before": before_rootzone,
        "support_rootzone_loss_after": after_rootzone,
        "support_rootzone_loss_delta": rootzone_delta,
        "support_candidate_objective_after": after_obj,
        "support_candidate_objective_delta": objective_delta,
        "support_candidate_loss_after": after_loss,
        "support_candidate_loss_delta": loss_delta,
        "support_candidate_surface_loss_after": after_surface,
        "support_candidate_surface_loss_delta": surface_delta,
        "support_candidate_rootzone_loss_after": after_rootzone,
        "support_candidate_rootzone_loss_delta": rootzone_delta,
        "support_gate_reject_reason": [],
    }
    if not enabled:
        summary["support_gate_status"] = "disabled"
        summary["stage3_posterior_decision"] = "accepted"
        return summary

    reject_reasons: List[str] = []
    if objective_delta is None:
        reject_reasons.append("missing_support_objective")
    elif objective_delta >= -float(min_delta):
        reject_reasons.append("objective_not_improved")
    if rootzone_delta is None:
        reject_reasons.append("missing_rootzone_guard")
    elif rootzone_delta > float(rootzone_tolerance):
        reject_reasons.append("rootzone_regression")

    if reject_reasons:
        summary["support_gate_status"] = "support_only_rejected_to_k0_anchor"
        summary["stage3_posterior_decision"] = "rejected_to_k0_anchor"
        summary["support_gate_reject_reason"] = reject_reasons
    else:
        summary["support_gate_status"] = "accepted"
        summary["stage3_posterior_decision"] = "accepted"
    return summary


def support_gate_summary_for_k0() -> Dict[str, Any]:
    return {
        "support_gate_enabled": False,
        "support_gate_status": "skipped_k0_no_support",
        "stage3_posterior_decision": "no_update",
        "stage3_no_update_contract": "K0_fixed_no_update_source_prior_identity",
        "support_gate_label_source": "none_k0_no_target_labels",
        "support_gate_policy_role": "not_applicable_k0_no_support",
        "support_gate_min_delta": 0.0,
        "support_gate_rootzone_tolerance": 0.0,
        "support_gate_reject_reason": [],
        "support_objective_before": None,
        "support_objective_after": None,
        "support_objective_delta": None,
        "support_loss_before": None,
        "support_loss_after": None,
        "support_loss_delta": None,
        "support_surface_loss_before": None,
        "support_surface_loss_after": None,
        "support_surface_loss_delta": None,
        "support_rootzone_loss_before": None,
        "support_rootzone_loss_after": None,
        "support_rootzone_loss_delta": None,
    }


def is_source_calibrated_kshot_no_update(args: argparse.Namespace) -> bool:
    """Return True when source-side SAFE policy intentionally chooses no update."""
    if int(getattr(args, "K", 0)) <= 0:
        return False
    if str(getattr(args, "policy_source", "")) != "source_side_episode_calibration":
        return False
    if str(getattr(args, "adapt_scope", "")) != "none":
        return False
    if int(getattr(args, "adaptation_steps", 0) or 0) != 0:
        return False
    if abs(float(getattr(args, "anchor_alpha", 0.0) or 0.0)) > 1e-12:
        return False
    return abs(float(getattr(args, "adapt_mix_rho", 0.0) or 0.0)) <= 1e-12


def support_gate_summary_for_source_calibrated_no_update(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "support_gate_enabled": False,
        "support_gate_status": "source_calibrated_no_update",
        "stage3_posterior_decision": "no_update",
        "stage3_no_update_contract": "Kshot_source_side_policy_selected_no_update",
        "support_gate_label_source": "source_side_episode_calibration",
        "support_gate_policy_role": "source_side_policy_selected_no_update",
        "support_gate_min_delta": float(getattr(args, "support_gate_min_delta", 0.0) or 0.0),
        "support_gate_rootzone_tolerance": float(getattr(args, "support_gate_rootzone_tolerance", 0.0) or 0.0),
        "support_gate_reject_reason": [],
        "source_policy_candidate_id": str(getattr(args, "source_policy_candidate_id", "")),
        "support_objective_before": None,
        "support_objective_after": None,
        "support_objective_delta": None,
        "support_loss_before": None,
        "support_loss_after": None,
        "support_loss_delta": None,
        "support_surface_loss_before": None,
        "support_surface_loss_after": None,
        "support_surface_loss_delta": None,
        "support_rootzone_loss_before": None,
        "support_rootzone_loss_after": None,
        "support_rootzone_loss_delta": None,
    }


def paper_facing_status_for_stage3(
    *,
    K: int,
    policy_source: str,
    stage3_posterior_decision: str,
) -> Dict[str, Any]:
    if int(K) == 0:
        return {"paper_facing_run": True, "diagnostic_run_reason": ""}
    if str(policy_source) != "source_side_episode_calibration":
        return {
            "paper_facing_run": False,
            "diagnostic_run_reason": "missing_source_side_safe_policy_json",
        }
    if str(stage3_posterior_decision) == "rejected_to_k0_anchor":
        return {
            "paper_facing_run": False,
            "diagnostic_run_reason": "source_policy_or_gate_rejected_to_k0_anchor",
        }
    return {"paper_facing_run": True, "diagnostic_run_reason": ""}


def coefficient_residual_vector(model: nn.Module) -> torch.Tensor:
    """Return concatenated adapter coefficient residual logits."""
    params = dict(model.named_parameters())
    pieces: List[torch.Tensor] = []
    for name in COEFF_RESIDUAL_PARAMETER_NAMES:
        if name not in params:
            raise ValueError(f"missing coefficient residual parameter: {name}")
        pieces.append(params[name].detach().flatten().float().cpu())
    return torch.cat(pieces, dim=0)


def set_coefficient_residual_vector(model: nn.Module, vector: torch.Tensor) -> None:
    """Set only adapter coefficient residual logits from a flat vector."""
    params = dict(model.named_parameters())
    cursor = 0
    with torch.no_grad():
        for name in COEFF_RESIDUAL_PARAMETER_NAMES:
            if name not in params:
                raise ValueError(f"missing coefficient residual parameter: {name}")
            param = params[name]
            count = int(param.numel())
            piece = vector.detach().to(device=param.device, dtype=param.dtype)[cursor: cursor + count]
            if piece.numel() != count:
                raise ValueError(
                    f"coefficient residual vector has wrong length: expected at least {cursor + count}, got {vector.numel()}"
                )
            param.copy_(piece.view_as(param))
            cursor += count
    if cursor != int(vector.numel()):
        raise ValueError(f"coefficient residual vector has extra values: expected {cursor}, got {vector.numel()}")


def solve_ridge_coefficients(
    design_t_design: torch.Tensor,
    design_t_residual: torch.Tensor,
    ridge_lambda: float,
) -> RidgeSolveResult:
    """Solve ``(X^T X + lambda I) delta = X^T y`` with finite diagnostics."""
    if design_t_design.ndim != 2 or design_t_design.shape[0] != design_t_design.shape[1]:
        raise ValueError("design_t_design must be square")
    if design_t_residual.ndim != 1 or design_t_residual.shape[0] != design_t_design.shape[0]:
        raise ValueError("design_t_residual shape must match design_t_design")
    if float(ridge_lambda) < 0.0:
        raise ValueError("ridge_lambda must be non-negative")
    feature_dim = int(design_t_residual.numel())
    lhs = design_t_design.detach().double().cpu()
    rhs = design_t_residual.detach().double().cpu()
    lhs = lhs + torch.eye(feature_dim, dtype=lhs.dtype) * float(ridge_lambda)
    status = "solved"
    try:
        delta = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        delta = torch.linalg.pinv(lhs) @ rhs
        status = "pinv_solved"
    singular_values = torch.linalg.svdvals(lhs)
    eps = torch.finfo(singular_values.dtype).eps
    min_sv = singular_values[-1].clamp_min(eps)
    condition_number = float((singular_values[0] / min_sv).item()) if singular_values.numel() else float("nan")
    rank = int(torch.linalg.matrix_rank(lhs).item())
    delta = torch.nan_to_num(delta.float(), nan=0.0, posinf=0.0, neginf=0.0)
    diagnostics = {
        "status": status,
        "ridge_lambda": float(ridge_lambda),
        "feature_dim": feature_dim,
        "condition_number": condition_number,
        "rank": rank,
        "raw_delta_norm": float(torch.linalg.vector_norm(delta).item()),
    }
    return RidgeSolveResult(delta=delta, diagnostics=diagnostics)


def _apply_vector_norm_clip(vector: torch.Tensor, max_norm: float) -> Tuple[torch.Tensor, bool]:
    if float(max_norm) <= 0.0:
        return torch.zeros_like(vector), bool(torch.linalg.vector_norm(vector).item() > 0.0)
    norm = torch.linalg.vector_norm(vector.float())
    if float(norm.item()) <= float(max_norm):
        return vector, False
    return vector * (float(max_norm) / float(norm.item())), True


def _ridge_observation_weights(
    batch: Dict[str, torch.Tensor],
    pred: torch.Tensor,
    surface_weight: float,
    rootzone_weight: float,
    use_lat_weighted_loss: bool,
) -> torch.Tensor:
    mask = batch["loss_mask"].to(pred.device).float()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    valid_weight = mask
    latitude_weight = batch.get("latitude_weight")
    if use_lat_weighted_loss and latitude_weight is not None:
        lat_w = latitude_weight.to(pred.device).float()
        if lat_w.ndim == 2:
            lat_w = lat_w.unsqueeze(0).unsqueeze(0)
        elif lat_w.ndim == 3:
            lat_w = lat_w.unsqueeze(1)
        valid_weight = valid_weight * lat_w
    channel_weight = torch.tensor(
        [float(surface_weight), float(rootzone_weight)],
        dtype=pred.dtype,
        device=pred.device,
    ).view(1, 2, 1, 1)
    return valid_weight.expand_as(pred) * channel_weight


def _ridge_flatten_weighted(
    tensor: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return (tensor.float() * weights.float().clamp_min(0.0).sqrt()).reshape(-1)


def _ridge_select_feature_observations(
    vector: torch.Tensor,
    valid_observation_mask: torch.Tensor,
    max_feature_pixels: int,
) -> torch.Tensor:
    """Return a deterministic positive-weight observation subset.

    A pixel contributes both output channels, so the cap is converted to
    observations after filtering masked support observations. Selection is
    evenly spaced to avoid adding a random target-side choice.
    """
    vector = vector.reshape(-1)
    valid_indices = torch.nonzero(valid_observation_mask.reshape(-1).to(torch.bool), as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return vector.new_empty((0,))
    valid_vector = vector.index_select(0, valid_indices.to(vector.device))
    valid_count = int(valid_vector.numel())
    max_feature_pixels = int(max_feature_pixels)
    if max_feature_pixels <= 0 or valid_count <= max_feature_pixels * 2:
        return valid_vector
    observation_cap = max(1, max_feature_pixels * 2)
    indices = torch.linspace(
        0,
        valid_count - 1,
        steps=observation_cap,
        dtype=torch.float64,
    ).round().long()
    indices = torch.unique_consecutive(indices.clamp_(0, valid_count - 1))
    if int(indices.numel()) < observation_cap:
        needed = observation_cap - int(indices.numel())
        extra = torch.arange(valid_count, dtype=torch.long)
        mask = torch.ones(valid_count, dtype=torch.bool)
        mask[indices] = False
        indices = torch.cat([indices, extra[mask][:needed]])
    return valid_vector.index_select(0, indices.to(vector.device))


def _ridge_subsample_rows(
    design: torch.Tensor,
    residual: torch.Tensor,
    max_feature_pixels: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the ridge feature-pixel cap once across all support rows."""
    row_count = int(residual.numel())
    max_feature_pixels = int(max_feature_pixels)
    if max_feature_pixels <= 0 or row_count <= max_feature_pixels * 2:
        return design, residual
    observation_cap = max(1, max_feature_pixels * 2)
    indices = torch.linspace(
        0,
        row_count - 1,
        steps=observation_cap,
        dtype=torch.float64,
    ).round().long()
    indices = torch.unique_consecutive(indices.clamp_(0, row_count - 1))
    if int(indices.numel()) < observation_cap:
        needed = observation_cap - int(indices.numel())
        extra = torch.arange(row_count, dtype=torch.long)
        mask = torch.ones(row_count, dtype=torch.bool)
        mask[indices] = False
        indices = torch.cat([indices, extra[mask][:needed]])
    return design.index_select(0, indices), residual.index_select(0, indices)


@torch.no_grad()
def _ridge_support_loss_from_loader(
    state: FewShotAdaptationState,
    loader: DataLoader,
    device: torch.device,
    target_context_prompt_state: Dict[str, Any],
    normalize_increment: bool,
    surface_weight: float,
    rootzone_weight: float,
    use_lat_weighted_loss: bool,
) -> Optional[float]:
    weighted_sq_sum = 0.0
    weight_sum = 0.0
    for batch in loader:
        pred = _ridge_forward_prediction(
            state=state,
            batch=batch,
            device=device,
            target_context_prompt_state=target_context_prompt_state,
        )
        target = _target_tensor(
            batch["increment_surface"].to(device),
            batch["increment_rootzone"].to(device),
            state.normalization,
            normalize_increment=normalize_increment,
        )
        weights = _ridge_observation_weights(
            batch=batch,
            pred=pred,
            surface_weight=surface_weight,
            rootzone_weight=rootzone_weight,
            use_lat_weighted_loss=use_lat_weighted_loss,
        )
        sq = (pred.float() - target.float()).square() * weights.float().clamp_min(0.0)
        weighted_sq_sum += float(sq.sum().detach().cpu().item())
        weight_sum += float(weights.float().clamp_min(0.0).sum().detach().cpu().item())
    if weight_sum <= 0.0:
        return None
    return float(weighted_sq_sum / weight_sum)


@torch.no_grad()
def _ridge_forward_prediction(
    state: FewShotAdaptationState,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    target_context_prompt_state: Dict[str, Any],
) -> torch.Tensor:
    x = batch["x"].to(device)
    months = batch["months"].to(device)
    x_norm = _normalize_x(x, state.normalization)
    z = compose_target_context_prompt_from_state(target_context_prompt_state, months, device=device)
    reliability_features = compose_target_context_reliability_features_from_state(
        target_context_prompt_state,
        months,
        device=device,
    )
    return _model_forward(state.model, x_norm, z, months, x, reliability_features=reliability_features)


def run_ridge_coeff_adaptation(
    state: FewShotAdaptationState,
    loader: DataLoader,
    device: torch.device,
    target_context_prompt_state: Dict[str, Any],
    normalize_increment: bool,
    ridge_lambda: float,
    ridge_clip_coeff_norm: float,
    ridge_trust_region_radius: float,
    ridge_max_feature_pixels: int,
    ridge_standardize_features: bool,
    surface_weight: float,
    rootzone_weight: float,
    use_lat_weighted_loss: bool,
    finite_difference_eps: float = 1e-3,
) -> Dict[str, Any]:
    """Solve a local linear ridge update for adapter coefficient residual logits."""
    state.model.eval()
    for name, param in state.model.named_parameters():
        param.requires_grad_(name in COEFF_RESIDUAL_PARAMETER_NAMES)
    base_vector = coefficient_residual_vector(state.model)
    feature_dim = int(base_vector.numel())
    xtx = torch.zeros(feature_dim, feature_dim, dtype=torch.float64)
    xtr = torch.zeros(feature_dim, dtype=torch.float64)
    column_sq_sum = torch.zeros(feature_dim, dtype=torch.float64)
    column_scale = torch.ones(feature_dim, dtype=torch.float64)
    design_blocks: List[torch.Tensor] = []
    residual_blocks: List[torch.Tensor] = []
    support_count = 0
    masked_pixel_count = 0
    masked_observation_count = 0
    support_loss_before = _ridge_support_loss_from_loader(
        state=state,
        loader=loader,
        device=device,
        target_context_prompt_state=target_context_prompt_state,
        normalize_increment=normalize_increment,
        surface_weight=surface_weight,
        rootzone_weight=rootzone_weight,
        use_lat_weighted_loss=use_lat_weighted_loss,
    )

    try:
        for batch in loader:
            support_count += int(batch["x"].shape[0])
            pred_base = _ridge_forward_prediction(
                state=state,
                batch=batch,
                device=device,
                target_context_prompt_state=target_context_prompt_state,
            )
            target = _target_tensor(
                batch["increment_surface"].to(device),
                batch["increment_rootzone"].to(device),
                state.normalization,
                normalize_increment=normalize_increment,
            )
            residual_flat = target - pred_base
            weights = _ridge_observation_weights(
                batch=batch,
                pred=pred_base,
                surface_weight=surface_weight,
                rootzone_weight=rootzone_weight,
                use_lat_weighted_loss=use_lat_weighted_loss,
            )
            valid_mask = batch["loss_mask"].to(device)
            masked_pixel_count += int((valid_mask > 0).sum().detach().cpu().item())
            masked_observation_count += int((weights > 0).sum().detach().cpu().item())
            weighted_residual = _ridge_flatten_weighted(residual_flat, weights).double().cpu()
            valid_observation_mask = (weights.reshape(-1) > 0).detach().cpu()
            weighted_residual = _ridge_select_feature_observations(
                weighted_residual,
                valid_observation_mask=valid_observation_mask,
                max_feature_pixels=0,
            )
            design_columns: List[torch.Tensor] = []
            for coeff_idx in range(feature_dim):
                perturb = torch.zeros_like(base_vector)
                perturb[coeff_idx] = float(finite_difference_eps)
                set_coefficient_residual_vector(state.model, base_vector + perturb)
                pred_perturbed = _ridge_forward_prediction(
                    state=state,
                    batch=batch,
                    device=device,
                    target_context_prompt_state=target_context_prompt_state,
                )
                feature = (pred_perturbed - pred_base) / float(finite_difference_eps)
                weighted_feature = _ridge_flatten_weighted(feature, weights).double().cpu()
                weighted_feature = _ridge_select_feature_observations(
                    weighted_feature,
                    valid_observation_mask=valid_observation_mask,
                    max_feature_pixels=0,
                )
                design_columns.append(weighted_feature)
            design = torch.stack(design_columns, dim=1)
            design_blocks.append(design)
            residual_blocks.append(weighted_residual)
            set_coefficient_residual_vector(state.model, base_vector)
    finally:
        set_coefficient_residual_vector(state.model, base_vector)

    if design_blocks and residual_blocks:
        design_all = torch.cat(design_blocks, dim=0)
        residual_all = torch.cat(residual_blocks, dim=0)
        design_all, residual_all = _ridge_subsample_rows(
            design_all,
            residual_all,
            max_feature_pixels=int(ridge_max_feature_pixels),
        )
        feature_observation_count = int(residual_all.numel())
        xtx += design_all.T @ design_all
        xtr += design_all.T @ residual_all
        column_sq_sum += design_all.square().sum(dim=0)
    else:
        feature_observation_count = 0

    if masked_observation_count <= 0:
        diagnostics = {
            "status": "no_masked_support_observations",
            "ridge_lambda": float(ridge_lambda),
            "coefficient_norm": float(torch.linalg.vector_norm(base_vector).item()),
            "delta_norm": 0.0,
            "raw_delta_norm": 0.0,
            "support_loss_before": support_loss_before,
            "support_loss_after": support_loss_before,
            "support_count": support_count,
            "masked_pixel_count": masked_pixel_count,
            "masked_observation_count": masked_observation_count,
            "feature_pixel_count": 0,
            "feature_observation_count": feature_observation_count,
            "feature_dim": feature_dim,
            "ridge_max_feature_pixels": int(ridge_max_feature_pixels),
            "ridge_standardize_features": bool(ridge_standardize_features),
            "condition_number": None,
            "rank": 0,
            "trust_region_clipped": False,
            "coefficient_norm_clipped": False,
        }
        return diagnostics

    normalizer = float(max(1, feature_observation_count))
    if bool(ridge_standardize_features) and feature_observation_count > 0:
        column_scale = torch.sqrt(column_sq_sum / normalizer).clamp_min(1e-12)
        inv_scale = torch.diag(1.0 / column_scale)
        xtx = inv_scale @ xtx @ inv_scale
        xtr = inv_scale @ xtr
    solve = solve_ridge_coefficients(
        design_t_design=xtx.float() / normalizer,
        design_t_residual=xtr.float() / normalizer,
        ridge_lambda=float(ridge_lambda),
    )
    raw_delta = solve.delta.double()
    if bool(ridge_standardize_features) and feature_observation_count > 0:
        raw_delta = raw_delta / column_scale
    raw_delta = raw_delta.float()
    trusted_delta, trust_region_clipped = _apply_vector_norm_clip(raw_delta, float(ridge_trust_region_radius))
    solved_vector = base_vector + trusted_delta
    clipped_vector, coefficient_norm_clipped = _apply_vector_norm_clip(solved_vector, float(ridge_clip_coeff_norm))
    set_coefficient_residual_vector(state.model, clipped_vector)
    final_delta = clipped_vector - base_vector
    support_loss_after = _ridge_support_loss_from_loader(
        state=state,
        loader=loader,
        device=device,
        target_context_prompt_state=target_context_prompt_state,
        normalize_increment=normalize_increment,
        surface_weight=surface_weight,
        rootzone_weight=rootzone_weight,
        use_lat_weighted_loss=use_lat_weighted_loss,
    )
    diagnostics = {
        **solve.diagnostics,
        "coefficient_norm": float(torch.linalg.vector_norm(clipped_vector).item()),
        "delta_norm": float(torch.linalg.vector_norm(final_delta).item()),
        "raw_delta_norm": float(torch.linalg.vector_norm(raw_delta).item()),
        "support_loss_before": support_loss_before,
        "support_loss_after": support_loss_after,
        "support_loss_delta": (
            None
            if support_loss_before is None or support_loss_after is None
            else float(support_loss_after) - float(support_loss_before)
        ),
        "support_count": support_count,
        "masked_pixel_count": masked_pixel_count,
        "masked_observation_count": masked_observation_count,
        "feature_pixel_count": int((feature_observation_count + 1) // 2),
        "feature_observation_count": feature_observation_count,
        "feature_dim": feature_dim,
        "ridge_max_feature_pixels": int(ridge_max_feature_pixels),
        "ridge_standardize_features": bool(ridge_standardize_features),
        "feature_column_scale_min": float(column_scale.min().item()) if column_scale.numel() else None,
        "feature_column_scale_max": float(column_scale.max().item()) if column_scale.numel() else None,
        "trust_region_clipped": bool(trust_region_clipped),
        "coefficient_norm_clipped": bool(coefficient_norm_clipped),
    }
    return diagnostics


def _drift_group_for_key(name: str) -> str:
    if name.startswith("target_prompt."):
        return "target_prompt"
    if name.startswith("target_adapter_coefficient_residual_b."):
        return "adapter_coeff_bottleneck"
    if name.startswith("target_adapter_coefficient_residual_d2."):
        return "adapter_coeff_dec2"
    if name.startswith("target_adapter_coefficient_residual_d1."):
        return "adapter_coeff_dec1"
    if name.startswith("residual_gain."):
        return "monthly_gain"
    if name.startswith("target_spatial_refine."):
        return "spatial_refine"
    return "other_target_specific"


def _trust_projection_group_for_key(name: str) -> str:
    group = _drift_group_for_key(name)
    if group.startswith("adapter_coeff_"):
        return "coeff"
    if group == "target_prompt":
        return "prompt"
    if group == "monthly_gain":
        return "gain"
    if group == "spatial_refine":
        return "spatial"
    return "other"


def _project_state_subset_l2(
    anchor_state: Dict[str, torch.Tensor],
    adapted_state: Dict[str, torch.Tensor],
    names: List[str],
    radius: float,
) -> bool:
    radius = float(radius)
    if radius <= 0.0:
        scale = 0.0
    else:
        sq = 0.0
        for name in names:
            diff = adapted_state[name].float() - anchor_state[name].float()
            sq += float(diff.square().sum().item())
        norm = sq ** 0.5
        if norm <= radius:
            return False
        scale = radius / max(norm, 1e-12)
    for name in names:
        adapted_state[name] = anchor_state[name] + (adapted_state[name] - anchor_state[name]) * scale
    return True


def project_target_state_to_trust_region(
    model: nn.Module,
    anchor_state: Dict[str, torch.Tensor],
    *,
    mode: str,
    total_radius: float,
    prompt_radius: float,
    gain_radius: float,
    coeff_radius: float,
    spatial_radius: float,
) -> Dict[str, Any]:
    """Project target-specific state to a fixed L2 ball around initialization."""
    if mode not in TRUST_REGION_MODES:
        raise ValueError(f"unsupported trust_region_mode={mode!r}")
    if mode == "none":
        return {
            "projection_applied": False,
            "pre_projection_drift": {},
            "post_projection_drift": {},
            "projected_groups": [],
        }
    if not anchor_state:
        raise ValueError("trust-region projection requires a target adapter anchor state")
    adapted_state = extract_target_adapter_state(model)
    pre_drift = target_parameter_l2_drift(anchor_state, adapted_state)
    projected_groups: List[str] = []
    if mode == "global":
        if _project_state_subset_l2(anchor_state, adapted_state, sorted(adapted_state), float(total_radius)):
            projected_groups.append("total")
    else:
        radii = {
            "prompt": float(prompt_radius),
            "gain": float(gain_radius),
            "coeff": float(coeff_radius),
            "spatial": float(spatial_radius),
        }
        for group, radius in radii.items():
            names = [name for name in sorted(adapted_state) if _trust_projection_group_for_key(name) == group]
            if names and _project_state_subset_l2(anchor_state, adapted_state, names, radius):
                projected_groups.append(group)
        if float(total_radius) > 0.0:
            if _project_state_subset_l2(anchor_state, adapted_state, sorted(adapted_state), float(total_radius)):
                projected_groups.append("total")
    apply_target_adapter_state(model, adapted_state)
    post_drift = target_parameter_l2_drift(anchor_state, extract_target_adapter_state(model))
    return {
        "projection_applied": bool(projected_groups),
        "pre_projection_drift": pre_drift,
        "post_projection_drift": post_drift,
        "projected_groups": projected_groups,
    }


def summarize_trust_projection_history(train_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups = [
        "total",
        "target_prompt",
        "monthly_gain",
        "adapter_coeff_bottleneck",
        "adapter_coeff_dec2",
        "adapter_coeff_dec1",
        "spatial_refine",
        "other_target_specific",
    ]
    if not train_history:
        empty = {
            "trust_projection_step_count": 0,
            "trust_projection_applied_count": 0,
        }
        for group in groups:
            empty[f"trust_projection_pre_step_drift_max_{group}"] = None
            empty[f"trust_projection_post_step_drift_max_{group}"] = None
            empty[f"trust_projection_pre_step_drift_last_{group}"] = None
            empty[f"trust_projection_post_step_drift_last_{group}"] = None
        return empty
    pre_values = [float(row.get("trust_projection_pre_total", 0.0) or 0.0) for row in train_history]
    post_values = [float(row.get("trust_projection_post_total", 0.0) or 0.0) for row in train_history]
    summary = {
        "trust_projection_step_count": len(train_history),
        "trust_projection_applied_count": int(
            sum(1 for row in train_history if bool(row.get("trust_projection_applied", 0.0)))
        ),
        "trust_projection_pre_step_drift_max_total": max(pre_values) if pre_values else None,
        "trust_projection_post_step_drift_max_total": max(post_values) if post_values else None,
        "trust_projection_pre_step_drift_last_total": pre_values[-1] if pre_values else None,
        "trust_projection_post_step_drift_last_total": post_values[-1] if post_values else None,
    }
    for group in groups:
        if group == "total":
            continue
        pre_group_values = [float(row.get(f"trust_projection_pre_{group}", 0.0) or 0.0) for row in train_history]
        post_group_values = [float(row.get(f"trust_projection_post_{group}", 0.0) or 0.0) for row in train_history]
        summary[f"trust_projection_pre_step_drift_max_{group}"] = max(pre_group_values) if pre_group_values else None
        summary[f"trust_projection_post_step_drift_max_{group}"] = max(post_group_values) if post_group_values else None
        summary[f"trust_projection_pre_step_drift_last_{group}"] = pre_group_values[-1] if pre_group_values else None
        summary[f"trust_projection_post_step_drift_last_{group}"] = post_group_values[-1] if post_group_values else None
    return summary


def _scope_group_for_parameter(name: str) -> str:
    return _drift_group_for_key(name)


def apply_adapt_scope(model: nn.Module, adapt_scope: str, freeze_monthly_gain: bool = False) -> List[str]:
    """Restrict trainable target-specific parameters for few-shot adaptation."""
    if adapt_scope not in ADAPT_SCOPES:
        raise ValueError(f"unsupported adapt_scope={adapt_scope!r}; expected one of {list(ADAPT_SCOPES)}")
    allowed_by_scope = {
        "none": set(),
        "safe_operator": {"target_prompt", "adapter_coeff_bottleneck", "adapter_coeff_dec2", "adapter_coeff_dec1", "monthly_gain"},
        "prompt_coeff_gain": {"target_prompt", "adapter_coeff_bottleneck", "adapter_coeff_dec2", "adapter_coeff_dec1", "monthly_gain"},
        "prompt_only": {"target_prompt"},
        "coeff_only": {"adapter_coeff_bottleneck", "adapter_coeff_dec2", "adapter_coeff_dec1"},
        "gain_only": {"monthly_gain"},
        "coeff_gain": {"adapter_coeff_bottleneck", "adapter_coeff_dec2", "adapter_coeff_dec1", "monthly_gain"},
        "all": {
            "target_prompt",
            "adapter_coeff_bottleneck",
            "adapter_coeff_dec2",
            "adapter_coeff_dec1",
            "monthly_gain",
            "spatial_refine",
            "other_target_specific",
        },
    }[adapt_scope]
    if freeze_monthly_gain:
        allowed_by_scope = set(allowed_by_scope)
        allowed_by_scope.discard("monthly_gain")
    for name, param in model.named_parameters():
        if not (
            name.startswith("target_prompt.")
            or name.startswith("target_adapter_coefficient_residual_")
            or name.startswith("residual_gain.")
            or name.startswith("target_spatial_refine.")
        ):
            param.requires_grad_(False)
            continue
        param.requires_grad_(_scope_group_for_parameter(name) in allowed_by_scope)
    return [name for name, param in model.named_parameters() if param.requires_grad]


def group_target_parameter_counts(model: nn.Module) -> Dict[str, int]:
    """Return trainable target-specific parameter counts by protocol group."""
    groups = {
        "target_prompt": 0,
        "adapter_coeff_bottleneck": 0,
        "adapter_coeff_dec2": 0,
        "adapter_coeff_dec1": 0,
        "monthly_gain": 0,
        "spatial_refine": 0,
        "other_target_specific": 0,
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = _scope_group_for_parameter(name)
        groups[group] = groups.get(group, 0) + int(param.numel())
    groups["total"] = int(sum(groups.values()))
    return groups


def trainable_target_groups_for_names(parameter_names: Iterable[str]) -> List[str]:
    """Map trainable parameter names to paper-facing target posterior groups."""
    groups: List[str] = []
    seen: set[str] = set()
    for name in parameter_names:
        group = _scope_group_for_parameter(str(name))
        if group == "target_prompt":
            label = "target_prompt"
        elif group.startswith("adapter_coeff_"):
            label = "adapter_coefficient_residuals"
        elif group == "monthly_gain":
            label = "monthly_residual_gain"
        elif group == "spatial_refine":
            label = "target_spatial_refine"
        else:
            label = "other_target_specific"
        if label not in seen:
            seen.add(label)
            groups.append(label)
    return groups


def target_parameter_l2_drift(
    anchor_state: Dict[str, torch.Tensor],
    adapted_state: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Return L2 drift from source/prior initialization for target-only state."""
    if set(anchor_state) != set(adapted_state):
        missing = sorted(set(anchor_state) - set(adapted_state))
        extra = sorted(set(adapted_state) - set(anchor_state))
        raise ValueError(f"adapter state keys differ; missing={missing[:5]} extra={extra[:5]}")
    group_sq: Dict[str, float] = {}
    total_sq = 0.0
    for name, anchor_tensor in anchor_state.items():
        adapted_tensor = adapted_state[name]
        if tuple(anchor_tensor.shape) != tuple(adapted_tensor.shape):
            raise ValueError(
                f"shape mismatch for {name}: anchor={tuple(anchor_tensor.shape)} adapted={tuple(adapted_tensor.shape)}"
            )
        diff = adapted_tensor.detach().cpu().float() - anchor_tensor.detach().cpu().float()
        sq = float(diff.square().sum().item())
        total_sq += sq
        group = _drift_group_for_key(name)
        group_sq[group] = group_sq.get(group, 0.0) + sq
    drift = {group: float(value ** 0.5) for group, value in sorted(group_sq.items())}
    drift["total"] = float(total_sq ** 0.5)
    return drift


def build_stage3_prior_snapshot_metadata(
    *,
    source_config: Dict[str, Any],
    prompt_state: Dict[str, Any],
    source_checkpoint_sha256: str,
    target_region: str,
    K: int,
) -> Dict[str, Any]:
    """Return an audit record for the frozen HyperDA prior used in Stage 3."""
    prompt_metadata = target_context_prompt_metadata(prompt_state)
    monthly_counts = {
        str(month): int(prompt_metadata.get("monthly_counts", {}).get(str(month), 0))
        for month in range(1, 13)
    }
    return {
        "schema_version": "hyperda_stage3_prior_snapshot_v1",
        "prior_operator": "frozen_source_hyperda",
        "source_hyperda_trainable_in_stage3": False,
        "source_hyperda_parameter_updates": 0,
        "source_checkpoint_sha256": str(source_checkpoint_sha256 or ""),
        "source_stage_checkpoint_provenance": source_config.get(
            "source_stage_checkpoint_provenance",
            "phase4_hyperda_staged",
        ),
        "source_model_type": source_config.get("model_type", ""),
        "source_regions": list(source_config.get("source_regions", []) or []),
        "target_region": str(target_region),
        "K": int(K),
        "target_context_prompt_schema": prompt_metadata.get("schema_version", ""),
        "target_context_prompt_source": prompt_metadata.get("prompt_source", ""),
        "target_context_label_usage": prompt_metadata.get("label_usage", ""),
        "target_context_hash": prompt_metadata.get("context_hash", ""),
        "target_context_date_hash": prompt_metadata.get("context_date_hash", ""),
        "target_context_n_samples": int(prompt_metadata.get("n_samples", 0) or 0),
        "target_context_date_start": prompt_metadata.get("date_start", ""),
        "target_context_date_end": prompt_metadata.get("date_end", ""),
        "monthly_prompt_counts": monthly_counts,
        "target_val_usage": "unused_in_main_protocol",
        "target_eval_usage": "final_eval_only_no_selection",
    }


def build_stage3_posterior_state_metadata(
    *,
    anchor_state: Dict[str, torch.Tensor],
    final_state: Dict[str, torch.Tensor],
    K: int,
    adapt_scope: str,
    anchor_alpha: float,
    adaptation_steps: int,
    target_labels_loaded: bool,
    target_labels_used: bool,
    source_prior_hash_before: str = "",
    source_prior_hash_after: str = "",
    stage3_posterior_policy: str = "safe_operator_ablation",
    stage3_posterior_decision: str = "accepted",
    support_gate_status: str = "disabled",
    paper_selection_basis: str = "",
    stage3_acceptance_basis: str = "",
    source_policy_candidate_id: str = "",
) -> Dict[str, Any]:
    """Return an audit record for target-only posterior variables in Stage 3."""
    drift = target_parameter_l2_drift(anchor_state, final_state)
    target_updates = int(
        sum(
            1
            for name, anchor_tensor in anchor_state.items()
            if not torch.allclose(anchor_tensor.detach().cpu(), final_state[name].detach().cpu())
        )
    )
    no_update_contract = int(K) == 0
    source_prior_unchanged = (
        bool(source_prior_hash_before)
        and bool(source_prior_hash_after)
        and str(source_prior_hash_before) == str(source_prior_hash_after)
    )
    resolved_paper_selection_basis = str(paper_selection_basis) if paper_selection_basis else (
        "source_side_safe_policy_only" if int(K) > 0 else "zero_shot_no_target_labels"
    )
    resolved_acceptance_basis = str(stage3_acceptance_basis or resolved_paper_selection_basis)
    anchor_hash = hash_tensor_state_dict(anchor_state)
    final_hash = hash_tensor_state_dict(final_state)
    return {
        "schema_version": "hyperda_stage3_target_posterior_v1",
        "posterior_form": "target_context_prompt_plus_safe_target_operator_posterior",
        "stage3_posterior_policy": str(stage3_posterior_policy),
        "stage3_posterior_decision": str(stage3_posterior_decision),
        "support_gate_status": str(support_gate_status),
        "support_only_gate_status": str(support_gate_status),
        "stage3_no_update_contract": (
            "K0_fixed_no_update_source_prior_identity"
            if no_update_contract
            else "Kshot_source_policy_constrained_posterior_update"
        ),
        "paper_selection_basis": resolved_paper_selection_basis,
        "stage3_acceptance_basis": resolved_acceptance_basis,
        "source_policy_candidate_id": str(source_policy_candidate_id or ""),
        "support_gate_policy_role": (
            "target_support_only_diagnostic_not_paper_selection"
            if int(K) > 0
            else "not_applicable_k0_no_support"
        ),
        "posterior_variables": [
            "target_prompt",
            "adapter_coefficient_residuals",
            "monthly_residual_gain",
        ],
        "optional_posterior_variables": ["target_spatial_refine"],
        "paper_main_posterior_variables": [
            "target_prompt",
            "adapter_coefficient_residuals",
            "monthly_residual_gain",
        ],
        "source_hyperda_parameter_updates": 0,
        "source_backbone_parameter_updates": 0,
        "hypernetwork_parameter_updates": 0,
        "adapter_basis_bank_parameter_updates": 0,
        "target_specific_parameter_updates": target_updates,
        "adapt_scope": str(adapt_scope),
        "K": int(K),
        "adaptation_steps": int(adaptation_steps),
        "anchor_alpha": float(anchor_alpha),
        "safe_anchor_formula": "theta_SAFE = theta_prior + alpha_K * (theta_adapt - theta_prior)",
        "anchor_state_key_count": len(anchor_state),
        "final_state_key_count": len(final_state),
        "drift_from_prior": drift,
        "target_adapter_anchor_hash": anchor_hash,
        "target_adapter_state_hash": final_hash,
        "k0_anchor_state_hash": anchor_hash,
        "source_prior_hash_before": str(source_prior_hash_before or ""),
        "source_prior_hash_after": str(source_prior_hash_after or ""),
        "source_prior_unchanged": bool(source_prior_unchanged),
        "k0_target_drift_zero": bool(no_update_contract and float(drift.get("total", 0.0)) == 0.0),
        "target_labels_loaded_for_adaptation": bool(target_labels_loaded),
        "target_labels_used_for_adaptation": bool(target_labels_used),
        "target_val_usage": "unused_in_main_protocol",
        "target_eval_usage": "final_eval_only_no_selection",
    }


def build_stage3_target_posterior_state(
    *,
    anchor_state: Dict[str, torch.Tensor],
    final_state: Dict[str, torch.Tensor],
    K: int,
    adapt_scope: str,
    anchor_alpha: float,
    adaptation_steps: int,
    target_labels_loaded: bool,
    target_labels_used: bool,
    source_prior_hash_before: str,
    source_prior_hash_after: str,
    stage3_posterior_policy: str = "safe_operator_ablation",
    stage3_posterior_decision: str = "accepted",
    support_gate_status: str = "disabled",
    paper_selection_basis: str = "",
    stage3_acceptance_basis: str = "",
    source_policy_candidate_id: str = "",
) -> Dict[str, Any]:
    """Build the replayable target-only Stage 3 posterior state object."""
    if not anchor_state:
        raise ValueError("Stage 3 posterior requires a non-empty target adapter anchor state")
    if not final_state:
        raise ValueError("Stage 3 posterior requires a non-empty target adapter final state")
    if set(anchor_state) != set(final_state):
        missing = sorted(set(anchor_state) - set(final_state))
        extra = sorted(set(final_state) - set(anchor_state))
        raise ValueError(f"target posterior state keys differ; missing={missing[:5]} extra={extra[:5]}")
    bad_anchor = [name for name in anchor_state if not _is_target_adapter_state_key(name)]
    bad_final = [name for name in final_state if not _is_target_adapter_state_key(name)]
    if bad_anchor or bad_final:
        raise ValueError(
            "Stage 3 posterior state must contain target-only keys; "
            f"bad_anchor={bad_anchor[:5]} bad_final={bad_final[:5]}"
        )
    metadata = build_stage3_posterior_state_metadata(
        anchor_state=anchor_state,
        final_state=final_state,
        K=K,
        adapt_scope=adapt_scope,
        anchor_alpha=anchor_alpha,
        adaptation_steps=adaptation_steps,
        target_labels_loaded=target_labels_loaded,
        target_labels_used=target_labels_used,
        source_prior_hash_before=source_prior_hash_before,
        source_prior_hash_after=source_prior_hash_after,
        stage3_posterior_policy=stage3_posterior_policy,
        stage3_posterior_decision=stage3_posterior_decision,
        support_gate_status=support_gate_status,
        paper_selection_basis=paper_selection_basis,
        stage3_acceptance_basis=stage3_acceptance_basis,
        source_policy_candidate_id=source_policy_candidate_id,
    )
    return {
        "schema_version": "hyperda_stage3_target_posterior_state_v1",
        "target_adapter_state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in sorted(final_state.items())
        },
        "target_adapter_anchor_state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in sorted(anchor_state.items())
        },
        "target_adapter_state_hash": metadata["target_adapter_state_hash"],
        "target_adapter_anchor_hash": metadata["target_adapter_anchor_hash"],
        "source_prior_hash_before": str(source_prior_hash_before or ""),
        "source_prior_hash_after": str(source_prior_hash_after or ""),
        "source_prior_unchanged": bool(metadata["source_prior_unchanged"]),
        "drift_from_prior": dict(metadata["drift_from_prior"]),
        "metadata": metadata,
    }


def apply_source_anchor_interpolation(
    model: nn.Module,
    anchor_state: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """Apply fixed ``theta_init + alpha * (theta_adapt - theta_init)`` to target tensors only."""
    adapted_state = extract_target_adapter_state(model)
    interpolated_state = interpolate_target_adapter_state(anchor_state, adapted_state, float(alpha))
    apply_target_adapter_state(model, interpolated_state)
    return interpolated_state


def support_loss_summary(train_history: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    if not train_history:
        return {
            "support_loss_before": None,
            "support_loss_after": None,
            "support_final_loss": None,
            "support_loss_delta": None,
        }
    first = train_history[0]
    last = train_history[-1]
    first_loss = first.get("total_loss", first.get("objective"))
    final_loss = last.get("total_loss", last.get("objective"))
    if first_loss is None or final_loss is None:
        return {
            "support_loss_before": None,
            "support_loss_after": None,
            "support_final_loss": None,
            "support_loss_delta": None,
        }
    return {
        "support_loss_before": float(first_loss),
        "support_loss_after": float(final_loss),
        "support_final_loss": float(final_loss),
        "support_loss_delta": float(final_loss) - float(first_loss),
    }


def _trainable_gradient_vector(model: nn.Module) -> torch.Tensor:
    pieces: List[torch.Tensor] = []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            pieces.append(torch.zeros(param.numel(), dtype=torch.float32))
        else:
            pieces.append(param.grad.detach().flatten().float().cpu())
    if not pieces:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(pieces, dim=0)


def _losses_for_support_batch(
    state: FewShotAdaptationState,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    target_context_prompt_state: Dict[str, Any],
    loss_fn: nn.Module,
    normalize_increment: bool,
    lambda_prior: float,
    lambda_latent: float,
    lambda_gain: float,
    lambda_gain_smooth: float,
    lambda_analysis: float,
    support_loss_reduction: str,
) -> Dict[str, torch.Tensor]:
    return few_shot_batch_loss(
        state=state,
        batch=batch,
        device=device,
        target_context_prompt_state=target_context_prompt_state,
        loss_fn=loss_fn,
        normalize_increment=normalize_increment,
        lambda_prior=lambda_prior,
        lambda_latent=lambda_latent,
        lambda_gain=lambda_gain,
        lambda_gain_smooth=lambda_gain_smooth,
        lambda_analysis=lambda_analysis,
        support_loss_reduction=support_loss_reduction,
    )


def support_cycle_losses_from_loader(
    state: FewShotAdaptationState,
    loader: DataLoader,
    device: torch.device,
    target_context_prompt_state: Dict[str, Any],
    loss_fn: nn.Module,
    normalize_increment: bool,
    lambda_prior: float,
    lambda_latent: float,
    lambda_gain: float,
    lambda_gain_smooth: float,
    lambda_analysis: float,
    support_loss_reduction: str,
) -> List[float]:
    losses_out: List[float] = []
    was_training = state.model.training
    state.model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                losses = _losses_for_support_batch(
                    state=state,
                    batch=batch,
                    device=device,
                    target_context_prompt_state=target_context_prompt_state,
                    loss_fn=loss_fn,
                    normalize_increment=normalize_increment,
                    lambda_prior=lambda_prior,
                    lambda_latent=lambda_latent,
                    lambda_gain=lambda_gain,
                    lambda_gain_smooth=lambda_gain_smooth,
                    lambda_analysis=lambda_analysis,
                    support_loss_reduction=support_loss_reduction,
                )
                batch_size = int(batch["x"].shape[0])
                losses_out.extend([float(losses["total_loss"].detach().cpu())] * batch_size)
    finally:
        state.model.train(was_training)
    return losses_out


def compute_support_gradient_conflict_diagnostics(
    state: FewShotAdaptationState,
    loader: DataLoader,
    device: torch.device,
    target_context_prompt_state: Dict[str, Any],
    loss_fn: nn.Module,
    normalize_increment: bool,
    lambda_prior: float,
    lambda_latent: float,
    lambda_gain: float,
    lambda_gain_smooth: float,
    lambda_analysis: float,
    support_loss_reduction: str,
    after_cycle_losses: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compute target-support-only per-cycle gradient conflict diagnostics."""
    anchor_state = extract_target_adapter_state(state.model)
    was_training = state.model.training
    state.model.train()
    cycle_losses: List[float] = []
    grad_vectors: List[torch.Tensor] = []
    grad_norms: List[float] = []
    try:
        for batch in loader:
            apply_target_adapter_state(state.model, anchor_state)
            state.model.zero_grad(set_to_none=True)
            losses = _losses_for_support_batch(
                state=state,
                batch=batch,
                device=device,
                target_context_prompt_state=target_context_prompt_state,
                loss_fn=loss_fn,
                normalize_increment=normalize_increment,
                lambda_prior=lambda_prior,
                lambda_latent=lambda_latent,
                lambda_gain=lambda_gain,
                lambda_gain_smooth=lambda_gain_smooth,
                lambda_analysis=lambda_analysis,
                support_loss_reduction=support_loss_reduction,
            )
            losses["objective"].backward()
            grad = _trainable_gradient_vector(state.model)
            batch_size = int(batch["x"].shape[0])
            loss_value = float(losses["total_loss"].detach().cpu())
            for _ in range(batch_size):
                cycle_losses.append(loss_value)
                grad_vectors.append(grad.clone())
                grad_norms.append(float(torch.linalg.vector_norm(grad).item()))
    finally:
        apply_target_adapter_state(state.model, anchor_state)
        state.model.zero_grad(set_to_none=True)
        state.model.train(was_training)

    cosines: List[float] = []
    for i in range(len(grad_vectors)):
        gi = grad_vectors[i]
        ni = torch.linalg.vector_norm(gi)
        for j in range(i + 1, len(grad_vectors)):
            gj = grad_vectors[j]
            nj = torch.linalg.vector_norm(gj)
            denom = float((ni * nj).item())
            if denom <= 0.0:
                cosine = 0.0
            else:
                cosine = float(torch.dot(gi, gj).item() / denom)
            cosines.append(max(-1.0, min(1.0, cosine)))
    improvements: List[float] = []
    if after_cycle_losses is not None:
        for before, after in zip(cycle_losses, after_cycle_losses):
            improvements.append(float(before) - float(after))
    return {
        "support_gradient_diagnostics_label_source": "target_support_only",
        "support_cycle_count": len(cycle_losses),
        "support_cycle_loss_before": cycle_losses,
        "support_cycle_loss_after": list(after_cycle_losses or []),
        "support_cycle_gradient_norm": grad_norms,
        "support_gradient_pair_count": len(cosines),
        "support_gradient_cosine_mean": float(np.mean(cosines)) if cosines else None,
        "support_gradient_cosine_min": float(np.min(cosines)) if cosines else None,
        "support_gradient_negative_fraction": float(np.mean([c < 0.0 for c in cosines])) if cosines else None,
        "support_cycle_loss_improvement_mean": float(np.mean(improvements)) if improvements else None,
        "support_cycle_loss_improvement_std": float(np.std(improvements)) if improvements else None,
    }


def _date_str_records(dataset: Optional[HydroDADataset], date_key: str) -> List[str]:
    if dataset is None:
        return []
    records = getattr(dataset, "_split_entry", {}).get(date_key, [])
    if not isinstance(records, list):
        return []
    return [str(record.get("date_str", "")) for record in records if isinstance(record, dict) and record.get("date_str")]


def support_batch_count(support_count: int, batch_size: int) -> int:
    """Return number of support batches in one deterministic pass."""
    support_count = int(support_count or 0)
    batch_size = int(batch_size or 0)
    if support_count <= 0 or batch_size <= 0:
        return 0
    return int((support_count + batch_size - 1) // batch_size)


def effective_support_passes(optimizer_steps_run: int, support_batches: int) -> float:
    """Return optimizer steps expressed as passes over the support dataloader."""
    optimizer_steps_run = int(optimizer_steps_run or 0)
    support_batches = int(support_batches or 0)
    if optimizer_steps_run <= 0 or support_batches <= 0:
        return 0.0
    return float(optimizer_steps_run) / float(support_batches)


def support_dates_subset(left_dates: Iterable[str], right_dates: Iterable[str]) -> bool:
    """Whether every date in ``left_dates`` is present in ``right_dates``."""
    return set(str(date) for date in left_dates) <= set(str(date) for date in right_dates)


def save_few_shot_checkpoint(
    path: Path,
    state: FewShotAdaptationState,
    optimizer_state_dict: Dict[str, Any],
    config: Dict[str, Any],
    target_context_prompt_state: Dict[str, Any],
    train_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable_names = state.model.target_trainable_parameter_names()
    actual_trainable_target_groups = trainable_target_groups_for_names(trainable_names)
    prompt_state = normalize_target_context_prompt_state(target_context_prompt_state)
    prompt_metadata = target_context_prompt_metadata(prompt_state)
    full_config = dict(config)
    anchor_state = {
        name: tensor.detach().cpu()
        for name, tensor in full_config.get("target_adapter_anchor_state", {}).items()
    }
    final_state = extract_target_adapter_state(state.model)
    anchor_state_source = "config_target_adapter_anchor_state"
    if not anchor_state:
        anchor_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in final_state.items()
        }
        anchor_state_source = "final_state_fallback_missing_anchor"
    source_prior_hash_before = str(
        full_config.get("stage3_source_prior_hash_before")
        or full_config.get("source_prior_hash_before")
        or hash_source_prior_state(state.model)
    )
    source_prior_hash_after = str(
        full_config.get("stage3_source_prior_hash_after")
        or full_config.get("source_prior_hash_after")
        or hash_source_prior_state(state.model)
    )
    source_checkpoint_sha256 = str(full_config.get("source_checkpoint_sha256", "") or "")
    config_context_hash = str(full_config.get("target_context_dates_hash", "") or "")
    prompt_context_hash = str(prompt_metadata.get("context_hash") or prompt_metadata.get("context_date_hash") or "")
    if config_context_hash and prompt_context_hash and config_context_hash != prompt_context_hash:
        raise ValueError(
            "target_context_dates_hash mismatch between split manifest and "
            f"target_context_prompt_state: {config_context_hash!r} != {prompt_context_hash!r}"
        )
    paper_facing_run = bool(
        full_config.get(
            "paper_facing_run",
            int(full_config.get("K", 0) or 0) == 0
            or full_config.get("policy_source") == "source_side_episode_calibration",
        )
    )
    if int(full_config.get("K", 0) or 0) > 0 and not paper_facing_run:
        full_config.setdefault("diagnostic_run_reason", "missing_source_side_safe_policy_json")
    stage3_acceptance_basis = str(
        full_config.get(
            "stage3_acceptance_basis",
            full_config.get(
                "paper_selection_basis",
                "zero_shot_no_target_labels"
                if int(full_config.get("K", 0) or 0) == 0
                else (
                    "source_side_safe_policy_only"
                    if paper_facing_run
                    else "diagnostic_no_source_safe_policy_json"
                ),
            ),
        )
    )
    if "stage3_prior_snapshot" not in full_config:
        full_config["stage3_prior_snapshot"] = build_stage3_prior_snapshot_metadata(
            source_config=state.source_config,
            prompt_state=prompt_state,
            source_checkpoint_sha256=source_checkpoint_sha256,
            target_region=str(full_config.get("target_region", "")),
            K=int(full_config.get("K", 0)),
        )
    if "stage3_posterior_state" not in full_config:
        full_config["stage3_posterior_state"] = build_stage3_posterior_state_metadata(
            anchor_state=anchor_state,
            final_state=final_state,
            K=int(full_config.get("K", 0)),
            adapt_scope=str(full_config.get("adapt_scope", "safe_operator")),
            anchor_alpha=float(full_config.get("anchor_alpha", default_anchor_alpha_for_K(int(full_config.get("K", 0))))),
            adaptation_steps=int(full_config.get("adaptation_steps", 0) or 0),
            target_labels_loaded=bool(full_config.get("target_labels_loaded_for_adaptation", False)),
            target_labels_used=bool(full_config.get("target_labels_used_for_adaptation", False)),
            source_prior_hash_before=source_prior_hash_before,
            source_prior_hash_after=source_prior_hash_after,
            stage3_posterior_policy=str(full_config.get("stage3_posterior_policy", "safe_operator_ablation")),
            stage3_posterior_decision=str(full_config.get("stage3_posterior_decision", "accepted")),
            support_gate_status=str(full_config.get("support_gate_status", "disabled")),
            paper_selection_basis=str(full_config.get("paper_selection_basis", "")),
            stage3_acceptance_basis=stage3_acceptance_basis,
            source_policy_candidate_id=str(full_config.get("source_policy_candidate_id", "")),
        )
    stage3_posterior_state_dict = build_stage3_target_posterior_state(
        anchor_state=anchor_state,
        final_state=final_state,
        K=int(full_config.get("K", 0)),
        adapt_scope=str(full_config.get("adapt_scope", "safe_operator")),
        anchor_alpha=float(full_config.get("anchor_alpha", default_anchor_alpha_for_K(int(full_config.get("K", 0))))),
        adaptation_steps=int(full_config.get("adaptation_steps", 0) or 0),
        target_labels_loaded=bool(full_config.get("target_labels_loaded_for_adaptation", False)),
        target_labels_used=bool(full_config.get("target_labels_used_for_adaptation", False)),
        source_prior_hash_before=source_prior_hash_before,
        source_prior_hash_after=source_prior_hash_after,
        stage3_posterior_policy=str(full_config.get("stage3_posterior_policy", "safe_operator_ablation")),
        stage3_posterior_decision=str(full_config.get("stage3_posterior_decision", "accepted")),
        support_gate_status=str(full_config.get("support_gate_status", "disabled")),
        paper_selection_basis=str(full_config.get("paper_selection_basis", "")),
        stage3_acceptance_basis=stage3_acceptance_basis,
        source_policy_candidate_id=str(full_config.get("source_policy_candidate_id", "")),
    )
    full_config["stage3_posterior_state"] = dict(stage3_posterior_state_dict["metadata"])
    full_config["stage3_posterior_state"]["anchor_state_source"] = anchor_state_source
    full_config["stage3_posterior_state"].update(
        {
            key: full_config.get(key)
            for key in (
                "support_gate_enabled",
                "support_gate_label_source",
                "support_gate_min_delta",
                "support_gate_rootzone_tolerance",
                "support_gate_reject_reason",
                "support_objective_before",
                "support_objective_after",
                "support_objective_delta",
                "support_surface_loss_before",
                "support_surface_loss_after",
                "support_surface_loss_delta",
                "support_rootzone_loss_before",
                "support_rootzone_loss_after",
                "support_rootzone_loss_delta",
            )
            if key in full_config
        }
    )
    full_config["stage3_source_prior_hash_before"] = source_prior_hash_before
    full_config["stage3_source_prior_hash_after"] = source_prior_hash_after
    full_config["stage3_source_prior_unchanged"] = bool(stage3_posterior_state_dict["source_prior_unchanged"])
    method_id = method_id_for_run(
        str(full_config.get("adaptation_setting", "")),
        int(full_config.get("K", 0)),
        paper_facing_run=paper_facing_run,
    )
    support_loss_before_alias = full_config.get("support_loss_before")
    if support_loss_before_alias is None:
        support_loss_before_alias = full_config.get("standard_support_loss_before_full_support")
    if support_loss_before_alias is None:
        support_loss_before_alias = full_config.get("ridge_design_loss_before_sampled_pixels")
    support_loss_after_alias = full_config.get("support_loss_after")
    if support_loss_after_alias is None:
        support_loss_after_alias = full_config.get("standard_support_loss_after_full_support")
    if support_loss_after_alias is None:
        support_loss_after_alias = full_config.get("ridge_design_loss_after_sampled_pixels")
    support_loss_delta_alias = full_config.get("support_loss_delta")
    if support_loss_delta_alias is None:
        support_loss_delta_alias = full_config.get("standard_support_loss_delta_full_support")
    if support_loss_delta_alias is None:
        support_loss_delta_alias = full_config.get("ridge_design_loss_delta_sampled_pixels")
    support_final_loss_alias = full_config.get("support_final_loss")
    if support_final_loss_alias is None:
        support_final_loss_alias = support_loss_after_alias
    full_config.update(
        {
            "method": method_id,
            "model_type": "hyperda_basis_adapter_target_adapt",
            "protocol_freeze_id": PROTOCOL_FREEZE_ID,
            "target_context_period": "2015-2021",
            "target_support_period": "2015-2021",
            "target_val_period": "unused_in_main_protocol",
            "target_eval_period": "2023-2025",
            "frozen_modules": ["source_backbone", "prompt_encoder", "hypernetwork", "adapter_basis_bank"],
            "frozen_source_groups": list(FROZEN_SOURCE_GROUPS),
            "trainable_modules": list(
                full_config.get("trainable_target_groups", actual_trainable_target_groups)
            ),
            "trainable_target_groups": list(
                full_config.get("trainable_target_groups", actual_trainable_target_groups)
            ),
            "trainable_parameter_names": trainable_names,
            "trainable_parameter_count": int(sum(p.numel() for p in state.model.parameters() if p.requires_grad)),
            "requires_grad_parameter_count": int(
                full_config.get(
                    "requires_grad_parameter_count",
                    sum(p.numel() for p in state.model.parameters() if p.requires_grad),
                )
            ),
            "requires_grad_param_count": int(
                full_config.get(
                    "requires_grad_param_count",
                    full_config.get(
                        "requires_grad_parameter_count",
                        sum(p.numel() for p in state.model.parameters() if p.requires_grad),
                    ),
                )
            ),
            "optimizer_parameter_count": int(full_config.get("optimizer_parameter_count", 0) or 0),
            "optimizer_param_count": int(
                full_config.get(
                    "optimizer_param_count",
                    full_config.get("optimizer_parameter_count", 0) or 0,
                )
            ),
            "target_parameter_count_by_group": dict(
                full_config.get("target_parameter_count_by_group", group_target_parameter_counts(state.model))
            ),
            "adapt_scope": full_config.get("adapt_scope", "safe_operator"),
            "stage3_posterior_policy": full_config.get("stage3_posterior_policy", "safe_operator_ablation"),
            "stage3_posterior_decision": full_config.get("stage3_posterior_decision", "accepted"),
            "stage3_no_update_contract": full_config.get("stage3_no_update_contract", ""),
            "paper_selection_basis": full_config.get("paper_selection_basis", ""),
            "stage3_acceptance_basis": stage3_acceptance_basis,
            "k0_anchor_state_hash": stage3_posterior_state_dict["target_adapter_anchor_hash"],
            "paper_facing_run": paper_facing_run,
            "diagnostic_run_reason": full_config.get("diagnostic_run_reason", ""),
            "support_gate_enabled": bool(full_config.get("support_gate_enabled", False)),
            "support_gate_status": full_config.get("support_gate_status", "disabled"),
            "support_only_gate_status": full_config.get(
                "support_only_gate_status",
                full_config.get("support_gate_status", "disabled"),
            ),
            "support_gate_label_source": full_config.get("support_gate_label_source", ""),
            "support_gate_policy_role": full_config.get(
                "support_gate_policy_role",
                "target_support_only_diagnostic_not_paper_selection"
                if int(full_config.get("K", 0) or 0) > 0
                else "not_applicable_k0_no_support",
            ),
            "support_gate_min_delta": full_config.get("support_gate_min_delta", 0.0),
            "support_gate_rootzone_tolerance": full_config.get("support_gate_rootzone_tolerance", 0.0),
            "support_gate_reject_reason": list(full_config.get("support_gate_reject_reason", []) or []),
            "support_objective_before": full_config.get("support_objective_before"),
            "support_objective_after": full_config.get("support_objective_after"),
            "support_objective_delta": full_config.get("support_objective_delta"),
            "support_surface_loss_before": full_config.get("support_surface_loss_before"),
            "support_surface_loss_after": full_config.get("support_surface_loss_after"),
            "support_surface_loss_delta": full_config.get("support_surface_loss_delta"),
            "support_rootzone_loss_before": full_config.get("support_rootzone_loss_before"),
            "support_rootzone_loss_after": full_config.get("support_rootzone_loss_after"),
            "support_rootzone_loss_delta": full_config.get("support_rootzone_loss_delta"),
            "freeze_monthly_gain": bool(full_config.get("freeze_monthly_gain", False)),
            "adapt_solver": full_config.get("adapt_solver", "adamw"),
            "trust_region_mode": full_config.get("trust_region_mode", "none"),
            "trust_total_radius": full_config.get("trust_total_radius", 0.0),
            "trust_prompt_radius": full_config.get("trust_prompt_radius", 0.0),
            "trust_gain_radius": full_config.get("trust_gain_radius", 0.0),
            "trust_coeff_radius": full_config.get("trust_coeff_radius", 0.0),
            "trust_spatial_radius": full_config.get("trust_spatial_radius", 0.0),
            "trust_projection_diagnostics": dict(full_config.get("trust_projection_diagnostics", {}) or {}),
            "support_loss_reduction": full_config.get("support_loss_reduction", "global_pixel"),
            "support_gradient_diagnostics": dict(full_config.get("support_gradient_diagnostics", {}) or {}),
            "support_gradient_cosine_mean": (full_config.get("support_gradient_diagnostics", {}) or {}).get(
                "support_gradient_cosine_mean"
            ),
            "support_gradient_cosine_min": (full_config.get("support_gradient_diagnostics", {}) or {}).get(
                "support_gradient_cosine_min"
            ),
            "support_gradient_negative_fraction": (full_config.get("support_gradient_diagnostics", {}) or {}).get(
                "support_gradient_negative_fraction"
            ),
            "support_cycle_loss_improvement_mean": (full_config.get("support_gradient_diagnostics", {}) or {}).get(
                "support_cycle_loss_improvement_mean"
            ),
            "support_cycle_loss_improvement_std": (full_config.get("support_gradient_diagnostics", {}) or {}).get(
                "support_cycle_loss_improvement_std"
            ),
            "ridge_lambda": full_config.get("ridge_lambda", None),
            "ridge_clip_coeff_norm": full_config.get("ridge_clip_coeff_norm", None),
            "ridge_trust_region_radius": full_config.get("ridge_trust_region_radius", None),
            "ridge_max_feature_pixels": full_config.get("ridge_max_feature_pixels", None),
            "ridge_standardize_features": bool(full_config.get("ridge_standardize_features", False)),
            "ridge_diagnostics": dict(full_config.get("ridge_diagnostics", {}) or {}),
            "ridge_status": (full_config.get("ridge_diagnostics", {}) or {}).get("status"),
            "ridge_coefficient_norm": (full_config.get("ridge_diagnostics", {}) or {}).get("coefficient_norm"),
            "ridge_delta_norm": (full_config.get("ridge_diagnostics", {}) or {}).get("delta_norm"),
            "ridge_raw_delta_norm": (full_config.get("ridge_diagnostics", {}) or {}).get("raw_delta_norm"),
            "ridge_coeff_norm": (full_config.get("ridge_diagnostics", {}) or {}).get("coefficient_norm"),
            "ridge_coeff_delta_norm": (full_config.get("ridge_diagnostics", {}) or {}).get("delta_norm"),
            "ridge_clip_applied": bool((full_config.get("ridge_diagnostics", {}) or {}).get("coefficient_norm_clipped", False)),
            "ridge_trust_region_clipped": bool((full_config.get("ridge_diagnostics", {}) or {}).get("trust_region_clipped", False)),
            "ridge_support_count": (full_config.get("ridge_diagnostics", {}) or {}).get("support_count"),
            "ridge_masked_pixel_count": (full_config.get("ridge_diagnostics", {}) or {}).get("masked_pixel_count"),
            "ridge_masked_observation_count": (full_config.get("ridge_diagnostics", {}) or {}).get("masked_observation_count"),
            "ridge_feature_pixel_count": (full_config.get("ridge_diagnostics", {}) or {}).get("feature_pixel_count"),
            "ridge_feature_observation_count": (full_config.get("ridge_diagnostics", {}) or {}).get("feature_observation_count"),
            "ridge_feature_dim": (full_config.get("ridge_diagnostics", {}) or {}).get("feature_dim"),
            "ridge_condition_number": (full_config.get("ridge_diagnostics", {}) or {}).get("condition_number"),
            "ridge_rank": (full_config.get("ridge_diagnostics", {}) or {}).get("rank"),
            "audit_identity": bool(full_config.get("audit_identity", False)),
            "audit_identity_tolerance": float(full_config.get("audit_identity_tolerance", 1e-8)),
            "source_checkpoint_sha256": full_config.get("source_checkpoint_sha256", ""),
            "staged_source_checkpoint_sha256": full_config.get(
                "staged_source_checkpoint_sha256",
                full_config.get("source_checkpoint_sha256", ""),
            ),
            "source_stage_checkpoint_provenance": full_config.get(
                "source_stage_checkpoint_provenance",
                "phase4_hyperda_staged",
            ),
            "stage3_source_prior_hash_before": source_prior_hash_before,
            "stage3_source_prior_hash_after": source_prior_hash_after,
            "stage3_source_prior_unchanged": bool(stage3_posterior_state_dict["source_prior_unchanged"]),
            "k0_target_drift_zero": bool(
                int(full_config.get("K", 0) or 0) == 0
                and float((full_config.get("target_parameter_l2_drift", {}) or {}).get("total", 0.0) or 0.0) == 0.0
            ),
            "target_support_count": int(full_config.get("target_support_count", 0) or 0),
            "target_labels_loaded_for_adaptation": bool(full_config.get("target_labels_loaded_for_adaptation", False)),
            "target_labels_used_for_adaptation": bool(full_config.get("target_labels_used_for_adaptation", False)),
            "adapt_batch_size": int(full_config.get("adapt_batch_size", full_config.get("batch_size", 0)) or 0),
            "max_steps_requested": int(
                full_config.get("max_steps_requested", full_config.get("adaptation_steps", 0)) or 0
            ),
            "actual_optimizer_steps": int(
                full_config.get(
                    "actual_optimizer_steps",
                    full_config.get("optimizer_steps_run", len(train_history)),
                )
                or 0
            ),
            "optimizer_steps_run": int(full_config.get("optimizer_steps_run", len(train_history)) or 0),
            "support_batch_count": int(full_config.get("support_batch_count", 0) or 0),
            "effective_support_passes": float(full_config.get("effective_support_passes", 0.0) or 0.0),
            "adapt_recipe": full_config.get("adapt_recipe", "source_anchor"),
            "policy_source": full_config.get("policy_source", "preregistered_default"),
            "safe_policy_json": full_config.get("safe_policy_json", ""),
            "safe_policy_json_sha256": full_config.get("safe_policy_json_sha256", ""),
            "safe_policy": dict(full_config.get("safe_policy", {}) or {}),
            "safe_policy_hash": full_config.get(
                "safe_policy_hash",
                (full_config.get("safe_policy", {}) or {}).get("policy_hash", ""),
            ),
            "source_policy_candidate_id": full_config.get("source_policy_candidate_id", ""),
            "source_policy_guard_config_hash": full_config.get("source_policy_guard_config_hash", ""),
            "source_episode_regions": list(full_config.get("source_episode_regions", []) or []),
            "rho_policy": full_config.get("rho_policy", "fixed_1.0"),
            "adapt_mix_rho": float(full_config.get("adapt_mix_rho", 1.0 if int(full_config.get("K", 0) or 0) == 0 else 0.0)),
            "anchor_alpha": float(full_config.get("anchor_alpha", default_anchor_alpha_for_K(int(full_config.get("K", 0))))),
            "schedule_label": full_config.get("schedule_label", ""),
            "requested_lr": full_config.get("requested_lr", full_config.get("lr", None)),
            "requested_max_steps": int(
                full_config.get("requested_max_steps", full_config.get("max_steps_requested", full_config.get("adaptation_steps", 0))) or 0
            ),
            "requested_anchor_alpha": float(
                full_config.get(
                    "requested_anchor_alpha",
                    full_config.get("anchor_alpha", default_anchor_alpha_for_K(int(full_config.get("K", 0)))),
                )
            ),
            "requested_weight_decay": full_config.get("requested_weight_decay", full_config.get("weight_decay", None)),
            "requested_grad_clip": full_config.get("requested_grad_clip", full_config.get("grad_clip", None)),
            "anchor_alpha_grid_preregistered": full_config.get(
                "anchor_alpha_grid_preregistered",
                default_anchor_alpha_grid_for_K(int(full_config.get("K", 0))),
            ),
            "source_anchor_hyperparameter_source": full_config.get(
                "source_anchor_hyperparameter_source",
                "source_side_episodic_validation_preregistered",
            ),
            "model_selection_source": "source_val_preregistered",
            "target_val_usage": "unused_in_main_protocol",
            "checkpoint_selection": "fixed_preregistered_final_step",
            "target_eval_usage": full_config.get("target_eval_usage", "final_eval_only_no_selection"),
            "support_manifest_hash": full_config.get("support_manifest_hash", ""),
            "support_nesting_hash": full_config.get("support_nesting_hash", ""),
            "support_nesting_status": full_config.get("support_nesting_status", ""),
            "standard_support_loss_before_full_support": full_config.get("standard_support_loss_before_full_support"),
            "standard_support_loss_after_full_support": full_config.get("standard_support_loss_after_full_support"),
            "standard_support_loss_delta_full_support": full_config.get("standard_support_loss_delta_full_support"),
            "standard_support_surface_loss_before_full_support": full_config.get("standard_support_surface_loss_before_full_support"),
            "standard_support_surface_loss_after_full_support": full_config.get("standard_support_surface_loss_after_full_support"),
            "standard_support_rootzone_loss_before_full_support": full_config.get("standard_support_rootzone_loss_before_full_support"),
            "standard_support_rootzone_loss_after_full_support": full_config.get("standard_support_rootzone_loss_after_full_support"),
            "standard_support_objective_before_full_support": full_config.get("standard_support_objective_before_full_support"),
            "standard_support_objective_after_full_support": full_config.get("standard_support_objective_after_full_support"),
            "standard_support_increment_loss_before_full_support": full_config.get("standard_support_increment_loss_before_full_support"),
            "standard_support_increment_loss_after_full_support": full_config.get("standard_support_increment_loss_after_full_support"),
            "standard_support_analysis_loss_before_full_support": full_config.get("standard_support_analysis_loss_before_full_support"),
            "standard_support_analysis_loss_after_full_support": full_config.get("standard_support_analysis_loss_after_full_support"),
            "standard_support_analysis_surface_loss_before_full_support": full_config.get("standard_support_analysis_surface_loss_before_full_support"),
            "standard_support_analysis_surface_loss_after_full_support": full_config.get("standard_support_analysis_surface_loss_after_full_support"),
            "standard_support_analysis_rootzone_loss_before_full_support": full_config.get("standard_support_analysis_rootzone_loss_before_full_support"),
            "standard_support_analysis_rootzone_loss_after_full_support": full_config.get("standard_support_analysis_rootzone_loss_after_full_support"),
            "standard_support_regularization_loss_before_full_support": full_config.get("standard_support_regularization_loss_before_full_support"),
            "standard_support_regularization_loss_after_full_support": full_config.get("standard_support_regularization_loss_after_full_support"),
            "ridge_design_loss_before_sampled_pixels": full_config.get("ridge_design_loss_before_sampled_pixels"),
            "ridge_design_loss_after_sampled_pixels": full_config.get("ridge_design_loss_after_sampled_pixels"),
            "ridge_design_loss_delta_sampled_pixels": full_config.get("ridge_design_loss_delta_sampled_pixels"),
            "support_loss_before": support_loss_before_alias,
            "support_loss_after": support_loss_after_alias,
            "support_final_loss": support_final_loss_alias,
            "support_loss_delta": support_loss_delta_alias,
            "target_parameter_l2_drift_pre_anchor": dict(full_config.get("target_parameter_l2_drift_pre_anchor", {})),
            "target_parameter_l2_drift_post_anchor": dict(
                full_config.get("target_parameter_l2_drift_post_anchor", full_config.get("target_parameter_l2_drift", {}))
            ),
            "target_context_prompt_state": prompt_state,
            "target_context_prompt_state_summary": prompt_metadata,
            "stage3_prior_snapshot": dict(full_config.get("stage3_prior_snapshot", {})),
            "stage3_posterior_state": dict(full_config.get("stage3_posterior_state", {})),
            "prompt_policy": prompt_metadata["prompt_source"],
            "prompt_label_usage": prompt_metadata["label_usage"],
            "eval_input_usage": prompt_metadata["eval_input_usage"],
            "eval_month_usage": prompt_metadata["eval_month_usage"],
            "normalization_source": "source_fit_only_from_source_checkpoint",
            "leakage_guard_status": "pass",
            "git_hash": get_git_hash(),
            "timestamp": get_timestamp(),
        }
    )
    checkpoint = {
        "tag": "final_preregistered",
        "epoch": 0,
        "protocol_freeze_id": PROTOCOL_FREEZE_ID,
        "model_state_dict": state.model.state_dict(),
        "prompt_encoder_state_dict": state.prompt_encoder.state_dict(),
        "target_context_prompt_state": prompt_state,
        "target_adapter_anchor_state": {
            name: tensor.detach().cpu()
            for name, tensor in anchor_state.items()
        },
        "stage3_posterior_state_dict": stage3_posterior_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "source_checkpoint_config": state.source_config,
        "train_history": train_history,
        "config": full_config,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
        },
    }
    torch.save(checkpoint, path)
    return full_config


def write_run_metadata_sidecar(output_dir: Path, checkpoint_path: Path, config: Dict[str, Any]) -> None:
    """Write JSON-safe run metadata mirroring required checkpoint metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_state = config.get("target_context_prompt_state")
    prompt_state_summary: Dict[str, Any] = {}
    if isinstance(prompt_state, dict):
        try:
            prompt_state_summary = target_context_prompt_metadata(prompt_state)
        except Exception:
            prompt_state_summary = {
                "schema_version": prompt_state.get("schema_version", ""),
                "prompt_source": prompt_state.get("prompt_source", ""),
                "label_usage": prompt_state.get("label_usage", ""),
                "context_hash": prompt_state.get("context_hash", ""),
                "context_date_hash": prompt_state.get("context_date_hash", prompt_state.get("context_hash", "")),
            }
    metadata = {
        "checkpoint": str(checkpoint_path),
        "protocol_freeze_id": PROTOCOL_FREEZE_ID,
        "method": config.get("method", ""),
        "adaptation_setting": config.get("adaptation_setting", ""),
        "K": config.get("K", None),
        "seed": config.get("seed", None),
        "target_region": config.get("target_region", ""),
        "split_manifest_path": config.get("split_manifest_path", ""),
        "split_manifest_sha256": config.get("split_manifest_sha256", ""),
        "source_checkpoint": config.get("source_checkpoint", ""),
        "source_checkpoint_sha256": config.get("source_checkpoint_sha256", ""),
        "staged_source_checkpoint_sha256": config.get(
            "staged_source_checkpoint_sha256",
            config.get("source_checkpoint_sha256", ""),
        ),
        "source_stage_checkpoint_provenance": config.get(
            "source_stage_checkpoint_provenance",
            "phase4_hyperda_staged",
        ),
        "target_context_dates_hash": config.get("target_context_dates_hash", ""),
        "target_support_dates_hash": config.get("target_support_dates_hash", ""),
        "target_support_dates": list(config.get("target_support_dates", [])),
        "support_manifest_hash": config.get("support_manifest_hash", ""),
        "support_nesting_hash": config.get("support_nesting_hash", ""),
        "support_nesting_status": config.get("support_nesting_status", ""),
        "target_support_count": config.get("target_support_count", 0),
        "target_eval_dates_hash": config.get("target_eval_dates_hash", ""),
        "target_context_prompt_state": prompt_state_summary,
        "stage3_prior_snapshot": dict(config.get("stage3_prior_snapshot", {}) or {}),
        "stage3_posterior_state": dict(config.get("stage3_posterior_state", {}) or {}),
        "stage3_source_prior_hash_before": config.get("stage3_source_prior_hash_before", ""),
        "stage3_source_prior_hash_after": config.get("stage3_source_prior_hash_after", ""),
        "stage3_source_prior_unchanged": bool(config.get("stage3_source_prior_unchanged", False)),
        "k0_target_drift_zero": bool(config.get("k0_target_drift_zero", False)),
        "adapt_scope": config.get("adapt_scope", "safe_operator"),
        "stage3_posterior_policy": config.get("stage3_posterior_policy", "safe_operator_ablation"),
        "stage3_posterior_decision": config.get("stage3_posterior_decision", "accepted"),
        "stage3_no_update_contract": config.get("stage3_no_update_contract", ""),
        "paper_selection_basis": config.get("paper_selection_basis", ""),
        "stage3_acceptance_basis": config.get("stage3_acceptance_basis", config.get("paper_selection_basis", "")),
        "k0_anchor_state_hash": config.get("k0_anchor_state_hash", ""),
        "paper_facing_run": bool(config.get("paper_facing_run", False)),
        "diagnostic_run_reason": config.get("diagnostic_run_reason", ""),
        "support_gate_enabled": bool(config.get("support_gate_enabled", False)),
        "support_gate_status": config.get("support_gate_status", "disabled"),
        "support_only_gate_status": config.get(
            "support_only_gate_status",
            config.get("support_gate_status", "disabled"),
        ),
        "support_gate_label_source": config.get("support_gate_label_source", ""),
        "support_gate_policy_role": config.get(
            "support_gate_policy_role",
            "target_support_only_diagnostic_not_paper_selection"
            if int(config.get("K", 0) or 0) > 0
            else "not_applicable_k0_no_support",
        ),
        "support_gate_min_delta": config.get("support_gate_min_delta", 0.0),
        "support_gate_rootzone_tolerance": config.get("support_gate_rootzone_tolerance", 0.0),
        "support_gate_reject_reason": list(config.get("support_gate_reject_reason", []) or []),
        "support_objective_before": config.get("support_objective_before", None),
        "support_objective_after": config.get("support_objective_after", None),
        "support_objective_delta": config.get("support_objective_delta", None),
        "support_surface_loss_before": config.get("support_surface_loss_before", None),
        "support_surface_loss_after": config.get("support_surface_loss_after", None),
        "support_surface_loss_delta": config.get("support_surface_loss_delta", None),
        "support_rootzone_loss_before": config.get("support_rootzone_loss_before", None),
        "support_rootzone_loss_after": config.get("support_rootzone_loss_after", None),
        "support_rootzone_loss_delta": config.get("support_rootzone_loss_delta", None),
        "freeze_monthly_gain": bool(config.get("freeze_monthly_gain", False)),
        "adapt_solver": config.get("adapt_solver", "adamw"),
        "trust_region_mode": config.get("trust_region_mode", "none"),
        "trust_total_radius": config.get("trust_total_radius", 0.0),
        "trust_prompt_radius": config.get("trust_prompt_radius", 0.0),
        "trust_gain_radius": config.get("trust_gain_radius", 0.0),
        "trust_coeff_radius": config.get("trust_coeff_radius", 0.0),
        "trust_spatial_radius": config.get("trust_spatial_radius", 0.0),
        "trust_projection_diagnostics": dict(config.get("trust_projection_diagnostics", {}) or {}),
        "support_loss_reduction": config.get("support_loss_reduction", "global_pixel"),
        "support_gradient_diagnostics": dict(config.get("support_gradient_diagnostics", {}) or {}),
        "support_gradient_cosine_mean": (config.get("support_gradient_diagnostics", {}) or {}).get(
            "support_gradient_cosine_mean"
        ),
        "support_gradient_cosine_min": (config.get("support_gradient_diagnostics", {}) or {}).get(
            "support_gradient_cosine_min"
        ),
        "support_gradient_negative_fraction": (config.get("support_gradient_diagnostics", {}) or {}).get(
            "support_gradient_negative_fraction"
        ),
        "support_cycle_loss_improvement_mean": (config.get("support_gradient_diagnostics", {}) or {}).get(
            "support_cycle_loss_improvement_mean"
        ),
        "support_cycle_loss_improvement_std": (config.get("support_gradient_diagnostics", {}) or {}).get(
            "support_cycle_loss_improvement_std"
        ),
        "ridge_lambda": config.get("ridge_lambda", None),
        "ridge_clip_coeff_norm": config.get("ridge_clip_coeff_norm", None),
        "ridge_trust_region_radius": config.get("ridge_trust_region_radius", None),
        "ridge_max_feature_pixels": config.get("ridge_max_feature_pixels", None),
        "ridge_standardize_features": bool(config.get("ridge_standardize_features", False)),
        "ridge_diagnostics": dict(config.get("ridge_diagnostics", {}) or {}),
        "ridge_status": (config.get("ridge_diagnostics", {}) or {}).get("status"),
        "ridge_coefficient_norm": (config.get("ridge_diagnostics", {}) or {}).get("coefficient_norm"),
        "ridge_delta_norm": (config.get("ridge_diagnostics", {}) or {}).get("delta_norm"),
        "ridge_raw_delta_norm": (config.get("ridge_diagnostics", {}) or {}).get("raw_delta_norm"),
        "ridge_coeff_norm": (config.get("ridge_diagnostics", {}) or {}).get("coefficient_norm"),
        "ridge_coeff_delta_norm": (config.get("ridge_diagnostics", {}) or {}).get("delta_norm"),
        "ridge_clip_applied": bool((config.get("ridge_diagnostics", {}) or {}).get("coefficient_norm_clipped", False)),
        "ridge_trust_region_clipped": bool((config.get("ridge_diagnostics", {}) or {}).get("trust_region_clipped", False)),
        "ridge_support_count": (config.get("ridge_diagnostics", {}) or {}).get("support_count"),
        "ridge_masked_pixel_count": (config.get("ridge_diagnostics", {}) or {}).get("masked_pixel_count"),
        "ridge_masked_observation_count": (config.get("ridge_diagnostics", {}) or {}).get("masked_observation_count"),
        "ridge_feature_pixel_count": (config.get("ridge_diagnostics", {}) or {}).get("feature_pixel_count"),
        "ridge_feature_observation_count": (config.get("ridge_diagnostics", {}) or {}).get("feature_observation_count"),
        "ridge_feature_dim": (config.get("ridge_diagnostics", {}) or {}).get("feature_dim"),
        "ridge_condition_number": (config.get("ridge_diagnostics", {}) or {}).get("condition_number"),
        "ridge_rank": (config.get("ridge_diagnostics", {}) or {}).get("rank"),
        "audit_identity": bool(config.get("audit_identity", False)),
        "audit_identity_tolerance": config.get("audit_identity_tolerance", 1e-8),
        "trainable_parameter_count": config.get("trainable_parameter_count", 0),
        "requires_grad_parameter_count": config.get("requires_grad_parameter_count", config.get("trainable_parameter_count", 0)),
        "requires_grad_param_count": config.get("requires_grad_param_count", config.get("requires_grad_parameter_count", config.get("trainable_parameter_count", 0))),
        "optimizer_parameter_count": config.get("optimizer_parameter_count", 0),
        "optimizer_param_count": config.get("optimizer_param_count", config.get("optimizer_parameter_count", 0)),
        "trainable_parameter_names": list(config.get("trainable_parameter_names", [])),
        "target_parameter_count_by_group": dict(config.get("target_parameter_count_by_group", {})),
        "adaptation_steps": config.get("adaptation_steps", 0),
        "lr": config.get("lr", None),
        "weight_decay": config.get("weight_decay", None),
        "grad_clip": config.get("grad_clip", None),
        "schedule_label": config.get("schedule_label", ""),
        "requested_lr": config.get("requested_lr", config.get("lr", None)),
        "requested_max_steps": config.get("requested_max_steps", config.get("max_steps_requested", config.get("adaptation_steps", 0))),
        "requested_anchor_alpha": config.get("requested_anchor_alpha", config.get("anchor_alpha", None)),
        "requested_weight_decay": config.get("requested_weight_decay", config.get("weight_decay", None)),
        "requested_grad_clip": config.get("requested_grad_clip", config.get("grad_clip", None)),
        "adapt_batch_size": config.get("adapt_batch_size", config.get("batch_size", 0)),
        "max_steps_requested": config.get("max_steps_requested", config.get("adaptation_steps", 0)),
        "actual_optimizer_steps": config.get("actual_optimizer_steps", config.get("optimizer_steps_run", 0)),
        "optimizer_steps_run": config.get("optimizer_steps_run", 0),
        "support_batch_count": config.get("support_batch_count", 0),
        "effective_support_passes": config.get("effective_support_passes", 0.0),
        "adapt_recipe": config.get("adapt_recipe", ""),
        "policy_source": config.get("policy_source", "preregistered_default"),
        "safe_policy_json": config.get("safe_policy_json", ""),
        "safe_policy_json_sha256": config.get("safe_policy_json_sha256", ""),
        "safe_policy": dict(config.get("safe_policy", {}) or {}),
        "safe_policy_hash": config.get(
            "safe_policy_hash",
            (config.get("safe_policy", {}) or {}).get("policy_hash", ""),
        ),
        "source_policy_candidate_id": config.get("source_policy_candidate_id", ""),
        "source_policy_guard_config_hash": config.get("source_policy_guard_config_hash", ""),
        "source_episode_regions": list(config.get("source_episode_regions", []) or []),
        "rho_policy": config.get("rho_policy", "fixed_1.0"),
        "adapt_mix_rho": config.get("adapt_mix_rho", 1.0),
        "anchor_alpha": config.get("anchor_alpha", None),
        "anchor_alpha_grid_preregistered": list(config.get("anchor_alpha_grid_preregistered", [])),
        "source_anchor_hyperparameter_source": config.get("source_anchor_hyperparameter_source", ""),
        "support_loss_before": config.get("support_loss_before", None),
        "support_loss_after": config.get("support_loss_after", None),
        "support_final_loss": config.get("support_final_loss", None),
        "support_loss_delta": config.get("support_loss_delta", None),
        "standard_support_loss_before_full_support": config.get("standard_support_loss_before_full_support", None),
        "standard_support_loss_after_full_support": config.get("standard_support_loss_after_full_support", None),
        "standard_support_loss_delta_full_support": config.get("standard_support_loss_delta_full_support", None),
        "standard_support_surface_loss_before_full_support": config.get("standard_support_surface_loss_before_full_support", None),
        "standard_support_surface_loss_after_full_support": config.get("standard_support_surface_loss_after_full_support", None),
        "standard_support_rootzone_loss_before_full_support": config.get("standard_support_rootzone_loss_before_full_support", None),
        "standard_support_rootzone_loss_after_full_support": config.get("standard_support_rootzone_loss_after_full_support", None),
        "standard_support_objective_before_full_support": config.get("standard_support_objective_before_full_support", None),
        "standard_support_objective_after_full_support": config.get("standard_support_objective_after_full_support", None),
        "standard_support_increment_loss_before_full_support": config.get("standard_support_increment_loss_before_full_support", None),
        "standard_support_increment_loss_after_full_support": config.get("standard_support_increment_loss_after_full_support", None),
        "standard_support_analysis_loss_before_full_support": config.get("standard_support_analysis_loss_before_full_support", None),
        "standard_support_analysis_loss_after_full_support": config.get("standard_support_analysis_loss_after_full_support", None),
        "standard_support_analysis_surface_loss_before_full_support": config.get("standard_support_analysis_surface_loss_before_full_support", None),
        "standard_support_analysis_surface_loss_after_full_support": config.get("standard_support_analysis_surface_loss_after_full_support", None),
        "standard_support_analysis_rootzone_loss_before_full_support": config.get("standard_support_analysis_rootzone_loss_before_full_support", None),
        "standard_support_analysis_rootzone_loss_after_full_support": config.get("standard_support_analysis_rootzone_loss_after_full_support", None),
        "standard_support_regularization_loss_before_full_support": config.get("standard_support_regularization_loss_before_full_support", None),
        "standard_support_regularization_loss_after_full_support": config.get("standard_support_regularization_loss_after_full_support", None),
        "ridge_design_loss_before_sampled_pixels": config.get("ridge_design_loss_before_sampled_pixels", None),
        "ridge_design_loss_after_sampled_pixels": config.get("ridge_design_loss_after_sampled_pixels", None),
        "ridge_design_loss_delta_sampled_pixels": config.get("ridge_design_loss_delta_sampled_pixels", None),
        "target_parameter_l2_drift_pre_anchor": dict(config.get("target_parameter_l2_drift_pre_anchor", {})),
        "target_parameter_l2_drift_post_anchor": dict(config.get("target_parameter_l2_drift_post_anchor", config.get("target_parameter_l2_drift", {}))),
        "target_parameter_l2_drift": dict(config.get("target_parameter_l2_drift", {})),
        "target_labels_loaded_for_adaptation": bool(config.get("target_labels_loaded_for_adaptation", False)),
        "target_labels_used_for_adaptation": bool(config.get("target_labels_used_for_adaptation", False)),
        "normalization_source": config.get("normalization_source", ""),
        "model_selection_source": config.get("model_selection_source", ""),
        "target_val_usage": config.get("target_val_usage", ""),
        "target_eval_usage": config.get("target_eval_usage", ""),
        "checkpoint_selection": config.get("checkpoint_selection", ""),
        "prompt_policy": config.get("prompt_policy", ""),
        "prompt_label_usage": config.get("prompt_label_usage", ""),
        "eval_input_usage": config.get("eval_input_usage", ""),
        "eval_month_usage": config.get("eval_month_usage", ""),
        "frozen_modules": list(config.get("frozen_modules", [])),
        "frozen_source_groups": list(config.get("frozen_source_groups", [])),
        "trainable_modules": list(config.get("trainable_modules", [])),
        "trainable_target_groups": list(config.get("trainable_target_groups", [])),
        "git_hash": config.get("git_hash", ""),
        "timestamp": config.get("timestamp", ""),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _split_hashes(dataset: HydroDADataset) -> Dict[str, str]:
    entry = dataset._split_entry
    return {
        "target_context_dates_hash": entry.get("target_context_dates_hash", entry.get("target_train_dates_hash", "")),
        "target_support_dates_hash": entry.get("target_support_dates_hash", entry.get("support_dates_hash", "")),
        "target_eval_dates_hash": entry.get("target_eval_dates_hash", entry.get("target_query_dates_hash", "")),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    run_name = args.run_name or f"hyperda_zero_few_shot_{args.target_region}_K{args.K}_s{args.seed}"
    output_dir = Path(args.output_dir) if args.output_dir else RunManager(PHASE).create_run_dir(run_name)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    state = load_source_checkpoint_for_few_shot(
        checkpoint_path=args.source_checkpoint,
        device=device,
        target_latent_dim=args.target_latent_dim,
        enable_target_spatial_refine=args.enable_target_spatial_refine,
    )
    apply_target_adaptation_stage(state.model, epoch=0, stage1_epochs=0)
    apply_adapt_scope(state.model, args.adapt_scope, freeze_monthly_gain=args.freeze_monthly_gain)
    anchor_adapter_state = extract_target_adapter_state(state.model)
    source_prior_hash_before = hash_source_prior_state(state.model)

    target_context_dataset = HydroDADataset(
        da_nc_path=args.da_nc,
        region_masks_nc=args.region_masks_nc,
        splits_json=args.splits_json,
        target_region=args.target_region,
        split_type="target_context",
        K=args.K,
        seed=args.seed,
        adaptation_setting=args.adaptation_setting,
        freeze_manifest=args.freeze_manifest,
    )
    support_dataset = None
    should_load_support_labels = args.K > 0 and not (args.audit_identity and args.adapt_scope == "none")
    if should_load_support_labels:
        support_dataset = HydroDADataset(
            da_nc_path=args.da_nc,
            region_masks_nc=args.region_masks_nc,
            splits_json=args.splits_json,
            target_region=args.target_region,
            split_type="target_support",
            K=args.K,
            seed=args.seed,
            adaptation_setting=args.adaptation_setting,
            freeze_manifest=args.freeze_manifest,
        )

    split_hashes = _split_hashes(target_context_dataset)
    target_context_sample_count = len(target_context_dataset)
    if int(args.target_context_max_samples) > 0:
        target_context_sample_count = min(target_context_sample_count, int(args.target_context_max_samples))
    target_context_samples = (
        target_context_dataset.get_input_side_sample(i)
        for i in range(target_context_sample_count)
    )
    target_context_prompt_state = build_few_shot_target_context_prompt_state(
        state=state,
        samples=target_context_samples,
        target_region=args.target_region,
        device=device,
        context_hash=split_hashes.get("target_context_dates_hash", ""),
    )
    prompt_metadata = target_context_prompt_metadata(target_context_prompt_state)
    print(
        "Target-context monthly prompt prototypes: "
        f"n={prompt_metadata['n_samples']} "
        f"dates={prompt_metadata['date_start']}..{prompt_metadata['date_end']} "
        f"labels={prompt_metadata['label_usage']}",
        flush=True,
    )
    train_history: List[Dict[str, Any]] = []
    ridge_diagnostics: Dict[str, Any] = {}
    standard_support_loss_before: Dict[str, Optional[float]] = {}
    standard_support_loss_after: Dict[str, Optional[float]] = {}
    support_gradient_diagnostics: Dict[str, Any] = {}
    try:
        trainable_params = [p for p in state.model.parameters() if p.requires_grad]
        optimizer = None
        loss_fn: Optional[nn.Module] = None
        loss_loader = None
        if args.K > 0 and support_dataset is not None:
            loss_loader = _loader(support_dataset, args.batch_size, args.num_workers, shuffle=False)
            if args.use_lat_weighted_loss:
                loss_fn = WeightedMaskedHuberLoss(
                    delta=1.0,
                    surface_weight=args.surface_weight,
                    rootzone_weight=args.rootzone_weight,
                )
            else:
                loss_fn = MaskedHuberLoss(
                    delta=1.0,
                    surface_weight=args.surface_weight,
                    rootzone_weight=args.rootzone_weight,
                )
        if trainable_params and args.adapt_solver == "adamw" and args.K > 0 and args.adaptation_steps > 0:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        if args.K > 0 and args.adapt_solver == "ridge_coeff":
            assert support_dataset is not None
            assert loss_loader is not None
            started = time.time()
            ridge_diagnostics = run_ridge_coeff_adaptation(
                state=state,
                loader=loss_loader,
                device=device,
                target_context_prompt_state=target_context_prompt_state,
                normalize_increment=state.normalization.get("inc_mean") is not None,
                ridge_lambda=args.ridge_lambda,
                ridge_clip_coeff_norm=args.ridge_clip_coeff_norm,
                ridge_trust_region_radius=args.ridge_trust_region_radius,
                ridge_max_feature_pixels=args.ridge_max_feature_pixels,
                ridge_standardize_features=args.ridge_standardize_features,
                surface_weight=args.surface_weight,
                rootzone_weight=args.rootzone_weight,
                use_lat_weighted_loss=args.use_lat_weighted_loss,
            )
            print(
                "Ridge coefficient adaptation finished in "
                f"{time.time() - started:.1f}s: "
                f"status={ridge_diagnostics.get('status')} "
                f"lambda={args.ridge_lambda} "
                f"coef_norm={ridge_diagnostics.get('coefficient_norm')} "
                f"masked_obs={ridge_diagnostics.get('masked_observation_count')}",
                flush=True,
            )
        elif args.K > 0 and args.adaptation_steps > 0 and trainable_params:
            assert support_dataset is not None
            assert loss_fn is not None
            loader = _loader(support_dataset, args.batch_size, args.num_workers, shuffle=True)
            assert loss_loader is not None
            standard_support_loss_before = standard_support_loss_from_loader(
                state=state,
                loader=loss_loader,
                device=device,
                target_context_prompt_state=target_context_prompt_state,
                loss_fn=loss_fn,
                normalize_increment=state.normalization.get("inc_mean") is not None,
                lambda_prior=args.lambda_prior,
                lambda_latent=args.lambda_latent,
                lambda_gain=args.lambda_gain,
                lambda_gain_smooth=args.lambda_gain_smooth,
                lambda_analysis=args.lambda_analysis,
                support_loss_reduction=args.support_loss_reduction,
            )
            cycle_loader = _loader(support_dataset, 1, args.num_workers, shuffle=False)
            support_gradient_diagnostics = compute_support_gradient_conflict_diagnostics(
                state=state,
                loader=cycle_loader,
                device=device,
                target_context_prompt_state=target_context_prompt_state,
                loss_fn=loss_fn,
                normalize_increment=state.normalization.get("inc_mean") is not None,
                lambda_prior=args.lambda_prior,
                lambda_latent=args.lambda_latent,
                lambda_gain=args.lambda_gain,
                lambda_gain_smooth=args.lambda_gain_smooth,
                lambda_analysis=args.lambda_analysis,
                support_loss_reduction=args.support_loss_reduction,
            )
            started = time.time()
            train_history = train_fixed_steps(
                state=state,
                loader=loader,
                optimizer=optimizer,
                device=device,
                target_context_prompt_state=target_context_prompt_state,
                loss_fn=loss_fn,
                normalize_increment=state.normalization.get("inc_mean") is not None,
                adaptation_steps=args.adaptation_steps,
                grad_clip=args.grad_clip,
                lambda_prior=args.lambda_prior,
                lambda_latent=args.lambda_latent,
                lambda_gain=args.lambda_gain,
                lambda_gain_smooth=args.lambda_gain_smooth,
                lambda_analysis=args.lambda_analysis,
                log_every_steps=args.log_every_steps,
                support_loss_reduction=args.support_loss_reduction,
                trust_region_anchor_state=anchor_adapter_state,
                trust_region_mode=args.trust_region_mode,
                trust_total_radius=args.trust_total_radius,
                trust_prompt_radius=args.trust_prompt_radius,
                trust_gain_radius=args.trust_gain_radius,
                trust_coeff_radius=args.trust_coeff_radius,
                trust_spatial_radius=args.trust_spatial_radius,
            )
            standard_support_loss_after = standard_support_loss_from_loader(
                state=state,
                loader=loss_loader,
                device=device,
                target_context_prompt_state=target_context_prompt_state,
                loss_fn=loss_fn,
                normalize_increment=state.normalization.get("inc_mean") is not None,
                lambda_prior=args.lambda_prior,
                lambda_latent=args.lambda_latent,
                lambda_gain=args.lambda_gain,
                lambda_gain_smooth=args.lambda_gain_smooth,
                lambda_analysis=args.lambda_analysis,
                support_loss_reduction=args.support_loss_reduction,
            )
            support_gradient_diagnostics["support_cycle_loss_after"] = support_cycle_losses_from_loader(
                state=state,
                loader=cycle_loader,
                device=device,
                target_context_prompt_state=target_context_prompt_state,
                loss_fn=loss_fn,
                normalize_increment=state.normalization.get("inc_mean") is not None,
                lambda_prior=args.lambda_prior,
                lambda_latent=args.lambda_latent,
                lambda_gain=args.lambda_gain,
                lambda_gain_smooth=args.lambda_gain_smooth,
                lambda_analysis=args.lambda_analysis,
                support_loss_reduction=args.support_loss_reduction,
            )
            before_losses = support_gradient_diagnostics.get("support_cycle_loss_before", [])
            after_losses = support_gradient_diagnostics.get("support_cycle_loss_after", [])
            improvements = [float(b) - float(a) for b, a in zip(before_losses, after_losses)]
            support_gradient_diagnostics["support_cycle_loss_improvement_mean"] = (
                float(np.mean(improvements)) if improvements else None
            )
            support_gradient_diagnostics["support_cycle_loss_improvement_std"] = (
                float(np.std(improvements)) if improvements else None
            )
            print(f"Fixed-step few-shot training finished in {time.time() - started:.1f}s")
        elif args.K > 0:
            print(
                "K>0: skipping target-label training "
                f"(steps={args.adaptation_steps}, adapt_scope={args.adapt_scope}, trainable_params={len(trainable_params)})."
            )
        else:
            print("K=0: skipping target-label training; saving source prior with target-context metadata.")

        pre_anchor_adapter_state = extract_target_adapter_state(state.model)
        pre_anchor_drift = target_parameter_l2_drift(anchor_adapter_state, pre_anchor_adapter_state)
        if args.K > 0:
            if args.adapt_recipe in {"source_anchor", "conservative", "episode_prior"}:
                apply_source_anchor_interpolation(state.model, anchor_adapter_state, alpha=args.anchor_alpha)
                print(
                    "Applied source-anchor interpolation: "
                    f"recipe={args.adapt_recipe} alpha={args.anchor_alpha:.4f}",
                    flush=True,
                )
        else:
            apply_source_anchor_interpolation(state.model, anchor_adapter_state, alpha=0.0)

        if args.K > 0 and loss_loader is not None and loss_fn is not None:
            standard_support_loss_after = standard_support_loss_from_loader(
                state=state,
                loader=loss_loader,
                device=device,
                target_context_prompt_state=target_context_prompt_state,
                loss_fn=loss_fn,
                normalize_increment=state.normalization.get("inc_mean") is not None,
                lambda_prior=args.lambda_prior,
                lambda_latent=args.lambda_latent,
                lambda_gain=args.lambda_gain,
                lambda_gain_smooth=args.lambda_gain_smooth,
                lambda_analysis=args.lambda_analysis,
                support_loss_reduction=args.support_loss_reduction,
            )
            if not standard_support_loss_before:
                apply_target_adapter_state(state.model, anchor_adapter_state)
                standard_support_loss_before = standard_support_loss_from_loader(
                    state=state,
                    loader=loss_loader,
                    device=device,
                    target_context_prompt_state=target_context_prompt_state,
                    loss_fn=loss_fn,
                    normalize_increment=state.normalization.get("inc_mean") is not None,
                    lambda_prior=args.lambda_prior,
                    lambda_latent=args.lambda_latent,
                    lambda_gain=args.lambda_gain,
                    lambda_gain_smooth=args.lambda_gain_smooth,
                    lambda_analysis=args.lambda_analysis,
                    support_loss_reduction=args.support_loss_reduction,
                )
                apply_target_adapter_state(state.model, pre_anchor_adapter_state)
                if args.adapt_recipe in {"source_anchor", "conservative", "episode_prior"}:
                    apply_source_anchor_interpolation(state.model, anchor_adapter_state, alpha=args.anchor_alpha)
            if is_source_calibrated_kshot_no_update(args):
                support_gate_summary = support_gate_summary_for_source_calibrated_no_update(args)
                print(
                    "Source-side SAFE policy selected no K-shot update; saving the frozen-prior anchor.",
                    flush=True,
                )
            else:
                support_gate_summary = decide_support_gate(
                    before=standard_support_loss_before,
                    after=standard_support_loss_after,
                    enabled=args.support_gate == "auto",
                    min_delta=float(args.support_gate_min_delta),
                    rootzone_tolerance=float(args.support_gate_rootzone_tolerance),
                )
            if support_gate_summary["stage3_posterior_decision"] == "rejected_to_k0_anchor":
                apply_target_adapter_state(state.model, anchor_adapter_state)
                standard_support_loss_after = dict(standard_support_loss_before)
                support_gate_summary.update(
                    {
                        "support_objective_after": support_gate_summary.get("support_objective_before"),
                        "support_objective_delta": 0.0
                        if support_gate_summary.get("support_objective_before") is not None
                        else None,
                        "support_loss_after": support_gate_summary.get("support_loss_before"),
                        "support_loss_delta": 0.0
                        if support_gate_summary.get("support_loss_before") is not None
                        else None,
                        "support_surface_loss_after": support_gate_summary.get("support_surface_loss_before"),
                        "support_surface_loss_delta": 0.0
                        if support_gate_summary.get("support_surface_loss_before") is not None
                        else None,
                        "support_rootzone_loss_after": support_gate_summary.get("support_rootzone_loss_before"),
                        "support_rootzone_loss_delta": 0.0
                        if support_gate_summary.get("support_rootzone_loss_before") is not None
                        else None,
                    }
                )
                print(
                    "Support gate rejected Stage-3 posterior; rolled back to frozen-prior anchor: "
                    f"reasons={support_gate_summary.get('support_gate_reject_reason')}",
                    flush=True,
                )
            else:
                print(
                    "Support gate decision: "
                    f"{support_gate_summary['stage3_posterior_decision']} "
                    f"objective_delta={support_gate_summary.get('support_candidate_objective_delta')}",
                    flush=True,
                )
        elif args.K == 0:
            support_gate_summary = support_gate_summary_for_k0()
        else:
            support_gate_summary = {
                "support_gate_enabled": False,
                "support_gate_status": "skipped_no_support_objective",
                "stage3_posterior_decision": "accepted",
                "support_gate_label_source": "target_support_only",
                "support_gate_min_delta": float(args.support_gate_min_delta),
                "support_gate_rootzone_tolerance": float(args.support_gate_rootzone_tolerance),
                "support_gate_reject_reason": [],
            }

        if args.K > 0 and args.policy_source != "source_side_episode_calibration":
            apply_target_adapter_state(state.model, anchor_adapter_state)
            support_gate_summary.update(
                {
                    "support_gate_enabled": bool(support_gate_summary.get("support_gate_enabled", False)),
                    "support_gate_status": "missing_source_policy_rejected_to_k0_anchor",
                    "support_only_gate_status": str(
                        support_gate_summary.get("support_gate_status", "skipped_no_source_policy")
                    ),
                    "stage3_posterior_decision": "rejected_to_k0_anchor",
                    "support_gate_reject_reason": list(
                        support_gate_summary.get("support_gate_reject_reason", []) or []
                    )
                    + ["missing_source_side_safe_policy_json"],
                }
            )
            print(
                "No source-side SAFE policy was provided for K-shot Stage 3; "
                "saving the K0 anchor state as a diagnostic run.",
                flush=True,
            )

        final_adapter_state = extract_target_adapter_state(state.model)
        drift = target_parameter_l2_drift(anchor_adapter_state, final_adapter_state)
        source_prior_hash_after = hash_source_prior_state(state.model)
        if args.K == 0:
            if source_prior_hash_before != source_prior_hash_after:
                raise RuntimeError("K=0 no-update protocol violation: source prior hash changed")
            if float(drift.get("total", 0.0)) != 0.0:
                raise RuntimeError(
                    "K=0 no-update protocol violation: target posterior drift is "
                    f"{drift.get('total')}, expected exactly 0.0"
                )
        if args.freeze_monthly_gain and float(drift.get("monthly_gain", 0.0)) != 0.0:
            raise RuntimeError(
                "FREEZE_MONTHLY_GAIN protocol violation: monthly_gain drift is "
                f"{drift.get('monthly_gain')}, expected exactly 0.0"
            )
        loss_summary = support_loss_summary(train_history)
        standard_loss_before = standard_support_loss_before.get("standard_support_loss_full_support")
        standard_loss_after = standard_support_loss_after.get("standard_support_loss_full_support")
        standard_loss_delta = (
            None
            if standard_loss_before is None or standard_loss_after is None
            else float(standard_loss_after) - float(standard_loss_before)
        )
        standard_loss_summary = {
            "standard_support_loss_before_full_support": standard_loss_before,
            "standard_support_loss_after_full_support": standard_loss_after,
            "standard_support_loss_delta_full_support": standard_loss_delta,
            "standard_support_surface_loss_before_full_support": standard_support_loss_before.get("standard_support_surface_loss_full_support"),
            "standard_support_surface_loss_after_full_support": standard_support_loss_after.get("standard_support_surface_loss_full_support"),
            "standard_support_rootzone_loss_before_full_support": standard_support_loss_before.get("standard_support_rootzone_loss_full_support"),
            "standard_support_rootzone_loss_after_full_support": standard_support_loss_after.get("standard_support_rootzone_loss_full_support"),
            "standard_support_objective_before_full_support": standard_support_loss_before.get("standard_support_objective_full_support"),
            "standard_support_objective_after_full_support": standard_support_loss_after.get("standard_support_objective_full_support"),
            "standard_support_increment_loss_before_full_support": standard_support_loss_before.get("standard_support_increment_loss_full_support"),
            "standard_support_increment_loss_after_full_support": standard_support_loss_after.get("standard_support_increment_loss_full_support"),
            "standard_support_analysis_loss_before_full_support": standard_support_loss_before.get("standard_support_analysis_loss_full_support"),
            "standard_support_analysis_loss_after_full_support": standard_support_loss_after.get("standard_support_analysis_loss_full_support"),
            "standard_support_analysis_surface_loss_before_full_support": standard_support_loss_before.get("standard_support_analysis_surface_loss_full_support"),
            "standard_support_analysis_surface_loss_after_full_support": standard_support_loss_after.get("standard_support_analysis_surface_loss_full_support"),
            "standard_support_analysis_rootzone_loss_before_full_support": standard_support_loss_before.get("standard_support_analysis_rootzone_loss_full_support"),
            "standard_support_analysis_rootzone_loss_after_full_support": standard_support_loss_after.get("standard_support_analysis_rootzone_loss_full_support"),
            "standard_support_regularization_loss_before_full_support": standard_support_loss_before.get("standard_support_regularization_loss_full_support"),
            "standard_support_regularization_loss_after_full_support": standard_support_loss_after.get("standard_support_regularization_loss_full_support"),
        }
        ridge_design_loss_before = None
        ridge_design_loss_after = None
        ridge_design_loss_delta = None
        if ridge_diagnostics:
            ridge_design_loss_before = ridge_diagnostics.get("support_loss_before")
            ridge_design_loss_after = ridge_diagnostics.get("support_loss_after")
            ridge_design_loss_delta = ridge_diagnostics.get("support_loss_delta")
        split_manifest_sha256 = compute_sha256(args.splits_json) if Path(args.splits_json).exists() else ""
        source_checkpoint_sha256 = compute_sha256(args.source_checkpoint) if Path(args.source_checkpoint).exists() else ""
        requires_grad_parameter_count = int(sum(p.numel() for p in state.model.parameters() if p.requires_grad))
        optimizer_parameter_count = int(sum(p.numel() for p in trainable_params)) if optimizer is not None else 0
        target_support_dates = _date_str_records(support_dataset, "target_support_dates")
        if not target_support_dates:
            target_support_dates = _date_str_records(target_context_dataset, "target_support_dates")
        target_support_count = len(target_support_dates)
        target_labels_used = bool(train_history) or bool(ridge_diagnostics) or bool(
            args.K > 0 and support_gate_summary.get("support_gate_enabled", False)
        )
        optimizer_steps_run = len(train_history)
        if ridge_diagnostics:
            optimizer_steps_run = 0
        support_batches = support_batch_count(target_support_count, args.batch_size)
        support_passes = effective_support_passes(optimizer_steps_run, support_batches)
        if ridge_diagnostics:
            loss_summary = {
                "support_loss_before": ridge_diagnostics.get("support_loss_before"),
                "support_loss_after": ridge_diagnostics.get("support_loss_after"),
                "support_final_loss": ridge_diagnostics.get("support_loss_after"),
                "support_loss_delta": ridge_diagnostics.get("support_loss_delta"),
            }
        elif standard_loss_before is not None or standard_loss_after is not None:
            loss_summary = {
                "support_loss_before": standard_loss_before,
                "support_loss_after": standard_loss_after,
                "support_final_loss": standard_loss_after,
                "support_loss_delta": standard_loss_delta,
            }
        trust_projection_diagnostics = summarize_trust_projection_history(train_history)
        support_manifest_hash = _support_manifest_hash(
            target_region=args.target_region,
            K=args.K,
            seed=args.seed,
            target_support_dates=target_support_dates,
            target_support_dates_hash=split_hashes.get("target_support_dates_hash", ""),
            split_manifest_sha256=split_manifest_sha256,
        )
        support_nesting_hash = _json_sha256_payload(
            {
                "target_region": args.target_region,
                "K": args.K,
                "seed": args.seed,
                "support_manifest_hash": support_manifest_hash,
                "target_support_dates": target_support_dates,
            }
        )
        stage3_prior_snapshot = build_stage3_prior_snapshot_metadata(
            source_config=state.source_config,
            prompt_state=target_context_prompt_state,
            source_checkpoint_sha256=source_checkpoint_sha256,
            target_region=args.target_region,
            K=args.K,
        )
        paper_facing_status = paper_facing_status_for_stage3(
            K=args.K,
            policy_source=args.policy_source,
            stage3_posterior_decision=str(support_gate_summary.get("stage3_posterior_decision", "accepted")),
        )
        paper_facing_run = bool(paper_facing_status["paper_facing_run"])
        paper_selection_basis = (
            "source_side_safe_policy_only"
            if args.K > 0 and args.policy_source == "source_side_episode_calibration"
            else (
                "diagnostic_no_source_safe_policy_json"
                if args.K > 0
                else "zero_shot_no_target_labels"
            )
        )
        if str(support_gate_summary.get("stage3_posterior_decision", "")) == "rejected_to_k0_anchor":
            stage3_acceptance_basis = (
                "source_policy_or_gate_rejected_to_k0_anchor"
                if paper_facing_run
                else "diagnostic_no_source_safe_policy_json_rejected_to_k0_anchor"
            )
        else:
            stage3_acceptance_basis = paper_selection_basis
        stage3_posterior_state = build_stage3_posterior_state_metadata(
            anchor_state=anchor_adapter_state,
            final_state=final_adapter_state,
            K=args.K,
            adapt_scope=args.adapt_scope,
            anchor_alpha=float(args.anchor_alpha),
            adaptation_steps=int(args.adaptation_steps),
            target_labels_loaded=bool(support_dataset is not None),
            target_labels_used=target_labels_used,
            source_prior_hash_before=source_prior_hash_before,
            source_prior_hash_after=source_prior_hash_after,
            stage3_posterior_policy=args.stage3_posterior_policy,
            stage3_posterior_decision=str(support_gate_summary.get("stage3_posterior_decision", "accepted")),
            support_gate_status=str(support_gate_summary.get("support_gate_status", "disabled")),
            paper_selection_basis=paper_selection_basis,
            stage3_acceptance_basis=stage3_acceptance_basis,
            source_policy_candidate_id=args.source_policy_candidate_id,
        )
        config = {
            "K": args.K,
            "adaptation_setting": args.adaptation_setting,
            "adapt_scope": args.adapt_scope,
            "stage3_posterior_policy": args.stage3_posterior_policy,
            "stage3_no_update_contract": (
                "K0_fixed_no_update_source_prior_identity"
                if args.K == 0
                else "Kshot_source_policy_constrained_posterior_update"
            ),
            "paper_selection_basis": paper_selection_basis,
            "stage3_acceptance_basis": stage3_acceptance_basis,
            "paper_facing_run": paper_facing_run,
            "diagnostic_run_reason": str(paper_facing_status["diagnostic_run_reason"]),
            "freeze_monthly_gain": bool(args.freeze_monthly_gain),
            "adapt_solver": args.adapt_solver,
            "trust_region_mode": args.trust_region_mode,
            "trust_total_radius": float(args.trust_total_radius),
            "trust_prompt_radius": float(args.trust_prompt_radius),
            "trust_gain_radius": float(args.trust_gain_radius),
            "trust_coeff_radius": float(args.trust_coeff_radius),
            "trust_spatial_radius": float(args.trust_spatial_radius),
            "trust_projection_diagnostics": trust_projection_diagnostics,
            "support_loss_reduction": args.support_loss_reduction,
            "support_gradient_diagnostics": support_gradient_diagnostics,
            "ridge_lambda": float(args.ridge_lambda),
            "ridge_clip_coeff_norm": float(args.ridge_clip_coeff_norm),
            "ridge_trust_region_radius": float(args.ridge_trust_region_radius),
            "ridge_max_feature_pixels": int(args.ridge_max_feature_pixels),
            "ridge_standardize_features": bool(args.ridge_standardize_features),
            "ridge_diagnostics": ridge_diagnostics,
            "audit_identity": bool(args.audit_identity),
            "audit_identity_tolerance": float(args.audit_identity_tolerance),
            "adapt_recipe": args.adapt_recipe,
            "policy_source": args.policy_source,
            "safe_policy_json": args.safe_policy_json or "",
            "safe_policy_json_sha256": args.safe_policy_json_sha256,
            "safe_policy_hash": args.safe_policy_hash,
            "safe_policy": args.safe_policy,
            "source_policy_candidate_id": args.source_policy_candidate_id,
            "source_policy_guard_config_hash": args.source_policy_guard_config_hash,
            "source_episode_regions": args.source_episode_regions,
            "rho_policy": args.rho_policy,
            "adapt_mix_rho": float(args.adapt_mix_rho),
            "anchor_alpha": float(args.anchor_alpha),
            "anchor_alpha_grid_preregistered": default_anchor_alpha_grid_for_K(args.K),
            "source_anchor_hyperparameter_source": args.source_anchor_hyperparameter_source,
            "target_eval_usage": args.target_eval_usage,
            "target_region": args.target_region,
            "seed": args.seed,
            "source_checkpoint": args.source_checkpoint,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "staged_source_checkpoint_sha256": source_checkpoint_sha256,
            "source_stage_checkpoint_provenance": "phase4_hyperda_staged",
            "stage3_source_prior_hash_before": source_prior_hash_before,
            "stage3_source_prior_hash_after": source_prior_hash_after,
            "stage3_source_prior_unchanged": source_prior_hash_before == source_prior_hash_after,
            "k0_target_drift_zero": bool(args.K == 0 and float(drift.get("total", 0.0)) == 0.0),
            "split_manifest_path": args.splits_json,
            "split_manifest_sha256": split_manifest_sha256,
            "adaptation_steps": args.adaptation_steps,
            "schedule_label": args.schedule_label,
            "requested_lr": args.lr,
            "requested_max_steps": args.adaptation_steps,
            "requested_anchor_alpha": float(args.anchor_alpha),
            "requested_weight_decay": args.weight_decay,
            "requested_grad_clip": args.grad_clip,
            "max_steps_requested": args.adaptation_steps,
            "actual_optimizer_steps": optimizer_steps_run,
            "optimizer_steps_run": optimizer_steps_run,
            "support_batch_count": support_batches,
            "effective_support_passes": support_passes,
            "adapt_batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "lambda_prior": args.lambda_prior,
            "lambda_latent": args.lambda_latent,
            "lambda_gain": args.lambda_gain,
            "lambda_gain_smooth": args.lambda_gain_smooth,
            "lambda_analysis": args.lambda_analysis,
            "surface_weight": args.surface_weight,
            "rootzone_weight": args.rootzone_weight,
            "use_lat_weighted_loss": bool(args.use_lat_weighted_loss),
            **loss_summary,
            **support_gate_summary,
            "support_gate_policy_role": support_gate_summary.get(
                "support_gate_policy_role",
                (
                    "target_support_only_diagnostic_not_paper_selection"
                    if args.K > 0
                    else "not_applicable_k0_no_support"
                ),
            ),
            **standard_loss_summary,
            "ridge_design_loss_before_sampled_pixels": ridge_design_loss_before,
            "ridge_design_loss_after_sampled_pixels": ridge_design_loss_after,
            "ridge_design_loss_delta_sampled_pixels": ridge_design_loss_delta,
            "target_parameter_l2_drift_pre_anchor": pre_anchor_drift,
            "target_parameter_l2_drift_post_anchor": drift,
            "target_parameter_l2_drift": drift,
            "stage3_prior_snapshot": stage3_prior_snapshot,
            "stage3_posterior_state": stage3_posterior_state,
            "target_parameter_count_by_group": group_target_parameter_counts(state.model),
            "requires_grad_parameter_count": requires_grad_parameter_count,
            "requires_grad_param_count": requires_grad_parameter_count,
            "optimizer_parameter_count": optimizer_parameter_count,
            "optimizer_param_count": optimizer_parameter_count,
            "target_support_dates": target_support_dates,
            "support_manifest_hash": support_manifest_hash,
            "support_nesting_hash": support_nesting_hash,
            "support_nesting_status": _support_nesting_status(args.K, target_support_dates),
            "target_support_count": target_support_count,
            "target_labels_loaded_for_adaptation": bool(support_dataset is not None),
            "target_labels_used_for_adaptation": target_labels_used,
            "target_latent_dim": args.target_latent_dim,
            "enable_target_spatial_refine": args.enable_target_spatial_refine,
            "target_adapter_anchor_state": anchor_adapter_state,
            "target_context_max_samples": int(args.target_context_max_samples),
            "target_context_samples_used": int(target_context_sample_count),
            **split_hashes,
        }
        final_path = checkpoints_dir / "checkpoint_final_preregistered.pt"
        saved_config = save_few_shot_checkpoint(
            path=final_path,
            state=state,
            optimizer_state_dict=optimizer.state_dict() if optimizer is not None else {},
            config=config,
            target_context_prompt_state=target_context_prompt_state,
            train_history=train_history,
        )
        write_run_metadata_sidecar(output_dir, final_path, saved_config)
        print(f"Saved: {final_path}")
    finally:
        target_context_dataset.close()
        if support_dataset is not None:
            support_dataset.close()


if __name__ == "__main__":
    main()
