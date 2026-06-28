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
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from hydroda.data.dataset import HydroDADataset, collate_hydroda_samples
from hydroda.data.file_hash import compute_sha256
from hydroda.data.protocol import ProtocolConfig
from hydroda.baselines.prompt_conditioned import (
    CONTEXT_TTA_MODES,
    CONTEXT_TTA_NONE,
    SOURCE_MANIFOLD_DISTANCE_KEY,
    build_target_context_prompt_state,
    compose_target_context_prompt_from_state,
    compose_target_context_source_trust_query_from_state,
    normalize_hyperda_source_trust_bank_state,
    normalize_source_prompt_manifold_guard_state,
    normalize_target_context_prompt_state,
    target_context_prompt_metadata,
    ROBUST_DA_CONTEXT_ENCODER,
    ROBUST_DA_RAW_CONTEXT_ENCODER,
)
from hydroda.models.hyper_conditional_unet import HyperAdapterConditionalResUNet
from hydroda.models.prompt_encoder import RegionPromptEncoder, RobustInputSideDAPromptEncoder
from hydroda.training.losses import MaskedHuberLoss, WeightedMaskedHuberLoss
from hydroda.training.calibration import calibrate_residual_affine, calibrate_residual_gain
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
RIDGE_WEIGHTINGS = ("global_pixel_l2", "cycle_variable_balanced_huber")
STAGE3_KSHOT_MODES = (
    "paper_safe",
    "diagnostic_direct_kshot",
    "diagnostic_direct_kshot_v2",
    "diagnostic_conservative_kshot_v3",
    "diagnostic_support_gain_v1",
    "diagnostic_support_gain_v2",
    "diagnostic_support_gain_v3_stable",
    "diagnostic_support_gain_v4_nested_stable",
    "diagnostic_support_gain_v12_nested_cv",
    "diagnostic_support_gain_v13_k12_aggressive_calibration_pool",
    "diagnostic_finetune_support_gain_v14_nested",
    "diagnostic_support_affine_v1_nested",
    "diagnostic_safe_operator_v5_nested",
    "diagnostic_linearized_coeff_ridge_v6_nested",
    "diagnostic_linearized_coeff_ridge_v7_balanced_nested",
    "diagnostic_linearized_coeff_ridge_v8_hybrid_nested",
    "diagnostic_linearized_coeff_ridge_v9_guarded_nested",
    "diagnostic_linearized_coeff_ridge_v10_support_pool_nested",
    "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested",
)
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


def is_diagnostic_direct_kshot_mode(stage3_kshot_mode: str) -> bool:
    return str(stage3_kshot_mode) in {
        "diagnostic_direct_kshot",
        "diagnostic_direct_kshot_v2",
        "diagnostic_conservative_kshot_v3",
        "diagnostic_support_gain_v1",
        "diagnostic_support_gain_v2",
        "diagnostic_support_gain_v3_stable",
        "diagnostic_support_gain_v4_nested_stable",
        "diagnostic_support_gain_v12_nested_cv",
        "diagnostic_support_gain_v13_k12_aggressive_calibration_pool",
        "diagnostic_finetune_support_gain_v14_nested",
        "diagnostic_support_affine_v1_nested",
        "diagnostic_safe_operator_v5_nested",
        "diagnostic_linearized_coeff_ridge_v6_nested",
        "diagnostic_linearized_coeff_ridge_v7_balanced_nested",
        "diagnostic_linearized_coeff_ridge_v8_hybrid_nested",
        "diagnostic_linearized_coeff_ridge_v9_guarded_nested",
        "diagnostic_linearized_coeff_ridge_v10_support_pool_nested",
        "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested",
    }


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
        return 100
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


def diagnostic_kshot_v2_defaults(K: int, strength: str) -> Dict[str, Any]:
    """Return explicit non-paper K-shot v2 defaults for target-support upper-bound diagnosis."""
    strength = str(strength or "strong").strip().lower()
    if strength not in {"strong", "medium"}:
        raise ValueError("DIAGNOSTIC_KSHOT_STRENGTH must be strong or medium")
    if int(K) == 4:
        steps = 100 if strength == "strong" else 50
    elif int(K) == 12:
        steps = 200 if strength == "strong" else 100
    else:
        steps = 0
    return {
        "strength": strength,
        "adaptation_steps": int(steps),
        "lr": 1e-3,
        "anchor_alpha": 1.0 if int(K) > 0 else 0.0,
        "adapt_mix_rho": 1.0,
        "adapt_scope": "safe_operator" if int(K) > 0 else "none",
        "support_gate": "off",
        "support_loss_reduction": "cycle_balanced",
        "source_anchor_hyperparameter_source": f"diagnostic_direct_kshot_v2_{strength}_fixed_defaults",
    }


def diagnostic_conservative_kshot_v3_defaults(K: int, strength: str) -> Dict[str, Any]:
    """Return fixed conservative K-shot defaults after v2 prompt-drift failure.

    The v3 diagnostic deliberately avoids target-prompt and monthly-gain
    updates. It only tests whether small coefficient residual changes supported
    by target_support can move predictions without the large v2 prompt drift.
    """
    strength = str(strength or "strong").strip().lower()
    if strength not in {"strong", "medium"}:
        raise ValueError("DIAGNOSTIC_KSHOT_STRENGTH must be strong or medium")
    if int(K) == 4:
        steps = 40 if strength == "strong" else 20
        coeff_radius = 0.20 if strength == "strong" else 0.10
        anchor_alpha = 0.35 if strength == "strong" else 0.25
        adapt_mix_rho = 0.50 if strength == "strong" else 0.35
    elif int(K) == 12:
        steps = 80 if strength == "strong" else 40
        coeff_radius = 0.30 if strength == "strong" else 0.15
        anchor_alpha = 0.50 if strength == "strong" else 0.35
        adapt_mix_rho = 0.60 if strength == "strong" else 0.45
    else:
        steps = 0
        coeff_radius = 0.0
        anchor_alpha = 0.0
        adapt_mix_rho = 1.0
    return {
        "strength": strength,
        "adaptation_steps": int(steps),
        "lr": 3e-4,
        "anchor_alpha": float(anchor_alpha),
        "adapt_mix_rho": float(adapt_mix_rho),
        "adapt_scope": "coeff_only" if int(K) > 0 else "none",
        "support_gate": "auto",
        "support_gate_min_delta": 1e-8,
        "support_gate_rootzone_tolerance": 1e-8,
        "support_loss_reduction": "cycle_balanced",
        "freeze_monthly_gain": True,
        "trust_region_mode": "groupwise" if int(K) > 0 else "none",
        "trust_prompt_radius": 0.0,
        "trust_gain_radius": 0.0,
        "trust_coeff_radius": float(coeff_radius),
        "trust_spatial_radius": 0.0,
        "source_anchor_hyperparameter_source": (
            f"diagnostic_conservative_kshot_v3_{strength}_fixed_defaults"
        ),
    }


def diagnostic_safe_operator_v5_defaults(K: int, strength: str) -> Dict[str, Any]:
    """Return fixed v5 nested SAFE diagnostic defaults.

    v5 keeps the same frozen backbone/prompt/hypernetwork/basis contract as
    conservative v3, but names the source-side policy explicitly and uses the
    nested K12 support comparison against a run-local K4 reference when K=12.
    """
    defaults = dict(diagnostic_conservative_kshot_v3_defaults(K, strength))
    defaults["source_anchor_hyperparameter_source"] = (
        "diagnostic_safe_operator_v5_nested_support_only_coeff_residual_policy"
    )
    return defaults


def diagnostic_linearized_coeff_ridge_v6_defaults(K: int, strength: str) -> Dict[str, Any]:
    """Return fixed v6 closed-form coefficient-ridge diagnostic defaults.

    v6 performs no Adam updates. It freezes the source prior path and solves a
    local linearized ridge system only for target adapter coefficient residual
    logits, then applies the same support-only K0/K4 gates as the nested SAFE
    diagnostics. The constants are fixed source-side policy defaults for this
    exploratory mode; target_eval remains final-eval only.
    """
    strength = str(strength or "strong").strip().lower()
    if strength not in {"strong", "medium"}:
        raise ValueError("DIAGNOSTIC_KSHOT_STRENGTH must be strong or medium")
    if int(K) == 4:
        ridge_lambda = 2.0 if strength == "strong" else 4.0
        trust_radius = 0.18 if strength == "strong" else 0.10
        coeff_norm = 0.25 if strength == "strong" else 0.15
        anchor_alpha = 0.40 if strength == "strong" else 0.30
        adapt_mix_rho = 0.50 if strength == "strong" else 0.35
    elif int(K) == 12:
        ridge_lambda = 1.0 if strength == "strong" else 2.0
        trust_radius = 0.28 if strength == "strong" else 0.18
        coeff_norm = 0.40 if strength == "strong" else 0.25
        anchor_alpha = 0.60 if strength == "strong" else 0.45
        adapt_mix_rho = 0.65 if strength == "strong" else 0.50
    else:
        ridge_lambda = 0.0
        trust_radius = 0.0
        coeff_norm = 0.0
        anchor_alpha = 0.0
        adapt_mix_rho = 1.0
    return {
        "strength": strength,
        "adaptation_steps": 0,
        "lr": 0.0,
        "anchor_alpha": float(anchor_alpha),
        "adapt_mix_rho": float(adapt_mix_rho),
        "adapt_scope": "coeff_only" if int(K) > 0 else "none",
        "adapt_solver": "ridge_coeff" if int(K) > 0 else "adamw",
        "support_gate": "auto",
        "support_gate_min_delta": 1e-8,
        "support_gate_rootzone_tolerance": 1e-8,
        "support_loss_reduction": "cycle_balanced",
        "freeze_monthly_gain": True,
        "trust_region_mode": "groupwise" if int(K) > 0 else "none",
        "trust_prompt_radius": 0.0,
        "trust_gain_radius": 0.0,
        "trust_coeff_radius": float(trust_radius),
        "trust_spatial_radius": 0.0,
        "ridge_lambda": float(ridge_lambda),
        "ridge_clip_coeff_norm": float(coeff_norm),
        "ridge_trust_region_radius": float(trust_radius),
        "ridge_max_feature_pixels": 20000,
        "ridge_standardize_features": True,
        "adaptation_step_policy_source": (
            "diagnostic_linearized_coeff_ridge_v6_closed_form_no_adam_steps"
        ),
        "source_anchor_hyperparameter_source": (
            "diagnostic_linearized_coeff_ridge_v6_nested_source_side_policy_defaults"
        ),
    }


def diagnostic_linearized_coeff_ridge_v7_defaults(K: int, strength: str) -> Dict[str, Any]:
    """Return fixed v7 balanced closed-form coefficient-ridge defaults.

    v7 keeps the v6 no-Adam, coeff-only, nested-K12 contract. Its only method
    change is the target_support-only ridge objective: support cycles and output
    variables are balanced before forming the local linear system, with Huber
    influence weights computed from frozen-prior support residuals.
    """
    defaults = dict(diagnostic_linearized_coeff_ridge_v6_defaults(K, strength))
    defaults.update(
        {
            "ridge_weighting": "cycle_variable_balanced_huber",
            "adaptation_step_policy_source": (
                "diagnostic_linearized_coeff_ridge_v7_balanced_closed_form_no_adam_steps"
            ),
            "source_anchor_hyperparameter_source": (
                "diagnostic_linearized_coeff_ridge_v7_balanced_nested_source_side_policy_defaults"
            ),
        }
    )
    return defaults


def diagnostic_linearized_coeff_ridge_v8_hybrid_defaults(K: int, strength: str) -> Dict[str, Any]:
    """Return fixed v8 hybrid coefficient-ridge diagnostic defaults.

    v8 is an exploratory US-R1 follow-up after v7 improved K4 but weakened K12:
    K4 uses v7's cycle/variable balanced Huber support objective, while K12
    keeps v6's global-pixel ridge objective. The solver contract remains
    no-Adam, coeff-only, nested-K12, and target_eval-final-only.
    """
    defaults = dict(diagnostic_linearized_coeff_ridge_v6_defaults(K, strength))
    if int(K) == 4:
        defaults["ridge_weighting"] = "cycle_variable_balanced_huber"
    else:
        defaults["ridge_weighting"] = "global_pixel_l2"
    defaults.update(
        {
            "adaptation_step_policy_source": (
                "diagnostic_linearized_coeff_ridge_v8_hybrid_closed_form_no_adam_steps"
            ),
            "source_anchor_hyperparameter_source": (
                "diagnostic_linearized_coeff_ridge_v8_hybrid_nested_source_side_policy_defaults"
            ),
        }
    )
    return defaults


def diagnostic_linearized_coeff_ridge_v9_guarded_defaults(K: int, strength: str) -> Dict[str, Any]:
    """Return fixed v9 guarded coefficient-ridge diagnostic defaults.

    v9 keeps v8's K-specific ridge objective, but makes the nested K12 gate more
    conservative: K12 must beat the run-local K4 reference by a fixed support
    objective margin before it is used. Otherwise K12 falls back to the K4
    checkpoint, which prevents a weak support-only K12 signal from degrading the
    final K=12 artifact.
    """
    defaults = dict(diagnostic_linearized_coeff_ridge_v8_hybrid_defaults(K, strength))
    if int(K) == 12:
        defaults["support_gate_min_delta"] = 3e-3 if defaults["strength"] == "strong" else 2e-3
        defaults["adapt_mix_rho"] = 0.50 if defaults["strength"] == "medium" else 0.65
    defaults.update(
        {
            "adaptation_step_policy_source": (
                "diagnostic_linearized_coeff_ridge_v9_guarded_closed_form_no_adam_steps"
            ),
            "source_anchor_hyperparameter_source": (
                "diagnostic_linearized_coeff_ridge_v9_guarded_nested_source_side_policy_defaults"
            ),
        }
    )
    return defaults


def diagnostic_linearized_coeff_ridge_v10_support_pool(K: int) -> List[Dict[str, Any]]:
    """Return the fixed v10 target-support candidate pool.

    v10 is an exploratory US-R1 redesign after target_eval exposure. The pool is
    intentionally fixed in code and selected only by target_support exact mixed
    prediction loss; target_eval remains final-eval only.
    """
    if int(K) == 4:
        return [
            {
                "candidate_id": "k4_conservative_balanced",
                "ridge_weighting": "cycle_variable_balanced_huber",
                "lambda": 4.0,
                "radius": 0.10,
                "alpha": 0.30,
                "rho": 0.35,
            },
            {
                "candidate_id": "k4_current_balanced",
                "ridge_weighting": "cycle_variable_balanced_huber",
                "lambda": 2.0,
                "radius": 0.18,
                "alpha": 0.40,
                "rho": 0.50,
            },
            {
                "candidate_id": "k4_stronger_balanced",
                "ridge_weighting": "cycle_variable_balanced_huber",
                "lambda": 1.0,
                "radius": 0.22,
                "alpha": 0.45,
                "rho": 0.55,
            },
        ]
    if int(K) == 12:
        return [
            {
                "candidate_id": "k12_conservative_global",
                "ridge_weighting": "global_pixel_l2",
                "lambda": 2.0,
                "radius": 0.18,
                "alpha": 0.45,
                "rho": 0.45,
            },
            {
                "candidate_id": "k12_current_global",
                "ridge_weighting": "global_pixel_l2",
                "lambda": 1.0,
                "radius": 0.28,
                "alpha": 0.60,
                "rho": 0.65,
            },
            {
                "candidate_id": "k12_balanced_huber",
                "ridge_weighting": "cycle_variable_balanced_huber",
                "lambda": 1.0,
                "radius": 0.28,
                "alpha": 0.60,
                "rho": 0.60,
            },
            {
                "candidate_id": "k12_medium_global",
                "ridge_weighting": "global_pixel_l2",
                "lambda": 2.0,
                "radius": 0.22,
                "alpha": 0.50,
                "rho": 0.50,
            },
        ]
    return []


def diagnostic_linearized_coeff_ridge_v11_support_pool(K: int) -> List[Dict[str, Any]]:
    """Return the fixed v11 conservative leave-one-cycle-out support pool.

    v11 keeps the v10 closed-form coeff-only surface, but it treats support loss
    as a noisy model-selection signal: candidate choice and gates use
    support-internal held-out cycles and a raw physical WRMSE objective.
    """
    if int(K) == 4:
        return [
            {
                "candidate_id": "k4_eb_shrink_balanced",
                "ridge_weighting": "cycle_variable_balanced_huber",
                "lambda": 8.0,
                "radius": 0.06,
                "alpha": 0.20,
                "rho": 0.20,
            },
            {
                "candidate_id": "k4_l2sp_balanced",
                "ridge_weighting": "cycle_variable_balanced_huber",
                "lambda": 4.0,
                "radius": 0.08,
                "alpha": 0.25,
                "rho": 0.25,
            },
            {
                "candidate_id": "k4_wiseft_balanced",
                "ridge_weighting": "cycle_variable_balanced_huber",
                "lambda": 2.0,
                "radius": 0.10,
                "alpha": 0.30,
                "rho": 0.30,
            },
        ]
    if int(K) == 12:
        return [
            {
                "candidate_id": "k12_eb_shrink_global",
                "ridge_weighting": "global_pixel_l2",
                "lambda": 6.0,
                "radius": 0.10,
                "alpha": 0.30,
                "rho": 0.30,
            },
            {
                "candidate_id": "k12_l2sp_balanced",
                "ridge_weighting": "cycle_variable_balanced_huber",
                "lambda": 4.0,
                "radius": 0.14,
                "alpha": 0.35,
                "rho": 0.35,
            },
            {
                "candidate_id": "k12_wiseft_global",
                "ridge_weighting": "global_pixel_l2",
                "lambda": 2.0,
                "radius": 0.16,
                "alpha": 0.40,
                "rho": 0.40,
            },
        ]
    return []


def diagnostic_support_gain_v13_k12_calibration_pool(K: int) -> List[Dict[str, Any]]:
    """Return the fixed v13 K12 aggressive support-only calibration pool."""
    if int(K) != 12:
        return []
    return [
        {
            "candidate_id": "k12_v12_global_alpha",
            "candidate_type": "alpha_global_grid",
            "alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
            "calibration_dof": 1.0,
        },
        {
            "candidate_id": "k12_alpha2d_fine",
            "candidate_type": "alpha2d_grid",
            "alpha_grid_surface": [0.35, 0.45, 0.50, 0.60, 0.75, 0.90, 1.00],
            "alpha_grid_rootzone": [0.35, 0.45, 0.50, 0.60, 0.75, 0.90, 1.00],
            "calibration_dof": 2.0,
        },
        {
            "candidate_id": "k12_global_affine_light",
            "candidate_type": "global_affine",
            "ridge_lambda": 0.1,
            "shrinkage_strength": 0.25,
            "calibration_dof": 4.0,
        },
        {
            "candidate_id": "k12_global_affine_stronger",
            "candidate_type": "global_affine",
            "ridge_lambda": 0.05,
            "shrinkage_strength": 0.10,
            "calibration_dof": 4.0,
        },
        {
            "candidate_id": "k12_seasonal_affine_light",
            "candidate_type": "seasonal_affine",
            "ridge_lambda": 0.1,
            "shrinkage_strength": 0.25,
            "season_shrinkage_strength": 8.0,
            "calibration_dof": 4.0,
        },
        {
            "candidate_id": "k12_seasonal_affine_aggressive",
            "candidate_type": "seasonal_affine",
            "ridge_lambda": 0.05,
            "shrinkage_strength": 0.10,
            "season_shrinkage_strength": 4.0,
            "calibration_dof": 4.0,
        },
        {
            "candidate_id": "k12_alpha2d_plus_global_affine",
            "candidate_type": "alpha2d_plus_global_affine",
            "alpha_grid_surface": [0.35, 0.45, 0.50, 0.60, 0.75, 0.90, 1.00],
            "alpha_grid_rootzone": [0.35, 0.45, 0.50, 0.60, 0.75, 0.90, 1.00],
            "ridge_lambda": 0.1,
            "shrinkage_strength": 0.25,
            "calibration_dof": 6.0,
        },
    ]


def diagnostic_support_gain_v13_defaults(K: int, strength: str = "strong") -> Dict[str, Any]:
    """Return parse-time defaults for the v13 aggressive K12 calibration pool."""
    del strength
    pool = diagnostic_support_gain_v13_k12_calibration_pool(K)
    if int(K) == 4:
        return {
            "strength": "v12_k4_reference",
            "adaptation_steps": 0,
            "lr": 1e-3,
            "anchor_alpha": 0.0,
            "adapt_mix_rho": 1.0,
            "adapt_scope": "none",
            "adapt_solver": "adamw",
            "support_gate": "off",
            "support_gate_min_delta": 0.0,
            "support_gate_rootzone_tolerance": 0.0,
            "support_loss_reduction": "cycle_balanced",
            "freeze_monthly_gain": True,
            "support_candidate_pool": [],
            "support_selection_objective": "v12_global_residual_gain_reference_for_v13",
            "target_eval_usage": "final_eval_only_no_selection",
            "adaptation_step_policy_source": "diagnostic_support_gain_v13_k4_uses_v12_no_update_gain_profile",
            "source_anchor_hyperparameter_source": (
                "diagnostic_support_gain_v12_nested_cv_checkpoint_no_harm_alpha_grid"
            ),
        }
    return {
        "strength": "k12_aggressive_fixed_calibration_pool",
        "adaptation_steps": 0,
        "lr": 1e-3,
        "anchor_alpha": 0.0,
        "adapt_mix_rho": 1.0,
        "adapt_scope": "none",
        "adapt_solver": "adamw",
        "support_gate": "off",
        "support_gate_min_delta": 0.0,
        "support_gate_rootzone_tolerance": 5e-6,
        "support_loss_reduction": "cycle_balanced",
        "freeze_monthly_gain": True,
        "support_candidate_pool": [dict(candidate) for candidate in pool],
        "support_selection_objective": "k12_aggressive_nested_cv_calibration_pool_support_only",
        "support_gate_cycle_improvement_min_fraction": 0.5,
        "k12_reference_policy": "k4_safe_nested_reference" if int(K) == 12 else "",
        "target_eval_usage": "final_eval_only_no_selection",
        "adaptation_step_policy_source": "diagnostic_support_gain_v13_k12_aggressive_calibration_pool_no_adam_steps",
        "source_anchor_hyperparameter_source": (
            "diagnostic_support_gain_v13_k12_aggressive_calibration_pool_fixed_exploratory_defaults"
        ),
    }


def diagnostic_finetune_support_gain_v14_defaults(K: int, strength: str = "strong") -> Dict[str, Any]:
    """Return fixed defaults for fine-tune plus support-gain diagnostic runs.

    v14 keeps the existing support-label calibration idea, but unlike v13 it
    first performs a lightweight target-support parameter update. The update is
    restricted to adapter coefficient residuals and is still gated by
    support-only evidence; target_eval remains final-evaluation only.
    """
    defaults = dict(diagnostic_safe_operator_v5_defaults(K, strength))
    defaults.update(
        {
            "adapt_solver": "adamw",
            "adapt_mix_rho": 1.0,
            "support_gain_post_finetune": True,
            "support_gain_post_finetune_policy": "v14_nested_cv_reject_gain_to_identity",
            "support_selection_objective": (
                "target_support_parameter_finetune_then_nested_cv_support_gain"
            ),
            "target_eval_usage": "final_eval_only_no_selection",
            "adaptation_step_policy_source": (
                "diagnostic_finetune_support_gain_v14_nested_fixed_coeff_only_adamw"
            ),
            "source_anchor_hyperparameter_source": (
                "diagnostic_finetune_support_gain_v14_nested_support_only_coeff_finetune_plus_gain"
            ),
        }
    )
    return defaults


def diagnostic_linearized_coeff_ridge_v10_defaults(K: int, strength: str = "strong") -> Dict[str, Any]:
    """Return parse-time defaults for the v10 support-pool diagnostic."""
    del strength
    pool = diagnostic_linearized_coeff_ridge_v10_support_pool(K)
    if not pool:
        return {
            "strength": "fixed_pool",
            "adaptation_steps": 0,
            "lr": 0.0,
            "anchor_alpha": 0.0,
            "adapt_mix_rho": 1.0,
            "adapt_scope": "none",
            "adapt_solver": "adamw",
            "support_gate": "off",
            "support_gate_min_delta": 0.0,
            "support_gate_rootzone_tolerance": 0.0,
            "support_loss_reduction": "cycle_balanced",
            "freeze_monthly_gain": True,
            "trust_region_mode": "none",
            "trust_prompt_radius": 0.0,
            "trust_gain_radius": 0.0,
            "trust_coeff_radius": 0.0,
            "trust_spatial_radius": 0.0,
            "ridge_lambda": 0.0,
            "ridge_clip_coeff_norm": 0.0,
            "ridge_trust_region_radius": 0.0,
            "ridge_max_feature_pixels": 20000,
            "ridge_standardize_features": True,
            "ridge_weighting": "global_pixel_l2",
            "support_candidate_pool": [],
            "support_gate_cycle_improvement_min_fraction": 0.0,
            "adaptation_step_policy_source": (
                "diagnostic_linearized_coeff_ridge_v10_support_pool_no_adam_steps"
            ),
            "source_anchor_hyperparameter_source": (
                "diagnostic_linearized_coeff_ridge_v10_support_pool_nested_fixed_exploratory_defaults"
            ),
        }
    first = dict(pool[0])
    return {
        "strength": "fixed_pool",
        "adaptation_steps": 0,
        "lr": 0.0,
        "anchor_alpha": float(first["alpha"]),
        "adapt_mix_rho": float(first["rho"]),
        "adapt_scope": "coeff_only",
        "adapt_solver": "ridge_coeff",
        "support_gate": "auto",
        "support_gate_min_delta": 1e-4 if int(K) == 4 else 1e-3,
        "support_gate_rootzone_tolerance": 0.0,
        "support_loss_reduction": "cycle_balanced",
        "freeze_monthly_gain": True,
        "trust_region_mode": "groupwise",
        "trust_prompt_radius": 0.0,
        "trust_gain_radius": 0.0,
        "trust_coeff_radius": float(first["radius"]),
        "trust_spatial_radius": 0.0,
        "ridge_lambda": float(first["lambda"]),
        "ridge_clip_coeff_norm": float(first["radius"]),
        "ridge_trust_region_radius": float(first["radius"]),
        "ridge_max_feature_pixels": 20000,
        "ridge_standardize_features": True,
        "ridge_weighting": str(first["ridge_weighting"]),
        "support_candidate_pool": [dict(candidate) for candidate in pool],
        "support_gate_cycle_improvement_min_fraction": 0.75 if int(K) == 4 else (8.0 / 12.0),
        "support_selection_objective": "exact_mixed_target_support_only",
        "k12_reference_policy": "k4_safe_nested_reference" if int(K) == 12 else "",
        "target_eval_usage": "final_eval_only_no_selection",
        "adaptation_step_policy_source": (
            "diagnostic_linearized_coeff_ridge_v10_support_pool_no_adam_steps"
        ),
        "source_anchor_hyperparameter_source": (
            "diagnostic_linearized_coeff_ridge_v10_support_pool_nested_fixed_exploratory_defaults"
        ),
    }


def diagnostic_linearized_coeff_ridge_v11_defaults(K: int, strength: str = "strong") -> Dict[str, Any]:
    """Return parse-time defaults for the v11 LOOCV support-pool diagnostic."""
    del strength
    pool = diagnostic_linearized_coeff_ridge_v11_support_pool(K)
    if not pool:
        defaults = diagnostic_linearized_coeff_ridge_v10_defaults(K)
        defaults.update(
            {
                "support_candidate_pool": [],
                "support_selection_objective": "loocv_mixed_raw_increment_wrmse_target_support_only",
                "support_gate_cycle_improvement_min_fraction": 0.0,
                "source_anchor_hyperparameter_source": (
                    "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested_fixed_exploratory_defaults"
                ),
                "adaptation_step_policy_source": (
                    "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_no_adam_steps"
                ),
            }
        )
        return defaults
    first = dict(pool[0])
    return {
        "strength": "loocv_fixed_pool",
        "adaptation_steps": 0,
        "lr": 0.0,
        "anchor_alpha": float(first["alpha"]),
        "adapt_mix_rho": float(first["rho"]),
        "adapt_scope": "coeff_only",
        "adapt_solver": "ridge_coeff",
        "support_gate": "auto",
        "support_gate_min_delta": 5e-6 if int(K) == 4 else 2e-6,
        "support_gate_rootzone_tolerance": 0.0,
        "support_loss_reduction": "cycle_balanced",
        "freeze_monthly_gain": True,
        "trust_region_mode": "groupwise",
        "trust_prompt_radius": 0.0,
        "trust_gain_radius": 0.0,
        "trust_coeff_radius": float(first["radius"]),
        "trust_spatial_radius": 0.0,
        "ridge_lambda": float(first["lambda"]),
        "ridge_clip_coeff_norm": float(first["radius"]),
        "ridge_trust_region_radius": float(first["radius"]),
        "ridge_max_feature_pixels": 20000,
        "ridge_standardize_features": True,
        "ridge_weighting": str(first["ridge_weighting"]),
        "support_candidate_pool": [dict(candidate) for candidate in pool],
        "support_gate_cycle_improvement_min_fraction": 1.0 if int(K) == 4 else (2.0 / 3.0),
        "support_selection_objective": "loocv_mixed_raw_increment_wrmse_target_support_only",
        "k12_reference_policy": "k4_safe_nested_reference" if int(K) == 12 else "",
        "target_eval_usage": "final_eval_only_no_selection",
        "adaptation_step_policy_source": (
            "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_no_adam_steps"
        ),
        "source_anchor_hyperparameter_source": (
            "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested_fixed_exploratory_defaults"
        ),
    }


def _v10_candidate_to_ridge_kwargs(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ridge_lambda": float(candidate["lambda"]),
        "ridge_clip_coeff_norm": float(candidate["radius"]),
        "ridge_trust_region_radius": float(candidate["radius"]),
        "ridge_weighting": str(candidate["ridge_weighting"]),
        "anchor_alpha": float(candidate["alpha"]),
        "adapt_mix_rho": float(candidate["rho"]),
    }


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
    explicit_adapt_mix_rho = getattr(args, "adapt_mix_rho", None)
    explicit_adapt_mix_rho_provided = (
        bool(getattr(args, "adapt_mix_rho_explicit", False))
        and explicit_adapt_mix_rho is not None
    )
    diagnostic_direct = (
        int(args.K) > 0
        and str(getattr(args, "stage3_kshot_mode", ""))
        in {
            "diagnostic_direct_kshot",
            "diagnostic_direct_kshot_v2",
            "diagnostic_conservative_kshot_v3",
            "diagnostic_support_gain_v1",
            "diagnostic_support_gain_v2",
            "diagnostic_support_gain_v3_stable",
            "diagnostic_support_gain_v4_nested_stable",
            "diagnostic_support_gain_v12_nested_cv",
            "diagnostic_support_gain_v13_k12_aggressive_calibration_pool",
            "diagnostic_finetune_support_gain_v14_nested",
            "diagnostic_support_affine_v1_nested",
            "diagnostic_safe_operator_v5_nested",
            "diagnostic_linearized_coeff_ridge_v6_nested",
            "diagnostic_linearized_coeff_ridge_v7_balanced_nested",
            "diagnostic_linearized_coeff_ridge_v8_hybrid_nested",
            "diagnostic_linearized_coeff_ridge_v9_guarded_nested",
            "diagnostic_linearized_coeff_ridge_v10_support_pool_nested",
            "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested",
        }
        and bool(getattr(args, "stage3_kshot_mode_explicit", False))
        and not args.safe_policy_json
    )
    args.policy_source = "diagnostic_direct_target_support" if diagnostic_direct else "preregistered_default"
    args.target_eval_usage = "final_eval_only_no_selection"
    args.source_episode_regions = []
    args.source_policy_candidate_id = ""
    args.source_policy_guard_config_hash = ""
    if int(args.K) == 0:
        if explicit_adapt_mix_rho_provided and abs(float(explicit_adapt_mix_rho) - 1.0) > 1e-12:
            parser.error("K=0 requires --adapt_mix_rho 1.0")
        args.rho_policy = "not_applicable_k0"
        args.adapt_mix_rho = 1.0
    elif diagnostic_direct:
        mode = str(getattr(args, "stage3_kshot_mode", ""))
        if mode == "diagnostic_conservative_kshot_v3":
            strength = str(getattr(args, "diagnostic_kshot_strength", "strong"))
            defaults = diagnostic_conservative_kshot_v3_defaults(args.K, strength)
            args.rho_policy = f"diagnostic_conservative_fixed_{defaults['adapt_mix_rho']}"
            args.adapt_mix_rho = float(defaults["adapt_mix_rho"])
            args.source_anchor_hyperparameter_source = str(defaults["source_anchor_hyperparameter_source"])
        elif mode == "diagnostic_finetune_support_gain_v14_nested":
            strength = str(getattr(args, "diagnostic_kshot_strength", "strong"))
            defaults = diagnostic_finetune_support_gain_v14_defaults(args.K, strength)
            args.rho_policy = f"diagnostic_finetune_support_gain_v14_fixed_{defaults['adapt_mix_rho']}"
            args.adapt_mix_rho = float(defaults["adapt_mix_rho"])
            args.source_anchor_hyperparameter_source = str(defaults["source_anchor_hyperparameter_source"])
        elif mode == "diagnostic_safe_operator_v5_nested":
            strength = str(getattr(args, "diagnostic_kshot_strength", "strong"))
            defaults = diagnostic_safe_operator_v5_defaults(args.K, strength)
            args.rho_policy = f"diagnostic_safe_operator_v5_fixed_{defaults['adapt_mix_rho']}"
            args.adapt_mix_rho = float(defaults["adapt_mix_rho"])
            args.source_anchor_hyperparameter_source = str(defaults["source_anchor_hyperparameter_source"])
        elif mode == "diagnostic_linearized_coeff_ridge_v6_nested":
            strength = str(getattr(args, "diagnostic_kshot_strength", "strong"))
            defaults = diagnostic_linearized_coeff_ridge_v6_defaults(args.K, strength)
            args.rho_policy = f"diagnostic_linearized_coeff_ridge_v6_fixed_{defaults['adapt_mix_rho']}"
            args.adapt_mix_rho = float(defaults["adapt_mix_rho"])
            args.source_anchor_hyperparameter_source = str(defaults["source_anchor_hyperparameter_source"])
        elif mode == "diagnostic_linearized_coeff_ridge_v7_balanced_nested":
            strength = str(getattr(args, "diagnostic_kshot_strength", "strong"))
            defaults = diagnostic_linearized_coeff_ridge_v7_defaults(args.K, strength)
            args.rho_policy = f"diagnostic_linearized_coeff_ridge_v7_balanced_fixed_{defaults['adapt_mix_rho']}"
            args.adapt_mix_rho = float(defaults["adapt_mix_rho"])
            args.source_anchor_hyperparameter_source = str(defaults["source_anchor_hyperparameter_source"])
        elif mode == "diagnostic_linearized_coeff_ridge_v8_hybrid_nested":
            strength = str(getattr(args, "diagnostic_kshot_strength", "strong"))
            defaults = diagnostic_linearized_coeff_ridge_v8_hybrid_defaults(args.K, strength)
            args.rho_policy = f"diagnostic_linearized_coeff_ridge_v8_hybrid_fixed_{defaults['adapt_mix_rho']}"
            args.adapt_mix_rho = float(defaults["adapt_mix_rho"])
            args.source_anchor_hyperparameter_source = str(defaults["source_anchor_hyperparameter_source"])
        elif mode == "diagnostic_linearized_coeff_ridge_v9_guarded_nested":
            strength = str(getattr(args, "diagnostic_kshot_strength", "strong"))
            defaults = diagnostic_linearized_coeff_ridge_v9_guarded_defaults(args.K, strength)
            args.rho_policy = f"diagnostic_linearized_coeff_ridge_v9_guarded_fixed_{defaults['adapt_mix_rho']}"
            args.adapt_mix_rho = float(defaults["adapt_mix_rho"])
            args.source_anchor_hyperparameter_source = str(defaults["source_anchor_hyperparameter_source"])
        elif mode == "diagnostic_linearized_coeff_ridge_v10_support_pool_nested":
            defaults = diagnostic_linearized_coeff_ridge_v10_defaults(args.K)
            args.rho_policy = "diagnostic_linearized_coeff_ridge_v10_support_pool_selected_by_support"
            args.adapt_mix_rho = float(defaults["adapt_mix_rho"])
            args.source_anchor_hyperparameter_source = str(defaults["source_anchor_hyperparameter_source"])
        elif mode == "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested":
            defaults = diagnostic_linearized_coeff_ridge_v11_defaults(args.K)
            args.rho_policy = "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_selected_by_support_cv"
            args.adapt_mix_rho = float(defaults["adapt_mix_rho"])
            args.source_anchor_hyperparameter_source = str(defaults["source_anchor_hyperparameter_source"])
        elif mode in {
            "diagnostic_support_gain_v2",
            "diagnostic_support_gain_v3_stable",
            "diagnostic_support_gain_v4_nested_stable",
            "diagnostic_support_gain_v12_nested_cv",
            "diagnostic_support_gain_v13_k12_aggressive_calibration_pool",
            "diagnostic_finetune_support_gain_v14_nested",
            "diagnostic_support_affine_v1_nested",
        }:
            args.rho_policy = f"{mode}_checkpoint_fixed_1.0"
            args.adapt_mix_rho = 1.0
            if mode == "diagnostic_support_affine_v1_nested":
                args.source_anchor_hyperparameter_source = (
                    "diagnostic_support_affine_v1_nested_support_only_ridge_affine"
                )
            elif mode == "diagnostic_support_gain_v13_k12_aggressive_calibration_pool":
                args.source_anchor_hyperparameter_source = str(
                    diagnostic_support_gain_v13_defaults(
                        args.K,
                        str(getattr(args, "diagnostic_kshot_strength", "strong")),
                    )["source_anchor_hyperparameter_source"]
                )
            elif mode == "diagnostic_support_gain_v12_nested_cv":
                args.source_anchor_hyperparameter_source = (
                    "diagnostic_support_gain_v12_nested_cv_checkpoint_no_harm_alpha_grid"
                )
            elif mode == "diagnostic_support_gain_v4_nested_stable":
                args.source_anchor_hyperparameter_source = (
                    "diagnostic_support_gain_v4_nested_stable_checkpoint_stability_alpha_grid"
                )
            elif mode == "diagnostic_support_gain_v3_stable":
                args.source_anchor_hyperparameter_source = (
                    "diagnostic_support_gain_v3_stable_checkpoint_stability_alpha_grid"
                )
            else:
                args.source_anchor_hyperparameter_source = "diagnostic_support_gain_v2_checkpoint_fixed_alpha_grid"
        else:
            args.rho_policy = "diagnostic_direct_fixed_1.0"
            args.adapt_mix_rho = 1.0
        if mode == "diagnostic_direct_kshot_v2":
            strength = str(getattr(args, "diagnostic_kshot_strength", "strong"))
            args.source_anchor_hyperparameter_source = f"diagnostic_direct_kshot_v2_{strength}_fixed_defaults"
        elif mode == "diagnostic_direct_kshot":
            args.source_anchor_hyperparameter_source = "diagnostic_direct_kshot_fixed_defaults"
    else:
        args.rho_policy = "diagnostic_no_source_safe_policy_json"
        args.adapt_mix_rho = 0.0
    if not args.safe_policy_json:
        if bool(args.require_safe_policy_json_for_kshot) and int(args.K) > 0:
            parser.error("--require_safe_policy_json_for_kshot requires --safe_policy_json for K>0")
        if explicit_adapt_mix_rho_provided:
            args.adapt_mix_rho = float(explicit_adapt_mix_rho)
            if int(args.K) > 0:
                args.rho_policy = f"explicit_wrapper_{float(explicit_adapt_mix_rho):g}"
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
        "ridge_lambda",
        "ridge_clip_coeff_norm",
        "ridge_trust_region_radius",
        "ridge_max_feature_pixels",
        "ridge_standardize_features",
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
        "ridge_lambda",
        "ridge_clip_coeff_norm",
        "ridge_trust_region_radius",
    ):
        if key in selected:
            setattr(args, key, float(selected[key]))
    if "ridge_max_feature_pixels" in selected:
        args.ridge_max_feature_pixels = int(selected["ridge_max_feature_pixels"])
    if "ridge_standardize_features" in selected:
        args.ridge_standardize_features = bool(selected["ridge_standardize_features"])
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
    args.adaptation_step_policy_source = "source_side_episode_calibration"
    args.resolved_mode_defaults = {
        "source": "safe_policy_json",
        "selected_keys": sorted(selected),
        "K": int(args.K),
        "adaptation_setting": str(args.adaptation_setting),
    }
    if bool(args.require_safe_policy_json_for_kshot) and int(args.K) > 0:
        nonzero_violations = []
        if args.adapt_scope == "none":
            nonzero_violations.append("adapt_scope=none")
        if int(args.adaptation_steps) <= 0 and str(args.adapt_solver) != "ridge_coeff":
            nonzero_violations.append(f"adaptation_steps={args.adaptation_steps}")
        if float(args.lr) <= 0.0 and str(args.adapt_solver) != "ridge_coeff":
            nonzero_violations.append(f"lr={args.lr}")
        if str(args.adapt_solver) == "ridge_coeff" and float(args.ridge_trust_region_radius) <= 0.0:
            nonzero_violations.append(f"ridge_trust_region_radius={args.ridge_trust_region_radius}")
        if float(args.adapt_mix_rho) <= 0.0:
            nonzero_violations.append(f"adapt_mix_rho={args.adapt_mix_rho}")
        if nonzero_violations:
            parser.error(
                "paper-facing K-shot requires a source-side SAFE policy with a nonzero "
                "target_support update and nonzero final mix; got "
                + ", ".join(nonzero_violations)
            )
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
    anchor_alpha_provided = _arg_was_provided("--anchor_alpha")
    adapt_mix_rho_provided = _arg_was_provided("--adapt_mix_rho")
    adaptation_steps_provided = _arg_was_provided("--adaptation_steps")
    lr_provided = _arg_was_provided("--lr")
    support_gate_provided = _arg_was_provided("--support_gate")
    support_loss_reduction_provided = _arg_was_provided("--support_loss_reduction")
    stage3_kshot_mode_explicit = _arg_was_provided("--stage3_kshot_mode") or bool(os.environ.get("STAGE3_KSHOT_MODE"))
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
        "--context_tta_mode",
        type=str,
        default=CONTEXT_TTA_NONE,
        choices=sorted(CONTEXT_TTA_MODES),
        help=(
            "Frozen target_context input-side TTA state. prompt_feature_alignment "
            "hashes target-context input embeddings and source-stat provenance; "
            "target_eval never updates this state."
        ),
    )
    parser.add_argument(
        "--context_tta_residual_scale",
        type=float,
        default=float(os.environ.get("CONTEXT_TTA_RESIDUAL_SCALE", "0.05")),
        help=(
            "Bound for context_prompt_residual_shift. Uses target_context input-side "
            "statistics only; default preserves the original conservative shift."
        ),
    )
    parser.add_argument(
        "--context_tta_residual_clip_l2",
        type=float,
        default=float(os.environ.get("CONTEXT_TTA_RESIDUAL_CLIP_L2", "0.0")),
        help="Optional L2 cap for context_prompt_residual_shift; 0 disables clipping.",
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
        "--adapt_mix_rho",
        type=float,
        default=None,
        help=(
            "Final-eval blend weight for the adapted posterior. Stage-3 wrappers may "
            "pass this after source-side policy resolution; K=0 resolves to 1.0."
        ),
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
        "--k4_reference_checkpoint",
        type=str,
        default=os.environ.get("K4_REFERENCE_CHECKPOINT", ""),
        help=(
            "Diagnostic v5 only: run-local K4 adapted checkpoint used as the "
            "support-only K12 fallback reference. target_eval is never consulted."
        ),
    )
    parser.add_argument(
        "--stage3_kshot_mode",
        type=str,
        default=os.environ.get("STAGE3_KSHOT_MODE", "paper_safe"),
        choices=list(STAGE3_KSHOT_MODES),
        help=(
            "K-shot run mode. paper_safe requires source-side SAFE policy for K>0; "
            "diagnostic_direct_kshot uses fixed target_support updates and is never paper-facing; "
            "diagnostic_conservative_kshot_v3 restricts diagnostics to small coefficient updates."
        ),
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
        "--ridge_weighting",
        type=str,
        default="global_pixel_l2",
        choices=list(RIDGE_WEIGHTINGS),
        help=(
            "Support-only weighting used to build the ridge coefficient design. "
            "cycle_variable_balanced_huber matches the support gate's cycle-balanced "
            "spirit and bounds large support residual influence."
        ),
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
    args.adaptation_step_policy_source = "legacy_default_steps_for_K"
    args.resolved_mode_defaults = {
        "source": "legacy_default_steps_for_K",
        "K": int(args.K),
        "adaptation_steps": int(args.adaptation_steps),
        "lr": float(args.lr),
        "anchor_alpha": float(args.anchor_alpha),
    }
    args.stage3_kshot_mode_explicit = bool(stage3_kshot_mode_explicit)
    args.adapt_mix_rho_explicit = bool(adapt_mix_rho_provided)
    args.diagnostic_kshot_strength = str(os.environ.get("DIAGNOSTIC_KSHOT_STRENGTH", "strong")).strip().lower()
    if args.diagnostic_kshot_strength not in {"strong", "medium"}:
        parser.error("DIAGNOSTIC_KSHOT_STRENGTH must be strong or medium")
    if (
        args.stage3_kshot_mode == "paper_safe"
        and int(args.K) > 0
        and not args.safe_policy_json
    ):
        args.require_safe_policy_json_for_kshot = True
    if (
        args.stage3_kshot_mode == "diagnostic_direct_kshot"
        and int(args.K) > 0
        and not args.safe_policy_json
        and args.stage3_kshot_mode_explicit
    ):
        if not anchor_alpha_provided:
            args.anchor_alpha = 1.0
        if not support_gate_provided:
            args.support_gate = "off"
        if not support_loss_reduction_provided:
            args.support_loss_reduction = "cycle_balanced"
    if (
        args.stage3_kshot_mode == "diagnostic_direct_kshot_v2"
        and int(args.K) > 0
        and not args.safe_policy_json
        and args.stage3_kshot_mode_explicit
    ):
        v2_defaults = diagnostic_kshot_v2_defaults(args.K, args.diagnostic_kshot_strength)
        if not adapt_scope_provided:
            args.adapt_scope = str(v2_defaults["adapt_scope"])
        if not adaptation_steps_provided:
            args.adaptation_steps = int(v2_defaults["adaptation_steps"])
        if not lr_provided:
            args.lr = float(v2_defaults["lr"])
        if not anchor_alpha_provided:
            args.anchor_alpha = float(v2_defaults["anchor_alpha"])
        if not support_gate_provided:
            args.support_gate = str(v2_defaults["support_gate"])
        if not support_loss_reduction_provided:
            args.support_loss_reduction = str(v2_defaults["support_loss_reduction"])
        if not freeze_monthly_gain_provided:
            args.freeze_monthly_gain = False
        args.source_anchor_hyperparameter_source = str(v2_defaults["source_anchor_hyperparameter_source"])
    if (
        args.stage3_kshot_mode == "diagnostic_conservative_kshot_v3"
        and int(args.K) > 0
        and not args.safe_policy_json
        and args.stage3_kshot_mode_explicit
    ):
        v3_defaults = diagnostic_conservative_kshot_v3_defaults(args.K, args.diagnostic_kshot_strength)
        if not adapt_scope_provided:
            args.adapt_scope = str(v3_defaults["adapt_scope"])
        if not adaptation_steps_provided:
            args.adaptation_steps = int(v3_defaults["adaptation_steps"])
        if not lr_provided:
            args.lr = float(v3_defaults["lr"])
        if not anchor_alpha_provided:
            args.anchor_alpha = float(v3_defaults["anchor_alpha"])
        if not support_gate_provided:
            args.support_gate = str(v3_defaults["support_gate"])
        if not support_loss_reduction_provided:
            args.support_loss_reduction = str(v3_defaults["support_loss_reduction"])
        if not freeze_monthly_gain_provided:
            args.freeze_monthly_gain = bool(v3_defaults["freeze_monthly_gain"])
        if args.trust_region_mode == "none":
            args.trust_region_mode = str(v3_defaults["trust_region_mode"])
        if float(args.trust_prompt_radius) == 0.0:
            args.trust_prompt_radius = float(v3_defaults["trust_prompt_radius"])
        if float(args.trust_gain_radius) == 0.0:
            args.trust_gain_radius = float(v3_defaults["trust_gain_radius"])
        if float(args.trust_coeff_radius) == 0.0:
            args.trust_coeff_radius = float(v3_defaults["trust_coeff_radius"])
        if float(args.trust_spatial_radius) == 0.0:
            args.trust_spatial_radius = float(v3_defaults["trust_spatial_radius"])
        if float(args.support_gate_min_delta) == 0.0:
            args.support_gate_min_delta = float(v3_defaults["support_gate_min_delta"])
        if float(args.support_gate_rootzone_tolerance) == 0.0:
            args.support_gate_rootzone_tolerance = float(v3_defaults["support_gate_rootzone_tolerance"])
        args.stage3_posterior_policy = "conservative_coeff_posterior"
        args.source_anchor_hyperparameter_source = str(v3_defaults["source_anchor_hyperparameter_source"])
    if (
        args.stage3_kshot_mode == "diagnostic_safe_operator_v5_nested"
        and int(args.K) > 0
        and not args.safe_policy_json
        and args.stage3_kshot_mode_explicit
    ):
        v5_defaults = diagnostic_safe_operator_v5_defaults(args.K, args.diagnostic_kshot_strength)
        if not adapt_scope_provided:
            args.adapt_scope = str(v5_defaults["adapt_scope"])
        if not adaptation_steps_provided:
            args.adaptation_steps = int(v5_defaults["adaptation_steps"])
        if not lr_provided:
            args.lr = float(v5_defaults["lr"])
        if not anchor_alpha_provided:
            args.anchor_alpha = float(v5_defaults["anchor_alpha"])
        if not support_gate_provided:
            args.support_gate = str(v5_defaults["support_gate"])
        if not support_loss_reduction_provided:
            args.support_loss_reduction = str(v5_defaults["support_loss_reduction"])
        if not freeze_monthly_gain_provided:
            args.freeze_monthly_gain = bool(v5_defaults["freeze_monthly_gain"])
        if args.trust_region_mode == "none":
            args.trust_region_mode = str(v5_defaults["trust_region_mode"])
        if float(args.trust_prompt_radius) == 0.0:
            args.trust_prompt_radius = float(v5_defaults["trust_prompt_radius"])
        if float(args.trust_gain_radius) == 0.0:
            args.trust_gain_radius = float(v5_defaults["trust_gain_radius"])
        if float(args.trust_coeff_radius) == 0.0:
            args.trust_coeff_radius = float(v5_defaults["trust_coeff_radius"])
        if float(args.trust_spatial_radius) == 0.0:
            args.trust_spatial_radius = float(v5_defaults["trust_spatial_radius"])
        if float(args.support_gate_min_delta) == 0.0:
            args.support_gate_min_delta = float(v5_defaults["support_gate_min_delta"])
        if float(args.support_gate_rootzone_tolerance) == 0.0:
            args.support_gate_rootzone_tolerance = float(v5_defaults["support_gate_rootzone_tolerance"])
        args.stage3_posterior_policy = "conservative_coeff_posterior"
        args.source_anchor_hyperparameter_source = str(v5_defaults["source_anchor_hyperparameter_source"])
        args.adaptation_step_policy_source = "diagnostic_safe_operator_v5_nested_mode_defaults"
        args.resolved_mode_defaults = dict(v5_defaults)
    if (
        args.stage3_kshot_mode == "diagnostic_finetune_support_gain_v14_nested"
        and int(args.K) > 0
        and not args.safe_policy_json
        and args.stage3_kshot_mode_explicit
    ):
        v14_defaults = diagnostic_finetune_support_gain_v14_defaults(
            args.K,
            args.diagnostic_kshot_strength,
        )
        if not adapt_scope_provided:
            args.adapt_scope = str(v14_defaults["adapt_scope"])
        args.adapt_solver = str(v14_defaults["adapt_solver"])
        if not adaptation_steps_provided:
            args.adaptation_steps = int(v14_defaults["adaptation_steps"])
        if not lr_provided:
            args.lr = float(v14_defaults["lr"])
        if not anchor_alpha_provided:
            args.anchor_alpha = float(v14_defaults["anchor_alpha"])
        if not support_gate_provided:
            args.support_gate = str(v14_defaults["support_gate"])
        if not support_loss_reduction_provided:
            args.support_loss_reduction = str(v14_defaults["support_loss_reduction"])
        if not freeze_monthly_gain_provided:
            args.freeze_monthly_gain = bool(v14_defaults["freeze_monthly_gain"])
        if args.trust_region_mode == "none":
            args.trust_region_mode = str(v14_defaults["trust_region_mode"])
        if float(args.trust_prompt_radius) == 0.0:
            args.trust_prompt_radius = float(v14_defaults["trust_prompt_radius"])
        if float(args.trust_gain_radius) == 0.0:
            args.trust_gain_radius = float(v14_defaults["trust_gain_radius"])
        if float(args.trust_coeff_radius) == 0.0:
            args.trust_coeff_radius = float(v14_defaults["trust_coeff_radius"])
        if float(args.trust_spatial_radius) == 0.0:
            args.trust_spatial_radius = float(v14_defaults["trust_spatial_radius"])
        if float(args.support_gate_min_delta) == 0.0:
            args.support_gate_min_delta = float(v14_defaults["support_gate_min_delta"])
        if float(args.support_gate_rootzone_tolerance) == 0.0:
            args.support_gate_rootzone_tolerance = float(v14_defaults["support_gate_rootzone_tolerance"])
        args.stage3_posterior_policy = "conservative_coeff_posterior"
        args.source_anchor_hyperparameter_source = str(v14_defaults["source_anchor_hyperparameter_source"])
        args.adaptation_step_policy_source = str(v14_defaults["adaptation_step_policy_source"])
        args.resolved_mode_defaults = dict(v14_defaults)
    if (
        args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v6_nested"
        or args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v7_balanced_nested"
        or args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v8_hybrid_nested"
        or args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v9_guarded_nested"
        or args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v10_support_pool_nested"
        or args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested"
    ) and int(args.K) > 0 and not args.safe_policy_json and args.stage3_kshot_mode_explicit:
        if args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v7_balanced_nested":
            v6_defaults = diagnostic_linearized_coeff_ridge_v7_defaults(args.K, args.diagnostic_kshot_strength)
        elif args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v8_hybrid_nested":
            v6_defaults = diagnostic_linearized_coeff_ridge_v8_hybrid_defaults(
                args.K,
                args.diagnostic_kshot_strength,
            )
        elif args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v9_guarded_nested":
            v6_defaults = diagnostic_linearized_coeff_ridge_v9_guarded_defaults(
                args.K,
                args.diagnostic_kshot_strength,
            )
        elif args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v10_support_pool_nested":
            v6_defaults = diagnostic_linearized_coeff_ridge_v10_defaults(
                args.K,
                args.diagnostic_kshot_strength,
            )
        elif args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested":
            v6_defaults = diagnostic_linearized_coeff_ridge_v11_defaults(
                args.K,
                args.diagnostic_kshot_strength,
            )
        else:
            v6_defaults = diagnostic_linearized_coeff_ridge_v6_defaults(args.K, args.diagnostic_kshot_strength)
        if not adapt_scope_provided:
            args.adapt_scope = str(v6_defaults["adapt_scope"])
        args.adapt_solver = str(v6_defaults["adapt_solver"])
        if not adaptation_steps_provided:
            args.adaptation_steps = int(v6_defaults["adaptation_steps"])
        if not lr_provided:
            args.lr = float(v6_defaults["lr"])
        if not anchor_alpha_provided:
            args.anchor_alpha = float(v6_defaults["anchor_alpha"])
        if not support_gate_provided:
            args.support_gate = str(v6_defaults["support_gate"])
        if not support_loss_reduction_provided:
            args.support_loss_reduction = str(v6_defaults["support_loss_reduction"])
        if not freeze_monthly_gain_provided:
            args.freeze_monthly_gain = bool(v6_defaults["freeze_monthly_gain"])
        if args.trust_region_mode == "none":
            args.trust_region_mode = str(v6_defaults["trust_region_mode"])
        if float(args.trust_prompt_radius) == 0.0:
            args.trust_prompt_radius = float(v6_defaults["trust_prompt_radius"])
        if float(args.trust_gain_radius) == 0.0:
            args.trust_gain_radius = float(v6_defaults["trust_gain_radius"])
        if float(args.trust_coeff_radius) == 0.0:
            args.trust_coeff_radius = float(v6_defaults["trust_coeff_radius"])
        if float(args.trust_spatial_radius) == 0.0:
            args.trust_spatial_radius = float(v6_defaults["trust_spatial_radius"])
        if float(args.support_gate_min_delta) == 0.0:
            args.support_gate_min_delta = float(v6_defaults["support_gate_min_delta"])
        if float(args.support_gate_rootzone_tolerance) == 0.0:
            args.support_gate_rootzone_tolerance = float(v6_defaults["support_gate_rootzone_tolerance"])
        if float(args.ridge_lambda) == 1.0:
            args.ridge_lambda = float(v6_defaults["ridge_lambda"])
        if float(args.ridge_clip_coeff_norm) == 1.0:
            args.ridge_clip_coeff_norm = float(v6_defaults["ridge_clip_coeff_norm"])
        if float(args.ridge_trust_region_radius) == 1.0:
            args.ridge_trust_region_radius = float(v6_defaults["ridge_trust_region_radius"])
        if int(args.ridge_max_feature_pixels) == 20000:
            args.ridge_max_feature_pixels = int(v6_defaults["ridge_max_feature_pixels"])
        if not args.ridge_standardize_features:
            args.ridge_standardize_features = bool(v6_defaults["ridge_standardize_features"])
        if args.ridge_weighting == "global_pixel_l2" and "ridge_weighting" in v6_defaults:
            args.ridge_weighting = str(v6_defaults["ridge_weighting"])
        args.stage3_posterior_policy = "conservative_coeff_posterior"
        args.source_anchor_hyperparameter_source = str(v6_defaults["source_anchor_hyperparameter_source"])
        args.adaptation_step_policy_source = str(v6_defaults["adaptation_step_policy_source"])
        args.resolved_mode_defaults = dict(v6_defaults)
    if (
        args.stage3_kshot_mode
        in {
            "diagnostic_support_gain_v1",
            "diagnostic_support_gain_v2",
            "diagnostic_support_gain_v3_stable",
            "diagnostic_support_gain_v4_nested_stable",
            "diagnostic_support_gain_v12_nested_cv",
            "diagnostic_support_gain_v13_k12_aggressive_calibration_pool",
            "diagnostic_support_affine_v1_nested",
        }
        and int(args.K) > 0
        and not args.safe_policy_json
        and args.stage3_kshot_mode_explicit
    ):
        if not adapt_scope_provided:
            args.adapt_scope = "none"
        if not adaptation_steps_provided:
            args.adaptation_steps = 0
        if not anchor_alpha_provided:
            args.anchor_alpha = 0.0
        if not support_gate_provided:
            args.support_gate = "off"
        if not support_loss_reduction_provided:
            args.support_loss_reduction = "cycle_balanced"
        if not freeze_monthly_gain_provided:
            args.freeze_monthly_gain = True
        args.stage3_posterior_policy = "source_calibrated_mix"
        if args.stage3_kshot_mode == "diagnostic_support_gain_v1":
            args.source_anchor_hyperparameter_source = "diagnostic_support_gain_v1_eval_side_fixed_alpha_grid"
        elif args.stage3_kshot_mode == "diagnostic_support_gain_v13_k12_aggressive_calibration_pool":
            v13_defaults = diagnostic_support_gain_v13_defaults(
                args.K,
                args.diagnostic_kshot_strength,
            )
            args.source_anchor_hyperparameter_source = str(v13_defaults["source_anchor_hyperparameter_source"])
            args.adaptation_step_policy_source = str(v13_defaults["adaptation_step_policy_source"])
            args.resolved_mode_defaults = dict(v13_defaults)
        elif args.stage3_kshot_mode == "diagnostic_support_gain_v3_stable":
            args.source_anchor_hyperparameter_source = (
                "diagnostic_support_gain_v3_stable_checkpoint_stability_alpha_grid"
            )
        elif args.stage3_kshot_mode == "diagnostic_support_gain_v4_nested_stable":
            args.source_anchor_hyperparameter_source = (
                "diagnostic_support_gain_v4_nested_stable_checkpoint_stability_alpha_grid"
            )
        elif args.stage3_kshot_mode == "diagnostic_support_gain_v12_nested_cv":
            args.source_anchor_hyperparameter_source = (
                "diagnostic_support_gain_v12_nested_cv_checkpoint_no_harm_alpha_grid"
            )
        elif args.stage3_kshot_mode == "diagnostic_support_affine_v1_nested":
            args.source_anchor_hyperparameter_source = (
                "diagnostic_support_affine_v1_nested_support_only_ridge_affine"
            )
        else:
            args.source_anchor_hyperparameter_source = "diagnostic_support_gain_v2_checkpoint_fixed_alpha_grid"
    apply_safe_policy_to_args(args, parser)
    if (
        args.stage3_posterior_policy in {"conservative_coeff_posterior", "source_calibrated_mix"}
        and int(args.K) > 0
        and not args.safe_policy_json
        and args.stage3_kshot_mode not in {
            "diagnostic_direct_kshot_v2",
            "diagnostic_conservative_kshot_v3",
            "diagnostic_safe_operator_v5_nested",
            "diagnostic_linearized_coeff_ridge_v6_nested",
            "diagnostic_linearized_coeff_ridge_v7_balanced_nested",
            "diagnostic_linearized_coeff_ridge_v8_hybrid_nested",
            "diagnostic_linearized_coeff_ridge_v9_guarded_nested",
            "diagnostic_linearized_coeff_ridge_v10_support_pool_nested",
            "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested",
            "diagnostic_finetune_support_gain_v14_nested",
            "diagnostic_support_gain_v1",
            "diagnostic_support_gain_v2",
            "diagnostic_support_gain_v3_stable",
            "diagnostic_support_gain_v4_nested_stable",
            "diagnostic_support_gain_v12_nested_cv",
            "diagnostic_support_gain_v13_k12_aggressive_calibration_pool",
            "diagnostic_support_affine_v1_nested",
        }
    ):
        if not adapt_scope_provided:
            args.adapt_scope = "coeff_only"
        if not freeze_monthly_gain_provided:
            args.freeze_monthly_gain = True
    if not 0.0 <= float(args.adapt_mix_rho) <= 1.0:
        parser.error("--adapt_mix_rho must be in [0, 1]")
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
    if float(args.context_tta_residual_scale) < 0.0:
        parser.error("--context_tta_residual_scale must be non-negative")
    if float(args.context_tta_residual_clip_l2) < 0.0:
        parser.error("--context_tta_residual_clip_l2 must be non-negative")
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
        if int(args.K) > 0 and args.policy_source == "source_side_episode_calibration":
            args.support_gate = "off"
        else:
            args.support_gate = (
                "auto"
                if args.stage3_posterior_policy in {"conservative_coeff_posterior", "source_calibrated_mix"}
                else "off"
            )
    if (
        args.stage3_posterior_policy in {"conservative_coeff_posterior", "source_calibrated_mix"}
        and args.stage3_kshot_mode
        not in {
            "diagnostic_direct_kshot_v2",
            "diagnostic_conservative_kshot_v3",
            "diagnostic_safe_operator_v5_nested",
            "diagnostic_linearized_coeff_ridge_v6_nested",
            "diagnostic_linearized_coeff_ridge_v7_balanced_nested",
            "diagnostic_linearized_coeff_ridge_v8_hybrid_nested",
            "diagnostic_linearized_coeff_ridge_v9_guarded_nested",
            "diagnostic_linearized_coeff_ridge_v10_support_pool_nested",
            "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested",
            "diagnostic_finetune_support_gain_v14_nested",
        }
    ):
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
                and not bool(args.require_safe_policy_json_for_kshot)
                and args.policy_source == "source_side_episode_calibration"
                and "adapt_scope" in set(getattr(args, "safe_policy_selected_keys", []))
                and args.adapt_scope == "none"
                and int(args.adaptation_steps) == 0
                and abs(float(args.anchor_alpha)) <= 1e-12
            )
            diagnostic_support_gain_no_update = (
                args.stage3_kshot_mode
                in {
                    "diagnostic_support_gain_v1",
                    "diagnostic_support_gain_v2",
                    "diagnostic_support_gain_v3_stable",
                    "diagnostic_support_gain_v4_nested_stable",
                    "diagnostic_support_gain_v12_nested_cv",
                    "diagnostic_support_gain_v13_k12_aggressive_calibration_pool",
                    "diagnostic_support_affine_v1_nested",
                }
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
                and not diagnostic_support_gain_no_update
            ):
                parser.error(
                    f"--stage3_posterior_policy {args.stage3_posterior_policy} requires "
                    "--adapt_scope coeff_only for K>0 unless source-side SAFE policy "
                    "explicitly selects coeff_gain or a diagnostic no-update path, or --audit_identity requests "
                    "a no-update path"
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
        hyper_source_manifold_guard=bool(source_config.get("hyper_source_manifold_guard", False)),
        hyper_source_manifold_guard_strength=float(source_config.get("hyper_source_manifold_guard_strength", 0.25)),
        hyper_source_manifold_guard_distance_key=source_config.get(
            "hyper_source_manifold_guard_distance_key",
            SOURCE_MANIFOLD_DISTANCE_KEY,
        ),
        hyper_source_manifold_guard_min_multiplier=float(
            source_config.get("hyper_source_manifold_guard_min_multiplier", 0.0)
        ),
        source_manifold_guard_calibration=source_config.get("source_manifold_guard_calibration", "disabled"),
        hyper_source_trust_routing=bool(source_config.get("hyper_source_trust_routing", False)),
        hyper_source_trust_strength=float(source_config.get("hyper_source_trust_strength", 0.0)),
        hyper_source_trust_top_m=int(source_config.get("hyper_source_trust_top_m", 4)),
        hyper_source_trust_variable_gate=bool(source_config.get("hyper_source_trust_variable_gate", False)),
        hyper_phys_agreement_guard=bool(source_config.get("hyper_phys_agreement_guard", False)),
        hyper_phys_agreement_guard_strength=float(source_config.get("hyper_phys_agreement_guard_strength", 1.0)),
        hyper_phys_agreement_guard_min_multiplier=float(
            source_config.get("hyper_phys_agreement_guard_min_multiplier", 0.0)
        ),
        hyper_phys_agreement_guard_risk_rule=source_config.get("hyper_phys_agreement_guard_risk_rule", "or"),
        hyper_phys_context_modulation=bool(source_config.get("hyper_phys_context_modulation", False)),
        hyper_phys_delta_scale=float(source_config.get("hyper_phys_delta_scale", 0.25)),
        hyper_phys_gate_init=float(source_config.get("hyper_phys_gate_init", 0.90)),
        hyper_operator_droppath_p=float(source_config.get("hyper_operator_droppath_p", 0.10)),
        phys_context_source=source_config.get("phys_context_source", "raw_input_side_da_diagnostics"),
        hyper_phys_gain_basis_residual=bool(source_config.get("hyper_phys_gain_basis_residual", False)),
        hyper_phys_gain_basis_coeff_scale=float(source_config.get("hyper_phys_gain_basis_coeff_scale", 0.05)),
        hyper_phys_gain_basis_residual_clip=float(source_config.get("hyper_phys_gain_basis_residual_clip", 0.25)),
        hyper_phys_gain_basis_beta_init=float(source_config.get("hyper_phys_gain_basis_beta_init", 0.50)),
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
    context_tta_mode: str = CONTEXT_TTA_NONE,
    context_tta_residual_scale: float = 0.05,
    context_tta_residual_clip_l2: float = 0.0,
) -> Dict[str, Any]:
    return build_target_context_prompt_state(
        samples=samples,
        prompt_encoder=state.prompt_encoder,
        normalize_x=lambda x: _normalize_x(x, state.normalization),
        target_region_embedding=_target_region_embedding(state, target_region, device),
        device=device,
        context_hash=context_hash,
        context_encoder=state.source_config.get("context_encoder", "current_mean_std"),
        source_prompt_manifold_guard_state=normalize_source_prompt_manifold_guard_state(
            state.source_checkpoint.get("source_prompt_manifold_guard_state")
            or state.source_config.get("source_prompt_manifold_guard_state")
        ),
        source_trust_bank_state=normalize_hyperda_source_trust_bank_state(
            state.source_checkpoint.get("source_trust_bank_state")
            or state.source_checkpoint.get("hyperda_source_trust_bank_state")
            or state.source_config.get("source_trust_bank_state")
            or state.source_config.get("hyperda_source_trust_bank_state")
        ),
        context_tta_mode=context_tta_mode,
        context_tta_residual_scale=float(context_tta_residual_scale),
        context_tta_residual_clip_l2=float(context_tta_residual_clip_l2),
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


def source_trust_bank_for_few_shot_state(state: FewShotAdaptationState) -> Dict[str, Any] | None:
    return normalize_hyperda_source_trust_bank_state(
        state.source_checkpoint.get("source_trust_bank_state")
        or state.source_checkpoint.get("hyperda_source_trust_bank_state")
        or state.source_config.get("source_trust_bank_state")
        or state.source_config.get("hyperda_source_trust_bank_state")
    )


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
    source_trust_query = compose_target_context_source_trust_query_from_state(
        target_context_prompt_state,
        months,
        device=device,
    )
    reliability_features = compose_target_context_reliability_features_from_state(
        target_context_prompt_state,
        months,
        device=device,
    )
    pred = _model_forward(
        state.model,
        x_norm,
        z,
        months,
        x,
        reliability_features=reliability_features,
        source_trust_bank=source_trust_bank_for_few_shot_state(state),
        source_trust_query=source_trust_query,
    )
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


def decide_k12_vs_k4_reference_gate(
    *,
    candidate: Dict[str, Optional[float]],
    k4_reference: Dict[str, Optional[float]],
    enabled: bool,
    min_delta: float,
    k4_reference_adapt_mix_rho: Optional[float] = None,
    support_nesting_policy: str = "",
    nested_support_dates_hash: str = "",
) -> Dict[str, Any]:
    """Compare K12 candidate against K4-equivalent reference on nested support only."""
    candidate_obj = _float_or_none(candidate.get("standard_support_objective_full_support"))
    reference_obj = _float_or_none(k4_reference.get("standard_support_objective_full_support"))
    candidate_loss = _float_or_none(candidate.get("standard_support_loss_full_support"))
    reference_loss = _float_or_none(k4_reference.get("standard_support_loss_full_support"))
    candidate_surface = _float_or_none(candidate.get("standard_support_surface_loss_full_support"))
    reference_surface = _float_or_none(k4_reference.get("standard_support_surface_loss_full_support"))
    candidate_rootzone = _float_or_none(candidate.get("standard_support_rootzone_loss_full_support"))
    reference_rootzone = _float_or_none(k4_reference.get("standard_support_rootzone_loss_full_support"))
    objective_delta = (
        None if candidate_obj is None or reference_obj is None else candidate_obj - reference_obj
    )
    accepted = (not enabled) or (
        objective_delta is not None and objective_delta < -float(min_delta)
    )
    summary: Dict[str, Any] = {
        "support_gate_enabled": bool(enabled),
        "support_gate_label_source": "target_support_only",
        "support_gate_policy_role": "target_support_only_k12_vs_k4_reference_diagnostic",
        "support_gate_min_delta": float(min_delta),
        "support_gate_rootzone_tolerance": 0.0,
        "target_eval_usage": "final_eval_only_no_selection",
        "k4_reference_support_objective": reference_obj,
        "k12_candidate_support_objective": candidate_obj,
        "k12_vs_k4_support_objective_delta": objective_delta,
        "k4_reference_support_loss": reference_loss,
        "k12_candidate_support_loss": candidate_loss,
        "k4_reference_surface_loss": reference_surface,
        "k12_candidate_surface_loss": candidate_surface,
        "k4_reference_rootzone_loss": reference_rootzone,
        "k12_candidate_rootzone_loss": candidate_rootzone,
        "k4_reference_adapt_mix_rho": (
            None if k4_reference_adapt_mix_rho is None else float(k4_reference_adapt_mix_rho)
        ),
        "support_nesting_policy": str(support_nesting_policy or ""),
        "nested_support_dates_hash": str(nested_support_dates_hash or ""),
        "support_objective_before": reference_obj,
        "support_objective_after": candidate_obj,
        "support_objective_delta": objective_delta,
        "support_loss_before": reference_loss,
        "support_loss_after": candidate_loss,
        "support_loss_delta": None if candidate_loss is None or reference_loss is None else candidate_loss - reference_loss,
        "support_surface_loss_before": reference_surface,
        "support_surface_loss_after": candidate_surface,
        "support_surface_loss_delta": (
            None if candidate_surface is None or reference_surface is None else candidate_surface - reference_surface
        ),
        "support_rootzone_loss_before": reference_rootzone,
        "support_rootzone_loss_after": candidate_rootzone,
        "support_rootzone_loss_delta": (
            None if candidate_rootzone is None or reference_rootzone is None else candidate_rootzone - reference_rootzone
        ),
    }
    if accepted:
        summary.update(
            {
                "stage3_posterior_decision": "accepted",
                "support_gate_status": (
                    "support_only_k12_vs_k4_reference_disabled"
                    if not enabled
                    else "support_only_k12_beats_k4_reference"
                ),
                "support_gate_reject_reason": [],
            }
        )
    else:
        summary.update(
            {
                "stage3_posterior_decision": "fallback_to_k4_reference",
                "support_gate_status": "support_only_k12_fallback_to_k4_reference",
                "support_gate_reject_reason": [
                    "k12_not_better_than_k4_reference_on_nested_support"
                ],
            }
        )
    return summary


def defer_k0_anchor_gate_to_k4_reference_gate(
    summary: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Preserve a K12 candidate so the nested K4 reference gate can arbitrate."""
    generic_summary = dict(summary)
    deferred = dict(summary)
    deferred.update(
        {
            "stage3_posterior_decision": "accepted",
            "support_gate_status": "support_only_k12_pending_k4_reference_gate",
            "support_only_gate_status": str(
                summary.get("support_gate_status", "support_only_rejected_to_k0_anchor")
            ),
            "support_gate_reject_reason": [],
            "k0_anchor_gate_deferred_to_k4_reference": True,
            "k0_anchor_gate": generic_summary,
            "target_eval_usage": "final_eval_only_no_selection",
        }
    )
    return deferred, generic_summary


def decide_v10_k4_support_pool_gate(
    *,
    candidate: Dict[str, Optional[float]],
    k0_anchor: Dict[str, Optional[float]],
    min_delta: float,
    min_cycle_improvement_fraction: float,
    cycle_improvement_fraction: float,
    selected_candidate_id: str,
    support_candidate_pool: List[Dict[str, Any]],
) -> Dict[str, Any]:
    candidate_obj = _float_or_none(candidate.get("standard_support_objective_full_support"))
    reference_obj = _float_or_none(k0_anchor.get("standard_support_objective_full_support"))
    candidate_surface = _float_or_none(candidate.get("standard_support_surface_loss_full_support"))
    reference_surface = _float_or_none(k0_anchor.get("standard_support_surface_loss_full_support"))
    candidate_rootzone = _float_or_none(candidate.get("standard_support_rootzone_loss_full_support"))
    reference_rootzone = _float_or_none(k0_anchor.get("standard_support_rootzone_loss_full_support"))
    candidate_loss = _float_or_none(candidate.get("standard_support_loss_full_support"))
    reference_loss = _float_or_none(k0_anchor.get("standard_support_loss_full_support"))
    objective_delta = None if candidate_obj is None or reference_obj is None else candidate_obj - reference_obj
    loss_delta = None if candidate_loss is None or reference_loss is None else candidate_loss - reference_loss
    rootzone_delta = None if candidate_rootzone is None or reference_rootzone is None else candidate_rootzone - reference_rootzone
    surface_delta = None if candidate_surface is None or reference_surface is None else candidate_surface - reference_surface
    reject_reasons: List[str] = []
    if objective_delta is None:
        reject_reasons.append("missing_support_objective")
    elif objective_delta >= -float(min_delta):
        reject_reasons.append("objective_not_improved")
    if rootzone_delta is None:
        reject_reasons.append("missing_rootzone_guard")
    elif rootzone_delta > 0.0:
        reject_reasons.append("rootzone_regression")
    if float(cycle_improvement_fraction) + 1e-12 < float(min_cycle_improvement_fraction):
        reject_reasons.append("insufficient_support_cycle_improvement_fraction")
    accepted = not reject_reasons
    return {
        "support_gate_enabled": True,
        "support_gate_label_source": "target_support_only",
        "support_gate_policy_role": "target_support_only_v10_candidate_pool_diagnostic",
        "support_selection_objective": "exact_mixed_target_support_only",
        "target_eval_usage": "final_eval_only_no_selection",
        "selected_support_candidate_id": str(selected_candidate_id or ""),
        "support_candidate_pool": [dict(candidate) for candidate in support_candidate_pool],
        "support_gate_min_delta": float(min_delta),
        "support_gate_rootzone_tolerance": 0.0,
        "support_gate_cycle_improvement_min_fraction": float(min_cycle_improvement_fraction),
        "support_cycle_improvement_fraction": float(cycle_improvement_fraction),
        "support_objective_before": reference_obj,
        "support_objective_after": candidate_obj if accepted else reference_obj,
        "support_objective_delta": 0.0 if (not accepted and reference_obj is not None) else objective_delta,
        "support_loss_before": reference_loss,
        "support_loss_after": candidate_loss if accepted else reference_loss,
        "support_loss_delta": 0.0 if (not accepted and reference_loss is not None) else loss_delta,
        "support_surface_loss_before": reference_surface,
        "support_surface_loss_after": candidate_surface if accepted else reference_surface,
        "support_surface_loss_delta": 0.0 if (not accepted and reference_surface is not None) else surface_delta,
        "support_rootzone_loss_before": reference_rootzone,
        "support_rootzone_loss_after": candidate_rootzone if accepted else reference_rootzone,
        "support_rootzone_loss_delta": 0.0 if (not accepted and reference_rootzone is not None) else rootzone_delta,
        "support_candidate_objective_after": candidate_obj,
        "support_candidate_objective_delta": objective_delta,
        "support_candidate_loss_after": candidate_loss,
        "support_candidate_loss_delta": loss_delta,
        "support_candidate_surface_loss_after": candidate_surface,
        "support_candidate_surface_loss_delta": surface_delta,
        "support_candidate_rootzone_loss_after": candidate_rootzone,
        "support_candidate_rootzone_loss_delta": rootzone_delta,
        "support_gate_status": "accepted" if accepted else "support_only_rejected_to_k0_anchor",
        "stage3_posterior_decision": "accepted" if accepted else "rejected_to_k0_anchor",
        "support_gate_reject_reason": reject_reasons,
    }


def decide_v10_k12_support_pool_gate(
    *,
    candidate: Dict[str, Optional[float]],
    k4_reference: Dict[str, Optional[float]],
    candidate_nested_k4: Dict[str, Optional[float]],
    k4_reference_nested: Dict[str, Optional[float]],
    min_delta: float,
    min_cycle_improvement_fraction: float,
    cycle_improvement_fraction: float,
    selected_candidate_id: str,
    support_candidate_pool: List[Dict[str, Any]],
    k4_reference_adapt_mix_rho: Optional[float] = None,
    support_nesting_policy: str = "",
    nested_support_dates_hash: str = "",
) -> Dict[str, Any]:
    candidate_obj = _float_or_none(candidate.get("standard_support_objective_full_support"))
    reference_obj = _float_or_none(k4_reference.get("standard_support_objective_full_support"))
    candidate_surface = _float_or_none(candidate.get("standard_support_surface_loss_full_support"))
    reference_surface = _float_or_none(k4_reference.get("standard_support_surface_loss_full_support"))
    candidate_rootzone = _float_or_none(candidate.get("standard_support_rootzone_loss_full_support"))
    reference_rootzone = _float_or_none(k4_reference.get("standard_support_rootzone_loss_full_support"))
    candidate_loss = _float_or_none(candidate.get("standard_support_loss_full_support"))
    reference_loss = _float_or_none(k4_reference.get("standard_support_loss_full_support"))
    objective_delta = None if candidate_obj is None or reference_obj is None else candidate_obj - reference_obj
    loss_delta = None if candidate_loss is None or reference_loss is None else candidate_loss - reference_loss
    rootzone_delta = None if candidate_rootzone is None or reference_rootzone is None else candidate_rootzone - reference_rootzone
    surface_delta = None if candidate_surface is None or reference_surface is None else candidate_surface - reference_surface
    nested_candidate_obj = _float_or_none(candidate_nested_k4.get("standard_support_objective_full_support"))
    nested_reference_obj = _float_or_none(k4_reference_nested.get("standard_support_objective_full_support"))
    nested_delta = (
        None if nested_candidate_obj is None or nested_reference_obj is None else nested_candidate_obj - nested_reference_obj
    )
    reject_reasons: List[str] = []
    if objective_delta is None:
        reject_reasons.append("missing_support_objective")
    elif objective_delta >= -float(min_delta):
        reject_reasons.append("k12_not_better_than_k4_reference_on_nested_support")
    if rootzone_delta is None:
        reject_reasons.append("missing_rootzone_guard")
    elif rootzone_delta > 0.0:
        reject_reasons.append("rootzone_regression")
    if float(cycle_improvement_fraction) + 1e-12 < float(min_cycle_improvement_fraction):
        reject_reasons.append("insufficient_support_cycle_improvement_fraction")
    if nested_delta is None:
        reject_reasons.append("missing_nested_k4_subset_guard")
    elif nested_delta > 0.0:
        reject_reasons.append("nested_k4_subset_worse_than_k4_reference")
    accepted = not reject_reasons
    return {
        "support_gate_enabled": True,
        "support_gate_label_source": "target_support_only",
        "support_gate_policy_role": "target_support_only_v10_k12_vs_k4_reference_diagnostic",
        "support_selection_objective": "exact_mixed_target_support_only",
        "target_eval_usage": "final_eval_only_no_selection",
        "selected_support_candidate_id": str(selected_candidate_id or ""),
        "support_candidate_pool": [dict(candidate) for candidate in support_candidate_pool],
        "k12_reference_policy": "k4_safe_nested_reference",
        "support_gate_min_delta": float(min_delta),
        "support_gate_rootzone_tolerance": 0.0,
        "support_gate_cycle_improvement_min_fraction": float(min_cycle_improvement_fraction),
        "support_cycle_improvement_fraction": float(cycle_improvement_fraction),
        "k4_reference_support_objective": reference_obj,
        "k12_candidate_support_objective": candidate_obj,
        "k12_vs_k4_support_objective_delta": objective_delta,
        "k4_reference_surface_loss": reference_surface,
        "k12_candidate_surface_loss": candidate_surface,
        "k4_reference_rootzone_loss": reference_rootzone,
        "k12_candidate_rootzone_loss": candidate_rootzone,
        "k12_candidate_nested_k4_support_objective": nested_candidate_obj,
        "k4_reference_nested_k4_support_objective": nested_reference_obj,
        "k12_nested_k4_vs_k4_reference_delta": nested_delta,
        "k4_reference_adapt_mix_rho": (
            None if k4_reference_adapt_mix_rho is None else float(k4_reference_adapt_mix_rho)
        ),
        "support_nesting_policy": str(support_nesting_policy or ""),
        "nested_support_dates_hash": str(nested_support_dates_hash or ""),
        "support_objective_before": reference_obj,
        "support_objective_after": candidate_obj if accepted else reference_obj,
        "support_objective_delta": 0.0 if (not accepted and reference_obj is not None) else objective_delta,
        "support_loss_before": reference_loss,
        "support_loss_after": candidate_loss if accepted else reference_loss,
        "support_loss_delta": 0.0 if (not accepted and reference_loss is not None) else loss_delta,
        "support_surface_loss_before": reference_surface,
        "support_surface_loss_after": candidate_surface if accepted else reference_surface,
        "support_surface_loss_delta": 0.0 if (not accepted and reference_surface is not None) else surface_delta,
        "support_rootzone_loss_before": reference_rootzone,
        "support_rootzone_loss_after": candidate_rootzone if accepted else reference_rootzone,
        "support_rootzone_loss_delta": 0.0 if (not accepted and reference_rootzone is not None) else rootzone_delta,
        "support_candidate_objective_after": candidate_obj,
        "support_candidate_objective_delta": objective_delta,
        "support_candidate_loss_after": candidate_loss,
        "support_candidate_loss_delta": loss_delta,
        "support_candidate_surface_loss_after": candidate_surface,
        "support_candidate_surface_loss_delta": surface_delta,
        "support_candidate_rootzone_loss_after": candidate_rootzone,
        "support_candidate_rootzone_loss_delta": rootzone_delta,
        "support_gate_status": (
            "support_only_v10_k12_beats_k4_reference"
            if accepted
            else "support_only_v10_k12_fallback_to_k4_reference"
        ),
        "stage3_posterior_decision": "accepted" if accepted else "fallback_to_k4_reference",
        "support_gate_reject_reason": reject_reasons,
    }


def apply_diagnostic_direct_v2_support_risk_guard(
    *,
    summary: Dict[str, Any],
    stage3_kshot_mode: str,
    support_gradient_diagnostics: Dict[str, Any],
    target_parameter_drift: Dict[str, float],
    max_target_prompt_drift: float = 1.0,
    max_negative_gradient_fraction: float = 0.25,
    min_gradient_cosine: float = -0.10,
) -> Dict[str, Any]:
    """Reject diagnostic v2/v3 updates with support-only conflict or excessive drift.

    This is a diagnostic safety guard, not paper-facing selection. It uses only
    target_support optimization diagnostics and target-specific parameter drift.
    """
    guarded = dict(summary)
    if str(stage3_kshot_mode) not in {"diagnostic_direct_kshot_v2", "diagnostic_conservative_kshot_v3"}:
        return guarded
    guard_record: Dict[str, Any] = {
        "schema_version": "diagnostic_kshot_support_risk_guard_v2",
        "stage3_kshot_mode": str(stage3_kshot_mode),
        "label_source": "target_support_only",
        "target_eval_usage": "final_eval_only_no_selection",
        "max_target_prompt_drift": float(max_target_prompt_drift),
        "max_negative_gradient_fraction": float(max_negative_gradient_fraction),
        "min_gradient_cosine": float(min_gradient_cosine),
        "target_prompt_drift": _float_or_none(target_parameter_drift.get("target_prompt")),
        "total_drift": _float_or_none(target_parameter_drift.get("total")),
        "support_gradient_negative_fraction": _float_or_none(
            support_gradient_diagnostics.get("support_gradient_negative_fraction")
        ),
        "support_gradient_cosine_min": _float_or_none(
            support_gradient_diagnostics.get("support_gradient_cosine_min")
        ),
    }
    reject_reasons = list(guarded.get("support_gate_reject_reason", []) or [])
    prompt_drift = guard_record["target_prompt_drift"]
    neg_frac = guard_record["support_gradient_negative_fraction"]
    cosine_min = guard_record["support_gradient_cosine_min"]
    if prompt_drift is not None and prompt_drift > float(max_target_prompt_drift):
        reject_reasons.append("target_prompt_drift_exceeds_v2_guard")
    if (
        neg_frac is not None
        and neg_frac > float(max_negative_gradient_fraction)
    ) or (
        cosine_min is not None
        and cosine_min < float(min_gradient_cosine)
    ):
        reject_reasons.append("support_gradient_conflict")
    if reject_reasons and str(guarded.get("stage3_posterior_decision", "accepted")) == "accepted":
        guarded["support_gate_status"] = (
            "diagnostic_v2_support_risk_rejected_to_k0_anchor"
            if str(stage3_kshot_mode) == "diagnostic_direct_kshot_v2"
            else "diagnostic_support_risk_rejected_to_k0_anchor"
        )
        guarded["stage3_posterior_decision"] = "rejected_to_k0_anchor"
        guarded["support_gate_reject_reason"] = sorted(set(str(reason) for reason in reject_reasons))
        guard_record["status"] = "rejected_to_k0_anchor"
    else:
        guarded["support_gate_reject_reason"] = reject_reasons
        guard_record["status"] = "passed"
    guarded["diagnostic_support_risk_guard"] = guard_record
    guarded["diagnostic_v2_support_risk_guard"] = guard_record
    return guarded


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
    stage3_kshot_mode: str = "paper_safe",
    context_tta_mode: str = "none",
    context_tta_effective: bool = False,
    context_tta_source_stat_status: str = "not_requested",
) -> Dict[str, Any]:
    if int(K) == 0:
        if str(context_tta_mode or "none") == "none":
            return {"paper_facing_run": True, "diagnostic_run_reason": ""}
        if (
            str(context_tta_mode) == "prompt_feature_alignment"
            and bool(context_tta_effective)
            and "identity_fallback" not in str(context_tta_source_stat_status)
        ):
            return {"paper_facing_run": True, "diagnostic_run_reason": ""}
        return {
            "paper_facing_run": False,
            "diagnostic_run_reason": f"context_tta_{context_tta_mode}_not_source_side_paper_safe",
        }
    if str(stage3_kshot_mode) in {
        "diagnostic_direct_kshot",
        "diagnostic_direct_kshot_v2",
        "diagnostic_conservative_kshot_v3",
        "diagnostic_support_gain_v2",
        "diagnostic_support_gain_v3_stable",
        "diagnostic_support_gain_v4_nested_stable",
        "diagnostic_support_gain_v12_nested_cv",
        "diagnostic_support_gain_v13_k12_aggressive_calibration_pool",
        "diagnostic_finetune_support_gain_v14_nested",
        "diagnostic_support_affine_v1_nested",
        "diagnostic_safe_operator_v5_nested",
        "diagnostic_linearized_coeff_ridge_v6_nested",
        "diagnostic_linearized_coeff_ridge_v7_balanced_nested",
        "diagnostic_linearized_coeff_ridge_v8_hybrid_nested",
        "diagnostic_linearized_coeff_ridge_v9_guarded_nested",
        "diagnostic_linearized_coeff_ridge_v10_support_pool_nested",
        "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested",
    }:
        return {
            "paper_facing_run": False,
            "diagnostic_run_reason": f"{stage3_kshot_mode}_target_support_update",
        }
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
    if str(stage3_posterior_decision) == "fallback_to_k4_reference":
        return {
            "paper_facing_run": False,
            "diagnostic_run_reason": f"{stage3_kshot_mode}_support_only_fallback_to_k4",
        }
    if str(stage3_posterior_decision) == "no_update":
        return {
            "paper_facing_run": False,
            "diagnostic_run_reason": "source_policy_selected_no_update_k0_equivalent",
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


def load_k4_reference_target_adapter_state(checkpoint_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Load a run-local K4 target adapter state for diagnostic K12 support fallback."""
    path = Path(str(checkpoint_path or "")).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"--k4_reference_checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("target_adapter_state_dict")
    if not isinstance(state, dict) or not state:
        nested = checkpoint.get("stage3_posterior_state_dict", {})
        if isinstance(nested, dict):
            state = nested.get("target_adapter_state_dict")
    if not isinstance(state, dict) or not state:
        model_state = checkpoint.get("model_state_dict", {})
        if isinstance(model_state, dict):
            state = {
                str(name): tensor
                for name, tensor in model_state.items()
                if _is_target_adapter_state_key(str(name))
            }
    if not isinstance(state, dict) or not state:
        raise ValueError(f"K4 reference checkpoint lacks target adapter state: {path}")
    bad_keys = [str(name) for name in state if not _is_target_adapter_state_key(str(name))]
    if bad_keys:
        raise ValueError(f"K4 reference target adapter state contains non-target keys: {bad_keys[:5]}")
    return {
        str(name): tensor.detach().to(device=device).clone()
        for name, tensor in state.items()
    }


def load_k4_reference_adapt_mix_rho(checkpoint_path: str) -> Optional[float]:
    """Return the final eval mix rho recorded by a K4 reference checkpoint."""
    path = Path(str(checkpoint_path or "")).expanduser()
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if isinstance(config, dict):
        value = config.get("adapt_mix_rho")
        if value is not None:
            return float(value)
    return None


def load_k4_reference_residual_gain_alphas(checkpoint_path: str) -> Tuple[float, float]:
    """Return support-gain alphas recorded by a K4 reference checkpoint."""
    path = Path(str(checkpoint_path or "")).expanduser()
    if not path.exists():
        return 1.0, 1.0
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    surface = checkpoint.get("residual_gain_alpha_surface")
    rootzone = checkpoint.get("residual_gain_alpha_rootzone")
    if surface is None and isinstance(config, dict):
        surface = config.get("residual_gain_alpha_surface")
    if rootzone is None and isinstance(config, dict):
        rootzone = config.get("residual_gain_alpha_rootzone")
    return (
        float(1.0 if surface is None else surface),
        float(1.0 if rootzone is None else rootzone),
    )


def load_k4_reference_support_affine_calibration(checkpoint_path: str) -> Dict[str, Any]:
    """Return frozen support-affine calibration recorded by a K4 reference checkpoint."""
    path = Path(str(checkpoint_path or "")).expanduser()
    if not path.exists():
        return {}
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    affine = checkpoint.get("support_affine_calibration")
    if not isinstance(affine, dict) and isinstance(config, dict):
        affine = config.get("support_affine_calibration")
    return dict(affine) if isinstance(affine, dict) else {}


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


def _ridge_balanced_huber_observation_weights(
    *,
    batch: Dict[str, torch.Tensor],
    pred: torch.Tensor,
    residual: torch.Tensor,
    surface_weight: float,
    rootzone_weight: float,
    use_lat_weighted_loss: bool,
    delta: float = 1.0,
) -> torch.Tensor:
    """Build support-only balanced Huber weights for the linearized ridge solve."""
    base = _ridge_observation_weights(
        batch=batch,
        pred=pred,
        surface_weight=surface_weight,
        rootzone_weight=rootzone_weight,
        use_lat_weighted_loss=use_lat_weighted_loss,
    ).float()
    valid = base > 0
    abs_residual = residual.detach().float().abs()
    delta_t = torch.as_tensor(float(delta), dtype=abs_residual.dtype, device=abs_residual.device)
    influence = torch.where(
        abs_residual <= delta_t,
        torch.ones_like(abs_residual),
        delta_t / abs_residual.clamp_min(1e-12),
    )
    balanced = base * influence
    batch_size = int(pred.shape[0])
    channel_weights = torch.tensor(
        [float(surface_weight), float(rootzone_weight)],
        dtype=balanced.dtype,
        device=balanced.device,
    )
    for sample_idx in range(batch_size):
        for channel_idx in range(2):
            mask_sc = valid[sample_idx, channel_idx]
            weight_sum = balanced[sample_idx, channel_idx][mask_sc].sum()
            if float(weight_sum.detach().cpu().item()) > 0.0:
                balanced[sample_idx, channel_idx] = balanced[sample_idx, channel_idx] * (
                    channel_weights[channel_idx] / weight_sum.clamp_min(1e-12)
                )
    return balanced


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
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Apply the ridge feature-pixel cap once across all support rows."""
    row_count = int(residual.numel())
    max_feature_pixels = int(max_feature_pixels)
    if max_feature_pixels <= 0 or row_count <= max_feature_pixels * 2:
        return design, residual, weights
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
    selected_weights = None if weights is None else weights.index_select(0, indices)
    return design.index_select(0, indices), residual.index_select(0, indices), selected_weights


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


def _support_batch_observation_weights(
    *,
    batch: Dict[str, torch.Tensor],
    pred: torch.Tensor,
    surface_weight: float,
    rootzone_weight: float,
    use_lat_weighted_loss: bool,
    weighting: str,
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if str(weighting) == "cycle_variable_balanced_huber":
        if residual is None:
            raise ValueError("cycle_variable_balanced_huber support scoring requires residual")
        return _ridge_balanced_huber_observation_weights(
            batch=batch,
            pred=pred,
            residual=residual,
            surface_weight=surface_weight,
            rootzone_weight=rootzone_weight,
            use_lat_weighted_loss=use_lat_weighted_loss,
            delta=1.0,
        )
    return _ridge_observation_weights(
        batch=batch,
        pred=pred,
        surface_weight=surface_weight,
        rootzone_weight=rootzone_weight,
        use_lat_weighted_loss=use_lat_weighted_loss,
    )


def _empty_mixed_support_objective() -> Dict[str, Any]:
    return {
        "standard_support_objective_full_support": None,
        "standard_support_loss_full_support": None,
        "standard_support_surface_loss_full_support": None,
        "standard_support_rootzone_loss_full_support": None,
        "standard_support_increment_loss_full_support": None,
        "standard_support_analysis_loss_full_support": None,
        "standard_support_analysis_surface_loss_full_support": None,
        "standard_support_analysis_rootzone_loss_full_support": None,
        "support_cycle_count": 0,
        "support_cycle_objectives": [],
        "support_cycle_surface_losses": [],
        "support_cycle_rootzone_losses": [],
    }


def _empty_mixed_raw_wrmse_objective() -> Dict[str, Any]:
    return {
        "standard_support_objective_full_support": None,
        "standard_support_loss_full_support": None,
        "standard_support_surface_loss_full_support": None,
        "standard_support_rootzone_loss_full_support": None,
        "standard_support_increment_loss_full_support": None,
        "standard_support_analysis_loss_full_support": None,
        "standard_support_increment_wrmse_full_support": None,
        "standard_support_surface_wrmse_full_support": None,
        "standard_support_rootzone_wrmse_full_support": None,
        "support_cycle_count": 0,
        "support_cycle_objectives": [],
        "support_cycle_surface_losses": [],
        "support_cycle_rootzone_losses": [],
    }


def _expand_support_weight(
    mask: torch.Tensor,
    pred: torch.Tensor,
    latitude_weight: Optional[torch.Tensor],
    *,
    use_lat_weighted_loss: bool,
) -> torch.Tensor:
    weight = mask.float().to(pred.device)
    if weight.ndim == 3:
        weight = weight.unsqueeze(1)
    if bool(use_lat_weighted_loss) and latitude_weight is not None:
        latw = latitude_weight.to(pred.device).float()
        if latw.ndim == 2:
            latw = latw.unsqueeze(0).unsqueeze(0)
        elif latw.ndim == 3:
            latw = latw.unsqueeze(1)
        weight = weight * latw
    return weight.expand_as(pred).float().clamp_min(0.0)


def _mixed_raw_increment_wrmse_from_cycle_values(
    *,
    surface_sse: float,
    rootzone_sse: float,
    surface_weight_sum: float,
    rootzone_weight_sum: float,
    surface_objective_weight: float,
    rootzone_objective_weight: float,
) -> Dict[str, Optional[float]]:
    surface = (
        float(np.sqrt(float(surface_sse) / max(float(surface_weight_sum), 1e-12)))
        if float(surface_weight_sum) > 0.0
        else None
    )
    rootzone = (
        float(np.sqrt(float(rootzone_sse) / max(float(rootzone_weight_sum), 1e-12)))
        if float(rootzone_weight_sum) > 0.0
        else None
    )
    if surface is None or rootzone is None:
        objective = None
    else:
        objective = float(surface_objective_weight) * surface + float(rootzone_objective_weight) * rootzone
    total_weight = float(surface_weight_sum) + float(rootzone_weight_sum)
    total = (
        float(np.sqrt((float(surface_sse) + float(rootzone_sse)) / max(total_weight, 1e-12)))
        if total_weight > 0.0
        else None
    )
    return {
        "objective": objective,
        "loss": total,
        "surface": surface,
        "rootzone": rootzone,
    }


@torch.no_grad()
def mixed_raw_increment_wrmse_objective_from_loader(
    *,
    state: FewShotAdaptationState,
    loader: DataLoader,
    device: torch.device,
    target_context_prompt_state: Dict[str, Any],
    anchor_state: Dict[str, torch.Tensor],
    candidate_state: Dict[str, torch.Tensor],
    rho: float,
    normalize_increment: bool,
    surface_weight: float,
    rootzone_weight: float,
    use_lat_weighted_loss: bool,
    max_cycles: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate the final mixed predictor with raw physical increment WRMSE.

    This intentionally mirrors the target-eval WRMSE scale more closely than
    the normalized support training loss while still using target_support only.
    """
    rho = float(rho)
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    original_state = extract_target_adapter_state(state.model)
    was_training = state.model.training
    surface_sse = 0.0
    rootzone_sse = 0.0
    surface_weight_sum = 0.0
    rootzone_weight_sum = 0.0
    cycle_objectives: List[float] = []
    cycle_surface_losses: List[float] = []
    cycle_rootzone_losses: List[float] = []
    seen = 0
    try:
        state.model.eval()
        for batch in loader:
            batch_size = int(batch["x"].shape[0])
            for sample_idx in range(batch_size):
                if max_cycles is not None and seen >= int(max_cycles):
                    break
                single_batch: Dict[str, torch.Tensor] = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        single_batch[key] = value[sample_idx: sample_idx + 1]
                    else:
                        single_batch[key] = value
                apply_target_adapter_state(state.model, anchor_state)
                pred_anchor_norm = _ridge_forward_prediction(
                    state=state,
                    batch=single_batch,
                    device=device,
                    target_context_prompt_state=target_context_prompt_state,
                )
                apply_target_adapter_state(state.model, candidate_state)
                pred_candidate_norm = _ridge_forward_prediction(
                    state=state,
                    batch=single_batch,
                    device=device,
                    target_context_prompt_state=target_context_prompt_state,
                )
                pred_norm = pred_anchor_norm + rho * (pred_candidate_norm - pred_anchor_norm)
                pred = _denormalize_increment(pred_norm, state.normalization, normalize_increment)
                target = torch.stack(
                    [
                        single_batch["increment_surface"].to(device),
                        single_batch["increment_rootzone"].to(device),
                    ],
                    dim=1,
                ).float()
                weights = _expand_support_weight(
                    single_batch["loss_mask"].to(device),
                    pred,
                    single_batch.get("latitude_weight"),
                    use_lat_weighted_loss=use_lat_weighted_loss,
                )
                sq = (pred.float() - target.float()).square() * weights
                channel_sse = sq.sum(dim=(0, 2, 3)).detach().double().cpu()
                channel_weight = weights.sum(dim=(0, 2, 3)).detach().double().cpu()
                surface_sse += float(channel_sse[0].item())
                rootzone_sse += float(channel_sse[1].item())
                surface_weight_sum += float(channel_weight[0].item())
                rootzone_weight_sum += float(channel_weight[1].item())
                cycle = _mixed_raw_increment_wrmse_from_cycle_values(
                    surface_sse=float(channel_sse[0].item()),
                    rootzone_sse=float(channel_sse[1].item()),
                    surface_weight_sum=float(channel_weight[0].item()),
                    rootzone_weight_sum=float(channel_weight[1].item()),
                    surface_objective_weight=surface_weight,
                    rootzone_objective_weight=rootzone_weight,
                )
                if cycle["objective"] is not None and cycle["surface"] is not None and cycle["rootzone"] is not None:
                    cycle_objectives.append(float(cycle["objective"]))
                    cycle_surface_losses.append(float(cycle["surface"]))
                    cycle_rootzone_losses.append(float(cycle["rootzone"]))
                seen += 1
            if max_cycles is not None and seen >= int(max_cycles):
                break
    finally:
        apply_target_adapter_state(state.model, original_state)
        state.model.train(was_training)

    if seen <= 0 or (surface_weight_sum + rootzone_weight_sum) <= 0.0:
        return _empty_mixed_raw_wrmse_objective()
    aggregate = _mixed_raw_increment_wrmse_from_cycle_values(
        surface_sse=surface_sse,
        rootzone_sse=rootzone_sse,
        surface_weight_sum=surface_weight_sum,
        rootzone_weight_sum=rootzone_weight_sum,
        surface_objective_weight=surface_weight,
        rootzone_objective_weight=rootzone_weight,
    )
    return {
        "standard_support_objective_full_support": aggregate["objective"],
        "standard_support_loss_full_support": aggregate["loss"],
        "standard_support_surface_loss_full_support": aggregate["surface"],
        "standard_support_rootzone_loss_full_support": aggregate["rootzone"],
        "standard_support_increment_loss_full_support": aggregate["loss"],
        "standard_support_analysis_loss_full_support": None,
        "standard_support_analysis_surface_loss_full_support": None,
        "standard_support_analysis_rootzone_loss_full_support": None,
        "standard_support_regularization_loss_full_support": 0.0,
        "standard_support_increment_wrmse_full_support": aggregate["loss"],
        "standard_support_surface_wrmse_full_support": aggregate["surface"],
        "standard_support_rootzone_wrmse_full_support": aggregate["rootzone"],
        "support_cycle_count": len(cycle_objectives),
        "support_cycle_objectives": cycle_objectives,
        "support_cycle_surface_losses": cycle_surface_losses,
        "support_cycle_rootzone_losses": cycle_rootzone_losses,
    }


@torch.no_grad()
def exact_mixed_support_objective_from_loader(
    *,
    state: FewShotAdaptationState,
    loader: DataLoader,
    device: torch.device,
    target_context_prompt_state: Dict[str, Any],
    anchor_state: Dict[str, torch.Tensor],
    candidate_state: Dict[str, torch.Tensor],
    rho: float,
    normalize_increment: bool,
    surface_weight: float,
    rootzone_weight: float,
    use_lat_weighted_loss: bool,
    support_weighting: str = "global_pixel_l2",
    max_cycles: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate the final-eval mixed predictor on target_support only.

    The prediction form matches evaluation:
    ``pred = rho * pred_candidate + (1-rho) * pred_K0_anchor``.
    """
    if str(support_weighting) not in RIDGE_WEIGHTINGS:
        raise ValueError(
            f"unsupported support_weighting={support_weighting!r}; expected one of {list(RIDGE_WEIGHTINGS)}"
        )
    rho = float(rho)
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    original_state = extract_target_adapter_state(state.model)
    was_training = state.model.training
    weighted_sum = torch.zeros(2, dtype=torch.float64)
    weight_sum = torch.zeros(2, dtype=torch.float64)
    cycle_objectives: List[float] = []
    cycle_surface_losses: List[float] = []
    cycle_rootzone_losses: List[float] = []
    seen = 0
    try:
        state.model.eval()
        for batch in loader:
            batch_size = int(batch["x"].shape[0])
            for sample_idx in range(batch_size):
                if max_cycles is not None and seen >= int(max_cycles):
                    break
                single_batch: Dict[str, torch.Tensor] = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        single_batch[key] = value[sample_idx: sample_idx + 1]
                    else:
                        single_batch[key] = value
                apply_target_adapter_state(state.model, anchor_state)
                pred_anchor = _ridge_forward_prediction(
                    state=state,
                    batch=single_batch,
                    device=device,
                    target_context_prompt_state=target_context_prompt_state,
                )
                apply_target_adapter_state(state.model, candidate_state)
                pred_candidate = _ridge_forward_prediction(
                    state=state,
                    batch=single_batch,
                    device=device,
                    target_context_prompt_state=target_context_prompt_state,
                )
                pred = pred_anchor + float(rho) * (pred_candidate - pred_anchor)
                target = _target_tensor(
                    single_batch["increment_surface"].to(device),
                    single_batch["increment_rootzone"].to(device),
                    state.normalization,
                    normalize_increment=normalize_increment,
                )
                residual = pred.float() - target.float()
                weights = _support_batch_observation_weights(
                    batch=single_batch,
                    pred=pred,
                    surface_weight=surface_weight,
                    rootzone_weight=rootzone_weight,
                    use_lat_weighted_loss=use_lat_weighted_loss,
                    weighting=str(support_weighting),
                    residual=residual,
                ).float().clamp_min(0.0)
                sq = residual.square() * weights
                channel_sum = sq.sum(dim=(0, 2, 3)).detach().double().cpu()
                channel_weight = weights.sum(dim=(0, 2, 3)).detach().double().cpu()
                weighted_sum += channel_sum
                weight_sum += channel_weight
                surface_loss = (
                    float((channel_sum[0] / channel_weight[0].clamp_min(1e-12)).item())
                    if float(channel_weight[0].item()) > 0.0
                    else None
                )
                rootzone_loss = (
                    float((channel_sum[1] / channel_weight[1].clamp_min(1e-12)).item())
                    if float(channel_weight[1].item()) > 0.0
                    else None
                )
                if surface_loss is not None and rootzone_loss is not None:
                    cycle_surface_losses.append(surface_loss)
                    cycle_rootzone_losses.append(rootzone_loss)
                    cycle_objectives.append(float(surface_weight) * surface_loss + float(rootzone_weight) * rootzone_loss)
                seen += 1
            if max_cycles is not None and seen >= int(max_cycles):
                break
    finally:
        apply_target_adapter_state(state.model, original_state)
        state.model.train(was_training)

    if seen <= 0 or float(weight_sum.sum().item()) <= 0.0:
        return _empty_mixed_support_objective()
    surface = (
        float((weighted_sum[0] / weight_sum[0].clamp_min(1e-12)).item())
        if float(weight_sum[0].item()) > 0.0
        else None
    )
    rootzone = (
        float((weighted_sum[1] / weight_sum[1].clamp_min(1e-12)).item())
        if float(weight_sum[1].item()) > 0.0
        else None
    )
    if surface is None or rootzone is None:
        total_loss = None
        objective = None
    else:
        total_loss = float(
            (weighted_sum.sum() / weight_sum.sum().clamp_min(1e-12)).item()
        )
        objective = float(surface_weight) * surface + float(rootzone_weight) * rootzone
    return {
        "standard_support_objective_full_support": objective,
        "standard_support_loss_full_support": total_loss,
        "standard_support_surface_loss_full_support": surface,
        "standard_support_rootzone_loss_full_support": rootzone,
        "standard_support_increment_loss_full_support": total_loss,
        "standard_support_analysis_loss_full_support": None,
        "standard_support_analysis_surface_loss_full_support": None,
        "standard_support_analysis_rootzone_loss_full_support": None,
        "standard_support_regularization_loss_full_support": 0.0,
        "support_cycle_count": len(cycle_objectives),
        "support_cycle_objectives": cycle_objectives,
        "support_cycle_surface_losses": cycle_surface_losses,
        "support_cycle_rootzone_losses": cycle_rootzone_losses,
    }


def support_cycle_improvement_fraction(
    before: Dict[str, Any],
    after: Dict[str, Any],
    *,
    min_improvement: float = 0.0,
    max_cycles: Optional[int] = None,
) -> float:
    before_values = [float(v) for v in before.get("support_cycle_objectives", []) or []]
    after_values = [float(v) for v in after.get("support_cycle_objectives", []) or []]
    count = min(len(before_values), len(after_values))
    if max_cycles is not None:
        count = min(count, int(max_cycles))
    if count <= 0:
        return 0.0
    improved = 0
    for before_value, after_value in zip(before_values[:count], after_values[:count]):
        if float(before_value) - float(after_value) > float(min_improvement):
            improved += 1
    return float(improved) / float(count)


def _support_cycle_count(loader: DataLoader) -> int:
    try:
        return int(len(loader.dataset))  # type: ignore[arg-type]
    except Exception:
        count = 0
        for batch in loader:
            count += int(batch["x"].shape[0])
        return count


def _support_subset_loader(loader: DataLoader, indices: Iterable[int]) -> DataLoader:
    selected = [int(index) for index in indices]
    return DataLoader(
        Subset(loader.dataset, selected),  # type: ignore[arg-type]
        batch_size=max(1, min(int(getattr(loader, "batch_size", 1) or 1), max(1, len(selected)))),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=getattr(loader, "collate_fn", _collate_target_batch),
    )


def _mean_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(np.mean(clean))


def _fraction_nonpositive(values: Iterable[Optional[float]], *, tolerance: float = 0.0) -> float:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return 0.0
    passed = sum(1 for value in clean if value <= float(tolerance))
    return float(passed) / float(len(clean))


def summarize_v11_cv_folds(
    folds: List[Dict[str, Any]],
    *,
    nested_count: int,
) -> Dict[str, Any]:
    if not folds:
        return {
            "cv_fold_count": 0,
            "cv_candidate_objective": None,
            "cv_reference_objective": None,
            "cv_objective_delta": None,
            "cv_cycle_improvement_fraction": 0.0,
            "cv_surface_delta": None,
            "cv_rootzone_delta": None,
            "cv_rootzone_nonregression_fraction": 0.0,
            "cv_nested_k4_objective_delta": None,
            "cv_nested_k4_improvement_fraction": 0.0,
            "cv_added_objective_delta": None,
            "cv_added_improvement_fraction": 0.0,
        }
    candidate_objectives = [fold.get("candidate_objective") for fold in folds]
    reference_objectives = [fold.get("reference_objective") for fold in folds]
    objective_deltas = [fold.get("objective_delta") for fold in folds]
    surface_deltas = [fold.get("surface_delta") for fold in folds]
    rootzone_deltas = [fold.get("rootzone_delta") for fold in folds]
    improved = [
        float(delta) < 0.0
        for delta in objective_deltas
        if delta is not None
    ]
    nested_folds = [fold for fold in folds if int(fold.get("holdout_index", -1)) < int(nested_count)]
    added_folds = [fold for fold in folds if int(fold.get("holdout_index", -1)) >= int(nested_count)]

    def _delta_values(items: List[Dict[str, Any]], key: str) -> List[Optional[float]]:
        return [item.get(key) for item in items]

    nested_deltas = _delta_values(nested_folds, "objective_delta")
    added_deltas = _delta_values(added_folds, "objective_delta")
    return {
        "cv_fold_count": int(len(folds)),
        "cv_candidate_objective": _mean_optional(candidate_objectives),
        "cv_reference_objective": _mean_optional(reference_objectives),
        "cv_objective_delta": _mean_optional(objective_deltas),
        "cv_cycle_improvement_fraction": (
            float(sum(1 for value in improved if value)) / float(len(improved)) if improved else 0.0
        ),
        "cv_surface_delta": _mean_optional(surface_deltas),
        "cv_rootzone_delta": _mean_optional(rootzone_deltas),
        "cv_rootzone_nonregression_fraction": _fraction_nonpositive(rootzone_deltas),
        "cv_nested_k4_objective_delta": _mean_optional(nested_deltas),
        "cv_nested_k4_improvement_fraction": _fraction_nonpositive(nested_deltas),
        "cv_added_objective_delta": _mean_optional(added_deltas),
        "cv_added_improvement_fraction": _fraction_nonpositive(added_deltas),
    }


def _v11_selected_cv_metric(result: Dict[str, Any]) -> float:
    summary = result.get("cv_summary", {}) or {}
    value = _float_or_none(summary.get("cv_candidate_objective"))
    if value is None:
        return float("inf")
    return float(value)


def _v11_fold_records_from_scores(
    *,
    holdout_indices: List[int],
    candidate_score: Dict[str, Any],
    reference_score: Dict[str, Any],
    train_policy: str,
) -> List[Dict[str, Any]]:
    candidate_obj = [float(value) for value in candidate_score.get("support_cycle_objectives", []) or []]
    reference_obj = [float(value) for value in reference_score.get("support_cycle_objectives", []) or []]
    candidate_surface = [float(value) for value in candidate_score.get("support_cycle_surface_losses", []) or []]
    reference_surface = [float(value) for value in reference_score.get("support_cycle_surface_losses", []) or []]
    candidate_rootzone = [float(value) for value in candidate_score.get("support_cycle_rootzone_losses", []) or []]
    reference_rootzone = [float(value) for value in reference_score.get("support_cycle_rootzone_losses", []) or []]
    count = min(len(holdout_indices), len(candidate_obj), len(reference_obj))
    records: List[Dict[str, Any]] = []
    for local_idx in range(count):
        surface_delta = (
            candidate_surface[local_idx] - reference_surface[local_idx]
            if local_idx < len(candidate_surface) and local_idx < len(reference_surface)
            else None
        )
        rootzone_delta = (
            candidate_rootzone[local_idx] - reference_rootzone[local_idx]
            if local_idx < len(candidate_rootzone) and local_idx < len(reference_rootzone)
            else None
        )
        records.append(
            {
                "holdout_index": int(holdout_indices[local_idx]),
                "train_policy": str(train_policy),
                "candidate_objective": float(candidate_obj[local_idx]),
                "reference_objective": float(reference_obj[local_idx]),
                "objective_delta": float(candidate_obj[local_idx] - reference_obj[local_idx]),
                "candidate_surface": (
                    float(candidate_surface[local_idx]) if local_idx < len(candidate_surface) else None
                ),
                "reference_surface": (
                    float(reference_surface[local_idx]) if local_idx < len(reference_surface) else None
                ),
                "surface_delta": surface_delta,
                "candidate_rootzone": (
                    float(candidate_rootzone[local_idx]) if local_idx < len(candidate_rootzone) else None
                ),
                "reference_rootzone": (
                    float(reference_rootzone[local_idx]) if local_idx < len(reference_rootzone) else None
                ),
                "rootzone_delta": rootzone_delta,
            }
        )
    return records


def decide_v11_k4_support_pool_gate(
    *,
    full_candidate: Dict[str, Optional[float]],
    full_reference: Dict[str, Optional[float]],
    cv_summary: Dict[str, Any],
    min_delta: float,
    min_cycle_improvement_fraction: float,
    selected_candidate_id: str,
    support_candidate_pool: List[Dict[str, Any]],
) -> Dict[str, Any]:
    full_candidate_obj = _float_or_none(full_candidate.get("standard_support_objective_full_support"))
    full_reference_obj = _float_or_none(full_reference.get("standard_support_objective_full_support"))
    full_candidate_rootzone = _float_or_none(full_candidate.get("standard_support_rootzone_loss_full_support"))
    full_reference_rootzone = _float_or_none(full_reference.get("standard_support_rootzone_loss_full_support"))
    full_delta = None if full_candidate_obj is None or full_reference_obj is None else full_candidate_obj - full_reference_obj
    full_rootzone_delta = (
        None if full_candidate_rootzone is None or full_reference_rootzone is None else full_candidate_rootzone - full_reference_rootzone
    )
    cv_delta = _float_or_none(cv_summary.get("cv_objective_delta"))
    cv_rootzone_delta = _float_or_none(cv_summary.get("cv_rootzone_delta"))
    cv_fraction = float(cv_summary.get("cv_cycle_improvement_fraction", 0.0) or 0.0)
    reject_reasons: List[str] = []
    if cv_delta is None:
        reject_reasons.append("missing_loocv_objective")
    elif cv_delta >= -float(min_delta):
        reject_reasons.append("loocv_objective_not_improved")
    if cv_fraction + 1e-12 < float(min_cycle_improvement_fraction):
        reject_reasons.append("insufficient_loocv_cycle_improvement_fraction")
    if cv_rootzone_delta is None:
        reject_reasons.append("missing_loocv_rootzone_guard")
    elif cv_rootzone_delta > 0.0:
        reject_reasons.append("loocv_rootzone_regression")
    if full_delta is None:
        reject_reasons.append("missing_full_support_objective")
    elif full_delta > 0.0:
        reject_reasons.append("full_support_objective_regression")
    if full_rootzone_delta is None:
        reject_reasons.append("missing_full_support_rootzone_guard")
    elif full_rootzone_delta > 0.0:
        reject_reasons.append("full_support_rootzone_regression")
    accepted = not reject_reasons
    return {
        "support_gate_enabled": True,
        "support_gate_label_source": "target_support_only",
        "support_gate_policy_role": "target_support_only_v11_loocv_support_pool_diagnostic",
        "support_selection_objective": "loocv_mixed_raw_increment_wrmse_target_support_only",
        "target_eval_usage": "final_eval_only_no_selection",
        "selected_support_candidate_id": str(selected_candidate_id or ""),
        "support_candidate_pool": [dict(candidate) for candidate in support_candidate_pool],
        "support_gate_min_delta": float(min_delta),
        "support_gate_rootzone_tolerance": 0.0,
        "support_gate_cycle_improvement_min_fraction": float(min_cycle_improvement_fraction),
        "support_cycle_improvement_fraction": cv_fraction,
        "support_cv_objective_before": cv_summary.get("cv_reference_objective"),
        "support_cv_objective_after": cv_summary.get("cv_candidate_objective") if accepted else cv_summary.get("cv_reference_objective"),
        "support_cv_objective_delta": 0.0 if (not accepted and cv_summary.get("cv_reference_objective") is not None) else cv_delta,
        "support_cv_rootzone_delta": 0.0 if not accepted else cv_rootzone_delta,
        "support_objective_before": full_reference_obj,
        "support_objective_after": full_candidate_obj if accepted else full_reference_obj,
        "support_objective_delta": 0.0 if (not accepted and full_reference_obj is not None) else full_delta,
        "support_rootzone_loss_before": full_reference_rootzone,
        "support_rootzone_loss_after": full_candidate_rootzone if accepted else full_reference_rootzone,
        "support_rootzone_loss_delta": 0.0 if (not accepted and full_reference_rootzone is not None) else full_rootzone_delta,
        "support_candidate_objective_after": full_candidate_obj,
        "support_candidate_objective_delta": full_delta,
        "support_candidate_rootzone_loss_after": full_candidate_rootzone,
        "support_candidate_rootzone_loss_delta": full_rootzone_delta,
        "support_gate_status": "accepted" if accepted else "support_only_v11_loocv_rejected_to_k0_anchor",
        "stage3_posterior_decision": "accepted" if accepted else "rejected_to_k0_anchor",
        "support_gate_reject_reason": reject_reasons,
    }


def decide_v11_k12_support_pool_gate(
    *,
    full_candidate: Dict[str, Optional[float]],
    full_reference: Dict[str, Optional[float]],
    cv_summary: Dict[str, Any],
    min_delta: float,
    min_cycle_improvement_fraction: float,
    selected_candidate_id: str,
    support_candidate_pool: List[Dict[str, Any]],
    k4_reference_adapt_mix_rho: Optional[float] = None,
    support_nesting_policy: str = "",
    nested_support_dates_hash: str = "",
) -> Dict[str, Any]:
    full_candidate_obj = _float_or_none(full_candidate.get("standard_support_objective_full_support"))
    full_reference_obj = _float_or_none(full_reference.get("standard_support_objective_full_support"))
    full_candidate_rootzone = _float_or_none(full_candidate.get("standard_support_rootzone_loss_full_support"))
    full_reference_rootzone = _float_or_none(full_reference.get("standard_support_rootzone_loss_full_support"))
    full_delta = None if full_candidate_obj is None or full_reference_obj is None else full_candidate_obj - full_reference_obj
    full_rootzone_delta = (
        None if full_candidate_rootzone is None or full_reference_rootzone is None else full_candidate_rootzone - full_reference_rootzone
    )
    cv_delta = _float_or_none(cv_summary.get("cv_objective_delta"))
    cv_rootzone_delta = _float_or_none(cv_summary.get("cv_rootzone_delta"))
    cv_fraction = float(cv_summary.get("cv_cycle_improvement_fraction", 0.0) or 0.0)
    nested_delta = _float_or_none(cv_summary.get("cv_nested_k4_objective_delta"))
    nested_fraction = float(cv_summary.get("cv_nested_k4_improvement_fraction", 0.0) or 0.0)
    added_delta = _float_or_none(cv_summary.get("cv_added_objective_delta"))
    added_fraction = float(cv_summary.get("cv_added_improvement_fraction", 0.0) or 0.0)
    reject_reasons: List[str] = []
    if cv_delta is None:
        reject_reasons.append("missing_loocv_objective")
    elif cv_delta >= -float(min_delta):
        reject_reasons.append("loocv_k12_not_better_than_k4_reference")
    if cv_fraction + 1e-12 < float(min_cycle_improvement_fraction):
        reject_reasons.append("insufficient_loocv_cycle_improvement_fraction")
    if cv_rootzone_delta is None:
        reject_reasons.append("missing_loocv_rootzone_guard")
    elif cv_rootzone_delta > 0.0:
        reject_reasons.append("loocv_rootzone_regression")
    if nested_delta is None:
        reject_reasons.append("missing_nested_k4_loocv_guard")
    elif nested_delta > 0.0 or nested_fraction + 1e-12 < 0.75:
        reject_reasons.append("nested_k4_loocv_worse_than_k4_reference")
    if added_delta is None:
        reject_reasons.append("missing_added_support_loocv_guard")
    elif added_delta > 0.0 or added_fraction + 1e-12 < 0.625:
        reject_reasons.append("added_support_loocv_not_stable")
    if full_delta is None:
        reject_reasons.append("missing_full_support_objective")
    elif full_delta > 0.0:
        reject_reasons.append("full_support_objective_regression")
    if full_rootzone_delta is None:
        reject_reasons.append("missing_full_support_rootzone_guard")
    elif full_rootzone_delta > 0.0:
        reject_reasons.append("full_support_rootzone_regression")
    accepted = not reject_reasons
    return {
        "support_gate_enabled": True,
        "support_gate_label_source": "target_support_only",
        "support_gate_policy_role": "target_support_only_v11_loocv_k12_vs_k4_reference_diagnostic",
        "support_selection_objective": "loocv_mixed_raw_increment_wrmse_target_support_only",
        "target_eval_usage": "final_eval_only_no_selection",
        "selected_support_candidate_id": str(selected_candidate_id or ""),
        "support_candidate_pool": [dict(candidate) for candidate in support_candidate_pool],
        "k12_reference_policy": "k4_safe_nested_reference",
        "support_gate_min_delta": float(min_delta),
        "support_gate_rootzone_tolerance": 0.0,
        "support_gate_cycle_improvement_min_fraction": float(min_cycle_improvement_fraction),
        "support_cycle_improvement_fraction": cv_fraction,
        "support_cv_objective_before": cv_summary.get("cv_reference_objective"),
        "support_cv_objective_after": cv_summary.get("cv_candidate_objective") if accepted else cv_summary.get("cv_reference_objective"),
        "support_cv_objective_delta": 0.0 if (not accepted and cv_summary.get("cv_reference_objective") is not None) else cv_delta,
        "support_cv_rootzone_delta": 0.0 if not accepted else cv_rootzone_delta,
        "support_cv_nested_k4_objective_delta": nested_delta,
        "support_cv_nested_k4_improvement_fraction": nested_fraction,
        "support_cv_added_objective_delta": added_delta,
        "support_cv_added_improvement_fraction": added_fraction,
        "k4_reference_support_objective": full_reference_obj,
        "k12_candidate_support_objective": full_candidate_obj,
        "k12_vs_k4_support_objective_delta": full_delta,
        "k4_reference_rootzone_loss": full_reference_rootzone,
        "k12_candidate_rootzone_loss": full_candidate_rootzone,
        "k4_reference_adapt_mix_rho": (
            None if k4_reference_adapt_mix_rho is None else float(k4_reference_adapt_mix_rho)
        ),
        "support_nesting_policy": str(support_nesting_policy or ""),
        "nested_support_dates_hash": str(nested_support_dates_hash or ""),
        "support_objective_before": full_reference_obj,
        "support_objective_after": full_candidate_obj if accepted else full_reference_obj,
        "support_objective_delta": 0.0 if (not accepted and full_reference_obj is not None) else full_delta,
        "support_rootzone_loss_before": full_reference_rootzone,
        "support_rootzone_loss_after": full_candidate_rootzone if accepted else full_reference_rootzone,
        "support_rootzone_loss_delta": 0.0 if (not accepted and full_reference_rootzone is not None) else full_rootzone_delta,
        "support_candidate_objective_after": full_candidate_obj,
        "support_candidate_objective_delta": full_delta,
        "support_candidate_rootzone_loss_after": full_candidate_rootzone,
        "support_candidate_rootzone_loss_delta": full_rootzone_delta,
        "support_gate_status": (
            "support_only_v11_loocv_k12_beats_k4_reference"
            if accepted
            else "support_only_v11_loocv_k12_fallback_to_k4_reference"
        ),
        "stage3_posterior_decision": "accepted" if accepted else "fallback_to_k4_reference",
        "support_gate_reject_reason": reject_reasons,
    }


@torch.no_grad()
def calibrate_support_residual_gain_from_loader(
    *,
    state: FewShotAdaptationState,
    loader: DataLoader,
    device: torch.device,
    target_context_prompt_state: Dict[str, Any],
    alpha_grid: List[float],
    selection_rule: str = "max_min_skill",
) -> Dict[str, Any]:
    """Calibrate residual gain alphas from target_support labels only.

    This is the checkpoint-side counterpart to the legacy evaluator diagnostic.
    It never reads target_eval; the resulting alphas are saved in the adapted
    checkpoint before final evaluation runs.
    """
    was_training = state.model.training
    state.model.eval()
    samples_s: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    samples_r: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    normalize_increment = state.normalization.get("inc_mean") is not None
    try:
        for batch in loader:
            pred_norm = _ridge_forward_prediction(
                state=state,
                batch=batch,
                device=device,
                target_context_prompt_state=target_context_prompt_state,
            )
            pred = _denormalize_increment(pred_norm, state.normalization, normalize_increment)
            pred_np = pred.detach().cpu().numpy()
            true_s = batch["increment_surface"].detach().cpu().numpy()
            true_r = batch["increment_rootzone"].detach().cpu().numpy()
            forecast_s = batch["forecast_surface"].detach().cpu().numpy()
            forecast_r = batch["forecast_rootzone"].detach().cpu().numpy()
            mask = batch["loss_mask"].detach().cpu().numpy()
            latw = batch.get("latitude_weight")
            if latw is None:
                latw_np = np.ones_like(mask, dtype=np.float32)
            else:
                latw_np = latw.detach().cpu().numpy()
            for idx in range(pred_np.shape[0]):
                samples_s.append((
                    pred_np[idx, 0].astype(np.float32),
                    true_s[idx].astype(np.float32),
                    forecast_s[idx].astype(np.float32),
                    mask[idx].astype(np.float32),
                    latw_np[idx].astype(np.float32),
                ))
                samples_r.append((
                    pred_np[idx, 1].astype(np.float32),
                    true_r[idx].astype(np.float32),
                    forecast_r[idx].astype(np.float32),
                    mask[idx].astype(np.float32),
                    latw_np[idx].astype(np.float32),
                ))
    finally:
        state.model.train(was_training)

    calibration = calibrate_residual_gain(
        samples_s,
        samples_r,
        alpha_grid,
        selection_rule=selection_rule,
    )
    if not calibration:
        return {
            "calibration_mode": (
                "target_support_residual_gain_v4_nested_stable_grid"
                if selection_rule == "support_uncertainty_stable_high_alpha_with_dual_guard"
                else (
                    "target_support_residual_gain_stable_grid"
                    if selection_rule == "stable_high_alpha_with_mean_skill_guard"
                    else "target_support_residual_gain_fixed_grid"
                )
            ),
            "status": "empty_support_or_invalid_samples",
            "label_source": "target_support_only",
            "target_eval_usage": "final_eval_only_no_selection",
            "selection_rule": selection_rule,
            "best_alpha_raw": 1.0,
            "stable_candidate_alphas": [],
            "selection_margin": 0.0,
            "alpha_grid": [float(alpha) for alpha in alpha_grid],
            "support_count": 0,
            "best_alpha_surface": 1.0,
            "best_alpha_rootzone": 1.0,
        }
    calibration = dict(calibration)
    calibration.update(
        {
            "calibration_mode": (
                "target_support_residual_gain_v4_nested_stable_grid"
                if selection_rule == "support_uncertainty_stable_high_alpha_with_dual_guard"
                else (
                    "target_support_residual_gain_stable_grid"
                    if selection_rule == "stable_high_alpha_with_mean_skill_guard"
                    else "target_support_residual_gain_fixed_grid"
                )
            ),
            "status": "calibrated",
            "label_source": "target_support_only",
            "target_eval_usage": "final_eval_only_no_selection",
            "selection_rule": selection_rule,
            "support_count": len(samples_s),
            "alpha_grid": [float(alpha) for alpha in alpha_grid],
        }
    )
    return calibration


def _support_gain_sample_objective(
    sample: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    alpha: float,
) -> Optional[float]:
    pred_inc, true_inc, _forecast, mask, latw = sample
    pred = float(alpha) * np.asarray(pred_inc, dtype=np.float64)
    target = np.asarray(true_inc, dtype=np.float64)
    weight = np.asarray(mask, dtype=np.float64) * np.asarray(latw, dtype=np.float64)
    finite = np.isfinite(pred) & np.isfinite(target) & np.isfinite(weight) & (weight > 0.0)
    if not np.any(finite):
        return None
    mse = float(np.sum(weight[finite] * (pred[finite] - target[finite]) ** 2) / np.sum(weight[finite]))
    return float(np.sqrt(max(0.0, mse)))


def _support_gain_pair_objective(
    surface_sample: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    rootzone_sample: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    alpha_surface: float,
    alpha_rootzone: float,
    surface_weight: float,
    rootzone_weight: float,
) -> Optional[float]:
    surface = _support_gain_sample_objective(surface_sample, alpha_surface)
    rootzone = _support_gain_sample_objective(rootzone_sample, alpha_rootzone)
    if surface is None or rootzone is None:
        return None
    return float(float(surface_weight) * surface + float(rootzone_weight) * rootzone)


def _support_sample_season_from_month(month: int) -> str:
    month = int(month)
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def _support_alpha_transform_samples(
    samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    alpha: float,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    return [
        (
            (float(alpha) * np.asarray(pred_inc, dtype=np.float32)).astype(np.float32),
            true_inc,
            forecast,
            mask,
            latw,
        )
        for pred_inc, true_inc, forecast, mask, latw in samples
    ]


def _support_affine_score_sample(
    sample: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    variable: str,
    calibration: Dict[str, Any],
    season: Optional[str] = None,
) -> Optional[float]:
    pred_inc, true_inc, _forecast, mask, latw = sample
    seasonal = calibration.get("seasonal_affine_coefficients", {})
    block: Dict[str, Any] = {}
    if season and isinstance(seasonal, dict):
        season_block = seasonal.get(str(season), {})
        if isinstance(season_block, dict):
            candidate = season_block.get(variable, {})
            if isinstance(candidate, dict):
                block = candidate
    if not block:
        coeffs = calibration.get("support_affine_coefficients", {})
        if isinstance(coeffs, dict):
            candidate = coeffs.get(variable, {})
            if isinstance(candidate, dict):
                block = candidate
    a = float(block.get("a", 1.0)) if block else 1.0
    b = float(block.get("b", 0.0)) if block else 0.0
    pred = a * np.asarray(pred_inc, dtype=np.float64) + b
    target = np.asarray(true_inc, dtype=np.float64)
    weight = np.asarray(mask, dtype=np.float64) * np.asarray(latw, dtype=np.float64)
    finite = np.isfinite(pred) & np.isfinite(target) & np.isfinite(weight) & (weight > 0.0)
    if not np.any(finite):
        return None
    mse = float(np.sum(weight[finite] * (pred[finite] - target[finite]) ** 2) / np.sum(weight[finite]))
    return float(np.sqrt(max(0.0, mse)))


def _support_affine_pair_objective(
    surface_sample: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    rootzone_sample: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    calibration: Dict[str, Any],
    season: Optional[str],
    surface_weight: float,
    rootzone_weight: float,
) -> Optional[float]:
    surface = _support_affine_score_sample(
        surface_sample,
        variable="surface",
        calibration=calibration,
        season=season,
    )
    rootzone = _support_affine_score_sample(
        rootzone_sample,
        variable="rootzone",
        calibration=calibration,
        season=season,
    )
    if surface is None or rootzone is None:
        return None
    return float(float(surface_weight) * surface + float(rootzone_weight) * rootzone)


def _support_alpha2d_fit(
    samples_s: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    samples_r: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    alpha_grid_surface: List[float],
    alpha_grid_rootzone: List[float],
    *,
    surface_weight: float,
    rootzone_weight: float,
) -> Dict[str, Any]:
    best_surface = 1.0
    best_rootzone = 1.0
    best_objective: Optional[float] = None
    per_pair: Dict[str, Any] = {}
    for alpha_s in alpha_grid_surface:
        for alpha_r in alpha_grid_rootzone:
            objectives = []
            for sample_s, sample_r in zip(samples_s, samples_r):
                objective = _support_gain_pair_objective(
                    sample_s,
                    sample_r,
                    alpha_surface=float(alpha_s),
                    alpha_rootzone=float(alpha_r),
                    surface_weight=surface_weight,
                    rootzone_weight=rootzone_weight,
                )
                if objective is not None:
                    objectives.append(float(objective))
            value = float(np.mean(objectives)) if objectives else None
            per_pair[f"{float(alpha_s):.6g},{float(alpha_r):.6g}"] = value
            if value is not None and (best_objective is None or value < best_objective):
                best_objective = value
                best_surface = float(alpha_s)
                best_rootzone = float(alpha_r)
    return {
        "best_alpha_surface": float(best_surface),
        "best_alpha_rootzone": float(best_rootzone),
        "selection_objective": best_objective,
        "per_alpha2d_objective": per_pair,
        "alpha_grid_surface": [float(alpha) for alpha in alpha_grid_surface],
        "alpha_grid_rootzone": [float(alpha) for alpha in alpha_grid_rootzone],
    }


def _support_calibrated_samples_score(
    *,
    samples_s: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    samples_r: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    seasons: List[str],
    calibration: Dict[str, Any],
    surface_weight: float,
    rootzone_weight: float,
) -> Dict[str, Any]:
    alpha_s = float(calibration.get("alpha_surface", calibration.get("best_alpha_surface", 1.0)))
    alpha_r = float(calibration.get("alpha_rootzone", calibration.get("best_alpha_rootzone", 1.0)))
    affine = calibration.get("support_affine_calibration")
    cycle_objectives: List[float] = []
    cycle_surface_losses: List[float] = []
    cycle_rootzone_losses: List[float] = []
    for idx, (sample_s, sample_r) in enumerate(zip(samples_s, samples_r)):
        scaled_s = _support_alpha_transform_samples([sample_s], alpha_s)[0]
        scaled_r = _support_alpha_transform_samples([sample_r], alpha_r)[0]
        season = seasons[idx] if idx < len(seasons) else None
        if isinstance(affine, dict) and affine:
            surface = _support_affine_score_sample(
                scaled_s,
                variable="surface",
                calibration=affine,
                season=season,
            )
            rootzone = _support_affine_score_sample(
                scaled_r,
                variable="rootzone",
                calibration=affine,
                season=season,
            )
            objective = (
                None
                if surface is None or rootzone is None
                else float(surface_weight) * float(surface) + float(rootzone_weight) * float(rootzone)
            )
        else:
            surface = _support_gain_sample_objective(scaled_s, 1.0)
            rootzone = _support_gain_sample_objective(scaled_r, 1.0)
            objective = (
                None
                if surface is None or rootzone is None
                else float(surface_weight) * float(surface) + float(rootzone_weight) * float(rootzone)
            )
        if objective is not None and surface is not None and rootzone is not None:
            cycle_objectives.append(float(objective))
            cycle_surface_losses.append(float(surface))
            cycle_rootzone_losses.append(float(rootzone))
    if not cycle_objectives:
        return _empty_mixed_raw_wrmse_objective()
    surface = float(np.mean(cycle_surface_losses))
    rootzone = float(np.mean(cycle_rootzone_losses))
    objective = float(surface_weight) * surface + float(rootzone_weight) * rootzone
    loss = float(np.mean([surface, rootzone]))
    return {
        "standard_support_objective_full_support": objective,
        "standard_support_loss_full_support": loss,
        "standard_support_surface_loss_full_support": surface,
        "standard_support_rootzone_loss_full_support": rootzone,
        "standard_support_increment_loss_full_support": loss,
        "standard_support_analysis_loss_full_support": None,
        "standard_support_regularization_loss_full_support": 0.0,
        "support_cycle_count": len(cycle_objectives),
        "support_cycle_objectives": cycle_objectives,
        "support_cycle_surface_losses": cycle_surface_losses,
        "support_cycle_rootzone_losses": cycle_rootzone_losses,
    }


def summarize_support_gain_cv_folds(
    folds: List[Dict[str, Any]],
    *,
    nested_count: int,
) -> Dict[str, Any]:
    if not folds:
        return {
            "cv_fold_count": 0,
            "cv_candidate_objective": None,
            "cv_reference_objective": None,
            "cv_objective_delta": None,
            "cv_objective_delta_std": None,
            "cv_objective_delta_se": None,
            "cv_objective_delta_t": None,
            "cv_cycle_improvement_fraction": 0.0,
            "cv_nested_k4_objective_delta": None,
            "cv_nested_k4_improvement_fraction": 0.0,
            "cv_added_objective_delta": None,
            "cv_added_improvement_fraction": 0.0,
        }
    candidate_values = [fold.get("candidate_objective") for fold in folds]
    reference_values = [fold.get("reference_objective") for fold in folds]
    deltas = [fold.get("objective_delta") for fold in folds]
    clean_deltas = [
        float(delta)
        for delta in deltas
        if delta is not None and np.isfinite(float(delta))
    ]
    delta_mean = _mean_optional(deltas)
    if len(clean_deltas) >= 2:
        delta_std = float(np.std(np.asarray(clean_deltas, dtype=np.float64), ddof=1))
        delta_se = float(delta_std / np.sqrt(float(len(clean_deltas))))
    else:
        delta_std = 0.0 if clean_deltas else None
        delta_se = 0.0 if clean_deltas else None
    delta_t = (
        None
        if delta_mean is None or delta_se is None or float(delta_se) <= 0.0
        else float(delta_mean) / float(delta_se)
    )
    nested_folds = [fold for fold in folds if int(fold.get("holdout_index", -1)) < int(nested_count)]
    added_folds = [fold for fold in folds if int(fold.get("holdout_index", -1)) >= int(nested_count)]
    nested_deltas = [fold.get("objective_delta") for fold in nested_folds]
    added_deltas = [fold.get("objective_delta") for fold in added_folds]
    return {
        "cv_fold_count": int(len(folds)),
        "cv_candidate_objective": _mean_optional(candidate_values),
        "cv_reference_objective": _mean_optional(reference_values),
        "cv_objective_delta": delta_mean,
        "cv_objective_delta_std": delta_std,
        "cv_objective_delta_se": delta_se,
        "cv_objective_delta_t": delta_t,
        "cv_cycle_improvement_fraction": _fraction_nonpositive(deltas, tolerance=0.0),
        "cv_nested_k4_objective_delta": _mean_optional(nested_deltas),
        "cv_nested_k4_improvement_fraction": _fraction_nonpositive(nested_deltas, tolerance=0.0),
        "cv_added_objective_delta": _mean_optional(added_deltas),
        "cv_added_improvement_fraction": _fraction_nonpositive(added_deltas, tolerance=0.0),
    }


def decide_support_gain_v12_gate(
    *,
    K: int,
    cv_summary: Dict[str, Any],
    selected_alpha: float,
    min_delta: float,
    min_cycle_improvement_fraction: float,
) -> Dict[str, Any]:
    cv_delta = _float_or_none(cv_summary.get("cv_objective_delta"))
    cv_se = _float_or_none(cv_summary.get("cv_objective_delta_se"))
    cv_t = _float_or_none(cv_summary.get("cv_objective_delta_t"))
    cv_fraction = float(cv_summary.get("cv_cycle_improvement_fraction", 0.0) or 0.0)
    nested_delta = _float_or_none(cv_summary.get("cv_nested_k4_objective_delta"))
    nested_fraction = float(cv_summary.get("cv_nested_k4_improvement_fraction", 0.0) or 0.0)
    added_delta = _float_or_none(cv_summary.get("cv_added_objective_delta"))
    added_fraction = float(cv_summary.get("cv_added_improvement_fraction", 0.0) or 0.0)
    reject_reasons: List[str] = []
    if abs(float(selected_alpha) - 1.0) <= 1e-12:
        reject_reasons.append("selected_alpha_identity")
    if cv_delta is None:
        reject_reasons.append("missing_support_gain_cv_objective")
    elif cv_delta >= -float(min_delta):
        reject_reasons.append("support_gain_cv_objective_not_materially_improved")
    if cv_fraction + 1e-12 < float(min_cycle_improvement_fraction):
        reject_reasons.append("insufficient_support_gain_cv_cycle_improvement_fraction")
    if cv_delta is not None and cv_se is not None and float(cv_se) > 0.0:
        if abs(float(cv_delta)) < 1.0 * float(cv_se):
            reject_reasons.append("support_gain_cv_effect_within_one_standard_error")
        if cv_t is not None and float(cv_t) > -1.0:
            reject_reasons.append("support_gain_cv_t_stat_too_weak")
    if int(K) == 12:
        if nested_delta is None:
            reject_reasons.append("missing_nested_k4_support_gain_cv_guard")
        elif nested_delta > 0.0 or nested_fraction + 1e-12 < 0.75:
            reject_reasons.append("nested_k4_support_gain_cv_worse_than_reference")
        if added_delta is None:
            reject_reasons.append("missing_added_support_gain_cv_guard")
        elif added_delta >= -float(min_delta) or added_fraction + 1e-12 < 0.75:
            reject_reasons.append("added_support_gain_cv_not_independently_material")
    accepted = not reject_reasons
    if accepted:
        status = "support_only_v12_nested_cv_support_gain_accepted"
        decision = "accepted"
    elif int(K) == 12:
        status = "support_only_v12_nested_cv_support_gain_fallback_to_k4_reference"
        decision = "fallback_to_k4_reference"
    else:
        status = "support_only_v12_nested_cv_support_gain_rejected_to_k0_anchor"
        decision = "rejected_to_k0_anchor"
    return {
        "support_gate_enabled": True,
        "support_gate_label_source": "target_support_only",
        "support_gate_policy_role": "target_support_only_v12_nested_cv_support_gain_diagnostic",
        "support_selection_objective": "nested_cv_raw_increment_wrmse_support_gain_target_support_only",
        "target_eval_usage": "final_eval_only_no_selection",
        "support_gate_min_delta": float(min_delta),
        "support_gate_rootzone_tolerance": 0.0,
        "support_gate_cycle_improvement_min_fraction": float(min_cycle_improvement_fraction),
        "support_cycle_improvement_fraction": cv_fraction,
        "support_cv_objective_before": cv_summary.get("cv_reference_objective"),
        "support_cv_objective_after": cv_summary.get("cv_candidate_objective") if accepted else cv_summary.get("cv_reference_objective"),
        "support_cv_objective_delta": 0.0 if (not accepted and cv_summary.get("cv_reference_objective") is not None) else cv_delta,
        "support_cv_objective_delta_se": cv_se,
        "support_cv_objective_delta_t": cv_t,
        "support_cv_nested_k4_objective_delta": nested_delta,
        "support_cv_nested_k4_improvement_fraction": nested_fraction,
        "support_cv_added_objective_delta": added_delta,
        "support_cv_added_improvement_fraction": added_fraction,
        "support_gain_selected_alpha_raw": float(selected_alpha),
        "support_gate_status": status,
        "stage3_posterior_decision": decision,
        "support_gate_reject_reason": reject_reasons,
        "k12_reference_policy": "k4_safe_nested_reference" if int(K) == 12 else "",
    }


def _collect_support_gain_samples_from_loader(
    *,
    state: FewShotAdaptationState,
    loader: DataLoader,
    device: torch.device,
    target_context_prompt_state: Dict[str, Any],
) -> Tuple[
    List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
]:
    was_training = state.model.training
    state.model.eval()
    samples_s: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    samples_r: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    normalize_increment = state.normalization.get("inc_mean") is not None
    try:
        for batch in loader:
            pred_norm = _ridge_forward_prediction(
                state=state,
                batch=batch,
                device=device,
                target_context_prompt_state=target_context_prompt_state,
            )
            pred = _denormalize_increment(pred_norm, state.normalization, normalize_increment)
            pred_np = pred.detach().cpu().numpy()
            true_s = batch["increment_surface"].detach().cpu().numpy()
            true_r = batch["increment_rootzone"].detach().cpu().numpy()
            forecast_s = batch["forecast_surface"].detach().cpu().numpy()
            forecast_r = batch["forecast_rootzone"].detach().cpu().numpy()
            mask = batch["loss_mask"].detach().cpu().numpy()
            latw = batch.get("latitude_weight")
            latw_np = (
                np.ones_like(mask, dtype=np.float32)
                if latw is None
                else latw.detach().cpu().numpy()
            )
            for idx in range(pred_np.shape[0]):
                samples_s.append((
                    pred_np[idx, 0].astype(np.float32),
                    true_s[idx].astype(np.float32),
                    forecast_s[idx].astype(np.float32),
                    mask[idx].astype(np.float32),
                    latw_np[idx].astype(np.float32),
                ))
                samples_r.append((
                    pred_np[idx, 1].astype(np.float32),
                    true_r[idx].astype(np.float32),
                    forecast_r[idx].astype(np.float32),
                    mask[idx].astype(np.float32),
                    latw_np[idx].astype(np.float32),
                ))
    finally:
        state.model.train(was_training)
    return samples_s, samples_r


def _collect_support_months_from_loader(loader: DataLoader) -> List[int]:
    months: List[int] = []
    for batch in loader:
        values = batch.get("months")
        if isinstance(values, torch.Tensor):
            months.extend(int(value) for value in values.detach().cpu().view(-1).tolist())
    return months


def _v13_fit_candidate_calibration(
    candidate: Dict[str, Any],
    train_samples_s: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    train_samples_r: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    train_seasons: List[str],
    *,
    surface_weight: float,
    rootzone_weight: float,
) -> Dict[str, Any]:
    candidate_type = str(candidate.get("candidate_type", ""))
    calibration: Dict[str, Any] = {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "candidate_type": candidate_type,
        "target_eval_usage": "final_eval_only_no_selection",
        "label_source": "target_support_only",
        "alpha_surface": 1.0,
        "alpha_rootzone": 1.0,
        "support_affine_calibration": {},
        "support_calibration_dof": float(candidate.get("calibration_dof", 0.0) or 0.0),
    }
    if candidate_type == "alpha_global_grid":
        alpha_grid = [float(alpha) for alpha in candidate.get("alpha_grid", [])]
        fitted = calibrate_residual_gain(
            train_samples_s,
            train_samples_r,
            alpha_grid,
            selection_rule="support_uncertainty_stable_high_alpha_with_dual_guard",
        )
        alpha = float(fitted.get("best_alpha_surface", 1.0) if fitted else 1.0)
        calibration.update(
            {
                "alpha_surface": alpha,
                "alpha_rootzone": alpha,
                "support_gain_calibration": dict(fitted or {}),
                "support_calibration_dof": 1.0,
            }
        )
        return calibration
    if candidate_type in {"alpha2d_grid", "alpha2d_plus_global_affine"}:
        alpha2d = _support_alpha2d_fit(
            train_samples_s,
            train_samples_r,
            [float(alpha) for alpha in candidate.get("alpha_grid_surface", [])],
            [float(alpha) for alpha in candidate.get("alpha_grid_rootzone", [])],
            surface_weight=surface_weight,
            rootzone_weight=rootzone_weight,
        )
        calibration.update(
            {
                "alpha_surface": float(alpha2d.get("best_alpha_surface", 1.0)),
                "alpha_rootzone": float(alpha2d.get("best_alpha_rootzone", 1.0)),
                "support_gain_calibration": alpha2d,
                "support_calibration_dof": 2.0,
            }
        )
        if candidate_type == "alpha2d_grid":
            return calibration
        scaled_s = _support_alpha_transform_samples(train_samples_s, calibration["alpha_surface"])
        scaled_r = _support_alpha_transform_samples(train_samples_r, calibration["alpha_rootzone"])
        affine = calibrate_residual_affine(
            scaled_s,
            scaled_r,
            seasons=None,
            ridge_lambda=float(candidate.get("ridge_lambda", 0.1)),
            shrinkage_strength=float(candidate.get("shrinkage_strength", 0.25)),
            season_shrinkage_strength=8.0,
            K=4,
        )
        calibration.update(
            {
                "support_affine_calibration": affine,
                "support_calibration_dof": 2.0 + float(affine.get("effective_calibration_dof", 4.0) or 4.0),
            }
        )
        return calibration
    if candidate_type in {"global_affine", "seasonal_affine"}:
        use_seasons = train_seasons if candidate_type == "seasonal_affine" else None
        affine = calibrate_residual_affine(
            train_samples_s,
            train_samples_r,
            seasons=use_seasons,
            ridge_lambda=float(candidate.get("ridge_lambda", 0.1)),
            shrinkage_strength=float(candidate.get("shrinkage_strength", 0.25)),
            season_shrinkage_strength=float(candidate.get("season_shrinkage_strength", 8.0)),
            K=12 if candidate_type == "seasonal_affine" else 4,
        )
        calibration.update(
            {
                "support_affine_calibration": affine,
                "support_calibration_dof": float(affine.get("effective_calibration_dof", 4.0) or 4.0),
            }
        )
        return calibration
    raise ValueError(f"unsupported v13 candidate_type={candidate_type!r}")


def decide_v13_k12_aggressive_calibration_pool_gate(
    *,
    selected_candidate_id: str,
    support_candidate_pool: List[Dict[str, Any]],
    cv_summary: Dict[str, Any],
    support_calibration_dof: float,
    rootzone_tolerance: float = 5e-6,
    min_cycle_improvement_fraction: float = 0.5,
    best_rejected_candidate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cv_delta = _float_or_none(cv_summary.get("cv_objective_delta"))
    cv_se = _float_or_none(cv_summary.get("cv_objective_delta_se"))
    cv_t = _float_or_none(cv_summary.get("cv_objective_delta_t"))
    rootzone_delta = _float_or_none(cv_summary.get("cv_rootzone_delta"))
    cv_fraction = float(cv_summary.get("cv_cycle_improvement_fraction", 0.0) or 0.0)
    reject_reasons = _v13_k12_candidate_gate_reject_reasons(
        cv_summary,
        rootzone_tolerance=float(rootzone_tolerance),
        min_cycle_improvement_fraction=float(min_cycle_improvement_fraction),
    )
    accepted = not reject_reasons
    reference_obj = cv_summary.get("cv_reference_objective")
    candidate_obj = cv_summary.get("cv_candidate_objective")
    return {
        "support_gate_enabled": True,
        "support_gate_label_source": "target_support_only",
        "support_gate_policy_role": "target_support_only_v13_k12_aggressive_calibration_pool_diagnostic",
        "support_selection_objective": "k12_aggressive_nested_cv_calibration_pool_support_only",
        "target_eval_usage": "final_eval_only_no_selection",
        "selected_support_candidate_id": str(selected_candidate_id or ""),
        "support_candidate_pool": [dict(candidate) for candidate in support_candidate_pool],
        "support_gate_min_delta": 0.0,
        "support_gate_rootzone_tolerance": float(rootzone_tolerance),
        "support_gate_cycle_improvement_min_fraction": float(min_cycle_improvement_fraction),
        "support_cycle_improvement_fraction": cv_fraction,
        "support_cv_objective_before": reference_obj,
        "support_cv_objective_after": candidate_obj if accepted else reference_obj,
        "support_cv_objective_delta": 0.0 if (not accepted and reference_obj is not None) else cv_delta,
        "support_cv_objective_delta_se": cv_se,
        "support_cv_objective_delta_t": cv_t,
        "support_cv_rootzone_delta": 0.0 if not accepted else rootzone_delta,
        "support_cv_nested_k4_objective_delta": cv_summary.get("cv_nested_k4_objective_delta"),
        "support_cv_added_objective_delta": cv_summary.get("cv_added_objective_delta"),
        "k12_vs_k4_cv_objective_delta": cv_delta,
        "k12_vs_k4_cv_rootzone_delta": rootzone_delta,
        "support_calibration_dof": float(support_calibration_dof),
        "best_rejected_support_candidate": dict(best_rejected_candidate or {}),
        "k12_reference_policy": "k4_safe_nested_reference",
        "support_gate_status": (
            "support_only_v13_k12_aggressive_calibration_pool_accepted"
            if accepted
            else "support_only_v13_k12_aggressive_calibration_pool_fallback_to_k4_reference"
        ),
        "stage3_posterior_decision": "accepted" if accepted else "fallback_to_k4_reference",
        "support_gate_reject_reason": reject_reasons,
    }


def _v13_k12_candidate_gate_reject_reasons(
    cv_summary: Dict[str, Any],
    *,
    rootzone_tolerance: float = 5e-6,
    min_cycle_improvement_fraction: float = 0.5,
) -> List[str]:
    cv_delta = _float_or_none(cv_summary.get("cv_objective_delta"))
    rootzone_delta = _float_or_none(cv_summary.get("cv_rootzone_delta"))
    cv_fraction = float(cv_summary.get("cv_cycle_improvement_fraction", 0.0) or 0.0)
    reject_reasons: List[str] = []
    if cv_delta is None:
        reject_reasons.append("missing_k12_vs_k4_cv_objective")
    elif cv_delta >= 0.0:
        reject_reasons.append("k12_candidate_not_better_than_k4_reference_cv")
    if rootzone_delta is None:
        reject_reasons.append("missing_k12_vs_k4_cv_rootzone_guard")
    elif rootzone_delta > float(rootzone_tolerance):
        reject_reasons.append("rootzone_cv_regression_gt_5e-6")
    if cv_fraction + 1e-12 < float(min_cycle_improvement_fraction):
        reject_reasons.append("insufficient_support_cycle_improvement_fraction")
    return reject_reasons


def _v13_cv_candidate_objective(result: Dict[str, Any]) -> float:
    value = _float_or_none(result.get("cv_summary", {}).get("cv_candidate_objective"))
    if value is None:
        return float("inf")
    return float(value)


def _v13_public_rejected_candidate(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if result is None:
        return {}
    return {
        "candidate_id": str(result.get("candidate_id", "")),
        "candidate_type": str(result.get("candidate_type", "")),
        "cv_summary": dict(result.get("cv_summary", {}) or {}),
        "gate_status": str(result.get("gate_status", "")),
        "gate_reject_reason": list(result.get("gate_reject_reason", []) or []),
        "support_calibration_dof": float(result.get("support_calibration_dof", 0.0) or 0.0),
    }


def _select_v13_k12_support_candidate_results(
    candidate_results: List[Dict[str, Any]],
    *,
    rootzone_tolerance: float = 5e-6,
    min_cycle_improvement_fraction: float = 0.5,
) -> Dict[str, Any]:
    if not candidate_results:
        raise RuntimeError("v13 aggressive calibration pool produced no candidate results")
    selected_result: Optional[Dict[str, Any]] = None
    best_rejected_result: Optional[Dict[str, Any]] = None
    for result in candidate_results:
        gate_reject_reasons = _v13_k12_candidate_gate_reject_reasons(
            dict(result.get("cv_summary", {}) or {}),
            rootzone_tolerance=float(rootzone_tolerance),
            min_cycle_improvement_fraction=float(min_cycle_improvement_fraction),
        )
        result["gate_reject_reason"] = gate_reject_reasons
        result["gate_status"] = (
            "support_only_v13_k12_aggressive_calibration_pool_eligible"
            if not gate_reject_reasons
            else "support_only_v13_k12_aggressive_calibration_pool_rejected"
        )
        if (
            not gate_reject_reasons
            and _v13_cv_candidate_objective(result) < _v13_cv_candidate_objective(selected_result or {})
        ):
            selected_result = result
        if (
            gate_reject_reasons
            and _v13_cv_candidate_objective(result) < _v13_cv_candidate_objective(best_rejected_result or {})
        ):
            best_rejected_result = result

    fallback_result = min(candidate_results, key=_v13_cv_candidate_objective)
    if selected_result is None:
        best_rejected_result = fallback_result
    return {
        "selected_result": selected_result,
        "gate_target_result": selected_result if selected_result is not None else fallback_result,
        "eligible_support_candidate_count": sum(
            1 for result in candidate_results if not list(result.get("gate_reject_reason", []) or [])
        ),
        "best_rejected_support_candidate": _v13_public_rejected_candidate(best_rejected_result),
    }


@torch.no_grad()
def calibrate_support_gain_v13_k12_aggressive_pool_from_loader(
    *,
    state: FewShotAdaptationState,
    loader: DataLoader,
    device: torch.device,
    target_context_prompt_state: Dict[str, Any],
    k4_reference_checkpoint: str,
    surface_weight: float,
    rootzone_weight: float,
    support_nesting_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Select a K12 support-only calibration candidate against run-local K4."""
    cycle_count = _support_cycle_count(loader)
    if cycle_count < 12:
        raise ValueError(f"v13 K12 aggressive calibration pool expects 12-cycle support, got {cycle_count}")
    if not k4_reference_checkpoint:
        raise ValueError("v13 K12 aggressive calibration pool requires --k4_reference_checkpoint")
    support_nesting_metadata = dict(support_nesting_metadata or {})
    full_samples_s, full_samples_r = _collect_support_gain_samples_from_loader(
        state=state,
        loader=loader,
        device=device,
        target_context_prompt_state=target_context_prompt_state,
    )
    months = _collect_support_months_from_loader(loader)
    seasons = [_support_sample_season_from_month(month) for month in months]
    if len(seasons) < len(full_samples_s):
        seasons.extend([""] * (len(full_samples_s) - len(seasons)))

    k4_alpha_s, k4_alpha_r = load_k4_reference_residual_gain_alphas(k4_reference_checkpoint)
    k4_affine = load_k4_reference_support_affine_calibration(k4_reference_checkpoint)
    reference_calibration = {
        "candidate_id": "run_local_k4_reference",
        "candidate_type": "k4_checkpoint_final_prediction_form",
        "alpha_surface": float(k4_alpha_s),
        "alpha_rootzone": float(k4_alpha_r),
        "support_affine_calibration": k4_affine,
    }
    full_reference_score = _support_calibrated_samples_score(
        samples_s=full_samples_s,
        samples_r=full_samples_r,
        seasons=seasons,
        calibration=reference_calibration,
        surface_weight=surface_weight,
        rootzone_weight=rootzone_weight,
    )
    pool = diagnostic_support_gain_v13_k12_calibration_pool(12)
    nested_count = 4
    nested_indices = list(range(0, min(4, cycle_count)))
    added_indices = list(range(min(4, cycle_count), cycle_count))
    candidate_results: List[Dict[str, Any]] = []

    def _subset(items: List[Any], indices: List[int]) -> List[Any]:
        return [items[index] for index in indices if index < len(items)]

    for candidate in pool:
        full_calibration = _v13_fit_candidate_calibration(
            candidate,
            full_samples_s,
            full_samples_r,
            seasons,
            surface_weight=surface_weight,
            rootzone_weight=rootzone_weight,
        )
        full_score = _support_calibrated_samples_score(
            samples_s=full_samples_s,
            samples_r=full_samples_r,
            seasons=seasons,
            calibration=full_calibration,
            surface_weight=surface_weight,
            rootzone_weight=rootzone_weight,
        )
        cv_folds: List[Dict[str, Any]] = []
        for train_indices, holdout_indices, policy in (
            (added_indices, nested_indices, "train_added8_validate_nested_k4"),
            (nested_indices, added_indices, "train_nested_k4_validate_added8"),
        ):
            if not train_indices or not holdout_indices:
                continue
            fold_calibration = _v13_fit_candidate_calibration(
                candidate,
                _subset(full_samples_s, train_indices),
                _subset(full_samples_r, train_indices),
                _subset(seasons, train_indices),
                surface_weight=surface_weight,
                rootzone_weight=rootzone_weight,
            )
            for holdout in holdout_indices:
                if holdout >= min(len(full_samples_s), len(full_samples_r)):
                    continue
                candidate_score = _support_calibrated_samples_score(
                    samples_s=[full_samples_s[holdout]],
                    samples_r=[full_samples_r[holdout]],
                    seasons=[seasons[holdout] if holdout < len(seasons) else ""],
                    calibration=fold_calibration,
                    surface_weight=surface_weight,
                    rootzone_weight=rootzone_weight,
                )
                reference_score = _support_calibrated_samples_score(
                    samples_s=[full_samples_s[holdout]],
                    samples_r=[full_samples_r[holdout]],
                    seasons=[seasons[holdout] if holdout < len(seasons) else ""],
                    calibration=reference_calibration,
                    surface_weight=surface_weight,
                    rootzone_weight=rootzone_weight,
                )
                candidate_obj = _float_or_none(candidate_score.get("standard_support_objective_full_support"))
                reference_obj = _float_or_none(reference_score.get("standard_support_objective_full_support"))
                candidate_rootzone = _float_or_none(candidate_score.get("standard_support_rootzone_loss_full_support"))
                reference_rootzone = _float_or_none(reference_score.get("standard_support_rootzone_loss_full_support"))
                candidate_surface = _float_or_none(candidate_score.get("standard_support_surface_loss_full_support"))
                reference_surface = _float_or_none(reference_score.get("standard_support_surface_loss_full_support"))
                cv_folds.append(
                    {
                        "holdout_index": int(holdout),
                        "train_policy": policy,
                        "candidate_objective": candidate_obj,
                        "reference_objective": reference_obj,
                        "objective_delta": (
                            None if candidate_obj is None or reference_obj is None else candidate_obj - reference_obj
                        ),
                        "candidate_surface": candidate_surface,
                        "reference_surface": reference_surface,
                        "surface_delta": (
                            None
                            if candidate_surface is None or reference_surface is None
                            else candidate_surface - reference_surface
                        ),
                        "candidate_rootzone": candidate_rootzone,
                        "reference_rootzone": reference_rootzone,
                        "rootzone_delta": (
                            None
                            if candidate_rootzone is None or reference_rootzone is None
                            else candidate_rootzone - reference_rootzone
                        ),
                        "fold_candidate_id": str(candidate.get("candidate_id", "")),
                        "fold_candidate_type": str(candidate.get("candidate_type", "")),
                    }
                )
        cv_summary = summarize_v11_cv_folds(cv_folds, nested_count=nested_count)
        deltas = [fold.get("objective_delta") for fold in cv_folds]
        cv_summary.update(
            {
                "cv_objective_delta_std": summarize_support_gain_cv_folds(
                    cv_folds,
                    nested_count=nested_count,
                ).get("cv_objective_delta_std"),
                "cv_objective_delta_se": summarize_support_gain_cv_folds(
                    cv_folds,
                    nested_count=nested_count,
                ).get("cv_objective_delta_se"),
                "cv_objective_delta_t": summarize_support_gain_cv_folds(
                    cv_folds,
                    nested_count=nested_count,
                ).get("cv_objective_delta_t"),
                "cv_cycle_improvement_fraction": _fraction_nonpositive(deltas, tolerance=0.0),
            }
        )
        result = {
            **dict(candidate),
            "target_eval_usage": "final_eval_only_no_selection",
            "label_source": "target_support_only",
            "full_calibration": full_calibration,
            "score": full_score,
            "cv_folds": cv_folds,
            "cv_summary": cv_summary,
            "support_calibration_dof": float(full_calibration.get("support_calibration_dof", 0.0) or 0.0),
        }
        candidate_results.append(result)

    selection = _select_v13_k12_support_candidate_results(
        candidate_results,
        rootzone_tolerance=5e-6,
        min_cycle_improvement_fraction=0.5,
    )
    gate_target_result = selection["gate_target_result"]
    gate = decide_v13_k12_aggressive_calibration_pool_gate(
        selected_candidate_id=str(gate_target_result.get("candidate_id", "")),
        support_candidate_pool=pool,
        cv_summary=dict(gate_target_result.get("cv_summary", {}) or {}),
        support_calibration_dof=float(gate_target_result.get("support_calibration_dof", 0.0) or 0.0),
        rootzone_tolerance=5e-6,
        min_cycle_improvement_fraction=0.5,
        best_rejected_candidate=dict(selection.get("best_rejected_support_candidate", {}) or {}),
    )
    accepted = str(gate.get("stage3_posterior_decision")) == "accepted"
    final_calibration = dict(gate_target_result.get("full_calibration", {}) or {})
    selected_affine = dict(final_calibration.get("support_affine_calibration", {}) or {})
    selected_alpha_s = float(final_calibration.get("alpha_surface", 1.0))
    selected_alpha_r = float(final_calibration.get("alpha_rootzone", 1.0))
    if not accepted:
        selected_alpha_s = float(k4_alpha_s)
        selected_alpha_r = float(k4_alpha_r)
        selected_affine = dict(k4_affine)
    public_results: List[Dict[str, Any]] = []
    for result in candidate_results:
        public_results.append(
            {
                key: value
                for key, value in result.items()
                if key != "full_calibration"
            }
            | {"full_calibration": dict(result.get("full_calibration", {}) or {})}
        )
    selected_score = (
        dict(gate_target_result.get("score", {}) or {})
        if accepted
        else dict(full_reference_score)
    )
    return {
        "schema_version": "diagnostic_support_gain_v13_k12_aggressive_calibration_pool_v1",
        "calibration_mode": "target_support_gain_v13_k12_aggressive_calibration_pool",
        "status": "accepted" if accepted else "fallback_to_k4_reference",
        "stage3_kshot_mode": "diagnostic_support_gain_v13_k12_aggressive_calibration_pool",
        "label_source": "target_support_only",
        "target_eval_usage": "final_eval_only_no_selection",
        "support_selection_objective": "k12_aggressive_nested_cv_calibration_pool_support_only",
        "support_candidate_pool": [dict(candidate) for candidate in pool],
        "support_candidate_results": public_results,
        "eligible_support_candidate_count": int(selection["eligible_support_candidate_count"]),
        "selected_support_candidate_id": str(gate_target_result.get("candidate_id", "")) if accepted else "k4_reference_fallback",
        "selected_support_candidate_id_before_gate": str(gate_target_result.get("candidate_id", "")),
        "best_rejected_support_candidate": dict(gate.get("best_rejected_support_candidate", {}) or {}),
        "selected_candidate": {
            key: value
            for key, value in gate_target_result.items()
            if key not in {"full_calibration"}
        },
        "selected_full_calibration": final_calibration,
        "support_gain_cv_summary": dict(gate_target_result.get("cv_summary", {}) or {}),
        "support_gain_cv_folds": list(gate_target_result.get("cv_folds", []) or []),
        "support_gain_gate": gate,
        "support_count": len(full_samples_s),
        "best_alpha_surface": selected_alpha_s,
        "best_alpha_rootzone": selected_alpha_r,
        "selected_alpha_before_gate_surface": float(final_calibration.get("alpha_surface", 1.0)),
        "selected_alpha_before_gate_rootzone": float(final_calibration.get("alpha_rootzone", 1.0)),
        "selected_alpha_after_gate_surface": selected_alpha_s,
        "selected_alpha_after_gate_rootzone": selected_alpha_r,
        "support_affine_calibration": selected_affine,
        "effective_calibration_dof": float(
            selected_affine.get("effective_calibration_dof", 0.0)
            if selected_affine
            else 0.0
        ),
        "support_calibration_dof": (
            float(gate_target_result.get("support_calibration_dof", 0.0) or 0.0)
            if accepted
            else 0.0
        ),
        "k4_reference_checkpoint": str(k4_reference_checkpoint),
        "k4_reference_gain_alpha_surface": float(k4_alpha_s),
        "k4_reference_gain_alpha_rootzone": float(k4_alpha_r),
        "k4_reference_support_affine_calibration": dict(k4_affine),
        "k4_reference_score": dict(full_reference_score),
        "selected_score": selected_score,
        "target_eval_usage_note": "final_eval_only_no_selection",
        **support_nesting_metadata,
    }


@torch.no_grad()
def calibrate_support_residual_gain_v12_nested_cv_from_loader(
    *,
    state: FewShotAdaptationState,
    loader: DataLoader,
    device: torch.device,
    target_context_prompt_state: Dict[str, Any],
    K: int,
    alpha_grid: List[float],
    surface_weight: float,
    rootzone_weight: float,
    support_nesting_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Support-only residual-gain calibration with cycle-level CV no-harm gates."""
    cycle_count = _support_cycle_count(loader)
    nested_count = 4 if int(K) == 12 else cycle_count
    if int(K) == 12 and cycle_count < 12:
        raise ValueError(f"v12 support gain expects nested 12-cycle support, got {cycle_count}")
    if int(K) == 4 and cycle_count < 4:
        raise ValueError(f"v12 support gain expects 4-cycle support, got {cycle_count}")

    full_samples_s, full_samples_r = _collect_support_gain_samples_from_loader(
        state=state,
        loader=loader,
        device=device,
        target_context_prompt_state=target_context_prompt_state,
    )
    base_calibration = calibrate_residual_gain(
        full_samples_s,
        full_samples_r,
        alpha_grid,
        selection_rule="support_uncertainty_stable_high_alpha_with_dual_guard",
    )
    if not base_calibration:
        base_calibration = {
            "best_alpha_surface": 1.0,
            "best_alpha_rootzone": 1.0,
            "best_alpha_raw": 1.0,
            "stable_candidate_alphas": [1.0],
            "selection_margin": 0.0,
            "per_alpha_results": {},
            "selection_rule": "support_uncertainty_stable_high_alpha_with_dual_guard",
        }

    def _fit_alpha(train_indices: List[int]) -> float:
        train_loader = _support_subset_loader(loader, train_indices)
        train_samples_s, train_samples_r = _collect_support_gain_samples_from_loader(
            state=state,
            loader=train_loader,
            device=device,
            target_context_prompt_state=target_context_prompt_state,
        )
        fold_calibration = calibrate_residual_gain(
            train_samples_s,
            train_samples_r,
            alpha_grid,
            selection_rule="support_uncertainty_stable_high_alpha_with_dual_guard",
        )
        return float(fold_calibration.get("best_alpha_surface", 1.0) if fold_calibration else 1.0)

    folds: List[Dict[str, Any]] = []
    all_indices = list(range(cycle_count))
    if int(K) == 4:
        fold_specs = [
            ([idx for idx in all_indices if idx != holdout], [holdout], "leave_one_cycle_out")
            for holdout in all_indices
        ]
    else:
        nested_indices = list(range(0, min(4, cycle_count)))
        added_indices = list(range(min(4, cycle_count), cycle_count))
        fold_specs = [
            (added_indices, nested_indices, "train_added8_validate_nested_k4"),
            (nested_indices, added_indices, "train_nested_k4_validate_added8"),
        ]
    for train_indices, holdout_indices, policy in fold_specs:
        if not train_indices or not holdout_indices:
            continue
        alpha = _fit_alpha(train_indices)
        for holdout in holdout_indices:
            if holdout >= min(len(full_samples_s), len(full_samples_r)):
                continue
            candidate = _support_gain_pair_objective(
                full_samples_s[holdout],
                full_samples_r[holdout],
                alpha_surface=alpha,
                alpha_rootzone=alpha,
                surface_weight=surface_weight,
                rootzone_weight=rootzone_weight,
            )
            reference = _support_gain_pair_objective(
                full_samples_s[holdout],
                full_samples_r[holdout],
                alpha_surface=1.0,
                alpha_rootzone=1.0,
                surface_weight=surface_weight,
                rootzone_weight=rootzone_weight,
            )
            delta = None if candidate is None or reference is None else float(candidate) - float(reference)
            folds.append(
                {
                    "holdout_index": int(holdout),
                    "train_policy": policy,
                    "fold_selected_alpha": float(alpha),
                    "candidate_objective": candidate,
                    "reference_objective": reference,
                    "objective_delta": delta,
                }
            )

    cv_summary = summarize_support_gain_cv_folds(folds, nested_count=nested_count)
    gate_defaults = {
        "min_delta": 5e-6 if int(K) == 4 else 2e-6,
        "min_fraction": 1.0 if int(K) == 4 else 0.75,
    }
    gate = decide_support_gain_v12_gate(
        K=int(K),
        cv_summary=cv_summary,
        selected_alpha=float(base_calibration.get("best_alpha_surface", 1.0)),
        min_delta=gate_defaults["min_delta"],
        min_cycle_improvement_fraction=gate_defaults["min_fraction"],
    )
    accepted = str(gate["stage3_posterior_decision"]) == "accepted"
    calibration = dict(base_calibration)
    calibration.update(
        {
            "calibration_mode": "target_support_residual_gain_v12_nested_cv",
            "status": "accepted" if accepted else "rejected_to_identity",
            "label_source": "target_support_only",
            "target_eval_usage": "final_eval_only_no_selection",
            "selection_rule": "support_uncertainty_nested_cv_no_harm",
            "support_selection_objective": "nested_cv_raw_increment_wrmse_support_gain_target_support_only",
            "support_gain_cv_summary": cv_summary,
            "support_gain_cv_folds": folds,
            "support_gain_gate": gate,
            "support_count": len(full_samples_s),
            "alpha_grid": [float(alpha) for alpha in alpha_grid],
            "best_alpha_surface": float(base_calibration.get("best_alpha_surface", 1.0)) if accepted else 1.0,
            "best_alpha_rootzone": float(base_calibration.get("best_alpha_rootzone", 1.0)) if accepted else 1.0,
            "selected_alpha_before_gate": float(base_calibration.get("best_alpha_surface", 1.0)),
            "selected_alpha_after_gate": float(base_calibration.get("best_alpha_surface", 1.0)) if accepted else 1.0,
        }
    )
    calibration.update(dict(support_nesting_metadata or {}))
    return calibration


@torch.no_grad()
def calibrate_support_residual_affine_from_loader(
    *,
    state: FewShotAdaptationState,
    loader: DataLoader,
    device: torch.device,
    target_context_prompt_state: Dict[str, Any],
    K: int,
    support_nesting_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fit target_support-only residual affine calibration from a support loader."""
    was_training = state.model.training
    state.model.eval()
    samples_s: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    samples_r: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    seasons: List[str] = []
    normalize_increment = state.normalization.get("inc_mean") is not None
    try:
        for batch in loader:
            pred_norm = _ridge_forward_prediction(
                state=state,
                batch=batch,
                device=device,
                target_context_prompt_state=target_context_prompt_state,
            )
            pred = _denormalize_increment(pred_norm, state.normalization, normalize_increment)
            pred_np = pred.detach().cpu().numpy()
            true_s = batch["increment_surface"].detach().cpu().numpy()
            true_r = batch["increment_rootzone"].detach().cpu().numpy()
            forecast_s = batch["forecast_surface"].detach().cpu().numpy()
            forecast_r = batch["forecast_rootzone"].detach().cpu().numpy()
            mask = batch["loss_mask"].detach().cpu().numpy()
            latw = batch.get("latitude_weight")
            if latw is None:
                latw_np = np.ones_like(mask, dtype=np.float32)
            else:
                latw_np = latw.detach().cpu().numpy()
            month_values = [int(value) for value in batch["months"].detach().cpu().view(-1).tolist()]
            for idx in range(pred_np.shape[0]):
                samples_s.append((
                    pred_np[idx, 0].astype(np.float32),
                    true_s[idx].astype(np.float32),
                    forecast_s[idx].astype(np.float32),
                    mask[idx].astype(np.float32),
                    latw_np[idx].astype(np.float32),
                ))
                samples_r.append((
                    pred_np[idx, 1].astype(np.float32),
                    true_r[idx].astype(np.float32),
                    forecast_r[idx].astype(np.float32),
                    mask[idx].astype(np.float32),
                    latw_np[idx].astype(np.float32),
                ))
                month = month_values[idx] if idx < len(month_values) else 1
                if month in (12, 1, 2):
                    seasons.append("DJF")
                elif month in (3, 4, 5):
                    seasons.append("MAM")
                elif month in (6, 7, 8):
                    seasons.append("JJA")
                else:
                    seasons.append("SON")
    finally:
        state.model.train(was_training)

    nesting = dict(support_nesting_metadata or {})
    calibration = calibrate_residual_affine(
        samples_s,
        samples_r,
        seasons=seasons if int(K) == 12 else None,
        ridge_lambda=0.1,
        shrinkage_strength=0.01,
        season_shrinkage_strength=8.0,
        K=int(K),
        support_nesting_policy=str(nesting.get("support_nesting_policy", "")),
        nested_support_dates_hash=str(nesting.get("nested_support_dates_hash", "")),
    )
    calibration.update(nesting)
    return calibration


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
    source_trust_query = compose_target_context_source_trust_query_from_state(
        target_context_prompt_state,
        months,
        device=device,
    )
    reliability_features = compose_target_context_reliability_features_from_state(
        target_context_prompt_state,
        months,
        device=device,
    )
    return _model_forward(
        state.model,
        x_norm,
        z,
        months,
        x,
        reliability_features=reliability_features,
        source_trust_bank=source_trust_bank_for_few_shot_state(state),
        source_trust_query=source_trust_query,
    )


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
    ridge_weighting: str,
    surface_weight: float,
    rootzone_weight: float,
    use_lat_weighted_loss: bool,
    finite_difference_eps: float = 1e-3,
    skip_cycle_indices: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    """Solve a local linear ridge update for adapter coefficient residual logits."""
    if str(ridge_weighting) not in RIDGE_WEIGHTINGS:
        raise ValueError(f"unsupported ridge_weighting={ridge_weighting!r}; expected one of {list(RIDGE_WEIGHTINGS)}")
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
    weight_blocks: List[torch.Tensor] = []
    support_count = 0
    cycle_index = 0
    skipped_cycle_indices = {int(index) for index in (skip_cycle_indices or [])}
    skipped_support_count = 0
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
            if skipped_cycle_indices:
                kept_indices: List[int] = []
                batch_size = int(batch["x"].shape[0])
                for local_idx in range(batch_size):
                    if cycle_index in skipped_cycle_indices:
                        skipped_support_count += 1
                    else:
                        kept_indices.append(local_idx)
                    cycle_index += 1
                if not kept_indices:
                    continue
                index_tensor = torch.as_tensor(kept_indices, dtype=torch.long)
                filtered_batch: Dict[str, Any] = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor) and int(value.shape[0]) == batch_size:
                        filtered_batch[key] = value.index_select(0, index_tensor)
                    else:
                        filtered_batch[key] = value
                batch = filtered_batch
            else:
                cycle_index += int(batch["x"].shape[0])
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
            if str(ridge_weighting) == "cycle_variable_balanced_huber":
                weights = _ridge_balanced_huber_observation_weights(
                    batch=batch,
                    pred=pred_base,
                    residual=residual_flat,
                    surface_weight=surface_weight,
                    rootzone_weight=rootzone_weight,
                    use_lat_weighted_loss=use_lat_weighted_loss,
                    delta=1.0,
                )
            else:
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
            selected_weight = _ridge_select_feature_observations(
                weights.detach().float().reshape(-1).cpu(),
                valid_observation_mask=valid_observation_mask,
                max_feature_pixels=0,
            ).double()
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
            weight_blocks.append(selected_weight)
            set_coefficient_residual_vector(state.model, base_vector)
    finally:
        set_coefficient_residual_vector(state.model, base_vector)

    if design_blocks and residual_blocks:
        design_all = torch.cat(design_blocks, dim=0)
        residual_all = torch.cat(residual_blocks, dim=0)
        weight_all = torch.cat(weight_blocks, dim=0) if weight_blocks else None
        design_all, residual_all, weight_all = _ridge_subsample_rows(
            design_all,
            residual_all,
            max_feature_pixels=int(ridge_max_feature_pixels),
            weights=weight_all,
        )
        feature_observation_count = int(residual_all.numel())
        xtx += design_all.T @ design_all
        xtr += design_all.T @ residual_all
        column_sq_sum += design_all.square().sum(dim=0)
        if str(ridge_weighting) == "cycle_variable_balanced_huber" and weight_all is not None:
            normalizer_value = float(weight_all.double().sum().item())
        else:
            normalizer_value = float(max(1, feature_observation_count))
    else:
        feature_observation_count = 0
        normalizer_value = 1.0

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
            "skipped_support_count": skipped_support_count,
            "skipped_cycle_indices": sorted(skipped_cycle_indices),
            "masked_pixel_count": masked_pixel_count,
            "masked_observation_count": masked_observation_count,
            "feature_pixel_count": 0,
            "feature_observation_count": feature_observation_count,
            "feature_dim": feature_dim,
            "ridge_max_feature_pixels": int(ridge_max_feature_pixels),
            "ridge_standardize_features": bool(ridge_standardize_features),
            "ridge_weighting": str(ridge_weighting),
            "ridge_normalizer": None,
            "condition_number": None,
            "rank": 0,
            "trust_region_clipped": False,
            "coefficient_norm_clipped": False,
        }
        return diagnostics

    normalizer = float(max(1e-12, normalizer_value))
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
        "solver": "linearized_coeff_ridge",
        "parameter_scope": "adapter_coefficient_residuals_only",
        "label_source": "target_support_only",
        "target_eval_usage": "final_eval_only_no_selection",
        "jacobian_storage": "streamed_small_design_no_full_jacobian_artifact",
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
        "skipped_support_count": skipped_support_count,
        "skipped_cycle_indices": sorted(skipped_cycle_indices),
        "masked_pixel_count": masked_pixel_count,
        "masked_observation_count": masked_observation_count,
        "feature_pixel_count": int((feature_observation_count + 1) // 2),
        "feature_observation_count": feature_observation_count,
        "feature_dim": feature_dim,
        "ridge_max_feature_pixels": int(ridge_max_feature_pixels),
        "ridge_standardize_features": bool(ridge_standardize_features),
        "ridge_weighting": str(ridge_weighting),
        "ridge_normalizer": float(normalizer),
        "feature_column_scale_min": float(column_scale.min().item()) if column_scale.numel() else None,
        "feature_column_scale_max": float(column_scale.max().item()) if column_scale.numel() else None,
        "trust_region_clipped": bool(trust_region_clipped),
        "coefficient_norm_clipped": bool(coefficient_norm_clipped),
        "effective_calibration_dof": float(solve.diagnostics.get("rank", 0) or 0),
    }
    return diagnostics


def run_v10_support_pool_adaptation(
    *,
    state: FewShotAdaptationState,
    loader: DataLoader,
    device: torch.device,
    target_context_prompt_state: Dict[str, Any],
    anchor_state: Dict[str, torch.Tensor],
    K: int,
    normalize_increment: bool,
    ridge_max_feature_pixels: int,
    ridge_standardize_features: bool,
    surface_weight: float,
    rootzone_weight: float,
    use_lat_weighted_loss: bool,
    k4_reference_state: Optional[Dict[str, torch.Tensor]] = None,
    k4_reference_adapt_mix_rho: Optional[float] = None,
) -> Dict[str, Any]:
    """Solve and score the fixed v10 support-pool candidates from one anchor."""
    pool = diagnostic_linearized_coeff_ridge_v10_support_pool(K)
    if not pool:
        raise ValueError(f"v10 support pool is not defined for K={K}")
    original_state = extract_target_adapter_state(state.model)
    candidate_results: List[Dict[str, Any]] = []
    selected_result: Optional[Dict[str, Any]] = None
    k0_anchor_objective = exact_mixed_support_objective_from_loader(
        state=state,
        loader=loader,
        device=device,
        target_context_prompt_state=target_context_prompt_state,
        anchor_state=anchor_state,
        candidate_state=anchor_state,
        rho=0.0,
        normalize_increment=normalize_increment,
        surface_weight=surface_weight,
        rootzone_weight=rootzone_weight,
        use_lat_weighted_loss=use_lat_weighted_loss,
        support_weighting="global_pixel_l2",
    )
    k4_reference_objective: Dict[str, Any] = {}
    k4_reference_nested_k4_objective: Dict[str, Any] = {}
    if int(K) == 12 and k4_reference_state is not None:
        rho = 1.0 if k4_reference_adapt_mix_rho is None else float(k4_reference_adapt_mix_rho)
        k4_reference_objective = exact_mixed_support_objective_from_loader(
            state=state,
            loader=loader,
            device=device,
            target_context_prompt_state=target_context_prompt_state,
            anchor_state=anchor_state,
            candidate_state=k4_reference_state,
            rho=rho,
            normalize_increment=normalize_increment,
            surface_weight=surface_weight,
            rootzone_weight=rootzone_weight,
            use_lat_weighted_loss=use_lat_weighted_loss,
            support_weighting="global_pixel_l2",
        )
        k4_reference_nested_k4_objective = exact_mixed_support_objective_from_loader(
            state=state,
            loader=loader,
            device=device,
            target_context_prompt_state=target_context_prompt_state,
            anchor_state=anchor_state,
            candidate_state=k4_reference_state,
            rho=rho,
            normalize_increment=normalize_increment,
            surface_weight=surface_weight,
            rootzone_weight=rootzone_weight,
            use_lat_weighted_loss=use_lat_weighted_loss,
            support_weighting="global_pixel_l2",
            max_cycles=4,
        )
    try:
        for candidate in pool:
            ridge_kwargs = _v10_candidate_to_ridge_kwargs(candidate)
            apply_target_adapter_state(state.model, anchor_state)
            diagnostics = run_ridge_coeff_adaptation(
                state=state,
                loader=loader,
                device=device,
                target_context_prompt_state=target_context_prompt_state,
                normalize_increment=normalize_increment,
                ridge_lambda=float(ridge_kwargs["ridge_lambda"]),
                ridge_clip_coeff_norm=float(ridge_kwargs["ridge_clip_coeff_norm"]),
                ridge_trust_region_radius=float(ridge_kwargs["ridge_trust_region_radius"]),
                ridge_max_feature_pixels=int(ridge_max_feature_pixels),
                ridge_standardize_features=bool(ridge_standardize_features),
                ridge_weighting=str(ridge_kwargs["ridge_weighting"]),
                surface_weight=surface_weight,
                rootzone_weight=rootzone_weight,
                use_lat_weighted_loss=use_lat_weighted_loss,
            )
            raw_state = extract_target_adapter_state(state.model)
            final_state = interpolate_target_adapter_state(
                anchor_state,
                raw_state,
                float(ridge_kwargs["anchor_alpha"]),
            )
            score = exact_mixed_support_objective_from_loader(
                state=state,
                loader=loader,
                device=device,
                target_context_prompt_state=target_context_prompt_state,
                anchor_state=anchor_state,
                candidate_state=final_state,
                rho=float(ridge_kwargs["adapt_mix_rho"]),
                normalize_increment=normalize_increment,
                surface_weight=surface_weight,
                rootzone_weight=rootzone_weight,
                use_lat_weighted_loss=use_lat_weighted_loss,
                support_weighting="global_pixel_l2",
            )
            nested_k4_score = exact_mixed_support_objective_from_loader(
                state=state,
                loader=loader,
                device=device,
                target_context_prompt_state=target_context_prompt_state,
                anchor_state=anchor_state,
                candidate_state=final_state,
                rho=float(ridge_kwargs["adapt_mix_rho"]),
                normalize_increment=normalize_increment,
                surface_weight=surface_weight,
                rootzone_weight=rootzone_weight,
                use_lat_weighted_loss=use_lat_weighted_loss,
                support_weighting="global_pixel_l2",
                max_cycles=4,
            )
            reference_for_cycles = k0_anchor_objective if int(K) == 4 else k4_reference_objective
            cycle_fraction = support_cycle_improvement_fraction(reference_for_cycles, score)
            result = {
                **dict(candidate),
                **ridge_kwargs,
                "support_selection_objective": "exact_mixed_target_support_only",
                "target_eval_usage": "final_eval_only_no_selection",
                "score": score,
                "nested_k4_score": nested_k4_score,
                "support_cycle_improvement_fraction": float(cycle_fraction),
                "ridge_diagnostics": diagnostics,
                "raw_state": raw_state,
                "final_state": final_state,
            }
            candidate_results.append(result)
            selected_obj = (
                None
                if selected_result is None
                else _float_or_none(selected_result.get("score", {}).get("standard_support_objective_full_support"))
            )
            result_obj = _float_or_none(score.get("standard_support_objective_full_support"))
            if selected_result is None:
                selected_result = result
            elif result_obj is not None and (selected_obj is None or result_obj < selected_obj):
                selected_result = result
    finally:
        apply_target_adapter_state(state.model, original_state)

    if selected_result is None:
        raise RuntimeError("v10 support pool produced no candidate results")
    apply_target_adapter_state(state.model, selected_result["raw_state"])
    public_results: List[Dict[str, Any]] = []
    for result in candidate_results:
        public = {
            key: value
            for key, value in result.items()
            if key not in {"raw_state", "final_state"}
        }
        public_results.append(public)
    selected_public = {
        key: value
        for key, value in selected_result.items()
        if key not in {"raw_state", "final_state"}
    }
    return {
        "schema_version": "diagnostic_linearized_coeff_ridge_v10_support_pool_v1",
        "stage3_kshot_mode": "diagnostic_linearized_coeff_ridge_v10_support_pool_nested",
        "label_source": "target_support_only",
        "target_eval_usage": "final_eval_only_no_selection",
        "support_selection_objective": "exact_mixed_target_support_only",
        "support_candidate_pool": [dict(candidate) for candidate in pool],
        "support_candidate_results": public_results,
        "selected_support_candidate_id": str(selected_result.get("candidate_id", "")),
        "selected_candidate": selected_public,
        "selected_ridge_diagnostics": dict(selected_result.get("ridge_diagnostics", {}) or {}),
        "selected_score": dict(selected_result.get("score", {}) or {}),
        "selected_nested_k4_score": dict(selected_result.get("nested_k4_score", {}) or {}),
        "selected_anchor_alpha": float(selected_result["anchor_alpha"]),
        "selected_adapt_mix_rho": float(selected_result["adapt_mix_rho"]),
        "selected_ridge_lambda": float(selected_result["ridge_lambda"]),
        "selected_ridge_clip_coeff_norm": float(selected_result["ridge_clip_coeff_norm"]),
        "selected_ridge_trust_region_radius": float(selected_result["ridge_trust_region_radius"]),
        "selected_ridge_weighting": str(selected_result["ridge_weighting"]),
        "support_cycle_improvement_fraction": float(
            selected_result.get("support_cycle_improvement_fraction", 0.0) or 0.0
        ),
        "k0_anchor_score": dict(k0_anchor_objective),
        "k4_reference_score": dict(k4_reference_objective),
        "k4_reference_nested_k4_score": dict(k4_reference_nested_k4_objective),
        "k4_reference_adapt_mix_rho": (
            None if k4_reference_adapt_mix_rho is None else float(k4_reference_adapt_mix_rho)
        ),
    }


def run_v11_loocv_support_pool_adaptation(
    *,
    state: FewShotAdaptationState,
    loader: DataLoader,
    device: torch.device,
    target_context_prompt_state: Dict[str, Any],
    anchor_state: Dict[str, torch.Tensor],
    K: int,
    normalize_increment: bool,
    ridge_max_feature_pixels: int,
    ridge_standardize_features: bool,
    surface_weight: float,
    rootzone_weight: float,
    use_lat_weighted_loss: bool,
    k4_reference_state: Optional[Dict[str, torch.Tensor]] = None,
    k4_reference_adapt_mix_rho: Optional[float] = None,
) -> Dict[str, Any]:
    """Solve and score v11 candidates using support-internal held-out cycles."""
    pool = diagnostic_linearized_coeff_ridge_v11_support_pool(K)
    if not pool:
        raise ValueError(f"v11 support pool is not defined for K={K}")
    cycle_count = _support_cycle_count(loader)
    if int(K) == 12 and cycle_count < 12:
        raise ValueError(f"v11 K12 expects nested 12-cycle support, got {cycle_count}")
    if int(K) == 4 and cycle_count < 4:
        raise ValueError(f"v11 K4 expects 4-cycle support, got {cycle_count}")
    nested_count = 4 if int(K) == 12 else cycle_count
    original_state = extract_target_adapter_state(state.model)
    candidate_results: List[Dict[str, Any]] = []
    selected_result: Optional[Dict[str, Any]] = None

    k0_anchor_score = mixed_raw_increment_wrmse_objective_from_loader(
        state=state,
        loader=loader,
        device=device,
        target_context_prompt_state=target_context_prompt_state,
        anchor_state=anchor_state,
        candidate_state=anchor_state,
        rho=0.0,
        normalize_increment=normalize_increment,
        surface_weight=surface_weight,
        rootzone_weight=rootzone_weight,
        use_lat_weighted_loss=use_lat_weighted_loss,
    )
    k4_reference_score: Dict[str, Any] = {}
    k4_reference_nested_k4_score: Dict[str, Any] = {}
    if int(K) == 12 and k4_reference_state is not None:
        rho = 1.0 if k4_reference_adapt_mix_rho is None else float(k4_reference_adapt_mix_rho)
        k4_reference_score = mixed_raw_increment_wrmse_objective_from_loader(
            state=state,
            loader=loader,
            device=device,
            target_context_prompt_state=target_context_prompt_state,
            anchor_state=anchor_state,
            candidate_state=k4_reference_state,
            rho=rho,
            normalize_increment=normalize_increment,
            surface_weight=surface_weight,
            rootzone_weight=rootzone_weight,
            use_lat_weighted_loss=use_lat_weighted_loss,
        )
        k4_reference_nested_k4_score = mixed_raw_increment_wrmse_objective_from_loader(
            state=state,
            loader=loader,
            device=device,
            target_context_prompt_state=target_context_prompt_state,
            anchor_state=anchor_state,
            candidate_state=k4_reference_state,
            rho=rho,
            normalize_increment=normalize_increment,
            surface_weight=surface_weight,
            rootzone_weight=rootzone_weight,
            use_lat_weighted_loss=use_lat_weighted_loss,
            max_cycles=4,
        )

    try:
        for candidate in pool:
            ridge_kwargs = _v10_candidate_to_ridge_kwargs(candidate)
            apply_target_adapter_state(state.model, anchor_state)
            diagnostics = run_ridge_coeff_adaptation(
                state=state,
                loader=loader,
                device=device,
                target_context_prompt_state=target_context_prompt_state,
                normalize_increment=normalize_increment,
                ridge_lambda=float(ridge_kwargs["ridge_lambda"]),
                ridge_clip_coeff_norm=float(ridge_kwargs["ridge_clip_coeff_norm"]),
                ridge_trust_region_radius=float(ridge_kwargs["ridge_trust_region_radius"]),
                ridge_max_feature_pixels=int(ridge_max_feature_pixels),
                ridge_standardize_features=bool(ridge_standardize_features),
                ridge_weighting=str(ridge_kwargs["ridge_weighting"]),
                surface_weight=surface_weight,
                rootzone_weight=rootzone_weight,
                use_lat_weighted_loss=use_lat_weighted_loss,
            )
            raw_state = extract_target_adapter_state(state.model)
            final_state = interpolate_target_adapter_state(
                anchor_state,
                raw_state,
                float(ridge_kwargs["anchor_alpha"]),
            )
            full_score = mixed_raw_increment_wrmse_objective_from_loader(
                state=state,
                loader=loader,
                device=device,
                target_context_prompt_state=target_context_prompt_state,
                anchor_state=anchor_state,
                candidate_state=final_state,
                rho=float(ridge_kwargs["adapt_mix_rho"]),
                normalize_increment=normalize_increment,
                surface_weight=surface_weight,
                rootzone_weight=rootzone_weight,
                use_lat_weighted_loss=use_lat_weighted_loss,
            )
            nested_k4_score = mixed_raw_increment_wrmse_objective_from_loader(
                state=state,
                loader=loader,
                device=device,
                target_context_prompt_state=target_context_prompt_state,
                anchor_state=anchor_state,
                candidate_state=final_state,
                rho=float(ridge_kwargs["adapt_mix_rho"]),
                normalize_increment=normalize_increment,
                surface_weight=surface_weight,
                rootzone_weight=rootzone_weight,
                use_lat_weighted_loss=use_lat_weighted_loss,
                max_cycles=4,
            )
            cv_folds: List[Dict[str, Any]] = []
            if int(K) == 4:
                for holdout_index in range(cycle_count):
                    apply_target_adapter_state(state.model, anchor_state)
                    fold_diagnostics = run_ridge_coeff_adaptation(
                        state=state,
                        loader=loader,
                        device=device,
                        target_context_prompt_state=target_context_prompt_state,
                        normalize_increment=normalize_increment,
                        ridge_lambda=float(ridge_kwargs["ridge_lambda"]),
                        ridge_clip_coeff_norm=float(ridge_kwargs["ridge_clip_coeff_norm"]),
                        ridge_trust_region_radius=float(ridge_kwargs["ridge_trust_region_radius"]),
                        ridge_max_feature_pixels=int(ridge_max_feature_pixels),
                        ridge_standardize_features=bool(ridge_standardize_features),
                        ridge_weighting=str(ridge_kwargs["ridge_weighting"]),
                        surface_weight=surface_weight,
                        rootzone_weight=rootzone_weight,
                        use_lat_weighted_loss=use_lat_weighted_loss,
                        skip_cycle_indices=[holdout_index],
                    )
                    fold_raw_state = extract_target_adapter_state(state.model)
                    fold_final_state = interpolate_target_adapter_state(
                        anchor_state,
                        fold_raw_state,
                        float(ridge_kwargs["anchor_alpha"]),
                    )
                    holdout_loader = _support_subset_loader(loader, [holdout_index])
                    fold_candidate = mixed_raw_increment_wrmse_objective_from_loader(
                        state=state,
                        loader=holdout_loader,
                        device=device,
                        target_context_prompt_state=target_context_prompt_state,
                        anchor_state=anchor_state,
                        candidate_state=fold_final_state,
                        rho=float(ridge_kwargs["adapt_mix_rho"]),
                        normalize_increment=normalize_increment,
                        surface_weight=surface_weight,
                        rootzone_weight=rootzone_weight,
                        use_lat_weighted_loss=use_lat_weighted_loss,
                    )
                    fold_reference = mixed_raw_increment_wrmse_objective_from_loader(
                        state=state,
                        loader=holdout_loader,
                        device=device,
                        target_context_prompt_state=target_context_prompt_state,
                        anchor_state=anchor_state,
                        candidate_state=anchor_state,
                        rho=0.0,
                        normalize_increment=normalize_increment,
                        surface_weight=surface_weight,
                        rootzone_weight=rootzone_weight,
                        use_lat_weighted_loss=use_lat_weighted_loss,
                    )
                    records = _v11_fold_records_from_scores(
                        holdout_indices=[holdout_index],
                        candidate_score=fold_candidate,
                        reference_score=fold_reference,
                        train_policy="leave_one_cycle_out",
                    )
                    for record in records:
                        record["fold_ridge_status"] = fold_diagnostics.get("status")
                    cv_folds.extend(records)
            else:
                nested_indices = list(range(0, min(4, cycle_count)))
                added_indices = list(range(min(4, cycle_count), cycle_count))
                if k4_reference_state is None:
                    raise ValueError("v11 K12 requires K4 reference state for nested cross-fit scoring")
                reference_rho = 1.0 if k4_reference_adapt_mix_rho is None else float(k4_reference_adapt_mix_rho)
                for train_indices, holdout_indices, policy in (
                    (added_indices, nested_indices, "train_added8_validate_nested_k4"),
                    (nested_indices, added_indices, "train_nested_k4_validate_added8"),
                ):
                    if not train_indices or not holdout_indices:
                        continue
                    apply_target_adapter_state(state.model, anchor_state)
                    train_loader = _support_subset_loader(loader, train_indices)
                    fold_diagnostics = run_ridge_coeff_adaptation(
                        state=state,
                        loader=train_loader,
                        device=device,
                        target_context_prompt_state=target_context_prompt_state,
                        normalize_increment=normalize_increment,
                        ridge_lambda=float(ridge_kwargs["ridge_lambda"]),
                        ridge_clip_coeff_norm=float(ridge_kwargs["ridge_clip_coeff_norm"]),
                        ridge_trust_region_radius=float(ridge_kwargs["ridge_trust_region_radius"]),
                        ridge_max_feature_pixels=int(ridge_max_feature_pixels),
                        ridge_standardize_features=bool(ridge_standardize_features),
                        ridge_weighting=str(ridge_kwargs["ridge_weighting"]),
                        surface_weight=surface_weight,
                        rootzone_weight=rootzone_weight,
                        use_lat_weighted_loss=use_lat_weighted_loss,
                    )
                    fold_raw_state = extract_target_adapter_state(state.model)
                    fold_final_state = interpolate_target_adapter_state(
                        anchor_state,
                        fold_raw_state,
                        float(ridge_kwargs["anchor_alpha"]),
                    )
                    holdout_loader = _support_subset_loader(loader, holdout_indices)
                    fold_candidate = mixed_raw_increment_wrmse_objective_from_loader(
                        state=state,
                        loader=holdout_loader,
                        device=device,
                        target_context_prompt_state=target_context_prompt_state,
                        anchor_state=anchor_state,
                        candidate_state=fold_final_state,
                        rho=float(ridge_kwargs["adapt_mix_rho"]),
                        normalize_increment=normalize_increment,
                        surface_weight=surface_weight,
                        rootzone_weight=rootzone_weight,
                        use_lat_weighted_loss=use_lat_weighted_loss,
                    )
                    fold_reference = mixed_raw_increment_wrmse_objective_from_loader(
                        state=state,
                        loader=holdout_loader,
                        device=device,
                        target_context_prompt_state=target_context_prompt_state,
                        anchor_state=anchor_state,
                        candidate_state=k4_reference_state,
                        rho=reference_rho,
                        normalize_increment=normalize_increment,
                        surface_weight=surface_weight,
                        rootzone_weight=rootzone_weight,
                        use_lat_weighted_loss=use_lat_weighted_loss,
                    )
                    records = _v11_fold_records_from_scores(
                        holdout_indices=holdout_indices,
                        candidate_score=fold_candidate,
                        reference_score=fold_reference,
                        train_policy=policy,
                    )
                    for record in records:
                        record["fold_ridge_status"] = fold_diagnostics.get("status")
                    cv_folds.extend(records)
            cv_summary = summarize_v11_cv_folds(cv_folds, nested_count=nested_count)
            reference_for_cycles = k0_anchor_score if int(K) == 4 else k4_reference_score
            cycle_fraction = support_cycle_improvement_fraction(reference_for_cycles, full_score)
            result = {
                **dict(candidate),
                **ridge_kwargs,
                "support_selection_objective": "loocv_mixed_raw_increment_wrmse_target_support_only",
                "target_eval_usage": "final_eval_only_no_selection",
                "score": full_score,
                "nested_k4_score": nested_k4_score,
                "cv_folds": cv_folds,
                "cv_summary": cv_summary,
                "support_cycle_improvement_fraction": float(cycle_fraction),
                "ridge_diagnostics": diagnostics,
                "raw_state": raw_state,
                "final_state": final_state,
            }
            candidate_results.append(result)
            if selected_result is None or _v11_selected_cv_metric(result) < _v11_selected_cv_metric(selected_result):
                selected_result = result
    finally:
        apply_target_adapter_state(state.model, original_state)

    if selected_result is None:
        raise RuntimeError("v11 support pool produced no candidate results")
    apply_target_adapter_state(state.model, selected_result["raw_state"])
    public_results: List[Dict[str, Any]] = []
    for result in candidate_results:
        public = {
            key: value
            for key, value in result.items()
            if key not in {"raw_state", "final_state"}
        }
        public_results.append(public)
    selected_public = {
        key: value
        for key, value in selected_result.items()
        if key not in {"raw_state", "final_state"}
    }
    return {
        "schema_version": "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_v1",
        "stage3_kshot_mode": "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested",
        "label_source": "target_support_only",
        "target_eval_usage": "final_eval_only_no_selection",
        "support_selection_objective": "loocv_mixed_raw_increment_wrmse_target_support_only",
        "support_candidate_pool": [dict(candidate) for candidate in pool],
        "support_candidate_results": public_results,
        "selected_support_candidate_id": str(selected_result.get("candidate_id", "")),
        "selected_candidate": selected_public,
        "selected_ridge_diagnostics": dict(selected_result.get("ridge_diagnostics", {}) or {}),
        "selected_score": dict(selected_result.get("score", {}) or {}),
        "selected_nested_k4_score": dict(selected_result.get("nested_k4_score", {}) or {}),
        "selected_cv_summary": dict(selected_result.get("cv_summary", {}) or {}),
        "selected_cv_folds": list(selected_result.get("cv_folds", []) or []),
        "selected_anchor_alpha": float(selected_result["anchor_alpha"]),
        "selected_adapt_mix_rho": float(selected_result["adapt_mix_rho"]),
        "selected_ridge_lambda": float(selected_result["ridge_lambda"]),
        "selected_ridge_clip_coeff_norm": float(selected_result["ridge_clip_coeff_norm"]),
        "selected_ridge_trust_region_radius": float(selected_result["ridge_trust_region_radius"]),
        "selected_ridge_weighting": str(selected_result["ridge_weighting"]),
        "support_cycle_improvement_fraction": float(
            selected_result.get("cv_summary", {}).get("cv_cycle_improvement_fraction", 0.0) or 0.0
        ),
        "full_support_cycle_improvement_fraction": float(
            selected_result.get("support_cycle_improvement_fraction", 0.0) or 0.0
        ),
        "k0_anchor_score": dict(k0_anchor_score),
        "k4_reference_score": dict(k4_reference_score),
        "k4_reference_nested_k4_score": dict(k4_reference_nested_k4_score),
        "k4_reference_adapt_mix_rho": (
            None if k4_reference_adapt_mix_rho is None else float(k4_reference_adapt_mix_rho)
        ),
    }


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
    source_calibrated_kshot_no_update = (
        int(K) > 0
        and str(stage3_posterior_decision) == "no_update"
        and str(support_gate_status) == "source_calibrated_no_update"
    )
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
            else (
                "Kshot_source_side_policy_selected_no_update"
                if source_calibrated_kshot_no_update
                else "Kshot_source_policy_constrained_posterior_update"
            )
        ),
        "paper_selection_basis": resolved_paper_selection_basis,
        "stage3_acceptance_basis": resolved_acceptance_basis,
        "source_policy_candidate_id": str(source_policy_candidate_id or ""),
        "support_gate_policy_role": (
            "not_applicable_k0_no_support"
            if int(K) == 0
            else (
                "source_side_policy_selected_no_update"
                if source_calibrated_kshot_no_update
                else "target_support_only_diagnostic_not_paper_selection"
            )
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


def build_stage3_adapter_stage_state(
    *,
    stage_name: str,
    anchor_state: Dict[str, torch.Tensor],
    stage_state: Dict[str, torch.Tensor],
    K: int,
    adapt_scope: str,
    anchor_alpha: float,
    adaptation_steps: int,
    stage3_posterior_decision: str,
    support_gate_status: str,
    adapt_mix_rho: float,
    support_loss_delta: Any = None,
    support_gate_reject_reason: Optional[Iterable[Any]] = None,
    source_policy_candidate_id: str = "",
) -> Dict[str, Any]:
    """Return a replayable target-adapter state for one Stage 3 audit point."""
    if stage_name not in {"raw_adapted", "post_gate"}:
        raise ValueError(f"unsupported Stage 3 adapter stage: {stage_name!r}")
    if not anchor_state or not stage_state:
        raise ValueError(f"{stage_name} Stage 3 state requires non-empty anchor and stage states")
    if set(anchor_state) != set(stage_state):
        missing = sorted(set(anchor_state) - set(stage_state))
        extra = sorted(set(stage_state) - set(anchor_state))
        raise ValueError(f"{stage_name} adapter state keys differ; missing={missing[:5]} extra={extra[:5]}")
    bad_keys = [name for name in stage_state if not _is_target_adapter_state_key(name)]
    if bad_keys:
        raise ValueError(f"{stage_name} state must contain target-only keys; bad_keys={bad_keys[:5]}")
    drift = target_parameter_l2_drift(anchor_state, stage_state)
    state_hash = hash_tensor_state_dict(stage_state)
    anchor_hash = hash_tensor_state_dict(anchor_state)
    return {
        "schema_version": f"hyperda_stage3_{stage_name}_state_v1",
        "stage_name": stage_name,
        "target_adapter_state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in sorted(stage_state.items())
        },
        "target_adapter_state_hash": state_hash,
        "target_adapter_anchor_hash": anchor_hash,
        "drift_from_prior": drift,
        "K": int(K),
        "adapt_scope": str(adapt_scope),
        "anchor_alpha": float(anchor_alpha),
        "adaptation_steps": int(adaptation_steps),
        "stage3_posterior_decision": str(stage3_posterior_decision),
        "support_gate_status": str(support_gate_status),
        "support_gate_reject_reason": [str(reason) for reason in (support_gate_reject_reason or [])],
        "support_loss_delta": None if support_loss_delta is None else float(support_loss_delta),
        "adapt_mix_rho": float(adapt_mix_rho),
        "source_policy_candidate_id": str(source_policy_candidate_id or ""),
    }


def stage3_adapter_stage_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return JSON-safe metadata for a tensor-bearing Stage 3 adapter state."""
    return {
        "schema_version": str(state.get("schema_version", "")),
        "stage_name": str(state.get("stage_name", "")),
        "target_adapter_state_hash": str(state.get("target_adapter_state_hash", "")),
        "target_adapter_anchor_hash": str(state.get("target_adapter_anchor_hash", "")),
        "drift_from_prior": dict(state.get("drift_from_prior", {}) or {}),
        "K": state.get("K"),
        "adapt_scope": state.get("adapt_scope", ""),
        "anchor_alpha": state.get("anchor_alpha"),
        "adaptation_steps": state.get("adaptation_steps"),
        "stage3_posterior_decision": state.get("stage3_posterior_decision", ""),
        "support_gate_status": state.get("support_gate_status", ""),
        "support_gate_reject_reason": list(state.get("support_gate_reject_reason", []) or []),
        "support_loss_delta": state.get("support_loss_delta"),
        "adapt_mix_rho": state.get("adapt_mix_rho"),
        "source_policy_candidate_id": state.get("source_policy_candidate_id", ""),
    }


def build_stage3_final_eval_mix_state(
    *,
    adapt_mix_rho: float,
    raw_state: Dict[str, Any],
    post_gate_state: Dict[str, Any],
    stage3_posterior_decision: str,
    support_gate_status: str,
) -> Dict[str, Any]:
    """Return metadata for the final output mixture applied during target_eval."""
    return {
        "schema_version": "hyperda_stage3_final_eval_mix_state_v1",
        "state_stage": "final_eval_output_mix",
        "adapt_mix_rho": float(adapt_mix_rho),
        "raw_adapted_state_hash": str(raw_state.get("target_adapter_state_hash", "")),
        "post_gate_state_hash": str(post_gate_state.get("target_adapter_state_hash", "")),
        "stage3_posterior_decision": str(stage3_posterior_decision),
        "support_gate_status": str(support_gate_status),
        "prediction_hash_source": "scripts/eval/evaluate_checkpoint.py",
        "prediction_hashes_recorded_in_eval_summary": True,
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
    raw_adapter_state = {
        name: tensor.detach().cpu()
        for name, tensor in full_config.get("raw_adapted_adapter_state", {}).items()
    }
    if not raw_adapter_state:
        raw_adapter_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in final_state.items()
        }
    support_loss_delta_for_stage = (
        full_config.get("support_loss_delta")
        if full_config.get("support_loss_delta") is not None
        else full_config.get("standard_support_loss_delta_full_support")
    )
    raw_adapted_state = build_stage3_adapter_stage_state(
        stage_name="raw_adapted",
        anchor_state=anchor_state,
        stage_state=raw_adapter_state,
        K=int(full_config.get("K", 0)),
        adapt_scope=str(full_config.get("adapt_scope", "safe_operator")),
        anchor_alpha=float(full_config.get("anchor_alpha", default_anchor_alpha_for_K(int(full_config.get("K", 0))))),
        adaptation_steps=int(full_config.get("adaptation_steps", 0) or 0),
        stage3_posterior_decision="raw_adapted_before_gate",
        support_gate_status=str(full_config.get("support_gate_status", "disabled")),
        adapt_mix_rho=float(full_config.get("adapt_mix_rho", 1.0 if int(full_config.get("K", 0) or 0) == 0 else 0.0)),
        support_loss_delta=support_loss_delta_for_stage,
        support_gate_reject_reason=full_config.get("support_gate_reject_reason", []),
        source_policy_candidate_id=str(full_config.get("source_policy_candidate_id", "")),
    )
    post_gate_state = build_stage3_adapter_stage_state(
        stage_name="post_gate",
        anchor_state=anchor_state,
        stage_state=final_state,
        K=int(full_config.get("K", 0)),
        adapt_scope=str(full_config.get("adapt_scope", "safe_operator")),
        anchor_alpha=float(full_config.get("anchor_alpha", default_anchor_alpha_for_K(int(full_config.get("K", 0))))),
        adaptation_steps=int(full_config.get("adaptation_steps", 0) or 0),
        stage3_posterior_decision=str(full_config.get("stage3_posterior_decision", "accepted")),
        support_gate_status=str(full_config.get("support_gate_status", "disabled")),
        adapt_mix_rho=float(full_config.get("adapt_mix_rho", 1.0 if int(full_config.get("K", 0) or 0) == 0 else 0.0)),
        support_loss_delta=support_loss_delta_for_stage,
        support_gate_reject_reason=full_config.get("support_gate_reject_reason", []),
        source_policy_candidate_id=str(full_config.get("source_policy_candidate_id", "")),
    )
    final_eval_mix_state = build_stage3_final_eval_mix_state(
        adapt_mix_rho=float(full_config.get("adapt_mix_rho", 1.0 if int(full_config.get("K", 0) or 0) == 0 else 0.0)),
        raw_state=raw_adapted_state,
        post_gate_state=post_gate_state,
        stage3_posterior_decision=str(full_config.get("stage3_posterior_decision", "accepted")),
        support_gate_status=str(full_config.get("support_gate_status", "disabled")),
    )
    full_config["raw_adapted_state"] = stage3_adapter_stage_summary(raw_adapted_state)
    full_config["post_gate_state"] = stage3_adapter_stage_summary(post_gate_state)
    full_config["final_eval_mix_state"] = dict(final_eval_mix_state)
    full_config["raw_adapted_state_hash"] = raw_adapted_state["target_adapter_state_hash"]
    full_config["post_gate_state_hash"] = post_gate_state["target_adapter_state_hash"]
    full_config["raw_adapted_drift_from_prior"] = dict(raw_adapted_state["drift_from_prior"])
    full_config["post_gate_drift_from_prior"] = dict(post_gate_state["drift_from_prior"])
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
    residual_gain_alpha_surface = float(full_config.get("residual_gain_alpha_surface", 1.0) or 1.0)
    residual_gain_alpha_rootzone = float(full_config.get("residual_gain_alpha_rootzone", 1.0) or 1.0)
    support_affine_calibration = dict(full_config.get("support_affine_calibration", {}) or {})
    full_config.update(
        {
            "method": method_id,
            "model_type": "hyperda_basis_adapter_target_adapt",
            "width": state.source_config.get("width", 32),
            "prompt_dim": state.source_config.get("prompt_dim", 64),
            "hyper_n_basis": state.source_config.get("hyper_n_basis", 8),
            "hyper_adapter_bottleneck": state.source_config.get("hyper_adapter_bottleneck"),
            "hyper_adapter_scale": state.source_config.get("hyper_adapter_scale", 1.0),
            "hyper_coeff_generator": state.source_config.get("hyper_coeff_generator", "per_adapter"),
            "hyper_rank_gate_top_k": state.source_config.get("hyper_rank_gate_top_k", 4),
            "hyper_rank_gate_temperature_init": state.source_config.get("hyper_rank_gate_temperature_init", 1.0),
            "hyper_adapter_param_style": state.source_config.get("hyper_adapter_param_style", "basis_1x1"),
            "hyper_reliability_gate": state.source_config.get("hyper_reliability_gate", "none"),
            "hyper_reliability_init": state.source_config.get("hyper_reliability_init", 0.95),
            "hyper_source_saliency_prior": state.source_config.get("hyper_source_saliency_prior"),
            "hyper_source_saliency_prior_beta": state.source_config.get("hyper_source_saliency_prior_beta", 0.0),
            "hyper_source_saliency_prior_path": state.source_config.get("hyper_source_saliency_prior_path", ""),
            "hyper_source_saliency_prior_application": state.source_config.get(
                "hyper_source_saliency_prior_application",
                "soft_regularization_metadata",
            ),
            "hyper_prompt_manifold_reliability": bool(
                state.source_config.get("hyper_prompt_manifold_reliability", False)
            ),
            "hyper_prompt_manifold_reliability_strength": state.source_config.get(
                "hyper_prompt_manifold_reliability_strength",
                0.0,
            ),
            "hyper_source_manifold_guard": bool(state.source_config.get("hyper_source_manifold_guard", False)),
            "hyper_source_manifold_guard_strength": state.source_config.get(
                "hyper_source_manifold_guard_strength",
                0.25,
            ),
            "hyper_source_manifold_guard_distance_key": state.source_config.get(
                "hyper_source_manifold_guard_distance_key",
                SOURCE_MANIFOLD_DISTANCE_KEY,
            ),
            "hyper_source_manifold_guard_min_multiplier": state.source_config.get(
                "hyper_source_manifold_guard_min_multiplier",
                0.0,
            ),
            "source_manifold_guard_calibration": state.source_config.get(
                "source_manifold_guard_calibration",
                "disabled",
            ),
            "hyper_source_trust_routing": bool(state.source_config.get("hyper_source_trust_routing", False)),
            "hyper_source_trust_strength": state.source_config.get("hyper_source_trust_strength", 0.0),
            "hyper_source_trust_top_m": state.source_config.get("hyper_source_trust_top_m", 4),
            "hyper_source_trust_variable_gate": bool(
                state.source_config.get("hyper_source_trust_variable_gate", False)
            ),
            "hyper_phys_agreement_guard": bool(state.source_config.get("hyper_phys_agreement_guard", False)),
            "hyper_phys_agreement_guard_strength": state.source_config.get(
                "hyper_phys_agreement_guard_strength",
                1.0,
            ),
            "hyper_phys_agreement_guard_min_multiplier": state.source_config.get(
                "hyper_phys_agreement_guard_min_multiplier",
                0.0,
            ),
            "hyper_phys_agreement_guard_risk_rule": state.source_config.get(
                "hyper_phys_agreement_guard_risk_rule",
                "or",
            ),
            "hyper_phys_context_modulation": bool(state.source_config.get("hyper_phys_context_modulation", False)),
            "hyper_phys_delta_scale": state.source_config.get("hyper_phys_delta_scale", 0.25),
            "hyper_phys_gate_init": state.source_config.get("hyper_phys_gate_init", 0.90),
            "hyper_operator_droppath_p": state.source_config.get("hyper_operator_droppath_p", 0.10),
            "phys_context_source": state.source_config.get(
                "phys_context_source",
                "raw_input_side_da_diagnostics",
            ),
            "hyper_enable_film": bool(state.source_config.get("hyper_enable_film", True)),
            "hyper_enable_adapters": bool(state.source_config.get("hyper_enable_adapters", True)),
            "zero_shot_prior_form": state.source_config.get("zero_shot_prior_form", "direct_hyper"),
            "source_residual_rho": state.source_config.get(
                "source_residual_rho",
                state.source_config.get("zero_shot_rho", 1.0),
            ),
            "source_residual_gate": state.source_config.get("source_residual_gate", "prompt_reliability_scalar"),
            "source_residual_gate_init": state.source_config.get("source_residual_gate_init", 0.95),
            "source_residual_reliability_dim": state.source_config.get("source_residual_reliability_dim", 5),
            "zero_raw_increment_init": bool(state.source_config.get("zero_raw_increment_init", False)),
            "context_encoder": state.source_config.get("context_encoder", "current_mean_std"),
            "num_regions": state.source_config.get("num_regions", 6),
            "source_regions": list(state.source_config.get("source_regions", [])),
            "source_region_global_indices": list(state.source_config.get("source_region_global_indices", [])),
            "ch_mean": state.source_config.get("ch_mean"),
            "ch_std": state.source_config.get("ch_std"),
            "inc_mean": state.source_config.get("inc_mean"),
            "inc_std": state.source_config.get("inc_std"),
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
            "eligible_support_candidate_count": int(
                full_config.get("eligible_support_candidate_count", 0) or 0
            ),
            "k0_anchor_gate_deferred_to_k4_reference": bool(
                full_config.get("k0_anchor_gate_deferred_to_k4_reference", False)
            ),
            "k0_anchor_gate": dict(full_config.get("k0_anchor_gate", {}) or {}),
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
            "stage3_kshot_mode": full_config.get("stage3_kshot_mode", "paper_safe"),
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
            "selected_support_candidate_id": full_config.get("selected_support_candidate_id", ""),
            "support_candidate_pool": list(full_config.get("support_candidate_pool", []) or []),
            "support_pool_selection": dict(full_config.get("support_pool_selection", {}) or {}),
            "support_selection_objective": full_config.get("support_selection_objective", ""),
            "support_cv_objective_delta": full_config.get("support_cv_objective_delta"),
            "support_cv_objective_delta_se": full_config.get("support_cv_objective_delta_se"),
            "support_cv_objective_delta_t": full_config.get("support_cv_objective_delta_t"),
            "support_cv_nested_k4_objective_delta": full_config.get("support_cv_nested_k4_objective_delta"),
            "support_cv_added_objective_delta": full_config.get("support_cv_added_objective_delta"),
            "k12_vs_k4_cv_objective_delta": full_config.get("k12_vs_k4_cv_objective_delta"),
            "k12_vs_k4_cv_rootzone_delta": full_config.get("k12_vs_k4_cv_rootzone_delta"),
            "support_calibration_dof": full_config.get("support_calibration_dof"),
            "best_rejected_support_candidate": dict(
                full_config.get("best_rejected_support_candidate", {}) or {}
            ),
            "support_cycle_improvement_fraction": full_config.get("support_cycle_improvement_fraction"),
            "support_gate_cycle_improvement_min_fraction": full_config.get(
                "support_gate_cycle_improvement_min_fraction"
            ),
            "k12_reference_policy": full_config.get("k12_reference_policy", ""),
            "ridge_lambda": full_config.get("ridge_lambda", None),
            "ridge_clip_coeff_norm": full_config.get("ridge_clip_coeff_norm", None),
            "ridge_trust_region_radius": full_config.get("ridge_trust_region_radius", None),
            "ridge_max_feature_pixels": full_config.get("ridge_max_feature_pixels", None),
            "ridge_standardize_features": bool(full_config.get("ridge_standardize_features", False)),
            "ridge_weighting": full_config.get("ridge_weighting", "global_pixel_l2"),
            "ridge_diagnostics": dict(full_config.get("ridge_diagnostics", {}) or {}),
            "linearized_ridge_calibration": dict(full_config.get("linearized_ridge_calibration", {}) or {}),
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
            "ridge_effective_calibration_dof": (full_config.get("ridge_diagnostics", {}) or {}).get(
                "effective_calibration_dof"
            ),
            "adaptation_step_policy_source": full_config.get("adaptation_step_policy_source", ""),
            "resolved_mode_defaults": dict(full_config.get("resolved_mode_defaults", {}) or {}),
            "residual_gain_alpha_surface": residual_gain_alpha_surface,
            "residual_gain_alpha_rootzone": residual_gain_alpha_rootzone,
            "support_gain_calibration": dict(full_config.get("support_gain_calibration", {}) or {}),
            "support_affine_calibration": support_affine_calibration,
            "k4_reference_checkpoint": full_config.get("k4_reference_checkpoint", ""),
            "k4_reference_checkpoint_sha256": full_config.get("k4_reference_checkpoint_sha256", ""),
            "k4_reference_gate": dict(full_config.get("k4_reference_gate", {}) or {}),
            "deferred_k0_anchor_gate": dict(full_config.get("deferred_k0_anchor_gate", {}) or {}),
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
            "target_support_labels_used_for_parameter_update": bool(
                full_config.get("target_support_labels_used_for_parameter_update", False)
            ),
            "target_support_labels_used_for_optimizer_update": bool(
                full_config.get(
                    "target_support_labels_used_for_optimizer_update",
                    full_config.get("target_support_labels_used_for_parameter_update", False),
                )
            ),
            "target_support_labels_used_for_ridge_solve": bool(
                full_config.get("target_support_labels_used_for_ridge_solve", False)
            ),
            "target_support_labels_used_for_calibration": bool(
                full_config.get("target_support_labels_used_for_calibration", False)
            ),
            "target_support_labels_used_for_support_gate": bool(
                full_config.get("target_support_labels_used_for_support_gate", False)
            ),
            "few_shot_update_type": full_config.get("few_shot_update_type", ""),
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
            "support_nesting_policy": full_config.get("support_nesting_policy", ""),
            "nested_support_dates_hash": full_config.get("nested_support_dates_hash", ""),
            "nested_support_manifest": full_config.get("nested_support_manifest", ""),
            "support_affine_calibration": dict(full_config.get("support_affine_calibration", {}) or {}),
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
            "raw_adapted_state": dict(full_config.get("raw_adapted_state", {})),
            "post_gate_state": dict(full_config.get("post_gate_state", {})),
            "final_eval_mix_state": dict(full_config.get("final_eval_mix_state", {})),
            "raw_adapted_state_hash": full_config.get("raw_adapted_state_hash", ""),
            "post_gate_state_hash": full_config.get("post_gate_state_hash", ""),
            "raw_adapted_drift_from_prior": dict(full_config.get("raw_adapted_drift_from_prior", {})),
            "post_gate_drift_from_prior": dict(full_config.get("post_gate_drift_from_prior", {})),
            "target_context_prompt_state": prompt_state,
            "target_context_prompt_state_summary": prompt_metadata,
            "context_tta_mode": full_config.get("context_tta_mode", prompt_metadata.get("context_tta_mode", "")),
            "context_tta_residual_scale": float(
                full_config.get(
                    "context_tta_residual_scale",
                    prompt_metadata.get("context_tta_residual_scale", 0.0),
                )
                or 0.0
            ),
            "context_tta_residual_clip_l2": float(
                full_config.get(
                    "context_tta_residual_clip_l2",
                    prompt_metadata.get("context_tta_residual_clip_l2", 0.0),
                )
                or 0.0
            ),
            "context_tta_state_hash": full_config.get(
                "context_tta_state_hash",
                prompt_metadata.get("context_tta_state_hash", ""),
            ),
            "context_tta_label_usage": full_config.get(
                "context_tta_label_usage",
                prompt_metadata.get("context_tta_label_usage", "none"),
            ),
            "context_tta_effective": bool(
                full_config.get("context_tta_effective", prompt_metadata.get("context_tta_effective", False))
            ),
            "context_tta_source_stat_status": full_config.get(
                "context_tta_source_stat_status",
                prompt_metadata.get("context_tta_source_stat_status", "not_requested"),
            ),
            "prompt_l2_delta_mean": float(
                full_config.get("prompt_l2_delta_mean", prompt_metadata.get("prompt_l2_delta_mean", 0.0)) or 0.0
            ),
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
        "raw_adapted_state_dict": raw_adapted_state,
        "post_gate_state_dict": post_gate_state,
        "final_eval_mix_state": final_eval_mix_state,
        "stage3_posterior_state_dict": stage3_posterior_state_dict,
        "residual_gain_alpha_surface": residual_gain_alpha_surface,
        "residual_gain_alpha_rootzone": residual_gain_alpha_rootzone,
        "support_gain_calibration": dict(full_config.get("support_gain_calibration", {}) or {}),
        "support_affine_calibration": support_affine_calibration,
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
        "support_nesting_policy": config.get("support_nesting_policy", ""),
        "nested_support_dates_hash": config.get("nested_support_dates_hash", ""),
        "nested_support_manifest": config.get("nested_support_manifest", ""),
        "target_support_count": config.get("target_support_count", 0),
        "target_eval_dates_hash": config.get("target_eval_dates_hash", ""),
        "context_tta_mode": config.get("context_tta_mode", ""),
        "context_tta_residual_scale": float(config.get("context_tta_residual_scale", 0.0) or 0.0),
        "context_tta_residual_clip_l2": float(config.get("context_tta_residual_clip_l2", 0.0) or 0.0),
        "context_tta_state_hash": config.get("context_tta_state_hash", ""),
        "context_tta_label_usage": config.get("context_tta_label_usage", "none"),
        "context_tta_effective": bool(config.get("context_tta_effective", False)),
        "context_tta_source_stat_status": config.get("context_tta_source_stat_status", "not_requested"),
        "prompt_l2_delta_mean": float(config.get("prompt_l2_delta_mean", 0.0) or 0.0),
        "target_context_prompt_state": prompt_state_summary,
        "stage3_prior_snapshot": dict(config.get("stage3_prior_snapshot", {}) or {}),
        "stage3_posterior_state": dict(config.get("stage3_posterior_state", {}) or {}),
        "raw_adapted_state": dict(config.get("raw_adapted_state", {}) or {}),
        "post_gate_state": dict(config.get("post_gate_state", {}) or {}),
        "final_eval_mix_state": dict(config.get("final_eval_mix_state", {}) or {}),
        "raw_adapted_state_hash": config.get("raw_adapted_state_hash", ""),
        "post_gate_state_hash": config.get("post_gate_state_hash", ""),
        "raw_adapted_drift_from_prior": dict(config.get("raw_adapted_drift_from_prior", {}) or {}),
        "post_gate_drift_from_prior": dict(config.get("post_gate_drift_from_prior", {}) or {}),
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
        "eligible_support_candidate_count": int(config.get("eligible_support_candidate_count", 0) or 0),
        "k0_anchor_gate_deferred_to_k4_reference": bool(
            config.get("k0_anchor_gate_deferred_to_k4_reference", False)
        ),
        "k0_anchor_gate": dict(config.get("k0_anchor_gate", {}) or {}),
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
        "stage3_kshot_mode": config.get("stage3_kshot_mode", "paper_safe"),
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
        "selected_support_candidate_id": config.get("selected_support_candidate_id", ""),
        "support_candidate_pool": list(config.get("support_candidate_pool", []) or []),
        "support_pool_selection": dict(config.get("support_pool_selection", {}) or {}),
        "support_selection_objective": config.get("support_selection_objective", ""),
        "support_cv_objective_delta": config.get("support_cv_objective_delta", None),
        "support_cv_objective_delta_se": config.get("support_cv_objective_delta_se", None),
        "support_cv_objective_delta_t": config.get("support_cv_objective_delta_t", None),
        "support_cv_nested_k4_objective_delta": config.get("support_cv_nested_k4_objective_delta", None),
        "support_cv_added_objective_delta": config.get("support_cv_added_objective_delta", None),
        "k12_vs_k4_cv_objective_delta": config.get("k12_vs_k4_cv_objective_delta", None),
        "k12_vs_k4_cv_rootzone_delta": config.get("k12_vs_k4_cv_rootzone_delta", None),
        "support_calibration_dof": config.get("support_calibration_dof", None),
        "best_rejected_support_candidate": dict(config.get("best_rejected_support_candidate", {}) or {}),
        "support_cycle_improvement_fraction": config.get("support_cycle_improvement_fraction", None),
        "support_gate_cycle_improvement_min_fraction": config.get(
            "support_gate_cycle_improvement_min_fraction",
            None,
        ),
        "k12_reference_policy": config.get("k12_reference_policy", ""),
        "ridge_lambda": config.get("ridge_lambda", None),
        "ridge_clip_coeff_norm": config.get("ridge_clip_coeff_norm", None),
        "ridge_trust_region_radius": config.get("ridge_trust_region_radius", None),
        "ridge_max_feature_pixels": config.get("ridge_max_feature_pixels", None),
        "ridge_standardize_features": bool(config.get("ridge_standardize_features", False)),
        "ridge_weighting": config.get("ridge_weighting", "global_pixel_l2"),
        "ridge_diagnostics": dict(config.get("ridge_diagnostics", {}) or {}),
        "linearized_ridge_calibration": dict(config.get("linearized_ridge_calibration", {}) or {}),
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
        "ridge_effective_calibration_dof": (config.get("ridge_diagnostics", {}) or {}).get(
            "effective_calibration_dof"
        ),
        "residual_gain_alpha_surface": config.get("residual_gain_alpha_surface", 1.0),
        "residual_gain_alpha_rootzone": config.get("residual_gain_alpha_rootzone", 1.0),
        "support_gain_calibration": dict(config.get("support_gain_calibration", {}) or {}),
        "support_affine_calibration": dict(config.get("support_affine_calibration", {}) or {}),
        "k4_reference_checkpoint": config.get("k4_reference_checkpoint", ""),
        "k4_reference_checkpoint_sha256": config.get("k4_reference_checkpoint_sha256", ""),
        "k4_reference_gate": dict(config.get("k4_reference_gate", {}) or {}),
        "deferred_k0_anchor_gate": dict(config.get("deferred_k0_anchor_gate", {}) or {}),
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
        "adaptation_step_policy_source": config.get("adaptation_step_policy_source", ""),
        "resolved_mode_defaults": dict(config.get("resolved_mode_defaults", {}) or {}),
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
        "target_support_labels_used_for_parameter_update": bool(
            config.get("target_support_labels_used_for_parameter_update", False)
        ),
        "target_support_labels_used_for_optimizer_update": bool(
            config.get(
                "target_support_labels_used_for_optimizer_update",
                config.get("target_support_labels_used_for_parameter_update", False),
            )
        ),
        "target_support_labels_used_for_ridge_solve": bool(
            config.get("target_support_labels_used_for_ridge_solve", False)
        ),
        "target_support_labels_used_for_calibration": bool(
            config.get("target_support_labels_used_for_calibration", False)
        ),
        "target_support_labels_used_for_support_gate": bool(
            config.get("target_support_labels_used_for_support_gate", False)
        ),
        "few_shot_update_type": config.get("few_shot_update_type", ""),
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


def _support_nesting_metadata(dataset: Optional[HydroDADataset]) -> Dict[str, Any]:
    """Return run-local nested support provenance from the active split entry."""
    if dataset is None:
        return {
            "support_nesting_policy": "not_applicable",
            "nested_support_dates_hash": "",
            "nested_support_manifest": "",
        }
    entry = dataset._split_entry
    policy = str(entry.get("support_nesting_policy", "") or "")
    nested_hash = str(entry.get("nested_support_dates_hash", "") or "")
    if not nested_hash and policy:
        nested_hash = str(entry.get("target_support_dates_hash", entry.get("support_dates_hash", "")) or "")
    return {
        "support_nesting_policy": policy or _support_nesting_status(dataset.K or 0, []),
        "nested_support_dates_hash": nested_hash,
        "nested_support_manifest": str(entry.get("nested_support_manifest", "") or ""),
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
        context_tta_mode=args.context_tta_mode,
        context_tta_residual_scale=float(args.context_tta_residual_scale),
        context_tta_residual_clip_l2=float(args.context_tta_residual_clip_l2),
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
    support_gain_calibration: Dict[str, Any] = {}
    support_affine_calibration: Dict[str, Any] = {}
    linearized_ridge_calibration: Dict[str, Any] = {}
    support_pool_selection: Dict[str, Any] = {}
    k4_reference_gate: Dict[str, Any] = {}
    k4_reference_checkpoint_sha256 = ""
    k4_reference_state: Optional[Dict[str, torch.Tensor]] = None
    k4_reference_adapt_mix_rho: Optional[float] = None
    deferred_k0_anchor_gate: Dict[str, Any] = {}
    residual_gain_alpha_surface = 1.0
    residual_gain_alpha_rootzone = 1.0
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
        if (
            args.K > 0
            and args.adapt_solver == "ridge_coeff"
            and args.stage3_kshot_mode in {
                "diagnostic_linearized_coeff_ridge_v10_support_pool_nested",
                "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested",
            }
        ):
            assert support_dataset is not None
            assert loss_loader is not None
            if args.K == 12:
                if not args.k4_reference_checkpoint:
                    raise ValueError(f"{args.stage3_kshot_mode} K12 requires --k4_reference_checkpoint for nested K4-safe selection")
                k4_reference_checkpoint_sha256 = compute_sha256(args.k4_reference_checkpoint)
                k4_reference_state = load_k4_reference_target_adapter_state(args.k4_reference_checkpoint, device)
                k4_reference_adapt_mix_rho = load_k4_reference_adapt_mix_rho(args.k4_reference_checkpoint)
            started = time.time()
            if args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested":
                support_pool_selection = run_v11_loocv_support_pool_adaptation(
                    state=state,
                    loader=loss_loader,
                    device=device,
                    target_context_prompt_state=target_context_prompt_state,
                    anchor_state=anchor_adapter_state,
                    K=args.K,
                    normalize_increment=state.normalization.get("inc_mean") is not None,
                    ridge_max_feature_pixels=args.ridge_max_feature_pixels,
                    ridge_standardize_features=args.ridge_standardize_features,
                    surface_weight=args.surface_weight,
                    rootzone_weight=args.rootzone_weight,
                    use_lat_weighted_loss=args.use_lat_weighted_loss,
                    k4_reference_state=k4_reference_state,
                    k4_reference_adapt_mix_rho=k4_reference_adapt_mix_rho,
                )
            else:
                support_pool_selection = run_v10_support_pool_adaptation(
                    state=state,
                    loader=loss_loader,
                    device=device,
                    target_context_prompt_state=target_context_prompt_state,
                    anchor_state=anchor_adapter_state,
                    K=args.K,
                    normalize_increment=state.normalization.get("inc_mean") is not None,
                    ridge_max_feature_pixels=args.ridge_max_feature_pixels,
                    ridge_standardize_features=args.ridge_standardize_features,
                    surface_weight=args.surface_weight,
                    rootzone_weight=args.rootzone_weight,
                    use_lat_weighted_loss=args.use_lat_weighted_loss,
                    k4_reference_state=k4_reference_state,
                    k4_reference_adapt_mix_rho=k4_reference_adapt_mix_rho,
                )
            args.anchor_alpha = float(support_pool_selection["selected_anchor_alpha"])
            args.adapt_mix_rho = float(support_pool_selection["selected_adapt_mix_rho"])
            args.ridge_lambda = float(support_pool_selection["selected_ridge_lambda"])
            args.ridge_clip_coeff_norm = float(support_pool_selection["selected_ridge_clip_coeff_norm"])
            args.ridge_trust_region_radius = float(support_pool_selection["selected_ridge_trust_region_radius"])
            args.ridge_weighting = str(support_pool_selection["selected_ridge_weighting"])
            args.trust_coeff_radius = float(args.ridge_trust_region_radius)
            args.rho_policy = (
                f"{args.stage3_kshot_mode}_selected_"
                f"{support_pool_selection['selected_support_candidate_id']}"
            )
            args.source_policy_candidate_id = str(support_pool_selection["selected_support_candidate_id"])
            ridge_diagnostics = dict(support_pool_selection["selected_ridge_diagnostics"])
            standard_support_loss_before = dict(support_pool_selection["k0_anchor_score"])
            standard_support_loss_after = dict(support_pool_selection["selected_score"])
            print(
                f"{args.stage3_kshot_mode} ridge adaptation finished in "
                f"{time.time() - started:.1f}s: "
                f"selected={support_pool_selection.get('selected_support_candidate_id')} "
                f"objective={support_pool_selection.get('selected_score', {}).get('standard_support_objective_full_support')} "
                f"cv_objective={support_pool_selection.get('selected_cv_summary', {}).get('cv_candidate_objective')} "
                f"rho={args.adapt_mix_rho}",
                flush=True,
            )
            support_selection_objective = str(
                support_pool_selection.get("support_selection_objective", "exact_mixed_target_support_only")
            )
            linearized_ridge_calibration = {
                "calibration_mode": (
                    "target_support_linearized_coeff_ridge_v11_loocv_support_pool_nested"
                    if args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested"
                    else "target_support_linearized_coeff_ridge_v10_support_pool_nested"
                ),
                "status": ridge_diagnostics.get("status"),
                "label_source": "target_support_only",
                "target_eval_usage": "final_eval_only_no_selection",
                "parameter_scope": "adapter_coefficient_residuals_only",
                "frozen_groups": list(FROZEN_SOURCE_GROUPS),
                "support_selection_objective": support_selection_objective,
                "selected_support_candidate_id": str(support_pool_selection["selected_support_candidate_id"]),
                "support_candidate_pool": [
                    dict(candidate)
                    for candidate in support_pool_selection.get("support_candidate_pool", [])
                ],
                "anchor_alpha": float(args.anchor_alpha),
                "adapt_mix_rho": float(args.adapt_mix_rho),
                "ridge_lambda": float(args.ridge_lambda),
                "ridge_weighting": str(args.ridge_weighting),
                "trust_region_radius": float(args.ridge_trust_region_radius),
                "coefficient_norm_radius": float(args.ridge_clip_coeff_norm),
                "support_loss_reduction": str(args.support_loss_reduction),
                "support_cycle_improvement_fraction": float(
                    support_pool_selection.get("support_cycle_improvement_fraction", 0.0) or 0.0
                ),
                "support_cv_summary": dict(support_pool_selection.get("selected_cv_summary", {}) or {}),
                "k12_reference_policy": "k4_safe_nested_reference" if args.K == 12 else "",
                "effective_calibration_dof": ridge_diagnostics.get("effective_calibration_dof"),
                "feature_dim": ridge_diagnostics.get("feature_dim"),
                "feature_observation_count": ridge_diagnostics.get("feature_observation_count"),
                "support_count": ridge_diagnostics.get("support_count"),
                "source_policy": {
                    "pool_source": str(args.source_anchor_hyperparameter_source),
                    "selection_source": (
                        "target_support_only_loocv_mixed_raw_increment_wrmse"
                        if args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested"
                        else "target_support_only_exact_mixed_predictor_loss"
                    ),
                    "rho_source": str(getattr(args, "rho_policy", "")),
                },
            }
        elif args.K > 0 and args.adapt_solver == "ridge_coeff":
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
                ridge_weighting=args.ridge_weighting,
                surface_weight=args.surface_weight,
                rootzone_weight=args.rootzone_weight,
                use_lat_weighted_loss=args.use_lat_weighted_loss,
            )
            print(
                "Ridge coefficient adaptation finished in "
                f"{time.time() - started:.1f}s: "
                f"status={ridge_diagnostics.get('status')} "
                f"lambda={args.ridge_lambda} "
                f"weighting={args.ridge_weighting} "
                f"coef_norm={ridge_diagnostics.get('coefficient_norm')} "
                f"masked_obs={ridge_diagnostics.get('masked_observation_count')}",
                flush=True,
            )
            ridge_calibration_mode = "target_support_linearized_coeff_ridge"
            if args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v6_nested":
                ridge_calibration_mode = "target_support_linearized_coeff_ridge_v6_nested"
            elif args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v7_balanced_nested":
                ridge_calibration_mode = "target_support_linearized_coeff_ridge_v7_balanced_nested"
            elif args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v8_hybrid_nested":
                ridge_calibration_mode = "target_support_linearized_coeff_ridge_v8_hybrid_nested"
            elif args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v9_guarded_nested":
                ridge_calibration_mode = "target_support_linearized_coeff_ridge_v9_guarded_nested"
            linearized_ridge_calibration = {
                "calibration_mode": ridge_calibration_mode,
                "status": ridge_diagnostics.get("status"),
                "label_source": "target_support_only",
                "target_eval_usage": "final_eval_only_no_selection",
                "parameter_scope": "adapter_coefficient_residuals_only",
                "frozen_groups": list(FROZEN_SOURCE_GROUPS),
                "ridge_lambda": float(args.ridge_lambda),
                "ridge_weighting": str(args.ridge_weighting),
                "anchor_alpha": float(args.anchor_alpha),
                "trust_region_radius": float(args.ridge_trust_region_radius),
                "coefficient_norm_radius": float(args.ridge_clip_coeff_norm),
                "support_loss_reduction": str(args.support_loss_reduction),
                "effective_calibration_dof": ridge_diagnostics.get("effective_calibration_dof"),
                "feature_dim": ridge_diagnostics.get("feature_dim"),
                "feature_observation_count": ridge_diagnostics.get("feature_observation_count"),
                "support_count": ridge_diagnostics.get("support_count"),
                "source_policy": {
                    "lambda_source": str(args.source_anchor_hyperparameter_source),
                    "trust_radius_source": str(args.source_anchor_hyperparameter_source),
                    "anchor_source": str(args.source_anchor_hyperparameter_source),
                    "rho_source": str(getattr(args, "rho_policy", "")),
                },
            }
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

        if (
            args.K > 0
            and loss_loader is not None
            and args.stage3_kshot_mode in {
                "diagnostic_linearized_coeff_ridge_v10_support_pool_nested",
                "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested",
            }
            and support_pool_selection
        ):
            pool_defaults = (
                diagnostic_linearized_coeff_ridge_v11_defaults(args.K)
                if args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested"
                else diagnostic_linearized_coeff_ridge_v10_defaults(args.K)
            )
            min_delta = float(pool_defaults["support_gate_min_delta"])
            min_fraction = float(pool_defaults["support_gate_cycle_improvement_min_fraction"])
            support_nesting_metadata = _support_nesting_metadata(support_dataset)
            standard_support_loss_before = (
                dict(support_pool_selection.get("k0_anchor_score", {}))
                if args.K == 4
                else dict(support_pool_selection.get("k4_reference_score", {}))
            )
            standard_support_loss_after = dict(support_pool_selection.get("selected_score", {}))
            if args.K == 4:
                if args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested":
                    support_gate_summary = decide_v11_k4_support_pool_gate(
                        full_candidate=support_pool_selection["selected_score"],
                        full_reference=support_pool_selection["k0_anchor_score"],
                        cv_summary=support_pool_selection.get("selected_cv_summary", {}),
                        min_delta=min_delta,
                        min_cycle_improvement_fraction=min_fraction,
                        selected_candidate_id=str(support_pool_selection["selected_support_candidate_id"]),
                        support_candidate_pool=list(support_pool_selection.get("support_candidate_pool", [])),
                    )
                else:
                    support_gate_summary = decide_v10_k4_support_pool_gate(
                        candidate=support_pool_selection["selected_score"],
                        k0_anchor=support_pool_selection["k0_anchor_score"],
                        min_delta=min_delta,
                        min_cycle_improvement_fraction=min_fraction,
                        cycle_improvement_fraction=float(
                            support_pool_selection.get("support_cycle_improvement_fraction", 0.0) or 0.0
                        ),
                        selected_candidate_id=str(support_pool_selection["selected_support_candidate_id"]),
                        support_candidate_pool=list(support_pool_selection.get("support_candidate_pool", [])),
                    )
                if support_gate_summary["stage3_posterior_decision"] == "rejected_to_k0_anchor":
                    apply_target_adapter_state(state.model, anchor_adapter_state)
                    standard_support_loss_after = dict(support_pool_selection.get("k0_anchor_score", {}))
                    args.adapt_mix_rho = 0.0
                    print(
                        f"{args.stage3_kshot_mode} K4 support pool rejected selected candidate; rolled back to K0 anchor: "
                        f"reasons={support_gate_summary.get('support_gate_reject_reason')}",
                        flush=True,
                    )
                else:
                    print(
                        f"{args.stage3_kshot_mode} K4 support pool accepted candidate: "
                        f"{support_pool_selection.get('selected_support_candidate_id')} "
                        f"delta={support_gate_summary.get('support_candidate_objective_delta')}",
                        flush=True,
                    )
            else:
                if k4_reference_state is None:
                    raise RuntimeError(f"{args.stage3_kshot_mode} K12 gate missing K4 reference state")
                if args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v11_loocv_support_pool_nested":
                    support_gate_summary = decide_v11_k12_support_pool_gate(
                        full_candidate=support_pool_selection["selected_score"],
                        full_reference=support_pool_selection["k4_reference_score"],
                        cv_summary=support_pool_selection.get("selected_cv_summary", {}),
                        min_delta=min_delta,
                        min_cycle_improvement_fraction=min_fraction,
                        selected_candidate_id=str(support_pool_selection["selected_support_candidate_id"]),
                        support_candidate_pool=list(support_pool_selection.get("support_candidate_pool", [])),
                        k4_reference_adapt_mix_rho=k4_reference_adapt_mix_rho,
                        support_nesting_policy=str(support_nesting_metadata.get("support_nesting_policy", "")),
                        nested_support_dates_hash=str(support_nesting_metadata.get("nested_support_dates_hash", "")),
                    )
                else:
                    support_gate_summary = decide_v10_k12_support_pool_gate(
                        candidate=support_pool_selection["selected_score"],
                        k4_reference=support_pool_selection["k4_reference_score"],
                        candidate_nested_k4=support_pool_selection["selected_nested_k4_score"],
                        k4_reference_nested=support_pool_selection["k4_reference_nested_k4_score"],
                        min_delta=min_delta,
                        min_cycle_improvement_fraction=min_fraction,
                        cycle_improvement_fraction=float(
                            support_pool_selection.get("support_cycle_improvement_fraction", 0.0) or 0.0
                        ),
                        selected_candidate_id=str(support_pool_selection["selected_support_candidate_id"]),
                        support_candidate_pool=list(support_pool_selection.get("support_candidate_pool", [])),
                        k4_reference_adapt_mix_rho=k4_reference_adapt_mix_rho,
                        support_nesting_policy=str(support_nesting_metadata.get("support_nesting_policy", "")),
                        nested_support_dates_hash=str(support_nesting_metadata.get("nested_support_dates_hash", "")),
                    )
                k4_reference_gate = dict(support_gate_summary)
                k4_reference_gate.update(
                    {
                        "k4_reference_checkpoint": str(args.k4_reference_checkpoint),
                        "k4_reference_checkpoint_sha256": k4_reference_checkpoint_sha256,
                    }
                )
                if support_gate_summary["stage3_posterior_decision"] == "fallback_to_k4_reference":
                    apply_target_adapter_state(state.model, k4_reference_state)
                    standard_support_loss_after = dict(support_pool_selection.get("k4_reference_score", {}))
                    if k4_reference_adapt_mix_rho is not None:
                        args.adapt_mix_rho = float(k4_reference_adapt_mix_rho)
                    support_gate_summary["support_only_gate_status"] = support_gate_summary["support_gate_status"]
                    print(
                        f"{args.stage3_kshot_mode} K12 support pool fell back to K4 reference: "
                        f"reasons={support_gate_summary.get('support_gate_reject_reason')}",
                        flush=True,
                    )
                else:
                    support_gate_summary["support_only_gate_status"] = support_gate_summary["support_gate_status"]
                    print(
                        f"{args.stage3_kshot_mode} K12 support pool accepted candidate over K4 reference: "
                        f"{support_pool_selection.get('selected_support_candidate_id')} "
                        f"delta={support_gate_summary.get('k12_vs_k4_support_objective_delta')}",
                        flush=True,
                    )
        elif args.K > 0 and loss_loader is not None and loss_fn is not None:
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
            defer_k0_anchor_gate = (
                args.K == 12
                and args.stage3_kshot_mode == "diagnostic_linearized_coeff_ridge_v9_guarded_nested"
                and loss_loader is not None
                and loss_fn is not None
                and bool(args.k4_reference_checkpoint)
                and support_gate_summary["stage3_posterior_decision"] == "rejected_to_k0_anchor"
            )
            if defer_k0_anchor_gate:
                support_gate_summary, deferred_k0_anchor_gate = defer_k0_anchor_gate_to_k4_reference_gate(
                    support_gate_summary
                )
                print(
                    "Support gate deferred K12-vs-K0 rejection to nested K4 reference gate: "
                    f"reasons={deferred_k0_anchor_gate.get('support_gate_reject_reason')}",
                    flush=True,
                )
            elif support_gate_summary["stage3_posterior_decision"] == "rejected_to_k0_anchor":
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
            candidate_drift = target_parameter_l2_drift(
                anchor_adapter_state,
                extract_target_adapter_state(state.model),
            )
            support_gate_summary = apply_diagnostic_direct_v2_support_risk_guard(
                summary=support_gate_summary,
                stage3_kshot_mode=args.stage3_kshot_mode,
                support_gradient_diagnostics=support_gradient_diagnostics,
                target_parameter_drift=candidate_drift,
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
                    "Diagnostic v2 support-risk guard rejected Stage-3 posterior; "
                    "rolled back to frozen-prior anchor: "
                    f"reasons={support_gate_summary.get('support_gate_reject_reason')}",
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

        if (
            args.K == 12
            and args.stage3_kshot_mode in {
                "diagnostic_safe_operator_v5_nested",
                "diagnostic_finetune_support_gain_v14_nested",
                "diagnostic_linearized_coeff_ridge_v6_nested",
                "diagnostic_linearized_coeff_ridge_v7_balanced_nested",
                "diagnostic_linearized_coeff_ridge_v8_hybrid_nested",
                "diagnostic_linearized_coeff_ridge_v9_guarded_nested",
            }
            and loss_loader is not None
            and loss_fn is not None
            and str(support_gate_summary.get("stage3_posterior_decision", "accepted")) == "accepted"
        ):
            support_nesting_metadata = _support_nesting_metadata(support_dataset)
            if not str(args.k4_reference_checkpoint or ""):
                support_gate_summary.update(
                    {
                        "support_gate_enabled": True,
                        "support_gate_status": "missing_k4_reference_rejected_to_k0_anchor",
                        "support_only_gate_status": str(
                            support_gate_summary.get("support_gate_status", "missing_k4_reference")
                        ),
                        "stage3_posterior_decision": "rejected_to_k0_anchor",
                        "support_gate_reject_reason": list(
                            support_gate_summary.get("support_gate_reject_reason", []) or []
                        )
                        + [f"missing_k4_reference_checkpoint_for_{args.stage3_kshot_mode}"],
                        "target_eval_usage": "final_eval_only_no_selection",
                        **support_nesting_metadata,
                    }
                )
                apply_target_adapter_state(state.model, anchor_adapter_state)
                standard_support_loss_after = dict(standard_support_loss_before)
            else:
                k12_candidate_state = extract_target_adapter_state(state.model)
                k4_reference_checkpoint_sha256 = (
                    compute_sha256(args.k4_reference_checkpoint)
                    if Path(args.k4_reference_checkpoint).exists()
                    else ""
                )
                k4_reference_state = load_k4_reference_target_adapter_state(
                    args.k4_reference_checkpoint,
                    device,
                )
                k4_reference_adapt_mix_rho = load_k4_reference_adapt_mix_rho(args.k4_reference_checkpoint)
                apply_target_adapter_state(state.model, k4_reference_state)
                k4_reference_support_loss = standard_support_loss_from_loader(
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
                apply_target_adapter_state(state.model, k12_candidate_state)
                k4_reference_gate = decide_k12_vs_k4_reference_gate(
                    candidate=standard_support_loss_after,
                    k4_reference=k4_reference_support_loss,
                    enabled=True,
                    min_delta=float(args.support_gate_min_delta),
                    k4_reference_adapt_mix_rho=k4_reference_adapt_mix_rho,
                    support_nesting_policy=str(support_nesting_metadata.get("support_nesting_policy", "")),
                    nested_support_dates_hash=str(support_nesting_metadata.get("nested_support_dates_hash", "")),
                )
                k4_reference_gate.update(
                    {
                        "k4_reference_checkpoint": str(args.k4_reference_checkpoint),
                        "k4_reference_checkpoint_sha256": k4_reference_checkpoint_sha256,
                    }
                )
                if k4_reference_gate["stage3_posterior_decision"] == "fallback_to_k4_reference":
                    apply_target_adapter_state(state.model, k4_reference_state)
                    standard_support_loss_after = dict(k4_reference_support_loss)
                    if k4_reference_adapt_mix_rho is not None:
                        args.adapt_mix_rho = float(k4_reference_adapt_mix_rho)
                    support_gate_summary.update(k4_reference_gate)
                    support_gate_summary["support_only_gate_status"] = k4_reference_gate["support_gate_status"]
                    print(
                        f"{args.stage3_kshot_mode} K12 support gate fell back to K4 reference on nested support: "
                        f"delta={k4_reference_gate.get('k12_vs_k4_support_objective_delta')}",
                        flush=True,
                    )
                else:
                    support_gate_summary.update(k4_reference_gate)
                    support_gate_summary["support_only_gate_status"] = k4_reference_gate["support_gate_status"]
                    print(
                        f"{args.stage3_kshot_mode} K12 support gate kept K12 over K4 reference: "
                        f"delta={k4_reference_gate.get('k12_vs_k4_support_objective_delta')}",
                        flush=True,
                    )

        if args.K > 0 and args.stage3_kshot_mode in {
            "diagnostic_support_gain_v2",
            "diagnostic_support_gain_v3_stable",
            "diagnostic_support_gain_v4_nested_stable",
            "diagnostic_support_gain_v12_nested_cv",
            "diagnostic_support_gain_v13_k12_aggressive_calibration_pool",
            "diagnostic_finetune_support_gain_v14_nested",
        }:
            if args.stage3_kshot_mode in {
                "diagnostic_support_gain_v12_nested_cv",
                "diagnostic_support_gain_v13_k12_aggressive_calibration_pool",
                "diagnostic_finetune_support_gain_v14_nested",
            }:
                support_gain_selection_rule = "support_uncertainty_nested_cv_no_harm"
            elif args.stage3_kshot_mode == "diagnostic_support_gain_v4_nested_stable":
                support_gain_selection_rule = "support_uncertainty_stable_high_alpha_with_dual_guard"
            elif args.stage3_kshot_mode == "diagnostic_support_gain_v3_stable":
                support_gain_selection_rule = "stable_high_alpha_with_mean_skill_guard"
            else:
                support_gain_selection_rule = "max_min_skill"
            support_nesting_metadata = _support_nesting_metadata(support_dataset)
            if loss_loader is None:
                if support_gain_selection_rule == "support_uncertainty_nested_cv_no_harm":
                    calibration_mode = "target_support_residual_gain_v12_nested_cv"
                elif support_gain_selection_rule == "support_uncertainty_stable_high_alpha_with_dual_guard":
                    calibration_mode = "target_support_residual_gain_v4_nested_stable_grid"
                elif support_gain_selection_rule == "stable_high_alpha_with_mean_skill_guard":
                    calibration_mode = "target_support_residual_gain_stable_grid"
                else:
                    calibration_mode = "target_support_residual_gain_fixed_grid"
                support_gain_calibration = {
                    "calibration_mode": calibration_mode,
                    "status": "skipped_no_support_loader",
                    "label_source": "target_support_only",
                    "target_eval_usage": "final_eval_only_no_selection",
                    "selection_rule": support_gain_selection_rule,
                    "best_alpha_raw": 1.0,
                    "stable_candidate_alphas": [],
                    "selection_margin": 0.0,
                    "alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
                    "support_count": 0,
                    "best_alpha_surface": 1.0,
                    "best_alpha_rootzone": 1.0,
                    "stability_tolerance": 0.0,
                    "paired_support_se_capped": 0.0,
                    **support_nesting_metadata,
                }
            elif (
                args.stage3_kshot_mode == "diagnostic_support_gain_v13_k12_aggressive_calibration_pool"
                and int(args.K) == 12
            ):
                k4_reference_checkpoint_sha256 = (
                    compute_sha256(args.k4_reference_checkpoint)
                    if args.k4_reference_checkpoint and Path(args.k4_reference_checkpoint).exists()
                    else ""
                )
                support_gain_calibration = calibrate_support_gain_v13_k12_aggressive_pool_from_loader(
                    state=state,
                    loader=loss_loader,
                    device=device,
                    target_context_prompt_state=target_context_prompt_state,
                    k4_reference_checkpoint=str(args.k4_reference_checkpoint or ""),
                    surface_weight=float(args.surface_weight),
                    rootzone_weight=float(args.rootzone_weight),
                    support_nesting_metadata=support_nesting_metadata,
                )
            elif args.stage3_kshot_mode == "diagnostic_finetune_support_gain_v14_nested":
                support_gain_calibration = calibrate_support_residual_gain_v12_nested_cv_from_loader(
                    state=state,
                    loader=loss_loader,
                    device=device,
                    target_context_prompt_state=target_context_prompt_state,
                    K=args.K,
                    alpha_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
                    surface_weight=float(args.surface_weight),
                    rootzone_weight=float(args.rootzone_weight),
                    support_nesting_metadata=support_nesting_metadata,
                )
                gain_gate = dict(support_gain_calibration.get("support_gain_gate", {}) or {})
                support_gain_calibration.update(
                    {
                        "calibration_mode": "target_support_residual_gain_v14_after_finetune_nested_cv",
                        "v14_role": "auxiliary_support_gain_after_target_parameter_finetune",
                        "gain_gate_decision_is_auxiliary": True,
                        "finetune_parameter_gate_decision": str(
                            support_gate_summary.get("stage3_posterior_decision", "accepted")
                        ),
                        "finetune_parameter_gate_status": str(
                            support_gate_summary.get("support_gate_status", "")
                        ),
                    }
                )
                if gain_gate.get("stage3_posterior_decision") != "accepted":
                    support_gain_calibration.update(
                        {
                            "status": "gain_identity_after_finetune",
                            "best_alpha_surface": 1.0,
                            "best_alpha_rootzone": 1.0,
                            "selected_alpha_after_gate": 1.0,
                            "support_gain_gate": gain_gate
                            | {
                                "stage3_posterior_decision": "gain_identity_after_finetune",
                                "support_gate_status": "support_gain_identity_after_finetune",
                            },
                        }
                    )
                else:
                    support_gain_calibration["status"] = "accepted_after_finetune"
            elif args.stage3_kshot_mode == "diagnostic_support_gain_v12_nested_cv":
                support_gain_calibration = calibrate_support_residual_gain_v12_nested_cv_from_loader(
                    state=state,
                    loader=loss_loader,
                    device=device,
                    target_context_prompt_state=target_context_prompt_state,
                    K=args.K,
                    alpha_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
                    surface_weight=float(args.surface_weight),
                    rootzone_weight=float(args.rootzone_weight),
                    support_nesting_metadata=support_nesting_metadata,
                )
            elif (
                args.stage3_kshot_mode == "diagnostic_support_gain_v13_k12_aggressive_calibration_pool"
                and int(args.K) == 4
            ):
                support_gain_calibration = calibrate_support_residual_gain_v12_nested_cv_from_loader(
                    state=state,
                    loader=loss_loader,
                    device=device,
                    target_context_prompt_state=target_context_prompt_state,
                    K=args.K,
                    alpha_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
                    surface_weight=float(args.surface_weight),
                    rootzone_weight=float(args.rootzone_weight),
                    support_nesting_metadata=support_nesting_metadata,
                )
                support_gain_calibration.update(
                    {
                        "v13_role": "run_local_k4_reference_uses_v12_global_residual_gain",
                        "support_selection_objective": "v12_global_residual_gain_reference_for_v13",
                    }
                )
            else:
                support_gain_calibration = calibrate_support_residual_gain_from_loader(
                    state=state,
                    loader=loss_loader,
                    device=device,
                    target_context_prompt_state=target_context_prompt_state,
                    alpha_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
                    selection_rule=support_gain_selection_rule,
                )
                support_gain_calibration.update(support_nesting_metadata)
            residual_gain_alpha_surface = float(support_gain_calibration.get("best_alpha_surface", 1.0))
            residual_gain_alpha_rootzone = float(support_gain_calibration.get("best_alpha_rootzone", 1.0))
            if args.stage3_kshot_mode in {
                "diagnostic_support_gain_v12_nested_cv",
                "diagnostic_support_gain_v13_k12_aggressive_calibration_pool",
                "diagnostic_finetune_support_gain_v14_nested",
            }:
                support_gain_gate_summary = dict(support_gain_calibration.get("support_gain_gate", {}) or {})
                if args.stage3_kshot_mode != "diagnostic_finetune_support_gain_v14_nested":
                    support_gate_summary = dict(support_gain_gate_summary)
                if support_gate_summary:
                    support_gate_summary.update(
                        {
                            "support_nesting_policy": support_nesting_metadata.get("support_nesting_policy", ""),
                            "nested_support_dates_hash": support_nesting_metadata.get("nested_support_dates_hash", ""),
                        }
                    )
                if args.stage3_kshot_mode == "diagnostic_finetune_support_gain_v14_nested":
                    support_gate_summary["support_gain_gate"] = support_gain_gate_summary
                    support_gate_summary["support_gain_gate_status"] = support_gain_gate_summary.get(
                        "support_gate_status"
                    )
                    support_gate_summary["support_gain_stage3_decision"] = support_gain_gate_summary.get(
                        "stage3_posterior_decision"
                    )
                    support_gate_summary["support_gain_selected_alpha_raw"] = support_gain_gate_summary.get(
                        "support_gain_selected_alpha_raw"
                    )
                if args.stage3_kshot_mode == "diagnostic_support_gain_v13_k12_aggressive_calibration_pool":
                    support_affine_calibration = dict(
                        support_gain_calibration.get("support_affine_calibration", {}) or {}
                    )
                    support_pool_selection = dict(support_gain_calibration)
                    if support_gain_calibration.get("selected_support_candidate_id"):
                        support_gate_summary["selected_support_candidate_id"] = str(
                            support_gain_calibration.get("selected_support_candidate_id")
                        )
                    support_gate_summary["support_candidate_pool"] = list(
                        support_gain_calibration.get("support_candidate_pool", []) or []
                    )
                    support_gate_summary["support_selection_objective"] = support_gain_calibration.get(
                        "support_selection_objective",
                        support_gate_summary.get("support_selection_objective", ""),
                    )
                    support_gate_summary["support_calibration_dof"] = support_gain_calibration.get(
                        "support_calibration_dof",
                        support_gate_summary.get("support_calibration_dof"),
                    )
                    support_gate_summary["best_rejected_support_candidate"] = dict(
                        support_gain_calibration.get(
                            "best_rejected_support_candidate",
                            support_gate_summary.get("best_rejected_support_candidate", {}),
                        )
                        or {}
                    )
                    selected_score = support_gain_calibration.get("selected_score", {})
                    reference_score = support_gain_calibration.get("k4_reference_score", {})
                    if isinstance(reference_score, dict):
                        standard_support_loss_before = dict(reference_score)
                    if isinstance(selected_score, dict):
                        standard_support_loss_after = dict(selected_score)
                    if args.K == 12 and args.k4_reference_checkpoint:
                        k4_reference_gate = dict(support_gate_summary)
                        k4_reference_gate.update(
                            {
                                "k4_reference_checkpoint": str(args.k4_reference_checkpoint),
                                "k4_reference_checkpoint_sha256": k4_reference_checkpoint_sha256,
                                "k4_reference_gain_alpha_surface": support_gain_calibration.get(
                                    "k4_reference_gain_alpha_surface"
                                ),
                                "k4_reference_gain_alpha_rootzone": support_gain_calibration.get(
                                    "k4_reference_gain_alpha_rootzone"
                                ),
                            }
                        )
                        support_gate_summary.update(
                            {
                                "k4_reference_checkpoint": str(args.k4_reference_checkpoint),
                                "k4_reference_checkpoint_sha256": k4_reference_checkpoint_sha256,
                            }
                        )
                    support_gate_summary["support_only_gate_status"] = support_gate_summary.get(
                        "support_gate_status"
                    )
                if support_gate_summary.get("stage3_posterior_decision") == "fallback_to_k4_reference":
                    support_gain_calibration["status"] = "fallback_to_k4_reference"
                    if args.stage3_kshot_mode == "diagnostic_support_gain_v13_k12_aggressive_calibration_pool":
                        support_gain_calibration["selected_alpha_after_gate"] = residual_gain_alpha_surface
                    elif args.k4_reference_checkpoint:
                        k4_reference_checkpoint_sha256 = (
                            compute_sha256(args.k4_reference_checkpoint)
                            if Path(args.k4_reference_checkpoint).exists()
                            else ""
                        )
                        k4_gain_surface, k4_gain_rootzone = load_k4_reference_residual_gain_alphas(
                            args.k4_reference_checkpoint
                        )
                        residual_gain_alpha_surface = k4_gain_surface
                        residual_gain_alpha_rootzone = k4_gain_rootzone
                        support_gain_calibration["best_alpha_surface"] = residual_gain_alpha_surface
                        support_gain_calibration["best_alpha_rootzone"] = residual_gain_alpha_rootzone
                        support_gain_calibration["fallback_reference_gain_alpha_surface"] = k4_gain_surface
                        support_gain_calibration["fallback_reference_gain_alpha_rootzone"] = k4_gain_rootzone
                        k4_reference_gate = dict(support_gate_summary)
                        k4_reference_gate.update(
                            {
                                "k4_reference_checkpoint": str(args.k4_reference_checkpoint),
                                "k4_reference_checkpoint_sha256": k4_reference_checkpoint_sha256,
                                "k4_reference_gain_alpha_surface": k4_gain_surface,
                                "k4_reference_gain_alpha_rootzone": k4_gain_rootzone,
                            }
                        )
                        support_gate_summary.update(
                            {
                                "k4_reference_checkpoint": str(args.k4_reference_checkpoint),
                                "k4_reference_checkpoint_sha256": k4_reference_checkpoint_sha256,
                            }
                        )
                    else:
                        residual_gain_alpha_surface = 1.0
                        residual_gain_alpha_rootzone = 1.0
                    support_gain_calibration["selected_alpha_after_gate"] = residual_gain_alpha_surface
                    support_gate_summary["support_only_gate_status"] = support_gate_summary.get("support_gate_status")
                elif support_gate_summary.get("stage3_posterior_decision") == "rejected_to_k0_anchor":
                    support_gain_calibration["status"] = "rejected_to_identity"
                    residual_gain_alpha_surface = 1.0
                    residual_gain_alpha_rootzone = 1.0
                    support_gain_calibration["best_alpha_surface"] = 1.0
                    support_gain_calibration["best_alpha_rootzone"] = 1.0
                    support_gain_calibration["selected_alpha_after_gate"] = 1.0
            print(
                "Support residual-gain calibration: "
                f"status={support_gain_calibration.get('status')} "
                f"surface={residual_gain_alpha_surface:.3f} "
                f"rootzone={residual_gain_alpha_rootzone:.3f}",
                flush=True,
            )

        if args.K > 0 and args.stage3_kshot_mode == "diagnostic_support_affine_v1_nested":
            support_nesting_metadata = _support_nesting_metadata(support_dataset)
            if loss_loader is None:
                support_affine_calibration = {
                    "calibration_mode": (
                        "target_support_residual_affine_v1_nested"
                        if int(args.K) == 12
                        else "target_support_residual_affine_v1"
                    ),
                    "status": "skipped_no_support_loader",
                    "label_source": "target_support_only",
                    "target_eval_usage": "final_eval_only_no_selection",
                    "support_affine_coefficients": {
                        "surface": {"a": 1.0, "b": 0.0, "status": "skipped_no_support_loader"},
                        "rootzone": {"a": 1.0, "b": 0.0, "status": "skipped_no_support_loader"},
                    },
                    "seasonal_affine_coefficients": {},
                    "effective_calibration_dof": 0.0,
                    **support_nesting_metadata,
                }
            else:
                support_affine_calibration = calibrate_support_residual_affine_from_loader(
                    state=state,
                    loader=loss_loader,
                    device=device,
                    target_context_prompt_state=target_context_prompt_state,
                    K=args.K,
                    support_nesting_metadata=support_nesting_metadata,
                )
            print(
                "Support residual-affine calibration: "
                f"status={support_affine_calibration.get('status')} "
                f"dof={support_affine_calibration.get('effective_calibration_dof')}",
                flush=True,
            )

        if (
            args.K > 0
            and args.policy_source != "source_side_episode_calibration"
            and not is_diagnostic_direct_kshot_mode(args.stage3_kshot_mode)
        ):
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
        target_labels_used = (
            bool(train_history)
            or bool(ridge_diagnostics)
            or bool(support_gain_calibration)
            or bool(support_affine_calibration)
            or bool(args.K > 0 and support_gate_summary.get("support_gate_enabled", False))
        )
        target_support_labels_used_for_parameter_update = bool(train_history)
        target_support_labels_used_for_ridge_solve = bool(ridge_diagnostics)
        target_support_labels_used_for_calibration = bool(
            support_gain_calibration or support_affine_calibration
        )
        if target_support_labels_used_for_parameter_update and target_support_labels_used_for_calibration:
            few_shot_update_type = "parameter_update_plus_support_calibration"
        elif target_support_labels_used_for_parameter_update:
            few_shot_update_type = "parameter_update_only"
        elif target_support_labels_used_for_ridge_solve and target_support_labels_used_for_calibration:
            few_shot_update_type = "ridge_solve_plus_support_calibration"
        elif target_support_labels_used_for_ridge_solve:
            few_shot_update_type = "ridge_solve_only"
        elif target_support_labels_used_for_calibration:
            few_shot_update_type = "support_calibration_only"
        elif args.K > 0 and bool(support_gate_summary.get("support_gate_enabled", False)):
            few_shot_update_type = "support_gate_only"
        else:
            few_shot_update_type = "no_target_label_update"
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
        support_nesting_metadata = _support_nesting_metadata(support_dataset)
        stage3_prior_snapshot = build_stage3_prior_snapshot_metadata(
            source_config=state.source_config,
            prompt_state=target_context_prompt_state,
            source_checkpoint_sha256=source_checkpoint_sha256,
            target_region=args.target_region,
            K=args.K,
        )
        stage3_prompt_metadata = target_context_prompt_metadata(target_context_prompt_state)
        paper_facing_status = paper_facing_status_for_stage3(
            K=args.K,
            policy_source=args.policy_source,
            stage3_posterior_decision=str(support_gate_summary.get("stage3_posterior_decision", "accepted")),
            stage3_kshot_mode=args.stage3_kshot_mode,
            context_tta_mode=str(args.context_tta_mode),
            context_tta_effective=bool(stage3_prompt_metadata.get("context_tta_effective", False)),
            context_tta_source_stat_status=str(
                stage3_prompt_metadata.get("context_tta_source_stat_status", "not_requested")
            ),
        )
        paper_facing_run = bool(paper_facing_status["paper_facing_run"])
        source_calibrated_no_update = (
            args.K > 0
            and args.policy_source == "source_side_episode_calibration"
            and str(support_gate_summary.get("stage3_posterior_decision", "")) == "no_update"
        )
        paper_selection_basis = (
            "source_side_safe_policy_selected_no_update"
            if source_calibrated_no_update
            else (
                "source_side_safe_policy_only"
                if args.K > 0 and args.policy_source == "source_side_episode_calibration"
                else (
                    "diagnostic_no_source_safe_policy_json"
                    if args.K > 0 and not is_diagnostic_direct_kshot_mode(args.stage3_kshot_mode)
                    else (
                        f"{args.stage3_kshot_mode}_target_support_only"
                        if args.K > 0
                        else "zero_shot_no_target_labels"
                    )
                )
            )
        )
        if str(support_gate_summary.get("stage3_posterior_decision", "")) == "rejected_to_k0_anchor":
            stage3_acceptance_basis = (
                "source_policy_or_gate_rejected_to_k0_anchor"
                if paper_facing_run
                else (
                    "diagnostic_direct_kshot_rejected_to_k0_anchor"
                    if is_diagnostic_direct_kshot_mode(args.stage3_kshot_mode)
                    else "diagnostic_no_source_safe_policy_json_rejected_to_k0_anchor"
                )
            )
        elif str(support_gate_summary.get("stage3_posterior_decision", "")) == "fallback_to_k4_reference":
            stage3_acceptance_basis = f"{args.stage3_kshot_mode}_support_only_fallback_to_k4"
        elif source_calibrated_no_update:
            stage3_acceptance_basis = "source_side_safe_policy_selected_no_update_k0_equivalent"
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
            "stage3_kshot_mode": args.stage3_kshot_mode,
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
            "selected_support_candidate_id": str(support_gate_summary.get("selected_support_candidate_id", "")),
            "support_candidate_pool": list(support_gate_summary.get("support_candidate_pool", []) or []),
            "support_pool_selection": support_pool_selection,
            "support_selection_objective": support_gate_summary.get("support_selection_objective", ""),
            "support_cv_objective_delta": support_gate_summary.get("support_cv_objective_delta"),
            "support_cv_objective_delta_se": support_gate_summary.get("support_cv_objective_delta_se"),
            "support_cv_objective_delta_t": support_gate_summary.get("support_cv_objective_delta_t"),
            "support_cv_nested_k4_objective_delta": support_gate_summary.get(
                "support_cv_nested_k4_objective_delta"
            ),
            "support_cv_added_objective_delta": support_gate_summary.get("support_cv_added_objective_delta"),
            "k12_vs_k4_cv_objective_delta": support_gate_summary.get("k12_vs_k4_cv_objective_delta"),
            "k12_vs_k4_cv_rootzone_delta": support_gate_summary.get("k12_vs_k4_cv_rootzone_delta"),
            "support_calibration_dof": support_gate_summary.get(
                "support_calibration_dof",
                support_gain_calibration.get("support_calibration_dof"),
            ),
            "eligible_support_candidate_count": int(
                support_gain_calibration.get("eligible_support_candidate_count", 0) or 0
            ),
            "best_rejected_support_candidate": dict(
                support_gate_summary.get(
                    "best_rejected_support_candidate",
                    support_gain_calibration.get("best_rejected_support_candidate", {}),
                )
                or {}
            ),
            "support_cycle_improvement_fraction": support_gate_summary.get("support_cycle_improvement_fraction"),
            "support_gate_cycle_improvement_min_fraction": support_gate_summary.get(
                "support_gate_cycle_improvement_min_fraction"
            ),
            "k12_reference_policy": support_gate_summary.get("k12_reference_policy", ""),
            "ridge_lambda": float(args.ridge_lambda),
            "ridge_clip_coeff_norm": float(args.ridge_clip_coeff_norm),
            "ridge_trust_region_radius": float(args.ridge_trust_region_radius),
            "ridge_max_feature_pixels": int(args.ridge_max_feature_pixels),
            "ridge_standardize_features": bool(args.ridge_standardize_features),
            "ridge_weighting": str(args.ridge_weighting),
            "ridge_diagnostics": ridge_diagnostics,
            "linearized_ridge_calibration": linearized_ridge_calibration,
            "residual_gain_alpha_surface": float(residual_gain_alpha_surface),
            "residual_gain_alpha_rootzone": float(residual_gain_alpha_rootzone),
            "support_gain_calibration": support_gain_calibration,
            "support_affine_calibration": support_affine_calibration,
            "k4_reference_checkpoint": str(args.k4_reference_checkpoint or ""),
            "k4_reference_checkpoint_sha256": k4_reference_checkpoint_sha256,
            "k4_reference_gate": k4_reference_gate,
            "deferred_k0_anchor_gate": deferred_k0_anchor_gate,
            "audit_identity": bool(args.audit_identity),
            "audit_identity_tolerance": float(args.audit_identity_tolerance),
            "adapt_recipe": args.adapt_recipe,
            "policy_source": args.policy_source,
            "safe_policy_json": args.safe_policy_json or "",
            "safe_policy_json_sha256": args.safe_policy_json_sha256,
            "safe_policy_hash": args.safe_policy_hash,
            "safe_policy": args.safe_policy,
            "adaptation_step_policy_source": args.adaptation_step_policy_source,
            "resolved_mode_defaults": args.resolved_mode_defaults,
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
            "raw_adapted_adapter_state": pre_anchor_adapter_state,
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
            "support_nesting_policy": support_nesting_metadata.get("support_nesting_policy", ""),
            "nested_support_dates_hash": support_nesting_metadata.get("nested_support_dates_hash", ""),
            "nested_support_manifest": support_nesting_metadata.get("nested_support_manifest", ""),
            "target_support_count": target_support_count,
            "target_labels_loaded_for_adaptation": bool(support_dataset is not None),
            "target_labels_used_for_adaptation": target_labels_used,
            "target_support_labels_used_for_parameter_update": target_support_labels_used_for_parameter_update,
            "target_support_labels_used_for_optimizer_update": target_support_labels_used_for_parameter_update,
            "target_support_labels_used_for_ridge_solve": target_support_labels_used_for_ridge_solve,
            "target_support_labels_used_for_calibration": target_support_labels_used_for_calibration,
            "target_support_labels_used_for_support_gate": bool(
                args.K > 0 and support_gate_summary.get("support_gate_enabled", False)
            ),
            "few_shot_update_type": few_shot_update_type,
            "target_latent_dim": args.target_latent_dim,
            "enable_target_spatial_refine": args.enable_target_spatial_refine,
            "target_adapter_anchor_state": anchor_adapter_state,
            "target_context_max_samples": int(args.target_context_max_samples),
            "target_context_samples_used": int(target_context_sample_count),
            "context_tta_mode": args.context_tta_mode,
            "context_tta_residual_scale": float(args.context_tta_residual_scale),
            "context_tta_residual_clip_l2": float(args.context_tta_residual_clip_l2),
            "context_tta_state_hash": prompt_metadata.get("context_tta_state_hash", ""),
            "context_tta_label_usage": prompt_metadata.get("context_tta_label_usage", "none"),
            "context_tta_effective": bool(prompt_metadata.get("context_tta_effective", False)),
            "context_tta_source_stat_status": prompt_metadata.get("context_tta_source_stat_status", "not_requested"),
            "prompt_l2_delta_mean": float(prompt_metadata.get("prompt_l2_delta_mean", 0.0) or 0.0),
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
