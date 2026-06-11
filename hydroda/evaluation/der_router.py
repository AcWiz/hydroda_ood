"""Variable-wise dual-expert routing for HydroDA-DER.

Legacy/secondary note:
    The router selects experts on target_val=2022 and is not part of the V4.4
    zero/few-shot main protocol.

The router is intentionally late-fusion: each expert predicts a full sample,
then surface and rootzone increments are copied from the preregistered experts.
Target-eval labels are never used for routing; target-eval evaluation must load
an existing ``router_config.json`` selected on target_val=2022.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

import numpy as np
import pandas as pd


VARIABLES = ("surface", "rootzone")
ROUTER_METHOD = "HydroDA-DER"
ROUTER_SELECTION_SOURCE = "target_val_2022"
MODEL_SELECTION_SOURCE = "target_val_2022_preregistered_variable_wise_expert_selection"
DEFAULT_SELECTION_METRIC = "increment_rmse_latw"


class DualExpertRouterPredictor:
    """Predictor that routes variables to two fixed experts.

    The surface expert supplies ``pred_increment_surface`` and the rootzone
    expert supplies ``pred_increment_rootzone``. Analysis fields are recomputed
    from the sample forecasts plus routed increments so residual-gain or other
    expert-local analysis reconstruction never leaks across variables.
    """

    method_name = "hydroda_der_variable_wise_dual_expert_router"

    def __init__(
        self,
        *,
        surface_expert: Any,
        rootzone_expert: Any,
        surface_metadata: Mapping[str, Any] | None = None,
        rootzone_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.surface_expert = surface_expert
        self.rootzone_expert = rootzone_expert
        self.surface_metadata = dict(surface_metadata or {})
        self.rootzone_metadata = dict(rootzone_metadata or {})

    def predict(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        surface_pred = self.surface_expert.predict(sample)
        rootzone_pred = self.rootzone_expert.predict(sample)

        _require_prediction_keys(surface_pred, ["pred_increment_surface"])
        _require_prediction_keys(rootzone_pred, ["pred_increment_rootzone"])

        pred_inc_surface = np.asarray(surface_pred["pred_increment_surface"], dtype=np.float32)
        pred_inc_rootzone = np.asarray(rootzone_pred["pred_increment_rootzone"], dtype=np.float32)
        forecast_surface = np.asarray(sample["forecast_surface"], dtype=np.float32)
        forecast_rootzone = np.asarray(sample["forecast_rootzone"], dtype=np.float32)

        return {
            "pred_increment_surface": pred_inc_surface,
            "pred_increment_rootzone": pred_inc_rootzone,
            "pred_analysis_surface": (forecast_surface + pred_inc_surface).astype(np.float32),
            "pred_analysis_rootzone": (forecast_rootzone + pred_inc_rootzone).astype(np.float32),
            "der_surface_expert_method": getattr(self.surface_expert, "method_name", ""),
            "der_rootzone_expert_method": getattr(self.rootzone_expert, "method_name", ""),
            "der_surface_expert_checkpoint": str(self.surface_metadata.get("checkpoint", "")),
            "der_rootzone_expert_checkpoint": str(self.rootzone_metadata.get("checkpoint", "")),
        }


def _require_prediction_keys(pred: Mapping[str, Any], keys: Iterable[str]) -> None:
    missing = [key for key in keys if key not in pred]
    if missing:
        raise KeyError(f"DER expert output missing keys: {missing}")


def create_predictor(
    *,
    checkpoint: str | Path,
    predictor_type: str,
    device: str = "cuda",
    target_region: str | None = None,
) -> Any:
    """Create a predictor matching ``scripts/eval/evaluate_checkpoint.py`` types."""
    predictor_type = str(predictor_type)
    if predictor_type == "source_only":
        from hydroda.baselines.source_only import SourceOnlyBackbonePredictor

        return SourceOnlyBackbonePredictor(checkpoint_path=str(checkpoint), device=device)
    if predictor_type in {"prompt_conditioned", "hyperda_target_adapt"}:
        from hydroda.baselines.prompt_conditioned import PromptConditionedBackbonePredictor

        return PromptConditionedBackbonePredictor(
            checkpoint_path=str(checkpoint),
            device=device,
            target_region=target_region,
        )
    raise ValueError(
        f"Unsupported predictor_type={predictor_type!r}. "
        "Allowed: source_only, prompt_conditioned, hyperda_target_adapt."
    )


def build_dual_expert_predictor_from_config(
    router_config: Mapping[str, Any],
    *,
    device: str = "cuda",
    target_region: str | None = None,
) -> DualExpertRouterPredictor:
    selected = router_config.get("selected_experts", {})
    if not isinstance(selected, Mapping):
        raise ValueError("router_config selected_experts must be a mapping")
    for variable in VARIABLES:
        if variable not in selected:
            raise ValueError(f"router_config missing selected_experts.{variable}")
    surface_cfg = dict(selected["surface"])
    rootzone_cfg = dict(selected["rootzone"])
    target_region = target_region or str(router_config.get("target_region", ""))

    surface_expert = create_predictor(
        checkpoint=surface_cfg["checkpoint"],
        predictor_type=surface_cfg["predictor_type"],
        device=device,
        target_region=target_region,
    )
    rootzone_expert = create_predictor(
        checkpoint=rootzone_cfg["checkpoint"],
        predictor_type=rootzone_cfg["predictor_type"],
        device=device,
        target_region=target_region,
    )
    return DualExpertRouterPredictor(
        surface_expert=surface_expert,
        rootzone_expert=rootzone_expert,
        surface_metadata=surface_cfg,
        rootzone_metadata=rootzone_cfg,
    )


def select_variable_experts(
    metrics: pd.DataFrame,
    *,
    metric: str = DEFAULT_SELECTION_METRIC,
    split_role: str = "target_val",
) -> Dict[str, Dict[str, Any]]:
    """Select the lowest target-val WRMSE candidate independently per variable."""
    required = {"candidate_id", "split_role", "variable", "metric", "value"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"metrics frame missing required columns: {sorted(missing)}")

    selected: Dict[str, Dict[str, Any]] = {}
    for variable in VARIABLES:
        subset = metrics[
            (metrics["split_role"].astype(str) == split_role)
            & (metrics["variable"].astype(str) == variable)
            & (metrics["metric"].astype(str) == metric)
        ].copy()
        subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
        subset = subset[np.isfinite(subset["value"])]
        if subset.empty:
            raise ValueError(f"No finite {split_role} {metric} rows found for variable={variable}")
        grouped = subset.groupby("candidate_id", as_index=False)["value"].mean()
        grouped = grouped.sort_values(["value", "candidate_id"], kind="mergesort").reset_index(drop=True)
        row = grouped.iloc[0]
        selected[variable] = {
            "candidate_id": str(row["candidate_id"]),
            "target_val_metric": metric,
            "target_val_metric_value": float(row["value"]),
            "target_val_split_role": split_role,
        }
    return selected


def build_router_config(
    *,
    candidates: Iterable[Mapping[str, Any]],
    selection: Mapping[str, Mapping[str, Any]],
    target_region: str,
    adaptation_setting: str,
    seed: int,
    split_manifest_path: str,
    split_manifest_sha256: str,
    target_val_dates_hash: str = "",
    target_train_dates_hash: str = "",
    target_eval_dates_hash: str = "",
    protocol_freeze_id: str = "",
    selection_metric: str = DEFAULT_SELECTION_METRIC,
) -> Dict[str, Any]:
    """Build a serializable target-val registered DER router config."""
    candidate_list = [dict(c) for c in candidates]
    candidate_map = {str(c["candidate_id"]): dict(c) for c in candidate_list}
    selected_experts: Dict[str, Dict[str, Any]] = {}
    for variable in VARIABLES:
        if variable not in selection:
            raise ValueError(f"selection missing variable={variable}")
        selected_candidate_id = str(selection[variable]["candidate_id"])
        if selected_candidate_id not in candidate_map:
            raise ValueError(f"selection for {variable} references unknown candidate_id={selected_candidate_id!r}")
        candidate = dict(candidate_map[selected_candidate_id])
        selected_experts[variable] = {
            "candidate_id": selected_candidate_id,
            "checkpoint": str(candidate["checkpoint"]),
            "predictor_type": str(candidate["predictor_type"]),
            "target_val_metric": str(selection[variable].get("target_val_metric", selection_metric)),
            "target_val_metric_value": float(selection[variable]["target_val_metric_value"]),
            "target_val_split_role": str(selection[variable].get("target_val_split_role", "target_val")),
        }

    return {
        "schema_version": "hydroda_der_router_v1",
        "method": ROUTER_METHOD,
        "router_name": "Variable-Wise Dual-Expert Router",
        "router_selection_source": ROUTER_SELECTION_SOURCE,
        "model_selection_source": MODEL_SELECTION_SOURCE,
        "selection_metric": selection_metric,
        "target_region": target_region,
        "adaptation_setting": adaptation_setting,
        "seed": int(seed),
        "protocol_freeze_id": protocol_freeze_id,
        "split_manifest_path": split_manifest_path,
        "split_manifest_sha256": split_manifest_sha256,
        "target_train_dates_hash": target_train_dates_hash,
        "target_val_dates_hash": target_val_dates_hash,
        "target_eval_dates_hash": target_eval_dates_hash,
        "candidates": candidate_list,
        "selected_experts": selected_experts,
        "no_leakage_declaration": {
            "target_val_used_for_variable_wise_selection": True,
            "target_eval_used_for_selection": False,
            "target_eval_used_for_training": False,
            "target_eval_used_for_threshold_or_gating": False,
            "dynamic_target_eval_gating": False,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_router_config(config: Mapping[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(dict(config), f, indent=2)
    return out


def load_router_config(path: str | Path) -> Dict[str, Any]:
    router_path = Path(path)
    with router_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    _validate_router_config_shape(config, source_path=router_path)
    return config


def _validate_router_config_shape(config: MutableMapping[str, Any], *, source_path: Path) -> None:
    if config.get("method") != ROUTER_METHOD:
        raise ValueError(f"{source_path} is not a {ROUTER_METHOD} router config")
    if config.get("router_selection_source") != ROUTER_SELECTION_SOURCE:
        raise ValueError(
            f"{source_path} router_selection_source must be {ROUTER_SELECTION_SOURCE!r}; "
            f"got {config.get('router_selection_source')!r}"
        )
    selected = config.get("selected_experts")
    if not isinstance(selected, Mapping):
        raise ValueError(f"{source_path} missing selected_experts")
    for variable in VARIABLES:
        expert = selected.get(variable)
        if not isinstance(expert, Mapping):
            raise ValueError(f"{source_path} missing selected_experts.{variable}")
        for key in ["checkpoint", "predictor_type"]:
            if key not in expert:
                raise ValueError(f"{source_path} missing selected_experts.{variable}.{key}")
    declaration = config.get("no_leakage_declaration", {})
    if declaration and declaration.get("target_eval_used_for_selection") is not False:
        raise ValueError(f"{source_path} declares target_eval_used_for_selection={declaration.get('target_eval_used_for_selection')}")


def validate_eval_uses_router_config(
    *,
    split_type: str,
    router_config_path: str | Path | None,
) -> Dict[str, Any]:
    """Require a preregistered router config for target_eval evaluation."""
    split_type = str(split_type)
    if split_type in {"target_eval", "target_query"}:
        if router_config_path is None:
            raise FileNotFoundError("target_eval DER evaluation requires an existing router_config.json")
        path = Path(router_config_path)
        if not path.exists():
            raise FileNotFoundError(f"target_eval DER evaluation requires an existing router_config.json: {path}")
        return load_router_config(path)
    if router_config_path is not None and Path(router_config_path).exists():
        return load_router_config(router_config_path)
    return {}


def dataset_date_hash(dataset: Any, key: str) -> str:
    entry = getattr(dataset, "_split_entry", {})
    return str(entry.get(key, ""))


def target_val_dates_hash(train_dataset: Any | None, val_dataset: Any) -> str:
    train_dataset = train_dataset or val_dataset
    return (
        dataset_date_hash(val_dataset, "target_val_dates_hash")
        or dataset_date_hash(train_dataset, "target_val_dates_hash")
        or dataset_date_hash(train_dataset, "source_val_dates_hash")
    )
