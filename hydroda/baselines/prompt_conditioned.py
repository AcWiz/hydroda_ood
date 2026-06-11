"""Prompt-conditioned backbone predictor for HydroDA-OOD / HyperDA V4.

No-leakage declaration:
    - Uses trained FiLMConditionalResUNet + RegionPromptEncoder checkpoint
    - Prompt uses input-side features only (x, region_id, month)
    - No target_eval/query labels used in prompt construction
    - Held-out target unseen-region prompt fallback is retained for split compatibility
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import numpy as np
import torch

from hydroda.models.conditional_unet import FiLMConditionalResUNet
from hydroda.models.hyper_conditional_unet import HyperAdapterConditionalResUNet
from hydroda.models.prompt_encoder import RegionPromptEncoder


# Mapping from region name (e.g. "US-R1") to region index (0..5)
_REGION_TO_IDX = {
    "US-R1": 0,
    "US-R2": 1,
    "US-R3": 2,
    "US-R4": 3,
    "US-R5": 4,
    "US-R6": 5,
}

TARGET_CONTEXT_PROMPT_SCHEMA_VERSION = "target_context_prompt_state_v1"
TARGET_CONTEXT_PROMPT_SOURCE = "target_context_monthly_prompt_prototypes"
_MAIN_HYPERDA_METHOD_IDS = {
    "hyperda_zero_shot_context",
    "hyperda_few_shot_k4",
    "hyperda_few_shot_k12",
}


def _hyperda_method_id_from_config(config: Dict[str, Any]) -> Optional[str]:
    method = config.get("method")
    if method in _MAIN_HYPERDA_METHOD_IDS:
        return str(method)
    setting = config.get("adaptation_setting")
    if setting == "zero_shot_context":
        return "hyperda_zero_shot_context"
    if setting == "few_shot_k4":
        return "hyperda_few_shot_k4"
    if setting == "few_shot_k12":
        return "hyperda_few_shot_k12"
    return None


def _coerce_month(value: Any, date_str: str = "") -> int:
    try:
        month = int(value)
    except Exception:
        month = int(date_str[5:7]) if date_str and len(date_str) >= 7 else 6
    if month < 1 or month > 12:
        raise ValueError(f"month must be in 1..12, got {month}")
    return month


def _prompt_tensor(value: Any, device: torch.device | str | None = None) -> Optional[torch.Tensor]:
    if value is None:
        return None
    tensor = value.detach().clone() if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
    tensor = tensor.to(dtype=torch.float32)
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def normalize_target_context_prompt_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalized target-context prompt state with CPU tensor prototypes."""
    if not state:
        raise ValueError("target_context_prompt_state is empty")
    schema = state.get("schema_version")
    if schema != TARGET_CONTEXT_PROMPT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported target_context_prompt_state schema_version={schema!r}; "
            f"expected {TARGET_CONTEXT_PROMPT_SCHEMA_VERSION!r}"
        )

    monthly_counts_raw = state.get("monthly_counts", {})
    monthly_counts = {str(month): int(monthly_counts_raw.get(str(month), 0)) for month in range(1, 13)}
    monthly_raw = state.get("monthly_prototypes", {})
    monthly_prototypes = {
        str(month): _prompt_tensor(monthly_raw.get(str(month)))
        for month in range(1, 13)
    }
    global_prototype = _prompt_tensor(state.get("global_prototype"))
    if global_prototype is None:
        raise ValueError("target_context_prompt_state missing global_prototype")

    normalized = dict(state)
    normalized["schema_version"] = TARGET_CONTEXT_PROMPT_SCHEMA_VERSION
    normalized["prompt_source"] = TARGET_CONTEXT_PROMPT_SOURCE
    normalized["label_usage"] = "none"
    normalized["monthly_counts"] = monthly_counts
    normalized["monthly_prototypes"] = monthly_prototypes
    normalized["global_prototype"] = global_prototype.detach().cpu()
    context_hash = str(normalized.get("context_hash") or normalized.get("context_date_hash") or "")
    normalized["context_hash"] = context_hash
    normalized["context_date_hash"] = context_hash
    normalized["metadata"] = dict(state.get("metadata", {}))
    normalized["metadata"].setdefault("eval_input_usage", "none_for_prompt_update")
    normalized["metadata"].setdefault("eval_month_usage", "known_seasonal_phase_selector_only")
    normalized["metadata"].setdefault("temporal_usage", "month_of_year_seasonal_phase")
    return normalized


def compose_target_context_prompt_from_state(
    state: Dict[str, Any],
    months: int | Sequence[int] | torch.Tensor,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Select monthly target-context prompt prototypes, falling back to global."""
    normalized = normalize_target_context_prompt_state(state)
    if isinstance(months, torch.Tensor):
        month_values = [int(v) for v in months.detach().cpu().view(-1).tolist()]
    elif isinstance(months, int):
        month_values = [int(months)]
    else:
        month_values = [int(v) for v in months]

    prompts = []
    for month in month_values:
        month = _coerce_month(month)
        prompt = normalized["monthly_prototypes"].get(str(month))
        if prompt is None:
            prompt = normalized["global_prototype"]
        prompts.append(prompt.to(device=device) if device is not None else prompt)
    return torch.stack(prompts, dim=0)


def target_context_prompt_metadata(state: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_target_context_prompt_state(state)
    metadata = dict(normalized.get("metadata", {}))
    metadata.update(
        {
            "schema_version": normalized["schema_version"],
            "prompt_source": normalized["prompt_source"],
            "label_usage": normalized["label_usage"],
            "context_hash": normalized.get("context_hash", ""),
            "context_date_hash": normalized.get("context_date_hash", normalized.get("context_hash", "")),
            "n_samples": int(normalized.get("n_samples", sum(normalized["monthly_counts"].values()))),
            "date_start": normalized.get("date_start", ""),
            "date_end": normalized.get("date_end", ""),
            "monthly_counts": dict(normalized["monthly_counts"]),
        }
    )
    return metadata


def _hash_context_dates(dates: Sequence[str], monthly_counts: Dict[str, int]) -> str:
    payload = json.dumps(
        {"dates": list(dates), "monthly_counts": monthly_counts},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_target_context_prompt_state(
    samples: Iterable[Dict[str, Any]],
    prompt_encoder: RegionPromptEncoder,
    normalize_x: Callable[[torch.Tensor], torch.Tensor],
    target_region_embedding: torch.Tensor,
    device: torch.device | str,
    context_hash: str = "",
) -> Dict[str, Any]:
    """Build monthly target-context prompt prototypes from input-side fields only.

    Reads only ``x``, ``month``, and ``date_str`` from each sample. Target
    labels, increments, residuals, validation scores, and eval inputs are not
    consulted.
    """
    device = torch.device(device)
    by_month: Dict[int, list[torch.Tensor]] = {month: [] for month in range(1, 13)}
    dates: list[str] = []
    all_input_embs: list[torch.Tensor] = []
    all_temporal_embs: list[torch.Tensor] = []

    target_region_embedding = target_region_embedding.to(device=device, dtype=torch.float32)
    if target_region_embedding.ndim == 1:
        target_region_embedding = target_region_embedding.unsqueeze(0)
    elif target_region_embedding.ndim != 2 or target_region_embedding.shape[0] != 1:
        raise ValueError("target_region_embedding must have shape [16] or [1,16]")

    with torch.no_grad():
        for sample in samples:
            x = torch.from_numpy(np.asarray(sample["x"], dtype=np.float32)).unsqueeze(0).to(device)
            x_norm = normalize_x(x)
            input_stats = prompt_encoder._compute_input_stats(x_norm)
            input_emb = prompt_encoder.input_proj(input_stats)

            date_str = str(sample.get("date_str", ""))
            month_value = _coerce_month(sample.get("month", None), date_str)
            month = torch.tensor([month_value], dtype=torch.long, device=device)
            temporal = prompt_encoder._temporal_encoding(month)
            temporal_emb = prompt_encoder.temporal_proj(temporal)

            by_month[month_value].append(input_emb.detach())
            all_input_embs.append(input_emb.detach())
            all_temporal_embs.append(temporal_emb.detach())
            if date_str:
                dates.append(date_str)

        if not all_input_embs:
            raise ValueError("Cannot build target_context prompt state from zero samples")

        r_emb = target_region_embedding
        global_i = torch.stack(all_input_embs, dim=0).mean(dim=0)
        global_t = torch.stack(all_temporal_embs, dim=0).mean(dim=0)
        global_prompt = prompt_encoder.mlp(torch.cat([r_emb, global_i, global_t], dim=1)).squeeze(0).detach().cpu()

        monthly_prototypes: Dict[str, Optional[torch.Tensor]] = {}
        monthly_counts: Dict[str, int] = {}
        for month_value in range(1, 13):
            month_key = str(month_value)
            input_embs = by_month[month_value]
            monthly_counts[month_key] = len(input_embs)
            if not input_embs:
                monthly_prototypes[month_key] = None
                continue
            month_i = torch.stack(input_embs, dim=0).mean(dim=0)
            month_tensor = torch.tensor([month_value], dtype=torch.long, device=device)
            month_t = prompt_encoder.temporal_proj(prompt_encoder._temporal_encoding(month_tensor))
            prompt = prompt_encoder.mlp(torch.cat([r_emb, month_i, month_t], dim=1)).squeeze(0)
            monthly_prototypes[month_key] = prompt.detach().cpu()

    return {
        "schema_version": TARGET_CONTEXT_PROMPT_SCHEMA_VERSION,
        "prompt_source": TARGET_CONTEXT_PROMPT_SOURCE,
        "label_usage": "none",
        "context_hash": context_hash or _hash_context_dates(dates, monthly_counts),
        "context_date_hash": context_hash or _hash_context_dates(dates, monthly_counts),
        "date_start": min(dates) if dates else "",
        "date_end": max(dates) if dates else "",
        "n_samples": int(sum(monthly_counts.values())),
        "monthly_counts": monthly_counts,
        "global_prototype": global_prompt,
        "monthly_prototypes": monthly_prototypes,
        "metadata": {
            "prompt_source": TARGET_CONTEXT_PROMPT_SOURCE,
            "input_usage": "target_context_normalized_input_summary_only",
            "region_usage": "target_region_embedding_or_source_mean_fallback",
            "temporal_usage": "month_of_year_seasonal_phase",
            "label_usage": "none",
            "eval_input_usage": "none_for_prompt_update",
            "eval_month_usage": "known_seasonal_phase_selector_only",
        },
    }


class PromptConditionedBackbonePredictor:
    """Neural predictor wrapping trained FiLMConditionalResUNet + RegionPromptEncoder.

    Loads checkpoint, sets model and prompt encoder to eval mode, and predicts
    with region-conditioned prompt.

    Held-out target unseen-region prompt fallback:
        When target region is not in the source region set, uses the mean of all
        source region embeddings as the target region embedding. This is correct
        because the prompt encoder's num_regions only covers source regions,
        so the target region index would otherwise alias a source region embedding.

    Args:
        checkpoint_path: path to trained .pt checkpoint
        device: device string (default "cuda")
        target_region: target region name (e.g. "US-R1")
        target_region_idx: override target region embedding index (default: from _REGION_TO_IDX)
        apply_residual_gain: apply residual gain alpha from calibration (default True)
    """

    method_name = "prompt_conditioned_shared_backbone"

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        target_region: Optional[str] = None,
        target_region_idx: Optional[int] = None,
        apply_residual_gain: bool = True,
    ) -> None:
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)

        if target_region_idx is None and target_region is not None:
            target_region_idx = _REGION_TO_IDX.get(target_region, 0)
        if target_region_idx is None:
            target_region_idx = 0
        self._target_region_idx = target_region_idx

        # Load checkpoint
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        saved_config = checkpoint.get("config", {})
        source_config = checkpoint.get("source_checkpoint_config", {})

        def cfg_get(name: str, default: Any = None) -> Any:
            if name in saved_config and saved_config[name] is not None:
                return saved_config[name]
            return source_config.get(name, default)

        if bool(cfg_get("enable_pigo", False)):
            raise ValueError(
                "PIGO target-adaptation checkpoints are no longer supported. "
                "Use a non-PIGO spatial-rootzone Phase 5 checkpoint."
            )

        # Init conditional backbone
        width = cfg_get("width", 32)
        prompt_dim = cfg_get("prompt_dim", 64)
        model_type = saved_config.get("model_type", "prompt_conditioned")
        self.model_type = model_type
        is_hyperda = model_type in {"hyperda_basis_adapter", "hyperda_basis_adapter_target_adapt"}
        is_target_adapt = model_type == "hyperda_basis_adapter_target_adapt"
        if is_target_adapt:
            self.method_name = (
                _hyperda_method_id_from_config(saved_config)
                or _hyperda_method_id_from_config(source_config)
                or "hyperda_target_adapt"
            )
        elif is_hyperda:
            self.method_name = "hyperda_basis_adapter_shared"
        else:
            self.method_name = "prompt_conditioned_shared_backbone"
        if is_hyperda:
            self.model = HyperAdapterConditionalResUNet(
                in_channels=12,
                out_channels=2,
                width=width,
                prompt_dim=prompt_dim,
                hyper_n_basis=cfg_get("hyper_n_basis", 8),
                hyper_adapter_bottleneck=cfg_get("hyper_adapter_bottleneck"),
                hyper_adapter_scale=cfg_get("hyper_adapter_scale", 1.0),
                zero_raw_increment_init=cfg_get("zero_raw_increment_init", False),
                enable_target_adaptation=is_target_adapt,
                target_latent_dim=cfg_get("target_latent_dim", 32),
                enable_target_spatial_refine=cfg_get("enable_target_spatial_refine", False),
                target_spatial_refine_hidden=cfg_get("target_spatial_refine_hidden", 16),
                target_spatial_refine_rootzone=cfg_get("target_spatial_refine_rootzone", False),
                target_spatial_refine_input=cfg_get("target_spatial_refine_input", "normalized"),
                target_spatial_refine_type=cfg_get("target_spatial_refine_type", "simple"),
                target_spatial_refine_gain_span=cfg_get("target_spatial_refine_gain_span", 0.25),
                hydro_msr_hidden=cfg_get("hydro_msr_hidden", cfg_get("target_spatial_refine_hidden", 16)),
                enable_hydro_msr_da_film=cfg_get("enable_hydro_msr_da_film", False),
            )
        else:
            self.model = FiLMConditionalResUNet(
                in_channels=12,
                out_channels=2,
                width=width,
                prompt_dim=prompt_dim,
                zero_raw_increment_init=cfg_get("zero_raw_increment_init", False),
            )
        if is_hyperda:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device).eval()
        self._requires_month = is_target_adapt

        # Init RegionPromptEncoder
        num_regions = cfg_get("num_regions", 6)
        self.prompt_encoder = RegionPromptEncoder(
            num_regions=num_regions,
            input_channels=12,
            hidden_dim=prompt_dim,
        )
        if "prompt_encoder_state_dict" in checkpoint:
            self.prompt_encoder.load_state_dict(checkpoint["prompt_encoder_state_dict"])
        self.prompt_encoder.to(device).eval()

        # Held-out target fallback: target region not in source region set.
        # The prompt_encoder was trained with num_regions = len(source_regions).
        # _REGION_TO_IDX maps global region names to indices 0..5, but the
        # prompt encoder's embedding indices correspond only to source regions.
        # We use source_region_global_indices from the checkpoint to determine
        # if the target is unseen. If so, use the mean of all source embeddings
        # as an "unknown target" embedding.
        self._is_target_unseen = False
        self._target_region_emb: Optional[torch.Tensor] = None
        self._source_global_to_prompt_idx: Dict[int, int] = {}
        source_global_indices = cfg_get("source_region_global_indices")
        if source_global_indices is not None:
            self.source_regions = [f"US-R{int(global_idx) + 1}" for global_idx in source_global_indices]
            self._source_global_to_prompt_idx = {
                int(global_idx): prompt_idx
                for prompt_idx, global_idx in enumerate(source_global_indices)
            }
            source_global_set = set(source_global_indices)
            if self._target_region_idx not in source_global_set:
                self._is_target_unseen = True
                with torch.no_grad():
                    all_emb = self.prompt_encoder.region_embed.weight.data.clone()  # [N, 16]
                    self._target_region_emb = all_emb.mean(dim=0)  # [16]
        elif self._target_region_idx >= num_regions:
            self.source_regions = [f"US-R{i + 1}" for i in range(num_regions)]
            # Fallback for old checkpoints without source_region_global_indices
            self._is_target_unseen = True
            with torch.no_grad():
                all_emb = self.prompt_encoder.region_embed.weight.data.clone()
                self._target_region_emb = all_emb.mean(dim=0)
        else:
            self.source_regions = [f"US-R{i + 1}" for i in range(num_regions)]

        # Normalization params
        ch_mean = cfg_get("ch_mean")
        ch_std = cfg_get("ch_std")
        self._ch_mean = np.array(ch_mean, dtype=np.float32) if ch_mean is not None else None
        self._ch_std = np.array(ch_std, dtype=np.float32) if ch_std is not None else None

        # Increment normalization params
        inc_mean = cfg_get("inc_mean")
        inc_std = cfg_get("inc_std")
        self._inc_mean = np.array(inc_mean, dtype=np.float32) if inc_mean is not None else None
        self._inc_std = np.array(inc_std, dtype=np.float32) if inc_std is not None else None
        self._has_inc_norm = self._inc_mean is not None and self._inc_std is not None

        # Residual gain alphas (from source_val calibration)
        self.alpha_surface = float(checkpoint.get("residual_gain_alpha_surface", 1.0))
        self.alpha_rootzone = float(checkpoint.get("residual_gain_alpha_rootzone", 1.0))
        self.apply_residual_gain = apply_residual_gain
        self._prompt_route_uses_target_fallback = False
        self._fixed_target_prompt: Optional[torch.Tensor] = None
        self._target_context_prompt_state: Optional[Dict[str, Any]] = None
        self._target_prompt_metadata: Dict[str, Any] = {}
        state_candidate = checkpoint.get("target_context_prompt_state") or saved_config.get("target_context_prompt_state")
        if state_candidate:
            self.load_target_context_prompt_state(state_candidate)
        elif is_target_adapt and self.method_name in _MAIN_HYPERDA_METHOD_IDS:
            raise ValueError(
                "Paper-facing HyperDA zero/few-shot checkpoints must include "
                "target_context_prompt_state so target_eval inputs cannot update prompts."
            )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise normalization with NaN/Inf guard."""
        if self._ch_mean is None or self._ch_std is None:
            return x
        mean_t = torch.from_numpy(self._ch_mean).to(x.device).view(1, 12, 1, 1)
        std_t = torch.from_numpy(self._ch_std).to(x.device).view(1, 12, 1, 1)
        x_norm = (x - mean_t) / std_t
        if torch.isnan(x_norm).any() or torch.isinf(x_norm).any():
            n_nan = torch.isnan(x_norm).sum().item()
            n_inf = torch.isinf(x_norm).sum().item()
            print(f"  WARNING: normalize produced {n_nan} NaN / {n_inf} Inf — returning raw input", flush=True)
            return x
        return x_norm

    def _build_prompt(self, x_norm: torch.Tensor, region_idx: int, month_val: int) -> torch.Tensor:
        """Build prompt vector z, handling unseen target region.

        When the target region is unseen during training (not in source set),
        uses pre-computed mean of source embeddings instead of aliasing a
        wrong source region embedding.
        """
        region_ids = torch.tensor([region_idx], dtype=torch.long, device=x_norm.device)
        month = torch.tensor([month_val], dtype=torch.long, device=x_norm.device)

        if not (self._is_target_unseen and self._prompt_route_uses_target_fallback):
            # Target is a source region: use standard prompt encoder forward
            return self.prompt_encoder(x_norm, region_ids, month)

        # Target region not in source set: manually assemble prompt
        # using mean of source region embeddings
        input_stats = self.prompt_encoder._compute_input_stats(x_norm)  # [1, C*2]
        i_emb = self.prompt_encoder.input_proj(input_stats)  # [1, 16]
        t_enc = self.prompt_encoder._temporal_encoding(month)  # [1, 2]
        t_emb = self.prompt_encoder.temporal_proj(t_enc)  # [1, 8]

        r_emb = self._target_region_emb.unsqueeze(0).to(x_norm.device)  # [1, 16]

        combined = torch.cat([r_emb, i_emb, t_emb], dim=1)  # [1, 40]
        z = self.prompt_encoder.mlp(combined)  # [1, hidden_dim]
        return z

    @staticmethod
    def _is_source_split(split_role: str) -> bool:
        return split_role in {"source_train", "source_fit", "source_val", "source_test"}

    def _resolve_prompt_region_idx(self, sample: Dict[str, Any]) -> tuple[int, bool]:
        """Return compact prompt id and whether to use held-out target fallback.

        Training uses compact source-region ids (0..Nsource-1). Source split
        evaluation must therefore route by the sample's source region, while
        target splits use the held-out target route.
        """
        split_role = str(sample.get("split_role", ""))
        if self._is_source_split(split_role):
            region_id_str = sample.get("sample_region_id") or sample.get("target_region_id", "")
            global_idx = _REGION_TO_IDX.get(region_id_str, self._target_region_idx)
            if self._source_global_to_prompt_idx:
                if global_idx not in self._source_global_to_prompt_idx:
                    raise ValueError(
                        f"Source split sample_region_id={region_id_str!r} is not in checkpoint "
                        f"source_region_global_indices={sorted(self._source_global_to_prompt_idx)}"
                    )
                return self._source_global_to_prompt_idx[global_idx], False
            return global_idx, False

        region_id_str = sample.get("target_region_id", "")
        return _REGION_TO_IDX.get(region_id_str, self._target_region_idx), True

    @property
    def uses_fixed_target_prompt(self) -> bool:
        return self._target_context_prompt_state is not None or self._fixed_target_prompt is not None

    @property
    def target_context_prompt_state(self) -> Dict[str, Any]:
        if self._target_context_prompt_state is None:
            raise RuntimeError("target_context_prompt_state has not been initialized")
        return normalize_target_context_prompt_state(self._target_context_prompt_state)

    def _target_region_embedding_for_prompt_state(self) -> torch.Tensor:
        if self._is_target_unseen:
            if self._target_region_emb is None:
                raise RuntimeError("Target region fallback embedding was not initialized")
            return self._target_region_emb.unsqueeze(0).to(self.device)
        target_ids = torch.tensor([self._target_region_idx], dtype=torch.long, device=self.device)
        return self.prompt_encoder.region_embed(target_ids)

    def load_target_context_prompt_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        normalized = normalize_target_context_prompt_state(state)
        self._target_context_prompt_state = normalized
        self._fixed_target_prompt = normalized["global_prototype"].unsqueeze(0).to(self.device)
        self._target_prompt_metadata = target_context_prompt_metadata(normalized)
        return dict(self._target_prompt_metadata)

    def compose_target_context_prompt(self, month: int | Sequence[int] | torch.Tensor) -> torch.Tensor:
        if self._target_context_prompt_state is None:
            raise RuntimeError("target_context_prompt_state has not been initialized")
        return compose_target_context_prompt_from_state(self._target_context_prompt_state, month, device=self.device)

    def set_target_context_prompt_from_samples(self, samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Build monthly target-context prompt prototypes from input-side fields only.

        The prompt summary reads only ``x``, ``month``, and ``date_str``. It
        deliberately does not read target analysis or increment labels.
        """
        state = build_target_context_prompt_state(
            samples=samples,
            prompt_encoder=self.prompt_encoder,
            normalize_x=self._normalize,
            target_region_embedding=self._target_region_embedding_for_prompt_state(),
            device=self.device,
        )
        return self.load_target_context_prompt_state(state)

    def set_target_prompt_from_samples(self, samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Legacy alias for older target_train prompt call sites."""
        metadata = self.set_target_context_prompt_from_samples(samples)
        metadata["legacy_alias"] = "set_target_prompt_from_samples"
        self._target_prompt_metadata = dict(metadata)
        return metadata

    def predict(self, sample: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Predict DA increments and analysis for a single sample with prompt conditioning.

        Args:
            sample: dict with keys:
                - x: raw input array [12, H, W]
                - forecast_surface: [H, W]
                - forecast_rootzone: [H, W]
                - target_region_id: str (e.g. "US-R1")
                - month: int (1-12)
                - (optional) date_str, metric_mask, etc.

        Returns:
            dict with pred_increment_*, pred_analysis_*
        """
        x = torch.from_numpy(np.asarray(sample["x"], dtype=np.float32))
        x = x.unsqueeze(0).to(self.device)  # [1, 12, H, W]

        x_norm = self._normalize(x)

        # Build prompt. Source splits route by sample_region_id; target splits
        # route by target_region_id and use held-out target fallback when needed.
        region_idx, use_target_fallback = self._resolve_prompt_region_idx(sample)
        month_val = int(sample.get("month", 6))

        with torch.no_grad():
            if use_target_fallback and self._target_context_prompt_state is not None:
                z = compose_target_context_prompt_from_state(
                    self._target_context_prompt_state,
                    month_val,
                    device=x_norm.device,
                )
            elif use_target_fallback and self._fixed_target_prompt is not None:
                z = self._fixed_target_prompt.to(x_norm.device)
            else:
                try:
                    self._prompt_route_uses_target_fallback = use_target_fallback
                    z = self._build_prompt(x_norm, region_idx, month_val)
                finally:
                    self._prompt_route_uses_target_fallback = False
            month_tensor = torch.tensor([month_val], dtype=torch.long, device=x_norm.device)
            if self._requires_month:
                pred = self.model(x_norm, z, month=month_tensor, x_raw=x)  # [1, 2, H, W]
            else:
                pred = self.model(x_norm, z)  # [1, 2, H, W]

        pred_inc_s = pred[0, 0].cpu().numpy().astype(np.float32)
        pred_inc_r = pred[0, 1].cpu().numpy().astype(np.float32)

        forecast_surface = np.asarray(sample["forecast_surface"], dtype=np.float32)
        forecast_rootzone = np.asarray(sample["forecast_rootzone"], dtype=np.float32)

        # Denormalize increments if needed
        if self._has_inc_norm:
            pred_inc_s = pred_inc_s * self._inc_std[0] + self._inc_mean[0]
            pred_inc_r = pred_inc_r * self._inc_std[1] + self._inc_mean[1]

        # Apply residual gain before returning so public outputs satisfy
        # pred_analysis = forecast + pred_increment.
        if self.apply_residual_gain:
            alpha_s = self.alpha_surface
            alpha_r = self.alpha_rootzone
        else:
            alpha_s = 1.0
            alpha_r = 1.0

        pred_inc_s = (alpha_s * pred_inc_s).astype(np.float32)
        pred_inc_r = (alpha_r * pred_inc_r).astype(np.float32)

        pred_analysis_surface = (forecast_surface + pred_inc_s).astype(np.float32)
        pred_analysis_rootzone = (forecast_rootzone + pred_inc_r).astype(np.float32)

        return {
            "pred_increment_surface": pred_inc_s,
            "pred_increment_rootzone": pred_inc_r,
            "pred_analysis_surface": pred_analysis_surface,
            "pred_analysis_rootzone": pred_analysis_rootzone,
            "residual_gain_alpha_surface": alpha_s,
            "residual_gain_alpha_rootzone": alpha_r,
        }
