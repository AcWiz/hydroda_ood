"""Prompt-conditioned backbone predictor for HydroDA-OOD / HyperDA V4.

No-leakage declaration:
    - Uses trained FiLMConditionalResUNet + RegionPromptEncoder checkpoint
    - Prompt uses input-side features only (x, region_id, month)
    - No target_eval/query labels used in prompt construction
    - Held-out target unseen-region prompt fallback is retained for split compatibility
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

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
            self.method_name = "hyperda_target_adapt"
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
        self._target_prompt_metadata: Dict[str, Any] = {}

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
        return self._fixed_target_prompt is not None

    def set_target_prompt_from_samples(self, samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a fixed target prompt from target_train input-side fields only.

        The prompt summary reads only ``x``, ``month``, and ``date_str``. It
        deliberately does not read target analysis or increment labels.
        """
        input_embs = []
        temporal_embs = []
        dates = []

        with torch.no_grad():
            for sample in samples:
                x = torch.from_numpy(np.asarray(sample["x"], dtype=np.float32))
                x = x.unsqueeze(0).to(self.device)
                x_norm = self._normalize(x)
                input_stats = self.prompt_encoder._compute_input_stats(x_norm)
                input_embs.append(self.prompt_encoder.input_proj(input_stats))

                month = torch.tensor([int(sample.get("month", 6))], dtype=torch.long, device=self.device)
                temporal = self.prompt_encoder._temporal_encoding(month)
                temporal_embs.append(self.prompt_encoder.temporal_proj(temporal))

                date_str = sample.get("date_str", "")
                if date_str:
                    dates.append(str(date_str))

            if not input_embs:
                raise ValueError("Cannot build target prompt from zero target_train samples")

            i_emb = torch.stack(input_embs, dim=0).mean(dim=0)
            t_emb = torch.stack(temporal_embs, dim=0).mean(dim=0)
            if self._is_target_unseen:
                if self._target_region_emb is None:
                    raise RuntimeError("Target region fallback embedding was not initialized")
                r_emb = self._target_region_emb.unsqueeze(0).to(self.device)
            else:
                target_ids = torch.tensor([self._target_region_idx], dtype=torch.long, device=self.device)
                r_emb = self.prompt_encoder.region_embed(target_ids)

            combined = torch.cat([r_emb, i_emb, t_emb], dim=1)
            self._fixed_target_prompt = self.prompt_encoder.mlp(combined).detach()

        self._target_prompt_metadata = {
            "prompt_source": "target_train_input_side_summary",
            "n_samples": len(input_embs),
            "date_start": min(dates) if dates else "",
            "date_end": max(dates) if dates else "",
            "label_usage": "none",
        }
        return dict(self._target_prompt_metadata)

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
            if use_target_fallback and self._fixed_target_prompt is not None:
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

        # Apply residual gain calibration if available
        if self.apply_residual_gain:
            alpha_s = self.alpha_surface
            alpha_r = self.alpha_rootzone
        else:
            alpha_s = 1.0
            alpha_r = 1.0

        pred_analysis_surface = (forecast_surface + alpha_s * pred_inc_s).astype(np.float32)
        pred_analysis_rootzone = (forecast_rootzone + alpha_r * pred_inc_r).astype(np.float32)

        return {
            "pred_increment_surface": pred_inc_s,
            "pred_increment_rootzone": pred_inc_r,
            "pred_analysis_surface": pred_analysis_surface,
            "pred_analysis_rootzone": pred_analysis_rootzone,
            "residual_gain_alpha_surface": alpha_s,
            "residual_gain_alpha_rootzone": alpha_r,
        }
