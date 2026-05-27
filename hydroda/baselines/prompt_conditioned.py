"""Prompt-conditioned backbone predictor for HydroDA-OOD / HyperDA V4.

No-leakage declaration:
    - Uses trained FiLMConditionalResUNet + RegionPromptEncoder checkpoint
    - Prompt uses input-side features only (x, region_id, month)
    - No target_query labels used in prompt construction
    - Target region uses either source-mean embedding (K=0) or target-specific
      embedding from calibration (K>0)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from hydroda.models.conditional_unet import FiLMConditionalResUNet
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

    K=0 inference:
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

        # Init FiLMConditionalResUNet
        width = saved_config.get("width", 32)
        prompt_dim = saved_config.get("prompt_dim", 64)
        self.model = FiLMConditionalResUNet(
            in_channels=12,
            out_channels=2,
            width=width,
            prompt_dim=prompt_dim,
            zero_raw_increment_init=saved_config.get("zero_raw_increment_init", False),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device).eval()

        # Init RegionPromptEncoder
        num_regions = saved_config.get("num_regions", 6)
        self.prompt_encoder = RegionPromptEncoder(
            num_regions=num_regions,
            input_channels=12,
            hidden_dim=prompt_dim,
        )
        if "prompt_encoder_state_dict" in checkpoint:
            self.prompt_encoder.load_state_dict(checkpoint["prompt_encoder_state_dict"])
        self.prompt_encoder.to(device).eval()

        # BUG FIX: K=0 inference — target region not in source region set.
        # The prompt_encoder was trained with num_regions = len(source_regions).
        # _REGION_TO_IDX maps global region names to indices 0..5, but the
        # prompt encoder's embedding indices correspond only to source regions.
        # We use source_region_global_indices from the checkpoint to determine
        # if the target is unseen. If so, use the mean of all source embeddings
        # as an "unknown target" embedding.
        self._is_target_unseen = False
        self._target_region_emb: Optional[torch.Tensor] = None
        source_global_indices = saved_config.get("source_region_global_indices")
        if source_global_indices is not None:
            source_global_set = set(source_global_indices)
            if self._target_region_idx not in source_global_set:
                self._is_target_unseen = True
                with torch.no_grad():
                    all_emb = self.prompt_encoder.region_embed.weight.data.clone()  # [N, 16]
                    self._target_region_emb = all_emb.mean(dim=0)  # [16]
        elif self._target_region_idx >= num_regions:
            # Fallback for old checkpoints without source_region_global_indices
            self._is_target_unseen = True
            with torch.no_grad():
                all_emb = self.prompt_encoder.region_embed.weight.data.clone()
                self._target_region_emb = all_emb.mean(dim=0)

        # Normalization params
        ch_mean = saved_config.get("ch_mean")
        ch_std = saved_config.get("ch_std")
        self._ch_mean = np.array(ch_mean, dtype=np.float32) if ch_mean is not None else None
        self._ch_std = np.array(ch_std, dtype=np.float32) if ch_std is not None else None

        # Increment normalization params
        inc_mean = saved_config.get("inc_mean")
        inc_std = saved_config.get("inc_std")
        self._inc_mean = np.array(inc_mean, dtype=np.float32) if inc_mean is not None else None
        self._inc_std = np.array(inc_std, dtype=np.float32) if inc_std is not None else None
        self._has_inc_norm = self._inc_mean is not None and self._inc_std is not None

        # Residual gain alphas (from source_val calibration)
        self.alpha_surface = float(checkpoint.get("residual_gain_alpha_surface", 1.0))
        self.alpha_rootzone = float(checkpoint.get("residual_gain_alpha_rootzone", 1.0))
        self.apply_residual_gain = apply_residual_gain

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

        When the target region is unseen during training (K=0, not in source set),
        uses pre-computed mean of source embeddings instead of aliasing a
        wrong source region embedding.
        """
        region_ids = torch.tensor([region_idx], dtype=torch.long, device=x_norm.device)
        month = torch.tensor([month_val], dtype=torch.long, device=x_norm.device)

        if not self._is_target_unseen:
            # Target is a source region: use standard prompt encoder forward
            return self.prompt_encoder(x_norm, region_ids, month)

        # Target region not in source set (K=0): manually assemble prompt
        # using mean of source region embeddings
        input_stats = self.prompt_encoder._compute_input_stats(x_norm)  # [1, C*2]
        i_emb = self.prompt_encoder.input_proj(input_stats)  # [1, 16]
        t_enc = self.prompt_encoder._temporal_encoding(month)  # [1, 2]
        t_emb = self.prompt_encoder.temporal_proj(t_enc)  # [1, 8]

        r_emb = self._target_region_emb.unsqueeze(0).to(x_norm.device)  # [1, 16]

        combined = torch.cat([r_emb, i_emb, t_emb], dim=1)  # [1, 40]
        z = self.prompt_encoder.mlp(combined)  # [1, hidden_dim]
        return z

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

        # Build prompt — handles K=0 target region via _build_prompt
        region_id_str = sample.get("target_region_id", "")
        region_idx = _REGION_TO_IDX.get(region_id_str, self._target_region_idx)
        month_val = int(sample.get("month", 6))

        with torch.no_grad():
            z = self._build_prompt(x_norm, region_idx, month_val)
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
