"""HyperDA v2 FiLM plus basis-adapter conditional UNet."""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from hydroda.models.hyper_adapters import BasisHyperAdapter
from hydroda.models.conditional_unet import FiLMLayer
from hydroda.models.resunet import ConvBlock
from hydroda.models.target_adaptation import (
    AdapterCoefficientResidual,
    HydroMSRGainOutputAdapter,
    HydroMSRGainLiteOutputAdapter,
    HydroMSROutputAdapter,
    HydroMSRROSEOutputAdapter,
    MonthlyResidualGain,
    TargetLatentPrompt,
    TargetSpatialResidualHead,
)


_ADAPTER_LAYER_NAMES = ("bottleneck", "dec2", "dec1")
_ADAPTER_LAYER_TO_INDEX = {name: idx for idx, name in enumerate(_ADAPTER_LAYER_NAMES)}
SOURCE_RESIDUAL_RELIABILITY_FEATURE_SCHEMA = [
    "monthly_count",
    "has_monthly_prototype",
    "global_context_count",
    "finite_input_coverage",
    "prompt_to_source_manifold_distance",
]
SOURCE_SALIENCY_PRIOR_APPLICATIONS = (
    "soft_regularization_metadata",
    "legacy_gate_logit_bias_before_topk",
)


class SharedLayerAwareCoefficientGenerator(nn.Module):
    """Shared prompt-to-basis-logit generator conditioned on adapter layer id."""

    def __init__(self, prompt_dim: int, n_basis: int, n_layers: int = 3) -> None:
        super().__init__()
        self.prompt_dim = int(prompt_dim)
        self.n_basis = int(n_basis)
        self.n_layers = int(n_layers)
        self.layer_embedding = nn.Embedding(self.n_layers, self.prompt_dim)
        self.coeff_head = nn.Linear(self.prompt_dim * 2, self.n_basis)

    def forward(self, z: torch.Tensor, layer_index: int | torch.Tensor) -> torch.Tensor:
        if isinstance(layer_index, int):
            layer_ids = torch.full(
                (z.shape[0],),
                int(layer_index),
                dtype=torch.long,
                device=z.device,
            )
        else:
            layer_ids = layer_index.to(device=z.device, dtype=torch.long)
            if layer_ids.ndim == 0:
                layer_ids = layer_ids.expand(z.shape[0])
            if layer_ids.shape != (z.shape[0],):
                raise ValueError("layer_index tensor must be scalar or shape [B]")
        layer_z = self.layer_embedding(layer_ids)
        return self.coeff_head(torch.cat([z, layer_z], dim=-1))


class RankGatedLayerAwareCoefficientGenerator(nn.Module):
    """Layer-aware basis-logit generator with a bounded top-k adapter budget."""

    def __init__(
        self,
        prompt_dim: int,
        n_basis: int,
        n_layers: int = 3,
        top_k: int = 4,
        temperature_init: float = 1.0,
    ) -> None:
        super().__init__()
        if n_basis < 1:
            raise ValueError("n_basis must be >= 1")
        if top_k < 1 or top_k > n_basis:
            raise ValueError("hyper_rank_gate_top_k must be in [1, n_basis]")
        if temperature_init <= 0.0:
            raise ValueError("hyper_rank_gate_temperature_init must be positive")
        self.prompt_dim = int(prompt_dim)
        self.n_basis = int(n_basis)
        self.n_layers = int(n_layers)
        self.top_k = int(top_k)
        self.temperature_init = float(temperature_init)
        self.layer_embedding = nn.Embedding(self.n_layers, self.prompt_dim)
        self.coeff_head = nn.Linear(self.prompt_dim * 2, self.n_basis)
        self.gate_head = nn.Linear(self.prompt_dim * 2, self.n_basis)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(self.temperature_init)))

    def _layer_ids(self, z: torch.Tensor, layer_index: int | torch.Tensor) -> torch.Tensor:
        if isinstance(layer_index, int):
            return torch.full(
                (z.shape[0],),
                int(layer_index),
                dtype=torch.long,
                device=z.device,
            )
        layer_ids = layer_index.to(device=z.device, dtype=torch.long)
        if layer_ids.ndim == 0:
            layer_ids = layer_ids.expand(z.shape[0])
        if layer_ids.shape != (z.shape[0],):
            raise ValueError("layer_index tensor must be scalar or shape [B]")
        return layer_ids

    def forward(self, z: torch.Tensor, layer_index: int | torch.Tensor) -> torch.Tensor:
        layer_ids = self._layer_ids(z, layer_index)
        h = torch.cat([z, self.layer_embedding(layer_ids)], dim=-1)
        logits = self.coeff_head(h)
        gate_logits = self.gate_head(h)
        logits = logits + gate_logits
        if self.top_k < self.n_basis:
            _, indices = torch.topk(gate_logits, k=self.top_k, dim=-1)
            mask = torch.zeros_like(gate_logits, dtype=torch.bool)
            mask.scatter_(1, indices, True)
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        temperature = self.log_temperature.exp().clamp_min(1e-6).to(dtype=logits.dtype)
        return logits / temperature


class StableRankGatedLayerAwareCoefficientGenerator(nn.Module):
    """Numerically stable layer-aware top-k basis-logit generator.

    This keeps the old rank-gated generator available as failed ablation
    evidence, while giving M2.1 a finite mask floor and bounded gate modulation.
    """

    def __init__(
        self,
        prompt_dim: int,
        n_basis: int,
        n_layers: int = 3,
        top_k: int = 4,
        temperature_init: float = 2.0,
        gate_logit_span: float = 2.0,
        mask_floor: float = -30.0,
        saliency_prior: torch.Tensor | None = None,
        saliency_prior_beta: float = 0.0,
        saliency_prior_application: str = "soft_regularization_metadata",
    ) -> None:
        super().__init__()
        if n_basis < 1:
            raise ValueError("n_basis must be >= 1")
        if top_k < 1 or top_k > n_basis:
            raise ValueError("hyper_rank_gate_top_k must be in [1, n_basis]")
        if temperature_init <= 0.0:
            raise ValueError("hyper_rank_gate_temperature_init must be positive")
        if gate_logit_span <= 0.0:
            raise ValueError("stable rank-gated gate_logit_span must be positive")
        if mask_floor >= 0.0:
            raise ValueError("stable rank-gated mask_floor must be negative")
        if saliency_prior_beta < 0.0:
            raise ValueError("hyper_source_saliency_prior_beta must be non-negative")
        if saliency_prior_application not in SOURCE_SALIENCY_PRIOR_APPLICATIONS:
            raise ValueError(
                "hyper_source_saliency_prior_application must be one of "
                f"{SOURCE_SALIENCY_PRIOR_APPLICATIONS}"
            )
        self.prompt_dim = int(prompt_dim)
        self.n_basis = int(n_basis)
        self.n_layers = int(n_layers)
        self.top_k = int(top_k)
        self.temperature_init = float(temperature_init)
        self.gate_logit_span = float(gate_logit_span)
        self.mask_floor = float(mask_floor)
        self.saliency_prior_beta = float(saliency_prior_beta)
        self.saliency_prior_application = str(saliency_prior_application)
        self.layer_embedding = nn.Embedding(self.n_layers, self.prompt_dim)
        self.coeff_head = nn.Linear(self.prompt_dim * 2, self.n_basis)
        self.gate_head = nn.Linear(self.prompt_dim * 2, self.n_basis)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(self.temperature_init)))
        self.register_buffer(
            "saliency_prior",
            self._coerce_saliency_prior(saliency_prior),
            persistent=False,
        )
        self._reset_stable_parameters()

    def _coerce_saliency_prior(self, prior: torch.Tensor | None) -> torch.Tensor:
        if prior is None:
            return torch.zeros(self.n_layers, self.n_basis, dtype=torch.float32)
        tensor = torch.as_tensor(prior, dtype=torch.float32).detach().clone()
        if tensor.shape != (self.n_layers, self.n_basis):
            raise ValueError(
                "source saliency prior must have shape "
                f"[{self.n_layers}, {self.n_basis}], got {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError("source saliency prior must be finite")
        return tensor

    def set_saliency_prior(
        self,
        prior: torch.Tensor | None,
        *,
        beta: float | None = None,
    ) -> None:
        if beta is not None:
            if float(beta) < 0.0:
                raise ValueError("hyper_source_saliency_prior_beta must be non-negative")
            self.saliency_prior_beta = float(beta)
        self.saliency_prior = self._coerce_saliency_prior(prior).to(
            device=self.saliency_prior.device,
            dtype=self.saliency_prior.dtype,
        )

    def _reset_stable_parameters(self) -> None:
        nn.init.normal_(self.layer_embedding.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.coeff_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.coeff_head.bias)
        nn.init.normal_(self.gate_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.gate_head.bias)

    def _layer_ids(self, z: torch.Tensor, layer_index: int | torch.Tensor) -> torch.Tensor:
        if isinstance(layer_index, int):
            return torch.full(
                (z.shape[0],),
                int(layer_index),
                dtype=torch.long,
                device=z.device,
            )
        layer_ids = layer_index.to(device=z.device, dtype=torch.long)
        if layer_ids.ndim == 0:
            layer_ids = layer_ids.expand(z.shape[0])
        if layer_ids.shape != (z.shape[0],):
            raise ValueError("layer_index tensor must be scalar or shape [B]")
        return layer_ids

    def forward(self, z: torch.Tensor, layer_index: int | torch.Tensor) -> torch.Tensor:
        layer_ids = self._layer_ids(z, layer_index)
        h = torch.cat([z, self.layer_embedding(layer_ids)], dim=-1)
        coeff_logits = self.coeff_head(h)
        gate_logits = self.gate_head(h)
        if (
            self.saliency_prior_beta > 0.0
            and self.saliency_prior_application == "legacy_gate_logit_bias_before_topk"
        ):
            prior = self.saliency_prior.to(device=gate_logits.device, dtype=gate_logits.dtype)
            gate_logits = gate_logits + float(self.saliency_prior_beta) * prior.index_select(0, layer_ids)
        bounded_gate = self.gate_logit_span * torch.tanh(gate_logits)
        temperature = self.log_temperature.exp().clamp_min(1e-3).to(dtype=coeff_logits.dtype)
        logits = (coeff_logits + bounded_gate) / temperature
        if self.top_k < self.n_basis:
            _, indices = torch.topk(gate_logits, k=self.top_k, dim=-1)
            mask = torch.zeros_like(gate_logits, dtype=torch.bool)
            mask.scatter_(1, indices, True)
            floor = torch.as_tensor(self.mask_floor, dtype=logits.dtype, device=logits.device)
            logits = torch.where(mask, logits, floor)
        return logits


class PromptScalarReliabilityGate(nn.Module):
    """Bounded scalar residual gate from prompt and adapter-layer embedding."""

    def __init__(self, prompt_dim: int, init_value: float = 0.95, n_layers: int = 3) -> None:
        super().__init__()
        if not 0.0 < float(init_value) < 1.0:
            raise ValueError("hyper_reliability_init must be between 0 and 1")
        self.prompt_dim = int(prompt_dim)
        self.n_layers = int(n_layers)
        self.init_value = float(init_value)
        self.layer_embedding = nn.Embedding(self.n_layers, self.prompt_dim)
        self.gate_head = nn.Linear(self.prompt_dim * 2, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, math.log(self.init_value / (1.0 - self.init_value)))

    def forward(self, z: torch.Tensor, layer_index: int | torch.Tensor) -> torch.Tensor:
        if isinstance(layer_index, int):
            layer_ids = torch.full(
                (z.shape[0],),
                int(layer_index),
                dtype=torch.long,
                device=z.device,
            )
        else:
            layer_ids = layer_index.to(device=z.device, dtype=torch.long)
            if layer_ids.ndim == 0:
                layer_ids = layer_ids.expand(z.shape[0])
            if layer_ids.shape != (z.shape[0],):
                raise ValueError("layer_index tensor must be scalar or shape [B]")
        layer_z = self.layer_embedding(layer_ids)
        return torch.sigmoid(self.gate_head(torch.cat([z, layer_z], dim=-1)))


class SourceResidualReliabilityGate(nn.Module):
    """Bounded scalar gate for source-base residual prior updates."""

    def __init__(
        self,
        prompt_dim: int,
        reliability_dim: int,
        init_value: float = 0.95,
    ) -> None:
        super().__init__()
        if reliability_dim < 1:
            raise ValueError("source_residual_reliability_dim must be >= 1")
        if not 0.0 < float(init_value) < 1.0:
            raise ValueError("source_residual_gate_init must be between 0 and 1")
        self.prompt_dim = int(prompt_dim)
        self.reliability_dim = int(reliability_dim)
        self.init_value = float(init_value)
        self.gate_head = nn.Linear(self.prompt_dim + self.reliability_dim, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, math.log(self.init_value / (1.0 - self.init_value)))

    def forward(self, z: torch.Tensor, reliability_features: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[-1] != self.prompt_dim:
            raise ValueError(f"z must have shape [B, {self.prompt_dim}]")
        if reliability_features.ndim != 2 or reliability_features.shape != (z.shape[0], self.reliability_dim):
            raise ValueError(
                "reliability_features must have shape "
                f"[B, {self.reliability_dim}]"
            )
        features = reliability_features.to(device=z.device, dtype=z.dtype)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.sigmoid(self.gate_head(torch.cat([z, features], dim=-1)))


class HyperAdapterConditionalResUNet(nn.Module):
    """SmallResUNet with FiLM modulation and prompt-mixture adapter residuals."""

    model_type = "hyperda_basis_adapter"

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 2,
        width: int = 32,
        prompt_dim: int = 64,
        hyper_n_basis: int = 8,
        hyper_adapter_bottleneck: Optional[int] = None,
        hyper_adapter_scale: float = 1.0,
        hyper_coeff_generator: str = "per_adapter",
        hyper_rank_gate_top_k: int = 4,
        hyper_rank_gate_temperature_init: float = 1.0,
        hyper_adapter_param_style: str = "basis_1x1",
        hyper_reliability_gate: str = "none",
        hyper_reliability_init: float = 0.95,
        hyper_source_saliency_prior: torch.Tensor | None = None,
        hyper_source_saliency_prior_beta: float = 0.0,
        hyper_source_saliency_prior_path: str = "",
        hyper_source_saliency_prior_application: str = "soft_regularization_metadata",
        hyper_prompt_manifold_reliability: bool = False,
        hyper_prompt_manifold_reliability_strength: float = 0.0,
        hyper_enable_film: bool = True,
        hyper_enable_adapters: bool = True,
        zero_shot_prior_form: str = "direct_hyper",
        source_residual_rho: float = 1.0,
        source_residual_gate: str = "prompt_reliability_scalar",
        source_residual_gate_init: float = 0.95,
        source_residual_reliability_dim: int = len(SOURCE_RESIDUAL_RELIABILITY_FEATURE_SCHEMA),
        zero_raw_increment_init: bool = False,
        enable_target_adaptation: bool = False,
        target_latent_dim: int = 32,
        enable_target_spatial_refine: bool = False,
        target_spatial_refine_hidden: int = 16,
        target_spatial_refine_rootzone: bool = False,
        target_spatial_refine_input: str = "normalized",
        target_spatial_refine_type: str = "simple",
        target_spatial_refine_gain_span: float = 0.25,
        hydro_msr_hidden: int | None = None,
        enable_hydro_msr_da_film: bool = False,
    ) -> None:
        super().__init__()
        self.prompt_dim = prompt_dim
        self.hyper_n_basis = int(hyper_n_basis)
        if hyper_coeff_generator not in {
            "per_adapter",
            "shared_layer_aware",
            "shared_layer_aware_rank_gated",
            "shared_layer_aware_rank_gated_stable",
        }:
            raise ValueError(
                "hyper_coeff_generator must be 'per_adapter', 'shared_layer_aware', "
                "'shared_layer_aware_rank_gated', or "
                "'shared_layer_aware_rank_gated_stable'"
            )
        if hyper_adapter_param_style not in {"basis_1x1", "dora_like_gain", "dora_like_gain_bounded"}:
            raise ValueError(
                "hyper_adapter_param_style must be 'basis_1x1', 'dora_like_gain', "
                "or 'dora_like_gain_bounded'"
            )
        if hyper_reliability_gate not in {"none", "prompt_scalar"}:
            raise ValueError("hyper_reliability_gate must be 'none' or 'prompt_scalar'")
        if float(hyper_source_saliency_prior_beta) < 0.0:
            raise ValueError("hyper_source_saliency_prior_beta must be non-negative")
        if hyper_source_saliency_prior_application not in SOURCE_SALIENCY_PRIOR_APPLICATIONS:
            raise ValueError(
                "hyper_source_saliency_prior_application must be one of "
                f"{SOURCE_SALIENCY_PRIOR_APPLICATIONS}"
            )
        if float(hyper_prompt_manifold_reliability_strength) < 0.0:
            raise ValueError("hyper_prompt_manifold_reliability_strength must be non-negative")
        self.hyper_coeff_generator = str(hyper_coeff_generator)
        self.hyper_rank_gate_top_k = int(hyper_rank_gate_top_k)
        self.hyper_rank_gate_temperature_init = float(hyper_rank_gate_temperature_init)
        self.hyper_adapter_param_style = str(hyper_adapter_param_style)
        self.hyper_reliability_gate = str(hyper_reliability_gate)
        self.hyper_reliability_init = float(hyper_reliability_init)
        self.hyper_source_saliency_prior_beta = float(hyper_source_saliency_prior_beta)
        self.hyper_source_saliency_prior_path = str(hyper_source_saliency_prior_path or "")
        self.hyper_source_saliency_prior_application = str(hyper_source_saliency_prior_application)
        self.hyper_prompt_manifold_reliability = bool(hyper_prompt_manifold_reliability)
        self.hyper_prompt_manifold_reliability_strength = float(hyper_prompt_manifold_reliability_strength)
        self.hyper_enable_film = bool(hyper_enable_film)
        self.hyper_enable_adapters = bool(hyper_enable_adapters)
        if zero_shot_prior_form not in {
            "direct_hyper",
            "source_residual_prior",
            "source_base_residual_reliability_gated",
        }:
            raise ValueError(
                "zero_shot_prior_form must be 'direct_hyper', 'source_residual_prior', "
                "or 'source_base_residual_reliability_gated'"
            )
        if source_residual_gate not in {"none", "prompt_reliability_scalar"}:
            raise ValueError("source_residual_gate must be 'none' or 'prompt_reliability_scalar'")
        if not 0.0 <= float(source_residual_rho) <= 1.0:
            raise ValueError("source_residual_rho must be in [0, 1]")
        self.zero_shot_prior_form = str(zero_shot_prior_form)
        self.uses_source_residual_prior = self.zero_shot_prior_form != "direct_hyper"
        self.source_residual_rho = float(source_residual_rho)
        self.source_residual_gate = str(source_residual_gate)
        self.source_residual_gate_init = float(source_residual_gate_init)
        self.source_residual_reliability_dim = int(source_residual_reliability_dim)
        self.reliability_feature_schema = list(SOURCE_RESIDUAL_RELIABILITY_FEATURE_SCHEMA)
        self.enable_target_adaptation = bool(enable_target_adaptation)
        self.hyper_adapter_bottleneck = (
            int(hyper_adapter_bottleneck)
            if hyper_adapter_bottleneck is not None
            else max(8, (width * 4) // 4)
        )
        self.hyper_adapter_scale = float(hyper_adapter_scale)
        self.enable_target_spatial_refine = bool(enable_target_spatial_refine)
        self.target_spatial_refine_hidden = int(target_spatial_refine_hidden)
        self.target_spatial_refine_rootzone = bool(target_spatial_refine_rootzone)
        if target_spatial_refine_input not in {"normalized", "raw"}:
            raise ValueError("target_spatial_refine_input must be 'normalized' or 'raw'")
        self.target_spatial_refine_input = str(target_spatial_refine_input)
        if target_spatial_refine_type not in {"simple", "hydro_msr", "hydro_msr_gain", "hydro_msr_gain_lite", "hydro_msr_rose"}:
            raise ValueError(
                "target_spatial_refine_type must be 'simple', 'hydro_msr', "
                "'hydro_msr_gain', 'hydro_msr_gain_lite', or 'hydro_msr_rose'"
            )
        self.target_spatial_refine_type = str(target_spatial_refine_type)
        self.target_spatial_refine_gain_span = float(target_spatial_refine_gain_span)
        self.hydro_msr_hidden = int(hydro_msr_hidden) if hydro_msr_hidden is not None else self.target_spatial_refine_hidden
        self.enable_hydro_msr_da_film = bool(enable_hydro_msr_da_film)

        self.enc1 = ConvBlock(in_channels, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.enc3 = ConvBlock(width * 2, width * 4)
        self.bottleneck = ConvBlock(width * 4, width * 4)
        self.film1 = FiLMLayer(width, prompt_dim)
        self.film2 = FiLMLayer(width * 2, prompt_dim)
        self.film3 = FiLMLayer(width * 4, prompt_dim)
        self.film_b = FiLMLayer(width * 4, prompt_dim)
        if self.hyper_coeff_generator == "shared_layer_aware_rank_gated":
            self.shared_coeff_generator = RankGatedLayerAwareCoefficientGenerator(
                prompt_dim=prompt_dim,
                n_basis=self.hyper_n_basis,
                n_layers=len(_ADAPTER_LAYER_NAMES),
                top_k=self.hyper_rank_gate_top_k,
                temperature_init=self.hyper_rank_gate_temperature_init,
            )
        elif self.hyper_coeff_generator == "shared_layer_aware_rank_gated_stable":
            self.shared_coeff_generator = StableRankGatedLayerAwareCoefficientGenerator(
                prompt_dim=prompt_dim,
                n_basis=self.hyper_n_basis,
                n_layers=len(_ADAPTER_LAYER_NAMES),
                top_k=self.hyper_rank_gate_top_k,
                temperature_init=self.hyper_rank_gate_temperature_init,
                saliency_prior=hyper_source_saliency_prior,
                saliency_prior_beta=self.hyper_source_saliency_prior_beta,
                saliency_prior_application=self.hyper_source_saliency_prior_application,
            )
        elif self.hyper_coeff_generator == "shared_layer_aware":
            self.shared_coeff_generator = SharedLayerAwareCoefficientGenerator(
                prompt_dim=prompt_dim,
                n_basis=self.hyper_n_basis,
                n_layers=len(_ADAPTER_LAYER_NAMES),
            )
        else:
            self.shared_coeff_generator = None
        self.reliability_gate = (
            PromptScalarReliabilityGate(
                prompt_dim=prompt_dim,
                init_value=self.hyper_reliability_init,
                n_layers=len(_ADAPTER_LAYER_NAMES),
            )
            if self.hyper_reliability_gate == "prompt_scalar"
            else None
        )
        self.hyper_adapter_b = BasisHyperAdapter(
            channels=width * 4,
            prompt_dim=prompt_dim,
            n_basis=self.hyper_n_basis,
            adapter_bottleneck=self.hyper_adapter_bottleneck,
            adapter_scale=self.hyper_adapter_scale,
            adapter_param_style=self.hyper_adapter_param_style,
        )
        self.dec2 = ConvBlock(width * 6, width * 2)
        self.dec1 = ConvBlock(width * 3, width)
        self.hyper_adapter_d2 = BasisHyperAdapter(
            channels=width * 2,
            prompt_dim=prompt_dim,
            n_basis=self.hyper_n_basis,
            adapter_bottleneck=max(4, self.hyper_adapter_bottleneck // 2),
            adapter_scale=self.hyper_adapter_scale,
            adapter_param_style=self.hyper_adapter_param_style,
        )
        self.hyper_adapter_d1 = BasisHyperAdapter(
            channels=width,
            prompt_dim=prompt_dim,
            n_basis=self.hyper_n_basis,
            adapter_bottleneck=max(4, self.hyper_adapter_bottleneck // 4),
            adapter_scale=self.hyper_adapter_scale,
            adapter_param_style=self.hyper_adapter_param_style,
        )
        self.hyper_adapter = self.hyper_adapter_b
        self.head = nn.Conv2d(width, out_channels, 1)
        self.residual_head = nn.Conv2d(width, out_channels, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.source_residual_gate_net = (
            SourceResidualReliabilityGate(
                prompt_dim=prompt_dim,
                reliability_dim=self.source_residual_reliability_dim,
                init_value=self.source_residual_gate_init,
            )
            if self.uses_source_residual_prior
            and self.source_residual_gate == "prompt_reliability_scalar"
            else None
        )
        if self.enable_target_adaptation:
            self.target_prompt = TargetLatentPrompt(prompt_dim=prompt_dim, latent_dim=target_latent_dim)
            self.target_adapter_coefficient_residual_b = AdapterCoefficientResidual(self.hyper_n_basis)
            self.target_adapter_coefficient_residual_d2 = AdapterCoefficientResidual(self.hyper_n_basis)
            self.target_adapter_coefficient_residual_d1 = AdapterCoefficientResidual(self.hyper_n_basis)
            self.residual_gain = MonthlyResidualGain(out_channels=out_channels)
            if self.enable_target_spatial_refine and self.target_spatial_refine_type == "hydro_msr_gain":
                self.target_spatial_refine = HydroMSRGainOutputAdapter(
                    input_channels=in_channels,
                    out_channels=out_channels,
                    hidden_channels=self.hydro_msr_hidden,
                    refine_rootzone=self.target_spatial_refine_rootzone,
                    enable_da_film=self.enable_hydro_msr_da_film,
                )
            elif self.enable_target_spatial_refine and self.target_spatial_refine_type == "hydro_msr_gain_lite":
                self.target_spatial_refine = HydroMSRGainLiteOutputAdapter(
                    input_channels=in_channels,
                    out_channels=out_channels,
                    hidden_channels=self.hydro_msr_hidden,
                    refine_rootzone=self.target_spatial_refine_rootzone,
                    enable_da_film=self.enable_hydro_msr_da_film,
                    gain_span=self.target_spatial_refine_gain_span,
                    learn_rootzone_gain=False,
                )
            elif self.enable_target_spatial_refine and self.target_spatial_refine_type == "hydro_msr":
                self.target_spatial_refine = HydroMSROutputAdapter(
                    input_channels=in_channels,
                    out_channels=out_channels,
                    hidden_channels=self.hydro_msr_hidden,
                    refine_rootzone=self.target_spatial_refine_rootzone,
                    enable_da_film=self.enable_hydro_msr_da_film,
                )
            elif self.enable_target_spatial_refine and self.target_spatial_refine_type == "hydro_msr_rose":
                self.target_spatial_refine = HydroMSRROSEOutputAdapter(
                    input_channels=in_channels,
                    out_channels=out_channels,
                    hidden_channels=self.hydro_msr_hidden,
                    refine_rootzone=self.target_spatial_refine_rootzone,
                    enable_da_film=self.enable_hydro_msr_da_film,
                )
            elif self.enable_target_spatial_refine:
                self.target_spatial_refine = TargetSpatialResidualHead(
                    input_channels=in_channels,
                    out_channels=out_channels,
                    hidden_channels=self.target_spatial_refine_hidden,
                    refine_rootzone=self.target_spatial_refine_rootzone,
                )
            else:
                self.target_spatial_refine = None
        else:
            self.target_prompt = None
            self.target_adapter_coefficient_residual_b = None
            self.target_adapter_coefficient_residual_d2 = None
            self.target_adapter_coefficient_residual_d1 = None
            self.residual_gain = None
            self.target_spatial_refine = None

        self._zero_raw_increment_init = zero_raw_increment_init
        if zero_raw_increment_init:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def _layer_index(self, layer_name: str) -> int:
        if layer_name not in _ADAPTER_LAYER_TO_INDEX:
            raise ValueError(f"unknown HyperDA adapter layer: {layer_name}")
        return _ADAPTER_LAYER_TO_INDEX[layer_name]

    def adapter_coefficient_logits(self, z: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Return basis logits for an adapter layer under the configured generator."""
        if self.hyper_coeff_generator in {
            "shared_layer_aware",
            "shared_layer_aware_rank_gated",
            "shared_layer_aware_rank_gated_stable",
        }:
            if self.shared_coeff_generator is None:
                raise RuntimeError("shared coefficient generator is not initialized")
            return self.shared_coeff_generator(z, self._layer_index(layer_name))
        adapter = self._adapter_module(layer_name)
        return adapter.coefficient_logits(z)

    def adapter_reliability_gate(self, z: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Return bounded adapter residual gate [B, 1] for a layer."""
        if self.hyper_reliability_gate == "none":
            return torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)
        if self.reliability_gate is None:
            raise RuntimeError("prompt scalar reliability gate is not initialized")
        return self.reliability_gate(z, self._layer_index(layer_name))

    def prompt_manifold_reliability_multiplier(
        self,
        z: torch.Tensor,
        reliability_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return conservative adapter multiplier from input-side prompt reliability."""
        if not self.hyper_prompt_manifold_reliability:
            return torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)
        if reliability_features is None:
            return torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)
        if reliability_features.ndim != 2 or reliability_features.shape[0] != z.shape[0]:
            raise ValueError("reliability_features must have shape [B, D]")
        if reliability_features.shape[1] < len(SOURCE_RESIDUAL_RELIABILITY_FEATURE_SCHEMA):
            raise ValueError(
                "reliability_features does not include prompt_to_source_manifold_distance"
            )
        features = reliability_features.to(device=z.device, dtype=z.dtype)
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        distance_index = SOURCE_RESIDUAL_RELIABILITY_FEATURE_SCHEMA.index(
            "prompt_to_source_manifold_distance"
        )
        bounded_distance = features[:, distance_index : distance_index + 1].clamp(0.0, 1.0)
        return (1.0 - self.hyper_prompt_manifold_reliability_strength * bounded_distance).clamp(0.0, 1.0)

    def _adapter_module(self, layer_name: str) -> BasisHyperAdapter:
        if layer_name == "bottleneck":
            return self.hyper_adapter_b
        if layer_name == "dec2":
            return self.hyper_adapter_d2
        if layer_name == "dec1":
            return self.hyper_adapter_d1
        raise ValueError(f"unknown HyperDA adapter layer: {layer_name}")

    def source_stage_trainable_modules(self) -> list[tuple[str, nn.Module]]:
        """Return enabled source-stage modules for staged HyperDA optimization."""
        modules: list[tuple[str, nn.Module]] = []
        if self.hyper_enable_film:
            modules.extend(
                [
                    ("film1", self.film1),
                    ("film2", self.film2),
                    ("film3", self.film3),
                    ("film_b", self.film_b),
                ]
            )
        if self.hyper_enable_adapters:
            modules.extend(
                [
                    ("hyper_adapter_b", self.hyper_adapter_b),
                    ("hyper_adapter_d2", self.hyper_adapter_d2),
                    ("hyper_adapter_d1", self.hyper_adapter_d1),
                ]
            )
            if self.shared_coeff_generator is not None:
                modules.append(("shared_coeff_generator", self.shared_coeff_generator))
            if self.reliability_gate is not None:
                modules.append(("reliability_gate", self.reliability_gate))
        if self.uses_source_residual_prior:
            modules.append(("residual_head", self.residual_head))
            if self.source_residual_gate_net is not None:
                modules.append(("source_residual_gate_net", self.source_residual_gate_net))
        return modules

    def _target_residual(self, name: str) -> torch.Tensor | None:
        module = getattr(self, name)
        return module() if module is not None else None

    def freeze_source_prior_for_target_adaptation(self) -> None:
        """Freeze source prior parameters and leave only target adaptation trainable."""
        if not self.enable_target_adaptation:
            raise ValueError("target adaptation modules are not enabled")
        for param in self.parameters():
            param.requires_grad_(False)
        for module in [
            self.target_prompt,
            self.target_adapter_coefficient_residual_b,
            self.target_adapter_coefficient_residual_d2,
            self.target_adapter_coefficient_residual_d1,
            self.residual_gain,
            self.target_spatial_refine,
        ]:
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad_(True)

    def target_trainable_parameter_names(self) -> list[str]:
        """Return trainable parameter names after target adaptation freezing."""
        return [name for name, param in self.named_parameters() if param.requires_grad]

    def source_base_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the frozen source-only enc/dec/head path."""
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        b = self.bottleneck(e3)
        d2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)

    def source_residual_reliability_gate(
        self,
        z: torch.Tensor,
        reliability_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return bounded source-residual gate with shape [B, 1]."""
        if self.source_residual_gate == "none":
            return torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)
        if self.source_residual_gate_net is None:
            raise RuntimeError("source residual reliability gate is not initialized")
        if reliability_features is None:
            reliability_features = torch.zeros(
                z.shape[0],
                self.source_residual_reliability_dim,
                dtype=z.dtype,
                device=z.device,
            )
        return self.source_residual_gate_net(z, reliability_features)

    def _conditional_features(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        reliability_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        e1 = self.enc1(x)
        if self.hyper_enable_film:
            e1 = self.film1(e1, z)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        if self.hyper_enable_film:
            e2 = self.film2(e2, z)
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        if self.hyper_enable_film:
            e3 = self.film3(e3, z)

        b = self.bottleneck(e3)
        if self.hyper_enable_film:
            b = self.film_b(b, z)
        reliability_multiplier = self.prompt_manifold_reliability_multiplier(
            z,
            reliability_features,
        )
        if self.hyper_enable_adapters:
            gate_b = self.adapter_reliability_gate(z, "bottleneck") * reliability_multiplier
            b = self.hyper_adapter_b(
                b,
                z,
                logit_residual=self._target_residual("target_adapter_coefficient_residual_b"),
                coeff_logits=self.adapter_coefficient_logits(z, "bottleneck"),
                residual_gate=gate_b,
            )

        d2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        if self.hyper_enable_adapters:
            gate_d2 = self.adapter_reliability_gate(z, "dec2") * reliability_multiplier
            d2 = self.hyper_adapter_d2(
                d2,
                z,
                logit_residual=self._target_residual("target_adapter_coefficient_residual_d2"),
                coeff_logits=self.adapter_coefficient_logits(z, "dec2"),
                residual_gate=gate_d2,
            )
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        if self.hyper_enable_adapters:
            gate_d1 = self.adapter_reliability_gate(z, "dec1") * reliability_multiplier
            d1 = self.hyper_adapter_d1(
                d1,
                z,
                logit_residual=self._target_residual("target_adapter_coefficient_residual_d1"),
                coeff_logits=self.adapter_coefficient_logits(z, "dec1"),
                residual_gate=gate_d1,
            )
        return d1

    def _apply_target_output_adaptation(
        self,
        y: torch.Tensor,
        *,
        month: Optional[torch.Tensor],
        x: torch.Tensor,
        x_raw: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.enable_target_adaptation and self.residual_gain is not None:
            if month is None:
                raise ValueError("target adaptation residual gain requires month tensor")
            y = self.residual_gain(y, month)
        if self.enable_target_adaptation and self.target_spatial_refine is not None:
            refine_x = x_raw if self.target_spatial_refine_input == "raw" and x_raw is not None else x
            if self.target_spatial_refine_type in {"hydro_msr_gain", "hydro_msr_gain_lite"}:
                y = y + self.target_spatial_refine(refine_x, y, x_raw=x_raw, month=month)
            else:
                y = y + self.target_spatial_refine(refine_x, y, x_raw=x_raw)
        return y

    def forward(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        month: Optional[torch.Tensor] = None,
        x_raw: Optional[torch.Tensor] = None,
        reliability_features: Optional[torch.Tensor] = None,
        rho: Optional[float] = None,
    ) -> torch.Tensor:
        if z is None:
            raise ValueError("HyperAdapterConditionalResUNet requires a prompt tensor z")
        if self.enable_target_adaptation:
            if month is None:
                raise ValueError("target adaptation residual gain requires month tensor")
            if self.target_prompt is not None:
                z = self.target_prompt(z)

        if self.uses_source_residual_prior:
            source_base = self.source_base_forward(x)
            conditional_features = self._conditional_features(x, z, reliability_features)
            delta = self.residual_head(conditional_features)
            gate = self.source_residual_reliability_gate(z, reliability_features).view(-1, 1, 1, 1)
            resolved_rho = self.source_residual_rho if rho is None else float(rho)
            if not 0.0 <= resolved_rho <= 1.0:
                raise ValueError("rho must be in [0, 1]")
            y = source_base + resolved_rho * gate.to(dtype=delta.dtype, device=delta.device) * delta
            return self._apply_target_output_adaptation(y, month=month, x=x, x_raw=x_raw)

        d1 = self._conditional_features(x, z, reliability_features)
        y = self.head(d1)
        return self._apply_target_output_adaptation(y, month=month, x=x, x_raw=x_raw)
