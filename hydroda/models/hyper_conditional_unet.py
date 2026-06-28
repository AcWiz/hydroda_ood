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
from hydroda.models.phys_trust import (
    PHYS_FORMULA_GAIN_SOURCE,
    PHYS_FORMULA_SOURCES,
    PHYS_GAIN_BASIS_NAMES,
    PHYS_GAIN_BASIS_SCHEMA_VERSION,
    phys_gain_basis_from_raw_tensor,
    phys_gain_basis_formula_schema,
    phys_formula_feature_schema_for_source,
)


_ADAPTER_LAYER_NAMES = ("bottleneck", "dec2", "dec1")
_ADAPTER_LAYER_TO_INDEX = {name: idx for idx, name in enumerate(_ADAPTER_LAYER_NAMES)}
SOURCE_RESIDUAL_RELIABILITY_FEATURE_SCHEMA = [
    "monthly_count",
    "has_monthly_prototype",
    "global_context_count",
    "finite_input_coverage",
    "source_manifold_distance_bounded",
]
LEGACY_SOURCE_MANIFOLD_DISTANCE_KEY = "prompt_to_source_manifold_distance"
SOURCE_MANIFOLD_DISTANCE_KEY = "source_manifold_distance_bounded"
SOURCE_SALIENCY_PRIOR_APPLICATIONS = (
    "soft_regularization_metadata",
    "legacy_gate_logit_bias_before_topk",
)
PHYS_AGREEMENT_GUARD_RISK_RULES = ("or", "and")
PHYS_CONTEXT_SOURCES = (
    "raw_input_side_da_diagnostics",
    *PHYS_FORMULA_SOURCES,
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


class PhysicalTokenOperatorResidual(nn.Module):
    """Small raw-physical-token residual on adapter coefficient logits.

    The residual is zero-initialized so enabling the module preserves the
    existing HyperDA-TRUST operator at step 0. DropPath applies only to this
    additive physical perturbation and is disabled by ``eval()``.
    """

    def __init__(
        self,
        prompt_dim: int,
        n_basis: int,
        *,
        delta_scale: float = 0.25,
        gate_init: float = 0.90,
        droppath_p: float = 0.10,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        if prompt_dim < 1:
            raise ValueError("prompt_dim must be >= 1")
        if n_basis < 1:
            raise ValueError("n_basis must be >= 1")
        if float(delta_scale) < 0.0:
            raise ValueError("hyper_phys_delta_scale must be non-negative")
        if not 0.0 < float(gate_init) < 1.0:
            raise ValueError("hyper_phys_gate_init must be between 0 and 1")
        if not 0.0 <= float(droppath_p) < 1.0:
            raise ValueError("hyper_operator_droppath_p must be in [0, 1)")
        self.prompt_dim = int(prompt_dim)
        self.n_basis = int(n_basis)
        self.n_layers = int(n_layers)
        self.delta_scale = float(delta_scale)
        self.gate_init = float(gate_init)
        self.droppath_p = float(droppath_p)
        self.layer_embedding = nn.Embedding(self.n_layers, self.prompt_dim)
        self.delta_head = nn.Linear(self.prompt_dim * 3, self.n_basis)
        self.gate_logit = nn.Parameter(torch.tensor(math.log(self.gate_init / (1.0 - self.gate_init))))
        nn.init.normal_(self.layer_embedding.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def _layer_ids(self, z_prompt: torch.Tensor, layer_index: int | torch.Tensor) -> torch.Tensor:
        if isinstance(layer_index, int):
            return torch.full(
                (z_prompt.shape[0],),
                int(layer_index),
                dtype=torch.long,
                device=z_prompt.device,
            )
        layer_ids = layer_index.to(device=z_prompt.device, dtype=torch.long)
        if layer_ids.ndim == 0:
            layer_ids = layer_ids.expand(z_prompt.shape[0])
        if layer_ids.shape != (z_prompt.shape[0],):
            raise ValueError("layer_index tensor must be scalar or shape [B]")
        return layer_ids

    def forward(
        self,
        z_prompt: torch.Tensor,
        z_phys: torch.Tensor,
        layer_index: int | torch.Tensor,
    ) -> torch.Tensor:
        if z_prompt.ndim != 2 or z_prompt.shape[-1] != self.prompt_dim:
            raise ValueError(f"z_prompt must have shape [B, {self.prompt_dim}]")
        if z_phys.ndim != 2 or z_phys.shape != z_prompt.shape:
            raise ValueError(f"z_phys must have shape [B, {self.prompt_dim}]")
        layer_ids = self._layer_ids(z_prompt, layer_index)
        layer_z = self.layer_embedding(layer_ids)
        delta = torch.tanh(
            self.delta_head(torch.cat([z_prompt, z_phys.to(dtype=z_prompt.dtype), layer_z], dim=-1))
        )
        gate = torch.sigmoid(self.gate_logit).to(device=delta.device, dtype=delta.dtype)
        delta = delta * gate * float(self.delta_scale)
        if self.training and self.droppath_p > 0.0:
            keep_prob = 1.0 - float(self.droppath_p)
            mask = torch.empty(
                delta.shape[0],
                1,
                dtype=delta.dtype,
                device=delta.device,
            ).bernoulli_(keep_prob)
            delta = delta * mask / keep_prob
        return delta


class FormulaPhysicalContextEncoder(nn.Module):
    """Small M3.8 encoder for bounded raw formula features."""

    def __init__(
        self,
        feature_dim: int,
        prompt_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if int(feature_dim) < 1:
            raise ValueError("phys formula feature_dim must be >= 1")
        if int(prompt_dim) < 1:
            raise ValueError("prompt_dim must be >= 1")
        self.feature_dim = int(feature_dim)
        self.prompt_dim = int(prompt_dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else max(16, int(prompt_dim) // 2)
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.prompt_dim),
            nn.LayerNorm(self.prompt_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"formula physical features must have shape [B, {self.feature_dim}]")
        features = torch.nan_to_num(features.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        return self.net(features)


class PhysGainBasisHyperResidual(nn.Module):
    """M3.12 signed physics-gain basis residual operator.

    Coefficient heads are zero-initialized, so enabling this branch preserves
    the existing HyperDA-TRUST design path exactly at step 0.
    """

    def __init__(
        self,
        prompt_dim: int,
        *,
        n_basis: int = len(PHYS_GAIN_BASIS_NAMES),
        coeff_scale: float = 0.05,
        residual_clip: float = 0.25,
        beta_init: float = 0.50,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if int(prompt_dim) < 1:
            raise ValueError("prompt_dim must be >= 1")
        if int(n_basis) != len(PHYS_GAIN_BASIS_NAMES):
            raise ValueError(f"phys gain basis residual expects {len(PHYS_GAIN_BASIS_NAMES)} basis maps")
        if float(coeff_scale) < 0.0:
            raise ValueError("hyper_phys_gain_basis_coeff_scale must be non-negative")
        if float(residual_clip) <= 0.0:
            raise ValueError("hyper_phys_gain_basis_residual_clip must be positive")
        if not 0.0 < float(beta_init) < 1.0:
            raise ValueError("hyper_phys_gain_basis_beta_init must be between 0 and 1")
        self.prompt_dim = int(prompt_dim)
        self.n_basis = int(n_basis)
        self.coeff_scale = float(coeff_scale)
        self.residual_clip = float(residual_clip)
        self.beta_init = float(beta_init)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else max(16, int(prompt_dim) // 2)
        self.basis_encoder = nn.Sequential(
            nn.Linear(self.n_basis * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.prompt_dim),
            nn.LayerNorm(self.prompt_dim),
        )
        self.coeff_head = nn.Linear(self.prompt_dim * 2, 2 * self.n_basis + 1)
        self.beta_head = nn.Linear(self.prompt_dim * 2, 2)
        nn.init.zeros_(self.coeff_head.weight)
        nn.init.zeros_(self.coeff_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.constant_(self.beta_head.bias, math.log(self.beta_init / (1.0 - self.beta_init)))
        self.last_summary: dict[str, object] = {"enabled": False}

    def _basis_stats(self, basis_maps: torch.Tensor) -> torch.Tensor:
        cleaned = torch.nan_to_num(basis_maps.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        flat = cleaned.flatten(start_dim=2)
        means = flat.mean(dim=-1)
        stds = flat.std(dim=-1, unbiased=False)
        return torch.cat([means, stds], dim=1).to(device=basis_maps.device, dtype=basis_maps.dtype)

    @staticmethod
    def _mean_list(values: torch.Tensor) -> list[float]:
        detached = values.detach().cpu().float()
        if detached.ndim == 0:
            return [float(detached.item())]
        return [float(v) for v in detached.reshape(-1).tolist()]

    def forward(
        self,
        z_prompt: torch.Tensor,
        basis_maps: torch.Tensor,
        *,
        source_gain_bank: dict[str, object] | None = None,
    ) -> torch.Tensor:
        if z_prompt.ndim != 2 or z_prompt.shape[-1] != self.prompt_dim:
            raise ValueError(f"z_prompt must have shape [B, {self.prompt_dim}]")
        if basis_maps.ndim != 4 or basis_maps.shape[1] != self.n_basis:
            raise ValueError(f"phys gain basis maps must have shape [B, {self.n_basis}, H, W]")
        if basis_maps.shape[0] != z_prompt.shape[0]:
            raise ValueError("phys gain basis batch size must match z_prompt")
        basis = torch.nan_to_num(basis_maps.to(device=z_prompt.device, dtype=z_prompt.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        basis = basis.clamp(-1.0, 1.0)
        z_phys = self.basis_encoder(self._basis_stats(basis).to(dtype=z_prompt.dtype))
        h = torch.cat([z_prompt, z_phys], dim=1)
        raw = self.coeff_head(h)
        coeff = torch.tanh(raw[:, : 2 * self.n_basis]).view(-1, 2, self.n_basis) * self.coeff_scale
        a_rz = torch.tanh(raw[:, -1:]) * self.coeff_scale
        beta = torch.sigmoid(self.beta_head(h)).to(dtype=basis.dtype)
        q_surface = (coeff[:, 0, :, None, None] * basis).sum(dim=1)
        q_rootzone_direct = (coeff[:, 1, :, None, None] * basis).sum(dim=1)
        q_rootzone = q_rootzone_direct + a_rz.view(-1, 1, 1) * q_surface
        q = torch.stack([q_surface, q_rootzone], dim=1).clamp(-self.residual_clip, self.residual_clip)
        residual = beta.view(-1, 2, 1, 1) * q

        coeff_detached = coeff.detach().cpu().float()
        beta_detached = beta.detach().cpu().float()
        residual_detached = residual.detach().cpu().float()
        sign_agreement = {}
        for var_idx, variable in enumerate(("surface", "rootzone")):
            hv = coeff_detached[:, var_idx, :2]
            sign_agreement[variable] = {
                "H_V_nonnegative_fraction": float((hv >= 0.0).float().mean().item()) if hv.numel() else 0.0,
                "H_mean": float(hv[:, 0].mean().item()) if hv.numel() else 0.0,
                "V_mean": float(hv[:, 1].mean().item()) if hv.numel() else 0.0,
            }
        bank_hash = ""
        if source_gain_bank:
            bank_hash = str(source_gain_bank.get("source_gain_bank_hash") or source_gain_bank.get("trust_bank_hash") or "")
        self.last_summary = {
            "enabled": True,
            "schema_version": PHYS_GAIN_BASIS_SCHEMA_VERSION,
            "formula_schema": phys_gain_basis_formula_schema(),
            "basis_names": list(PHYS_GAIN_BASIS_NAMES),
            "coefficient_scale": float(self.coeff_scale),
            "residual_clip": float(self.residual_clip),
            "beta_init": float(self.beta_init),
            "beta_mean_surface": float(beta_detached[:, 0].mean().item()) if beta_detached.numel() else 0.0,
            "beta_mean_rootzone": float(beta_detached[:, 1].mean().item()) if beta_detached.numel() else 0.0,
            "coefficient_mean_surface": self._mean_list(coeff_detached[:, 0].mean(dim=0)),
            "coefficient_mean_rootzone": self._mean_list(coeff_detached[:, 1].mean(dim=0)),
            "rootzone_surface_coupling_mean": float(a_rz.detach().cpu().float().mean().item()) if a_rz.numel() else 0.0,
            "residual_abs_mean": float(residual_detached.abs().mean().item()) if residual_detached.numel() else 0.0,
            "sign_agreement": sign_agreement,
            "source_gain_bank_hash": bank_hash,
            "source_gain_bank_used_for_forward": False,
            "source_gain_bank_role": "interpretation_and_weak_regularization_metadata_only",
            "label_usage": "none_for_forward",
            "target_val_usage": "unused_in_main_protocol",
            "target_eval_usage": "final_eval_only_no_selection",
            "base_valid_mask_usage": "diagnostic_only_not_loss_metric_obs_region_mask_or_gate_mask",
        }
        return residual


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
        hyper_source_manifold_guard: bool = False,
        hyper_source_manifold_guard_strength: float = 0.25,
        hyper_source_manifold_guard_distance_key: str = SOURCE_MANIFOLD_DISTANCE_KEY,
        hyper_source_manifold_guard_min_multiplier: float = 0.0,
        source_manifold_guard_calibration: str = "source_fit_source_val_only",
        hyper_source_trust_routing: bool = False,
        hyper_source_trust_strength: float = 0.0,
        hyper_source_trust_top_m: int = 4,
        hyper_source_trust_temperature: float | None = None,
        hyper_source_trust_variable_gate: bool = False,
        hyper_phys_agreement_guard: bool = False,
        hyper_phys_agreement_guard_strength: float = 1.0,
        hyper_phys_agreement_guard_min_multiplier: float = 0.0,
        hyper_phys_agreement_guard_risk_rule: str = "or",
        hyper_phys_context_modulation: bool = False,
        hyper_phys_delta_scale: float = 0.25,
        hyper_phys_gate_init: float = 0.90,
        hyper_operator_droppath_p: float = 0.10,
        phys_context_source: str = "raw_input_side_da_diagnostics",
        hyper_phys_gain_basis_residual: bool = False,
        hyper_phys_gain_basis_coeff_scale: float = 0.05,
        hyper_phys_gain_basis_residual_clip: float = 0.25,
        hyper_phys_gain_basis_beta_init: float = 0.50,
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
        if float(hyper_source_manifold_guard_strength) < 0.0:
            raise ValueError("hyper_source_manifold_guard_strength must be non-negative")
        if hyper_source_manifold_guard_distance_key not in {
            SOURCE_MANIFOLD_DISTANCE_KEY,
            LEGACY_SOURCE_MANIFOLD_DISTANCE_KEY,
        }:
            raise ValueError(
                "hyper_source_manifold_guard_distance_key must be "
                f"{SOURCE_MANIFOLD_DISTANCE_KEY!r}"
            )
        if not 0.0 <= float(hyper_source_manifold_guard_min_multiplier) <= 1.0:
            raise ValueError("hyper_source_manifold_guard_min_multiplier must be in [0, 1]")
        if not 0.0 <= float(hyper_source_trust_strength) <= 1.0:
            raise ValueError("hyper_source_trust_strength must be in [0, 1]")
        if int(hyper_source_trust_top_m) < 1:
            raise ValueError("hyper_source_trust_top_m must be >= 1")
        if hyper_source_trust_temperature is not None and float(hyper_source_trust_temperature) <= 0.0:
            raise ValueError("hyper_source_trust_temperature must be positive")
        if not 0.0 <= float(hyper_phys_agreement_guard_strength) <= 1.0:
            raise ValueError("hyper_phys_agreement_guard_strength must be in [0, 1]")
        if not 0.0 <= float(hyper_phys_agreement_guard_min_multiplier) <= 1.0:
            raise ValueError("hyper_phys_agreement_guard_min_multiplier must be in [0, 1]")
        if hyper_phys_agreement_guard_risk_rule not in PHYS_AGREEMENT_GUARD_RISK_RULES:
            raise ValueError(
                "hyper_phys_agreement_guard_risk_rule must be one of "
                f"{PHYS_AGREEMENT_GUARD_RISK_RULES}"
            )
        if float(hyper_phys_delta_scale) < 0.0:
            raise ValueError("hyper_phys_delta_scale must be non-negative")
        if not 0.0 < float(hyper_phys_gate_init) < 1.0:
            raise ValueError("hyper_phys_gate_init must be between 0 and 1")
        if not 0.0 <= float(hyper_operator_droppath_p) < 1.0:
            raise ValueError("hyper_operator_droppath_p must be in [0, 1)")
        if phys_context_source not in PHYS_CONTEXT_SOURCES:
            raise ValueError(f"phys_context_source must be one of {PHYS_CONTEXT_SOURCES}")
        if float(hyper_phys_gain_basis_coeff_scale) < 0.0:
            raise ValueError("hyper_phys_gain_basis_coeff_scale must be non-negative")
        if float(hyper_phys_gain_basis_residual_clip) <= 0.0:
            raise ValueError("hyper_phys_gain_basis_residual_clip must be positive")
        if not 0.0 < float(hyper_phys_gain_basis_beta_init) < 1.0:
            raise ValueError("hyper_phys_gain_basis_beta_init must be between 0 and 1")
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
        self.hyper_source_manifold_guard = bool(hyper_source_manifold_guard)
        self.hyper_source_manifold_guard_strength = float(hyper_source_manifold_guard_strength)
        self.hyper_source_manifold_guard_distance_key = str(hyper_source_manifold_guard_distance_key)
        self.hyper_source_manifold_guard_min_multiplier = float(hyper_source_manifold_guard_min_multiplier)
        self.source_manifold_guard_calibration = str(source_manifold_guard_calibration)
        self.hyper_source_trust_routing = bool(hyper_source_trust_routing)
        self.hyper_source_trust_strength = float(hyper_source_trust_strength)
        self.hyper_source_trust_top_m = int(hyper_source_trust_top_m)
        self.hyper_source_trust_temperature = (
            float(hyper_source_trust_temperature)
            if hyper_source_trust_temperature is not None
            else None
        )
        self.hyper_source_trust_variable_gate = bool(hyper_source_trust_variable_gate)
        self.hyper_phys_agreement_guard = bool(hyper_phys_agreement_guard)
        self.hyper_phys_agreement_guard_strength = float(hyper_phys_agreement_guard_strength)
        self.hyper_phys_agreement_guard_min_multiplier = float(hyper_phys_agreement_guard_min_multiplier)
        self.hyper_phys_agreement_guard_risk_rule = str(hyper_phys_agreement_guard_risk_rule)
        self.hyper_phys_context_modulation = bool(hyper_phys_context_modulation)
        self.hyper_phys_delta_scale = float(hyper_phys_delta_scale)
        self.hyper_phys_gate_init = float(hyper_phys_gate_init)
        self.hyper_operator_droppath_p = float(hyper_operator_droppath_p)
        self.phys_context_source = str(phys_context_source)
        self.hyper_phys_gain_basis_residual = bool(hyper_phys_gain_basis_residual)
        self.hyper_phys_gain_basis_coeff_scale = float(hyper_phys_gain_basis_coeff_scale)
        self.hyper_phys_gain_basis_residual_clip = float(hyper_phys_gain_basis_residual_clip)
        self.hyper_phys_gain_basis_beta_init = float(hyper_phys_gain_basis_beta_init)
        self.trust_routing_geometry = "prompt_embedding"
        self.last_trust_routing_summary: dict[str, dict[str, object]] = {}
        self.last_variable_trust_gate_summary: dict[str, dict[str, float]] = {}
        self.last_phys_agreement_guard_summary: dict[str, object] = {}
        self.last_phys_agreement_guard_query_source: str = ""
        self.last_phys_agreement_guard_query: torch.Tensor | None = None
        self.last_phys_agreement_guard_multiplier: torch.Tensor | None = None
        self.last_phys_operator_summary: dict[str, object] = {"enabled": False}
        self.last_phys_gain_basis_summary: dict[str, object] = {"enabled": False}
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
        self.phys_operator_residual: PhysicalTokenOperatorResidual | None = None
        self.formula_phys_context_encoder: FormulaPhysicalContextEncoder | None = None
        self.phys_gain_basis_residual: PhysGainBasisHyperResidual | None = None
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

        if self.hyper_phys_context_modulation:
            if self.phys_context_source in PHYS_FORMULA_SOURCES:
                self.formula_phys_context_encoder = FormulaPhysicalContextEncoder(
                    feature_dim=len(phys_formula_feature_schema_for_source(self.phys_context_source)),
                    prompt_dim=prompt_dim,
                )
            self.phys_operator_residual = PhysicalTokenOperatorResidual(
                prompt_dim=prompt_dim,
                n_basis=self.hyper_n_basis,
                delta_scale=self.hyper_phys_delta_scale,
                gate_init=self.hyper_phys_gate_init,
                droppath_p=self.hyper_operator_droppath_p,
                n_layers=len(_ADAPTER_LAYER_NAMES),
            )

        if self.hyper_phys_gain_basis_residual:
            self.phys_gain_basis_residual = PhysGainBasisHyperResidual(
                prompt_dim=prompt_dim,
                coeff_scale=self.hyper_phys_gain_basis_coeff_scale,
                residual_clip=self.hyper_phys_gain_basis_residual_clip,
                beta_init=self.hyper_phys_gain_basis_beta_init,
            )

        self._zero_raw_increment_init = zero_raw_increment_init
        if zero_raw_increment_init:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def _layer_index(self, layer_name: str) -> int:
        if layer_name not in _ADAPTER_LAYER_TO_INDEX:
            raise ValueError(f"unknown HyperDA adapter layer: {layer_name}")
        return _ADAPTER_LAYER_TO_INDEX[layer_name]

    def adapter_coefficient_logits(
        self,
        z: torch.Tensor,
        layer_name: str,
        source_trust_bank: dict[str, object] | None = None,
        source_trust_query: torch.Tensor | None = None,
        z_phys: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return basis logits for an adapter layer under the configured generator."""
        if self.hyper_coeff_generator in {
            "shared_layer_aware",
            "shared_layer_aware_rank_gated",
            "shared_layer_aware_rank_gated_stable",
        }:
            if self.shared_coeff_generator is None:
                raise RuntimeError("shared coefficient generator is not initialized")
            logits = self.shared_coeff_generator(z, self._layer_index(layer_name))
            return self.trust_routed_coefficient_logits(
                z,
                layer_name,
                logits,
                source_trust_bank=source_trust_bank,
                source_trust_query=source_trust_query,
                z_phys=z_phys,
            )
        adapter = self._adapter_module(layer_name)
        logits = adapter.coefficient_logits(z)
        return self.trust_routed_coefficient_logits(
            z,
            layer_name,
            logits,
            source_trust_bank=source_trust_bank,
            source_trust_query=source_trust_query,
            z_phys=z_phys,
        )

    @staticmethod
    def _trust_bank_tensor(
        value: object,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        tensor = value.detach().clone() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        tensor = tensor.to(device=device, dtype=dtype)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _trust_bank_distance_temperature(
        self,
        source_trust_bank: dict[str, object],
        *,
        quantile_key: str,
        fallback_quantile_keys: tuple[str, ...] = ("distance_quantiles",),
    ) -> float:
        if self.hyper_source_trust_temperature is not None:
            value = float(self.hyper_source_trust_temperature)
            return value if math.isfinite(value) and value > 0.0 else 1.0
        for key in (quantile_key, *fallback_quantile_keys):
            q = source_trust_bank.get(key, {})
            if not isinstance(q, dict):
                continue
            value = float(q.get("q75") or q.get("q90") or q.get("max") or 1.0)
            if math.isfinite(value) and value > 0.0:
                return value
        return 1.0

    def trust_routed_coefficient_logits(
        self,
        z: torch.Tensor,
        layer_name: str,
        target_logits: torch.Tensor,
        source_trust_bank: dict[str, object] | None = None,
        source_trust_query: torch.Tensor | None = None,
        z_phys: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Blend target-prompt logits with nearest source-neighborhood consensus.

        The source trust bank is source-fit/source-val input-side metadata. A
        zero trust strength is an exact identity for M2.1-style forwards.
        """
        if (
            not self.hyper_source_trust_routing
            or self.hyper_source_trust_strength <= 0.0
            or not source_trust_bank
        ):
            return self.phys_modulated_coefficient_logits(
                z,
                z_phys,
                layer_name,
                target_logits,
            )
        if target_logits.ndim != 2 or target_logits.shape[0] != z.shape[0]:
            raise ValueError("target_logits must have shape [B, n_basis]")

        source_embedding_key = "source_prompt_embeddings"
        guard_uses_phys_query = self.hyper_phys_agreement_guard and source_trust_query is not None
        if source_trust_query is not None and not guard_uses_phys_query:
            if source_trust_bank.get("source_trust_query_embeddings") is None:
                raise ValueError(
                    "source_trust_query was provided but source trust bank is missing "
                    "source_trust_query_embeddings; refusing prompt-space fallback"
                )
            source_embedding_key = "source_trust_query_embeddings"
        source_prompt_embeddings = self._trust_bank_tensor(
            source_trust_bank.get(source_embedding_key),
            device=z.device,
            dtype=z.dtype,
        )
        layer_map = source_trust_bank.get("layer_consensus_logits", {})
        if not isinstance(layer_map, dict):
            raise ValueError("source_trust_bank layer_consensus_logits must be a dict")
        consensus_logits = self._trust_bank_tensor(
            layer_map.get(layer_name),
            device=target_logits.device,
            dtype=target_logits.dtype,
        )
        if source_prompt_embeddings is None or consensus_logits is None:
            return target_logits
        if source_prompt_embeddings.ndim != 2 or consensus_logits.ndim != 2:
            raise ValueError("source trust bank tensors must be rank-2")
        if source_prompt_embeddings.shape[0] != consensus_logits.shape[0]:
            raise ValueError("source prompt embeddings and consensus logits row counts differ")
        trust_query = z if source_trust_query is None or guard_uses_phys_query else source_trust_query
        if trust_query.ndim != 2 or trust_query.shape[0] != z.shape[0]:
            raise ValueError("source_trust_query must have shape [B, prompt_dim]")
        if source_prompt_embeddings.shape[1] != trust_query.shape[1]:
            raise ValueError(
                "source trust bank prompt dimension mismatch: "
                f"source={tuple(source_prompt_embeddings.shape)} query={tuple(trust_query.shape)}"
            )
        if consensus_logits.shape[1] != target_logits.shape[1]:
            raise ValueError(
                "source trust bank consensus dimension mismatch: "
                f"source={tuple(consensus_logits.shape)} logits={tuple(target_logits.shape)}"
            )

        top_m = min(self.hyper_source_trust_top_m, int(source_prompt_embeddings.shape[0]))
        distances = torch.cdist(
            trust_query.to(dtype=torch.float32),
            source_prompt_embeddings.to(dtype=torch.float32),
            p=2,
        )
        neighbor_distances, neighbor_indices = torch.topk(distances, k=top_m, dim=1, largest=False)
        temperature = self._trust_bank_distance_temperature(
            source_trust_bank,
            quantile_key=(
                "prompt_distance_quantiles"
                if source_embedding_key == "source_prompt_embeddings"
                else "source_trust_query_distance_quantiles"
            ),
        )
        weights = torch.softmax(
            -neighbor_distances.to(device=target_logits.device, dtype=target_logits.dtype) / float(temperature),
            dim=1,
        )
        neighbor_logits = consensus_logits[neighbor_indices.to(device=consensus_logits.device)]
        consensus = (weights.unsqueeze(-1) * neighbor_logits).sum(dim=1)
        guard_multiplier = self.phys_agreement_guard_multiplier(
            z,
            source_trust_bank=source_trust_bank,
            source_trust_query=source_trust_query,
        )
        effective_strength = (
            float(self.hyper_source_trust_strength)
            * guard_multiplier.to(device=target_logits.device, dtype=target_logits.dtype)
        )
        routed = (1.0 - effective_strength) * target_logits + effective_strength * consensus
        routed = self.phys_modulated_coefficient_logits(
            z,
            z_phys,
            layer_name,
            routed,
        )
        nearest = neighbor_distances[:, 0].detach().cpu()
        bounded = (nearest / max(float(temperature), 1e-6)).clamp(0.0, 1.0)
        effective_detached = effective_strength.detach().cpu().view(-1)
        self.last_trust_routing_summary[layer_name] = {
            "enabled": True,
            "trust_strength": float(self.hyper_source_trust_strength),
            "trust_routing_geometry": "prompt_embedding",
            "effective_trust_strength_mean": (
                float(effective_detached.mean().item()) if effective_detached.numel() else 0.0
            ),
            "effective_trust_strength_min": (
                float(effective_detached.min().item()) if effective_detached.numel() else 0.0
            ),
            "effective_trust_strength_max": (
                float(effective_detached.max().item()) if effective_detached.numel() else 0.0
            ),
            "tau_layer": float(1.0 - float(self.hyper_source_trust_strength)),
            "effective_tau_layer_mean": (
                float((1.0 - effective_detached).mean().item()) if effective_detached.numel() else 1.0
            ),
            "source_neighbor_top_m": int(top_m),
            "distance_temperature": float(temperature),
            "trust_query_source": (
                "target_prompt"
                if source_trust_query is None or guard_uses_phys_query
                else "source_trust_query"
            ),
            "source_bank_embedding_key": source_embedding_key,
            "nearest_distance_mean": float(nearest.mean().item()) if nearest.numel() else 0.0,
            "nearest_distance_bounded_mean": float(bounded.mean().item()) if bounded.numel() else 0.0,
            "nearest_neighbor_indices": neighbor_indices.detach().cpu().tolist(),
            "phys_agreement_guard": dict(self.last_phys_agreement_guard_summary),
        }
        return routed

    def phys_modulated_coefficient_logits(
        self,
        z_prompt: torch.Tensor,
        z_phys: torch.Tensor | None,
        layer_name: str,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Add the M3.6 physical-token residual to coefficient logits."""
        if not self.hyper_phys_context_modulation:
            self.last_phys_operator_summary = {"enabled": False}
            return base_logits
        if self.phys_operator_residual is None:
            raise RuntimeError("physical operator residual module is not initialized")
        if z_phys is None:
            raise ValueError("hyper_phys_context_modulation requires z_phys")
        delta = self.phys_operator_residual(z_prompt, z_phys, self._layer_index(layer_name))
        if delta.shape != base_logits.shape:
            raise ValueError("physical operator residual shape mismatch")
        delta_detached = delta.detach().cpu()
        source_name = (
            "x_raw_month_region_mask_formula_features_only"
            if self.phys_context_source in PHYS_FORMULA_SOURCES
            else "x_x_raw_month_region_mask_only"
        )
        self.last_phys_operator_summary = {
            "enabled": True,
            "layer_name": layer_name,
            "phys_context_source": self.phys_context_source,
            "trust_routing_geometry": "prompt_embedding",
            "coefficient_injection_role": "bounded_operator_coefficient_logit_delta_only",
            "coefficient_injection_formula": (
                "logits_l = logits_l_M3design + sigmoid(g_l) * "
                "delta_scale * DeltaLogits_l(z_prompt,z_phys)"
            ),
            "final_output_residual_allowed": False,
            "operator_droppath_training_only": True,
            "operator_droppath_active": bool(self.training and self.hyper_operator_droppath_p > 0.0),
            "operator_droppath_p": float(self.hyper_operator_droppath_p),
            "phys_delta_scale": float(self.hyper_phys_delta_scale),
            "phys_gate_init": float(self.hyper_phys_gate_init),
            "phys_gate_value": float(torch.sigmoid(self.phys_operator_residual.gate_logit).detach().cpu().item()),
            "delta_abs_mean": float(delta_detached.abs().mean().item()) if delta_detached.numel() else 0.0,
            "label_usage": "none",
            "target_val_usage": "unused_in_main_protocol",
            "target_eval_usage": "final_eval_only_no_selection",
            "source": source_name,
            "base_valid_mask_usage": "bounded_diagnostic_coverage_only_not_loss_metric_obs_or_region_mask",
        }
        if self.phys_context_source == PHYS_FORMULA_GAIN_SOURCE:
            lite_source_only = (
                abs(float(self.hyper_phys_delta_scale) - 0.03) < 1e-12
                and abs(float(self.hyper_phys_gate_init) - 0.25) < 1e-12
                and not bool(self.hyper_phys_gain_basis_residual)
            )
            self.last_phys_operator_summary.update(
                {
                    "method_id": (
                        "M3_16_source_only_phys_m3trust_lite"
                        if lite_source_only
                        else "M3_14_source_trained_phys_formula_gain_hypertrust"
                    ),
                    "checkpoint_start": "source_pooled_global_backbone",
                    "warm_start_policy": "none_clean_source_only_checkpoint_full_hypernetwork_training",
                    "stage2_source_only_invariant": True,
                    "second_model_forward_allowed": False,
                    "source_fit_regularization_lambda_default": 0.0 if lite_source_only else 0.01,
                }
            )
        return base_logits + delta

    def phys_agreement_guard_multiplier(
        self,
        z: torch.Tensor,
        *,
        source_trust_bank: dict[str, object] | None = None,
        source_trust_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return shrink-only guard multiplier from prompt/physical neighbor agreement.

        M3.5 keeps prompt embeddings as the coefficient-routing geometry. The
        optional raw/blended physical query may only reduce the trust strength
        and residual branch when its nearest source neighbors disagree with the
        prompt-neighbor set or are physically OOD.
        """
        ones = torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)
        if not self.hyper_phys_agreement_guard:
            self.last_phys_agreement_guard_summary = {"enabled": False}
            self.last_phys_agreement_guard_multiplier = ones.detach()
            return ones
        if source_trust_query is None or not source_trust_bank:
            self.last_phys_agreement_guard_summary = {
                "enabled": True,
                "active": False,
                "reason": "missing_source_trust_query_or_bank",
                "multiplier_mean": 1.0,
            }
            self.last_phys_agreement_guard_multiplier = ones.detach()
            return ones
        if source_trust_bank.get("source_trust_query_embeddings") is None:
            raise ValueError(
                "hyper_phys_agreement_guard requires source_trust_query_embeddings; "
                "refusing to use physical diagnostics without a source-side query bank"
            )
        prompt_emb = self._trust_bank_tensor(
            source_trust_bank.get("source_prompt_embeddings"),
            device=z.device,
            dtype=z.dtype,
        )
        phys_emb = self._trust_bank_tensor(
            source_trust_bank.get("source_trust_query_embeddings"),
            device=z.device,
            dtype=z.dtype,
        )
        if prompt_emb is None or phys_emb is None:
            self.last_phys_agreement_guard_summary = {
                "enabled": True,
                "active": False,
                "reason": "empty_source_embedding_bank",
                "multiplier_mean": 1.0,
            }
            self.last_phys_agreement_guard_multiplier = ones.detach()
            return ones
        if prompt_emb.ndim != 2 or phys_emb.ndim != 2:
            raise ValueError("phys agreement guard source embeddings must be rank-2")
        if prompt_emb.shape != phys_emb.shape:
            raise ValueError(
                "phys agreement guard prompt and physical source banks must have matching shape: "
                f"prompt={tuple(prompt_emb.shape)} phys={tuple(phys_emb.shape)}"
            )
        query = source_trust_query.to(device=z.device, dtype=z.dtype)
        if query.ndim != 2 or query.shape[0] != z.shape[0]:
            raise ValueError("source_trust_query must have shape [B, prompt_dim]")
        if query.shape[1] != phys_emb.shape[1] or z.shape[1] != prompt_emb.shape[1]:
            raise ValueError(
                "phys agreement guard embedding dimension mismatch: "
                f"prompt_bank={tuple(prompt_emb.shape)} phys_bank={tuple(phys_emb.shape)} "
                f"z={tuple(z.shape)} query={tuple(query.shape)}"
            )
        top_m = min(self.hyper_source_trust_top_m, int(prompt_emb.shape[0]))
        prompt_distances = torch.cdist(z.to(dtype=torch.float32), prompt_emb.to(dtype=torch.float32), p=2)
        phys_distances = torch.cdist(query.to(dtype=torch.float32), phys_emb.to(dtype=torch.float32), p=2)
        prompt_neighbor_distances, prompt_indices = torch.topk(
            prompt_distances,
            k=top_m,
            dim=1,
            largest=False,
        )
        phys_neighbor_distances, phys_indices = torch.topk(
            phys_distances,
            k=top_m,
            dim=1,
            largest=False,
        )
        agreement_values: list[float] = []
        prompt_index_rows = prompt_indices.detach().cpu().tolist()
        phys_index_rows = phys_indices.detach().cpu().tolist()
        for prompt_row, phys_row in zip(prompt_index_rows, phys_index_rows):
            overlap = len(set(int(v) for v in prompt_row) & set(int(v) for v in phys_row))
            agreement_values.append(float(overlap) / float(max(top_m, 1)))
        agreement = torch.as_tensor(
            agreement_values,
            dtype=z.dtype,
            device=z.device,
        ).view(-1, 1)
        temperature = self._trust_bank_distance_temperature(
            source_trust_bank,
            quantile_key="source_trust_query_distance_quantiles",
        )
        phys_nearest = phys_neighbor_distances[:, 0].to(device=z.device, dtype=z.dtype).view(-1, 1)
        phys_bounded = (phys_nearest / max(float(temperature), 1e-6)).clamp(0.0, 1.0)
        disagreement = (1.0 - agreement).clamp(0.0, 1.0)
        if self.hyper_phys_agreement_guard_risk_rule == "and":
            shrink_risk = torch.minimum(disagreement, phys_bounded)
        else:
            shrink_risk = torch.maximum(disagreement, phys_bounded)
        multiplier = (1.0 - float(self.hyper_phys_agreement_guard_strength) * shrink_risk).clamp(
            float(self.hyper_phys_agreement_guard_min_multiplier),
            1.0,
        )
        detached_multiplier = multiplier.detach().cpu().view(-1)
        detached_agreement = agreement.detach().cpu().view(-1)
        detached_phys = phys_bounded.detach().cpu().view(-1)
        detached_risk = shrink_risk.detach().cpu().view(-1)
        self.last_phys_agreement_guard_query = source_trust_query
        self.last_phys_agreement_guard_summary = {
            "enabled": True,
            "active": True,
            "role": "shrink_only_no_enhance",
            "trust_routing_geometry": "prompt_embedding",
            "phys_query_usage": "guard_only_not_neighbor_geometry",
            "source": "x_x_raw_month_region_mask_only",
            "label_usage": "none",
            "target_val_usage": "unused_in_main_protocol",
            "target_eval_usage": "final_eval_only_no_selection",
            "guard_strength": float(self.hyper_phys_agreement_guard_strength),
            "min_multiplier": float(self.hyper_phys_agreement_guard_min_multiplier),
            "risk_rule": self.hyper_phys_agreement_guard_risk_rule,
            "source_neighbor_top_m": int(top_m),
            "distance_temperature": float(temperature),
            "prompt_neighbor_indices": prompt_index_rows,
            "phys_neighbor_indices": phys_index_rows,
            "soft_neighbor_agreement_mean": (
                float(detached_agreement.mean().item()) if detached_agreement.numel() else 1.0
            ),
            "soft_neighbor_agreement_min": (
                float(detached_agreement.min().item()) if detached_agreement.numel() else 1.0
            ),
            "phys_ood_distance_bounded_mean": (
                float(detached_phys.mean().item()) if detached_phys.numel() else 0.0
            ),
            "phys_ood_distance_bounded_max": (
                float(detached_phys.max().item()) if detached_phys.numel() else 0.0
            ),
            "shrink_risk_mean": (
                float(detached_risk.mean().item()) if detached_risk.numel() else 0.0
            ),
            "shrink_risk_max": (
                float(detached_risk.max().item()) if detached_risk.numel() else 0.0
            ),
            "multiplier_mean": (
                float(detached_multiplier.mean().item()) if detached_multiplier.numel() else 1.0
            ),
            "multiplier_min": (
                float(detached_multiplier.min().item()) if detached_multiplier.numel() else 1.0
            ),
            "multiplier_max": (
                float(detached_multiplier.max().item()) if detached_multiplier.numel() else 1.0
            ),
            "effective_trust_strength_mean": (
                float((detached_multiplier * float(self.hyper_source_trust_strength)).mean().item())
                if detached_multiplier.numel()
                else float(self.hyper_source_trust_strength)
            ),
            "prompt_nearest_distance_mean": (
                float(prompt_neighbor_distances[:, 0].detach().cpu().mean().item())
                if prompt_neighbor_distances.numel()
                else 0.0
            ),
            "phys_nearest_distance_mean": (
                float(phys_neighbor_distances[:, 0].detach().cpu().mean().item())
                if phys_neighbor_distances.numel()
                else 0.0
            ),
        }
        self.last_phys_agreement_guard_multiplier = multiplier.detach()
        return multiplier

    def adapter_reliability_gate(self, z: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Return bounded adapter residual gate [B, 1] for a layer."""
        if self.hyper_reliability_gate == "none":
            return torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)
        if self.reliability_gate is None:
            raise RuntimeError("prompt scalar reliability gate is not initialized")
        return self.reliability_gate(z, self._layer_index(layer_name))

    def source_manifold_guard_multiplier(
        self,
        z: torch.Tensor,
        reliability_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return conservative adapter/residual multiplier from source-manifold distance."""
        guard_enabled = self.hyper_source_manifold_guard or self.hyper_prompt_manifold_reliability
        if not guard_enabled:
            return torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)
        if reliability_features is None:
            return torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)
        if reliability_features.ndim != 2 or reliability_features.shape[0] != z.shape[0]:
            raise ValueError("reliability_features must have shape [B, D]")
        if reliability_features.shape[1] < len(SOURCE_RESIDUAL_RELIABILITY_FEATURE_SCHEMA):
            raise ValueError(
                "reliability_features does not include source_manifold_distance_bounded"
            )
        features = reliability_features.to(device=z.device, dtype=z.dtype)
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        distance_index = SOURCE_RESIDUAL_RELIABILITY_FEATURE_SCHEMA.index(
            SOURCE_MANIFOLD_DISTANCE_KEY
        )
        bounded_distance = features[:, distance_index : distance_index + 1].clamp(0.0, 1.0)
        strength = (
            self.hyper_source_manifold_guard_strength
            if self.hyper_source_manifold_guard
            else self.hyper_prompt_manifold_reliability_strength
        )
        return (1.0 - strength * bounded_distance).clamp(
            self.hyper_source_manifold_guard_min_multiplier,
            1.0,
        )

    def prompt_manifold_reliability_multiplier(
        self,
        z: torch.Tensor,
        reliability_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Backward-compatible alias for the source-manifold guard multiplier."""
        return self.source_manifold_guard_multiplier(z, reliability_features)

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
            if self.phys_operator_residual is not None:
                modules.append(("phys_operator_residual", self.phys_operator_residual))
            if self.formula_phys_context_encoder is not None:
                modules.append(("formula_phys_context_encoder", self.formula_phys_context_encoder))
        if self.phys_gain_basis_residual is not None:
            modules.append(("phys_gain_basis_residual", self.phys_gain_basis_residual))
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
        source_trust_bank: dict[str, object] | None = None,
        source_trust_query: torch.Tensor | None = None,
        z_phys: torch.Tensor | None = None,
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
        reliability_multiplier = self.source_manifold_guard_multiplier(
            z,
            reliability_features,
        )
        if self.hyper_phys_agreement_guard:
            reliability_multiplier = reliability_multiplier * self.phys_agreement_guard_multiplier(
                z,
                source_trust_bank=source_trust_bank,
                source_trust_query=source_trust_query,
            ).to(device=reliability_multiplier.device, dtype=reliability_multiplier.dtype)
        if self.hyper_enable_adapters:
            gate_b = self.adapter_reliability_gate(z, "bottleneck") * reliability_multiplier
            b = self.hyper_adapter_b(
                b,
                z,
                logit_residual=self._target_residual("target_adapter_coefficient_residual_b"),
                coeff_logits=self.adapter_coefficient_logits(
                    z,
                    "bottleneck",
                    source_trust_bank=source_trust_bank,
                    source_trust_query=source_trust_query,
                    z_phys=z_phys,
                ),
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
                coeff_logits=self.adapter_coefficient_logits(
                    z,
                    "dec2",
                    source_trust_bank=source_trust_bank,
                    source_trust_query=source_trust_query,
                    z_phys=z_phys,
                ),
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
                coeff_logits=self.adapter_coefficient_logits(
                    z,
                    "dec1",
                    source_trust_bank=source_trust_bank,
                    source_trust_query=source_trust_query,
                    z_phys=z_phys,
                ),
                residual_gate=gate_d1,
            )
        return d1

    def variable_trust_gate_tensor(
        self,
        variable_trust_gate: torch.Tensor | None,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return per-variable output residual gate with shape [B, 2, 1, 1]."""
        if variable_trust_gate is None or not self.hyper_source_trust_variable_gate:
            gate = torch.ones(batch_size, 2, dtype=dtype, device=device)
        else:
            gate = variable_trust_gate.to(device=device, dtype=dtype)
            if gate.ndim == 1:
                if gate.shape[0] != 2:
                    raise ValueError("variable_trust_gate vector must have length 2")
                gate = gate.view(1, 2).expand(batch_size, -1)
            if gate.shape != (batch_size, 2):
                raise ValueError("variable_trust_gate must have shape [B, 2] or [2]")
            gate = gate.clamp(0.0, 1.0)
        detached = gate.detach().cpu()
        names = ("surface", "rootzone")
        self.last_variable_trust_gate_summary = {
            name: {
                "mean": float(detached[:, idx].mean().item()),
                "min": float(detached[:, idx].min().item()),
                "max": float(detached[:, idx].max().item()),
            }
            for idx, name in enumerate(names)
        }
        return gate.view(batch_size, 2, 1, 1)

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

    def _apply_phys_gain_basis_residual(
        self,
        y: torch.Tensor,
        *,
        z: torch.Tensor,
        x_raw: torch.Tensor | None,
        phys_gain_basis_maps: torch.Tensor | None,
        phys_gain_bank: dict[str, object] | None,
    ) -> torch.Tensor:
        if not self.hyper_phys_gain_basis_residual:
            self.last_phys_gain_basis_summary = {"enabled": False}
            return y
        if self.phys_gain_basis_residual is None:
            raise RuntimeError("phys gain basis residual module is not initialized")
        basis_maps = phys_gain_basis_maps
        if basis_maps is None:
            if x_raw is None:
                raise ValueError("hyper_phys_gain_basis_residual requires x_raw or phys_gain_basis_maps")
            basis_maps, _ = phys_gain_basis_from_raw_tensor(x_raw, return_summary=False)
        if basis_maps.ndim == 3:
            basis_maps = basis_maps.unsqueeze(0)
        residual = self.phys_gain_basis_residual(
            z,
            basis_maps.to(device=y.device, dtype=y.dtype),
            source_gain_bank=phys_gain_bank,
        )
        self.last_phys_gain_basis_summary = dict(self.phys_gain_basis_residual.last_summary)
        return y + residual.to(device=y.device, dtype=y.dtype)

    def forward(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        month: Optional[torch.Tensor] = None,
        x_raw: Optional[torch.Tensor] = None,
        reliability_features: Optional[torch.Tensor] = None,
        rho: Optional[float] = None,
        source_trust_bank: Optional[dict[str, object]] = None,
        source_trust_query: Optional[torch.Tensor] = None,
        z_phys: Optional[torch.Tensor] = None,
        variable_trust_gate: Optional[torch.Tensor] = None,
        phys_gain_basis_maps: Optional[torch.Tensor] = None,
        phys_gain_bank: Optional[dict[str, object]] = None,
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
            conditional_features = self._conditional_features(
                x,
                z,
                reliability_features,
                source_trust_bank=source_trust_bank,
                source_trust_query=source_trust_query,
                z_phys=z_phys,
            )
            delta = self.residual_head(conditional_features)
            gate = self.source_residual_reliability_gate(z, reliability_features).view(-1, 1, 1, 1)
            if self.hyper_source_manifold_guard:
                guard = self.source_manifold_guard_multiplier(z, reliability_features).view(-1, 1, 1, 1)
            else:
                guard = torch.ones(z.shape[0], 1, 1, 1, dtype=delta.dtype, device=delta.device)
            if self.hyper_phys_agreement_guard:
                guard = guard * self.phys_agreement_guard_multiplier(
                    z,
                    source_trust_bank=source_trust_bank,
                    source_trust_query=source_trust_query,
                ).view(-1, 1, 1, 1).to(dtype=delta.dtype, device=delta.device)
            resolved_rho = self.source_residual_rho if rho is None else float(rho)
            if not 0.0 <= resolved_rho <= 1.0:
                raise ValueError("rho must be in [0, 1]")
            variable_gate = self.variable_trust_gate_tensor(
                variable_trust_gate,
                batch_size=z.shape[0],
                dtype=delta.dtype,
                device=delta.device,
            )
            y = source_base + resolved_rho * variable_gate * gate.to(dtype=delta.dtype, device=delta.device) * guard.to(
                dtype=delta.dtype,
                device=delta.device,
            ) * delta
            y = self._apply_phys_gain_basis_residual(
                y,
                z=z,
                x_raw=x_raw,
                phys_gain_basis_maps=phys_gain_basis_maps,
                phys_gain_bank=phys_gain_bank,
            )
            return self._apply_target_output_adaptation(y, month=month, x=x, x_raw=x_raw)

        d1 = self._conditional_features(
            x,
            z,
            reliability_features,
            source_trust_bank=source_trust_bank,
            source_trust_query=source_trust_query,
            z_phys=z_phys,
        )
        y = self.head(d1)
        y = self._apply_phys_gain_basis_residual(
            y,
            z=z,
            x_raw=x_raw,
            phys_gain_basis_maps=phys_gain_basis_maps,
            phys_gain_bank=phys_gain_bank,
        )
        return self._apply_target_output_adaptation(y, month=month, x=x, x_raw=x_raw)
