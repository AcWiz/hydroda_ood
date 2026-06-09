"""HyperDA v2 FiLM plus basis-adapter conditional UNet."""
from __future__ import annotations

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
        self.hyper_adapter_b = BasisHyperAdapter(
            channels=width * 4,
            prompt_dim=prompt_dim,
            n_basis=self.hyper_n_basis,
            adapter_bottleneck=self.hyper_adapter_bottleneck,
            adapter_scale=self.hyper_adapter_scale,
        )
        self.dec2 = ConvBlock(width * 6, width * 2)
        self.dec1 = ConvBlock(width * 3, width)
        self.hyper_adapter_d2 = BasisHyperAdapter(
            channels=width * 2,
            prompt_dim=prompt_dim,
            n_basis=self.hyper_n_basis,
            adapter_bottleneck=max(4, self.hyper_adapter_bottleneck // 2),
            adapter_scale=self.hyper_adapter_scale,
        )
        self.hyper_adapter_d1 = BasisHyperAdapter(
            channels=width,
            prompt_dim=prompt_dim,
            n_basis=self.hyper_n_basis,
            adapter_bottleneck=max(4, self.hyper_adapter_bottleneck // 4),
            adapter_scale=self.hyper_adapter_scale,
        )
        self.hyper_adapter = self.hyper_adapter_b
        self.head = nn.Conv2d(width, out_channels, 1)
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

    def forward(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        month: Optional[torch.Tensor] = None,
        x_raw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if z is None:
            raise ValueError("HyperAdapterConditionalResUNet requires a prompt tensor z")
        if self.enable_target_adaptation:
            if month is None:
                raise ValueError("target adaptation residual gain requires month tensor")
            if self.target_prompt is not None:
                z = self.target_prompt(z)

        e1 = self.enc1(x)
        e1 = self.film1(e1, z)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e2 = self.film2(e2, z)
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        e3 = self.film3(e3, z)

        b = self.bottleneck(e3)
        b = self.film_b(b, z)
        b = self.hyper_adapter_b(
            b,
            z,
            logit_residual=self._target_residual("target_adapter_coefficient_residual_b"),
        )

        d2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d2 = self.hyper_adapter_d2(
            d2,
            z,
            logit_residual=self._target_residual("target_adapter_coefficient_residual_d2"),
        )
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d1 = self.hyper_adapter_d1(
            d1,
            z,
            logit_residual=self._target_residual("target_adapter_coefficient_residual_d1"),
        )
        y = self.head(d1)
        if self.enable_target_adaptation and self.residual_gain is not None:
            y = self.residual_gain(y, month)
        if self.enable_target_adaptation and self.target_spatial_refine is not None:
            refine_x = x_raw if self.target_spatial_refine_input == "raw" and x_raw is not None else x
            if self.target_spatial_refine_type in {"hydro_msr_gain", "hydro_msr_gain_lite"}:
                y = y + self.target_spatial_refine(refine_x, y, x_raw=x_raw, month=month)
            else:
                y = y + self.target_spatial_refine(refine_x, y, x_raw=x_raw)
        return y
