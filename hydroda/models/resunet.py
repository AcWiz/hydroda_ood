"""Small ResUNet backbone for neural DA increment emulation."""
from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from hydroda.models.mixstyle import MixStyle2d


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class SmallResUNet(nn.Module):
    """Compact UNet-style dense predictor.

    Output channels are [ΔSM_surface, ΔSM_rootzone].
    """

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 2,
        width: int = 32,
        zero_raw_increment_init: bool = False,
        dg_method: str = "none",
        mixstyle_p: float = 0.5,
        mixstyle_alpha: float = 0.1,
        mixstyle_layers: Iterable[str] | str = ("enc1", "enc2"),
    ) -> None:
        super().__init__()
        self.dg_method = str(dg_method or "none")
        if isinstance(mixstyle_layers, str):
            self.mixstyle_layers = {layer.strip() for layer in mixstyle_layers.split(",") if layer.strip()}
        else:
            self.mixstyle_layers = {str(layer) for layer in mixstyle_layers}
        self.enc1 = ConvBlock(in_channels, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.enc3 = ConvBlock(width * 2, width * 4)
        self.bottleneck = ConvBlock(width * 4, width * 4)
        self.dec2 = ConvBlock(width * 6, width * 2)
        self.dec1 = ConvBlock(width * 3, width)
        self.head = nn.Conv2d(width, out_channels, 1)
        if self.dg_method == "mixstyle":
            self.mixstyle_enc1 = MixStyle2d(p=mixstyle_p, alpha=mixstyle_alpha)
            self.mixstyle_enc2 = MixStyle2d(p=mixstyle_p, alpha=mixstyle_alpha)
        else:
            self.mixstyle_enc1 = None
            self.mixstyle_enc2 = None
        self._zero_raw_increment_init = zero_raw_increment_init
        if zero_raw_increment_init:
            # Weight zero — bias will be set by Trainer when increment stats are available
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def _maybe_mixstyle(self, layer_name: str, x: torch.Tensor) -> torch.Tensor:
        if self.dg_method != "mixstyle" or layer_name not in self.mixstyle_layers:
            return x
        if layer_name == "enc1" and self.mixstyle_enc1 is not None:
            return self.mixstyle_enc1(x)
        if layer_name == "enc2" and self.mixstyle_enc2 is not None:
            return self.mixstyle_enc2(x)
        return x

    def forward_features(self, x: torch.Tensor, return_layer: str = "bottleneck") -> torch.Tensor:
        """Return an intermediate feature map without changing forward semantics."""
        e1 = self.enc1(x)
        e1 = self._maybe_mixstyle("enc1", e1)
        if return_layer == "enc1":
            return e1
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e2 = self._maybe_mixstyle("enc2", e2)
        if return_layer == "enc2":
            return e2
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        if return_layer == "enc3":
            return e3
        b = self.bottleneck(e3)
        if return_layer == "bottleneck":
            return b
        raise ValueError(f"Unsupported return_layer={return_layer!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e1 = self._maybe_mixstyle("enc1", e1)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e2 = self._maybe_mixstyle("enc2", e2)
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        b = self.bottleneck(e3)
        d2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)
