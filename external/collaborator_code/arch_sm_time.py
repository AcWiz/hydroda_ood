# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial, lru_cache
import numpy as np
import torch
import torch.nn as nn
from timm.layers import DropPath
from timm.models.vision_transformer import trunc_normal_
import collections.abc
from einops import rearrange
import torch.nn.functional as F


def exists(val):
    return val is not None


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


class PeriodicConv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert max(self.padding) > 0

    def forward(self, x):
        x = F.pad(x, (self.padding[1], self.padding[1], 0, 0), mode="circular")
        x = F.pad(x, (0, 0, self.padding[0], self.padding[0]), mode="constant", value=0)
        x = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            0,
            self.dilation,
            self.groups,
        )
        return x


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


class GeGLUFFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        inner_dim = int(hidden_features * (2 / 3))
        self.fc1 = nn.Linear(in_features, inner_dim * 2, bias=False)
        self.act = GEGLU()
        self.fc2 = nn.Linear(inner_dim, out_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(
        B,
        H // window_size[0],
        window_size[0],
        W // window_size[1],
        window_size[1],
        C,
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size[0], window_size[1], C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(
        B,
        H // window_size[0],
        W // window_size[1],
        window_size[0],
        window_size[1],
        -1,
    )
    x = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(B, H, W, -1)
    )
    return x


class WindowAttentionV2(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        pretrained_window_size=[0, 0],
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))),
            requires_grad=True,
        )

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        # get relative_coords_table
        relative_coords_h = torch.arange(
            -(self.window_size[0] - 1),
            self.window_size[0],
            dtype=torch.float32,
        )
        relative_coords_w = torch.arange(
            -(self.window_size[1] - 1),
            self.window_size[1],
            dtype=torch.float32,
        )
        relative_coords_table = torch.stack(
            torch.meshgrid(
                [relative_coords_h, relative_coords_w],
                indexing="ij",
            )
        ).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (
                pretrained_window_size[0] - 1
            )
            relative_coords_table[:, :, :, 1] /= (
                pretrained_window_size[1] - 1
            )
        else:
            relative_coords_table[:, :, :, 0] /= (
                self.window_size[0] - 1
            )
            relative_coords_table[:, :, :, 1] /= (
                self.window_size[1] - 1
            )
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            torch.sign(relative_coords_table)
            * torch.log2(torch.abs(relative_coords_table) + 1.0)
            / np.log2(8)
        )

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None]
            - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = (
            relative_coords.permute(1, 2, 0)
            .contiguous()
        )  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer(
            "relative_position_index",
            relative_position_index,
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows*B, N, C)
            mask: (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(
                        self.v_bias,
                        requires_grad=False,
                    ),
                    self.v_bias,
                )
            )
        qkv = F.linear(
            input=x,
            weight=self.qkv.weight,
            bias=qkv_bias,
        )
        qkv = (
            qkv.reshape(B_, N, 3, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(
            k, dim=-1
        ).transpose(-2, -1)
        logit_scale = torch.clamp(
            self.logit_scale,
            max=torch.log(
                torch.tensor(1.0 / 0.01)
            ).to(self.logit_scale.device),
        ).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(
            self.relative_coords_table.to(x)
        ).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = (
            relative_position_bias.permute(2, 0, 1)
            .contiguous()
        )  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = (
            16 * torch.sigmoid(relative_position_bias)
        )
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = mask.to(x)
            attn = (
                attn.view(B_ // nW, nW, self.num_heads, N, N)
                + mask.unsqueeze(1).unsqueeze(0)
            )
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (
            (attn @ v)
            .transpose(1, 2)
            .reshape(B_, N, C)
        )
        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x


class SwinBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        input_size,
        window_size=7,
        shift_size=0,
        mask_type="h",
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        drop_path=0.0,
        attn_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_size = input_size
        self.num_heads = num_heads
        self.window_size = list(to_2tuple(window_size))
        self.shift_size = list(to_2tuple(shift_size))
        self.mlp_ratio = mlp_ratio

        if self.input_size[0] <= self.window_size[0]:
            self.shift_size[0] = 0
            self.window_size[0] = self.input_size[0]

        if self.input_size[1] <= self.window_size[1]:
            self.shift_size[1] = 0
            self.window_size[1] = self.input_size[1]

        assert (
            0 <= self.shift_size[0] < self.window_size[0]
        ), "shift_size must in 0-window_size"
        assert (
            0 <= self.shift_size[1] < self.window_size[1]
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttentionV2(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = GeGLUFFN(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if max(self.shift_size) > 0:
            H, W = self.input_size
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size[0]),
                slice(-self.window_size[0], -self.shift_size[0]),
                slice(-self.shift_size[0], None),
            )
            w_slices = (
                slice(0, -self.window_size[1]),
                slice(-self.window_size[1], -self.shift_size[1]),
                slice(-self.shift_size[1], None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    if mask_type == "h":
                        img_mask[:, h, :, :] = cnt
                    elif mask_type == "w":
                        img_mask[:, :, w, :] = cnt
                    else:
                        img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask,
                self.window_size,
            )
            mask_windows = mask_windows.view(
                -1,
                self.window_size[0] * self.window_size[1],
            )
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0,
                float(-100.0),
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def swin_attn(self, x):
        H, W = self.input_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        if max(self.shift_size) > 0:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size[0], -self.shift_size[1]),
                dims=(1, 2),
            )
        else:
            shifted_x = x

        x_windows = window_partition(
            shifted_x,
            self.window_size,
        )
        x_windows = x_windows.view(
            -1,
            self.window_size[0] * self.window_size[1],
            C,
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(
            -1,
            self.window_size[0],
            self.window_size[1],
            C,
        )
        shifted_x = window_reverse(
            attn_windows,
            self.window_size,
            H,
            W,
        )

        if max(self.shift_size) > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size[0], self.shift_size[1]),
                dims=(1, 2),
            )
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        return x

    def forward(self, x):
        # 原本就没有时间调制，这里保持不变
        x = x + self.drop_path1(self.swin_attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


norm_layer = partial(nn.LayerNorm, eps=1e-6)


class SwinLayer(nn.Module):
    def __init__(
        self,
        in_chans,
        embed_dim,
        input_size,
        window_size,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        attn_drop=0.0,
    ):
        super().__init__()
        self.depth = depth
        self.input_size = input_size
        self.blocks = nn.ModuleList()

        for i in range(depth):
            blk = SwinBlock(
                dim=embed_dim,
                input_size=input_size,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
            )
            self.blocks.append(blk)

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, h):
        for blk in self.blocks:
            h = blk(h)
        return h


# ======================================================
#            时间编码 → Token 级季节性调制
# ======================================================

class TimeFiLMToken(nn.Module):
    """
    使用时间编码 [sin(doy), cos(doy)] 对 token 特征做 FiLM：
        h' = (1 + gamma(t)) * h + beta(t)
    其中 h: [B, L, C], gamma/beta: [B, C]
    """
    def __init__(self, time_dim=2, hidden=128, feat_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * feat_dim),
        )

    def forward(self, time_code, h):
        """
        time_code: [B, time_dim]
        h        : [B, L, C]
        """
        B, L, C = h.shape
        gamma_beta = self.mlp(time_code)  # [B, 2*C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, C], [B, C]

        gamma = gamma.unsqueeze(1)  # [B, 1, C]
        beta = beta.unsqueeze(1)    # [B, 1, C]

        h_mod = h * (1.0 + gamma) + beta
        return h_mod


class Sformer(nn.Module):
    def __init__(
        self,
        img_size=[128, 256],
        window_size=8,
        patch_size=4,
        num_vars=9,   # 9 input variables
        embed_dim=768,
        num_heads=16,
        depths=[4, 4, 4, 4],
        mlp_ratio=4,
        drop_path=0.2,
        drop_rate=0.2,
        attn_drop=0.0,
        const_dir="../../data/train_pred",
    ):
        super().__init__()
        # Define default variables (9 input variables, only 1 output variable)
        self.default_vars = [
            "Wind_f_inst",
            "Rainf_f_tavg",
            "Tair_f_inst",
            "Qair_f_inst",
            "Psurf_f_inst",
            "SWdown_f_tavg",
            "LWdown_f_tavg",
            "SoilTMP0_10cm_inst",
            "SoilMoi0_10cm_inst",
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        norm_layer_local = partial(nn.LayerNorm, eps=1e-6)
        self.c = 2  # Number of output variables
        self.h = self.img_size[0] // patch_size
        self.w = self.img_size[1] // patch_size
        self.feat_size = [sz // patch_size for sz in img_size]
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.num_vars = num_vars  # 9 input variables

        # Variable tokenization: separate embedding layer for each variable
        self.var_map = self.create_var_map()

        # 9 separate patch embedding layers (每个变量单独一个 Conv2d)
        for i in range(1, 10):
            setattr(
                self,
                f"patch_embed_{i}",
                nn.Conv2d(
                    in_channels=1,
                    out_channels=embed_dim,
                    kernel_size=patch_size,
                    stride=patch_size,
                ),
            )

        self.num_patches = self.h * self.w
        self.in_norm = norm_layer_local(
            self.num_vars * embed_dim,
            eps=1e-6,
        )

        # Swin Transformer layers
        layers = []
        input_size = [sz // patch_size for sz in self.img_size]
        for i in range(self.num_layers):
            layer = SwinLayer(
                in_chans=self.num_vars * embed_dim,
                embed_dim=self.num_vars * embed_dim,
                input_size=input_size,
                window_size=window_size,
                depth=depths[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=drop_path,
                attn_drop=attn_drop,
            )
            layers.append(layer)
            self.add_module(
                f"norm{i}",
                norm_layer_local(self.num_vars * embed_dim, eps=1e-6),
            )

        self.layers = nn.ModuleList(layers)
        self.fpn = nn.Sequential(
            nn.Linear(self.num_vars * embed_dim * self.num_layers, self.num_vars * embed_dim),
            nn.GELU(),
        )
        self.out_norm = nn.LayerNorm(self.num_vars * embed_dim)

        self.head = nn.Linear(
            self.num_vars * embed_dim,
            patch_size**2 * self.c,
        )

        # ====== 时间调制模块：使用 sin/cos(doy) ======
        self.time_film = TimeFiLMToken(
            time_dim=2,
            hidden=128,
            feat_dim=self.num_vars * embed_dim,
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights for the num_vars patch embedding layers
        for i in range(1, 10):
            embed_layer = getattr(self, f"patch_embed_{i}")
            w = embed_layer.weight.data
            trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_map(self):
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def unpatchify(self, x: torch.Tensor):
        """
        x: (B, L, patch_size**2 * c)
        return imgs: (B, c, H, W)
        """
        x = x.reshape(
            shape=(x.shape[0], self.h, self.w, self.patch_size, self.patch_size, self.c)
        )
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(
            shape=(x.shape[0], self.c, self.h * self.patch_size, self.w * self.patch_size)
        )
        return imgs

    def forward_encoder(self, x: torch.Tensor, time_code=None, noise=None):
        """
        x: [B, num_vars, H, W]  (只包含 9 个物理变量，不含 mask)
        time_code: [B, 2], sin/cos(doy)
        """
        # Split input into num_vars variables
        var1, var2, var3, var4, var5, var6, var7, var8, var9 = x.split(1, dim=1)

        # Separate patch embeddings
        in_1 = self.patch_embed_1(var1)
        in_2 = self.patch_embed_2(var2)
        in_3 = self.patch_embed_3(var3)
        in_4 = self.patch_embed_4(var4)
        in_5 = self.patch_embed_5(var5)
        in_6 = self.patch_embed_6(var6)
        in_7 = self.patch_embed_7(var7)
        in_8 = self.patch_embed_8(var8)
        in_9 = self.patch_embed_9(var9)

        # Concatenate embeddings
        h = torch.cat(
            [in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9],
            dim=1,
        )  # [B, num_vars*embed_dim, H/p, W/p]
        h = h.flatten(2).transpose(1, 2)  # [B, L, C]
        h = self.in_norm(h)

        # Add noise if provided
        if exists(noise):
            noise = F.interpolate(
                noise,
                size=self.feat_size,
                mode="bilinear",
                align_corners=False,
            )
            noise = rearrange(noise, "n c h w -> n (h w) c")
            h = h + noise.to(h)

        # ===== Seasonally modulate tokens by time_code =====
        if time_code is not None:
            h = self.time_film(time_code, h)

        outs = []
        for i, blk in enumerate(self.layers):
            h = blk(h)
            out = getattr(self, f"norm{i}")(h)
            outs.append(out)
        h = self.fpn(torch.cat(outs, dim=-1))
        h = self.out_norm(h)

        return h

    def forward(self, x, time_code=None, noise=None):
        """
        x: [B, 10, H, W]
           前 9 个通道是物理变量，第 10 个通道是 mask
        time_code: [B, 2]  (sin(doy), cos(doy))
        """
        mask = x[:, -1:]
        x = x[:, :-1]  # Remove the mask variable, keep 9 vars

        out_transformers = self.forward_encoder(
            x,
            time_code=time_code,
            noise=noise,
        )  # [B, L, num_vars*embed_dim]
        params = self.head(out_transformers)  # [B, L, p*p*c]
        params = self.unpatchify(params)      # [B, c, H, W]
        params *= mask                        # Apply mask

        return params


class Sformer_phy(nn.Module):
    def __init__(
        self,
        img_size=[128, 256],
        window_size=8,
        patch_size=4,
        num_vars=9,  # 9 input variables
        embed_dim=768,
        num_heads=16,
        depths=[4, 4, 4, 4],
        mlp_ratio=4,
        drop_path=0.2,
        drop_rate=0.2,
        attn_drop=0.0,
    ):
        super().__init__()

        self.model = Sformer(
            img_size=img_size,
            window_size=window_size,
            patch_size=patch_size,
            num_vars=num_vars,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depths=depths,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            drop_rate=drop_rate,
            attn_drop=attn_drop,
        )
        self.ssm_mean = 0.13334474
        self.ssm_std = 0.06109
        self.rsm_mean = 0.18498646
        self.rsm_std = 0.056409765

    def denormalize_params(self, data, mean, std):
        return data * std + mean

    def normalize_params(self, data, mean, std):
        return (data - mean) / std

    def latent2et(self, latent):
        # Convert latent variable to ET
        et = latent / 2.45e6  # Scale factor for ET
        return et

    def forward(self, x, time_code=None, noise=None):
        """
        x        : [B, 10, H, W]  (9 vars + 1 mask)
        time_code: [B, 2] (sin(doy), cos(doy))
        """
        params = self.model(x, time_code=time_code, noise=noise)
        sm = x[:, 7:9]   # background SM (表层+根区)
        params += sm
        
        if not self.training:
        # 通道 0 → 限制在 [0, 1]
            params[:, 0, :, :] = torch.clamp(
                params[:, 0, :, :],
                (0.0 - self.ssm_mean) / self.ssm_std,
                (1.0 - self.ssm_mean) / self.ssm_std,
            )
            # 通道 1 → 限制在 [0, 2]
            params[:, 1, :, :] = torch.clamp(
                params[:, 1, :, :],
                (0.0 - self.rsm_mean) / self.rsm_std,
                (1.0 - self.rsm_mean) / self.rsm_std,
            )

        return params