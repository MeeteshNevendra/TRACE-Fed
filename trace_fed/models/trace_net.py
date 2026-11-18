"""TRACE-Net: the network of TRACE-Fed.

pretrained CNN stages (stride 16) -> 1x1 projection + positional embedding
-> N hybrid blocks, each mixing a 4-direction selective SSM and windowed
self-attention through a learned per-channel gate
-> weak-localisation head (lesion map) -> mask-guided + global pooling
-> classifier.

Every component has a switch so ablations change exactly one thing.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import inject_lora
from .ssm import SelectiveSSM2D


class WindowAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, window: int = 7) -> None:
        super().__init__()
        self.heads, self.window = heads, window
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        b, L, c = x.shape
        h, w = hw
        ws = self.window
        assert h % ws == 0 and w % ws == 0, "feature grid must be divisible by the window"
        x = x.view(b, h // ws, ws, w // ws, ws, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, c)
        q, k, v = self.qkv(x).view(x.size(0), ws * ws, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(-1, ws * ws, c)
        x = self.proj(x).view(b, h // ws, w // ws, ws, ws, c).permute(0, 1, 3, 2, 4, 5).reshape(b, L, c)
        return x


class HybridBlock(nn.Module):
    """mixer in {"hybrid" (gated), "sum" (fixed 0.5/0.5), "ssm", "attn"}."""

    def __init__(self, dim: int, mixer: str = "hybrid", heads: int = 8, window: int = 7, mlp_ratio: float = 2.0,
                 drop_path: float = 0.0) -> None:
        super().__init__()
        self.mixer = mixer
        self.norm1 = nn.LayerNorm(dim)
        self.ssm = SelectiveSSM2D(dim) if mixer in {"hybrid", "sum", "ssm"} else None
        self.attn = WindowAttention(dim, heads, window) if mixer in {"hybrid", "sum", "attn"} else None
        self.gate = nn.Linear(dim, dim) if mixer == "hybrid" else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop_path = drop_path
        self.last_gate_mean: float | None = None

    def _dp(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_path == 0.0:
            return x
        keep = torch.rand(x.size(0), 1, 1, device=x.device) >= self.drop_path
        return x * keep / (1 - self.drop_path)

    def forward(self, x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        h = self.norm1(x)
        if self.mixer == "ssm":
            mixed = self.ssm(h, hw)
        elif self.mixer == "attn":
            mixed = self.attn(h, hw)
        elif self.mixer == "sum":
            mixed = 0.5 * self.ssm(h, hw) + 0.5 * self.attn(h, hw)
        else:
            g = torch.sigmoid(self.gate(h))
            self.last_gate_mean = float(g.detach().float().mean())
            mixed = g * self.ssm(h, hw) + (1 - g) * self.attn(h, hw)
        x = x + self._dp(mixed)
        return x + self._dp(self.mlp(self.norm2(x)))


class TRACENet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "convnext_tiny.fb_in22k_ft_in1k",
        backbone_stage: int | None = None,  # default: the stride-16 stage of any timm backbone
        dim: int = 256,
        depth: int = 2,
        mixer: str = "hybrid",
        pooling: str = "mask",  # "mask" (global + mask-guided) or "gap"
        loc_head: bool = True,
        drop_path: float = 0.1,
        pretrained: bool = True,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        if backbone_stage is None:
            probe = timm.create_model(backbone, pretrained=False, features_only=True)
            backbone_stage = probe.feature_info.reduction().index(16)
            del probe
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=(backbone_stage,))
        in_ch = self.backbone.feature_info.channels()[-1]
        reduction = self.backbone.feature_info.reduction()[-1]
        self.grid = image_size // reduction
        self.proj = nn.Sequential(nn.Conv2d(in_ch, dim, 1), nn.GroupNorm(1, dim))
        self.pos = nn.Parameter(torch.zeros(1, self.grid * self.grid, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [HybridBlock(dim, mixer, drop_path=drop_path * (i + 1) / max(depth, 1)) for i in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.pooling = pooling if loc_head else "gap"
        self.loc = nn.Sequential(nn.Conv2d(dim, dim // 2, 3, padding=1), nn.GELU(), nn.Conv2d(dim // 2, 1, 1)) if loc_head else None
        self.feat_dim = dim * (2 if self.pooling == "mask" else 1)
        self.head_norm = nn.LayerNorm(self.feat_dim)
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.dim = dim

    # ------------------------------------------------------------------ parameter groups
    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        skip = {id(p) for p in self.backbone.parameters()}
        return [p for p in self.parameters() if id(p) not in skip]

    def add_lora(self, rank: int = 8, alpha: float = 16.0) -> int:
        return inject_lora(self.backbone, rank, alpha)

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> dict:
        fmap = self.proj(self.backbone(x)[-1])
        b, c, h, w = fmap.shape
        tokens = fmap.flatten(2).transpose(1, 2)
        if tokens.size(1) == self.pos.size(1):
            tokens = tokens + self.pos
        for blk in self.blocks:
            tokens = blk(tokens, (h, w))
        tokens = self.norm(tokens)
        feat = tokens.transpose(1, 2).reshape(b, c, h, w)
        gap = feat.mean(dim=(2, 3))
        out = {"feature_map": feat}
        if self.loc is not None:
            mask_logits = self.loc(feat)  # (b, 1, h, w)
            out["mask_logits_lowres"] = mask_logits
            out["mask_logits"] = F.interpolate(mask_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if self.pooling == "mask":
            wmap = torch.sigmoid(out["mask_logits_lowres"]).float()
            masked = (feat.float() * wmap).sum(dim=(2, 3)) / (wmap.sum(dim=(2, 3)) + 1e-4)
            pooled = torch.cat([gap.float(), masked], dim=1).to(gap.dtype)
        else:
            pooled = gap
        pooled = self.head_norm(pooled)
        out["features"] = pooled
        out["logits"] = self.classifier(pooled)
        # class-activation map from the global-pooling half of the classifier
        w_gap = self.classifier.weight[:, : self.dim]
        out["cam_all"] = torch.einsum("kc,bchw->bkhw", w_gap.to(feat.dtype), feat)
        return out

    def loc_enabled(self) -> bool:
        return self.loc is not None

    def gate_means(self) -> list[float | None]:
        return [blk.last_gate_mean for blk in self.blocks]
