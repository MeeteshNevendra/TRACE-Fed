"""ImageNet-pretrained timm baselines with a common output dictionary."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

BASELINES = {
    "resnet18": "resnet18.a1_in1k",
    "resnet50": "resnet50.a1_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in1k",
}


class TimmClassifier(nn.Module):
    def __init__(self, name: str, num_classes: int, pretrained: bool = True, drop_path: float = 0.1) -> None:
        super().__init__()
        self.arch = BASELINES.get(name, name)
        kwargs = {"drop_path_rate": drop_path} if "efficientnet" not in self.arch else {"drop_path_rate": drop_path}
        self.net = timm.create_model(self.arch, pretrained=pretrained, num_classes=num_classes, **kwargs)
        self.is_vit = "vit" in self.arch
        self.is_swin = "swin" in self.arch
        self._vit_map: torch.Tensor | None = None
        if self.is_vit:
            # Logits depend only on the CLS token, so Grad-CAM uses the patch tokens entering the last
            # block; the block input is rebuilt from that map so the map lies on the gradient path.
            self.net.blocks[-1].register_forward_pre_hook(self._vit_pre_hook)

    def _vit_pre_hook(self, module, args):
        x = args[0]
        p = self.net.num_prefix_tokens
        tokens = x[:, p:]
        side = int(tokens.size(1) ** 0.5)
        fmap = tokens.transpose(1, 2).reshape(x.size(0), -1, side, side)
        self._vit_map = fmap
        return (torch.cat([x[:, :p], fmap.flatten(2).transpose(1, 2)], dim=1),) + tuple(args[1:])

    def forward(self, x: torch.Tensor) -> dict:
        feats = self.net.forward_features(x)
        if self.is_swin:  # (b, h, w, c): route the head through the channel-first map so it carries gradients
            fmap = feats.permute(0, 3, 1, 2)
            feats = fmap.permute(0, 2, 3, 1)
        elif self.is_vit:
            fmap = self._vit_map
        else:
            fmap = feats
        pooled = self.net.forward_head(feats, pre_logits=True)
        return {"logits": self.net.get_classifier()(pooled), "features": pooled, "feature_map": fmap}

    def backbone_parameters(self):
        head = {id(p) for p in self.net.get_classifier().parameters()}
        return [p for p in self.parameters() if id(p) not in head]

    def head_parameters(self):
        return list(self.net.get_classifier().parameters())
