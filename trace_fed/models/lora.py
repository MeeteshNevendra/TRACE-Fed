"""Low-rank adapters for frozen pretrained layers (Linear and 1x1 Conv2d)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scale


class LoRAConv1x1(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        self.lora_down = nn.Conv2d(base.in_channels, rank, 1, stride=base.stride, bias=False)
        self.lora_up = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scale


def inject_lora(module: nn.Module, rank: int, alpha: float, last_n_blocks: int | None = None) -> int:
    """Wrap Linear and 1x1 Conv layers inside `module`; returns number wrapped."""
    targets = []
    for name, child in module.named_modules():
        for cname, sub in child.named_children():
            if isinstance(sub, nn.Linear) or (isinstance(sub, nn.Conv2d) and sub.kernel_size == (1, 1) and sub.groups == 1):
                targets.append((child, cname, sub))
    if last_n_blocks is not None:
        targets = targets[-last_n_blocks:]
    for parent, cname, sub in targets:
        wrapped = LoRALinear(sub, rank, alpha) if isinstance(sub, nn.Linear) else LoRAConv1x1(sub, rank, alpha)
        setattr(parent, cname, wrapped)
    return len(targets)


def is_lora_param(name: str) -> bool:
    return "lora_" in name


def _unused(*_: object) -> None:  # keep F imported for type checkers
    return F.relu
