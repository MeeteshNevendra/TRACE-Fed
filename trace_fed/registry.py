"""Model specs -> modules.

Spec strings: a baseline name (see models/baselines.py) or
"trace[:key=value,...]", e.g. "trace:mixer=ssm,pooling=gap,xc=0".
Keys `xc`, `seg` override the loss weights; `peft=1` freezes the backbone and
adds LoRA adapters (rank `lora_r`). All other keys go to TRACENet.
"""
from __future__ import annotations

import torch.nn as nn

from .models.baselines import BASELINES, TimmClassifier
from .models.trace_net import TRACENet

LOSS_KEYS = {"xc": "xc_weight", "seg": "seg_weight"}
TRAIN_KEYS = {"lr": float, "backbone_lr": float, "epochs": int, "batch_size": int, "label_smoothing": float}


def parse_spec(spec: str) -> tuple[str, dict]:
    name, _, rest = spec.partition(":")
    kv = {}
    for item in filter(None, rest.split(",")):
        k, _, v = item.partition("=")
        kv[k.strip()] = _cast(v.strip())
    return name, kv


def _cast(v: str):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return v


def build(spec: str, num_classes: int, pretrained: bool = True) -> tuple[nn.Module, dict, dict]:
    """Returns (model, loss_overrides, train_overrides)."""
    name, kv = parse_spec(spec)
    loss = {LOSS_KEYS[k]: float(kv.pop(k)) for k in list(kv) if k in LOSS_KEYS}
    train = {k: TRAIN_KEYS[k](kv.pop(k)) for k in list(kv) if k in TRAIN_KEYS}
    if name == "trace":
        peft = bool(kv.pop("peft", 0))
        rank = int(kv.pop("lora_r", 8))
        model = TRACENet(num_classes, pretrained=pretrained, **kv)
        if peft:
            for p in model.backbone.parameters():
                p.requires_grad_(False)
            model.add_lora(rank)
            model.freeze_bn = True
        if not model.loc_enabled():
            loss.setdefault("seg_weight", 0.0)
            loss.setdefault("xc_weight", 0.0)
        return model, loss, train
    if name in BASELINES:
        if kv:
            raise ValueError(f"unexpected options for baseline {name}: {kv}")
        return TimmClassifier(name, num_classes, pretrained=pretrained), {"seg_weight": 0.0, "xc_weight": 0.0}, train
    raise ValueError(f"unknown model spec {spec}")
