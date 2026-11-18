"""Training, evaluation and metrics shared by centralized and federated runs."""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import Split, flip_batch, prepare_batch


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 32
    lr: float = 1e-3  # new layers / heads
    backbone_lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    label_smoothing: float = 0.1
    patience: int = 8
    seg_weight: float = 0.5
    xc_weight: float = 0.1  # explanation-consistency loss
    image_size: int = 224
    amp: bool = True
    grad_clip: float = 1.0
    balanced_sampling: bool = False
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- losses
def seg_loss(out: dict, batch: dict) -> torch.Tensor:
    """BCE + soft Dice at feature resolution, only on images with a known mask."""
    if "mask_logits_lowres" not in out:
        return out["logits"].new_zeros(())
    has = batch["has_mask"] > 0
    if not bool(has.any()):
        return out["logits"].new_zeros(())
    logits = out["mask_logits_lowres"][has].float()
    target = F.adaptive_avg_pool2d(batch["mask"][has], logits.shape[-2:]).clamp(0, 1)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    nonempty = target.flatten(1).sum(1) > 0
    if bool(nonempty.any()):
        p = torch.sigmoid(logits[nonempty]).flatten(1)
        t = target[nonempty].flatten(1)
        dice = 1 - ((2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)).mean()
    else:
        dice = logits.new_zeros(())
    return bce + dice


def xc_loss(out: dict, batch: dict, lesion_classes: torch.Tensor | None) -> torch.Tensor:
    """Share of positive class evidence (CAM of the true class) falling outside the predicted lesion map.

    Applied only to lesion-bearing images; the lesion map is detached so the
    loss moves the class evidence, not the map.
    """
    if "mask_logits_lowres" not in out or lesion_classes is None:
        return out["logits"].new_zeros(())
    y = batch["label"]
    lesion = lesion_classes[y]
    if not bool(lesion.any()):
        return out["logits"].new_zeros(())
    cam = out["cam_all"][lesion, y[lesion]].float().relu()  # (n, h, w)
    wmap = torch.sigmoid(out["mask_logits_lowres"][lesion, 0]).float().detach()
    inside = (cam * wmap).flatten(1).sum(1)
    total = cam.flatten(1).sum(1) + 1e-6
    return (1 - inside / total).mean()


# --------------------------------------------------------------------------- optimisation
def make_optimizer(model: nn.Module, cfg: TrainConfig, trainable_only: bool = True) -> torch.optim.Optimizer:
    bb = {id(p) for p in model.backbone_parameters()} if hasattr(model, "backbone_parameters") else set()
    groups = {"bb_decay": [], "bb_no": [], "hd_decay": [], "hd_no": []}
    for name, p in model.named_parameters():
        if trainable_only and not p.requires_grad:
            continue
        is_bb = id(p) in bb and "lora_" not in name
        no_decay = p.ndim <= 1 or name.endswith(".bias") or "pos" in name.split(".")[-1] or "A_log" in name or name.endswith(".D") or "dt_bias" in name
        groups[("bb_" if is_bb else "hd_") + ("no" if no_decay else "decay")].append(p)
    param_groups = []
    for key, params in groups.items():
        if params:
            param_groups.append(
                {
                    "params": params,
                    "lr": cfg.backbone_lr if key.startswith("bb") else cfg.lr,
                    "weight_decay": 0.0 if key.endswith("no") else cfg.weight_decay,
                }
            )
    for g in param_groups:
        g["base_lr"] = g["lr"]
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))


def set_lr(opt: torch.optim.Optimizer, progress: float, warmup_frac: float) -> None:
    """Linear warm-up then cosine decay; progress in [0, 1]."""
    if progress < warmup_frac:
        factor = progress / max(warmup_frac, 1e-8)
    else:
        factor = 0.5 * (1 + math.cos(math.pi * (progress - warmup_frac) / max(1 - warmup_frac, 1e-8)))
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * max(factor, 0.01)


def lesion_classes_for(split: Split) -> torch.Tensor:
    names = [n.lower() for n in split.class_names]
    return torch.tensor([n not in {"no tumor", "normal"} for n in names], device=split.idx.device)


def class_weights(split: Split) -> torch.Tensor:
    counts = torch.bincount(split.labels, minlength=split.num_classes).float()
    w = counts.sum() / (counts.clamp(min=1) * split.num_classes)
    return w / w.mean()


def train_steps(
    model: nn.Module,
    split: Split,
    opt: torch.optim.Optimizer,
    cfg: TrainConfig,
    generator: torch.Generator,
    progress_fn: Callable[[int], float] | None = None,
    extra_loss: Callable[[dict, dict, nn.Module], torch.Tensor] | None = None,
    post_step: Callable[[nn.Module, torch.optim.Optimizer], None] | None = None,
) -> dict:
    """One pass over `split`. Returns mean losses and number of optimiser steps."""
    model.train()
    if getattr(model, "freeze_bn", False):  # frozen (PEFT) backbone keeps its pretrained BN statistics
        for m in model.backbone.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
    lesion = lesion_classes_for(split)
    weight = class_weights(split) if cfg.balanced_sampling else None
    sums = {"loss": 0.0, "cls": 0.0, "seg": 0.0, "xc": 0.0, "extra": 0.0}
    steps = examples = 0
    for step, raw in enumerate(split.batches(cfg.batch_size, shuffle=True, generator=generator, drop_last=True)):
        if progress_fn is not None:
            set_lr(opt, progress_fn(step), 0.0 if cfg.warmup_epochs == 0 else cfg.extra.get("warmup_frac", 0.05))
        batch = prepare_batch(raw, cfg.image_size, train=True, generator=generator)
        with torch.autocast(device_type=batch["image"].device.type, dtype=torch.bfloat16, enabled=cfg.amp):
            out = model(batch["image"])
            cls = F.cross_entropy(out["logits"].float(), batch["label"], weight=weight, label_smoothing=cfg.label_smoothing)
            seg = seg_loss(out, batch) if cfg.seg_weight > 0 else cls.new_zeros(())
            xc = xc_loss(out, batch, lesion) if cfg.xc_weight > 0 else cls.new_zeros(())
            ext = extra_loss(out, batch, model) if extra_loss is not None else cls.new_zeros(())
            loss = cls + cfg.seg_weight * seg + cfg.xc_weight * xc + ext
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
        opt.step()
        if post_step is not None:
            post_step(model, opt)
        for k, v in zip(sums, [loss, cls, seg, xc, ext]):
            sums[k] += float(v.detach())
        steps += 1
        examples += int(batch["label"].numel())  # images in this step (shape only; no computation)
    return {k: v / max(steps, 1) for k, v in sums.items()} | {"steps": steps, "examples": examples}


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def predict(model: nn.Module, split: Split, cfg: TrainConfig, with_maps: bool = False, tta: bool = False) -> dict:
    model.eval()
    probs, labels, uids, dices, ious, stab = [], [], [], [], [], []
    lesion = lesion_classes_for(split)
    for raw in split.batches(64, shuffle=False):
        batch = prepare_batch(raw, cfg.image_size, train=False)
        with torch.autocast(device_type=batch["image"].device.type, dtype=torch.bfloat16, enabled=cfg.amp):
            out = model(batch["image"])
            logits = out["logits"].float()
            if tta:
                logits = 0.5 * (logits + model(flip_batch(batch["image"]))["logits"].float())
        probs.append(torch.softmax(logits, 1).cpu())
        labels.append(batch["label"].cpu())
        uids.append(batch["uid"].cpu())
        if with_maps and "mask_logits" in out:
            sel = (batch["has_mask"] > 0) & lesion[batch["label"]]
            if bool(sel.any()):
                pred = (out["mask_logits"][sel].float() > 0).float()
                gt = batch["mask"][sel]
                inter = (pred * gt).flatten(1).sum(1)
                dices.append(((2 * inter + 1e-6) / (pred.flatten(1).sum(1) + gt.flatten(1).sum(1) + 1e-6)).cpu())
                ious.append(((inter + 1e-6) / ((pred + gt).clamp(max=1).flatten(1).sum(1) + 1e-6)).cpu())
    res = {"probs": torch.cat(probs).numpy(), "labels": torch.cat(labels).numpy(), "uids": torch.cat(uids).numpy()}
    if dices:
        res["dice"] = float(torch.cat(dices).mean())
        res["iou"] = float(torch.cat(ious).mean())
    return res


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def compute_metrics(pred: dict, num_classes: int) -> dict:
    probs, labels = pred["probs"], pred["labels"]
    y_hat = probs.argmax(1)
    m = {
        "n": int(len(labels)),
        "accuracy": accuracy_score(labels, y_hat),
        "balanced_accuracy": balanced_accuracy_score(labels, y_hat),
        "precision": precision_score(labels, y_hat, average="macro", zero_division=0),
        "recall": recall_score(labels, y_hat, average="macro", zero_division=0),
        "f1": f1_score(labels, y_hat, average="macro", zero_division=0),
        "ece": expected_calibration_error(probs, labels),
        "nll": float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
    }
    try:
        if num_classes == 2:
            m["auc"] = roc_auc_score(labels, probs[:, 1])
        else:
            m["auc"] = roc_auc_score(labels, probs, multi_class="ovr", average="macro", labels=list(range(num_classes)))
    except ValueError:
        m["auc"] = float("nan")
    m["confusion"] = confusion_matrix(labels, y_hat, labels=list(range(num_classes))).tolist()
    for k in ("dice", "iou"):
        if k in pred:
            m[k] = pred[k]
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}


def evaluate(model: nn.Module, split: Split, cfg: TrainConfig, with_maps: bool = False, tta: bool = False) -> tuple[dict, dict]:
    pred = predict(model, split, cfg, with_maps=with_maps, tta=tta)
    return compute_metrics(pred, split.num_classes), pred


# --------------------------------------------------------------------------- centralized fit
def fit(
    model: nn.Module,
    train: Split,
    val: Split,
    cfg: TrainConfig,
    seed: int,
    log: Callable[[dict], None] | None = None,
) -> dict:
    """Train with early stopping on validation macro-F1; returns the best state."""
    gen = torch.Generator().manual_seed(seed)
    opt = make_optimizer(model, cfg)
    steps_per_epoch = max(1, len(train) // cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    cfg.extra.setdefault("warmup_frac", cfg.warmup_epochs / max(cfg.epochs, 1))
    best = {"f1": -1.0, "epoch": -1, "state": None, "val": None}
    bad = 0
    history = []
    t0 = time.time()
    for epoch in range(cfg.epochs):
        stats = train_steps(model, train, opt, cfg, gen, progress_fn=lambda s, e=epoch: (e * steps_per_epoch + s) / total)
        val_metrics, _ = evaluate(model, val, cfg)
        row = {"epoch": epoch + 1, **{f"train_{k}": v for k, v in stats.items()},
               **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion"}, "time": time.time() - t0}
        history.append(row)
        if log is not None:
            log(row)
        if val_metrics["f1"] > best["f1"] + 1e-4 or (abs(val_metrics["f1"] - best["f1"]) <= 1e-4 and best["val"] is not None and val_metrics["nll"] < best["val"]["nll"]):
            best = {"f1": val_metrics["f1"], "epoch": epoch + 1, "state": copy.deepcopy(model.state_dict()), "val": val_metrics}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best["state"])
    return {"best_epoch": best["epoch"], "val": best["val"], "history": history, "train_time_sec": time.time() - t0}
