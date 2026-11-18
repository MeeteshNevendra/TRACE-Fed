"""Quantitative explanation analysis on dataset1 test images with tumour boxes.

For every saved final centralized checkpoint:
  * Grad-CAM of the true class on the model's last feature map,
  * pointing game: is the CAM maximum inside the ground-truth box?
  * energy-in-box: share of positive CAM mass inside the box,
  * flip stability: correlation of CAM(x) and unflip(CAM(flip x)),
  * TRACE-Net only: Dice / IoU of the predicted lesion map.
Writes results_v2/tables/xai_metrics.csv and an example figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import IMAGENET_MEAN, IMAGENET_STD, DataStore, flip_batch, prepare_batch
from .engine import lesion_classes_for
from .registry import build


def gradcam(model, images: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16):
        out = model(images)
    fmap = out["feature_map"]
    fmap.retain_grad()
    score = out["logits"].float().gather(1, target[:, None]).sum()
    score.backward()
    w = fmap.grad.float().mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * fmap.float()).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return cam.detach(), out


def analyse(ckpt: Path, store: DataStore, image_size: int = 224) -> tuple[dict, dict]:
    blob = torch.load(ckpt, map_location=store.device, weights_only=False)
    model, _, _ = build(blob["spec"], blob["num_classes"], pretrained=False)
    model.load_state_dict(blob["state"])
    model = model.to(store.device).eval()
    test = store.split("dataset1", "test", 0)
    lesion = lesion_classes_for(test)
    hits, energy, stab, dice, iou = [], [], [], [], []
    example = None
    for raw in test.batches(32, shuffle=False):
        batch = prepare_batch(raw, image_size, train=False)
        sel = lesion[batch["label"]] & (batch["mask"].flatten(1).sum(1) > 0)
        if not bool(sel.any()):
            continue
        img, y, gt = batch["image"][sel], batch["label"][sel], batch["mask"][sel][:, 0]
        cam, out = gradcam(model, img, y)
        cam_f, _ = gradcam(model, flip_batch(img), y)
        cam_f = flip_batch(cam_f)
        flat = cam.flatten(1)
        arg = flat.argmax(1)
        hits.append(gt.flatten(1).gather(1, arg[:, None])[:, 0].cpu())
        energy.append(((flat * gt.flatten(1)).sum(1) / (flat.sum(1) + 1e-8)).cpu())
        a = flat - flat.mean(1, keepdim=True)
        b = cam_f.flatten(1) - cam_f.flatten(1).mean(1, keepdim=True)
        stab.append(((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)).cpu())
        if "mask_logits" in out:
            pred = (out["mask_logits"][:, 0].float() > 0).float().detach()
            inter = (pred * gt).flatten(1).sum(1)
            dice.append((2 * inter / (pred.flatten(1).sum(1) + gt.flatten(1).sum(1) + 1e-6)).cpu())
            iou.append((inter / ((pred + gt).clamp(max=1).flatten(1).sum(1) + 1e-6)).cpu())
        if example is None:
            example = {
                "image": (img[:6, 0].float() * IMAGENET_STD[0] + IMAGENET_MEAN[0]).clamp(0, 1).cpu().numpy(),
                "cam": cam[:6].cpu().numpy(),
                "gt": gt[:6].cpu().numpy(),
                "mask": torch.sigmoid(out["mask_logits"][:6, 0].float()).detach().cpu().numpy() if "mask_logits" in out else None,
            }
    row = {
        "pointing_game": float(torch.cat(hits).mean()),
        "energy_in_box": float(torch.cat(energy).mean()),
        "flip_stability": float(torch.cat(stab).mean()),
        "n_images": int(torch.cat(hits).numel()),
    }
    if dice:
        row["mask_dice"] = float(torch.cat(dice).mean())
        row["mask_iou"] = float(torch.cat(iou).mean())
    return row, example


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("results_v2"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stages", nargs="+", default=["final", "xai_control"],
                    help="checkpoint stages to analyse; xai_control = TRACE-Net trained without box supervision")
    args = ap.parse_args()
    store = DataStore(Path("."), args.work, torch.device(args.device))
    rows, examples = [], {}
    ckpts = [c for s in args.stages for c in sorted((args.work / "checkpoints" / s / "dataset1").glob("*/seed*.pt"))]
    for ckpt in ckpts:
        row, ex = analyse(ckpt, store)
        row.update({"model": ckpt.parent.name, "seed": int(ckpt.stem.replace("seed", ""))})
        rows.append(row)
        if row["seed"] == 0:
            examples[ckpt.parent.name] = ex
        print(json.dumps(row), flush=True)
    df = pd.DataFrame(rows)
    out = args.work / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "xai_per_seed.csv", index=False)
    summary = df.drop(columns="seed").groupby("model").agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.to_csv(out / "xai_metrics.csv")
    print(summary)
    _figure(examples, args.work / "figures" / "xai_examples.png")


def _figure(examples: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not examples:
        return
    names = list(examples)
    n = 4
    fig, axes = plt.subplots(len(names) + 1, n, figsize=(2.2 * n, 2.2 * (len(names) + 1)))
    first = examples[names[0]]
    for j in range(n):
        axes[0, j].imshow(first["image"][j], cmap="gray")
        axes[0, j].contour(first["gt"][j], levels=[0.5], colors="lime", linewidths=0.8)
        axes[0, j].set_axis_off()
    axes[0, 0].set_title("image + GT box", fontsize=8, loc="left")
    for i, name in enumerate(names, start=1):
        ex = examples[name]
        for j in range(n):
            ax = axes[i, j]
            ax.imshow(ex["image"][j], cmap="gray")
            ax.imshow(ex["cam"][j] / (ex["cam"][j].max() + 1e-8), cmap="jet", alpha=0.45)
            ax.contour(ex["gt"][j], levels=[0.5], colors="lime", linewidths=0.8)
            ax.set_axis_off()
        axes[i, 0].set_title(name.split("__")[0] if "__" in name else name, fontsize=8, loc="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
