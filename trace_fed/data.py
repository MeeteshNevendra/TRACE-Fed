"""Leakage-free data pipeline for TRACE-Fed (v2).

* Both datasets are pooled per dataset, duplicate groups from the audit
  (`tools/check_duplicates.py`) are collapsed to one representative, and new
  stratified splits are drawn so no image (or its near-copy) crosses splits.
* dataset1: fixed 70/15/15 train/val/test split (test is touched once).
  dataset2: 400 images -> stratified 5-fold CV; fold k is used with seed k.
* Images are cached as 256x256 uint8 grayscale and kept on the GPU; augmentation
  is done on-GPU with an affine grid that is applied identically to box masks.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CACHE_SIZE = 256
D1_CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
D2_CLASSES = ["Normal", "Tumor"]
BINARY_CLASSES = ["Normal", "Tumor"]
D1_NO_TUMOR_YOLO_ID = 2
SPLIT_SEED = 2026
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- catalog
def _read_yolo(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            try:
                cid = int(float(parts[0]))
                vals = [float(v) for v in parts[1:5]]
            except ValueError:
                continue
            if all(math.isfinite(v) for v in vals):
                boxes.append((cid, *vals))
    return boxes


def build_catalog(root: Path, audit_csv: Path) -> pd.DataFrame:
    """One row per *unique* image (duplicate groups collapsed)."""
    rows = []
    for split_dir in ["Train", "Val"]:
        for cls in D1_CLASSES:
            img_dir = root / "archive" / split_dir / cls / "images"
            for p in sorted(img_dir.iterdir()):
                if p.suffix.lower() not in IMG_EXT:
                    continue
                boxes = _read_yolo(root / "archive" / split_dir / cls / "labels" / f"{p.stem}.txt")
                tumor_boxes = [b for b in boxes if b[0] != D1_NO_TUMOR_YOLO_ID] if cls != "No Tumor" else []
                rows.append(
                    {
                        "dataset": "dataset1",
                        "source_split": split_dir.lower(),
                        "class_name": cls,
                        "label": D1_CLASSES.index(cls),
                        "binary_label": int(cls != "No Tumor"),
                        "path": str(p.relative_to(root)).replace("\\", "/"),
                        "boxes": json.dumps([b[1:] for b in tumor_boxes]),
                        "has_mask": 1,  # tumour boxes, or an empty mask for No Tumor
                        "box_class_mismatch": int(
                            cls != "No Tumor" and any(b[0] != D1_CLASSES.index(cls) for b in tumor_boxes)
                        ),
                    }
                )
    d2 = root / "archive_2" / "Brain MRI Images" / "Brain MRI Images"
    for split_dir in ["Train", "Validation"]:
        for cls in D2_CLASSES:
            for p in sorted((d2 / split_dir / cls).iterdir()):
                if p.suffix.lower() not in IMG_EXT:
                    continue
                rows.append(
                    {
                        "dataset": "dataset2",
                        "source_split": "train" if split_dir == "Train" else "val",
                        "class_name": cls,
                        "label": D2_CLASSES.index(cls),
                        "binary_label": D2_CLASSES.index(cls),
                        "path": str(p.relative_to(root)).replace("\\", "/"),
                        "boxes": "[]",
                        # no localisation ground truth; Normal images have a known-empty mask
                        "has_mask": int(cls == "Normal"),
                        "box_class_mismatch": 0,
                    }
                )
    df = pd.DataFrame(rows)
    dups = pd.read_csv(audit_csv)
    path_to_group = dict(zip(dups["path"], dups["group"]))
    df["dup_group"] = [path_to_group.get(p, -1) for p in df["path"]]
    # Representative per group = lexicographically first path; drop the rest.
    df = df.sort_values("path")
    redundant = df["dup_group"].ne(-1) & df.duplicated("dup_group", keep="first")
    df = df[~redundant].sort_values(["dataset", "path"]).reset_index(drop=True)
    df["uid"] = np.arange(len(df))
    return df


# --------------------------------------------------------------------------- splits
def _stratified_split(labels: np.ndarray, fractions: list[float], rng: np.random.Generator) -> list[np.ndarray]:
    parts: list[list[int]] = [[] for _ in fractions]
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        cuts = np.cumsum([int(round(f * len(idx))) for f in fractions[:-1]])
        for k, chunk in enumerate(np.split(idx, cuts)):
            parts[k].extend(chunk.tolist())
    return [np.array(sorted(p), dtype=int) for p in parts]


def make_splits(catalog: pd.DataFrame, out_dir: Path) -> dict:
    """dataset1: one fixed split. dataset2: 5 CV folds (each with its own val)."""
    rng = np.random.default_rng(SPLIT_SEED)
    splits: dict = {}
    d1 = catalog[catalog.dataset == "dataset1"]
    tr, va, te = _stratified_split(d1["label"].to_numpy(), [0.70, 0.15, 0.15], rng)
    uids = d1["uid"].to_numpy()
    splits["dataset1"] = [{"train": uids[tr].tolist(), "val": uids[va].tolist(), "test": uids[te].tolist()}]

    d2 = catalog[catalog.dataset == "dataset2"]
    labels = d2["label"].to_numpy()
    uids = d2["uid"].to_numpy()
    folds = _stratified_split(labels, [0.2] * 5, rng)
    splits["dataset2"] = []
    for k in range(5):
        test_idx = folds[k]
        rest = np.setdiff1d(np.arange(len(d2)), test_idx)
        va_local, tr_local = _stratified_split(labels[rest], [0.15, 0.85], rng)
        splits["dataset2"].append(
            {"train": uids[rest[tr_local]].tolist(), "val": uids[rest[va_local]].tolist(), "test": uids[test_idx].tolist()}
        )
    _assert_disjoint(splits)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "splits.json").write_text(json.dumps(splits))
    return splits


def _assert_disjoint(splits: dict) -> None:
    for name, folds in splits.items():
        for f in folds:
            a, b, c = set(f["train"]), set(f["val"]), set(f["test"])
            assert not (a & b or a & c or b & c), f"split overlap in {name}"


# --------------------------------------------------------------------------- cache
def _box_mask(boxes: list[list[float]], size: int) -> np.ndarray:
    m = np.zeros((size, size), dtype=np.uint8)
    for xc, yc, w, h in boxes:
        x0 = int(max(0, (xc - w / 2) * size)); x1 = int(min(size, math.ceil((xc + w / 2) * size)))
        y0 = int(max(0, (yc - h / 2) * size)); y1 = int(min(size, math.ceil((yc + h / 2) * size)))
        m[y0:y1, x0:x1] = 1
    return m


def build_cache(root: Path, catalog: pd.DataFrame, cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    img_f, mask_f = cache_dir / f"images_{CACHE_SIZE}.npy", cache_dir / f"masks_{CACHE_SIZE}.npy"
    key_f = cache_dir / "catalog_paths.json"
    paths = catalog["path"].tolist()
    if img_f.exists() and mask_f.exists() and key_f.exists() and json.loads(key_f.read_text()) == paths:
        return np.load(img_f), np.load(mask_f)
    images = np.zeros((len(catalog), CACHE_SIZE, CACHE_SIZE), dtype=np.uint8)
    masks = np.zeros((len(catalog), CACHE_SIZE, CACHE_SIZE), dtype=np.uint8)
    for i, row in enumerate(catalog.itertuples()):
        with Image.open(root / row.path) as im:
            images[i] = np.asarray(im.convert("L").resize((CACHE_SIZE, CACHE_SIZE), Image.BICUBIC))
        masks[i] = _box_mask(json.loads(row.boxes), CACHE_SIZE)
    np.save(img_f, images)
    np.save(mask_f, masks)
    key_f.write_text(json.dumps(paths))
    return images, masks


# --------------------------------------------------------------------------- tensors on device
@dataclass
class Split:
    """Index view into the device-resident tensors of one split."""

    store: "DataStore"
    idx: torch.Tensor  # rows into the store
    labels: torch.Tensor
    num_classes: int
    class_names: list[str]
    domain: torch.Tensor = field(default=None)  # 0 = dataset1, 1 = dataset2

    def __len__(self) -> int:
        return int(self.idx.numel())

    def subset(self, positions: np.ndarray | list[int]) -> "Split":
        pos = torch.as_tensor(np.asarray(positions, dtype=np.int64), device=self.idx.device)
        return Split(self.store, self.idx[pos], self.labels[pos], self.num_classes, self.class_names, self.domain[pos])

    def label_array(self) -> np.ndarray:
        return self.labels.cpu().numpy()

    def batches(self, batch_size: int, shuffle: bool, generator: torch.Generator | None = None, drop_last: bool = False):
        n = len(self)
        order = torch.randperm(n, generator=generator, device="cpu").to(self.idx.device) if shuffle else torch.arange(n, device=self.idx.device)
        stop = n - (n % batch_size) if drop_last and n >= batch_size else n
        for s in range(0, stop, batch_size):
            pos = order[s:s + batch_size]
            if drop_last and pos.numel() < 2:
                continue
            rows = self.idx[pos]
            yield {
                "image": self.store.images[rows],
                "mask": self.store.masks[rows],
                "has_mask": self.store.has_mask[rows],
                "label": self.labels[pos],
                "domain": self.domain[pos],
                "uid": self.store.uids[rows],
            }


class DataStore:
    def __init__(self, root: Path, work_dir: Path, device: torch.device) -> None:
        audit = work_dir / "data_audit" / "duplicate_groups.csv"
        if not audit.exists():
            raise FileNotFoundError(f"Run trace_fed.tools.check_duplicates first ({audit})")
        self.catalog = build_catalog(root, audit)
        split_file = work_dir / "splits" / "splits.json"
        self.splits = json.loads(split_file.read_text()) if split_file.exists() else make_splits(self.catalog, split_file.parent)
        _assert_disjoint(self.splits)
        images, masks = build_cache(root, self.catalog, work_dir / "cache")
        self.device = device
        self.images = torch.from_numpy(images).to(device)  # N x H x W uint8
        self.masks = torch.from_numpy(masks).to(device)
        self.has_mask = torch.tensor(self.catalog["has_mask"].to_numpy(), device=device, dtype=torch.float32)
        self.uids = torch.tensor(self.catalog["uid"].to_numpy(), device=device)
        self.native_labels = torch.tensor(self.catalog["label"].to_numpy(), device=device)
        self.binary_labels = torch.tensor(self.catalog["binary_label"].to_numpy(), device=device)
        self.domains = torch.tensor((self.catalog["dataset"] == "dataset2").astype(int).to_numpy(), device=device)

    def num_folds(self, dataset: str) -> int:
        return len(self.splits[dataset])

    def split(self, dataset: str, part: str, fold: int = 0, label_mode: str = "native") -> Split:
        uids = torch.tensor(self.splits[dataset][fold % len(self.splits[dataset])][part], device=self.device)
        labels = (self.binary_labels if label_mode == "binary" else self.native_labels)[uids]
        if label_mode == "binary":
            names = BINARY_CLASSES
        else:
            names = D1_CLASSES if dataset == "dataset1" else D2_CLASSES
        return Split(self, uids, labels, len(names), names, self.domains[uids])

    def summary(self) -> pd.DataFrame:
        rows = []
        for ds, folds in self.splits.items():
            for k, f in enumerate(folds):
                for part in ["train", "val", "test"]:
                    sub = self.catalog.iloc[f[part]]
                    row = {"dataset": ds, "fold": k, "split": part, "n": len(sub)}
                    row.update({f"n_{c}": int(v) for c, v in sub["class_name"].value_counts().sort_index().items()})
                    rows.append(row)
        return pd.DataFrame(rows)


def concat_splits(splits: list[Split]) -> Split:
    first = splits[0]
    return Split(
        first.store,
        torch.cat([s.idx for s in splits]),
        torch.cat([s.labels for s in splits]),
        first.num_classes,
        first.class_names,
        torch.cat([s.domain for s in splits]),
    )


# --------------------------------------------------------------------------- GPU augmentation
_MEAN = None
_STD = None


def _norm(x: torch.Tensor) -> torch.Tensor:
    global _MEAN, _STD
    if _MEAN is None or _MEAN.device != x.device:
        _MEAN = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        _STD = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x.expand(-1, 3, -1, -1) - _MEAN) / _STD


def prepare_batch(batch: dict, image_size: int, train: bool, generator: torch.Generator | None = None) -> dict:
    """uint8 grayscale -> normalized 3-channel float; masks follow the same geometry."""
    img = batch["image"].unsqueeze(1).float() / 255.0
    mask = batch["mask"].unsqueeze(1).float()
    b = img.size(0)
    if train:
        dev = img.device

        def rand(lo: float, hi: float) -> torch.Tensor:
            return torch.empty(b, device="cpu").uniform_(lo, hi, generator=generator).to(dev)

        angle = rand(-15, 15) * math.pi / 180
        scale = rand(0.85, 1.15)
        flip = torch.where(rand(0, 1) < 0.5, -1.0, 1.0)
        tx, ty = rand(-0.08, 0.08), rand(-0.08, 0.08)
        cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
        theta = torch.stack(
            [torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1
        )
        grid = F.affine_grid(theta, (b, 1, image_size, image_size), align_corners=False)
        img = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        mask = (F.grid_sample(mask, grid, mode="nearest", padding_mode="zeros", align_corners=False) > 0.5).float()
        # intensity: contrast, brightness, gamma
        contrast = rand(0.8, 1.2).view(b, 1, 1, 1)
        bright = rand(-0.08, 0.08).view(b, 1, 1, 1)
        gamma = rand(0.8, 1.25).view(b, 1, 1, 1)
        img = ((img.clamp(0, 1) ** gamma - 0.5) * contrast + 0.5 + bright).clamp(0, 1)
    else:
        img = F.interpolate(img, size=(image_size, image_size), mode="bilinear", align_corners=False, antialias=True)
        mask = (F.interpolate(mask, size=(image_size, image_size), mode="nearest") > 0.5).float()
    out = dict(batch)
    out["image"] = _norm(img)
    out["mask"] = mask
    return out


def flip_batch(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-1])
