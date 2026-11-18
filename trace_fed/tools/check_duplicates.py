"""Exact and near-duplicate audit across all splits of both datasets.

Writes a CSV of duplicate groups and a JSON summary; the v2 data module reads
the CSV to drop leaked images before any split is built.

Usage: python -m trace_fed.tools.check_duplicates --root . --out results_v2/data_audit
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(root: Path) -> list[tuple[str, str, str, Path]]:
    """Return (dataset, split, class, path) for every image of both datasets."""
    items = []
    for split_dir, split in [("Train", "train"), ("Val", "val")]:
        for cls_dir in sorted((root / "archive" / split_dir).iterdir()):
            for p in sorted((cls_dir / "images").iterdir()):
                if p.suffix.lower() in IMG_EXT:
                    items.append(("dataset1", split, cls_dir.name, p))
    d2 = root / "archive_2" / "Brain MRI Images" / "Brain MRI Images"
    for split_dir, split in [("Train", "train"), ("Validation", "val")]:
        for cls_dir in sorted((d2 / split_dir).iterdir()):
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in IMG_EXT:
                    items.append(("dataset2", split, cls_dir.name, p))
    return items


def dhash(img: Image.Image, size: int = 16) -> np.ndarray:
    """Difference hash on a size x size grid -> size*size bits."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.float32)
    return (g[:, 1:] > g[:, :-1]).flatten()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("results_v2/data_audit"))
    ap.add_argument("--near_threshold", type=int, default=10, help="max Hamming distance on 256-bit dHash")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    items = list_images(args.root)
    md5s, hashes = [], []
    for _, _, _, p in items:
        data = p.read_bytes()
        md5s.append(hashlib.md5(data).hexdigest())
        with Image.open(p) as im:
            hashes.append(dhash(im))
    bits = np.packbits(np.stack(hashes), axis=1)  # N x 32 bytes
    n = len(items)

    # Pairwise Hamming distance in chunks (popcount via lookup table).
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    by_md5 = defaultdict(list)
    for i, h in enumerate(md5s):
        by_md5[h].append(i)
    for group in by_md5.values():
        for j in group[1:]:
            union(group[0], j)

    near_pairs = 0
    chunk = 512
    for start in range(0, n, chunk):
        block = bits[start:start + chunk]
        dist = lut[np.bitwise_xor(block[:, None, :], bits[None, :, :])].sum(axis=2, dtype=np.int32)
        ii, jj = np.nonzero(dist <= args.near_threshold)
        for a, b in zip(ii + start, jj):
            if a < b:
                union(int(a), int(b))
                near_pairs += 1

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    dup_groups = [g for g in groups.values() if len(g) > 1]

    rows = []
    cross_split = cross_label = cross_dataset = 0
    for gid, g in enumerate(dup_groups):
        splits = {(items[i][0], items[i][1]) for i in g}
        labels = {(items[i][0], items[i][2]) for i in g}
        datasets = {items[i][0] for i in g}
        exact = len({md5s[i] for i in g}) == 1
        cross_split += len(splits) > 1
        cross_label += len(labels) > 1
        cross_dataset += len(datasets) > 1
        for i in g:
            ds, split, cls, p = items[i]
            rows.append(
                {
                    "group": gid,
                    "dataset": ds,
                    "split": split,
                    "class": cls,
                    "path": str(p.relative_to(args.root)).replace("\\", "/"),
                    "exact": exact,
                    "group_size": len(g),
                    "group_cross_split": len(splits) > 1,
                    "group_conflicting_labels": len(labels) > 1,
                }
            )

    import csv

    with open(args.out / "duplicate_groups.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["group"])
        w.writeheader()
        w.writerows(rows)

    per_split = defaultdict(int)
    for ds, split, _, _ in items:
        per_split[f"{ds}/{split}"] += 1
    dup_images = sum(len(g) for g in dup_groups)
    removable = sum(len(g) - 1 for g in dup_groups)
    summary = {
        "images_total": n,
        "images_per_split": dict(per_split),
        "near_threshold_bits_of_256": args.near_threshold,
        "near_pairs": near_pairs,
        "duplicate_groups": len(dup_groups),
        "images_in_duplicate_groups": dup_images,
        "redundant_images_removable": removable,
        "groups_crossing_splits": cross_split,
        "groups_with_conflicting_labels": cross_label,
        "groups_crossing_datasets": cross_dataset,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
