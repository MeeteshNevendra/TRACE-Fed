"""Centralized training of one (model spec, dataset, seed) cell. Resumable.

Example:
  python -m trace_fed.run_central --dataset dataset1 --model trace --seeds 0 1 2 3 4
  python -m trace_fed.run_central --dataset dataset1 --model convnext_tiny --stage search --seeds 0
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data import DataStore
from .engine import TrainConfig, evaluate, fit
from .registry import build
from .utils import count_params, device_from, safe_name, set_seed, write_json


def run_one(store: DataStore, dataset: str, spec: str, seed: int, args, out_root: Path) -> dict | None:
    out = out_root / args.stage / dataset / safe_name(spec) / f"seed{seed}.json"
    if out.exists() and not args.overwrite:
        print(f"skip (exists) {out}")
        return None
    fold = seed % store.num_folds(dataset)
    label_mode = args.label_mode
    train = store.split(dataset, "train", fold, label_mode)
    val = store.split(dataset, "val", fold, label_mode)
    set_seed(seed)
    model, loss_over, train_over = build(spec, train.num_classes)
    model = model.to(store.device).to(memory_format=torch.channels_last)
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, image_size=args.image_size, amp=store.device.type == "cuda")
    for k, v in {**loss_over, **train_over}.items():
        setattr(cfg, k, v)
    t0 = time.time()
    log_rows = []
    result = fit(model, train, val, cfg, seed, log=lambda r: (log_rows.append(r), print(
        f"[{dataset} {spec} s{seed}] ep{r['epoch']:>3} loss {r['train_loss']:.3f} val_f1 {r['val_f1']:.4f} val_acc {r['val_accuracy']:.4f} {r['time']:.0f}s",
        flush=True)) if args.verbose else log_rows.append(r))
    record = {
        "stage": args.stage,
        "dataset": dataset,
        "fold": fold,
        "label_mode": label_mode,
        "model": spec,
        "seed": seed,
        "config": asdict(cfg),
        "params_total": count_params(model),
        "params_trainable": count_params(model, trainable_only=True),
        "best_epoch": result["best_epoch"],
        "val": result["val"],
        "history": result["history"],
        "train_time_sec": result["train_time_sec"],
    }
    if hasattr(model, "gate_means"):
        record["gate_means"] = model.gate_means()
    if args.stage != "search":
        test = store.split(dataset, "test", fold, label_mode)
        test_metrics, pred = evaluate(model, test, cfg, with_maps=True)
        record["test"] = test_metrics
        pred_path = out.with_suffix(".npz")
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pred_path, probs=pred["probs"], labels=pred["labels"], uids=pred["uids"])
        # cross-dataset evaluation of binary models on the other dataset's test split
        if label_mode == "binary":
            other = "dataset2" if dataset == "dataset1" else "dataset1"
            record["cross_test"] = {}
            for k in range(store.num_folds(other)):
                m, _ = evaluate(model, store.split(other, "test", k, "binary"), cfg)
                record["cross_test"][f"{other}_fold{k}"] = m
        if args.save_ckpt:
            ck = args.work / "checkpoints" / args.stage / dataset / safe_name(spec) / f"seed{seed}.pt"
            ck.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"spec": spec, "num_classes": train.num_classes, "state": model.state_dict()}, ck)
            record["checkpoint"] = str(ck)
    record["wall_time_sec"] = time.time() - t0
    write_json(out, record)
    val_f1 = record["val"]["f1"]
    test_f1 = record.get("test", {}).get("f1", float("nan"))
    print(f"DONE {dataset} {spec} seed{seed}: best_ep {record['best_epoch']} val_f1 {val_f1:.4f} test_f1 {test_f1:.4f} ({record['wall_time_sec']:.0f}s)", flush=True)
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--work", type=Path, default=Path("results_v2"))
    ap.add_argument("--dataset", nargs="+", default=["dataset1"])
    ap.add_argument("--model", nargs="+", default=["trace"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--stage", default="final", choices=["search", "final", "ablation", "cross", "xai_control", "smoke"])
    ap.add_argument("--label_mode", default="native", choices=["native", "binary"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_ckpt", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: cap split sizes")
    args = ap.parse_args()
    if args.label_mode == "binary":
        args.stage = args.stage if args.stage != "final" else "cross"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    store = DataStore(args.root, args.work, device_from(args.device))
    if args.limit:
        _limit_store(store, args.limit)
    out_root = args.work / "runs" / "central"
    for dataset in args.dataset:
        for spec in args.model:
            for seed in args.seeds:
                run_one(store, dataset, spec, seed, args, out_root)


def _limit_store(store: DataStore, n: int) -> None:
    for folds in store.splits.values():
        for f in folds:
            for part in f:
                f[part] = f[part][:: max(1, len(f[part]) // n)][:n]


if __name__ == "__main__":
    main()
