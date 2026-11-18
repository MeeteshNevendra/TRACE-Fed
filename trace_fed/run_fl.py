"""Federated experiments. Resumable; one JSON (+ predictions) per cell.

Examples:
  python -m trace_fed.run_fl --dataset dataset1 --method fedavg fedprox trace --seeds 0 1 2
  python -m trace_fed.run_fl --dataset dataset1 --method trace --opts "trust=0" --tag no_trust
  python -m trace_fed.run_fl --scenario mixed --method fedavg trace --seeds 0
  PYTHONHASHSEED=0 python -m trace_fed.run_fl --dataset dataset2 --method trace --seeds 2 --deterministic --provenance

Two opt-in switches. With both off the runner behaves exactly as before.
  --deterministic  cuDNN autotuning off, cuDNN restricted to deterministic algorithms, cuBLAS workspace pinned
                   (CUBLAS_WORKSPACE_CONFIG=:4096:8) and torch.use_deterministic_algorithms(True) WITHOUT warn_only.
                   An operation that has no deterministic implementation therefore raises: the run writes
                   failure_seed<k>.json (operation, code location, traceback, and the per-client results already
                   computed in the round in progress) and stops. It never falls back to non-deterministic
                   execution. Requires PYTHONHASHSEED to be set before Python starts.
  --provenance     adds backend settings, environment, data/split/partition hashes and model-state hashes
                   (initial and selected) to the result JSON. Read-only; does not change the computation.
  --round-hashes   adds, for every round, SHA-256 of each client's compressed update and error-feedback state,
                   the server aggregate, the server residual, the new global shared state and the prototypes,
                   plus counts of non-finite values. Read-only; its time is recorded separately.
  --save-checkpoint  saves the selected round's full model state (state_dict, CPU tensors) as seed<k>.pt next to
                   the result JSON, and records the file's SHA-256 in it.
  --client-train-partition iid_size_matched  (diagnostic) after the usual partition, re-deal the clients'
                   training images class-stratified with every client's training size and validation set kept
                   (fl.iid_size_matched_train). Default: off.
  --no-test-eval   skip the final test pass (no test predictions or metrics are produced); test membership is
                   still recorded in the provenance. Training and selection are unaffected. Default: off.
  --trust-terms    (TRACE-Fed, ablation) comma list of the trust factors applied, from agree,stab,util (default:
                   all three, the method as published). A factor left out is computed and recorded as
                   <factor>_unused but enters the product as exactly 1; floor and normalisation are unchanged.
  --trust-diagnostics  (TRACE-Fed) writes seed<k>_diagnostics.json: per round and client the pre/post local-
                   validation losses, utility, validation probabilities and predictions, update geometry against
                   the coordinate-wise median, trust factors and weights; with trust=0 the trust statistics are
                   computed as a shadow and never applied. Observation only (see fl.run_federated).
  --client-optimizer-state persistent  (diagnostic) each client keeps its own AdamW moments and step counters
                   across rounds; the optimiser is still built fresh (same learning rates and settings) and the model
                   is synchronised to the global state first. Default: fresh (a new optimiser every round).

In deterministic mode the trust rule's coordinate-wise median uses a sort-based equivalent (fl.coordinate_median),
because CUDA median has no deterministic implementation. With --deterministic off it is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch

from .data import DataStore, concat_splits
from .engine import TrainConfig
from .fl import CLIENT_OPTIMIZER_STATES, TRUST_TERMS, FLConfig, build_clients, iid_size_matched_train, run_federated
from .registry import build, parse_spec
from .utils import count_params, device_from, safe_name, set_seed, write_json

TRACE_MODEL = "trace:peft=1"
BASE_MODEL = "trace"  # baselines federate the same TRACE-Net with full fine-tuning
FL_FIELDS = {f.name: f.type for f in fields(FLConfig)}


def parse_opts(s: str) -> dict:
    _, kv = parse_spec("x:" + s) if s else (None, {})
    for k, v in kv.items():
        if k not in FL_FIELDS:
            raise ValueError(f"unknown FL option {k}")
        if FL_FIELDS[k] in ("bool", bool):
            kv[k] = bool(int(v))
    return kv


# ----------------------------------------------------------------------------- determinism / provenance (opt-in)
DETERMINISM_ERROR = "does not have a deterministic implementation"
TRAIN_CONFIG_KEYS = ("lr", "backbone_lr", "batch_size", "image_size", "seg_weight", "xc_weight")


def configure_backends(deterministic: bool) -> None:
    """deterministic=False is exactly the original setup (cuDNN autotuning on, TF32 matmuls)."""
    if deterministic:
        if os.environ.get("PYTHONHASHSEED") is None:
            raise SystemExit("--deterministic needs PYTHONHASHSEED set before Python starts, e.g. PYTHONHASHSEED=0")
        if torch.cuda.is_initialized():
            raise SystemExit("--deterministic settings must be applied before CUDA is initialised")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # read when the first cuBLAS handle is created
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)  # strict: ops without a deterministic implementation raise
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensor_bytes(t: torch.Tensor) -> bytes:
    """Raw bytes in logical (row-major) order, independent of memory format."""
    return t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001 - provenance must never stop a run
        return f"unavailable: {type(e).__name__}"


def state_hashes(model: torch.nn.Module) -> dict:
    """SHA-256 over the state dict: all tensors, trainable parameters, and frozen weights + buffers."""
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    hs = {"all": hashlib.sha256(), "trainable": hashlib.sha256(), "frozen_and_buffers": hashlib.sha256()}
    sd = model.state_dict()
    for k, v in sd.items():
        meta = f"{k}|{v.dtype}|{tuple(v.shape)}|".encode()
        data = _tensor_bytes(v)
        for name in ("all", "trainable" if k in trainable else "frozen_and_buffers"):
            hs[name].update(meta)
            hs[name].update(data)
    return {k: h.hexdigest() for k, h in hs.items()} | {"tensors": len(sd), "trainable_tensors": len(trainable)}


def backend_settings() -> dict:
    def get(fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - attribute names differ across torch versions
            return f"unavailable: {type(e).__name__}"

    return {
        "cudnn.enabled": get(lambda: torch.backends.cudnn.enabled),
        "cudnn.benchmark": get(lambda: torch.backends.cudnn.benchmark),
        "cudnn.deterministic": get(lambda: torch.backends.cudnn.deterministic),
        "cudnn.allow_tf32": get(lambda: torch.backends.cudnn.allow_tf32),
        "cuda.matmul.allow_tf32": get(lambda: torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": get(torch.get_float32_matmul_precision),
        "deterministic_algorithms": get(torch.are_deterministic_algorithms_enabled),
        "deterministic_algorithms_warn_only": get(torch.is_deterministic_algorithms_warn_only_enabled),
        "fill_uninitialized_memory": get(lambda: torch.utils.deterministic.fill_uninitialized_memory),
        "sdp_flash": get(torch.backends.cuda.flash_sdp_enabled),
        "sdp_mem_efficient": get(torch.backends.cuda.mem_efficient_sdp_enabled),
        "sdp_math": get(torch.backends.cuda.math_sdp_enabled),
        "sdp_cudnn": get(torch.backends.cuda.cudnn_sdp_enabled),
        "torch_num_threads": torch.get_num_threads(),
        "env": {k: os.environ.get(k) for k in ("PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES",
                                                "CUDA_LAUNCH_BLOCKING", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    }


def environment_info(root: Path) -> dict:
    import importlib.metadata as md

    packages = {}
    for p in ("torch", "torchvision", "timm", "numpy", "scikit-learn", "scipy", "pandas", "huggingface_hub",
              "safetensors", "pillow"):
        try:
            packages[p] = md.version(p)
        except md.PackageNotFoundError:
            packages[p] = None
    info = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": packages, "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = {"name": p.name, "capability": f"{p.major}.{p.minor}", "total_memory_mb": p.total_memory // 2**20}
        info["driver"] = _cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        info["gpu_state_at_start"] = _cmd(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                                           "--format=csv,noheader"])
        info["gpu_compute_apps_at_start"] = _cmd(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"])
    info["git_head"] = _cmd(["git", "-C", str(root), "rev-parse", "HEAD"])
    info["git_status_trace_fed"] = _cmd(["git", "-C", str(root), "status", "--porcelain", "--", "trace_fed"])
    pkg = Path(__file__).resolve().parent
    info["source_sha256"] = {p.relative_to(pkg.parent).as_posix(): _sha256_file(p) for p in sorted(pkg.rglob("*.py"))}
    return info


def pretrained_files(model: torch.nn.Module) -> dict:
    out = {}
    try:
        from huggingface_hub import try_to_load_from_cache

        hub_id = (getattr(getattr(model, "backbone", None), "pretrained_cfg", None) or {}).get("hf_hub_id")
        for fname in ("model.safetensors", "pytorch_model.bin"):
            p = try_to_load_from_cache(hub_id, fname) if hub_id else None
            if isinstance(p, str):
                out[f"{hub_id}/{fname}"] = {"path": p, "bytes": Path(p).stat().st_size, "sha256": _sha256_file(Path(p))}
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def data_provenance(store: DataStore, work: Path, dataset: str, fold: int, clients: list, tests: dict) -> dict:
    files = {rel: _sha256_file(work / rel) for rel in ("data_audit/duplicate_groups.csv", "splits/splits.json",
                                                       "cache/images_256.npy", "cache/masks_256.npy",
                                                       "cache/catalog_paths.json")}
    cat = store.catalog[["uid", "dataset", "path", "label", "binary_label", "has_mask"]]
    fold_lists = store.splits[dataset][fold]
    return {
        "files_sha256": files,
        "catalog_sha256": _sha256_bytes(cat.to_csv(index=False).encode()),
        "fold": fold,
        "fold_split_sha256": {part: _sha256_bytes(json.dumps(fold_lists[part]).encode()) for part in ("train", "val", "test")},
        "fold_split_sizes": {part: len(fold_lists[part]) for part in ("train", "val", "test")},
        "clients": [{"cid": c.cid, "n_train": len(c.train), "n_val": len(c.val),
                     "train_class_counts": np.bincount(c.train.label_array(), minlength=c.train.num_classes).tolist(),
                     "val_class_counts": np.bincount(c.val.label_array(), minlength=c.val.num_classes).tolist(),
                     "train_rows_sha256": _sha256_bytes(_tensor_bytes(c.train.idx)),
                     "val_rows_sha256": _sha256_bytes(_tensor_bytes(c.val.idx)),
                     "train_labels_sha256": _sha256_bytes(_tensor_bytes(c.train.labels)),
                     "val_labels_sha256": _sha256_bytes(_tensor_bytes(c.val.labels))} for c in clients],
        "tests": {name: {"n": len(s), "rows_sha256": _sha256_bytes(_tensor_bytes(s.idx)),
                         "labels_sha256": _sha256_bytes(_tensor_bytes(s.labels))} for name, s in tests.items()},
    }


def _hash_tensors(tensors: dict) -> str:
    h = hashlib.sha256()
    for k, v in tensors.items():
        h.update(f"{k}|{v.dtype}|{tuple(v.shape)}|".encode())
        h.update(_tensor_bytes(v))
    return h.hexdigest()


class RoundHasher:
    """run_federated observer for --round-hashes: hashes each round's tensors; never modifies them."""

    def __init__(self) -> None:
        self.rounds: list[dict] = []
        self.seconds = 0.0

    @staticmethod
    def _nonfinite(tensors) -> int:
        counts = [(~torch.isfinite(v)).sum() for v in tensors if torch.is_floating_point(v)]
        return int(torch.stack(counts).sum()) if counts else 0

    def __call__(self, rnd: int, t: dict) -> None:
        t0 = time.time()
        self.rounds.append({
            "round": rnd,
            "client_update_sha256": [_hash_tensors(d) for d in t["client_updates"]],
            "client_error_feedback_sha256": [_hash_tensors(e) for e in t["client_error_feedback"]],
            "aggregate_sha256": _hash_tensors(t["aggregate"]),
            "server_error_feedback_sha256": _hash_tensors(t["server_error_feedback"]),
            "global_shared_sha256": _hash_tensors(t["global_shared"]),
            "prototype_sums_sha256": _hash_tensors(t["prototype_sums"]) if t["prototype_sums"] else None,
            "prototypes_sha256": _hash_tensors({"protos": t["prototypes"]}) if t["prototypes"] is not None else None,
            "nonfinite_values": {
                "client_updates": sum(self._nonfinite(d.values()) for d in t["client_updates"]),
                "client_error_feedback": sum(self._nonfinite(e.values()) for e in t["client_error_feedback"]),
                "aggregate": self._nonfinite(t["aggregate"].values()),
                "global_shared": self._nonfinite(t["global_shared"].values())},
        })
        self.seconds += time.time() - t0


def state_before_failure(err: BaseException) -> dict | None:
    """Per-client results of the round in progress when the error was raised, read (not changed) from the
    run_federated frame: client statistics, and hashes of each client's compressed update and error-feedback
    residual and of the prototype sums. Lets two failed runs be compared up to the point where they stopped."""
    try:
        for frame, _ in traceback.walk_tb(err.__traceback__):
            if frame.f_code.co_name != "run_federated":
                continue
            loc = frame.f_locals
            protos = {k: loc[k] for k in ("proto_sum", "proto_cnt") if isinstance(loc.get(k), torch.Tensor)}
            return {"round": loc.get("rnd"),
                    "client_infos": [dict(i) for i in loc.get("infos") or []],
                    "client_update_sha256": [_hash_tensors(d) for d in loc.get("deltas") or []],
                    "client_error_feedback_sha256": [_hash_tensors(c.ef) for c in loc.get("clients") or []],
                    "prototype_sums_sha256": _hash_tensors(protos) if protos else None}
    except Exception as e:  # noqa: BLE001 - must never mask the original error
        return {"error": f"{type(e).__name__}: {e}"}
    return None


def write_failure(out: Path, err: BaseException, base: dict) -> Path:
    """Record an operation that has no deterministic implementation (deterministic mode only)."""
    frames = [{"file": f.filename, "line": f.lineno, "function": f.name, "code": f.line}
              for f in traceback.extract_tb(err.__traceback__)]
    ours = [f for f in frames if "trace_fed" in f["file"].replace("\\", "/")]
    msg = str(err)
    rec = {**base, "status": "failed: operation without a deterministic implementation",
           "operation": msg.split(DETERMINISM_ERROR)[0].strip(), "error_type": type(err).__name__, "error": msg,
           "raised_at": ours[-1] if ours else (frames[-1] if frames else None), "traceback": frames,
           "state_before_failure": state_before_failure(err)}
    path = out.parent / f"failure_seed{base['seed']}.json"
    write_json(path, rec)
    return path


def run_cell(store: DataStore, args, scenario: str, dataset: str, method: str, seed: int) -> None:
    opts = parse_opts(args.opts)
    tag = args.tag or method
    if scenario == "main":
        sub = f"{dataset}/{args.partition}{'' if args.partition == 'iid' else args.alpha}_c{args.clients}"
    else:
        sub = f"mixed/c{args.clients}"
    out = args.work / "runs" / "fl" / args.stage / sub / safe_name(tag) / f"seed{seed}.json"
    if out.exists() and not args.overwrite:
        print(f"skip (exists) {out}")
        return
    cfg = FLConfig(method=method, rounds=args.rounds, clients=args.clients, local_epochs=args.local_epochs,
                   partition=args.partition, alpha=args.alpha, verbose=args.verbose)
    for k, v in opts.items():
        setattr(cfg, k, v)
    spec = args.model or (TRACE_MODEL if method == "trace" else BASE_MODEL)
    if scenario == "main":
        fold = seed % store.num_folds(dataset)
        train = store.split(dataset, "train", fold)
        sites = [(dataset, train, args.clients)]
        tests = {dataset: store.split(dataset, "test", fold)}
        num_classes = train.num_classes
    else:  # mixed sites, binary task: dataset1 -> clients-2 clients, dataset2 -> 2 clients
        fold = seed % store.num_folds("dataset2")
        d1 = store.split("dataset1", "train", 0, "binary")
        d2 = store.split("dataset2", "train", fold, "binary")
        sites = [("dataset1", d1, args.clients - 2), ("dataset2", d2, 2)]
        t1 = store.split("dataset1", "test", 0, "binary")
        t2 = store.split("dataset2", "test", fold, "binary")
        tests = {"dataset1": t1, "dataset2": t2, "pooled": concat_splits([t1, t2])}
        num_classes = 2
    set_seed(seed)
    model, loss_over, train_over = build(spec, num_classes)
    model = model.to(store.device).to(memory_format=torch.channels_last)
    tcfg = TrainConfig(batch_size=args.batch_size, image_size=args.image_size, amp=store.device.type == "cuda",
                       lr=args.lr, backbone_lr=args.backbone_lr)
    tcfg.extra["warmup_frac"] = 2 / max(args.rounds, 1)
    for k, v in {**loss_over, **train_over}.items():
        setattr(tcfg, k, v)
    clients = build_clients(sites, cfg, seed)
    repartition = None
    if args.client_train_partition == "iid_size_matched":
        if scenario != "main":
            raise SystemExit("--client-train-partition is defined for single-dataset runs only")
        repartition = iid_size_matched_train(clients, seed)
    extra_record = {}  # stays empty unless --deterministic / --provenance
    if args.deterministic or args.provenance:
        extra_record["deterministic"] = bool(args.deterministic)
        extra_record["backend"] = backend_settings() | {"autocast_bf16": tcfg.amp}
    if args.provenance:
        extra_record["provenance"] = {
            "environment": args.env_info,
            "seeds": {"seed": seed, "python_numpy_torch_cuda": seed, "data_order_and_augmentation_generator": seed,
                      "partition_rng": seed, "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")},
            "pretrained_files": pretrained_files(model),
            "data": data_provenance(store, args.work, dataset, fold, clients, tests),
            "initial_state_sha256": state_hashes(model),
        }
    hasher = RoundHasher() if args.round_hashes else None
    diagnostics = [] if args.trust_diagnostics else None
    terms = TRUST_TERMS
    if args.trust_terms:
        terms = tuple(t.strip() for t in args.trust_terms.split(",") if t.strip())
        if not terms or set(terms) - set(TRUST_TERMS) or method != "trace" or not cfg.trust:
            raise SystemExit(f"--trust-terms needs TRACE-Fed with trust on and a subset of {TRUST_TERMS}")
    t0 = time.time()
    print(f"== {scenario} {dataset} {tag} seed{seed}: {len(clients)} clients sizes {[len(c.train) for c in clients]}", flush=True)
    try:
        res = run_federated(model, clients, [c.val for c in clients], {} if args.no_test_eval else tests, tcfg, cfg,
                            seed, observer=hasher, diagnostics=diagnostics, trust_terms=terms,
                            client_optimizer_state=args.client_optimizer_state)
    except RuntimeError as err:
        if args.deterministic and DETERMINISM_ERROR in str(err):
            base = {"stage": args.stage, "scenario": scenario, "dataset": dataset, "fold": fold, "tag": tag, "model": spec,
                    "seed": seed, "config": asdict(cfg), "train_config": {k: getattr(tcfg, k) for k in TRAIN_CONFIG_KEYS},
                    **extra_record, "wall_time_sec": time.time() - t0}
            if hasher is not None:
                base["round_hashes"] = hasher.rounds
            print(f"FAILED {tag} seed{seed}: no deterministic implementation; see {write_failure(out, err, base)}", flush=True)
        raise
    preds = {}
    for name, m in res["test"].items():
        pred = m.pop("_pred", None)
        if pred is not None:
            preds[name] = pred
    record = {
        "stage": args.stage,
        "scenario": scenario,
        "dataset": dataset,
        "fold": fold,
        "tag": tag,
        "model": spec,
        "seed": seed,
        "train_config": {k: getattr(tcfg, k) for k in ("lr", "backbone_lr", "batch_size", "image_size", "seg_weight", "xc_weight")},
        "params_total": count_params(model),
        "client_sizes": [len(c.train) for c in clients],
        "client_class_counts": [np.bincount(c.train.label_array(), minlength=num_classes).tolist() for c in clients],
        **res,
        "wall_time_sec": time.time() - t0,
    }
    record.update(extra_record)
    if args.trust_terms:
        record["trust_terms"] = list(terms)
    if repartition is not None:
        record["client_train_partition"] = {"rule": "iid_size_matched", **repartition}
    if args.client_optimizer_state != "fresh":
        record["client_optimizer_state"] = args.client_optimizer_state
    if args.no_test_eval:
        record["test_evaluation"] = "skipped (--no-test-eval)"
    if hasher is not None:
        record["round_hashes"] = hasher.rounds
        record["round_hash_time_sec"] = hasher.seconds  # included in train_time_sec / wall_time_sec
    if diagnostics is not None:
        dpath = out.with_name(f"{out.stem}_diagnostics.json")
        write_json(dpath, {"fold": fold, "seed": seed, "tag": tag, "trust": bool(cfg.trust), "kappa": cfg.trust_kappa,
                           "rounds": diagnostics})
        record["trust_diagnostics"] = {"file": dpath.name, "sha256": _sha256_file(dpath)}
    if args.save_checkpoint:  # model holds the selected round's state
        ckpt = out.with_suffix(".pt")
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ckpt)
        record["checkpoint"] = {"file": ckpt.name, "round": record["best_round"], "sha256_file": _sha256_file(ckpt)}
    if args.provenance:
        record["provenance"]["selected_state_sha256"] = state_hashes(model)  # model holds the selected round's state
        record["provenance"]["test_outputs_sha256"] = {
            name: {k: _sha256_bytes(np.ascontiguousarray(v).tobytes())
                   for k, v in (("probs", p["probs"]), ("pred", p["probs"].argmax(1)), ("labels", p["labels"]), ("uids", p["uids"]))}
            for name, p in preds.items()}
    write_json(out, record)
    if preds:
        np.savez_compressed(out.with_suffix(".npz"), **{f"{k}_{f}": v[f] for k, v in preds.items() for f in ("probs", "labels", "uids")})
    tf = {k: round(v.get("f1", float("nan")), 4) for k, v in record["test"].items()}
    print(f"DONE {tag} seed{seed}: best_round {record['best_round']} client-val F1 {record['best_client_val_f1']:.4f} "
          f"test F1 {tf} uplink {record['uplink_mb_total']:.1f}MB ({record['wall_time_sec']:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--work", type=Path, default=Path("results_v2"))
    ap.add_argument("--scenario", default="main", choices=["main", "mixed"])
    ap.add_argument("--dataset", nargs="+", default=["dataset1"])
    ap.add_argument("--method", nargs="+", default=["fedavg"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--stage", default="final")
    ap.add_argument("--model", default="", help="override model spec")
    ap.add_argument("--opts", default="", help="FLConfig overrides, e.g. 'topk=0.25,trust=0'")
    ap.add_argument("--tag", default="", help="result folder name (default: method)")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--partition", default="dirichlet", choices=["dirichlet", "iid"])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone_lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true",
                    help="deterministic backends; an op without a deterministic implementation raises (default: off)")
    ap.add_argument("--provenance", action="store_true",
                    help="record settings, environment, data/split and model-state hashes (default: off)")
    ap.add_argument("--round-hashes", action="store_true",
                    help="record per-round hashes of client updates, aggregation and global state (default: off)")
    ap.add_argument("--save-checkpoint", action="store_true",
                    help="save the selected round's model state next to the result JSON (default: off)")
    ap.add_argument("--client-train-partition", default="", choices=["", "iid_size_matched"],
                    help="diagnostic: re-deal client training images IID with sizes and validation sets kept (default: off)")
    ap.add_argument("--no-test-eval", action="store_true",
                    help="skip the final test pass; training and selection are unchanged (default: off)")
    ap.add_argument("--trust-terms", default="",
                    help="ablation: trust factors applied, subset of agree,stab,util (default: all)")
    ap.add_argument("--trust-diagnostics", action="store_true",
                    help="record observation-only trust diagnostics (shadow trust when trust=0) (default: off)")
    ap.add_argument("--client-optimizer-state", default="fresh", choices=list(CLIENT_OPTIMIZER_STATES),
                    help="diagnostic: 'persistent' keeps each client's own AdamW state across rounds (default: fresh)")
    args = ap.parse_args()
    configure_backends(args.deterministic)
    store = DataStore(args.root, args.work, device_from(args.device))
    args.env_info = environment_info(args.root) if args.provenance else None
    if args.limit:
        from .run_central import _limit_store

        _limit_store(store, args.limit)
    datasets = ["mixed"] if args.scenario == "mixed" else args.dataset
    for dataset in datasets:
        for method in args.method:
            for seed in args.seeds:
                run_cell(store, args, args.scenario, dataset, method, seed)


if __name__ == "__main__":
    main()
