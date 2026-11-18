"""End-to-end, resumable experiment pipeline for TRACE-Fed v2.

Stages (each skips cells whose result JSON already exists):
  search_central  validation-only search, seeds 100/101 (never the final seeds)
  search_fl       validation-only tuning of FL hyper-parameters (seed 100)
  select          picks configs from *validation* scores only -> results_v2/selection.json
  central         final centralized comparison, seeds 0-4 (dataset2: 5 CV folds)
  fl_main         final FL comparison, 9 methods x 2 datasets x 5 seeds
  ablation        TRACE-Net component ablations (central) and TRACE-Fed ablations (FL)
  hetero          non-IID severity and client-count sweeps
  robust          noisy-client robustness
  mixed           mixed-site (dataset1 + dataset2) binary federation
  cross           cross-dataset binary generalisation
Usage:
  python -m trace_fed.pipeline --stages all
  python -m trace_fed.pipeline --stages search_central search_fl select
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PY = sys.executable
WORK = Path("results_v2")
SEARCH_SEEDS = [100, 101]
FINAL_SEEDS = [0, 1, 2, 3, 4]
SWEEP_SEEDS = [0, 1, 2]
BASELINES = ["resnet18", "resnet50", "efficientnet_b0", "convnext_tiny", "vit_small", "swin_tiny"]
BASELINE_LRS = [1e-4, 3e-5]
TRACE_SEARCH = [
    "trace",
    "trace:depth=1",
    "trace:dim=384",
    "trace:backbone=resnet50.a1_in1k",
    "trace:backbone_lr=5e-05",
    "trace:peft=1",
    "trace:peft=1,lora_r=16",
]
FL_SEARCH = {
    "fedprox": [("mu0.001", "prox_mu=0.001"), ("mu0.01", "prox_mu=0.01"), ("mu0.1", "prox_mu=0.1")],
    "fedadam": [("eta0.001", "server_lr=0.001"), ("eta0.003", "server_lr=0.003"), ("eta0.01", "server_lr=0.01")],
    "moon": [("mu0.1", "moon_mu=0.1"), ("mu1", "moon_mu=1.0")],
    "trace": [
        ("scaffold_k0.1", "drift=scaffold,topk=0.1"),
        ("prox_k0.1", "drift=prox,topk=0.1"),
        ("none_k0.1", "drift=none,topk=0.1"),
        ("prox_k0.05", "drift=prox,topk=0.05"),
        ("prox_k0.25", "drift=prox,topk=0.25"),
    ],
}
FL_METHODS = ["fedavg", "fedprox", "scaffold", "fedadam", "fedbn", "fedper", "fedrep", "moon", "trace"]


def sh(args: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {' '.join(args)}", flush=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(args)}\n")
        f.flush()
        rc = subprocess.call([PY, "-m", *args], stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        raise RuntimeError(f"command failed ({rc}); see {log}")


def central(models, datasets, seeds, stage, extra=(), log=None):
    sh(["trace_fed.run_central", "--stage", stage, "--dataset", *datasets, "--model", *models,
        "--seeds", *map(str, seeds), *extra], log or WORK / "logs" / f"{stage}.log")


def fl(methods, datasets, seeds, stage, tag="", opts="", extra=(), log=None):
    args = ["trace_fed.run_fl", "--stage", stage, "--dataset", *datasets, "--method", *methods,
            "--seeds", *map(str, seeds), *extra]
    if tag:
        args += ["--tag", tag]
    if opts:
        args += ["--opts", opts]
    sh(args, log or WORK / "logs" / f"fl_{stage}.log")


# ----------------------------------------------------------------------------- selection
def _mean_val(folder: Path, key: str = "val") -> float:
    vals = []
    for f in folder.glob("seed*.json"):
        r = json.loads(f.read_text())
        vals.append(r[key]["f1"] if key == "val" else r[key])
    return float(np.mean(vals)) if vals else float("nan")


def select() -> dict:
    from .utils import safe_name

    root = WORK / "runs" / "central" / "search" / "dataset1"
    sel = {"rule": "highest mean validation macro-F1 (dataset1, search seeds 100/101); test data never read", "central": {}, "fl": {}}
    # baselines: choose backbone lr per architecture
    for b in BASELINES:
        scores = {lr: _mean_val(root / safe_name(f"{b}:backbone_lr={lr}")) for lr in BASELINE_LRS}
        best_lr = max(scores, key=lambda k: scores[k])
        sel["central"][b] = {"spec": f"{b}:backbone_lr={best_lr}", "val_scores": scores}
    scores = {s: _mean_val(root / safe_name(s)) for s in TRACE_SEARCH}
    full = {s: v for s, v in scores.items() if "peft" not in s}
    best = max(full, key=lambda k: full[k])
    sel["central"]["trace"] = {"spec": best, "val_scores": scores}
    peft = {s: v for s, v in scores.items() if "peft" in s}
    sel["trace_fl_model"] = _merge_spec(best, max(peft, key=lambda k: peft[k]))
    # FL hyper-parameters
    fl_root = WORK / "runs" / "fl" / "search" / "dataset1" / "dirichlet0.5_c5"
    for method, grid in FL_SEARCH.items():
        s = {tag: _mean_val(fl_root / safe_name(f"{method}_{tag}"), key="best_client_val_f1") for tag, _ in grid}
        best_tag = max(s, key=lambda k: s[k])
        sel["fl"][method] = {"opts": dict(grid)[best_tag], "val_scores": s}
    (WORK / "selection.json").write_text(json.dumps(sel, indent=2))
    print(json.dumps(sel, indent=2))
    return sel


def _merge_spec(full_spec: str, peft_spec: str) -> str:
    """TRACE-Fed model: architecture of the selected central model + PEFT settings of the best PEFT run."""
    from .registry import parse_spec

    _, a = parse_spec(full_spec)
    _, b = parse_spec(peft_spec)
    a.pop("backbone_lr", None)
    a.update(b)
    return "trace:" + ",".join(f"{k}={v}" for k, v in a.items())


def load_selection() -> dict:
    p = WORK / "selection.json"
    if not p.exists():
        raise FileNotFoundError("run the 'select' stage first")
    return json.loads(p.read_text())


# ----------------------------------------------------------------------------- stages
def stage_search_central():
    specs = [f"{b}:backbone_lr={lr}" for b in BASELINES for lr in BASELINE_LRS]
    central(TRACE_SEARCH + specs, ["dataset1"], SEARCH_SEEDS, "search")


def stage_search_fl():
    from .registry import parse_spec

    for method, grid in FL_SEARCH.items():
        for tag, opts in grid:
            extra = []
            if method == "trace":
                # PEFT model chosen from the central search (validation only)
                scores = {s: _mean_val(WORK / "runs" / "central" / "search" / "dataset1" / _sn(s)) for s in TRACE_SEARCH if "peft" in s}
                extra = ["--model", max(scores, key=lambda k: scores[k])]
            fl([method], ["dataset1"], [100], "search", tag=f"{method}_{tag}", opts=opts, extra=extra)
    _ = parse_spec


def _sn(s):
    from .utils import safe_name

    return safe_name(s)


def _fl_args(sel: dict, method: str) -> tuple[str, list[str]]:
    opts = sel["fl"].get(method, {}).get("opts", "")
    extra = ["--model", sel["trace_fl_model"]] if method == "trace" else ["--model", sel["central"]["trace"]["spec"]]
    return opts, extra


def stage_central():
    sel = load_selection()
    specs = [v["spec"] for v in sel["central"].values()]
    central(specs, ["dataset1", "dataset2"], FINAL_SEEDS, "final", extra=["--save_ckpt"])


def stage_fl_main():
    sel = load_selection()
    for method in FL_METHODS:
        opts, extra = _fl_args(sel, method)
        fl([method], ["dataset1", "dataset2"], FINAL_SEEDS, "final", opts=opts, extra=extra)


def stage_ablation():
    sel = load_selection()
    base = sel["central"]["trace"]["spec"]
    sep = "," if ":" in base else ":"
    variants = [f"{base}{sep}{v}" for v in ["mixer=ssm", "mixer=attn", "mixer=sum", "pooling=gap", "loc_head=0", "xc=0", "seg=0,xc=0"]]
    central(variants, ["dataset1", "dataset2"], FINAL_SEEDS, "ablation")
    t_opts, t_extra = _fl_args(sel, "trace")
    fl_variants = [
        ("trace_no_trust", "trust=0"),
        ("trace_no_ef", "error_feedback=0"),
        ("trace_no_compression", "topk=1.0,qbits=32"),
        ("trace_no_proto", "prototypes=0"),
    ]
    # drift correction is only ablated if the selected TRACE-Fed configuration actually uses it
    if "drift=none" not in t_opts:
        fl_variants.append(("trace_no_drift", "drift=none"))
    for tag, o in fl_variants:
        fl(["trace"], ["dataset1", "dataset2"], SWEEP_SEEDS, "ablation", tag=tag, opts=",".join(filter(None, [t_opts, o])), extra=t_extra)
    # TRACE-Fed protocol on the full (non-PEFT) model: isolates the adapter design
    fl(["trace"], ["dataset1", "dataset2"], SWEEP_SEEDS, "ablation", tag="trace_full_model", opts=t_opts,
       extra=["--model", sel["central"]["trace"]["spec"]])


def stage_hetero():
    sel = load_selection()
    methods = ["fedavg", "fedprox", "scaffold", "trace"]
    for method in methods:
        opts, extra = _fl_args(sel, method)
        for part in [("--partition", "iid"), ("--alpha", "0.1"), ("--alpha", "1.0")]:
            fl([method], ["dataset1"], SWEEP_SEEDS, "hetero", opts=opts, extra=extra + list(part))
        for n in ["3", "10"]:
            fl([method], ["dataset1"], SWEEP_SEEDS, "clients", opts=opts, extra=extra + ["--clients", n])


def stage_robust():
    sel = load_selection()
    noise = "noisy_clients=1,noise_rate=0.8"
    for method, tag in [("fedavg", "fedavg"), ("fedprox", "fedprox"), ("trace", "trace"), ("trace", "trace_no_trust")]:
        opts, extra = _fl_args(sel, method)
        o = ",".join(filter(None, [opts, noise] + (["trust=0"] if tag == "trace_no_trust" else [])))
        fl([method], ["dataset1"], SWEEP_SEEDS, "robust", tag=tag, opts=o, extra=extra)


def stage_mixed():
    sel = load_selection()
    for method in ["fedavg", "fedprox", "scaffold", "fedbn", "trace"]:
        opts, extra = _fl_args(sel, method)
        fl([method], ["mixed"], SWEEP_SEEDS, "final", opts=opts, extra=extra + ["--scenario", "mixed"])


def stage_cross():
    sel = load_selection()
    best_baseline = _best_central_baseline(sel)
    central([sel["central"]["trace"]["spec"], best_baseline], ["dataset1", "dataset2"], SWEEP_SEEDS, "cross",
            extra=["--label_mode", "binary"])


def _best_central_baseline(sel: dict) -> str:
    best, score = None, -1
    for b in BASELINES:
        s = max(sel["central"][b]["val_scores"].values())
        if s > score:
            best, score = sel["central"][b]["spec"], s
    return best


STAGES = {
    "search_central": stage_search_central,
    "search_fl": stage_search_fl,
    "select": select,
    "central": stage_central,
    "fl_main": stage_fl_main,
    "ablation": stage_ablation,
    "hetero": stage_hetero,
    "robust": stage_robust,
    "mixed": stage_mixed,
    "cross": stage_cross,
}


def wait_for_gpu(pattern: str, poll: int = 120) -> None:
    """Block while another process whose command line contains `pattern` is running."""
    if not pattern:
        return
    while True:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"@(Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -like '*{pattern}*' -and "
             f"$_.CommandLine -notlike '*trace_fed*' -and $_.CommandLine -notlike '*Get-CimInstance*' }}).Count"],
            capture_output=True, text=True).stdout.strip()
        if out in ("", "0"):
            return
        print(f"{time.strftime('%H:%M:%S')} waiting for GPU: '{pattern}' still running", flush=True)
        time.sleep(poll)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["all"])
    ap.add_argument("--wait_for", default="", help="wait until no process command line contains this text")
    args = ap.parse_args()
    stages = list(STAGES) if args.stages == ["all"] else args.stages
    wait_for_gpu(args.wait_for)
    for s in stages:
        t0 = time.time()
        print(f"\n######## stage {s} ########", flush=True)
        STAGES[s]()
        print(f"######## stage {s} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
