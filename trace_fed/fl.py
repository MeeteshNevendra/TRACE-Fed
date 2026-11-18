"""Federated learning: FedAvg, FedProx, SCAFFOLD, FedAdam, FedBN, FedPer, FedRep,
MOON and TRACE-Fed, on a shared simulation loop.

Design notes
* Every client keeps 15 % of its partition as a private validation set. The
  server never sees data: the round used for the final model is the one with
  the best sample-weighted mean of client-reported validation macro-F1.
* Communication is counted per round in bits, uplink and downlink separately,
  including control variates (SCAFFOLD), prototypes and sparse indices.
* Personalised methods (FedBN/FedPer/FedRep) are evaluated with each client's
  personal parameters on the common test set and averaged over clients.

TRACE-Fed (Trust-Robust Adapter Communication-Efficient):
  1. frozen pretrained backbone + LoRA adapters; only adapters and the TRACE
     head are trained and communicated,
  2. top-k sparsification + 8-bit quantisation with client-side error feedback,
  3. drift correction on the shared parameters (SCAFFOLD or proximal),
  4. trust-weighted aggregation: sample count x update agreement with the
     coordinate-wise median update x local validation utility x measured
     explanation stability,
  5. global class prototypes that pull local features toward a shared space.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Split, concat_splits, flip_batch, prepare_batch
from .engine import TrainConfig, compute_metrics, make_optimizer, predict, set_lr, train_steps

NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
HEAD_TOKENS = ("classifier", "head_norm", "loc.", "net.head", "net.fc", "net.classifier")


@dataclass
class FLConfig:
    method: str = "fedavg"
    rounds: int = 30
    clients: int = 5
    local_epochs: int = 1
    partition: str = "dirichlet"  # "dirichlet" | "iid"
    alpha: float = 0.5
    local_val_frac: float = 0.15
    min_client_samples: int = 20
    prox_mu: float = 0.01
    moon_mu: float = 1.0
    moon_tau: float = 0.5
    server_lr: float = 1e-2  # FedAdam
    server_tau: float = 1e-3
    fedrep_head_epochs: int = 1
    # TRACE-Fed switches (also usable as ablations)
    topk: float = 0.1
    qbits: int = 8
    error_feedback: bool = True
    down_compress: bool = True  # TRACE: compress the broadcast too (server-side error feedback)
    trust: bool = True
    trust_kappa: float = 10.0
    drift: str = "scaffold"  # "scaffold" | "prox" | "none"
    prototypes: bool = True
    proto_weight: float = 0.1
    # robustness scenario
    noisy_clients: int = 0
    noise_rate: float = 0.0
    verbose: bool = True
    extra: dict = field(default_factory=dict)


# =========================================================================== partitioning
def partition(labels: np.ndarray, cfg: FLConfig, rng: np.random.Generator) -> list[np.ndarray]:
    n_cls = int(labels.max()) + 1
    if cfg.partition == "iid":
        parts = [[] for _ in range(cfg.clients)]
        for c in range(n_cls):
            idx = rng.permutation(np.flatnonzero(labels == c))
            for k, chunk in enumerate(np.array_split(idx, cfg.clients)):
                parts[k].extend(chunk.tolist())
        return [np.array(sorted(p)) for p in parts]
    for _ in range(1000):
        parts = [[] for _ in range(cfg.clients)]
        for c in range(n_cls):
            idx = rng.permutation(np.flatnonzero(labels == c))
            props = rng.dirichlet([cfg.alpha] * cfg.clients)
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= cfg.min_client_samples:
            return [np.array(sorted(p)) for p in parts]
    raise RuntimeError("could not draw a Dirichlet partition with the minimum client size")


def split_local_val(labels: np.ndarray, frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Per-client stratified hold-out (at least one val sample per class with >=3 samples)."""
    tr, va = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        n_val = int(round(frac * len(idx)))
        if n_val == 0 and len(idx) >= 3:
            n_val = 1
        va.extend(idx[:n_val].tolist())
        tr.extend(idx[n_val:].tolist())
    return np.array(sorted(tr)), np.array(sorted(va))


@dataclass
class Client:
    cid: int
    train: Split
    val: Split
    site: str = ""
    personal: dict = field(default_factory=dict)
    c_local: dict = field(default_factory=dict)
    ef: dict = field(default_factory=dict)
    prev_state: dict | None = None
    noisy: bool = False


IID_SIZE_MATCHED_SEED_OFFSET = 7919  # fixed; part of the rule below


def iid_size_matched_train(clients: list["Client"], seed: int) -> dict:
    """Re-deal the clients' training images class-stratified (IID) while keeping every client's training size
    and its validation set unchanged (diagnostic partition; opt-in via run_fl --client-train-partition).

    Rule: pool the clients' training images in client order. For every class except the last, allocate
    N_c * n_i / N images to client i by largest remainder (ties by client order); the last class fills each
    client up to its size n_i. Within each class, permute the pooled images with
    numpy.default_rng(seed + 7919) and deal them to clients in order. Returns the allocation table.
    """
    pool = concat_splits([c.train for c in clients])
    labels = pool.label_array()
    sizes = np.array([len(c.train) for c in clients])
    classes = np.unique(labels)
    alloc = np.zeros((len(clients), len(classes)), dtype=int)
    remaining = sizes.copy()
    for j, cls in enumerate(classes[:-1]):
        q = (labels == cls).sum() * sizes / sizes.sum()
        base = np.floor(q).astype(int)
        order = np.argsort(-(q - base), kind="stable")
        base[order[: int((labels == cls).sum()) - int(base.sum())]] += 1
        alloc[:, j] = base
        remaining = remaining - base
    alloc[:, -1] = remaining
    rng = np.random.default_rng(seed + IID_SIZE_MATCHED_SEED_OFFSET)
    parts: list[list[int]] = [[] for _ in clients]
    for j, cls in enumerate(classes):
        idx = rng.permutation(np.flatnonzero(labels == cls))
        if alloc[:, j].sum() != len(idx) or (alloc[:, j] < 0).any():
            raise ValueError("size-matched IID allocation is infeasible")
        start = 0
        for i in range(len(clients)):
            parts[i].extend(idx[start:start + alloc[i, j]].tolist())
            start += alloc[i, j]
    for c, p in zip(clients, parts):
        c.train = pool.subset(np.array(sorted(p)))
    return {"sizes": sizes.tolist(), "class_counts": alloc.tolist(), "classes": classes.tolist()}


def build_clients(train_sites: list[tuple[str, Split, int]], cfg: FLConfig, seed: int) -> list[Client]:
    """train_sites: (site name, train split, number of clients drawn from it)."""
    rng = np.random.default_rng(seed)
    clients: list[Client] = []
    for site, split, n_clients in train_sites:
        site_cfg = copy.copy(cfg)
        site_cfg.clients = n_clients
        parts = partition(split.label_array(), site_cfg, rng)
        for pos in parts:
            sub = split.subset(pos)
            tr, va = split_local_val(sub.label_array(), cfg.local_val_frac, rng)
            clients.append(Client(len(clients), sub.subset(tr), sub.subset(va), site))
    for c in clients[: cfg.noisy_clients]:
        c.noisy = True
        labels = c.train.labels.clone()
        flip = torch.rand(labels.shape, generator=torch.Generator().manual_seed(seed + 99)).to(labels.device) < cfg.noise_rate
        rand_labels = torch.randint(0, c.train.num_classes, labels.shape, generator=torch.Generator().manual_seed(seed + 7)).to(labels.device)
        labels[flip] = rand_labels[flip]
        c.train = Split(c.train.store, c.train.idx, labels, c.train.num_classes, c.train.class_names, c.train.domain)
    return clients


# =========================================================================== key sets
def norm_keys(model: nn.Module) -> set[str]:
    keys = set()
    for mname, mod in model.named_modules():
        if isinstance(mod, NORM_TYPES):
            for pname, _ in list(mod.named_parameters(recurse=False)) + list(mod.named_buffers(recurse=False)):
                keys.add(f"{mname}.{pname}" if mname else pname)
    return keys


def head_keys(model: nn.Module) -> set[str]:
    return {k for k in model.state_dict() if any(t in k for t in HEAD_TOKENS)}


def float_keys(model: nn.Module) -> list[str]:
    return [k for k, v in model.state_dict().items() if torch.is_floating_point(v)]


def key_sets(model: nn.Module, method: str) -> tuple[list[str], set[str]]:
    """(shared keys communicated every round, personal keys kept on the client)."""
    all_float = float_keys(model)
    if method == "trace":
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        return [k for k in all_float if k in trainable], set()
    if method == "fedbn":
        personal = norm_keys(model)
    elif method in {"fedper", "fedrep"}:
        personal = head_keys(model)
    else:
        personal = set()
    return [k for k in all_float if k not in personal], personal


# =========================================================================== compression
def compress(delta: torch.Tensor, topk: float, qbits: int) -> tuple[torch.Tensor, int]:
    """Top-k magnitude sparsification + symmetric uniform quantisation. Returns (decoded, bits)."""
    n = delta.numel()
    flat = delta.flatten().float()
    if topk < 1.0:
        k = max(1, int(math.ceil(topk * n)))
        idx = torch.topk(flat.abs(), k, sorted=False).indices
        vals = flat[idx]
        index_bits = min(k * 32, n)  # explicit indices or a bitmap, whichever is smaller
    else:
        k, idx, vals, index_bits = n, None, flat, 0
    if qbits < 32:
        qmax = 2 ** (qbits - 1) - 1
        scale = vals.abs().max().clamp(min=1e-12) / qmax
        vals = torch.round(vals / scale).clamp(-qmax, qmax) * scale
        value_bits = k * qbits + 32
    else:
        value_bits = k * 32
    if idx is None:
        out = vals
    else:
        out = torch.zeros_like(flat)
        out[idx] = vals
    return out.view_as(delta).to(delta.dtype), value_bits + index_bits


# =========================================================================== explanation stability
@torch.no_grad()
def explanation_stability(model: nn.Module, split: Split, tcfg: TrainConfig) -> float:
    """Mean Pearson correlation between CAM(x) and unflip(CAM(flip(x))) of the predicted class, mapped to [0, 1]."""
    model.eval()
    vals = []
    for raw in split.batches(64, shuffle=False):
        batch = prepare_batch(raw, tcfg.image_size, train=False)
        with torch.autocast(device_type=batch["image"].device.type, dtype=torch.bfloat16, enabled=tcfg.amp):
            o1 = model(batch["image"])
            o2 = model(flip_batch(batch["image"]))
        if "cam_all" in o1:
            y = o1["logits"].argmax(1)
            a = o1["cam_all"][torch.arange(len(y)), y].float()
            b = flip_batch(o2["cam_all"][torch.arange(len(y)), y].float())
        else:
            a = o1["feature_map"].float().mean(1)
            b = flip_batch(o2["feature_map"].float().mean(1))
        a = a.flatten(1) - a.flatten(1).mean(1, keepdim=True)
        b = b.flatten(1) - b.flatten(1).mean(1, keepdim=True)
        r = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)
        vals.append(r.cpu())
    if not vals:
        return 0.5
    return float(((torch.cat(vals).mean() + 1) / 2).clamp(0, 1))


# =========================================================================== client optimiser state
CLIENT_OPTIMIZER_STATES = ("fresh", "persistent")


def _optimizer_state_copy(opt: torch.optim.Optimizer, model: nn.Module) -> dict:
    """Detached copy of the optimiser's per-parameter state (e.g. AdamW moments and step counters), keyed by
    parameter name. Param groups (learning rates and other settings) are not included."""
    out = {}
    for name, p in model.named_parameters():
        st = opt.state.get(p)
        if st:
            out[name] = {k: v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v) for k, v in st.items()}
    return out


def _load_optimizer_state(opt: torch.optim.Optimizer, model: nn.Module, saved: dict) -> None:
    """Put a copy of a state saved by _optimizer_state_copy into a freshly built optimiser over the same named
    parameters. The optimiser's param groups are left exactly as built."""
    for name, p in model.named_parameters():
        if name in saved:
            opt.state[p] = {k: v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v) for k, v in saved[name].items()}


@torch.no_grad()
def _optimizer_state_digest(opt: torch.optim.Optimizer, model: nn.Module) -> tuple[str, list[float] | None]:
    """Diagnostics: SHA-256 of the per-parameter optimiser state in parameter order (the hash of nothing when the
    state is empty), and the [min, max] step counter over parameters that have state (None when empty)."""
    h = hashlib.sha256()
    steps = []
    for name, p in model.named_parameters():
        st = opt.state.get(p)
        if not st:
            continue
        for k in sorted(st):
            v = st[k]
            if torch.is_tensor(v):
                h.update(f"{name}|{k}|{v.dtype}|{tuple(v.shape)}|".encode())
                h.update(v.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
            else:
                h.update(f"{name}|{k}|{v!r}|".encode())
        if "step" in st:
            steps.append(float(st["step"]))
    return h.hexdigest(), ([min(steps), max(steps)] if steps else None)


# =========================================================================== main loop
def run_federated(model: nn.Module, clients: list[Client], val_report: list[Split], test_sets: dict[str, Split],
                  tcfg: TrainConfig, cfg: FLConfig, seed: int,
                  observer: Callable[[int, dict], None] | None = None,
                  diagnostics: list | None = None,
                  trust_terms: tuple[str, ...] = ("agree", "stab", "util"),
                  client_optimizer_state: str = "fresh") -> dict:
    """observer (default None): called once per round, after aggregation, with that round's client updates,
    error-feedback states, aggregate, server residual, new global shared state and prototypes. It must only
    read them (used to hash every round for reproducibility checks).

    diagnostics (default None; TRACE-Fed only): a list that receives one observation record per round:
    pre/post local-validation losses and predictions, update geometry against the coordinate-wise median,
    trust factors and weights, server-side validation predictions, aggregate/broadcast/residual norms, each
    client's local change before compression (SHA-256 and norm) and the client and server compression errors. With
    trust off it also computes the trust statistics as a shadow that is never applied. Everything it adds is
    an evaluation pass, a read of existing tensors, or arithmetic on copies: no gradients, no random numbers,
    no change to model, optimiser, client order or aggregation.

    trust_terms (default: all three): the trust factors applied (TRACE-Fed with trust on). See trust_factors.

    client_optimizer_state (default "fresh", the method as published): "fresh" builds a new local AdamW for every
    client in every round. "persistent" (diagnostic intervention on the optimisation policy) still builds the
    optimiser fresh, so its param groups (round-wise learning rates, betas, eps, weight decay) are exactly as in
    "fresh", and then loads that client's own per-parameter state (moments and step counters) saved at the end of
    its previous local training. Model parameters are always synchronised to the received global state first.
    States are kept per client and never shared."""
    if client_optimizer_state not in CLIENT_OPTIMIZER_STATES:
        raise ValueError(f"client_optimizer_state must be one of {CLIENT_OPTIMIZER_STATES}")
    if client_optimizer_state != "fresh" and cfg.method == "fedrep":
        raise ValueError("client_optimizer_state is not defined for FedRep's own local optimisers")
    persistent_opt = client_optimizer_state == "persistent"
    diag = diagnostics is not None and cfg.method == "trace"
    shadow = diag and not cfg.trust  # trust statistics computed but never applied
    method = cfg.method
    device = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    global_model = model
    worker = copy.deepcopy(model)
    aux = copy.deepcopy(model) if method == "moon" else None  # previous local model
    shared, personal_keys = key_sets(global_model, method)
    gstate = {k: v.detach().clone() for k, v in global_model.state_dict().items()}
    trainable_names = {n for n, p in global_model.named_parameters() if p.requires_grad}
    shared_trainable = [k for k in shared if k in trainable_names]
    numel_shared = sum(gstate[k].numel() for k in shared)
    use_scaffold = method == "scaffold" or (method == "trace" and cfg.drift == "scaffold")
    use_prox = method == "fedprox" or (method == "trace" and cfg.drift == "prox")
    compress_on = method == "trace" and (cfg.topk < 1.0 or cfg.qbits < 32)
    c_global = {k: torch.zeros_like(gstate[k]) for k in shared_trainable} if use_scaffold else {}
    adam_m = {k: torch.zeros_like(gstate[k]) for k in shared} if method == "fedadam" else {}
    adam_v = {k: torch.zeros_like(gstate[k]) for k in shared} if method == "fedadam" else {}
    protos: torch.Tensor | None = None
    num_classes = clients[0].train.num_classes
    for c in clients:
        if personal_keys:
            c.personal = {k: gstate[k].clone() for k in personal_keys}
        if use_scaffold:
            c.c_local = {k: torch.zeros_like(gstate[k]) for k in shared_trainable}

    down_compress = compress_on and cfg.down_compress
    server_ef = {k: torch.zeros_like(gstate[k], dtype=torch.float32) for k in shared} if down_compress else {}
    broadcast_bits = numel_shared * 32  # round 1: the initial shared parameters are sent dense
    c_broadcast_bits = sum(v.numel() for v in c_global.values()) * 32

    history = []
    best = {"score": -1.0}
    up_bits_total = down_bits_total = 0
    opt_states: dict[int, dict] = {}  # client_optimizer_state="persistent": each client's own optimiser state
    t0 = time.time()
    for rnd in range(1, cfg.rounds + 1):
        deltas, weights_n, infos = [], [], []
        new_c_locals = []
        proto_sum = proto_cnt = None
        up_bits = down_bits = 0
        dround = {"round": rnd, "clients": []} if diag else None
        raw_deltas = [] if diag else None  # diagnostics only: each client's local change before compression
        for c in clients:
            # ---------------------------------------------------------- download
            worker.load_state_dict(gstate)
            if c.personal:
                worker.load_state_dict(c.personal, strict=False)
            down_bits += broadcast_bits + c_broadcast_bits
            if protos is not None:
                down_bits += protos.numel() * 32
            pre_nll = pre_pred = None
            if method == "trace" and (cfg.trust or shadow):
                pre_pred = predict(worker, c.val, tcfg)
                pre_nll = compute_metrics(pre_pred, num_classes)["nll"]
            # ---------------------------------------------------------- local objective
            g_params = {n: p.detach().clone() for n, p in worker.named_parameters() if n in shared_trainable} if (use_prox or method == "moon") else {}
            if method == "moon":
                aux.load_state_dict(c.prev_state if c.prev_state is not None else gstate)
                gmodel_frozen = copy.deepcopy(worker).eval()
            opt = make_optimizer(worker, tcfg)
            if persistent_opt and c.cid in opt_states:
                _load_optimizer_state(opt, worker, opt_states[c.cid])
            if diag:  # diagnostics only: the optimiser state this client starts from (empty when fresh)
                opt_in = _optimizer_state_digest(opt, worker)
            name_of = {id(p): n for n, p in worker.named_parameters()}
            lr_used: dict[str, float] = {}

            def extra_loss(out, batch, mdl, _c=c):
                loss = out["logits"].new_zeros(())
                if use_prox:
                    mu = cfg.prox_mu
                    loss = loss + 0.5 * mu * sum(((p - g_params[n]) ** 2).sum() for n, p in mdl.named_parameters() if n in g_params)
                if method == "moon":
                    with torch.no_grad():
                        zg = gmodel_frozen(batch["image"])["features"].float()
                        zp = aux.eval()(batch["image"])["features"].float()
                    z = out["features"].float()
                    pos = F.cosine_similarity(z, zg) / cfg.moon_tau
                    neg = F.cosine_similarity(z, zp) / cfg.moon_tau
                    loss = loss + cfg.moon_mu * (-pos + torch.logsumexp(torch.stack([pos, neg]), 0)).mean()
                if method == "trace" and cfg.prototypes and protos is not None:
                    z = F.normalize(out["features"].float(), dim=1)
                    p = F.normalize(protos[batch["label"]], dim=1)
                    valid = protos.abs().sum(1)[batch["label"]] > 0
                    if bool(valid.any()):
                        loss = loss + cfg.proto_weight * (1 - (z * p).sum(1))[valid].mean()
                return loss

            post_step = None
            if use_scaffold:
                # SCAFFOLD correction applied in parameter space (w <- w - lr (c - c_i)) after each
                # optimiser step, so it is consistent with an adaptive local optimiser.
                corr = {k: (c_global[k] - c.c_local[k]) for k in shared_trainable}

                @torch.no_grad()
                def post_step(mdl, optim, _corr=corr):
                    for g in optim.param_groups:
                        for p in g["params"]:
                            n = name_of.get(id(p))
                            if n in _corr:
                                p.add_(_corr[n].to(p.dtype), alpha=-g["lr"])
                                lr_used[n] = g["lr"]

            progress = lambda s, r=rnd: (r - 1) / cfg.rounds  # cosine decay across rounds
            steps = examples = 0
            if method == "fedrep":
                steps += _fedrep_local(worker, c, tcfg, cfg, gen, personal_keys, progress)
            else:
                for _ in range(cfg.local_epochs):
                    stats = train_steps(worker, c.train, opt, tcfg, gen, progress_fn=progress,
                                        extra_loss=extra_loss if (use_prox or method in {"moon", "trace"}) else None,
                                        post_step=post_step)
                    steps += stats["steps"]
                    examples += stats.get("examples", 0)
            if diag:  # diagnostics only: the optimiser state after local training, and the settings it used
                opt_out = _optimizer_state_digest(opt, worker)
                opt_groups = [{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups]
            if persistent_opt:
                opt_states[c.cid] = _optimizer_state_copy(opt, worker)
            lstate = worker.state_dict()
            # ---------------------------------------------------------- upload
            delta = {}
            # diagnostics only: squared norms of local change, compressor input, residual, sent update, sent update minus local change
            raw_sq = inp_sq = res_sq = sent_sq = dev_sq = 0.0
            raw_hash = hashlib.sha256() if diag else None  # diagnostics only: hash of the local change before compression
            raw_delta = {} if diag else None
            for k in shared:
                d = lstate[k].detach().float() - gstate[k].float()
                if diag:
                    raw_delta[k] = d
                    raw_sq += float(d.pow(2).sum())
                    raw_hash.update(f"{k}|{d.dtype}|{tuple(d.shape)}|".encode())
                    raw_hash.update(d.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
                if compress_on:
                    if cfg.error_feedback:
                        d = d + c.ef.get(k, torch.zeros_like(d))
                    if diag:
                        inp_sq += float(d.pow(2).sum())
                    dq, bits = compress(d, cfg.topk, cfg.qbits)
                    if cfg.error_feedback:
                        c.ef[k] = d - dq
                        if diag:
                            res_sq += float(c.ef[k].pow(2).sum())
                    d = dq
                    up_bits += bits
                else:
                    up_bits += d.numel() * 32
                if diag:
                    sent_sq += float(d.pow(2).sum())
                    dev_sq += float((d - raw_delta[k]).pow(2).sum())
                delta[k] = d
            if use_scaffold and steps > 0:
                # option II of SCAFFOLD: c_i+ = c_i - c + (w_global - w_local) / (K * lr)
                new_c = {}
                for k in shared_trainable:
                    lr_k = max(lr_used.get(k, tcfg.lr), 1e-12)
                    new_c[k] = c.c_local[k] - c_global[k] + (gstate[k].float() - lstate[k].float()) / (steps * lr_k)
                    if compress_on:  # TRACE sends the control-variate change through the same compressor
                        dc = new_c[k] - c.c_local[k]
                        if cfg.error_feedback:
                            dc = dc + c.ef.get("c::" + k, torch.zeros_like(dc))
                        dcq, bits = compress(dc, cfg.topk, cfg.qbits)
                        if cfg.error_feedback:
                            c.ef["c::" + k] = dc - dcq
                        new_c[k] = c.c_local[k] + dcq
                        up_bits += bits
                    else:
                        up_bits += new_c[k].numel() * 32
                new_c_locals.append((c, new_c))
            if personal_keys:
                c.personal = {k: lstate[k].detach().clone() for k in personal_keys}
            if method == "moon":
                c.prev_state = {k: v.detach().clone() for k, v in lstate.items()}
            info = {"cid": c.cid, "site": c.site, "n": len(c.train), "noisy": c.noisy, "steps": steps}
            if method == "trace":
                if cfg.trust or shadow:
                    post_pred = predict(worker, c.val, tcfg)
                    post_nll = compute_metrics(post_pred, num_classes)["nll"]
                    utility = (pre_nll - post_nll) / max(pre_nll, 1e-6)  # relative local-val NLL change
                    stability = explanation_stability(worker, c.val, tcfg)
                    if cfg.trust:
                        info["utility"] = utility
                        info["stability"] = stability
                    if diag:
                        dround["clients"].append({
                            "cid": c.cid, "n": len(c.train), "steps": steps, "examples": examples,
                            "local_change_sha256": raw_hash.hexdigest(),
                            "local_change_l2": math.sqrt(raw_sq), "compressor_input_l2": math.sqrt(inp_sq),
                            "ef_residual_l2": math.sqrt(res_sq), "sent_update_l2": math.sqrt(sent_sq),
                            "sent_minus_local_change_l2": math.sqrt(dev_sq),
                            "opt_state_in_sha256": opt_in[0], "opt_step_in": opt_in[1],
                            "opt_state_out_sha256": opt_out[0], "opt_step_out": opt_out[1],
                            "opt_group_lrs": [g["lr"] for g in opt_groups],
                            "opt_group_settings_sha256": hashlib.sha256(
                                json.dumps(opt_groups, sort_keys=True, default=str).encode()).hexdigest(),
                            "pre_nll": pre_nll, "post_nll": post_nll,
                            "utility": utility, "stability": stability,
                            "f_util_exp_overflow": bool(-cfg.trust_kappa * utility > np.log(np.finfo(np.float64).max)),
                            "pre_val": _val_summary(pre_pred), "post_val": _val_summary(post_pred)})
                if cfg.prototypes:
                    ps, pc = _class_feature_sums(worker, c.train, tcfg, num_classes)
                    proto_sum = ps if proto_sum is None else proto_sum + ps
                    proto_cnt = pc if proto_cnt is None else proto_cnt + pc
                    up_bits += ps.numel() * 32 + pc.numel() * 32
            deltas.append(delta)
            if diag:
                raw_deltas.append(raw_delta)
            weights_n.append(float(len(c.train)))
            infos.append(info)
        # -------------------------------------------------------------- aggregation
        weights = np.array(weights_n)
        if method == "trace" and cfg.trust:
            factors = trust_factors(deltas, shared_trainable, infos, cfg.trust_kappa, trust_terms)
            weights = weights * factors
        weights = weights / weights.sum()
        for i, w in zip(infos, weights):
            i["weight"] = float(w)
        avg = {k: sum(float(w) * d[k] for w, d in zip(weights, deltas)) for k in shared}
        if method == "fedadam":
            for k in shared:
                adam_m[k] = 0.9 * adam_m[k] + 0.1 * avg[k]
                adam_v[k] = 0.99 * adam_v[k] + 0.01 * avg[k] ** 2
                gstate[k] = (gstate[k].float() + cfg.server_lr * adam_m[k] / (adam_v[k].sqrt() + cfg.server_tau)).to(gstate[k].dtype)
        elif down_compress:
            # one compressed broadcast shared by all clients; the server keeps the residual (error feedback)
            broadcast_bits = 0
            broadcast_sq = s_in_sq = b_dev_sq = 0.0  # diagnostics only: broadcast, compressor input, broadcast - aggregate
            for k in shared:
                s = avg[k] + server_ef[k]
                sq, bits = compress(s, cfg.topk, cfg.qbits)
                server_ef[k] = s - sq
                gstate[k] = (gstate[k].float() + sq).to(gstate[k].dtype)
                broadcast_bits += bits
                if diag:
                    broadcast_sq += float(sq.float().pow(2).sum())
                    s_in_sq += float(s.float().pow(2).sum())
                    b_dev_sq += float((sq.float() - avg[k].float()).pow(2).sum())
        else:
            broadcast_bits = numel_shared * 32
            for k in shared:
                gstate[k] = (gstate[k].float() + avg[k]).to(gstate[k].dtype)
        if use_scaffold and new_c_locals:
            for k in shared_trainable:
                c_global[k] = c_global[k] + sum(nc[k] - cl.c_local[k] for cl, nc in new_c_locals) / len(clients)
            for cl, nc in new_c_locals:
                cl.c_local = nc
        if proto_sum is not None:
            protos = proto_sum / proto_cnt.clamp(min=1).unsqueeze(1)
        up_bits_total += up_bits
        down_bits_total += down_bits
        if observer is not None:
            observer(rnd, {"client_updates": deltas, "client_error_feedback": [c.ef for c in clients],
                           "aggregate": avg, "server_error_feedback": server_ef,
                           "global_shared": {k: gstate[k] for k in shared},
                           "prototype_sums": {"proto_sum": proto_sum, "proto_cnt": proto_cnt} if proto_sum is not None else {},
                           "prototypes": protos})
        if diag:  # observation only: computed after the aggregation and never fed back
            n_arr = np.array(weights_n)
            dround["sample_weights"] = (n_arr / n_arr.sum()).tolist()
            dround["weights_applied"] = weights.tolist()
            dround["trust_applied"] = bool(cfg.trust)
            if shadow:  # the trust rule on copies of this round's statistics; the result is recorded, not used
                sh_infos = [dict(i, utility=d["utility"], stability=d["stability"]) for i, d in zip(infos, dround["clients"])]
                sh_factors = trust_factors(deltas, shared_trainable, sh_infos, cfg.trust_kappa)
                sh_w = n_arr * sh_factors
                dround["shadow_weights"] = (sh_w / sh_w.sum()).tolist()
                src = sh_infos
            else:
                src = infos
            dround["factors"] = [{k: v for k, v in i.items() if k in ("agreement", "f_agree", "f_stab", "f_util", "trust")
                                  or k.endswith("_unused")} for i in src]
            dround["geometry"] = _agreement_geometry(deltas, shared_trainable)
            dround["aggregate_l2"] = float(torch.sqrt(sum(avg[k].float().pow(2).sum() for k in shared)))
            dround["broadcast_l2"] = float(np.sqrt(broadcast_sq)) if down_compress else None
            dround["server_residual_l2"] = float(torch.sqrt(sum(server_ef[k].pow(2).sum() for k in shared))) if down_compress else None
            dround["server_compressor_input_l2"] = float(np.sqrt(s_in_sq)) if down_compress else None
            dround["broadcast_minus_aggregate_l2"] = float(np.sqrt(b_dev_sq)) if down_compress else None
            raw_avg = {k: sum(float(w) * d[k] for w, d in zip(weights, raw_deltas)) for k in shared}
            dround["local_change_aggregate_l2"] = float(torch.sqrt(sum(raw_avg[k].float().pow(2).sum() for k in shared)))
            del raw_avg
        # -------------------------------------------------------------- server-side model selection from client reports
        # Clients report confusion-matrix counts on their private validation data; the server
        # computes one pooled macro-F1 (robust to clients whose validation set lacks a class).
        global_model.load_state_dict(gstate)
        vals, conf = [], np.zeros((num_classes, num_classes), dtype=np.int64)
        for ci, c in enumerate(clients):
            if c.personal:
                worker.load_state_dict(gstate)
                worker.load_state_dict(c.personal, strict=False)
                mdl = worker
            else:
                mdl = global_model
            vpred = predict(mdl, c.val, tcfg)
            m = compute_metrics(vpred, num_classes)
            conf += np.array(m["confusion"])
            vals.append((len(c.val), m["f1"]))
            if diag:
                dround["clients"][ci]["server_val"] = _val_summary(vpred)
        score = _macro_f1_from_confusion(conf)
        row = {"round": rnd, "val_f1_clients": score, "client_val_f1": [f for _, f in vals],
               "val_confusion": conf.tolist(),  # pooled client-validation counts (rows true, columns predicted)
               "uplink_mb": up_bits / 8 / 2**20, "downlink_mb": down_bits / 8 / 2**20, "clients": infos,
               "time": time.time() - t0}
        history.append(row)
        if diag:
            diagnostics.append(dround)
        if cfg.verbose:
            w = " ".join(f"{i['weight']:.2f}" for i in infos)
            print(f"  [{method}] round {rnd:>2} client-val F1 {score:.4f} up {row['uplink_mb']:.2f}MB weights [{w}] {row['time']:.0f}s", flush=True)
        if score > best["score"]:
            best = {"score": score, "round": rnd, "gstate": {k: v.clone() for k, v in gstate.items()},
                    "personal": [dict(c.personal) for c in clients]}
    # ------------------------------------------------------------------ final test (once)
    global_model.load_state_dict(best["gstate"])
    test = {}
    for name, split in test_sets.items():
        if personal_keys:
            per_client = []
            for c, pers in zip(clients, best["personal"]):
                worker.load_state_dict(best["gstate"])
                worker.load_state_dict(pers, strict=False)
                per_client.append(compute_metrics(predict(worker, split, tcfg, with_maps=True), split.num_classes))
            test[name] = _mean_metrics(per_client)
            test[name]["per_client"] = per_client
        else:
            pred = predict(global_model, split, tcfg, with_maps=True)
            test[name] = compute_metrics(pred, split.num_classes)
            test[name]["_pred"] = pred
    return {
        "method": method,
        "config": asdict(cfg),
        "best_round": best["round"],
        "best_client_val_f1": best["score"],
        "history": history,
        "test": test,
        "uplink_mb_total": up_bits_total / 8 / 2**20,
        "downlink_mb_total": down_bits_total / 8 / 2**20,
        "shared_params": int(numel_shared),
        "train_time_sec": time.time() - t0,
    }


def _fedrep_local(worker, c, tcfg, cfg, gen, head, progress) -> int:
    """FedRep: train the personal head with the body frozen, then the body with the head frozen."""
    steps = 0
    for phase, epochs in (("head", cfg.fedrep_head_epochs), ("body", cfg.local_epochs)):
        for n, p in worker.named_parameters():
            p.requires_grad_((n in head) == (phase == "head"))
        opt = make_optimizer(worker, tcfg)
        for _ in range(epochs):
            steps += train_steps(worker, c.train, opt, tcfg, gen, progress_fn=progress)["steps"]
    for p in worker.parameters():
        p.requires_grad_(True)
    return steps


def coordinate_median(flat: torch.Tensor) -> torch.Tensor:
    """Coordinate-wise median over clients (dim 0 of a clients x P tensor), values only.

    Deterministic mode off (the default): flat.median(dim=0).values, exactly as before.

    Deterministic mode on (torch.use_deterministic_algorithms(True), set by run_fl --deterministic): CUDA
    median raises there because its indices output is non-deterministic. The same values come from a stable
    sort along dim 0 at index (K - 1) // 2, K = number of clients: the middle value for odd K and the lower
    of the two middle values for even K, which is torch.median's convention. Non-finite values keep
    torch.median's behaviour: +/-inf are ordered like any other value, and a coordinate that has a NaN in
    any client returns NaN (a sort alone would move the NaN last and could return a finite value). NaNs
    also raise a warning, so they cannot pass unnoticed.
    """
    if not torch.are_deterministic_algorithms_enabled():
        return flat.median(dim=0).values
    med = torch.sort(flat, dim=0, stable=True).values[(flat.shape[0] - 1) // 2]
    nan_cols = torch.isnan(flat).any(dim=0)
    if bool(nan_cols.any()):
        warnings.warn(f"coordinate_median: {int(nan_cols.sum())} coordinate(s) contain NaN; "
                      "returning NaN there, as torch.median does", RuntimeWarning, stacklevel=2)
        med = med.masked_fill(nan_cols, float("nan"))
    return med


def _val_summary(pred: dict) -> dict:
    """Diagnostics: one client's validation probabilities, predicted classes, labels and confusion counts."""
    probs, labels = pred["probs"], pred["labels"]
    yhat = probs.argmax(1)
    conf = np.zeros((probs.shape[1], probs.shape[1]), dtype=np.int64)
    np.add.at(conf, (labels, yhat), 1)
    return {"probs": probs.tolist(), "pred": yhat.tolist(), "labels": labels.tolist(), "confusion": conf.tolist()}


@torch.no_grad()
def _agreement_geometry(deltas: list[dict], keys: list[str]) -> dict:
    """Diagnostics: the geometry behind the agreement score, using the same median and cosine as
    _median_agreement: raw cosine before clamping, update norms and nonzero counts, support overlap and sign
    agreement with the coordinate-wise median update, share of each update's energy on the median's support,
    and pairwise cosines between client updates."""
    flat = torch.stack([torch.cat([d[k].flatten() for k in keys]) for d in deltas])  # clients x P
    med = coordinate_median(flat)
    raw = F.cosine_similarity(flat, med.unsqueeze(0).expand_as(flat), dim=1)
    nz, mnz = flat != 0, (med != 0).unsqueeze(0)
    both = nz & mnz
    prod = flat * med.unsqueeze(0)
    sq = flat.pow(2)
    norms = flat.norm(dim=1)
    unit = flat / norms.clamp_min(1e-30).unsqueeze(1)
    return {"P": int(flat.shape[1]), "raw_cosine": raw.tolist(), "update_l2": norms.tolist(),
            "update_nnz": nz.sum(1).tolist(), "median_l2": float(med.norm()), "median_nnz": int(mnz.sum()),
            "overlap_nnz": both.sum(1).tolist(), "overlap_same_sign": (both & (prod > 0)).sum(1).tolist(),
            "overlap_opposite_sign": (both & (prod < 0)).sum(1).tolist(),
            "dot_positive": prod.clamp_min(0).sum(1).tolist(), "dot_negative": prod.clamp_max(0).sum(1).tolist(),
            "energy_on_median_support": ((sq * mnz).sum(1) / sq.sum(1).clamp_min(1e-30)).tolist(),
            "pairwise_cosine": (unit @ unit.T).tolist()}


def _median_agreement(deltas: list[dict], keys: list[str]) -> np.ndarray:
    flat = torch.stack([torch.cat([d[k].flatten() for k in keys]) for d in deltas])  # clients x P
    med = coordinate_median(flat)
    cos = F.cosine_similarity(flat, med.unsqueeze(0).expand_as(flat), dim=1)
    return cos.clamp(min=0).cpu().numpy()


TRUST_TERMS = ("agree", "stab", "util")


def trust_factors(deltas: list[dict], keys: list[str], infos: list[dict], kappa: float,
                  terms: tuple[str, ...] = TRUST_TERMS) -> np.ndarray:
    """Multiplicative trust in (0, 1]: it only discounts clients that look anomalous.

    agreement : cosine of the update with the coordinate-wise median update, relative to the cohort median
    utility   : relative change of the client's own validation NLL after local training (worse -> discount)
    stability : flip-consistency of the client model's class-activation maps, relative to the cohort median
    Honest, typical clients get factor ~1, so the rule reduces to sample-size weighting (FedAvg).

    terms (default: all three) are the factors applied. A factor left out (ablation) is still computed exactly
    as above and recorded as "<factor>_unused", but enters the product as exactly 1; the floor and the weight
    normalisation are unchanged.
    """
    agree = _median_agreement(deltas, keys)
    stab = np.array([i["stability"] for i in infos])
    util = np.array([i["utility"] for i in infos])
    f_agree = np.clip(agree / max(np.median(agree), 1e-8), 0.0, 1.0)
    f_stab = np.clip(stab / max(np.median(stab), 1e-8), 0.0, 1.0)
    f_util = np.minimum(1.0, 2.0 / (1.0 + np.exp(-kappa * util)))
    unused = {}
    if "agree" not in terms:
        unused["f_agree"], f_agree = f_agree, np.ones_like(f_agree)
    if "stab" not in terms:
        unused["f_stab"], f_stab = f_stab, np.ones_like(f_stab)
    if "util" not in terms:
        unused["f_util"], f_util = f_util, np.ones_like(f_util)
    factors = np.maximum(f_agree * f_stab * f_util, 1e-3)
    for j, (i, a, fa, fs, fu, f) in enumerate(zip(infos, agree, f_agree, f_stab, f_util, factors)):
        i.update({"agreement": float(a), "f_agree": float(fa), "f_stab": float(fs), "f_util": float(fu), "trust": float(f)})
        for k, v in unused.items():
            i[f"{k}_unused"] = float(v[j])
    return factors


def _macro_f1_from_confusion(conf: np.ndarray) -> float:
    """Macro-F1 over classes present in the pooled validation labels."""
    tp = np.diag(conf).astype(float)
    support = conf.sum(1)
    pred = conf.sum(0)
    present = support > 0
    f1 = np.where(present, 2 * tp / np.maximum(support + pred, 1), 0.0)
    return float(f1[present].mean()) if present.any() else 0.0


@torch.no_grad()
def _class_feature_sums(model, split: Split, tcfg: TrainConfig, num_classes: int):
    model.eval()
    sums = counts = None
    for raw in split.batches(64, shuffle=False):
        batch = prepare_batch(raw, tcfg.image_size, train=False)
        with torch.autocast(device_type=batch["image"].device.type, dtype=torch.bfloat16, enabled=tcfg.amp):
            z = model(batch["image"])["features"].float()
        if sums is None:
            sums = torch.zeros(num_classes, z.size(1), device=z.device)
            counts = torch.zeros(num_classes, device=z.device)
        sums.index_add_(0, batch["label"], z)
        counts += torch.bincount(batch["label"], minlength=num_classes).float()
    return sums, counts


def _mean_metrics(rows: list[dict]) -> dict:
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        if len(vals) == len(rows):
            out[k] = float(np.mean(vals))
    return out


def pooled(splits: list[Split]) -> Split:
    return concat_splits(splits)
