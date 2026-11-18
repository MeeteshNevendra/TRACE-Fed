from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_name(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", spec.replace(":", "__").replace(",", "_"))


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_default))
    tmp.replace(path)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def device_from(arg: str) -> torch.device:
    if arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(arg)


def count_params(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
