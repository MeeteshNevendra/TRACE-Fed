"""Selective state-space mixer for 2D feature maps (pure PyTorch).

The scan is the Mamba-2 selective SSM with a scalar decay per head:

    h_t = exp(dt_t * A) h_{t-1} + dt_t * B_t x_t^T,      y_t = C_t h_t + D x_t

with input-dependent dt, B and C. For the short token sequences used here
(14 x 14 = 196) we evaluate its exact "state-space dual" form,

    y_t = sum_{s<=t} exp(sum_{r=s+1..t} dt_r A) (C_t . B_s) dt_s x_s,

which is a causally masked, decay-weighted product and needs no Python loop
over time. `selective_scan_reference` is the step-by-step recurrence used by
the unit test to check equivalence.

The 2D block scans the token grid in four orders (row-major, column-major and
both reversed), each with its own dt/B/C projection, and sums the results.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan_dual(x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """x: (b, L, h, p)  dt: (b, L, h)  A: (h,) negative  B, C: (b, L, n)  ->  (b, L, h, p)."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, dt, B, C = x.float(), dt.float(), B.float(), C.float()
        L = x.size(1)
        log_decay = torch.cumsum(dt * A.float(), dim=1).transpose(1, 2)  # (b, h, L), non-increasing
        seg = log_decay.unsqueeze(-1) - log_decay.unsqueeze(-2)  # (b, h, t, s) = sum_{s<r<=t} dt_r A
        causal = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
        decay = torch.exp(seg.masked_fill(~causal, float("-inf")))
        scores = torch.einsum("btn,bsn->bts", C, B).unsqueeze(1)  # (b, 1, t, s)
        return torch.einsum("bhts,bshp->bthp", decay * scores, x * dt.unsqueeze(-1))


def selective_scan_reference(x, dt, A, B, C):
    b, L, h, p = x.shape
    n = B.size(-1)
    state = x.new_zeros(b, h, p, n)
    ys = []
    for t in range(L):
        a = torch.exp(dt[:, t] * A).view(b, h, 1, 1)
        state = a * state + (dt[:, t].view(b, h, 1, 1) * x[:, t].unsqueeze(-1) * B[:, t].view(b, 1, 1, n))
        ys.append(torch.einsum("bhpn,bn->bhp", state, C[:, t]))
    return torch.stack(ys, dim=1)


def _scan_orders(h: int, w: int, device) -> list[torch.Tensor]:
    grid = torch.arange(h * w, device=device).view(h, w)
    row = grid.flatten()
    col = grid.t().flatten()
    return [row, row.flip(0), col, col.flip(0)]


class SelectiveSSM2D(nn.Module):
    """Four-direction selective SSM over a (b, h*w, dim) token grid."""

    def __init__(self, dim: int, expand: int = 2, head_dim: int = 64, d_state: int = 16, directions: int = 4,
                 dt_min: float = 1e-3, dt_max: float = 1e-1) -> None:
        super().__init__()
        self.inner = dim * expand
        assert self.inner % head_dim == 0
        self.heads = self.inner // head_dim
        self.head_dim = head_dim
        self.d_state = d_state
        self.directions = directions
        self.in_proj = nn.Linear(dim, 2 * self.inner)
        self.dwconv = nn.Conv2d(self.inner, self.inner, 3, padding=1, groups=self.inner)
        self.x_proj = nn.Linear(self.inner, directions * (self.heads + 2 * d_state), bias=False)
        # dt bias initialised so softplus(dt_bias) is log-uniform in [dt_min, dt_max] (as in Mamba)
        dt = torch.exp(torch.rand(directions, self.heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.linspace(1, 16, self.heads)).repeat(directions, 1))
        self.D = nn.Parameter(torch.ones(directions, self.heads))
        self.out_norm = nn.LayerNorm(self.inner)
        self.out_proj = nn.Linear(self.inner, dim)

    def forward(self, tokens: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        b, L, _ = tokens.shape
        h, w = hw
        x, z = self.in_proj(tokens).chunk(2, dim=-1)
        x = x.transpose(1, 2).reshape(b, self.inner, h, w)
        x = F.silu(self.dwconv(x)).flatten(2).transpose(1, 2)  # (b, L, inner)
        proj = self.x_proj(x).view(b, L, self.directions, self.heads + 2 * self.d_state)
        y = torch.zeros(b, L, self.heads, self.head_dim, device=x.device, dtype=torch.float32)
        xh = x.view(b, L, self.heads, self.head_dim)
        for k, order in enumerate(_scan_orders(h, w, x.device)[: self.directions]):
            dt_raw, Bk, Ck = proj[:, order, k].split([self.heads, self.d_state, self.d_state], dim=-1)
            dt = F.softplus(dt_raw.float() + self.dt_bias[k])
            A = -torch.exp(self.A_log[k].float())
            xs = xh[:, order]
            yk = selective_scan_dual(xs, dt, A, Bk, Ck) + xs.float() * self.D[k].view(1, 1, -1, 1)
            y[:, order] += yk  # scatter back to row-major positions
        y = y.reshape(b, L, self.inner).to(tokens.dtype)
        return self.out_proj(self.out_norm(y) * F.silu(z))
