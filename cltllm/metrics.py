from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch


def _standardize_pair(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.cat([x, y], dim=0).float()
    mu = z.mean(dim=0, keepdim=True)
    sd = z.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    return (x.float() - mu) / sd, (y.float() - mu) / sd


def coarse_grain(x: torch.Tensor, mode: str, seed: int) -> torch.Tensor:
    if mode == "identity":
        return x
    d = x.shape[-1]
    if mode in {"rp50", "rp75"}:
        frac = 0.50 if mode == "rp50" else 0.75
        target = max(8, min(int(round(d * frac)), 1024))
        g = torch.Generator(device=x.device)
        g.manual_seed(seed)
        R = torch.randn(d, target, generator=g, device=x.device, dtype=torch.float32) / math.sqrt(target)
        return x.float() @ R
    if mode == "block4":
        block = 4
        use = (d // block) * block
        if use < block:
            return x
        return x[..., :use].reshape(*x.shape[:-1], use // block, block).mean(dim=-1)
    raise ValueError(f"Unknown coarse graining: {mode}")


def sliced_wasserstein(x: torch.Tensor, y: torch.Tensor, n_proj: int, seed: int) -> float:
    x, y = _standardize_pair(x, y)
    d = x.shape[-1]
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)
    dirs = torch.randn(d, n_proj, generator=g, device=x.device, dtype=torch.float32)
    dirs = dirs / dirs.norm(dim=0, keepdim=True).clamp_min(1e-8)
    xp = x @ dirs
    yp = y @ dirs
    # Equal sample sizes in this suite.
    xp = torch.sort(xp, dim=0).values
    yp = torch.sort(yp, dim=0).values
    return float((xp - yp).abs().mean().item())


def _median_bandwidth(z: torch.Tensor, max_n: int = 128) -> float:
    if z.shape[0] > max_n:
        z = z[:max_n]
    with torch.no_grad():
        D = torch.cdist(z.float(), z.float())
        vals = D[D > 0]
        if vals.numel() == 0:
            return 1.0
        return float(vals.median().item()) + 1e-6


def mmd_rff(x: torch.Tensor, y: torch.Tensor, n_features: int, seed: int) -> float:
    x, y = _standardize_pair(x, y)
    z = torch.cat([x, y], dim=0)
    bw = _median_bandwidth(z)
    d = x.shape[-1]
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)
    W = torch.randn(d, n_features, generator=g, device=x.device, dtype=torch.float32) / bw
    b = torch.rand(n_features, generator=g, device=x.device, dtype=torch.float32) * (2 * math.pi)
    scale = math.sqrt(2.0 / n_features)
    phix = scale * torch.cos(x @ W + b)
    phiy = scale * torch.cos(y @ W + b)
    diff = phix.mean(dim=0) - phiy.mean(dim=0)
    return float(diff.square().sum().sqrt().item())


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = _standardize_pair(x, y)
    # Normalize Euclidean scale so the statistic is comparable across admissible
    # dimensional coarse-grainings.
    scale = math.sqrt(max(1, x.shape[-1]))
    x = x / scale
    y = y / scale
    xy = torch.cdist(x, y).mean()
    xx = torch.cdist(x, x).mean()
    yy = torch.cdist(y, y).mean()
    val = 2 * xy - xx - yy
    return float(torch.clamp(val, min=0).item())


def hidden_metric(name: str, x: torch.Tensor, y: torch.Tensor, *, seed: int, swd_projections: int, mmd_features: int) -> float:
    if name == "swd":
        return sliced_wasserstein(x, y, swd_projections, seed)
    if name == "mmd_rff":
        return mmd_rff(x, y, mmd_features, seed)
    if name == "energy":
        return energy_distance(x, y)
    raise ValueError(name)


def js_divergence_from_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    p = p.float().clamp_min(eps)
    q = q.float().clamp_min(eps)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl1 = (p * (p.log() - m.log())).sum()
    kl2 = (q * (q.log() - m.log())).sum()
    return float((0.5 * (kl1 + kl2)).item())


def js_divergence_logits(a: torch.Tensor, b: torch.Tensor) -> float:
    return js_divergence_from_probs(torch.softmax(a.float(), -1), torch.softmax(b.float(), -1))


def normalized_mediation(base_js: float, patched_js: float, eps: float = 1e-12) -> tuple[float, float]:
    raw = 1.0 - patched_js / max(base_js, eps)
    return raw, min(1.0, max(0.0, raw))
