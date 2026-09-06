from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .utils import stable_seed


@dataclass
class GovernanceSnapshot:
    state: torch.Tensor
    A: torch.Tensor
    B: torch.Tensor
    layer_index: int
    rank: int
    eta: float
    gain: float
    decay: float


class LowRankGovernance:
    """A small endogenous, persistent low-rank governance state.

    The residual intervention is
        delta h = gain * (h A) S B^T
    where A and B are fixed random orthonormal-ish projections and S is
    a persistent rank x rank state. After a discrimination d in {-1,+1},
    S updates from the model's own hidden state:
        S <- decay*S + eta*d*outer(tanh(hA), tanh(hA))

    No external labels or gradients are used.
    """

    def __init__(self, hidden_size: int, rank: int, layer_index: int, eta: float, gain: float, decay: float, device: torch.device, seed: int):
        self.hidden_size = hidden_size
        self.rank = rank
        self.layer_index = layer_index
        self.eta = float(eta)
        self.gain = float(gain)
        self.decay = float(decay)
        self.device = device
        g = torch.Generator(device=device)
        g.manual_seed(stable_seed(seed, "governance", hidden_size, rank, layer_index))
        A = torch.randn(hidden_size, rank, generator=g, device=device, dtype=torch.float32)
        B = torch.randn(hidden_size, rank, generator=g, device=device, dtype=torch.float32)
        self.A = torch.linalg.qr(A, mode="reduced").Q.to(torch.float16)
        self.B = torch.linalg.qr(B, mode="reduced").Q.to(torch.float16)
        self.state: torch.Tensor | None = None
        self.enabled = True
        self.handle: Any | None = None

    def reset(self, batch: int) -> None:
        self.state = torch.zeros(batch, self.rank, self.rank, device=self.device, dtype=torch.float32)

    def attach(self, layer: torch.nn.Module) -> None:
        if self.handle is not None:
            raise RuntimeError("governance hook already attached")

        def hook(_module, _inputs, output):
            if not self.enabled or self.state is None:
                return output
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = None
            if h.ndim != 3:
                return output
            batch = h.shape[0]
            S = self.state
            if S.shape[0] == 1 and batch > 1:
                S = S.expand(batch, -1, -1)
            elif S.shape[0] != batch:
                raise RuntimeError(f"governance batch mismatch {S.shape[0]} vs {batch}")
            A = self.A.to(dtype=h.dtype)
            B = self.B.to(dtype=h.dtype)
            u = torch.einsum("bsd,dr->bsr", h, A)
            mid = torch.einsum("bsr,brq->bsq", u, S.to(h.dtype))
            delta = torch.einsum("bsq,dq->bsd", mid, B) * self.gain
            h2 = h + delta
            return (h2, *rest) if rest is not None else h2

        self.handle = layer.register_forward_hook(hook)

    def detach(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def update(self, hidden_last: torch.Tensor, decision_sign: float | torch.Tensor) -> None:
        if self.state is None:
            self.reset(hidden_last.shape[0])
        h = hidden_last.detach().float()
        A = self.A.float()
        u = torch.tanh(h @ A)
        outer = torch.einsum("br,bq->brq", u, u)
        if not torch.is_tensor(decision_sign):
            sign = torch.full((h.shape[0], 1, 1), float(decision_sign), device=h.device)
        else:
            sign = decision_sign.to(h.device, torch.float32).reshape(-1, 1, 1)
        if self.state.shape[0] == 1 and h.shape[0] > 1:
            self.state = self.state.expand(h.shape[0], -1, -1).clone()
        self.state = self.decay * self.state + self.eta * sign * outer

    def snapshot_cpu(self) -> GovernanceSnapshot:
        if self.state is None:
            raise RuntimeError("governance state uninitialized")
        return GovernanceSnapshot(
            state=self.state.detach().cpu().clone(),
            A=self.A.detach().cpu().clone(),
            B=self.B.detach().cpu().clone(),
            layer_index=self.layer_index,
            rank=self.rank,
            eta=self.eta,
            gain=self.gain,
            decay=self.decay,
        )

    @classmethod
    def from_snapshot(cls, snap: GovernanceSnapshot, device: torch.device) -> "LowRankGovernance":
        obj = cls(
            hidden_size=snap.A.shape[0], rank=snap.rank, layer_index=snap.layer_index,
            eta=snap.eta, gain=snap.gain, decay=snap.decay, device=device, seed=0,
        )
        obj.A = snap.A.to(device=device, dtype=torch.float16)
        obj.B = snap.B.to(device=device, dtype=torch.float16)
        obj.state = snap.state.to(device=device, dtype=torch.float32)
        return obj

    def state_hash(self) -> str:
        import hashlib
        if self.state is None:
            return "none"
        return hashlib.sha256(self.state.detach().cpu().numpy().tobytes()).hexdigest()
