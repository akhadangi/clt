from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .governance import LowRankGovernance
from .hf import ModelBundle, repeat_inputs
from .utils import stable_seed


@dataclass
class RolloutResult:
    hidden: dict[int, dict[int, torch.Tensor]]
    mean_probs: dict[int, torch.Tensor]
    token_ids: torch.Tensor
    forced_hidden: torch.Tensor | None
    final_cache: Any
    governance_hash: str | None


def _sample_gumbel(logits: torch.Tensor, temperature: float, seed: int) -> torch.Tensor:
    # Common-random-number categorical sample via Gumbel max.
    g = torch.Generator(device=logits.device)
    g.manual_seed(seed)
    u = torch.rand(logits.shape, generator=g, device=logits.device, dtype=torch.float32)
    u = u.clamp_(1e-7, 1 - 1e-7)
    noise = -torch.log(-torch.log(u))
    return torch.argmax(logits.float() / max(temperature, 1e-5) + noise, dim=-1)


def _full_attention_mask(batch: int, length: int, device: torch.device) -> torch.Tensor:
    return torch.ones((batch, length), dtype=torch.long, device=device)


def rollout_counterfactual(
    bundle: ModelBundle,
    prompt_inputs: dict[str, torch.Tensor],
    forced_token_id: int,
    *,
    n_rollouts: int,
    batch_size: int,
    max_horizon: int,
    capture_layers: list[int],
    temperature: float,
    seed_namespace: tuple,
    governance: LowRankGovernance | None = None,
    governance_decision_sign: float = 1.0,
    condition: str = "frozen_live",
) -> RolloutResult:
    # Zamba2 runs through the pure-PyTorch Mamba fallback on Iris.
    # The validated configuration uses microbatch 1 to avoid transient
    # memory exhaustion. This changes evaluation batching only; n_rollouts,
    # interventions, horizons, and estimators remain unchanged.
    if bundle.spec.key in {"zamba2_1p2b_base", "gemma3_4b_pt"}:
        batch_size = 1

    all_hidden: dict[int, dict[int, list[torch.Tensor]]] = {
        h: {l: [] for l in capture_layers} for h in range(1, max_horizon + 1)
    }
    probs_sum: dict[int, torch.Tensor] = {}
    all_tokens: list[torch.Tensor] = []
    forced_hiddens: list[torch.Tensor] = []
    final_cache = None

    for b0 in range(0, n_rollouts, batch_size):
        bsz = min(batch_size, n_rollouts - b0)
        inp = repeat_inputs(prompt_inputs, bsz)
        plen = inp["input_ids"].shape[1]
        if governance is not None:
            governance.reset(bsz)
            governance.enabled = True

        with torch.inference_mode():
            pre = bundle.model(
                **inp,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
            forced_ids = torch.full((bsz, 1), forced_token_id, dtype=torch.long, device=bundle.device)
            forced_out = bundle.model(
                input_ids=forced_ids,
                attention_mask=_full_attention_mask(bsz, plen + 1, bundle.device),
                past_key_values=pre.past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = forced_out.past_key_values
            if governance is not None:
                hforced = forced_out.hidden_states[governance.layer_index + 1][:, -1, :].detach()
                forced_hiddens.append(hforced.cpu())
                governance.update(hforced, governance_decision_sign)

                if condition == "persistent_copy":
                    _unused_copy = governance.snapshot_cpu()
                elif condition == "reconstructed":
                    # Detached record: token history + governance state. Rebuild sequence state
                    # from prompt+forced token with governance disabled, then restore a fresh
                    # governance object with numerically identical persistent state.
                    snap = governance.snapshot_cpu()
                    governance.enabled = False
                    full_ids = torch.cat([inp["input_ids"], forced_ids], dim=1)
                    rec = bundle.model(
                        input_ids=full_ids,
                        attention_mask=_full_attention_mask(bsz, plen + 1, bundle.device),
                        use_cache=True,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    past = rec.past_key_values
                    governance.enabled = True
                    # We retain the same hook object but replace its state tensor with a fresh allocation.
                    governance.state = snap.state.to(bundle.device, torch.float32).clone()

            logits = forced_out.logits[:, -1, :]
            token_hist: list[torch.Tensor] = []
            for step in range(1, max_horizon + 1):
                # logits at the start of this iteration are the distribution of the
                # next discrimination at exactly this future horizon. Accumulate it
                # before sampling so horizon=1 means the immediate next decision.
                probs = torch.softmax(logits.float(), dim=-1).sum(dim=0).detach().cpu()
                probs_sum[step] = probs_sum.get(step, torch.zeros_like(probs)) + probs
                sample_seed = stable_seed(*seed_namespace, b0, step)
                next_ids = _sample_gumbel(logits, temperature, sample_seed).unsqueeze(1)
                token_hist.append(next_ids.detach().cpu())
                out = bundle.model(
                    input_ids=next_ids,
                    attention_mask=_full_attention_mask(bsz, plen + 1 + step, bundle.device),
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                past = out.past_key_values
                logits = out.logits[:, -1, :]
                hs = out.hidden_states
                for layer in capture_layers:
                    all_hidden[step][layer].append(hs[layer + 1][:, -1, :].detach().cpu())

            all_tokens.append(torch.cat(token_hist, dim=1))
            final_cache = past

    hidden: dict[int, dict[int, torch.Tensor]] = {
        h: {l: torch.cat(parts, dim=0) for l, parts in per.items()} for h, per in all_hidden.items()
    }
    mean_probs = {h: v / float(n_rollouts) for h, v in probs_sum.items()}
    return RolloutResult(
        hidden=hidden,
        mean_probs=mean_probs,
        token_ids=torch.cat(all_tokens, dim=0),
        forced_hidden=torch.cat(forced_hiddens, dim=0) if forced_hiddens else None,
        final_cache=final_cache,
        governance_hash=governance.state_hash() if governance is not None else None,
    )
