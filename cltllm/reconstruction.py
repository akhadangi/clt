from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .hf import ModelBundle, choose_top2_tokens, encode_prompt
from .metrics import js_divergence_logits
from .utils import write_csv_rows


def _mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.ones((1, length), dtype=torch.long, device=device)


def run_reconstruction_equivalence(
    bundle: ModelBundle,
    prompts: list[dict[str, Any]],
    cfg: dict[str, Any],
    outdir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = int(cfg["reconstruction_prompt_count_core"])
    for prompt in prompts[:n]:
        inp = encode_prompt(bundle, prompt["text"], cfg["max_prompt_tokens"])
        discr = choose_top2_tokens(bundle, inp)
        plen = inp["input_ids"].shape[1]
        forced = torch.tensor([[discr["d1_id"]]], device=bundle.device)
        with torch.inference_mode():
            pre = bundle.model(**inp, use_cache=True, return_dict=True)
            cut = bundle.model(
                input_ids=forced,
                attention_mask=_mask(plen + 1, bundle.device),
                past_key_values=pre.past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            next_id = int(torch.argmax(cut.logits[0, -1]).item())
            nxt = torch.tensor([[next_id]], device=bundle.device)
            live = bundle.model(
                input_ids=nxt,
                attention_mask=_mask(plen + 2, bundle.device),
                past_key_values=cut.past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )

            # Detached informational record only: original token history is replayed.
            record = torch.cat([inp["input_ids"], forced], dim=1).detach().clone()
            rebuilt = bundle.model(
                input_ids=record,
                attention_mask=_mask(plen + 1, bundle.device),
                use_cache=True,
                return_dict=True,
            )
            rec = bundle.model(
                input_ids=nxt,
                attention_mask=_mask(plen + 2, bundle.device),
                past_key_values=rebuilt.past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )

        last_layer_live = live.hidden_states[-1][0, -1].float()
        last_layer_rec = rec.hidden_states[-1][0, -1].float()
        rows.append({
            "experiment": "reconstruction_equivalence",
            "model": bundle.spec.key,
            "prompt_id": prompt["id"],
            "next_token_id": next_id,
            "logit_js": js_divergence_logits(live.logits[0, -1], rec.logits[0, -1]),
            "hidden_max_abs": float((last_layer_live - last_layer_rec).abs().max().item()),
            "hidden_rmse": float(torch.sqrt(torch.mean((last_layer_live - last_layer_rec) ** 2)).item()),
            "live_C_protocol": 1,
            "live_N_protocol": 1,
            "reconstructed_C_protocol": 0,
            "reconstructed_N_protocol": 0,
            **discr,
        })
    write_csv_rows(rows, outdir / "reconstruction_equivalence.csv")
    return rows


def make_fresh_instance_reference(bundle: ModelBundle, prompt: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    inp = encode_prompt(bundle, prompt["text"], cfg["max_prompt_tokens"])
    discr = choose_top2_tokens(bundle, inp)
    forced = torch.tensor([[discr["d1_id"]]], device=bundle.device)
    plen = inp["input_ids"].shape[1]
    with torch.inference_mode():
        pre = bundle.model(**inp, use_cache=True, return_dict=True)
        cut = bundle.model(
            input_ids=forced,
            attention_mask=_mask(plen + 1, bundle.device),
            past_key_values=pre.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        next_id = int(torch.argmax(cut.logits[0, -1]).item())
        nxt = torch.tensor([[next_id]], device=bundle.device)
        live = bundle.model(
            input_ids=nxt,
            attention_mask=_mask(plen + 2, bundle.device),
            past_key_values=cut.past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
    return {
        "prompt_id": prompt["id"],
        "prompt_ids": inp["input_ids"].detach().cpu(),
        "forced_id": discr["d1_id"],
        "next_id": next_id,
        "live_logits": live.logits[0, -1].detach().float().cpu(),
        "live_hidden": live.hidden_states[-1][0, -1].detach().float().cpu(),
        **discr,
    }


def compare_fresh_instance(bundle: ModelBundle, ref: dict[str, Any]) -> dict[str, Any]:
    prompt_ids = ref["prompt_ids"].to(bundle.device)
    forced = torch.tensor([[ref["forced_id"]]], device=bundle.device)
    next_id = torch.tensor([[ref["next_id"]]], device=bundle.device)
    plen = prompt_ids.shape[1]
    with torch.inference_mode():
        # Fresh model realization receives only detached token records.
        rec_ids = torch.cat([prompt_ids, forced], dim=1)
        rebuilt = bundle.model(
            input_ids=rec_ids,
            attention_mask=_mask(plen + 1, bundle.device),
            use_cache=True,
            return_dict=True,
        )
        out = bundle.model(
            input_ids=next_id,
            attention_mask=_mask(plen + 2, bundle.device),
            past_key_values=rebuilt.past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
    h = out.hidden_states[-1][0, -1].detach().float().cpu()
    logits = out.logits[0, -1].detach().float().cpu()
    return {
        "experiment": "fresh_instance_reconstruction",
        "model": bundle.spec.key,
        "prompt_id": ref["prompt_id"],
        "logit_js": js_divergence_logits(ref["live_logits"], logits),
        "hidden_max_abs": float((ref["live_hidden"] - h).abs().max().item()),
        "hidden_rmse": float(torch.sqrt(torch.mean((ref["live_hidden"] - h) ** 2)).item()),
        "source_realization_terminated": 1,
        "reconstructed_C_protocol": 0,
        "reconstructed_N_protocol": 0,
        "d1_id": ref["d1_id"],
        "d1_text": ref["d1_text"],
        "d2_id": ref["d2_id"],
        "d2_text": ref["d2_text"],
    }
