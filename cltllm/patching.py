from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .hf import ModelBundle, choose_top2_tokens, encode_prompt
from .metrics import js_divergence_logits, normalized_mediation
from .utils import stable_seed, write_csv_rows


def _greedy_suffix(bundle: ModelBundle, prompt_ids: torch.Tensor, forced_id: int, length: int) -> list[int]:
    ids = torch.cat([prompt_ids, torch.tensor([[forced_id]], device=bundle.device)], dim=1)
    suffix: list[int] = []
    with torch.inference_mode():
        for _ in range(length):
            out = bundle.model(input_ids=ids, use_cache=False, return_dict=True)
            tid = int(torch.argmax(out.logits[0, -1]).item())
            suffix.append(tid)
            ids = torch.cat([ids, torch.tensor([[tid]], device=bundle.device)], dim=1)
    return suffix


def _forward_sequence(bundle: ModelBundle, ids: torch.Tensor, hidden: bool = False):
    with torch.inference_mode():
        return bundle.model(
            input_ids=ids,
            use_cache=False,
            output_hidden_states=hidden,
            return_dict=True,
        )


def run_activation_patching(
    bundle: ModelBundle,
    prompts: list[dict[str, Any]],
    cfg: dict[str, Any],
    outdir: Path,
    prompt_limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    layer_stride = int(cfg["patch_layer_stride"])
    patch_layers = list(range(0, bundle.num_layers, layer_stride))
    if patch_layers[-1] != bundle.num_layers - 1:
        patch_layers.append(bundle.num_layers - 1)

    for prompt in prompts[:prompt_limit]:
        inputs = encode_prompt(bundle, prompt["text"], cfg["max_prompt_tokens"])
        discr = choose_top2_tokens(bundle, inputs)
        prompt_ids = inputs["input_ids"]
        plen = prompt_ids.shape[1]
        suffix = _greedy_suffix(bundle, prompt_ids, discr["d2_id"], int(cfg["patch_horizon"]))
        suffix_tensor = torch.tensor([suffix], device=bundle.device, dtype=torch.long)
        seq1 = torch.cat([prompt_ids, torch.tensor([[discr["d1_id"]]], device=bundle.device), suffix_tensor], dim=1)
        seq2 = torch.cat([prompt_ids, torch.tensor([[discr["d2_id"]]], device=bundle.device), suffix_tensor], dim=1)

        out1 = _forward_sequence(bundle, seq1, hidden=True)
        out2 = _forward_sequence(bundle, seq2, hidden=False)
        log1 = out1.logits[0, -1].float()
        log2 = out2.logits[0, -1].float()
        base_js = js_divergence_logits(log1, log2)
        effect_supported = int(base_js >= float(cfg.get("patch_js_epsilon", 1e-5)))

        for layer_idx in patch_layers:
            # hidden_states[layer_idx] is the residual stream ENTERING
            # this transformer layer.
            #
            # Patching the layer output after a full-sequence forward creates
            # a late-layer artifact: later positions in that same layer have
            # already attended to the unpatched past state.  We therefore
            # intervene with a forward pre-hook so the patched state enters
            # attention/KV computation for this layer.
            source_h = out1.hidden_states[layer_idx].detach()
            layer_module = bundle.layers[layer_idx]

            for rel_pos in cfg["patch_positions"]:
                abs_pos = plen + int(rel_pos)
                if abs_pos >= seq2.shape[1]:
                    continue

                def pre_hook(_module, args, kwargs, *, pos=abs_pos, src=source_h):
                    if args and torch.is_tensor(args[0]) and args[0].ndim == 3:
                        h = args[0].clone()
                        h[:, pos, :] = src[:, pos, :].to(h.dtype)
                        return (h, *args[1:]), kwargs

                    if (
                        "hidden_states" in kwargs
                        and torch.is_tensor(kwargs["hidden_states"])
                    ):
                        new_kwargs = dict(kwargs)
                        h = new_kwargs["hidden_states"].clone()
                        h[:, pos, :] = src[:, pos, :].to(h.dtype)
                        new_kwargs["hidden_states"] = h
                        return args, new_kwargs

                    raise RuntimeError(
                        "Could not locate layer hidden_states "
                        "for activation patch"
                    )

                handle = layer_module.register_forward_pre_hook(
                    pre_hook,
                    with_kwargs=True,
                )

                try:
                    patched = _forward_sequence(
                        bundle,
                        seq2,
                        hidden=False,
                    )
                finally:
                    handle.remove()
                patched_js = js_divergence_logits(log1, patched.logits[0, -1].float())
                if effect_supported:
                    raw, clipped = normalized_mediation(base_js, patched_js)
                else:
                    raw = clipped = float("nan")
                rows.append({
                    "experiment": "activation_patching",
                    "model": bundle.spec.key,
                    "prompt_id": prompt["id"],
                    "layer": layer_idx,
                    "patch_rel_pos": int(rel_pos),
                    "base_js": base_js,
                    "base_effect_supported": effect_supported,
                    "patched_js": patched_js,
                    "mediation_raw": raw,
                    "mediation_clipped": clipped,
                    **discr,
                })

    write_csv_rows(rows, outdir / "activation_patching.csv")
    return rows
