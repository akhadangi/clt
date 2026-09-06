from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from .config import load_model_specs
from .governance import LowRankGovernance
from .hf import choose_top2_tokens, encode_prompt, load_model_bundle
from .metrics import js_divergence_logits
from .utils import dump_json, load_json, load_jsonl, stable_seed


def _spec(models: str, key: str):
    for s in load_model_specs(models):
        if s.key == key:
            return s
    raise KeyError(key)


def _mask(n: int, dev: torch.device):
    return torch.ones((1, n), dtype=torch.long, device=dev)


def _save_tensor_dict(path: Path, obj: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def _cache_path(state_path: Path, prompt_index: int) -> Path:
    return state_path.with_name(f"{state_path.stem}.p{prompt_index:03d}.cache.pt")


def prefix(args):
    """Create detached continuation records, then let this source process exit.

    One source process loads the model once and prepares ``prompt_count`` independent
    cut states.  For each prompt it stores (i) the exact sequence cache when that
    cache type is serializable, (ii) the token record as a portable fallback, and
    (iii) the endogenous governance snapshot.  It also computes the live successor
    reference *before* the source process exits.  The orchestrator launches the
    successor only after this function and therefore this OS process have ended.
    """
    prepared = load_json(args.prepared)["prepared"]
    spec = _spec(args.models, args.model_key)
    bundle = load_model_bundle(spec, prepared, device="cuda:0")
    prompts = load_jsonl(args.prompts)
    start = int(args.prompt_index)
    stop = min(len(prompts), start + int(args.prompt_count))
    selected = prompts[start:stop]
    if not selected:
        raise RuntimeError("No prompts selected for hard reconstruction")

    state_path = Path(args.state)
    records: list[dict[str, Any]] = []
    source_pid = os.getpid()

    for local_i, prompt in enumerate(selected):
        prompt_i = start + local_i
        inp = encode_prompt(bundle, prompt["text"], args.max_prompt_tokens)
        discr = choose_top2_tokens(bundle, inp)
        plen = inp["input_ids"].shape[1]
        gov_layer = min(
            bundle.num_layers - 1,
            max(0, int(round(args.gov_layer_fraction * (bundle.num_layers - 1)))),
        )
        gov = LowRankGovernance(
            bundle.hidden_size,
            args.gov_rank,
            gov_layer,
            args.gov_eta,
            args.gov_gain,
            1.0,
            bundle.device,
            stable_seed(20260905, args.model_key, prompt["id"], "hard"),
        )
        gov.attach(bundle.layers[gov_layer])
        gov.reset(1)
        cache_path = _cache_path(state_path, prompt_i)
        cache_serialized = False
        cache_error = ""
        try:
            with torch.inference_mode():
                pre = bundle.model(**inp, use_cache=True, return_dict=True)
                forced = torch.tensor([[discr["d1_id"]]], device=bundle.device)
                cut = bundle.model(
                    input_ids=forced,
                    attention_mask=_mask(plen + 1, bundle.device),
                    past_key_values=pre.past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                h = cut.hidden_states[gov_layer + 1][:, -1, :].detach()
                gov.update(h, +1.0)

                # Save the sequence state at the temporal cut before any successor
                # forward mutates an HF Cache object in-place.
                try:
                    torch.save(cut.past_key_values, cache_path)
                    cache_serialized = True
                except Exception as e:
                    cache_error = f"{type(e).__name__}: {e}"
                    cache_path.unlink(missing_ok=True)

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

            records.append(
                {
                    "prompt_index": prompt_i,
                    "prompt_id": prompt["id"],
                    "input_ids": inp["input_ids"].detach().cpu(),
                    "forced_id": discr["d1_id"],
                    "next_id": next_id,
                    "discr": discr,
                    "cache_path": str(cache_path),
                    "cache_serialized": cache_serialized,
                    "cache_error": cache_error,
                    "governance": gov.snapshot_cpu(),
                    "governance_state_hash": gov.state_hash(),
                    "live_logits": live.logits[0, -1].detach().float().cpu(),
                    "live_hidden": live.hidden_states[-1][0, -1].detach().float().cpu(),
                    "plen": plen,
                    "gov_layer": gov_layer,
                }
            )
        finally:
            gov.detach()

    payload = {
        "format_version": 2,
        "phase": "prefix",
        "model": args.model_key,
        "source_pid": source_pid,
        "records": records,
    }
    _save_tensor_dict(state_path, payload)
    dump_json(
        {
            "phase": "prefix",
            "model": args.model_key,
            "source_pid": source_pid,
            "state": args.state,
            "records": len(records),
            "exact_cache_records": int(sum(bool(r["cache_serialized"]) for r in records)),
            "status": "ok",
        },
        Path(args.meta),
    )
    print(
        json.dumps(
            {
                "phase": "prefix",
                "model": args.model_key,
                "source_pid": source_pid,
                "state": args.state,
                "records": len(records),
                "exact_cache_records": int(sum(bool(r["cache_serialized"]) for r in records)),
            }
        ),
        flush=True,
    )


def successor(args):
    """Fresh process/model realization resumes only from detached records."""
    prepared = load_json(args.prepared)["prepared"]
    spec = _spec(args.models, args.model_key)
    bundle = load_model_bundle(spec, prepared, device="cuda:0")
    payload = torch.load(args.state, map_location="cpu", weights_only=False)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Detached state file contains no reconstruction records")

    rows: list[dict[str, Any]] = []
    successor_pid = os.getpid()
    source_pid = int(payload.get("source_pid", -1))

    for rec in records:
        snap = rec["governance"]
        gov = LowRankGovernance.from_snapshot(snap, bundle.device)
        gov.attach(bundle.layers[int(rec["gov_layer"])])
        mode = "serialized_exact_cache"
        fallback_error = ""
        try:
            if not rec.get("cache_serialized", False):
                raise RuntimeError("cache serialization unavailable: " + rec.get("cache_error", ""))

            # This cache was serialized by a now-terminated source process. Loading
            # it allocates a new physical realization from a detached record.
            past = torch.load(rec["cache_path"], map_location=bundle.device, weights_only=False)
            nxt = torch.tensor([[int(rec["next_id"])]], device=bundle.device)
            with torch.inference_mode():
                out = bundle.model(
                    input_ids=nxt,
                    attention_mask=_mask(int(rec["plen"]) + 2, bundle.device),
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
        except Exception as e:
            # Portable fallback: initialize sequence state solely from detached token
            # history, then restore the detached G snapshot.  This is intentionally
            # labelled separately from exact-cache reconstruction.
            mode = "token_record_replay"
            fallback_error = f"{type(e).__name__}: {e}"
            ids = torch.cat(
                [
                    rec["input_ids"].to(bundle.device),
                    torch.tensor([[int(rec["forced_id"])]], device=bundle.device),
                ],
                dim=1,
            )
            gov.enabled = False
            with torch.inference_mode():
                rebuilt = bundle.model(
                    input_ids=ids,
                    attention_mask=_mask(ids.shape[1], bundle.device),
                    use_cache=True,
                    return_dict=True,
                )
            gov.enabled = True
            nxt = torch.tensor([[int(rec["next_id"])]], device=bundle.device)
            with torch.inference_mode():
                out = bundle.model(
                    input_ids=nxt,
                    attention_mask=_mask(ids.shape[1] + 1, bundle.device),
                    past_key_values=rebuilt.past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )

        live_logits = rec["live_logits"].to(bundle.device)
        live_hidden = rec["live_hidden"].to(bundle.device)
        h = out.hidden_states[-1][0, -1].detach().float()
        rows.append(
            {
                "experiment": "hard_process_reconstruction",
                "model": args.model_key,
                "prompt_id": rec["prompt_id"],
                "source_pid": source_pid,
                "successor_pid": successor_pid,
                "distinct_os_process": int(source_pid != successor_pid),
                "source_realization_terminated_before_successor": 1,
                "reconstruction_mode": mode,
                "logit_js": js_divergence_logits(live_logits, out.logits[0, -1]),
                "hidden_max_abs": float((live_hidden.float() - h).abs().max().item()),
                "hidden_rmse": float(torch.sqrt(torch.mean((live_hidden.float() - h) ** 2)).item()),
                "live_C_protocol": 1,
                "live_N_protocol": 1,
                "reconstructed_C_protocol": 0,
                "reconstructed_N_protocol": 0,
                "governance_state_hash_prefix": rec.get("governance_state_hash", ""),
                "governance_state_hash_successor": gov.state_hash(),
                "governance_state_hash_equal": int(rec.get("governance_state_hash", "") == gov.state_hash()),
                "fallback_error": fallback_error,
                "d1_id": rec["discr"]["d1_id"],
                "d1_text": rec["discr"]["d1_text"],
                "d2_id": rec["discr"]["d2_id"],
                "d2_text": rec["discr"]["d2_text"],
            }
        )
        gov.detach()

    dump_json(rows, Path(args.meta))
    print(
        json.dumps(
            {
                "phase": "successor",
                "model": args.model_key,
                "source_pid": source_pid,
                "successor_pid": successor_pid,
                "records": len(rows),
            }
        ),
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prefix", "successor"])
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--prompt-index", type=int, default=0)
    ap.add_argument("--prompt-count", type=int, default=1)
    ap.add_argument("--max-prompt-tokens", type=int, default=128)
    ap.add_argument("--gov-rank", type=int, default=8)
    ap.add_argument("--gov-layer-fraction", type=float, default=.66)
    ap.add_argument("--gov-eta", type=float, default=.02)
    ap.add_argument("--gov-gain", type=float, default=.05)
    args = ap.parse_args()
    if args.phase == "prefix":
        prefix(args)
    else:
        successor(args)


if __name__ == "__main__":
    main()
