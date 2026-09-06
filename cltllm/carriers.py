from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import torch

from .cacheops import cache_summary, clone_cache, mix_legacy_cache, to_legacy
from .hf import ModelBundle, choose_top2_tokens, encode_prompt
from .metrics import js_divergence_logits, normalized_mediation
from .utils import dump_json, write_csv_rows


def _mask(batch: int, length: int, device: torch.device) -> torch.Tensor:
    return torch.ones((batch, length), dtype=torch.long, device=device)


def _branch_at_cut(bundle: ModelBundle, prompt_ids: torch.Tensor, forced_id: int):
    """Build a branch from detached token history to avoid mutable-cache aliasing."""
    ids = torch.cat(
        [prompt_ids, torch.tensor([[forced_id]], dtype=torch.long, device=bundle.device)],
        dim=1,
    )
    with torch.inference_mode():
        return bundle.model(
            input_ids=ids,
            attention_mask=_mask(1, ids.shape[1], bundle.device),
            use_cache=True,
            return_dict=True,
        )


def _greedy_common_suffix(bundle: ModelBundle, ids: torch.Tensor, length: int) -> list[int]:
    suffix: list[int] = []
    cur = ids
    with torch.inference_mode():
        for _ in range(length):
            out = bundle.model(input_ids=cur, use_cache=False, return_dict=True)
            tid = int(torch.argmax(out.logits[0, -1]).item())
            suffix.append(tid)
            cur = torch.cat([cur, torch.tensor([[tid]], device=bundle.device)], dim=1)
    return suffix


def _advance(bundle: ModelBundle, cache: Any, prefix_len: int, suffix: list[int]) -> torch.Tensor:
    past = cache
    logits = None
    with torch.inference_mode():
        for j, tid in enumerate(suffix, 1):
            token = torch.tensor([[tid]], dtype=torch.long, device=bundle.device)
            out = bundle.model(
                input_ids=token,
                attention_mask=_mask(1, prefix_len + j, bundle.device),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values
            logits = out.logits[0, -1].float()
    if logits is None:
        raise ValueError("carrier_horizon must be >=1")
    return logits


def _contiguous_groups(n_layers: int, n_groups: int) -> list[set[int]]:
    n_groups = max(1, min(n_groups, n_layers))
    # Exact contiguous partition, with at most one-layer size imbalance.
    bounds = [round(i * n_layers / n_groups) for i in range(n_groups + 1)]
    groups = [set(range(bounds[i], bounds[i + 1])) for i in range(n_groups)]
    return [g for g in groups if g]


def _subset_mask(indices: tuple[int, ...]) -> str:
    return ",".join(map(str, indices)) if indices else "EMPTY"


def run_carrier_audit(
    bundle: ModelBundle,
    prompts: list[dict[str, Any]],
    cfg: dict[str, Any],
    outdir: Path,
) -> list[dict[str, Any]]:
    """Interventional candidate-carrier audit over layerwise sequence state.

    The experiment holds a teacher-forced future suffix fixed. It starts from the
    d2 branch and transplants selected cache-layer groups from the d1 branch. The
    smallest preregistered group sets that transport >= threshold of the d1/d2
    future-logit effect are marked as candidate causal carrier sets.

    This approximates a causal carrier search. It does not, by itself, establish a
    CLT bearer or phenomenality.
    """
    rows: list[dict[str, Any]] = []
    n_prompt = int(cfg["carrier_prompt_count_core"])
    horizon = int(cfg["carrier_horizon"])
    threshold = float(cfg["carrier_mediation_threshold"])
    eps = float(cfg.get("carrier_js_epsilon", 1e-5))

    for prompt in prompts[:n_prompt]:
        inp = encode_prompt(bundle, prompt["text"], int(cfg["max_prompt_tokens"]))
        discr = choose_top2_tokens(bundle, inp)
        pids = inp["input_ids"]
        seq1 = torch.cat([pids, torch.tensor([[discr["d1_id"]]], device=bundle.device)], dim=1)
        seq2 = torch.cat([pids, torch.tensor([[discr["d2_id"]]], device=bundle.device)], dim=1)

        # Branches are independently reconstructed from the same prompt token record,
        # preventing DynamicCache in-place mutation from contaminating the comparison.
        cut1 = _branch_at_cut(bundle, pids, discr["d1_id"])
        cut2 = _branch_at_cut(bundle, pids, discr["d2_id"])
        suffix = _greedy_common_suffix(bundle, seq2, horizon)
        prefix_len = seq1.shape[1]

        log1 = _advance(bundle, clone_cache(cut1.past_key_values), prefix_len, suffix)
        log2 = _advance(bundle, clone_cache(cut2.past_key_values), prefix_len, suffix)
        base_js = js_divergence_logits(log1, log2)
        base_supported = int(base_js >= eps)

        meta = {
            "model": bundle.spec.key,
            "prompt_id": prompt["id"],
            "carrier_horizon": horizon,
            "base_js": base_js,
            "base_effect_supported": base_supported,
            "threshold": threshold,
            **discr,
        }

        legacy1 = to_legacy(cut1.past_key_values)
        legacy2 = to_legacy(cut2.past_key_values)
        layerwise_available = (
            legacy1 is not None
            and legacy2 is not None
            and len(legacy1) == len(legacy2)
            and len(legacy1) > 0
        )

        if not layerwise_available:
            # Whole-state transport is still a useful positive control. It is labeled
            # separately and never presented as a minimal layerwise carrier result.
            try:
                whole_logits = _advance(bundle, clone_cache(cut1.past_key_values), prefix_len, suffix)
                patched_js = js_divergence_logits(log1, whole_logits)
                raw, clipped = normalized_mediation(base_js, patched_js) if base_supported else (float("nan"), float("nan"))
                rows.append({
                    "experiment": "carrier_audit", **meta,
                    "supported": 1,
                    "layerwise_supported": 0,
                    "carrier_mode": "whole_sequence_state_only",
                    "partition_groups": 1,
                    "groups_from_d1": "ALL_SEQUENCE_STATE",
                    "group_count": 1,
                    "source_layer_count": -1,
                    "total_cache_layers": -1,
                    "source_layer_fraction": float("nan"),
                    "patched_js": patched_js,
                    "mediation_raw": raw,
                    "mediation_clipped": clipped,
                    "passes_threshold": int(base_supported and clipped >= threshold),
                    "is_minimal_at_threshold": 0,
                    "support_note": "Cache has no compatible layerwise legacy representation; whole-state transport only.",
                })
            except Exception as e:
                rows.append({
                    "experiment": "carrier_audit", **meta,
                    "supported": 0,
                    "layerwise_supported": 0,
                    "carrier_mode": "unsupported",
                    "partition_groups": -1,
                    "groups_from_d1": "",
                    "group_count": -1,
                    "source_layer_count": -1,
                    "total_cache_layers": -1,
                    "source_layer_fraction": float("nan"),
                    "patched_js": float("nan"),
                    "mediation_raw": float("nan"),
                    "mediation_clipped": float("nan"),
                    "passes_threshold": 0,
                    "is_minimal_at_threshold": 0,
                    "support_note": f"{type(e).__name__}: {e}",
                })
            dump_json(
                {"d1": cache_summary(cut1.past_key_values), "d2": cache_summary(cut2.past_key_values)},
                outdir / "representative" / f"carrier_cache_{prompt['id']}.json",
            )
            continue

        n_cache_layers = len(legacy1)
        for requested_groups in cfg["carrier_partitions"]:
            groups = _contiguous_groups(n_cache_layers, int(requested_groups))
            records: list[dict[str, Any]] = []
            # Include the empty intervention as an auditable baseline row.
            candidate_subsets = [tuple()] + [
                comb
                for k in range(1, len(groups) + 1)
                for comb in itertools.combinations(range(len(groups)), k)
            ]
            for subset in candidate_subsets:
                source_layers: set[int] = set()
                for gi in subset:
                    source_layers |= groups[gi]
                try:
                    mixed = mix_legacy_cache(cut1.past_key_values, cut2.past_key_values, source_layers)
                    patched_logits = _advance(bundle, mixed, prefix_len, suffix)
                    patched_js = js_divergence_logits(log1, patched_logits)
                    raw, clipped = normalized_mediation(base_js, patched_js) if base_supported else (float("nan"), float("nan"))
                    passes = int(bool(subset) and base_supported and clipped >= threshold)
                    rec = {
                        "experiment": "carrier_audit", **meta,
                        "supported": 1,
                        "layerwise_supported": 1,
                        "carrier_mode": "layerwise_cache_transplant",
                        "partition_groups": len(groups),
                        "groups_from_d1": _subset_mask(subset),
                        "group_count": len(subset),
                        "source_layer_count": len(source_layers),
                        "total_cache_layers": n_cache_layers,
                        "source_layer_fraction": len(source_layers) / float(n_cache_layers),
                        "source_layers": ",".join(map(str, sorted(source_layers))),
                        "patched_js": patched_js,
                        "mediation_raw": raw,
                        "mediation_clipped": clipped,
                        "passes_threshold": passes,
                        "is_minimal_at_threshold": 0,
                        "support_note": "",
                        "_subset": subset,
                    }
                except Exception as e:
                    rec = {
                        "experiment": "carrier_audit", **meta,
                        "supported": 0,
                        "layerwise_supported": 1,
                        "carrier_mode": "layerwise_cache_transplant",
                        "partition_groups": len(groups),
                        "groups_from_d1": _subset_mask(subset),
                        "group_count": len(subset),
                        "source_layer_count": len(source_layers),
                        "total_cache_layers": n_cache_layers,
                        "source_layer_fraction": len(source_layers) / float(n_cache_layers),
                        "source_layers": ",".join(map(str, sorted(source_layers))),
                        "patched_js": float("nan"),
                        "mediation_raw": float("nan"),
                        "mediation_clipped": float("nan"),
                        "passes_threshold": 0,
                        "is_minimal_at_threshold": 0,
                        "support_note": f"{type(e).__name__}: {e}",
                        "_subset": subset,
                    }
                records.append(rec)

            passing = {r["_subset"] for r in records if r["supported"] == 1 and r["passes_threshold"] == 1}
            for r in records:
                s = r["_subset"]
                if s in passing:
                    r["is_minimal_at_threshold"] = int(
                        not any(set(t) < set(s) for t in passing)
                    )
                r.pop("_subset", None)
                rows.append(r)

        dump_json(
            {"d1": cache_summary(cut1.past_key_values), "d2": cache_summary(cut2.past_key_values)},
            outdir / "representative" / f"carrier_cache_{prompt['id']}.json",
        )

    write_csv_rows(rows, outdir / "carrier_audit.csv")
    return rows
