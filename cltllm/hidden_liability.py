from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .hf import ModelBundle, choose_top2_tokens, encode_prompt
from .metrics import coarse_grain, hidden_metric, js_divergence_from_probs
from .rollout import rollout_counterfactual
from .utils import stable_seed, write_csv_rows


def run_hidden_liability(
    bundle: ModelBundle,
    prompts: list[dict[str, Any]],
    cfg: dict[str, Any],
    outdir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    capture_layers = list(range(0, bundle.num_layers, int(cfg["hidden_layer_stride"])))
    if capture_layers[-1] != bundle.num_layers - 1:
        capture_layers.append(bundle.num_layers - 1)
    max_h = max(cfg["horizons"])

    for pi, prompt in enumerate(prompts):
        inputs = encode_prompt(bundle, prompt["text"], cfg["max_prompt_tokens"])
        discr = choose_top2_tokens(bundle, inputs)
        common_seed = (cfg["seed"], bundle.spec.key, prompt["id"], "hidden")
        r1 = rollout_counterfactual(
            bundle, inputs, discr["d1_id"], n_rollouts=cfg["rollouts"],
            batch_size=cfg["rollout_batch_size"], max_horizon=max_h,
            capture_layers=capture_layers, temperature=cfg["temperature"],
            seed_namespace=common_seed, condition="frozen_live",
        )
        r2 = rollout_counterfactual(
            bundle, inputs, discr["d2_id"], n_rollouts=cfg["rollouts"],
            batch_size=cfg["rollout_batch_size"], max_horizon=max_h,
            capture_layers=capture_layers, temperature=cfg["temperature"],
            seed_namespace=common_seed, condition="frozen_live",
        )

        for h in cfg["horizons"]:
            js = js_divergence_from_probs(r1.mean_probs[h], r2.mean_probs[h])
            rows.append({
                "experiment": "hidden_liability", "model": bundle.spec.key,
                "prompt_id": prompt["id"], "horizon": h, "layer": -1,
                "measurement_map": "output_distribution", "metric": "js",
                "value": js, **discr,
            })
            for layer in capture_layers:
                x0 = r1.hidden[h][layer].to(bundle.device)
                y0 = r2.hidden[h][layer].to(bundle.device)
                for cg in cfg["measurement_maps"]:
                    seed = stable_seed(cfg["metric_seed"], bundle.spec.key, prompt["id"], h, layer, cg)
                    x = coarse_grain(x0, cg, seed)
                    y = coarse_grain(y0, cg, seed)
                    for metric in cfg["hidden_metrics"]:
                        val = hidden_metric(
                            metric, x, y, seed=seed,
                            swd_projections=cfg["swd_projections"],
                            mmd_features=cfg["mmd_rff_features"],
                        )
                        rows.append({
                            "experiment": "hidden_liability", "model": bundle.spec.key,
                            "prompt_id": prompt["id"], "horizon": h, "layer": layer,
                            "measurement_map": cg, "metric": metric, "value": val,
                            **discr,
                        })
                del x0, y0
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if cfg.get("save_representative_arrays", True) and pi == 0:
            rep = outdir / "representative"
            rep.mkdir(parents=True, exist_ok=True)
            last_layer = capture_layers[-1]
            np.savez_compressed(
                rep / "hidden_liability_example.npz",
                d1=r1.hidden[max_h][last_layer].numpy(),
                d2=r2.hidden[max_h][last_layer].numpy(),
                tokens_d1=r1.token_ids.numpy(), tokens_d2=r2.token_ids.numpy(),
            )

    write_csv_rows(rows, outdir / "hidden_liability.csv")
    return rows
