from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .governance import LowRankGovernance
from .hf import ModelBundle, choose_top2_tokens, encode_prompt
from .metrics import js_divergence_from_probs
from .rollout import rollout_counterfactual
from .utils import stable_seed, write_csv_rows


def run_adaptive_continuity(
    bundle: ModelBundle,
    prompts: list[dict[str, Any]],
    cfg: dict[str, Any],
    outdir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_h = max(cfg["horizons"])
    capture_layers = sorted({
        min(bundle.num_layers - 1, max(0, int(round(f * (bundle.num_layers - 1)))))
        for f in cfg["governance_layer_fractions"]
    })
    prompt_subset = prompts[: int(cfg["adaptive_prompt_count_core"])]

    for gov_layer in capture_layers:
        for eta in cfg["governance_sensitivity_etas"]:
            for prompt in prompt_subset:
                inputs = encode_prompt(bundle, prompt["text"], cfg["max_prompt_tokens"])
                discr = choose_top2_tokens(bundle, inputs)
                for condition in cfg["adaptive_conditions"]:
                    # Frozen is a true control: no writable governance state.
                    if condition == "frozen_live":
                        g1 = g2 = None
                    else:
                        g1 = LowRankGovernance(
                            bundle.hidden_size, int(cfg["governance_rank"]), gov_layer,
                            float(eta), float(cfg["governance_gain"]), float(cfg["governance_decay"]),
                            bundle.device, stable_seed(cfg["seed"], bundle.spec.key, gov_layer, eta, "A"),
                        )
                        g2 = LowRankGovernance(
                            bundle.hidden_size, int(cfg["governance_rank"]), gov_layer,
                            float(eta), float(cfg["governance_gain"]), float(cfg["governance_decay"]),
                            bundle.device, stable_seed(cfg["seed"], bundle.spec.key, gov_layer, eta, "A"),
                        )
                        # Same fixed A/B basis in both counterfactual worlds.
                        g2.A = g1.A.clone()
                        g2.B = g1.B.clone()

                    # Use common random numbers across both counterfactual
                    # branches AND architectural conditions.  Including
                    # `condition` here would inject avoidable Monte Carlo
                    # noise into live/copy/reconstructed comparisons.
                    namespace = (
                        cfg["seed"],
                        bundle.spec.key,
                        prompt["id"],
                        "adaptive",
                        gov_layer,
                        eta,
                    )
                    try:
                        if g1 is not None:
                            g1.attach(bundle.layers[gov_layer])
                        r1 = rollout_counterfactual(
                            bundle, inputs, discr["d1_id"], n_rollouts=int(cfg.get("adaptive_rollouts", cfg["rollouts"])),
                            batch_size=int(cfg.get("adaptive_rollout_batch_size", cfg["rollout_batch_size"])), max_horizon=max_h,
                            capture_layers=[gov_layer, bundle.num_layers - 1], temperature=cfg["temperature"],
                            seed_namespace=namespace, governance=g1, governance_decision_sign=+1.0,
                            condition=condition,
                        )
                        if g1 is not None:
                            g1.detach()
                        if g2 is not None:
                            g2.attach(bundle.layers[gov_layer])
                        r2 = rollout_counterfactual(
                            bundle, inputs, discr["d2_id"], n_rollouts=int(cfg.get("adaptive_rollouts", cfg["rollouts"])),
                            batch_size=int(cfg.get("adaptive_rollout_batch_size", cfg["rollout_batch_size"])), max_horizon=max_h,
                            capture_layers=[gov_layer, bundle.num_layers - 1], temperature=cfg["temperature"],
                            seed_namespace=namespace, governance=g2, governance_decision_sign=-1.0,
                            condition=condition,
                        )
                    finally:
                        if g1 is not None:
                            g1.detach()
                        if g2 is not None:
                            g2.detach()

                    # C and N are experimental-architecture predicates across the cut.
                    C = 0 if condition == "reconstructed" else 1
                    N = 0 if condition == "reconstructed" else 1
                    for h in cfg["horizons"]:
                        js = js_divergence_from_probs(r1.mean_probs[h], r2.mean_probs[h])
                        final_x = r1.hidden[h][bundle.num_layers - 1].float()
                        final_y = r2.hidden[h][bundle.num_layers - 1].float()
                        hidden_rmse = float(torch.sqrt(torch.mean((final_x - final_y) ** 2)).item())
                        rows.append({
                            "experiment": "adaptive_continuity",
                            "model": bundle.spec.key,
                            "prompt_id": prompt["id"],
                            "condition": condition,
                            "governance_layer": gov_layer,
                            "governance_eta": eta,
                            "governance_rank": cfg["governance_rank"],
                            "horizon": h,
                            "output_js": js,
                            "final_hidden_rmse": hidden_rmse,
                            "C_protocol": C,
                            "E_protocol": 1,
                            "future_counterfactual_effect_detected": int(js > float(cfg.get("carrier_js_epsilon", 1e-5))),
                            "N_protocol": N,
                            "copy_created": int(condition == "persistent_copy"),
                            "adaptive_state_enabled": int(condition != "frozen_live"),
                            "governance_hash_d1": r1.governance_hash,
                            "governance_hash_d2": r2.governance_hash,
                            **discr,
                        })
    write_csv_rows(rows, outdir / "adaptive_continuity.csv")
    return rows
