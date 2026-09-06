from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .config import load_model_specs, load_run_config, select_specs
from .utils import ensure_dir, load_json

ASSIGN4 = {
    0: ["qwen3_4b_base"],
    1: ["phi4_mini_instruct", "llama32_3b_base"],
    2: ["gemma3_4b_pt"],
    3: ["zamba2_1p2b_base"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rank = int(os.getenv("SLURM_PROCID", os.getenv("RANK", "0")))
    world = int(os.getenv("SLURM_NTASKS", os.getenv("WORLD_SIZE", "1")))
    cfg = load_run_config(args.config)
    specs = {s.key: s for s in load_model_specs(args.models)}
    prepared = load_json(args.prepared)["prepared"]
    core = [s.key for s in select_specs(list(specs.values()), [cfg["core_model_group"]])]
    if world == 4:
        assigned = [k for k in ASSIGN4.get(rank, []) if k in core]
    else:
        assigned = [k for i, k in enumerate(core) if i % world == rank]
    root = ensure_dir(Path(args.output) / f"rank_{rank:02d}" / "hard_reconstruction")
    rows: list[dict] = []
    prompt_count = int(cfg.get("hard_reconstruction_prompt_count", 3))

    for key in assigned:
        if key not in prepared:
            continue
        mdir = ensure_dir(root / key)
        state = mdir / "detached_state.pt"
        pmeta = mdir / "prefix.json"
        smeta = mdir / "successor.json"
        base = [
            "--model-key", key,
            "--models", args.models,
            "--prepared", args.prepared,
            "--prompts", args.prompts,
            "--state", str(state),
            "--prompt-count", str(prompt_count),
            "--max-prompt-tokens", str(cfg["max_prompt_tokens"]),
            "--gov-rank", str(cfg["governance_rank"]),
            "--gov-layer-fraction", str(cfg["governance_layer_fractions"][len(cfg["governance_layer_fractions"]) // 2]),
            "--gov-eta", str(cfg["governance_eta"]),
            "--gov-gain", str(cfg["governance_gain"]),
        ]
        print(f"[hard rank {rank}] prefix {key} ({prompt_count} states)", flush=True)
        subprocess.run(
            [sys.executable, "-m", "cltllm.hard_reconstruct", "prefix", *base, "--meta", str(pmeta)],
            check=True,
        )
        # The prefix subprocess has exited here.  This next subprocess necessarily
        # owns a new model realization and is the only process allowed to consume
        # the detached continuation records.
        print(f"[hard rank {rank}] successor {key}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "cltllm.hard_reconstruct", "successor", *base, "--meta", str(smeta)],
            check=True,
        )
        with open(smeta, "r", encoding="utf-8") as f:
            payload = json.load(f)
            rows.extend(payload if isinstance(payload, list) else [payload])
        if not bool(cfg.get("keep_hard_reconstruction_state", False)):
            state.unlink(missing_ok=True)
            for cp in mdir.glob("detached_state.p*.cache.pt"):
                cp.unlink(missing_ok=True)

    if rows:
        pd.DataFrame(rows).to_csv(root / "hard_process_reconstruction.csv", index=False)
    print(f"[hard rank {rank}] complete {assigned}", flush=True)


if __name__ == "__main__":
    main()
