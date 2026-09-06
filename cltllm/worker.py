from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

from .adaptive import run_adaptive_continuity
from .carriers import run_carrier_audit
from .config import load_model_specs, load_run_config, select_specs
from .hf import load_model_bundle
from .hidden_liability import run_hidden_liability
from .patching import run_activation_patching
from .reconstruction import run_reconstruction_equivalence
from .surface import run_surface_stimuli
from .utils import cleanup_cuda, dump_json, ensure_dir, load_json, load_jsonl, runtime_metadata, seed_all, timer


# For exactly four Iris V100 workers. One OLMo post-training stage is placed on
# each rank so the expensive 7B checkpoints are distributed instead of stacked.
PREFERRED_RANK = {
    "qwen3_4b_base": 0,
    "qwen3_4b_instruct": 0,
    "olmo2_7b_dpo": 0,
    "phi4_mini_instruct": 1,
    "llama32_3b_base": 1,
    "llama32_3b_instruct": 1,
    "olmo2_7b_sft": 1,
    "gemma3_4b_pt": 2,
    "gemma3_4b_it": 2,
    "olmo2_7b_base": 2,
    "zamba2_1p2b_base": 3,
    "olmo2_7b_instruct": 3,
}


def _assignment(specs, rank: int, world: int):
    if world == 4:
        return [s for s in specs if PREFERRED_RANK.get(s.key, hash(s.key) % 4) == rank]
    # Portable fallback for non-4-rank pilot runs.
    return [s for i, s in enumerate(specs) if i % world == rank]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--surface-prompts", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rank = int(os.getenv("SLURM_PROCID", os.getenv("RANK", "0")))
    world = int(os.getenv("SLURM_NTASKS", os.getenv("WORLD_SIZE", "1")))
    local_rank = int(os.getenv("SLURM_LOCALID", os.getenv("LOCAL_RANK", "0")))
    # Under --gpus-per-task=1 each task should see one logical GPU. If Slurm
    # exposes all GPUs, local_rank selects the task-local card deterministically.
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the production worker")
    visible = torch.cuda.device_count()
    device = "cuda:0" if visible == 1 else f"cuda:{local_rank % visible}"
    torch.cuda.set_device(device)

    cfg = load_run_config(args.config)
    specs = select_specs(load_model_specs(args.models), cfg["model_groups"])
    prep = load_json(args.prepared)["prepared"]
    prompts = load_jsonl(args.prompts)
    surface_prompts = load_jsonl(args.surface_prompts)
    assigned = _assignment(specs, rank, world)
    root = ensure_dir(Path(args.output) / f"rank_{rank:02d}")
    dump_json({"rank": rank, "world": world, "device": device, "assigned": [s.key for s in assigned], "runtime": runtime_metadata()}, root / "worker_meta.json")
    print(f"[rank {rank}/{world}] device={device}; assigned={[s.key for s in assigned]}", flush=True)

    for spec in assigned:
        if spec.key not in prep:
            print(f"[rank {rank}] SKIP unavailable {spec.key}", flush=True)
            continue
        model_out = ensure_dir(root / spec.key)
        status = {"model": spec.key, "rank": rank, "status": "running"}
        dump_json(status, model_out / "status.json")
        try:
            seed_all(int(cfg["seed"]))
            with timer(f"load:{spec.key}", root / "timings.jsonl"):
                bundle = load_model_bundle(spec, prep, device=device)
            dump_json({
                "model": spec.key, "repo": spec.repo, "hidden_size": bundle.hidden_size,
                "num_layers": bundle.num_layers, "device": str(bundle.device),
                "revision": prep[spec.key].get("revision"),
            }, model_out / "model_meta.json")
            print(f"[rank {rank}] {spec.key}: {bundle.num_layers} layers, d={bundle.hidden_size}", flush=True)

            mech_groups = set(cfg["mechanistic_model_groups"])
            is_mech = bool(mech_groups.intersection(spec.groups))
            is_core = cfg["core_model_group"] in spec.groups
            is_olmo = cfg["posttraining_model_group"] in spec.groups
            is_surface = bool(set(cfg["surface_model_groups"]).intersection(spec.groups))

            if is_mech:
                n = int(cfg["mechanistic_prompt_count_olmo"] if is_olmo else cfg["mechanistic_prompt_count_core"])
                with timer(f"hidden:{spec.key}", root / "timings.jsonl"):
                    run_hidden_liability(bundle, prompts[:n], cfg, model_out)
                patch_n = int(cfg["patch_prompt_count_olmo"] if is_olmo else cfg["patch_prompt_count_core"])
                with timer(f"patch:{spec.key}", root / "timings.jsonl"):
                    run_activation_patching(bundle, prompts, cfg, model_out, patch_n)
                with timer(f"reconstruct:{spec.key}", root / "timings.jsonl"):
                    run_reconstruction_equivalence(bundle, prompts, cfg, model_out)

            # The four-way causal continuation experiment is reserved for the core
            # cross-family panel. OLMo's contribution is the post-training-history control.
            if is_core:
                with timer(f"adaptive:{spec.key}", root / "timings.jsonl"):
                    run_adaptive_continuity(bundle, prompts, cfg, model_out)
                with timer(f"carrier:{spec.key}", root / "timings.jsonl"):
                    run_carrier_audit(bundle, prompts, cfg, model_out)

            if is_surface:
                with timer(f"surface:{spec.key}", root / "timings.jsonl"):
                    run_surface_stimuli(bundle, surface_prompts, cfg, model_out)

            status["status"] = "ok"
            dump_json(status, model_out / "status.json")
            del bundle
            cleanup_cuda()
            print(f"[rank {rank}] DONE {spec.key}", flush=True)
        except Exception as e:
            status.update({"status": "failed", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
            dump_json(status, model_out / "status.json")
            print(f"[rank {rank}] FAILED {spec.key}: {e}", flush=True)
            traceback.print_exc()
            cleanup_cuda()
            if cfg.get("strict_experiments", False):
                raise

    print(f"[rank {rank}] complete", flush=True)


if __name__ == "__main__":
    main()
