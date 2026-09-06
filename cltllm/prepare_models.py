from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from .config import load_model_specs, load_run_config, select_specs
from .utils import dump_json, ensure_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Download all model snapshots into scratch before GPU workers start.")
    p.add_argument("--config", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--strict", type=int, default=1)
    args = p.parse_args()

    cfg = load_run_config(args.config)
    specs = load_model_specs(args.models)
    selected = select_specs(specs, cfg["model_groups"])
    token = os.getenv("HF_TOKEN") or None
    cache_dir = ensure_dir(args.cache_dir)
    output = Path(args.output)
    ensure_dir(output.parent)

    api = HfApi(token=token)
    prepared: dict[str, dict] = {}
    failures: dict[str, str] = {}

    print(f"Preparing {len(selected)} Hugging Face snapshots in {cache_dir}", flush=True)
    for i, spec in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {spec.key}: {spec.repo}", flush=True)
        try:
            info = api.model_info(spec.repo)
            local = snapshot_download(
                repo_id=spec.repo,
                revision=info.sha,
                cache_dir=str(cache_dir),
                token=token,
            )
            prepared[spec.key] = {
                "repo": spec.repo,
                "revision": info.sha,
                "local_path": local,
                "loader": spec.loader,
                "trust_remote_code": spec.trust_remote_code,
                "groups": list(spec.groups),
                "family": spec.family,
            }
            dump_json({"prepared": prepared, "failures": failures}, output)
        except (GatedRepoError, RepositoryNotFoundError) as e:
            msg = f"ACCESS ERROR for {spec.repo}: {e}"
            print(msg, file=sys.stderr, flush=True)
            failures[spec.key] = msg
            dump_json({"prepared": prepared, "failures": failures}, output)
            if args.strict:
                print(
                    "For gated models, accept the model license on Hugging Face and export HF_TOKEN before sbatch.",
                    file=sys.stderr,
                )
                sys.exit(3)
        except Exception as e:
            msg = f"DOWNLOAD ERROR for {spec.repo}: {type(e).__name__}: {e}"
            print(msg, file=sys.stderr, flush=True)
            failures[spec.key] = msg
            dump_json({"prepared": prepared, "failures": failures}, output)
            if args.strict:
                sys.exit(4)

    dump_json({"prepared": prepared, "failures": failures}, output)
    print(f"Prepared {len(prepared)} models; failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
