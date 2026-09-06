from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from packaging.version import Version

from .utils import dump_json, package_versions, runtime_metadata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--expected-gpus", type=int, default=4)
    p.add_argument("--min-vram-gb", type=float, default=30.0)
    args = p.parse_args()

    versions = package_versions()
    problems: list[str] = []
    warnings: list[str] = []
    if versions.get("transformers") == "missing":
        problems.append("transformers>=4.51,<5 is required for the validated cache/hook API")
    else:
        tv = Version(versions["transformers"])
        if tv < Version("4.51.0") or tv >= Version("5.0.0"):
            problems.append("transformers>=4.51,<5 is required for the validated cache/hook API")
    if versions.get("torch") == "missing" or Version(versions["torch"].split("+")[0]) < Version("2.2.0"):
        problems.append("torch>=2.2 is required")
    if not os.getenv("SCRATCH"):
        problems.append("$SCRATCH is not defined")

    if not torch.cuda.is_available():
        problems.append("CUDA is not visible in the batch allocation")
    else:
        n = torch.cuda.device_count()
        if n != args.expected_gpus:
            problems.append(f"expected {args.expected_gpus} visible GPUs in batch shell, found {n}")
        for i in range(n):
            prop = torch.cuda.get_device_properties(i)
            gb = prop.total_memory / (1024**3)
            if gb < args.min_vram_gb:
                problems.append(
                    f"GPU {i} ({prop.name}) exposes {gb:.1f} GiB; full suite expects >= {args.min_vram_gb:.1f} GiB"
                )
            if "V100" not in prop.name:
                warnings.append(f"GPU {i} is {prop.name}; suite was designed for Iris V100 nodes")

    meta = runtime_metadata()
    meta["preflight_problems"] = problems
    meta["preflight_warnings"] = warnings
    dump_json(meta, Path(args.output))
    for w in warnings:
        print("PRE-FLIGHT WARNING:", w)
    if problems:
        print("PRE-FLIGHT FAILED:")
        for x in problems:
            print(" -", x)
        sys.exit(2)
    print("Pre-flight OK")
    print(versions)


if __name__ == "__main__":
    main()
