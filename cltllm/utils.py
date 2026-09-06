from __future__ import annotations

import contextlib
import csv
import gc
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(*parts: Any, mod: int = 2**31 - 1) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % mod


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if is_dataclass(obj):
        obj = asdict(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def append_jsonl(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, default=str) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_csv_rows(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    rows = list(rows)
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: str | Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def package_versions() -> dict[str, str]:
    names = ["torch", "transformers", "huggingface_hub", "numpy", "pandas", "matplotlib", "packaging"]
    out: dict[str, str] = {}
    import importlib.metadata as md
    for n in names:
        try:
            out[n] = md.version(n)
        except Exception:
            out[n] = "missing"
    return out


def runtime_metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "versions": package_versions(),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_procid": os.getenv("SLURM_PROCID"),
        "scratch": os.getenv("SCRATCH"),
    }
    if torch.cuda.is_available():
        meta["cuda_device_count_visible"] = torch.cuda.device_count()
        meta["cuda_devices"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
                "total_memory": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ]
    return meta


def run_capture(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as e:
        return f"ERROR: {e}"


@contextlib.contextmanager
def timer(label: str, log_path: str | Path | None = None):
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        if log_path is not None:
            append_jsonl({"event": "timing", "label": label, "seconds": dt}, log_path)


def chunked(seq: list[Any], n: int) -> Iterable[list[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]
