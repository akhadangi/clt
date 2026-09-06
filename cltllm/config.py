from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import load_json


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    groups: tuple[str, ...]
    loader: str = "causal_lm"
    trust_remote_code: bool = False
    gated: bool = False
    family: str | None = None


def load_run_config(path: str | Path) -> dict[str, Any]:
    return load_json(path)


def load_model_specs(path: str | Path) -> list[ModelSpec]:
    raw = load_json(path)["models"]
    specs: list[ModelSpec] = []
    for x in raw:
        specs.append(
            ModelSpec(
                key=x["key"],
                repo=x["repo"],
                groups=tuple(x.get("groups", [])),
                loader=x.get("loader", "causal_lm"),
                trust_remote_code=bool(x.get("trust_remote_code", False)),
                gated=bool(x.get("gated", False)),
                family=x.get("family"),
            )
        )
    return specs


def select_specs(specs: list[ModelSpec], groups: list[str]) -> list[ModelSpec]:
    wanted = set(groups)
    return [s for s in specs if wanted.intersection(s.groups)]
