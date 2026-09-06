from __future__ import annotations

import copy
from typing import Any

import torch


def cache_summary(cache: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"type": type(cache).__name__}
    if cache is None:
        return out
    try:
        legacy = to_legacy(cache)
        if legacy is not None:
            out["legacy_layers"] = len(legacy)
            out["legacy_shapes"] = [
                [list(t.shape) for t in layer if torch.is_tensor(t)]
                if isinstance(layer, (tuple, list)) else []
                for layer in legacy[:4]
            ]
            return out
    except Exception as e:
        out["legacy_error"] = f"{type(e).__name__}: {e}"
    if hasattr(cache, "__dict__"):
        attrs = {}
        for k, v in vars(cache).items():
            if torch.is_tensor(v):
                attrs[k] = list(v.shape)
            elif isinstance(v, (list, tuple)):
                attrs[k] = f"{type(v).__name__}[{len(v)}]"
            else:
                attrs[k] = type(v).__name__
        out["attrs"] = attrs
    return out


def to_legacy(cache: Any):
    if cache is None:
        return None
    if isinstance(cache, tuple):
        return cache
    if isinstance(cache, list):
        return tuple(cache)
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return None


def _clone_legacy(legacy: tuple) -> tuple:
    cloned = []
    for layer in legacy:
        if isinstance(layer, (tuple, list)):
            cloned.append(tuple(t.clone() if torch.is_tensor(t) else copy.deepcopy(t) for t in layer))
        else:
            cloned.append(copy.deepcopy(layer))
    return tuple(cloned)


def restore_cache_type(legacy: tuple, template: Any) -> Any:
    """Restore a legacy KV tuple to the template cache type when HF supports it.

    Decoder-only models usually still accept the legacy tuple directly. Newer
    DynamicCache variants can reconstruct their native type with from_legacy_cache.
    Hybrid/SSM caches that lack this conversion are deliberately left unsupported
    by the layerwise carrier audit rather than being silently coerced.
    """
    if isinstance(template, tuple):
        return legacy
    if isinstance(template, list):
        return list(legacy)
    cls = type(template)
    factory = getattr(cls, "from_legacy_cache", None)
    if callable(factory):
        try:
            return factory(legacy)
        except Exception:
            pass
    inst_factory = getattr(template, "from_legacy_cache", None)
    if callable(inst_factory):
        try:
            return inst_factory(legacy)
        except Exception:
            pass
    # Most standard HF decoder models accept legacy tuples even when they emitted
    # DynamicCache. The caller performs an actual forward pass and records failure
    # if a family rejects it.
    return legacy


def mix_legacy_cache(source: Any, target: Any, source_layers: set[int]) -> Any:
    """Return target sequence state with selected layer states taken from source."""
    a = to_legacy(source)
    b = to_legacy(target)
    if a is None or b is None or len(a) != len(b):
        raise TypeError("Layerwise legacy-cache mixing is unavailable for this cache type")
    mixed = []
    for i, (la, lb) in enumerate(zip(a, b)):
        pick = la if i in source_layers else lb
        if isinstance(pick, (tuple, list)):
            mixed.append(tuple(t.clone() if torch.is_tensor(t) else copy.deepcopy(t) for t in pick))
        else:
            mixed.append(copy.deepcopy(pick))
    return restore_cache_type(tuple(mixed), target)


def clone_cache(cache: Any) -> Any:
    legacy = to_legacy(cache)
    if legacy is not None:
        return restore_cache_type(_clone_legacy(legacy), cache)
    return copy.deepcopy(cache)
