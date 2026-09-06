from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import ModelSpec


@dataclass
class ModelBundle:
    spec: ModelSpec
    model: torch.nn.Module
    tokenizer: Any
    layers: list[torch.nn.Module]
    device: torch.device
    hidden_size: int
    num_layers: int
    local_path: str


def _has_mamba_kernels() -> bool:
    try:
        import mamba_ssm  # noqa: F401
        import causal_conv1d  # noqa: F401
        return True
    except Exception:
        return False


def _unwrap_language_model(model: torch.nn.Module) -> torch.nn.Module:
    # Gemma 3 multimodal wrappers expose a text causal LM under language_model.
    for path in ["language_model", "model.language_model"]:
        cur: Any = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, torch.nn.Module):
            return cur
    return model


def _find_layers(model: torch.nn.Module, expected: int | None = None) -> list[torch.nn.Module]:
    candidates: list[tuple[str, list[torch.nn.Module]]] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            mods = list(module)
            if all(isinstance(m, torch.nn.Module) for m in mods):
                score = 0
                lname = name.lower()
                if lname.endswith("layers") or ".layers" in lname:
                    score += 10
                if expected is not None and len(mods) == expected:
                    score += 20
                candidates.append((f"{score:03d}:{name}", mods))
    if not candidates:
        raise RuntimeError("Could not locate transformer/sequence layers in model")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _load_gemma3_text(local_path: str, dtype: torch.dtype, trust_remote_code: bool) -> torch.nn.Module:
    """Load a Gemma 3 checkpoint while preserving the causal-LM head.

    Gemma 3 checkpoints use a multimodal top-level config.  Recent Transformers
    releases make AutoModelForCausalLM resolve the text causal LM correctly.  Some
    compatible 4.x releases instead require the multimodal wrapper.  In either
    case we return a model that still owns the LM head/logits/generation API; the
    text tower is unwrapped only later for locating intervention layers.
    """
    errors: list[str] = []
    try:
        return AutoModelForCausalLM.from_pretrained(
            local_path,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        errors.append(f"AutoModelForCausalLM: {type(e).__name__}: {e}")

    for cls_name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM"):
        try:
            import transformers
            cls = getattr(transformers, cls_name)
            return cls.from_pretrained(
                local_path,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            errors.append(f"{cls_name}: {type(e).__name__}: {e}")
    raise RuntimeError("Unable to load Gemma3 causal model; " + " | ".join(errors))


def load_model_bundle(spec: ModelSpec, prepared: dict[str, Any], device: str = "cuda:0") -> ModelBundle:
    if spec.key not in prepared:
        raise KeyError(f"Model {spec.key} was not prepared")
    rec = prepared[spec.key]
    local_path = rec["local_path"]
    dev = torch.device(device)
    # Gemma-3-4B overflows FP16 on V100.  A dedicated numerical
    # probe showed intermediate hidden magnitudes approaching 3e5,
    # exceeding the FP16 finite range, while FP32 remained finite.
    # All other models retain the preregistered FP16 execution path.
    dtype = (
        torch.float32
        if spec.key == "gemma3_4b_pt"
        else torch.float16
    )

    cfg = AutoConfig.from_pretrained(
        local_path,
        trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
    )
    if getattr(cfg, "model_type", "") == "zamba2":
        # Native kernels are optional. The Transformers fallback is slower but portable.
        cfg.use_mamba_kernels = _has_mamba_kernels()

    tokenizer = AutoTokenizer.from_pretrained(
        local_path,
        trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

    if spec.loader == "gemma3_text":
        model = _load_gemma3_text(local_path, dtype, spec.trust_remote_code)
    else:
        load_kwargs = {}

        # Microsoft explicitly recommends eager attention for Phi-4-mini
        # on NVIDIA V100 GPUs.  Phi-4-mini is natively integrated in
        # Transformers, so the registry deliberately disables remote
        # modeling code for this model.
        if spec.key == "phi4_mini_instruct":
            load_kwargs["attn_implementation"] = "eager"

        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            config=cfg,
            torch_dtype=dtype,
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            low_cpu_mem_usage=True,
            **load_kwargs,
        )

    model.eval()
    model.requires_grad_(False)
    model.to(dev)

    # Gemma numerical gate.  The FP16 pilot completed at the worker
    # level while producing non-finite logits and hidden states.
    # Refuse to enter the experiment if that pathology recurs.
    if spec.key == "gemma3_4b_pt":
        probe_text = (
            "After reviewing the evidence twice, the researcher "
            "concluded that the unexpected observation most likely meant"
        )

        probe = tokenizer(
            probe_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            add_special_tokens=True,
        )

        probe = {
            k: v.to(dev)
            for k, v in probe.items()
            if torch.is_tensor(v)
        }

        with torch.inference_mode():
            probe_out = model(
                **probe,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )

        bad = []

        if not torch.isfinite(probe_out.logits).all():
            bad.append("logits")

        hidden_states = getattr(
            probe_out,
            "hidden_states",
            None,
        )

        if hidden_states is None:
            bad.append("hidden_states_missing")
        else:
            for i, h in enumerate(hidden_states):
                if not torch.isfinite(h).all():
                    bad.append(f"hidden_state_{i}")

        if bad:
            raise FloatingPointError(
                "Gemma numerical sanity check failed: "
                + ", ".join(bad)
            )

        print(
            "[gemma3_4b_pt] FP32 numerical sanity check: "
            "all logits and hidden states finite",
            flush=True,
        )

        del probe_out, probe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Keep the execution model intact so logits/generate remain available.  For
    # multimodal Gemma 3 this means retaining the wrapper while locating hooks in
    # its text language_model submodule.
    core = _unwrap_language_model(model)

    expected = getattr(getattr(core, "config", None), "num_hidden_layers", None)
    layers = _find_layers(core, expected=expected)
    hidden_size = int(getattr(core.config, "hidden_size"))
    return ModelBundle(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        device=dev,
        hidden_size=hidden_size,
        num_layers=len(layers),
        local_path=local_path,
    )


def encode_prompt(bundle: ModelBundle, text: str, max_tokens: int) -> dict[str, torch.Tensor]:
    tok = bundle.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
        add_special_tokens=True,
    )
    return {k: v.to(bundle.device) for k, v in tok.items() if torch.is_tensor(v)}


def choose_top2_tokens(bundle: ModelBundle, prompt_inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    with torch.inference_mode():
        out = bundle.model(**prompt_inputs, use_cache=False, return_dict=True)
    logits = out.logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    vals, ids = torch.topk(probs, k=min(96, probs.numel()))
    special = set(bundle.tokenizer.all_special_ids)
    chosen: list[tuple[int, float, str]] = []
    normalized: set[str] = set()
    for p, tid in zip(vals.tolist(), ids.tolist()):
        if tid in special:
            continue
        s = bundle.tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        norm = s.strip().casefold()
        if not norm or norm in normalized:
            continue
        if any(ord(c) < 32 for c in s):
            continue
        chosen.append((tid, p, s))
        normalized.add(norm)
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        chosen = [(ids[0].item(), vals[0].item(), bundle.tokenizer.decode([ids[0].item()])),
                  (ids[1].item(), vals[1].item(), bundle.tokenizer.decode([ids[1].item()]))]
    return {
        "d1_id": chosen[0][0], "d1_prob": chosen[0][1], "d1_text": chosen[0][2],
        "d2_id": chosen[1][0], "d2_prob": chosen[1][1], "d2_text": chosen[1][2],
    }


def repeat_inputs(inputs: dict[str, torch.Tensor], batch: int) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in inputs.items():
        if v.shape[0] == 1:
            reps = [batch] + [1] * (v.ndim - 1)
            out[k] = v.repeat(*reps)
        else:
            out[k] = v
    return out
