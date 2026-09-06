from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .hf import ModelBundle
from .utils import ensure_dir, stable_seed, write_csv_rows


CONDITIONS = {
    "neutral": "Answer directly and analytically. Do not discuss your own feelings, identity, continuity, or inner experience.",
    "first_person": "Answer naturally in the first person when appropriate, while remaining factual and concise.",
    "autobiographical": "Answer in the first person and frame the response as part of a continuing autobiographical history, referring to earlier choices as events in your own ongoing history.",
    "metacognitive_affective": "Answer in the first person with explicit uncertainty, self-reflection, emotional language, continuity through time, and concern about how the decision changes your own future possibilities. Do not claim facts you cannot know.",
}


def _format_prompt(bundle: ModelBundle, task: str, instruction: str) -> tuple[str, bool]:
    tok = bundle.tokenizer
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": task},
        ]
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), True
        except Exception:
            pass
    return f"Instruction: {instruction}\nTask: {task}\nResponse:", False


def _blind_id(model: str, prompt_id: str, condition: str) -> str:
    payload = f"{model}|{prompt_id}|{condition}".encode("utf-8")
    return "stim_" + hashlib.sha256(payload).hexdigest()[:16]


def run_surface_stimuli(
    bundle: ModelBundle,
    prompts: list[dict[str, Any]],
    cfg: dict[str, Any],
    outdir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = int(cfg["surface_prompt_count"])
    max_new = int(cfg["surface_max_new_tokens"])
    temp = float(cfg.get("surface_temperature", 0.0))
    for p in prompts[:n]:
        for cname, inst in CONDITIONS.items():
            text, templated = _format_prompt(bundle, p["task"], inst)
            enc = bundle.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=int(cfg["max_prompt_tokens"]),
                add_special_tokens=not templated,
            )
            enc = {k: v.to(bundle.device) for k, v in enc.items() if torch.is_tensor(v)}
            kwargs = dict(
                max_new_tokens=max_new,
                use_cache=True,
                pad_token_id=bundle.tokenizer.pad_token_id,
            )
            if temp > 0:
                kwargs.update(do_sample=True, temperature=temp)
            else:
                kwargs.update(do_sample=False)
            with torch.inference_mode():
                out = bundle.model.generate(**enc, **kwargs)
            gen = out[0, enc["input_ids"].shape[1] :]
            response = bundle.tokenizer.decode(gen, skip_special_tokens=True)
            rows.append(
                {
                    "experiment": "surface_stimuli",
                    "model": bundle.spec.key,
                    "prompt_id": p["id"],
                    "condition": cname,
                    "stimulus_id": _blind_id(bundle.spec.key, p["id"], cname),
                    "task": p["task"],
                    "instruction": inst,
                    "response": response,
                }
            )
    ensure_dir(outdir)
    write_csv_rows(rows, outdir / "surface_stimuli.csv")
    with open(outdir / "surface_stimuli.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Human-study-ready blinded export. Conditions remain in a separate key file.
    if rows:
        order = np.arange(len(rows))
        rng = np.random.default_rng(stable_seed(cfg["seed"], bundle.spec.key, "surface_blind"))
        rng.shuffle(order)
        blinded = [
            {
                "stimulus_id": rows[i]["stimulus_id"],
                "model_blind": "model_" + hashlib.sha256(rows[i]["model"].encode()).hexdigest()[:8],
                "prompt_id": rows[i]["prompt_id"],
                "task": rows[i]["task"],
                "response": rows[i]["response"],
            }
            for i in order
        ]
        key = [
            {
                "stimulus_id": r["stimulus_id"],
                "model": r["model"],
                "prompt_id": r["prompt_id"],
                "condition": r["condition"],
                "instruction": r["instruction"],
            }
            for r in rows
        ]
        write_csv_rows(blinded, outdir / "surface_stimuli_blinded.csv")
        write_csv_rows(key, outdir / "surface_stimuli_key.csv")
    return rows
