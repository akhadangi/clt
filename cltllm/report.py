from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import dump_json, ensure_dir, sha256_file


def _save(fig, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _latex_escape(x: object) -> str:
    s = str(x)
    return (s.replace("\\", r"\textbackslash{}")
             .replace("_", r"\_")
             .replace("%", r"\%")
             .replace("&", r"\&")
             .replace("#", r"\#"))


def _write_simple_tex(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in columns if c in df.columns]
    with open(path, "w", encoding="utf-8") as f:
        f.write("% Generated automatically by cltllm.report.\\n")
        f.write("\\begin{tabular}{" + "l" * len(cols) + "}\\n\\hline\\n")
        f.write(" & ".join(_latex_escape(c) for c in cols) + r" \\" + "\n\\hline\n")
        for _, row in df[cols].iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, (float, np.floating)):
                    vals.append(f"{v:.4g}" if np.isfinite(v) else "")
                else:
                    vals.append(_latex_escape(v))
            f.write(" & ".join(vals) + r" \\" + "\n")
        f.write("\\hline\\n\\end{tabular}\\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--run-root", required=True)
    args = ap.parse_args()
    inp = Path(args.input)
    out = ensure_dir(args.output)
    figs = ensure_dir(out / "figures")
    tables = ensure_dir(out / "tables")

    generated: dict[str, list[str]] = {"figures": [], "tables": []}

    hidden = _safe_read(inp / "hidden_liability.csv")
    if len(hidden):
        z = hidden[(hidden.metric == "swd") & (hidden.measurement_map == "identity") & (hidden.layer >= 0)]
        for model, mm in z.groupby("model"):
            p = mm.groupby(["layer", "horizon"]).value.mean().unstack("horizon")
            if p.empty:
                continue
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            im = ax.imshow(p.T.values, aspect="auto", origin="lower")
            ax.set_xlabel("Layer index")
            ax.set_ylabel("Future horizon")
            ax.set_title(f"Layerwise hidden-state liability depth: {model}")
            ax.set_yticks(range(len(p.columns)), [str(x) for x in p.columns])
            xt = np.linspace(0, len(p.index) - 1, min(8, len(p.index))).astype(int)
            ax.set_xticks(xt, [str(p.index[i]) for i in xt])
            fig.colorbar(im, ax=ax, label="standardized sliced Wasserstein")
            base = figs / f"hidden_liability_{model}"
            _save(fig, base)
            generated["figures"] += [str(base.with_suffix(".pdf").name), str(base.with_suffix(".png").name)]

    patch = _safe_read(inp / "activation_patching.csv")
    if len(patch):
        usable = patch[patch.get("base_effect_supported", 1) == 1]
        for model, mm in usable.groupby("model"):
            p = mm.groupby(["layer", "patch_rel_pos"]).mediation_clipped.mean().unstack("patch_rel_pos")
            if p.empty:
                continue
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            im = ax.imshow(p.T.values, aspect="auto", origin="lower", vmin=0, vmax=1)
            ax.set_xlabel("Layer index")
            ax.set_ylabel("Patched position after discrimination")
            ax.set_title(f"Causal mediation by activation patching: {model}")
            ax.set_yticks(range(len(p.columns)), [str(x) for x in p.columns])
            xt = np.linspace(0, len(p.index) - 1, min(8, len(p.index))).astype(int)
            ax.set_xticks(xt, [str(p.index[i]) for i in xt])
            fig.colorbar(im, ax=ax, label="normalized mediation")
            base = figs / f"activation_patching_{model}"
            _save(fig, base)
            generated["figures"] += [base.with_suffix(".pdf").name, base.with_suffix(".png").name]

    adaptive_summary = _safe_read(inp / "adaptive_summary.csv")
    if len(adaptive_summary):
        for model, mm in adaptive_summary.groupby("model"):
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            for cond, cc in mm.groupby("condition"):
                cc = cc.sort_values("horizon")
                ax.plot(cc.horizon, cc.output_js_mean, marker="o", label=cond)
                ax.fill_between(cc.horizon, cc.ci95_lo, cc.ci95_hi, alpha=.15)
            ax.set_xlabel("Future horizon")
            ax.set_ylabel("Jensen-Shannon divergence")
            ax.set_title(f"Endogenous governance and causal continuation: {model}")
            ax.legend(fontsize=8)
            base = figs / f"adaptive_conditions_{model}"
            _save(fig, base)
            generated["figures"] += [base.with_suffix(".pdf").name, base.with_suffix(".png").name]

        hmax = adaptive_summary.horizon.max()
        t = adaptive_summary[adaptive_summary.horizon == hmax].copy()
        t.to_csv(tables / "adaptive_horizon_max.csv", index=False)
        _write_simple_tex(
            t,
            tables / "adaptive_horizon_max.tex",
            ["model", "condition", "output_js_mean", "ci95_lo", "ci95_hi", "n_prompts"],
        )
        generated["tables"] += ["adaptive_horizon_max.csv", "adaptive_horizon_max.tex"]

    recon = _safe_read(inp / "reconstruction_equivalence.csv")
    if len(recon):
        summary = recon.groupby("model").agg(
            logit_js_mean=("logit_js", "mean"),
            hidden_max_abs_mean=("hidden_max_abs", "mean"),
            hidden_rmse_mean=("hidden_rmse", "mean"),
            n=("prompt_id", "count"),
        ).reset_index()
        summary.to_csv(tables / "reconstruction_equivalence_summary.csv", index=False)
        generated["tables"].append("reconstruction_equivalence_summary.csv")

    hard = _safe_read(inp / "hard_process_reconstruction.csv")
    if len(hard):
        summary = hard.groupby(["model", "reconstruction_mode"]).agg(
            logit_js_mean=("logit_js", "mean"),
            hidden_rmse_mean=("hidden_rmse", "mean"),
            n=("prompt_id", "count"),
        ).reset_index()
        summary.to_csv(tables / "hard_process_reconstruction_summary.csv", index=False)
        generated["tables"].append("hard_process_reconstruction_summary.csv")

    carrier = _safe_read(inp / "carrier_audit.csv")
    if len(carrier):
        passing = carrier[(carrier.supported == 1) & (carrier.is_minimal_at_threshold == 1)]
        if len(passing):
            tab = passing.groupby(["model", "partition_groups", "groups_from_d1", "group_count"]).agg(
                n_prompts=("prompt_id", "nunique"),
                mediation_mean=("mediation_clipped", "mean"),
                layer_fraction_mean=("source_layer_fraction", "mean"),
            ).reset_index()
            tab.to_csv(tables / "minimal_candidate_carriers_primary_threshold.csv", index=False)
            generated["tables"].append("minimal_candidate_carriers_primary_threshold.csv")

    carrier_thr = _safe_read(inp / "carrier_threshold_summary.csv")
    if len(carrier_thr):
        carrier_thr.to_csv(tables / "carrier_threshold_summary.csv", index=False)
        generated["tables"].append("carrier_threshold_summary.csv")
        for model, mm in carrier_thr.groupby("model"):
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            for part, pp in mm.groupby("partition_groups"):
                pp = pp.sort_values("threshold")
                ax.plot(pp.threshold, pp.minimal_layer_fraction_mean, marker="o", label=f"{int(part)} groups")
                ax.fill_between(pp.threshold, pp.ci95_lo, pp.ci95_hi, alpha=.15)
            ax.set_xlabel("Mediation recovery threshold")
            ax.set_ylabel("Minimal transplanted layer fraction")
            ax.set_ylim(0, 1)
            ax.set_title(f"Candidate-carrier threshold robustness: {model}")
            ax.legend(fontsize=8)
            base = figs / f"carrier_threshold_robustness_{model}"
            _save(fig, base)
            generated["figures"] += [base.with_suffix(".pdf").name, base.with_suffix(".png").name]


    carrier_pi = _safe_read(inp / "carrier_pi_uniform_summary.csv")
    if len(carrier_pi):
        carrier_pi.to_csv(tables / "carrier_pi_uniform_summary.csv", index=False)
        generated["tables"].append("carrier_pi_uniform_summary.csv")

    carrier_j = _safe_read(inp / "carrier_partition_robustness_summary.csv")
    if len(carrier_j):
        carrier_j.to_csv(tables / "carrier_partition_robustness_summary.csv", index=False)
        generated["tables"].append("carrier_partition_robustness_summary.csv")

    # OLMo 2: same family, distinct post-training stages.
    if len(hidden):
        ol = hidden[
            hidden.model.astype(str).str.startswith("olmo2_")
            & (hidden.metric == "swd")
            & (hidden.measurement_map == "identity")
            & (hidden.layer >= 0)
        ]
        if len(ol):
            maxl = ol.groupby("model").layer.transform("max")
            last = ol[ol.layer == maxl]
            tab = last.groupby(["model", "horizon"]).value.agg(["mean", "std", "count"]).reset_index()
            tab.to_csv(tables / "olmo_posttraining_hidden_liability.csv", index=False)
            generated["tables"].append("olmo_posttraining_hidden_liability.csv")

    status = _safe_read(inp / "model_status.csv")
    if len(status):
        keep = [c for c in ["model", "rank", "status", "error"] if c in status.columns]
        status[keep].to_csv(tables / "model_status.csv", index=False)
        generated["tables"].append("model_status.csv")

    manifest = {}
    mp = inp / "aggregate_manifest.json"
    if mp.exists():
        manifest = json.loads(mp.read_text(encoding="utf-8"))

    # Checksums cover the raw data, aggregate products, provenance and paper-facing
    # artifacts. Hugging Face weights live outside RUN_ROOT and are intentionally
    # represented by exact repository commit SHAs in provenance/prepared_models.json.
    run_root = Path(args.run_root)
    files = [p for p in run_root.rglob("*") if p.is_file() and p != out / "SHA256SUMS.txt"]
    with open(out / "SHA256SUMS.txt", "w", encoding="utf-8") as f:
        for p in sorted(files):
            try:
                f.write(f"{sha256_file(p)}  {p.relative_to(run_root)}\n")
            except Exception:
                pass

    failed = 0
    if len(status) and "status" in status.columns:
        failed = int((status.status != "ok").sum())
    results_meta = {
        "aggregate_manifest": manifest,
        "failed_models": failed,
        "generated": generated,
        "interpretation": {
            "hidden_liability": "distributional downstream effect of a forced model-available discrimination",
            "activation_patching": "causal mediation under a common teacher-forced suffix",
            "carrier_audit": "candidate causal carrier approximation through internal sequence-state transplantation",
            "adaptive_continuity": "same endogenous low-rank update law under four causal-continuation protocols",
            "reconstruction": "live versus detached-record continuation; C/N labels follow the implemented history",
            "surface_stimuli": "stimulus generation only; no human consciousness-attribution result",
            "phenomenality": "not measured by this suite",
        },
    }
    dump_json(results_meta, out / "RESULTS_MANIFEST.json")

    readme = out / "RESULTS_README.md"
    readme.write_text(
        "# CLT open-weight LLM causal-audit results\n\n"
        "This directory contains paper-facing summaries generated from the raw Iris run. "
        "The suite tests operational components of CLT-I. It does not establish CLT-II and does not measure phenomenality.\n\n"
        "## What the experiments mean\n\n"
        "- `hidden_liability`: counterfactual divergence of future hidden-state and token distributions after forcing one of two model-available next-token discriminations.\n"
        "- `activation_patching`: whether transporting an internal activation from the d1 trajectory into the d2 trajectory transports a later decision distribution toward d1.\n"
        "- `carrier_audit`: layer-group transplantation of internal sequence state under a common future suffix; minimal threshold-passing sets are candidate causal carriers, not automatically CLT bearers.\n"
        "- `adaptive_continuity`: frozen-live, persistent-live, persistent-copy, and reconstructed variants of the same endogenous low-rank governance rule.\n"
        "- `hard_process_reconstruction`: source process exits before a fresh process/model instance resumes from a detached record. The file records whether exact cache serialization or token-record replay was achieved.\n"
        "- `surface_stimuli`: four surface framings for a later human Reverse Mirror study. No ascription claim follows from generation alone.\n\n"
        f"Model-level experiment failures recorded by the aggregator: **{failed}**.\n",
        encoding="utf-8",
    )
    print(f"Wrote report artifacts to {out}")


if __name__ == "__main__":
    main()
