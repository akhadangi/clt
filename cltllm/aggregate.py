from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import dump_json, ensure_dir, load_json


def _bootstrap_mean_ci(values: np.ndarray, seed: int = 20260905, reps: int = 4000):
    x = values[np.isfinite(values)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    if len(x) == 1:
        return float(x[0]), float(x[0]), float(x[0]), 1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(reps, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975)), int(len(x))


def _layer_set(value) -> set[int]:
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return {int(x) for x in str(value).split(",") if str(x).strip()}


def _minimal_rows_at_threshold(group: pd.DataFrame, threshold: float) -> pd.DataFrame:
    g = group[
        (group["supported"] == 1)
        & (group.get("layerwise_supported", 1) == 1)
        & (group["group_count"] > 0)
        & (group.get("base_effect_supported", 1) == 1)
        & (group["mediation_clipped"] >= threshold)
    ].copy()
    if not len(g):
        return g
    sets = [_layer_set(x) for x in g["source_layers"]]
    keep = []
    for i, s in enumerate(sets):
        keep.append(not any(t < s for j, t in enumerate(sets) if j != i))
    return g[np.array(keep, dtype=bool)]


def _carrier_threshold_analysis(d: pd.DataFrame, thresholds: list[float], out: Path) -> None:
    if not len(d) or "source_layers" not in d.columns:
        return
    rows = []
    chosen_masks = []
    layerwise = d[(d.get("layerwise_supported", 0) == 1) & (d["supported"] == 1)].copy()
    for (model, prompt_id, part), gg in layerwise.groupby(["model", "prompt_id", "partition_groups"]):
        for thr in thresholds:
            mm = _minimal_rows_at_threshold(gg, float(thr))
            if len(mm):
                # Multiple incomparable minimal sets are retained. For cross-partition
                # robustness we choose the smallest physical layer set, then the one
                # with the strongest mediation as a deterministic representative.
                best = mm.sort_values(["source_layer_count", "mediation_clipped"], ascending=[True, False]).iloc[0]
                rows.append({
                    "model": model,
                    "prompt_id": prompt_id,
                    "partition_groups": int(part),
                    "threshold": float(thr),
                    "n_minimal_sets": int(len(mm)),
                    "minimal_group_count": int(mm["group_count"].min()),
                    "minimal_layer_count": int(best["source_layer_count"]),
                    "minimal_layer_fraction": float(best["source_layer_fraction"]),
                    "representative_source_layers": best["source_layers"],
                    "representative_mediation": float(best["mediation_clipped"]),
                })
                chosen_masks.append({
                    "model": model, "prompt_id": prompt_id, "partition_groups": int(part),
                    "threshold": float(thr), "mask": _layer_set(best["source_layers"])
                })
            else:
                rows.append({
                    "model": model, "prompt_id": prompt_id, "partition_groups": int(part),
                    "threshold": float(thr), "n_minimal_sets": 0,
                    "minimal_group_count": np.nan, "minimal_layer_count": np.nan,
                    "minimal_layer_fraction": np.nan, "representative_source_layers": "",
                    "representative_mediation": np.nan,
                })
    detail = pd.DataFrame(rows)
    if len(detail):
        detail.to_csv(out / "carrier_threshold_sensitivity.csv", index=False)
        summary_rows = []
        for (model, thr, part), gg in detail.groupby(["model", "threshold", "partition_groups"]):
            vals = gg["minimal_layer_fraction"].to_numpy(dtype=float)
            m, lo, hi, n = _bootstrap_mean_ci(vals)
            summary_rows.append({
                "model": model, "threshold": thr, "partition_groups": part,
                "minimal_layer_fraction_mean": m, "ci95_lo": lo, "ci95_hi": hi,
                "n_prompts_recovered": n, "n_prompts_total": int(gg["prompt_id"].nunique()),
            })
        pd.DataFrame(summary_rows).to_csv(out / "carrier_threshold_summary.csv", index=False)

    if chosen_masks:
        cm = pd.DataFrame(chosen_masks)
        jrows = []
        for (model, prompt_id, thr), gg in cm.groupby(["model", "prompt_id", "threshold"]):
            recs = gg.to_dict("records")
            js = []
            for i in range(len(recs)):
                for j in range(i + 1, len(recs)):
                    a, b = recs[i]["mask"], recs[j]["mask"]
                    union = a | b
                    js.append(len(a & b) / len(union) if union else 1.0)
            if js:
                jrows.append({
                    "model": model, "prompt_id": prompt_id, "threshold": thr,
                    "partition_jaccard_mean": float(np.mean(js)),
                    "n_partitions": int(len(recs)),
                })
        if jrows:
            jd = pd.DataFrame(jrows)
            jd.to_csv(out / "carrier_partition_robustness.csv", index=False)
            srows = []
            for (model, thr), gg in jd.groupby(["model", "threshold"]):
                m, lo, hi, n = _bootstrap_mean_ci(gg["partition_jaccard_mean"].to_numpy(dtype=float))
                srows.append({"model": model, "threshold": thr, "jaccard_mean": m, "ci95_lo": lo, "ci95_hi": hi, "n_prompts": n})
            pd.DataFrame(srows).to_csv(out / "carrier_partition_robustness_summary.csv", index=False)


def _carrier_pi_uniform(d: pd.DataFrame, thresholds: list[float], out: Path) -> None:
    """Operational audit-persistence statistic across preregistered layer partitions.

    For each prompt and threshold, take every inclusion-minimal candidate from the
    finest supported partition as a reference physical layer set B*. For each
    coarser partition, map B* to the union of coarse groups that intersect it and
    test whether that projected set is itself inclusion-minimal at the same
    mediation threshold. With uniform weight over the declared partitions, the
    fraction of successful projections is an operational analogue of Pi_nu for
    this cache-carrier audit. It remains audit-relative and is not a physical
    observable.
    """
    required = {"model", "prompt_id", "partition_groups", "source_layers",
                "group_count", "supported", "layerwise_supported",
                "base_effect_supported", "mediation_clipped"}
    if not len(d) or not required.issubset(d.columns):
        return
    z = d[(d.supported == 1) & (d.layerwise_supported == 1) & (d.base_effect_supported == 1)].copy()
    if not len(z):
        return
    detail_rows = []
    prompt_rows = []
    for (model, prompt_id), pg in z.groupby(["model", "prompt_id"]):
        parts = sorted(int(x) for x in pg.partition_groups.unique())
        if not parts:
            continue
        finest = max(parts)
        fine = pg[pg.partition_groups == finest]
        for thr in thresholds:
            refs = _minimal_rows_at_threshold(fine, thr)
            if not len(refs):
                continue
            candidate_pis = []
            for _, ref in refs.iterrows():
                B = _layer_set(ref.source_layers)
                successes = 0
                supported_parts = 0
                projections = []
                for part in parts:
                    q = pg[pg.partition_groups == part]
                    singles = q[(q.group_count == 1) & (q.supported == 1)]
                    group_masks = [_layer_set(x) for x in singles.source_layers]
                    if not group_masks:
                        continue
                    # The image of B under this finite grouping is the union of every
                    # group touched by B. This is deterministic and declared before
                    # examining whether the projected set passes.
                    proj = set().union(*(g for g in group_masks if g & B)) if B else set()
                    supported_parts += 1
                    minima = _minimal_rows_at_threshold(q, thr)
                    min_masks = [_layer_set(x) for x in minima.source_layers]
                    ok = any(m == proj for m in min_masks)
                    successes += int(ok)
                    projections.append({"partition_groups": part, "projection": ",".join(map(str, sorted(proj))), "minimal": int(ok)})
                pi = successes / supported_parts if supported_parts else np.nan
                candidate_pis.append(pi)
                detail_rows.append({
                    "model": model, "prompt_id": prompt_id, "threshold": float(thr),
                    "reference_partition": finest,
                    "reference_source_layers": ",".join(map(str, sorted(B))),
                    "pi_uniform": pi, "n_supported_partitions": supported_parts,
                    "n_successful_projections": successes,
                    "projection_audit_json": json.dumps(projections, sort_keys=True),
                })
            vals = np.asarray(candidate_pis, dtype=float)
            prompt_rows.append({
                "model": model, "prompt_id": prompt_id, "threshold": float(thr),
                "n_finest_minimal_candidates": int(len(vals)),
                "pi_uniform_mean_candidates": float(np.nanmean(vals)),
                "pi_uniform_min_candidates": float(np.nanmin(vals)),
                "pi_uniform_max_candidates": float(np.nanmax(vals)),
            })
    if detail_rows:
        pd.DataFrame(detail_rows).to_csv(out / "carrier_pi_uniform_detail.csv", index=False)
    if prompt_rows:
        pr = pd.DataFrame(prompt_rows)
        pr.to_csv(out / "carrier_pi_uniform_by_prompt.csv", index=False)
        srows=[]
        for (model, thr), gg in pr.groupby(["model", "threshold"]):
            m,lo,hi,n=_bootstrap_mean_ci(gg.pi_uniform_mean_candidates.to_numpy(dtype=float))
            srows.append({
                "model": model, "threshold": thr, "pi_uniform_mean": m,
                "ci95_lo": lo, "ci95_hi": hi, "n_prompts": n,
                "pi_uniform_min_across_prompts": float(gg.pi_uniform_min_candidates.min()),
                "pi_uniform_max_across_prompts": float(gg.pi_uniform_max_candidates.max()),
            })
        pd.DataFrame(srows).to_csv(out / "carrier_pi_uniform_summary.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", required=False)
    args = ap.parse_args()
    root = Path(args.input)
    out = ensure_dir(args.output)
    cfg = load_json(args.config) if args.config else {}

    names = [
        "hidden_liability", "activation_patching", "adaptive_continuity",
        "reconstruction_equivalence", "carrier_audit", "surface_stimuli",
        "surface_stimuli_blinded", "surface_stimuli_key", "hard_process_reconstruction",
    ]
    manifest = {}
    for name in names:
        files = sorted(set(root.glob(f"rank_*/**/{name}.csv")))
        frames = []
        for f in files:
            try:
                df = pd.read_csv(f)
                if len(df):
                    frames.append(df)
            except pd.errors.EmptyDataError:
                pass
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df.to_csv(out / f"{name}.csv", index=False)
            manifest[name] = {"rows": int(len(df)), "files": [str(f) for f in files]}

    # Collect model-level success/failure records so a partial run can never look complete.
    status_rows = []
    for f in sorted(root.glob("rank_*/*/status.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            r["status_file"] = str(f)
            status_rows.append(r)
        except Exception:
            pass
    if status_rows:
        pd.DataFrame(status_rows).to_csv(out / "model_status.csv", index=False)
        manifest["model_status"] = {
            "models": len(status_rows),
            "failed": int(sum(r.get("status") != "ok" for r in status_rows)),
        }

    # Model-level, prompt-clustered summaries. Technical repetitions are first
    # averaged within prompt, then prompts are bootstrapped.
    hp = out / "hidden_liability.csv"
    if hp.exists():
        d = pd.read_csv(hp)
        z = d[(d.metric == "swd") & (d.measurement_map == "identity") & (d.layer >= 0)].copy()
        if len(z):
            max_layer = z.groupby("model")["layer"].transform("max").replace(0, 1)
            z["depth"] = z.layer / max_layer
            z["depth_bin"] = pd.cut(z.depth, [-.001,.2,.4,.6,.8,1.001], labels=["0-.2",".2-.4",".4-.6",".6-.8",".8-1"])
            g = z.groupby(["model","prompt_id","horizon","depth_bin"], observed=True).value.mean().reset_index()
            rows=[]
            for keys, gg in g.groupby(["model","horizon","depth_bin"], observed=True):
                m,lo,hi,n=_bootstrap_mean_ci(gg.value.to_numpy())
                rows.append({"model":keys[0],"horizon":keys[1],"depth_bin":str(keys[2]),"mean":m,"ci95_lo":lo,"ci95_hi":hi,"n_prompts":n})
            pd.DataFrame(rows).to_csv(out / "hidden_swd_summary.csv", index=False)

            # Measurement-map robustness: within each prompt/layer/horizon/metric,
            # quantify relative dispersion across preregistered representation maps.
            cg = d[(d.layer >= 0) & d.metric.isin(["swd", "mmd_rff", "energy"])].copy()
            if len(cg):
                cg_prompt = cg.groupby(["model","prompt_id","horizon","layer","metric"]).value.agg(["mean","std","count"]).reset_index()
                cg_prompt["cv_across_measurement_maps"] = cg_prompt["std"] / cg_prompt["mean"].abs().clip(lower=1e-12)
                cg_prompt.to_csv(out / "hidden_measurement_map_robustness.csv", index=False)

    pp = out / "activation_patching.csv"
    if pp.exists():
        d = pd.read_csv(pp)
        if len(d):
            usable = d[d.get("base_effect_supported", 1) == 1].copy()
            if len(usable):
                gp=usable.groupby(["model","prompt_id"]).mediation_clipped.max().reset_index()
                rows=[]
                for model, gg in gp.groupby("model"):
                    m,lo,hi,n=_bootstrap_mean_ci(gg.mediation_clipped.to_numpy())
                    rows.append({"model":model,"max_mediation_mean":m,"ci95_lo":lo,"ci95_hi":hi,"n_prompts":n})
                pd.DataFrame(rows).to_csv(out / "patching_summary.csv", index=False)

    cpath = out / "carrier_audit.csv"
    if cpath.exists():
        d = pd.read_csv(cpath)
        if len(d):
            passing = d[(d.supported == 1) & (d.is_minimal_at_threshold == 1)]
            if len(passing):
                tab = passing.groupby(["model", "partition_groups", "group_count"]).agg(
                    n_prompts=("prompt_id", "nunique"),
                    mediation_mean=("mediation_clipped", "mean"),
                    layer_fraction_mean=("source_layer_fraction", "mean"),
                ).reset_index()
                tab.to_csv(out / "carrier_minimal_summary.csv", index=False)
            thresholds = [float(x) for x in cfg.get("carrier_robustness_thresholds", [0.7,0.8,0.9])]
            _carrier_threshold_analysis(d, thresholds, out)
            _carrier_pi_uniform(d, thresholds, out)

    apath = out / "adaptive_continuity.csv"
    if apath.exists():
        d=pd.read_csv(apath)
        if len(d):
            g=d.groupby(["model","condition","prompt_id","horizon"]).output_js.mean().reset_index()
            rows=[]
            for keys,gg in g.groupby(["model","condition","horizon"]):
                m,lo,hi,n=_bootstrap_mean_ci(gg.output_js.to_numpy())
                rows.append({"model":keys[0],"condition":keys[1],"horizon":keys[2],"output_js_mean":m,"ci95_lo":lo,"ci95_hi":hi,"n_prompts":n})
            pd.DataFrame(rows).to_csv(out / "adaptive_summary.csv", index=False)

            # Direct equality check for copyability and reconstruction controls.
            pivot = g.pivot_table(index=["model","prompt_id","horizon"], columns="condition", values="output_js")
            if "persistent_live" in pivot.columns:
                if "persistent_copy" in pivot.columns:
                    pivot["abs_live_minus_copy"] = (pivot["persistent_live"] - pivot["persistent_copy"]).abs()
                if "reconstructed" in pivot.columns:
                    pivot["abs_live_minus_reconstructed"] = (pivot["persistent_live"] - pivot["reconstructed"]).abs()
                pivot.reset_index().to_csv(out / "adaptive_equivalence_checks.csv", index=False)

    dump_json(manifest, out / "aggregate_manifest.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
