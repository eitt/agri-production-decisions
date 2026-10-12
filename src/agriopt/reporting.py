from __future__ import annotations

from pathlib import Path
import json
import shutil
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiment import resolve_run_dir
from .instances import CATALOG
from .metrics import (
    additive_epsilon,
    as_minimization_front,
    hypervolume_2d,
    igd_plus,
    nondominated,
    normalize,
    pooled_empirical_reference,
    reference_scaling,
)
from .paths import repo_root

METHOD_LABELS = {
    "exact": "SCIP epsilon-constraint",
    "nsga2": "NSGA-II",
    "nsga2_qp": "NSGA-II + full refinement",
    "hnsga2": "Surrogate-guided H-NSGA-II",
}
METHOD_ORDER = ["exact", "nsga2", "nsga2_qp", "hnsga2"]
MARKERS = {"exact": "o", "nsga2": "s", "nsga2_qp": "^", "hnsga2": "D"}
LINESTYLES = {"exact": "-", "nsga2": "--", "nsga2_qp": "-.", "hnsga2": ":"}


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _front_path(method_dir: Path, method: str) -> Path | None:
    candidates = {
        "exact": ["exact_front_scip.csv"],
        "nsga2": ["nsga2_front.csv"],
        "nsga2_qp": ["nsga2_qp_front.csv"],
        "hnsga2": ["hnsga2_front.csv"],
    }.get(method, [])
    for name in candidates:
        p = method_dir / name
        if p.exists():
            return p
    return None


def collect_fronts(run_dir: str | Path) -> pd.DataFrame:
    run = Path(run_dir)
    index_path = run / "run_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing {index_path}")
    idx = pd.read_csv(index_path)
    rows = []
    for _, r in idx[idx["status"].eq("ok")].iterrows():
        d = run / str(r["path"])
        p = _front_path(d, str(r["method"]))
        if p is None:
            continue
        f = pd.read_csv(p)
        if "income_mcop" not in f or "risk" not in f:
            continue
        for j, x in f.iterrows():
            if not np.isfinite(x["income_mcop"]) or not np.isfinite(x["risk"]):
                continue
            rows.append(
                {
                    "instance": r["instance"],
                    "method": r["method"],
                    "seed": int(r["seed"]),
                    "solution_id": int(x.get("solution_id", j)),
                    "income_mcop": float(x["income_mcop"]),
                    "risk": float(x["risk"]),
                    "runtime_s": float(r["runtime_s"]),
                    "path": str(d),
                }
            )
    return pd.DataFrame(rows)


def _run_extra_metrics(method_dir: Path, method: str) -> dict[str, float]:
    if method not in {"nsga2_qp", "hnsga2"}:
        return {"qp_calls": 0.0, "useful_refinements": 0.0, "refinement_hit_rate": math.nan}
    prefix = "nsga2_qp" if method == "nsga2_qp" else "hnsga2"
    hist = method_dir / f"{prefix}_history.csv"
    if not hist.exists():
        return {"qp_calls": math.nan, "useful_refinements": math.nan, "refinement_hit_rate": math.nan}
    h = pd.read_csv(hist)
    calls = float(h.get("refinement_calls", pd.Series(dtype=float)).sum())
    useful = float(h.get("useful_refinements", pd.Series(dtype=float)).sum())
    return {
        "qp_calls": calls,
        "useful_refinements": useful,
        "refinement_hit_rate": useful / calls if calls > 0 else math.nan,
    }


def compute_run_metrics(run_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run = Path(run_dir)
    fronts = collect_fronts(run)
    idx = pd.read_csv(run / "run_index.csv")
    if fronts.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    for instance, fi in fronts.groupby("instance"):
        ref_raw = pooled_empirical_reference([
            as_minimization_front(front["income_mcop"], front["risk"])
            for _, front in fi.groupby(["method", "seed"], sort=False)
        ])
        ref_source = "pooled_empirical_nondominated_union"
        ideal, span = reference_scaling(ref_raw)
        ref = normalize(ref_raw, ideal, span)
        for (method, seed), fm in fi.groupby(["method", "seed"]):
            approx = as_minimization_front(fm["income_mcop"], fm["risk"])
            approx = nondominated(approx)
            approx_n = normalize(approx, ideal, span)
            run_row = idx[(idx["instance"].eq(instance)) & (idx["method"].eq(method)) & (idx["seed"].eq(seed))]
            runtime = float(run_row.iloc[0]["runtime_s"]) if len(run_row) else float(fm["runtime_s"].iloc[0])
            method_dir = run / str(run_row.iloc[0]["path"]) if len(run_row) else Path(fm["path"].iloc[0])
            extra = _run_extra_metrics(method_dir, str(method))
            rows.append(
                {
                    "instance": instance,
                    "method": method,
                    "seed": int(seed),
                    "reference_source": ref_source,
                    "front_points": int(len(approx)),
                    "hypervolume": hypervolume_2d(approx_n, reference_point=(1.10, 1.10)),
                    "igd_plus": igd_plus(approx_n, ref),
                    "epsilon_additive": additive_epsilon(approx_n, ref),
                    "runtime_s": runtime,
                    "max_income_mcop": float(fm["income_mcop"].max()),
                    "min_risk": float(fm["risk"].min()),
                    **extra,
                }
            )
    seed_level = pd.DataFrame(rows)
    summary = (
        seed_level.groupby(["instance", "method"], observed=True)
        .agg(
            runs=("seed", "size"),
            hv_median=("hypervolume", "median"),
            hv_q1=("hypervolume", lambda s: s.quantile(0.25)),
            hv_q3=("hypervolume", lambda s: s.quantile(0.75)),
            igd_median=("igd_plus", "median"),
            epsilon_median=("epsilon_additive", "median"),
            runtime_median_s=("runtime_s", "median"),
            runtime_q1_s=("runtime_s", lambda s: s.quantile(0.25)),
            runtime_q3_s=("runtime_s", lambda s: s.quantile(0.75)),
            qp_calls_median=("qp_calls", "median"),
            hit_rate_median=("refinement_hit_rate", "median"),
            front_points_median=("front_points", "median"),
        )
        .reset_index()
    )
    return seed_level, summary


def _declutter(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.25)
    ax.tick_params(length=3)




def _story_title(fig, ax, title: str, subtitle: str | None = None):
    fig.suptitle(title, x=0.125, y=0.98, ha="left", va="top", fontsize=15, fontweight="bold")
    if subtitle:
        fig.text(0.125, 0.925, subtitle, ha="left", va="top", fontsize=9)
    fig.subplots_adjust(top=0.82)


def _save_figure(fig, stem: Path, dpi: int = 300, export_pdf: bool = True):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    if export_pdf:
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_pareto(fronts: pd.DataFrame, seed_level: pd.DataFrame, instance: str, out_dir: Path, dpi: int, export_pdf: bool):
    f = fronts[fronts["instance"].eq(instance)].copy()
    if f.empty:
        return
    # Plot one representative seed per stochastic method: median HV selection is handled through seed order
    # when summary metrics are available; otherwise the smallest seed gives deterministic reproducibility.
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for method in METHOD_ORDER:
        fm = f[f["method"].eq(method)]
        if fm.empty:
            continue
        candidates = seed_level[(seed_level["instance"].eq(instance)) & (seed_level["method"].eq(method))]
        if len(candidates):
            med = float(candidates["hypervolume"].median())
            seed = int(candidates.iloc[(candidates["hypervolume"] - med).abs().argmin()]["seed"])
        else:
            seed = int(sorted(fm["seed"].unique())[0])
        x = fm[fm["seed"].eq(seed)].sort_values("income_mcop")
        ax.plot(
            x["income_mcop"],
            x["risk"],
            marker=MARKERS.get(method, "o"),
            linestyle=LINESTYLES.get(method, "-"),
            linewidth=1.4,
            markersize=4,
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_xlabel("Expected income (million COP)")
    ax.set_ylabel("Correlated price risk")
    _story_title(fig, ax, f"Income–risk decisions for {instance}",
                 "Preferred movement is toward higher income and lower risk.")
    _declutter(ax)
    ax.legend(frameon=False, ncol=2, loc="best")
    _save_figure(fig, out_dir / f"pareto_{instance}", dpi, export_pdf)


def plot_runtime_scaling(summary: pd.DataFrame, out_dir: Path, dpi: int, export_pdf: bool):
    if summary.empty:
        return
    cat = CATALOG[["instance", "lots"]]
    z = summary.merge(cat, on="instance", how="left").dropna(subset=["lots", "runtime_median_s"])
    if z.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for method in METHOD_ORDER:
        m = z[z["method"].eq(method)].sort_values("lots")
        if m.empty:
            continue
        ax.plot(m["lots"], m["runtime_median_s"], marker=MARKERS.get(method, "o"),
                linestyle=LINESTYLES.get(method, "-"), linewidth=1.4, label=METHOD_LABELS.get(method, method))
    ax.set_yscale("log")
    ax.set_xlabel("Number of lots")
    ax.set_ylabel("Median runtime (s, log scale)")
    _story_title(fig, ax, "Computational cost as planning scale increases",
                 "Log scaling keeps exact and evolutionary methods comparable on one axis.")
    _declutter(ax)
    ax.legend(frameon=False, ncol=2, loc="best")
    _save_figure(fig, out_dir / "runtime_scaling", dpi, export_pdf)


def plot_quality_time(summary: pd.DataFrame, out_dir: Path, dpi: int, export_pdf: bool):
    z = summary.dropna(subset=["hv_median", "runtime_median_s"]).copy()
    if z.empty:
        return
    agg = z.groupby("method", observed=True).agg(hv=("hv_median", "median"), runtime=("runtime_median_s", "median")).reset_index()
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for _, r in agg.iterrows():
        ax.scatter(r["runtime"], r["hv"], marker=MARKERS.get(r["method"], "o"), s=45)
        ax.annotate(METHOD_LABELS.get(r["method"], r["method"]), (r["runtime"], r["hv"]),
                    xytext=(6, 5), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Median runtime across instances (s, log scale)")
    ax.set_ylabel("Median normalized hypervolume")
    _story_title(fig, ax, "Decision-front quality versus computational effort",
                 "Higher hypervolume and lower runtime indicate a more efficient decision method.")
    _declutter(ax)
    _save_figure(fig, out_dir / "quality_vs_time", dpi, export_pdf)


def plot_refinement_efficiency(seed_level: pd.DataFrame, out_dir: Path, dpi: int, export_pdf: bool):
    z = seed_level[seed_level["method"].isin(["nsga2_qp", "hnsga2"])].dropna(subset=["qp_calls", "hypervolume"])
    if z.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for method in ["nsga2_qp", "hnsga2"]:
        m = z[z["method"].eq(method)]
        if m.empty:
            continue
        ax.scatter(m["qp_calls"], m["hypervolume"], marker=MARKERS.get(method, "o"), s=32,
                   label=METHOD_LABELS.get(method, method))
    ax.set_xlabel("Exact refinement calls")
    ax.set_ylabel("Normalized hypervolume")
    _story_title(fig, ax, "Does the surrogate spend the refinement budget efficiently?",
                 "Better outcomes appear higher and farther left.")
    _declutter(ax)
    ax.legend(frameon=False)
    _save_figure(fig, out_dir / "surrogate_refinement_efficiency", dpi, export_pdf)


def _write_table(df: pd.DataFrame, stem: Path, export_latex: bool = True):
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    df.to_markdown(stem.with_suffix(".md"), index=False)
    if export_latex:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stem.with_suffix(".tex").write_text(
                df.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.4g}"), encoding="utf-8"
            )


def build_report(run: str | Path = "latest", *, dpi: int = 300, export_pdf: bool = True, export_latex: bool = True) -> Path:
    root = repo_root()
    run_dir = resolve_run_dir(run, root)
    fronts = collect_fronts(run_dir)
    seed_level, summary = compute_run_metrics(run_dir)
    artifact_root = root / "artifacts" / run_dir.name
    fig_dir = artifact_root / "figures"
    table_dir = artifact_root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    if not seed_level.empty:
        _write_table(seed_level, table_dir / "seed_level_metrics", export_latex=False)
    if not summary.empty:
        ordered = summary.copy()
        ordered["method_order"] = ordered["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
        ordered = ordered.sort_values(["instance", "method_order"]).drop(columns="method_order")
        _write_table(ordered, table_dir / "method_summary", export_latex=export_latex)
    _write_table(CATALOG, table_dir / "benchmark_instances", export_latex=export_latex)

    if not fronts.empty:
        for instance in sorted(fronts["instance"].unique()):
            plot_pareto(fronts, seed_level, instance, fig_dir, dpi, export_pdf)
    plot_runtime_scaling(summary, fig_dir, dpi, export_pdf)
    plot_quality_time(summary, fig_dir, dpi, export_pdf)
    plot_refinement_efficiency(seed_level, fig_dir, dpi, export_pdf)

    forecast_selection = root / "data" / "interim" / "forecast_model_selection.csv"
    if forecast_selection.exists():
        fs = pd.read_csv(forecast_selection)
        _write_table(fs, table_dir / "forecast_model_selection", export_latex=export_latex)

    # Convenience mirror for LaTeX/Overleaf: immutable artifacts remain under artifacts/<run_id>,
    # while paper/generated always reflects the most recently reported run.
    paper_generated = root / "paper" / "generated"
    if paper_generated.exists():
        shutil.rmtree(paper_generated)
    (paper_generated / "figures").mkdir(parents=True, exist_ok=True)
    (paper_generated / "tables").mkdir(parents=True, exist_ok=True)
    for src in fig_dir.glob("*.pdf"):
        shutil.copy2(src, paper_generated / "figures" / src.name)
    for src in table_dir.glob("*.tex"):
        shutil.copy2(src, paper_generated / "tables" / src.name)
    (paper_generated / "RUN_ID.txt").write_text(run_dir.name, encoding="utf-8")

    readme = artifact_root / "README.md"
    readme.write_text(
        "# Generated publication artifacts\n\n"
        f"Source experiment: `{run_dir.relative_to(root)}`.\n\n"
        "## Figures\n\n"
        "- `pareto_<instance>`: income-risk decision front for each benchmark.\n"
        "- `runtime_scaling`: computational cost as the number of lots grows.\n"
        "- `quality_vs_time`: normalized hypervolume against runtime.\n"
        "- `surrogate_refinement_efficiency`: quality against exact-refinement calls.\n\n"
        "The figure design follows a message-first workflow: one question per chart, direct axis labels, "
        "minimal decoration, limited visual encodings, and explicit preferred decision directions.\n\n"
        "## Tables\n\n"
        "- `benchmark_instances`: controlled instance design.\n"
        "- `method_summary`: median decision-front quality, runtime, and refinement efficiency.\n"
        "- `seed_level_metrics`: audit table for every stochastic run.\n"
        "- `forecast_model_selection`: model selected per crop when forecasting outputs are available.\n",
        encoding="utf-8",
    )
    return artifact_root
