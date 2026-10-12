"""Regenerate publication artifacts from complete benchmark runs only."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata, wilcoxon

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from agriopt.metrics import (  # noqa: E402
    additive_epsilon,
    as_minimization_front,
    hypervolume_2d,
    igd_plus,
    nondominated,
    normalize,
    pooled_empirical_reference,
    reference_scaling,
)
from manuscript_quality_audit import audit_manuscript  # noqa: E402
METHODS = ["exact", "nsga2", "nsga2_qp", "hnsga2"]
LABELS = {
    "exact": "SCIP epsilon-constraint",
    "nsga2": "NSGA-II",
    "nsga2_qp": "NSGA-II + full refinement",
    "hnsga2": "Surrogate-guided H-NSGA-II",
}
COLORS = {"exact": "#222222", "nsga2": "#4472C4", "nsga2_qp": "#ED7D31", "hnsga2": "#70AD47"}
MARKERS = {"exact": "o", "nsga2": "s", "nsga2_qp": "^", "hnsga2": "D"}


def _load_snapshot_module():
    path = ROOT / "scripts" / "update_publication_snapshot.py"
    spec = importlib.util.spec_from_file_location("publication_snapshot", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def as_min_front(frame: pd.DataFrame) -> np.ndarray:
    return as_minimization_front(frame["income_mcop"], frame["risk"])


def _latex(text: object) -> str:
    out = str(text)
    for old, new in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&")]:
        out = out.replace(old, new)
    return out


def _fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    if value != 0 and (abs(value) < 0.001 or abs(value) >= 10000):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def write_latex_table(path: Path, caption: str, label: str, columns: list[str], rows: list[list[object]], align: str) -> None:
    lines = [r"\begin{table}", rf"\tbl{{{caption}}}{{", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(columns) + r" \\", r"\midrule"]
    lines.extend(" & ".join(str(x) for x in row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}}", rf"\label{{{label}}}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_resized_latex_table(path: Path, caption: str, label: str, columns: list[str],
                              rows: list[list[object]], align: str) -> None:
    lines = [
        r"\begin{table}", rf"\tbl{{{caption}}}{{", r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(columns) + " \\\\", r"\midrule",
    ]
    lines.extend(" & ".join(str(x) for x in row) + " \\\\" for row in rows)
    lines.extend([
        r"\bottomrule", r"\end{tabular}", r"}%", r"}", rf"\label{{{label}}}", r"\end{table}", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def collect_fronts(runs: pd.DataFrame, fx: float) -> pd.DataFrame:
    rows = []
    for run in runs.itertuples(index=False):
        front = pd.read_csv(ROOT / run.front_path)
        for j, point in front.iterrows():
            if not np.isfinite(point.get("income_mcop", np.nan)) or not np.isfinite(point.get("risk", np.nan)):
                continue
            rows.append({
                "instance": run.instance, "method": run.method, "seed": int(run.seed),
                "source_run": getattr(run, "source_run", ""),
                "solution_id": int(point.get("solution_id", j)), "income_mcop": float(point.income_mcop),
                "risk": float(point.risk), "income_kusd": float(point.income_mcop) * 1000.0 / fx,
                "risk_kusd2": float(point.risk) * (1000.0 / fx) ** 2,
                "runtime_s": float(run.runtime_s), "front_path": run.front_path,
                "events_path": run.events_path,
            })
    return pd.DataFrame(rows)


def refinement_metrics(run) -> dict[str, float]:
    if run.method not in {"nsga2_qp", "hnsga2"}:
        return {"refinement_calls": 0.0, "useful_refinements": 0.0, "refinement_time_s": 0.0, "hit_rate": np.nan}
    directory = (ROOT / run.metadata_path).parent
    prefix = run.method
    history_path = directory / f"{prefix}_history.csv"
    if not history_path.exists():
        return {"refinement_calls": np.nan, "useful_refinements": np.nan, "refinement_time_s": np.nan, "hit_rate": np.nan}
    history = pd.read_csv(history_path)
    calls = float(history["refinement_calls"].sum())
    useful = float(history["useful_refinements"].sum())
    seconds = float(history["refinement_time_s"].sum())
    return {"refinement_calls": calls, "useful_refinements": useful, "refinement_time_s": seconds,
            "hit_rate": useful / calls if calls else np.nan}


def _summarize_metrics(seed: pd.DataFrame) -> pd.DataFrame:
    return seed.groupby(["reference_source", "instance", "method"], sort=False, observed=True).agg(
        runs=("seed", "size"), hv_median=("normalized_hypervolume", "median"),
        igd_plus_median=("igd_plus", "median"), epsilon_median=("additive_epsilon", "median"),
        points_median=("nondominated_points", "median"), runtime_median_s=("runtime_s", "median"),
        runtime_q1_s=("runtime_s", lambda x: x.quantile(0.25)),
        runtime_q3_s=("runtime_s", lambda x: x.quantile(0.75)),
        runtime_mean_s=("runtime_s", "mean"), refinement_calls_median=("refinement_calls", "median"),
        refinement_time_median_s=("refinement_time_s", "median"), hit_rate_median=("hit_rate", "median"),
    ).reset_index()


def compute_metrics(fronts: pd.DataFrame, runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary_records, sensitivity_records = [], []
    run_lookup = {(r.instance, r.method, int(r.seed)): r for r in runs.itertuples(index=False)}
    for instance, frame in fronts.groupby("instance", sort=False):
        exact = frame[frame.method.eq("exact")]
        pooled_raw = pooled_empirical_reference([
            as_min_front(run_frame)
            for _, run_frame in frame.groupby(["method", "seed"], sort=False)
        ])
        references = [("pooled empirical nondominated union", pooled_raw)]
        if len(exact) >= 2:
            references.append(("SCIP incumbent front", nondominated(as_min_front(exact))))
        for reference_source, reference_raw in references:
            ideal, span = reference_scaling(reference_raw)
            reference = normalize(reference_raw, ideal, span)
            destination = primary_records if reference_source.startswith("pooled") else sensitivity_records
            for (method, seed), approx_frame in frame.groupby(["method", "seed"], sort=False):
                approx_raw = nondominated(as_min_front(approx_frame))
                approx = normalize(approx_raw, ideal, span)
                run = run_lookup[(instance, method, int(seed))]
                destination.append({
                    "instance": instance, "method": method, "seed": int(seed),
                    "reference_source": reference_source,
                    "normalized_hypervolume": hypervolume_2d(approx),
                    "igd_plus": igd_plus(approx, reference),
                    "additive_epsilon": additive_epsilon(approx, reference),
                    "nondominated_points": len(approx_raw), "runtime_s": float(run.runtime_s),
                    **refinement_metrics(run),
                })
    primary_seed = pd.DataFrame(primary_records)
    sensitivity_seed = pd.DataFrame(sensitivity_records)
    return (
        primary_seed,
        _summarize_metrics(primary_seed),
        sensitivity_seed,
        _summarize_metrics(sensitivity_seed),
    )


def paired_inference(seed: pd.DataFrame, bootstrap_repetitions: int = 5000) -> pd.DataFrame:
    comparisons = [("nsga2", "nsga2_qp"), ("nsga2", "hnsga2"), ("nsga2_qp", "hnsga2")]
    metrics = ["normalized_hypervolume", "igd_plus", "additive_epsilon", "runtime_s"]
    rows = []
    for instance_index, (instance, frame) in enumerate(seed.groupby("instance", sort=False)):
        for metric_index, metric in enumerate(metrics):
            for comparison_index, (method_a, method_b) in enumerate(comparisons):
                left = frame[frame.method.eq(method_a)][["seed", metric]].rename(columns={metric: "a"})
                right = frame[frame.method.eq(method_b)][["seed", metric]].rename(columns={metric: "b"})
                paired = left.merge(right, on="seed").dropna()
                differences = paired.a.to_numpy(float) - paired.b.to_numpy(float)
                nonzero = differences[~np.isclose(differences, 0.0)]
                if len(nonzero):
                    ranks = rankdata(np.abs(nonzero))
                    positive, negative = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
                    effect = float((positive - negative) / (positive + negative))
                    test = wilcoxon(paired.a, paired.b, alternative="two-sided", zero_method="wilcox", method="auto")
                    statistic, p_value = float(test.statistic), float(test.pvalue)
                else:
                    statistic, p_value, effect = 0.0, 1.0, 0.0
                rng = np.random.default_rng(2026 + 100 * instance_index + 10 * metric_index + comparison_index)
                if len(differences):
                    draws = rng.choice(differences, size=(bootstrap_repetitions, len(differences)), replace=True)
                    medians = np.median(draws, axis=1)
                    ci_low, ci_high = np.quantile(medians, [0.025, 0.975])
                else:
                    ci_low = ci_high = np.nan
                rows.append({
                    "instance": instance, "metric": metric, "method_a": method_a, "method_b": method_b,
                    "n_pairs": len(differences), "median_a": np.median(paired.a) if len(paired) else np.nan,
                    "median_b": np.median(paired.b) if len(paired) else np.nan,
                    "median_paired_difference_a_minus_b": np.median(differences),
                    "bootstrap_ci_low": ci_low, "bootstrap_ci_high": ci_high,
                    "wilcoxon_w": statistic, "p_raw": p_value, "rank_biserial_a_minus_b": effect,
                })
    output = pd.DataFrame(rows)
    output["p_holm"] = np.nan
    for _, indices in output.groupby(["instance", "metric"], sort=False).groups.items():
        ordered = output.loc[list(indices), "p_raw"].sort_values()
        adjusted = np.maximum.accumulate([min(1.0, p * (len(ordered) - rank)) for rank, p in enumerate(ordered)])
        output.loc[ordered.index, "p_holm"] = adjusted
    output["interpretation"] = np.where(
        output.p_holm < 0.05,
        "paired difference detected after Holm correction",
        "no paired difference detected after Holm correction",
    )
    return output


def compare_baseline_initialization(fronts: pd.DataFrame, primary_seed: pd.DataFrame,
                                    original_run: str, instances: list[str]) -> pd.DataFrame:
    """Compare the archived and corrected baselines under the new pooled scaling."""
    rows = []
    for instance in instances:
        instance_fronts = fronts[fronts.instance.eq(instance)]
        reference_raw = pooled_empirical_reference([
            as_min_front(run_front)
            for _, run_front in instance_fronts.groupby(["method", "seed"], sort=False)
        ])
        ideal, span = reference_scaling(reference_raw)
        for seed in range(2026, 2036):
            path = ROOT / "results" / original_run / instance / "nsga2" / f"seed_{seed}" / "nsga2_front.csv"
            if not path.exists():
                continue
            archived = pd.read_csv(path)
            archived_raw = nondominated(as_min_front(archived))
            archived_hv = hypervolume_2d(normalize(archived_raw, ideal, span))
            corrected = primary_seed[
                primary_seed.instance.eq(instance) & primary_seed.method.eq("nsga2") & primary_seed.seed.eq(seed)
            ]
            if corrected.empty:
                continue
            rows.append({
                "instance": instance, "seed": seed,
                "archived_privileged_init_hv": archived_hv,
                "matched_init_hv": float(corrected.iloc[0].normalized_hypervolume),
                "paired_difference_matched_minus_archived":
                    float(corrected.iloc[0].normalized_hypervolume) - archived_hv,
            })
    return pd.DataFrame(rows)


def _axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.22, linewidth=0.65)
    ax.tick_params(direction="out", width=0.8, length=3.5)


def save_figure(fig, path: Path) -> None:
    for suffix, kwargs in [(".pdf", {}), (".svg", {}), (".png", {"dpi": 300})]:
        fig.savefig(
            path.with_suffix(suffix),
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def representative_seed(seed_metrics: pd.DataFrame, instance: str, method: str) -> int:
    z = seed_metrics[(seed_metrics.instance == instance) & (seed_metrics.method == method)]
    median = z.normalized_hypervolume.median()
    return int(z.loc[(z.normalized_hypervolume - median).abs().idxmin(), "seed"])


def plot_pareto(fronts: pd.DataFrame, seed_metrics: pd.DataFrame, instance: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.9, 4.3))
    for method in METHODS:
        z = fronts[(fronts.instance == instance) & (fronts.method == method)]
        if z.empty:
            continue
        seed = int(z.seed.iloc[0]) if method == "exact" else representative_seed(seed_metrics, instance, method)
        z = z[z.seed == seed].sort_values("income_kusd")
        ax.plot(z.income_kusd, z.risk_kusd2, color=COLORS[method], marker=MARKERS[method],
                markersize=3.5, linewidth=1.1, label=LABELS[method])
    ax.set_xlabel("Expected gross revenue (thousand USD)")
    ax.set_ylabel(r"Correlated price risk ((thousand USD)$^2$)")
    _axes(ax)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.tight_layout()
    save_figure(fig, out / f"pareto_{instance}.pdf")


def plot_pareto_grid(fronts: pd.DataFrame, seed_metrics: pd.DataFrame, instances: list[str], out: Path) -> None:
    """Assemble publication-size Pareto panels with one shared legend and shared labels."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    legend_handles: dict[str, object] = {}
    for ax, instance in zip(axes.flat, instances):
        for method in METHODS:
            z = fronts[(fronts.instance == instance) & (fronts.method == method)]
            if z.empty:
                continue
            seed = int(z.seed.iloc[0]) if method == "exact" else representative_seed(seed_metrics, instance, method)
            z = z[z.seed == seed].sort_values("income_kusd")
            markevery = 1 if method == "exact" else max(1, len(z) // 18)
            line, = ax.plot(
                z.income_kusd,
                z.risk_kusd2,
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=3.8,
                linewidth=1.45,
                markevery=markevery,
                label=LABELS[method],
            )
            legend_handles.setdefault(method, line)
        ax.set_title(instance.replace("_", " "), fontweight="bold", pad=5)
        _axes(ax)
    fig.supxlabel("Expected gross revenue (thousand USD)", y=0.035)
    fig.supylabel(r"Correlated price risk ((thousand USD)$^2$)", x=0.025)
    fig.legend(
        [legend_handles[m] for m in METHODS if m in legend_handles],
        [LABELS[m] for m in METHODS if m in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.12, top=0.86, wspace=0.24, hspace=0.28)
    save_figure(fig, out / "pareto_fronts.pdf")


def plot_runtime(summary: pd.DataFrame, catalog: pd.DataFrame, out: Path) -> None:
    z = summary.merge(catalog[["instance", "lots"]], on="instance")
    instances = catalog.instance.tolist()
    positions = {instance: i for i, instance in enumerate(instances)}
    fig, ax = plt.subplots(figsize=(7.1, 3.7))
    for method in METHODS:
        m = z[z.method == method].copy()
        m["position"] = m.instance.map(positions)
        m = m.sort_values("position")
        ax.plot(
            m.position,
            m.runtime_median_s,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=6,
            linewidth=1.6,
            label=LABELS[method],
        )
    ax.set_yscale("log")
    ax.set_xticks(range(len(instances)))
    ax.set_xticklabels([
        f"{instance.split('_')[0]}\n{int(catalog.loc[catalog.instance.eq(instance), 'lots'].iloc[0])} lots"
        for instance in instances
    ])
    ax.set_xlabel("Benchmark instance")
    ax.set_ylabel("Runtime (s, logarithmic scale)")
    _axes(ax)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout(pad=0.8)
    save_figure(fig, out / "runtime_scaling.pdf")


def plot_quality_time(summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), sharex=False, sharey=False)
    legend_handles: dict[str, object] = {}
    for ax, (instance, z) in zip(axes.flat, summary.groupby("instance", sort=False)):
        for row in z.itertuples(index=False):
            point = ax.scatter(
                row.runtime_median_s,
                row.hv_median,
                color=COLORS[row.method],
                marker=MARKERS[row.method],
                s=48,
                linewidths=0.7,
                edgecolors="white" if row.method != "exact" else COLORS[row.method],
                zorder=3,
            )
            legend_handles.setdefault(row.method, point)
        ax.set_xscale("log")
        ax.set_title(instance.replace("_", " "), fontweight="bold", pad=5)
        _axes(ax)
    fig.supxlabel("Runtime (s, logarithmic scale)", y=0.035)
    fig.supylabel("Normalized hypervolume", x=0.025)
    fig.legend(
        [legend_handles[m] for m in METHODS if m in legend_handles],
        [LABELS[m] for m in METHODS if m in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.12, top=0.84, wspace=0.24, hspace=0.30)
    save_figure(fig, out / "quality_vs_time.pdf")


def plot_refinement(seed_metrics: pd.DataFrame, out: Path) -> None:
    z = seed_metrics[seed_metrics.method.isin(["nsga2_qp", "hnsga2"])]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), sharey=True)
    legend_handles: dict[str, object] = {}
    for ax, (instance, instance_data) in zip(axes.flat, z.groupby("instance", sort=False)):
        for method in ["nsga2_qp", "hnsga2"]:
            m = instance_data[instance_data.method == method]
            points = ax.scatter(
                m.refinement_time_s,
                m.normalized_hypervolume,
                color=COLORS[method],
                marker=MARKERS[method],
                s=42,
                alpha=0.78,
                linewidths=0.7,
                edgecolors=COLORS[method],
                label=LABELS[method],
            )
            legend_handles.setdefault(method, points)
        ax.set_xscale("log")
        ax.set_title(instance.replace("_", " "), fontweight="bold", pad=5)
        _axes(ax)
    fig.supxlabel("Cumulative refinement time (s, logarithmic scale)", y=0.035)
    fig.supylabel("Normalized hypervolume", x=0.025)
    fig.legend(
        [legend_handles[m] for m in ["nsga2_qp", "hnsga2"]],
        [LABELS[m] for m in ["nsga2_qp", "hnsga2"]],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.12, top=0.86, wspace=0.20, hspace=0.30)
    save_figure(fig, out / "surrogate_refinement_efficiency.pdf")


def make_schedule(fronts: pd.DataFrame, seed_metrics: pd.DataFrame, names: dict, fx: float,
                  catalog: pd.DataFrame, out: Path) -> dict:
    instance, method = "I02_small", "hnsga2"
    seed = representative_seed(seed_metrics, instance, method)
    z = fronts[(fronts.instance == instance) & (fronts.method == method) & (fronts.seed == seed)].copy()
    income = z.income_kusd.to_numpy(); risk = z.risk_kusd2.to_numpy()
    inc_norm = (income - income.min()) / max(income.max() - income.min(), 1e-12)
    risk_norm = (risk - risk.min()) / max(risk.max() - risk.min(), 1e-12)
    z["knee_distance"] = np.sqrt((1 - inc_norm) ** 2 + risk_norm ** 2)
    choice = z.loc[z.knee_distance.idxmin()]
    events = pd.read_csv(ROOT / choice.events_path)
    events = events[events.solution_id == int(choice.solution_id)].copy()
    events["crop_en"] = events.crop_name.map(names).fillna(events.crop_name)
    events["area_ha"] = events.area_m2 / 10000.0
    events["price_usd_kg"] = events.price_cop_kg / fx
    schedule_export = events[
        ["solution_id", "lot", "farm", "municipality", "crop", "crop_en", "sow", "harvest",
         "area_m2", "area_ha", "production_kg", "price_usd_kg"]
    ].rename(columns={"crop_en": "crop_name"})
    schedule_export.to_csv(out.parent / "representative_schedule.csv", index=False)

    row = catalog[catalog.instance == instance].iloc[0]
    src_path = str(ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from agriopt.instances import _lot_allocation
    lot_areas, _ = _lot_allocation(
        int(row.farms), int(row.lots), float(row.total_area_ha), str(row.heterogeneity),
        np.random.default_rng(int(row.seed)),
    )
    horizon = int(row.horizon_weeks)
    used_area = np.zeros((len(lot_areas), horizon), dtype=float)
    for event in events.itertuples(index=False):
        used_area[int(event.lot), int(event.sow):int(event.harvest)] += float(event.area_m2)
    used_area = np.minimum(used_area, lot_areas[:, None])
    occupied = used_area > 1e-10
    occupied_weeks = occupied.sum(axis=1)
    lot_occupancy = occupied_weeks / horizon
    lot_area_utilization = used_area.sum(axis=1) / (horizon * lot_areas)
    space_time_utilization = float(used_area.sum() / (horizon * lot_areas.sum()))
    mean_lot_occupancy = float(lot_occupancy.mean())
    mean_idle_weeks = float((horizon - occupied_weeks).mean())
    pd.DataFrame({
        "lot": np.arange(len(lot_areas), dtype=int),
        "lot_area_m2": lot_areas,
        "occupied_weeks": occupied_weeks,
        "idle_weeks": horizon - occupied_weeks,
        "temporal_occupancy": lot_occupancy,
        "area_adjusted_space_time_utilization": lot_area_utilization,
    }).to_csv(out.parent / "representative_schedule_utilization.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    crops = list(dict.fromkeys(events.crop_en.tolist()))
    schedule_palette = ["#4472C4", "#70AD47", "#ED7D31", "#8064A2", "#5B9BD5", "#A5A5A5"]
    colors = {c: schedule_palette[i % len(schedule_palette)] for i, c in enumerate(crops)}
    for row in events.sort_values(["lot", "sow"]).itertuples(index=False):
        color = colors[row.crop_en]
        rgb = matplotlib.colors.to_rgb(color)
        luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        text_color = "#111111" if luminance > 0.62 else "white"
        ax.barh(row.lot, row.harvest - row.sow, left=row.sow, height=0.66, color=color,
                edgecolor="white", linewidth=0.7)
        ax.text((row.sow + row.harvest) / 2, row.lot, f"{row.crop_en}\n{row.area_ha:.2f} ha",
                ha="center", va="center", fontsize=8, color=text_color, weight="bold", clip_on=True)
    lots = sorted(events.lot.unique())
    ax.set_yticks(lots)
    ax.set_yticklabels([f"Lot {int(x)+1} | occupied {100 * lot_occupancy[int(x)]:.1f}%" for x in lots])
    ax.invert_yaxis(); ax.set_xlabel("Planning week"); ax.set_ylabel("Lot"); _axes(ax)
    ax.set_xlim(0, horizon)
    ax.text(1.0, 1.025,
            f"Area-adjusted space-time utilization: {100 * space_time_utilization:.1f}%  |  "
            f"Mean lot occupancy: {100 * mean_lot_occupancy:.1f}%  |  "
            f"Mean idle time: {mean_idle_weeks:.1f} weeks/lot",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.3, color="#333333")
    fig.tight_layout(pad=0.8)
    save_figure(fig, out / "representative_schedule.pdf")
    return {"instance": instance, "method": method, "seed": seed, "solution_id": int(choice.solution_id),
            "income_kusd": float(choice.income_kusd), "risk_kusd2": float(choice.risk_kusd2),
            "events": int(len(events)), "space_time_utilization": space_time_utilization,
            "mean_lot_occupancy": mean_lot_occupancy, "mean_idle_weeks": mean_idle_weeks,
            "lot_temporal_occupancy": [float(value) for value in lot_occupancy],
            "lot_area_adjusted_utilization": [float(value) for value in lot_area_utilization]}


def schedule_macros(schedule: dict) -> list[str]:
    return [
        rf"\newcommand{{\RepresentativeIncomeKUSD}}{{{schedule['income_kusd']:.2f}}}",
        rf"\newcommand{{\RepresentativeRiskKUSDsq}}{{{schedule['risk_kusd2']:.2f}}}",
        rf"\newcommand{{\RepresentativeLandUtilizationPct}}{{{100 * schedule['space_time_utilization']:.1f}}}",
        rf"\newcommand{{\RepresentativeMeanLotOccupancyPct}}{{{100 * schedule['mean_lot_occupancy']:.1f}}}",
        rf"\newcommand{{\RepresentativeMeanIdleWeeks}}{{{schedule['mean_idle_weeks']:.1f}}}",
        rf"\newcommand{{\RepresentativeEventCount}}{{{schedule['events']}}}",
    ]


def build_tables(snapshot: dict, runs: pd.DataFrame, seed: pd.DataFrame, summary: pd.DataFrame,
                 sensitivity_summary: pd.DataFrame, inference: pd.DataFrame,
                 catalog: pd.DataFrame, schedule: dict, table_dir: Path, artifact_dir: Path) -> None:
    included = snapshot["included_complete_instances"]
    cat = catalog[catalog.instance.isin(included)]
    write_latex_table(table_dir / "benchmark_instances.tex",
        "Data-informed synthetic benchmark instances included in the controlled experiment.", "tab:instances",
        ["Instance", "Farms", "Lots", "Area (ha)", "Crops", "Weeks"],
        [[_latex(r.instance), r.farms, r.lots, _fmt(r.total_area_ha, 1), r.products, r.horizon_weeks] for r in cat.itertuples()],
        "lrrrrr")

    forecast = pd.read_csv(ROOT / "data" / "interim" / "forecast_model_selection.csv")
    forecast.to_csv(table_dir.parent / "forecast_model_selection.csv", index=False)
    forecast_summary = forecast.groupby("selected_model", sort=True).agg(
        products=("product", "size"), median_smape=("cv_smape", "median"),
        median_rmse=("cv_rmse", "median"),
    ).reset_index()
    rows = [[_latex(r.selected_model), int(r.products), _fmt(r.median_smape, 2), _fmt(r.median_rmse, 2)]
            for r in forecast_summary.itertuples()]
    write_latex_table(table_dir / "forecast_validation.tex",
        "Selected forecasting models and terminal 52-week holdout errors aggregated across products; selection minimizes product-level sMAPE.",
        "tab:forecast", ["Selected model", "Products", "Median sMAPE (\\%)", "Median RMSE (COP/kg)"],
        rows, "lrrr")

    exact_rows, status_records = [], []
    for run in runs[runs.method == "exact"].itertuples(index=False):
        front = pd.read_csv(ROOT / run.front_path)
        counts = front.status.value_counts()
        finite_gap = front.mip_gap.replace([np.inf, -np.inf], np.nan).dropna()
        finite_gap = finite_gap[finite_gap < 1e10]  # SCIP uses a large sentinel when no finite dual bound exists.
        exact_rows.append([_latex(run.instance), _fmt(run.runtime_s, 2), int(len(front)), int(counts.get("optimal", 0)),
                           int(counts.get("gaplimit", 0)), int(counts.get("timelimit", 0)),
                           _fmt(finite_gap.max() if len(finite_gap) else np.nan, 3)])
        for row in front.itertuples(index=False):
            status_records.append({"instance": run.instance, "solution_id": row.solution_id,
                                   "status": row.status, "mip_gap": row.mip_gap, "solve_time_s": row.solve_time_s})
    pd.DataFrame(status_records).to_csv(artifact_dir / "exact_point_status.csv", index=False)
    write_latex_table(table_dir / "exact_performance.tex", "SCIP epsilon-constraint run and point status.",
        "tab:exact", ["Instance", "Runtime (s)", "Points", "Optimal", "Gap limit", "Time limit", "Max gap"],
        exact_rows, "lrrrrrr")

    table_labels = {"exact": "SCIP", "nsga2": "NSGA-II", "nsga2_qp": "NSGA-II + full", "hnsga2": "H-NSGA-II"}
    rows = []
    for r in summary.itertuples(index=False):
        rows.append([_latex(r.instance), _latex(table_labels[r.method]), _fmt(r.hv_median), _fmt(r.igd_plus_median),
                     _fmt(r.epsilon_median), _fmt(r.points_median, 1), _fmt(r.runtime_median_s, 2)])
    write_latex_table(table_dir / "method_comparison.tex",
        "Median Pareto quality under the pooled empirical reference and median runtime; stochastic methods use ten matched seeds.", "tab:comparison",
        ["Instance", "Method", "HV", "IGD$^+$", "$\\epsilon^+$", "ND points", "Runtime (s)"], rows, "llrrrrr")

    rows = []
    for r in summary[summary.method.isin(["nsga2_qp", "hnsga2"])].itertuples(index=False):
        rows.append([_latex(r.instance), _latex(table_labels[r.method]), _fmt(r.refinement_calls_median, 0),
                     _fmt(r.refinement_time_median_s, 2), _fmt(100 * r.hit_rate_median, 1), _fmt(r.hv_median)])
    write_latex_table(table_dir / "refinement_ablation.tex", "Full versus surrogate-guided SLSQP refinement.",
        "tab:refinement", ["Instance", "Method", "Calls", "Refinement time (s)", "Useful (\\%)", "HV"], rows, "llrrrr")

    pooled_stochastic = summary[summary.method.ne("exact")].copy()
    scip_stochastic = sensitivity_summary[sensitivity_summary.method.ne("exact")].copy()
    sensitivity = pooled_stochastic.merge(
        scip_stochastic,
        on=["instance", "method"],
        suffixes=("_pooled", "_scip"),
    )
    sensitivity["rank_pooled"] = sensitivity.groupby("instance").hv_median_pooled.rank(ascending=False, method="min")
    sensitivity["rank_scip"] = sensitivity.groupby("instance").hv_median_scip.rank(ascending=False, method="min")
    sensitivity.to_csv(artifact_dir / "reference_sensitivity.csv", index=False)
    sensitivity.to_csv(table_dir / "reference_sensitivity.csv", index=False)
    rows = [[_latex(r.instance), _latex(table_labels[r.method]), _fmt(r.hv_median_pooled),
             int(r.rank_pooled), _fmt(r.hv_median_scip), int(r.rank_scip)] for r in sensitivity.itertuples()]
    write_latex_table(table_dir / "reference_sensitivity.tex",
        "Sensitivity of stochastic-method median hypervolume and rank to the pooled empirical and SCIP-only references.",
        "tab:reference-sensitivity", ["Instance", "Method", "HV pooled", "Rank", "HV SCIP", "Rank"], rows, "llrrrr")

    inference.to_csv(artifact_dir / "stochastic_pairwise_tests.csv", index=False)
    inference.to_csv(table_dir / "stochastic_pairwise_tests.csv", index=False)
    hv_tests = inference[inference.metric.eq("normalized_hypervolume")]
    rows = [[_latex(r.instance), "HV", f"{_latex(table_labels[r.method_a])} $-$ {_latex(table_labels[r.method_b])}",
             int(r.n_pairs), _fmt(r.median_a), _fmt(r.median_b), _fmt(r.median_paired_difference_a_minus_b),
             _fmt(r.wilcoxon_w, 1), _fmt(r.p_raw), _fmt(r.p_holm), _fmt(r.rank_biserial_a_minus_b)]
            for r in hv_tests.itertuples()]
    write_resized_latex_table(table_dir / "stochastic_pairwise_tests.tex",
        "Paired hypervolume comparisons by matched seed; differences and rank-biserial effects use method A minus method B, with Holm-adjusted two-sided Wilcoxon probabilities.",
        "tab:paired-hv", ["Instance", "Metric", "Comparison", "$n$", "Median A", "Median B", "Median diff.",
                          "$W$", "$p$", "$p_{Holm}$", "$r_{rb}$"],
        rows, "lllrrrrrrrr")

    macro = [
        rf"\newcommand{{\NIncludedInstances}}{{{len(included)}}}",
        rf"\newcommand{{\LargestIncludedInstance}}{{{_latex(cat.iloc[-1].instance)}}}",
        rf"\newcommand{{\LargestIncludedLots}}{{{int(cat.iloc[-1].lots)}}}",
        *schedule_macros(schedule),
    ]
    (table_dir.parent / "results_macros.tex").write_text("\n".join(macro) + "\n", encoding="utf-8")


def compile_paper() -> None:
    for pass_no in (1, 2):
        result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                                cwd=ROOT / "paper", text=True, capture_output=True)
        if result.returncode:
            tail = "\n".join((result.stdout + "\n" + result.stderr).splitlines()[-40:])
            raise RuntimeError(f"LaTeX pass {pass_no} failed:\n{tail}")


def refresh_schedule_only(run: str, compile_tex: bool = True) -> None:
    artifact = ROOT / "artifacts" / run
    snapshot_path = artifact / "paper_snapshot.json"
    runs_path = artifact / "paper_snapshot_runs.csv"
    if not snapshot_path.exists() or not runs_path.exists():
        raise FileNotFoundError(f"Frozen publication snapshot is missing for run {run}")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    runs = pd.read_csv(runs_path)
    cfg = yaml.safe_load((ROOT / snapshot["configuration_path"]).read_text(encoding="utf-8"))
    fx = float(cfg["reporting"]["fx"]["cop_per_usd"])
    names = yaml.safe_load((ROOT / "configs" / "product_names_en.yaml").read_text(encoding="utf-8"))
    catalog = pd.read_csv(ROOT / "data" / "seed" / "benchmark_instance_catalog.csv")
    fronts = collect_fronts(runs, fx)
    seed_metrics, _, _, _ = compute_metrics(fronts, runs)
    generated = ROOT / "paper" / "generated"
    figures = generated / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    schedule = make_schedule(fronts, seed_metrics, names, fx, catalog, figures)
    (artifact / "representative_schedule.json").write_text(json.dumps(schedule, indent=2), encoding="utf-8")

    macro_path = generated / "results_macros.tex"
    existing = macro_path.read_text(encoding="utf-8").splitlines() if macro_path.exists() else []
    schedule_names = {
        "RepresentativeIncomeKUSD", "RepresentativeRiskKUSDsq", "RepresentativeLandUtilizationPct",
        "RepresentativeMeanLotOccupancyPct", "RepresentativeMeanIdleWeeks", "RepresentativeEventCount",
    }
    retained = [line for line in existing if not any(f"{{\\{name}}}" in line for name in schedule_names)]
    macro_path.write_text("\n".join(retained + schedule_macros(schedule)) + "\n", encoding="utf-8")
    if compile_tex:
        compile_paper()
    audit_path, audit_status, _ = audit_manuscript(ROOT)
    print(f"Schedule regenerated from frozen snapshot: {figures / 'representative_schedule.pdf'}")
    print(f"Utilization audit: {generated / 'representative_schedule_utilization.csv'}")
    print(f"Narrative-quality audit: {audit_path}")
    print(audit_status)


def overleaf_bundle() -> Path:
    paper = ROOT / "paper"
    destination = paper / "overleaf_ready"
    destination.mkdir(exist_ok=True)
    for name in ["main.tex", "interact.cls", "natbib.sty", "subfig.sty", "rotating.sty", "epsfig.sty", "booktabs.sty"]:
        shutil.copy2(paper / name, destination / name)
    shutil.copytree(paper / "generated", destination / "generated", dirs_exist_ok=True)
    if (paper / "main.pdf").exists():
        try:
            shutil.copy2(paper / "main.pdf", destination / "main.pdf")
        except PermissionError:
            warnings.warn("overleaf_ready/main.pdf is open in another program and was not replaced")
    return destination


def refresh(run: str, baseline_run: str | None = None, snapshot_id: str | None = None,
            compile_tex: bool = True, bundle_output: bool = True) -> None:
    snapshot_module = _load_snapshot_module()
    json_path, csv_path = snapshot_module.write_snapshot(run, baseline_run, snapshot_id)
    snapshot = json.loads(json_path.read_text(encoding="utf-8"))
    runs = pd.read_csv(csv_path)
    cfg = yaml.safe_load((ROOT / snapshot["configuration_path"]).read_text(encoding="utf-8"))
    fx = float(cfg["reporting"]["fx"]["cop_per_usd"])
    names = yaml.safe_load((ROOT / "configs" / "product_names_en.yaml").read_text(encoding="utf-8"))
    catalog = pd.read_csv(ROOT / "data" / "seed" / "benchmark_instance_catalog.csv")
    fronts = collect_fronts(runs, fx)
    seed_metrics, summary, sensitivity_seed, sensitivity_summary = compute_metrics(fronts, runs)
    inference = paired_inference(seed_metrics)
    initialization_comparison = compare_baseline_initialization(
        fronts, seed_metrics, run, snapshot["included_complete_instances"]
    )

    artifact = ROOT / "artifacts" / (snapshot_id or run)
    artifact.mkdir(parents=True, exist_ok=True)
    seed_metrics.to_csv(artifact / "seed_level_metrics.csv", index=False)
    summary.to_csv(artifact / "method_summary.csv", index=False)
    sensitivity_seed.to_csv(artifact / "seed_level_metrics_scip_reference.csv", index=False)
    sensitivity_summary.to_csv(artifact / "method_summary_scip_reference.csv", index=False)
    initialization_comparison.to_csv(artifact / "baseline_initialization_comparison.csv", index=False)
    fronts.to_csv(artifact / "publication_fronts_usd.csv", index=False)

    generated = ROOT / "paper" / "generated"
    figures, tables = generated / "figures", generated / "tables"
    figures.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    initialization_comparison.to_csv(tables / "baseline_initialization_comparison.csv", index=False)
    result_figure_names = {
        "runtime_scaling.pdf", "quality_vs_time.pdf", "surrogate_refinement_efficiency.pdf",
        "representative_schedule.pdf", "pareto_fronts.pdf",
        *{f"pareto_{instance}.pdf" for instance in snapshot["included_complete_instances"]},
    }
    for old in [figures / name for name in result_figure_names] + list(tables.glob("*.tex")):
        if not old.exists():
            continue
        old.unlink()
    for instance in snapshot["included_complete_instances"]:
        plot_pareto(fronts, seed_metrics, instance, figures)
    plot_pareto_grid(fronts, seed_metrics, snapshot["included_complete_instances"], figures)
    cat_included = catalog[catalog.instance.isin(snapshot["included_complete_instances"])]
    plot_runtime(summary, cat_included, figures)
    plot_quality_time(summary, figures)
    plot_refinement(seed_metrics, figures)
    schedule = make_schedule(fronts, seed_metrics, names, fx, catalog, figures)
    (artifact / "representative_schedule.json").write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    build_tables(snapshot, runs, seed_metrics, summary, sensitivity_summary, inference,
                 catalog, schedule, tables, artifact)
    (generated / "RUN_ID.txt").write_text((snapshot_id or run) + "\n", encoding="utf-8")
    (generated / "method_run_sources.json").write_text(
        json.dumps(snapshot["method_run_sources"], indent=2) + "\n", encoding="utf-8")
    shutil.copy2(json_path, generated / "submission_snapshot.json")

    if compile_tex:
        compile_paper()
    audit_path, audit_status, _ = audit_manuscript(ROOT)
    bundle = overleaf_bundle() if bundle_output else None
    print(f"Included: {', '.join(snapshot['included_complete_instances'])}")
    print(f"Excluded: {', '.join(snapshot['excluded_incomplete_instances'])}")
    print(f"Generated: {generated}")
    print(f"Narrative-quality audit: {audit_path}")
    print(audit_status)
    if bundle is not None:
        print(f"Overleaf bundle: {bundle}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="publication_slsqp_test")
    parser.add_argument("--baseline-run")
    parser.add_argument("--snapshot-id", default="submission_fairness_revision")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-bundle", action="store_true")
    parser.add_argument("--schedule-only", action="store_true",
                        help="Regenerate the representative schedule without rewriting the publication snapshot")
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if args.schedule_only:
            refresh_schedule_only(args.run, compile_tex=not args.no_compile)
        else:
            refresh(args.run, args.baseline_run, args.snapshot_id,
                    compile_tex=not args.no_compile, bundle_output=not args.no_bundle)


if __name__ == "__main__":
    main()
