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


ROOT = Path(__file__).resolve().parents[1]
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


def nondominated(points: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    keep = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        if not keep[i]:
            continue
        dominated = np.all(points <= points[i] + tol, axis=1) & np.any(points < points[i] - tol, axis=1)
        dominated[i] = False
        if dominated.any():
            keep[i] = False
    return points[keep]


def as_min_front(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([-frame["income_mcop"].to_numpy(float), frame["risk"].to_numpy(float)])


def normalize(points: np.ndarray, ideal: np.ndarray, span: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=float) - ideal) / span


def hypervolume_2d(points: np.ndarray, reference=(1.1, 1.1)) -> float:
    rp = np.asarray(reference, dtype=float)
    p = nondominated(points)
    p = p[np.all(p <= rp, axis=1)]
    if not len(p):
        return 0.0
    p = p[np.argsort(p[:, 0])]
    hv, current_y = 0.0, rp[1]
    for x, y in p:
        if y < current_y:
            hv += max(0.0, rp[0] - x) * max(0.0, current_y - y)
            current_y = y
    return float(hv)


def igd_plus(approx: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean([np.min(np.linalg.norm(np.maximum(approx - r, 0), axis=1)) for r in ref]))


def additive_epsilon(approx: np.ndarray, ref: np.ndarray) -> float:
    return float(max(min(np.max(a - r) for a in approx) for r in ref))


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


def collect_fronts(runs: pd.DataFrame, fx: float) -> pd.DataFrame:
    rows = []
    for run in runs.itertuples(index=False):
        front = pd.read_csv(ROOT / run.front_path)
        for j, point in front.iterrows():
            if not np.isfinite(point.get("income_mcop", np.nan)) or not np.isfinite(point.get("risk", np.nan)):
                continue
            rows.append({
                "instance": run.instance, "method": run.method, "seed": int(run.seed),
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


def compute_metrics(fronts: pd.DataFrame, runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    run_lookup = {(r.instance, r.method, int(r.seed)): r for r in runs.itertuples(index=False)}
    for instance, frame in fronts.groupby("instance", sort=False):
        exact = frame[frame.method.eq("exact")]
        source = exact if len(exact) >= 2 else frame
        reference_raw = nondominated(as_min_front(source))
        ideal = np.min(reference_raw, axis=0)
        span = np.maximum(np.max(reference_raw, axis=0) - ideal, 1e-12)
        reference = normalize(reference_raw, ideal, span)
        reference_source = "SCIP incumbent front" if len(exact) >= 2 else "pooled empirical front"
        for (method, seed), approx_frame in frame.groupby(["method", "seed"], sort=False):
            approx_raw = nondominated(as_min_front(approx_frame))
            approx = normalize(approx_raw, ideal, span)
            run = run_lookup[(instance, method, int(seed))]
            records.append({
                "instance": instance, "method": method, "seed": int(seed), "reference_source": reference_source,
                "normalized_hypervolume": hypervolume_2d(approx), "igd_plus": igd_plus(approx, reference),
                "additive_epsilon": additive_epsilon(approx, reference), "nondominated_points": len(approx_raw),
                "runtime_s": float(run.runtime_s), **refinement_metrics(run),
            })
    seed = pd.DataFrame(records)
    summary = seed.groupby(["instance", "method"], sort=False, observed=True).agg(
        runs=("seed", "size"), hv_median=("normalized_hypervolume", "median"),
        igd_plus_median=("igd_plus", "median"), epsilon_median=("additive_epsilon", "median"),
        points_median=("nondominated_points", "median"), runtime_median_s=("runtime_s", "median"),
        runtime_mean_s=("runtime_s", "mean"), refinement_calls_median=("refinement_calls", "median"),
        refinement_time_median_s=("refinement_time_s", "median"), hit_rate_median=("hit_rate", "median"),
    ).reset_index()
    return seed, summary


def _axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
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
    ax.set_xlabel("Expected income (thousand USD)")
    ax.set_ylabel(r"Correlated price risk ((thousand USD)$^2$)")
    _axes(ax)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.tight_layout()
    save_figure(fig, out / f"pareto_{instance}.pdf")


def plot_runtime(summary: pd.DataFrame, catalog: pd.DataFrame, out: Path) -> None:
    z = summary.merge(catalog[["instance", "lots"]], on="instance")
    fig, ax = plt.subplots(figsize=(6.9, 4.3))
    for method in METHODS:
        m = z[z.method == method].sort_values("lots")
        ax.plot(m.lots, m.runtime_median_s, color=COLORS[method], marker=MARKERS[method], label=LABELS[method])
    ax.set_yscale("log")
    ax.set_xlabel("Lots")
    ax.set_ylabel("Runtime (s, logarithmic scale)")
    _axes(ax); ax.legend(frameon=False, fontsize=7.5, ncol=2); fig.tight_layout()
    save_figure(fig, out / "runtime_scaling.pdf")


def plot_quality_time(summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.0), sharex=False, sharey=False)
    for ax, (instance, z) in zip(axes.flat, summary.groupby("instance", sort=False)):
        for row in z.itertuples(index=False):
            ax.scatter(row.runtime_median_s, row.hv_median, color=COLORS[row.method], marker=MARKERS[row.method], s=32)
        ax.set_xscale("log"); ax.set_title(instance.replace("_", " "), fontsize=9)
        ax.set_xlabel("Runtime (s)"); ax.set_ylabel("Normalized hypervolume"); _axes(ax)
    fig.tight_layout()
    save_figure(fig, out / "quality_vs_time.pdf")


def plot_refinement(seed_metrics: pd.DataFrame, out: Path) -> None:
    z = seed_metrics[seed_metrics.method.isin(["nsga2_qp", "hnsga2"])]
    fig, ax = plt.subplots(figsize=(6.9, 4.3))
    for method in ["nsga2_qp", "hnsga2"]:
        m = z[z.method == method]
        ax.scatter(m.refinement_time_s, m.normalized_hypervolume, color=COLORS[method], marker=MARKERS[method],
                   s=24, alpha=0.75, label=LABELS[method])
    ax.set_xscale("log")
    ax.set_xlabel("Cumulative refinement time (s, logarithmic scale)")
    ax.set_ylabel("Normalized hypervolume")
    _axes(ax); ax.legend(frameon=False, fontsize=8); fig.tight_layout()
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

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    crops = list(dict.fromkeys(events.crop_en.tolist()))
    palette = plt.get_cmap("tab20")
    colors = {c: palette(i % 20) for i, c in enumerate(crops)}
    for row in events.sort_values(["lot", "sow"]).itertuples(index=False):
        ax.barh(row.lot, row.harvest - row.sow, left=row.sow, height=0.62, color=colors[row.crop_en],
                edgecolor="white", linewidth=0.4)
        ax.text((row.sow + row.harvest) / 2, row.lot, f"{row.crop_en}\n{row.area_ha:.2f} ha",
                ha="center", va="center", fontsize=6, clip_on=True)
    lots = sorted(events.lot.unique())
    ax.set_yticks(lots)
    ax.set_yticklabels([f"Lot {int(x)+1} | occupied {100 * lot_occupancy[int(x)]:.1f}%" for x in lots])
    ax.invert_yaxis(); ax.set_xlabel("Planning week"); ax.set_ylabel("Lot"); _axes(ax)
    ax.text(1.0, 1.025,
            f"Area-adjusted space-time utilization: {100 * space_time_utilization:.1f}%  |  "
            f"Mean lot occupancy: {100 * mean_lot_occupancy:.1f}%  |  "
            f"Mean idle time: {mean_idle_weeks:.1f} weeks/lot",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2, color="#333333")
    fig.tight_layout()
    for extension in ("pdf", "svg", "png"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        fig.savefig(out / f"representative_schedule.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)
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
                 catalog: pd.DataFrame, schedule: dict, table_dir: Path, artifact_dir: Path) -> None:
    included = snapshot["included_complete_instances"]
    cat = catalog[catalog.instance.isin(included)]
    write_latex_table(table_dir / "benchmark_instances.tex",
        "Publication-complete benchmark instances.", "tab:instances",
        ["Instance", "Farms", "Lots", "Area (ha)", "Crops", "Weeks"],
        [[_latex(r.instance), r.farms, r.lots, _fmt(r.total_area_ha, 1), r.products, r.horizon_weeks] for r in cat.itertuples()],
        "lrrrrr")

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
        "Median Pareto quality and runtime; stochastic methods use ten seeds.", "tab:comparison",
        ["Instance", "Method", "HV", "IGD$^+$", "$\\epsilon^+$", "ND points", "Runtime (s)"], rows, "llrrrrr")

    rows = []
    for r in summary[summary.method.isin(["nsga2_qp", "hnsga2"])].itertuples(index=False):
        rows.append([_latex(r.instance), _latex(table_labels[r.method]), _fmt(r.refinement_calls_median, 0),
                     _fmt(r.refinement_time_median_s, 2), _fmt(100 * r.hit_rate_median, 1), _fmt(r.hv_median)])
    write_latex_table(table_dir / "refinement_ablation.tex", "Full versus surrogate-guided SLSQP refinement.",
        "tab:refinement", ["Instance", "Method", "Calls", "Refinement time (s)", "Useful (\\%)", "HV"], rows, "llrrrr")

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
    seed_metrics, _ = compute_metrics(fronts, runs)
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
    print(f"Schedule regenerated from frozen snapshot: {figures / 'representative_schedule.pdf'}")
    print(f"Utilization audit: {generated / 'representative_schedule_utilization.csv'}")


def overleaf_bundle() -> Path:
    paper = ROOT / "paper"
    destination = paper / "overleaf_ready"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir()
    for name in ["main.tex", "interact.cls", "natbib.sty", "subfig.sty", "rotating.sty", "epsfig.sty", "booktabs.sty"]:
        shutil.copy2(paper / name, destination / name)
    shutil.copytree(paper / "generated", destination / "generated")
    if (paper / "main.pdf").exists():
        shutil.copy2(paper / "main.pdf", destination / "main.pdf")
    return destination


def refresh(run: str, compile_tex: bool = True) -> None:
    snapshot_module = _load_snapshot_module()
    json_path, csv_path = snapshot_module.write_snapshot(run)
    snapshot = json.loads(json_path.read_text(encoding="utf-8"))
    runs = pd.read_csv(csv_path)
    cfg = yaml.safe_load((ROOT / snapshot["configuration_path"]).read_text(encoding="utf-8"))
    fx = float(cfg["reporting"]["fx"]["cop_per_usd"])
    names = yaml.safe_load((ROOT / "configs" / "product_names_en.yaml").read_text(encoding="utf-8"))
    catalog = pd.read_csv(ROOT / "data" / "seed" / "benchmark_instance_catalog.csv")
    fronts = collect_fronts(runs, fx)
    seed_metrics, summary = compute_metrics(fronts, runs)

    artifact = ROOT / "artifacts" / run
    artifact.mkdir(parents=True, exist_ok=True)
    seed_metrics.to_csv(artifact / "seed_level_metrics.csv", index=False)
    summary.to_csv(artifact / "method_summary.csv", index=False)
    fronts.to_csv(artifact / "publication_fronts_usd.csv", index=False)

    generated = ROOT / "paper" / "generated"
    figures, tables = generated / "figures", generated / "tables"
    figures.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    for old in list(figures.glob("*.pdf")) + list(tables.glob("*.tex")):
        old.unlink()
    for instance in snapshot["included_complete_instances"]:
        plot_pareto(fronts, seed_metrics, instance, figures)
    cat_included = catalog[catalog.instance.isin(snapshot["included_complete_instances"])]
    plot_runtime(summary, cat_included, figures)
    plot_quality_time(summary, figures)
    plot_refinement(seed_metrics, figures)
    schedule = make_schedule(fronts, seed_metrics, names, fx, catalog, figures)
    (artifact / "representative_schedule.json").write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    build_tables(snapshot, runs, seed_metrics, summary, catalog, schedule, tables, artifact)
    (generated / "RUN_ID.txt").write_text(run + "\n", encoding="utf-8")

    if compile_tex:
        compile_paper()
    bundle = overleaf_bundle()
    print(f"Included: {', '.join(snapshot['included_complete_instances'])}")
    print(f"Excluded: {', '.join(snapshot['excluded_incomplete_instances'])}")
    print(f"Generated: {generated}")
    print(f"Overleaf bundle: {bundle}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="publication_slsqp_test")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--schedule-only", action="store_true",
                        help="Regenerate the representative schedule without rewriting the publication snapshot")
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if args.schedule_only:
            refresh_schedule_only(args.run, compile_tex=not args.no_compile)
        else:
            refresh(args.run, compile_tex=not args.no_compile)


if __name__ == "__main__":
    main()
