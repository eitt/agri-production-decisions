"""Vector-first exports from completed, audited maintenance task results."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .persistence import atomic_json

LABELS = {
    "exact": "SCIP reference", "nsga2": "NSGA-II", "random_search": "Random search",
    "random_local": "Random local", "proxy_local": "Occupancy proxy", "surrogate_local": "Surrogate",
}


def _save(figure, destination: Path, stem: str) -> None:
    for extension in ("pdf", "svg", "png"):
        descriptor, temporary = tempfile.mkstemp(prefix=f".{stem}.", suffix=".tmp", dir=destination)
        os.close(descriptor)
        try:
            figure.savefig(temporary, format=extension, dpi=300, bbox_inches="tight", facecolor="white")
            os.replace(temporary, destination / f"{stem}.{extension}")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def export_figures(destination: Path, metrics: pd.DataFrame, results: dict) -> None:
    """Export descriptive HV means and one explicitly selected physical calendar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    destination.mkdir(parents=True, exist_ok=True)
    if metrics.empty:
        return
    instances = sorted(metrics.instance.unique())
    order = [method for method in LABELS if method in set(metrics.method)]
    colors = ["#64748b" if method != "surrogate_local" else "#067a75" for method in order]
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42}):
        columns = 2 if len(instances) > 1 else 1
        rows = (len(instances) + columns - 1) // columns
        figure, axes = plt.subplots(rows, columns, figsize=(7.0, 2.3 * rows), squeeze=False)
        for axis, instance in zip(axes.flat, instances):
            block = metrics[metrics.instance == instance]
            means, deviations = [], []
            for method in order:
                values = block[block.method == method].hypervolume
                means.append(float(values.mean()))
                deviations.append(float(values.std(ddof=1)) if len(values) > 1 else 0.)
            axis.barh(np.arange(len(order)), means, xerr=deviations, color=colors,
                      error_kw={"elinewidth": .8, "capsize": 2}, height=.65)
            axis.set_yticks(np.arange(len(order)), [LABELS[method] for method in order])
            axis.invert_yaxis()
            axis.set_title(instance.split("_")[0], loc="left", fontweight="bold")
            axis.set_xlabel("Normalized hypervolume (mean ± SD)")
            axis.grid(axis="x", alpha=.18)
            axis.set_axisbelow(True)
        for axis in list(axes.flat)[len(instances):]:
            axis.set_visible(False)
        figure.tight_layout(pad=1.0)
        _save(figure, destination, "hypervolume_comparison")
        plt.close(figure)

        instance = "I02_small" if "I02_small" in instances else instances[0]
        preferred = "surrogate_local" if "surrogate_local" in order else order[0]
        keys = sorted(key for key in results if key[0] == instance and key[1] == preferred)
        if not keys:
            return
        key = keys[0]
        result = results[key]
        objectives = np.asarray([decision["objectives"] for decision in result["decisions"]])
        scaled = (objectives - np.asarray(result["ideal"])) / np.asarray(result["span"])
        chosen = int(np.argmin(np.sum(scaled * scaled, axis=1)))
        decision = result["decisions"][chosen]
        events = decision.get("cultivation", decision.get("events", []))
        care = decision.get("maintenance", [])
        lots = len(decision["crops"])
        # A large-network display uses the first twelve lots, explicitly labelled.
        shown = min(lots, 12)
        horizon = int(result.get("instance_metadata", {}).get("horizon_weeks", 0))
        if not horizon:
            horizon = 104 if instance in {"I03_diverse", "I05_coop_small", "I06_coop_medium", "I07_network", "I08_network_large"} else 52
        figure, axis = plt.subplots(figsize=(7.0, max(2.3, .33 * shown + 1.2)))
        crops = sorted({event["crop_name"] for event in events})
        palette = plt.get_cmap("tab20")
        crop_colors = {crop: palette(i % 20) for i, crop in enumerate(crops)}
        for event in events:
            if event["lot"] < shown:
                axis.broken_barh([(event["sow"], event["harvest"] - event["sow"])],
                                 (event["lot"] - .3, .6), facecolors=crop_colors[event["crop_name"]])
        for activity in care:
            if activity["lot"] < shown:
                axis.plot(activity["week"] + .5, activity["lot"], marker="|", color="#222222", markersize=10, markeredgewidth=1.8)
        axis.set_yticks(np.arange(shown), [f"Plot {i + 1}" for i in range(shown)])
        axis.invert_yaxis()
        axis.set_xlim(0, horizon)
        axis.set_xlabel("Week of planning horizon")
        axis.set_title(f"{instance.split('_')[0]} · {LABELS[preferred]} · seed {key[2]}", loc="left")
        axis.grid(axis="x", alpha=.15)
        handles = [Line2D([0], [0], color=crop_colors[crop], lw=6, label=crop) for crop in crops]
        handles.append(Line2D([0], [0], color="#222222", marker="|", linestyle="none", markersize=10, label="Maintenance"))
        axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.23), ncol=3, frameon=False, fontsize=8)
        _save(figure, destination, "maintenance_calendar")
        plt.close(figure)
        atomic_json(destination / "figure_selection.json", {
            "instance": instance, "method": preferred, "seed": key[2], "decision": chosen,
            "selection": "Minimum squared distance to shared normalized ideal; descriptive illustration only.",
            "shown_plots": shown, "total_plots": lots,
            "hypervolume_error_bars": "Sample standard deviation across completed seeds, not confidence intervals.",
        })
