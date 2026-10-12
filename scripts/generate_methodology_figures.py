"""Generate explanatory diagrams at the Interact template's final text width.

Only methodology artwork is regenerated; no experiments or result plots run.
Geometry uses millimetres and typography uses publication-size points.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.transforms import Bbox
import yaml

ROOT = Path(__file__).resolve().parents[1]
# interact.cls specifies 34 TeX picas, not the generic 175-mm artwork width.
WIDTH_MM = 34 * 12 / 72.27 * 25.4
COLORS = dict(ink="#263238", muted="#586870", line="#A8B2B8",
              panel="#F5F7F8", blue="#4477AA", blue_light="#E5EDF5",
              orange="#A55C14", orange_light="#FAEDDF",
              green="#24783A", green_light="#E6F1E8",
              teal="#007F73", teal_light="#E1F2EF",
              grey_light="#F0F1F2")


def _configure_style():
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9, "mathtext.fontset": "dejavusans",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "figure.facecolor": "white", "savefig.facecolor": "white"})


def _new_canvas(height_mm):
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, height_mm / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH_MM), ylim=(height_mm, 0))
    ax.set_axis_off()
    return fig, ax


def _text(ax, x, y, label, *, size=9, bold=False, ha="left", va="center", color=None):
    return ax.text(x, y, label, fontsize=size, weight="bold" if bold else "normal",
                   ha=ha, va=va, linespacing=1.22, color=color or COLORS["ink"])


def _panel(ax, y, marker, label):
    _text(ax, 1, y, f"{marker}  {label}", size=10, bold=True)


def _box(ax, x, y, w, h, label, *, face="panel", edge="line", bold=False, size=9):
    patch = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0,rounding_size=1.1",
                          linewidth=.75, facecolor=COLORS.get(face, face),
                          edgecolor=COLORS.get(edge, edge))
    ax.add_patch(patch)
    patch._boxed_text = _text(ax, x + w / 2, y + h / 2, label,
                             size=size, bold=bold, ha="center")
    patch._layout_box = True
    return patch


def _arrow(ax, start, end, *, color="muted", dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8,
                                linewidth=.85, shrinkA=1.5, shrinkB=1.5,
                                linestyle="--" if dashed else "-",
                                color=COLORS.get(color, color)))


def _validate_layout(fig):
    """Reject overflow, text collisions, overlapping boxes and clipped artists."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    safety = fig.dpi / 72  # one point of interior clearance
    for ax in fig.axes:
        boxes = [p for p in ax.patches if getattr(p, "_layout_box", False)]
        for p in boxes:
            outer = p.get_window_extent(renderer)
            inner = p._boxed_text.get_window_extent(renderer)
            if (inner.x0 < outer.x0 + safety or inner.x1 > outer.x1 - safety
                    or inner.y0 < outer.y0 + safety or inner.y1 > outer.y1 - safety):
                raise ValueError(f"Text overflows box: {p._boxed_text.get_text()!r}")
        for i, p in enumerate(boxes):
            for q in boxes[i + 1:]:
                if p.get_window_extent(renderer).overlaps(q.get_window_extent(renderer)):
                    raise ValueError("Overlapping diagram boxes")
        texts = [(t, t.get_window_extent(renderer)) for t in ax.texts]
        for i, (t, b) in enumerate(texts):
            if t.get_fontsize() < 9:
                raise ValueError("Essential text must be at least 9 pt")
            if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.x1 or b.y1 > fig.bbox.y1:
                raise ValueError(f"Text outside canvas: {t.get_text()!r}")
            for other, c in texts[i + 1:]:
                if b.overlaps(c):
                    raise ValueError(f"Text collision: {t.get_text()!r} / {other.get_text()!r}")


def _matrix(ax, x, y, title, values, color):
    w, h = 37, 21
    _text(ax, x, y - 9, title, bold=True)
    for c in range(4):
        _text(ax, x + (c + .5) * w / 4, y - 3, f"S{c+1}", ha="center")
    for r, row in enumerate(values):
        _text(ax, x - 1.5, y + (r + .5) * h / 3, f"P{r+1}", ha="right")
        for c, value in enumerate(row):
            hit = (r, c) == (0, 0)
            ax.add_patch(Rectangle((x + c * w / 4, y + r * h / 3), w / 4, h / 3,
                                   facecolor=COLORS[f"{color}_light"] if hit else "white",
                                   edgecolor=COLORS[color], linewidth=1.2 if hit else .5))
            _text(ax, x + (c + .5) * w / 4, y + (r + .5) * h / 3,
                  value, ha="center", bold=hit)


def figure_chromosome(names=None):
    fig, ax = _new_canvas(170)
    _panel(ax, 3, "(a)", "Aligned genes: production plots × slots")
    crops = [["k2", "k5", "−1", "k1"], ["k3", "k2", "k7", "−1"], ["k5", "k5", "k1", "k4"]]
    waits = [["1", "0", "2", "1"], ["0", "2", "1", "0"], ["1", "3", "0", "2"]]
    areas = [["0.65", "0.80", "0.00", "0.35"], ["0.55", "0.70", "0.45", "0.00"], ["0.60", "0.40", "0.75", "0.50"]]
    for x, title, values, color in [(6, "Crop index", crops, "blue"),
                                    (54, "Wait (weeks)", waits, "orange"),
                                    (102, "Area fraction [0, 1]", areas, "green")]:
        _matrix(ax, x, 21, title, values, color)
    _text(ax, 1, 47, "P: plot; S: slot; k: crop index; −1: unused slot")
    _box(ax, 1, 52, WIDTH_MM - 2, 10,
         "Highlighted slot: crop k2 · wait 1 week · area fraction 0.65", bold=True)

    _panel(ax, 69, "(b)", "Deterministic decoding")
    xs, w = [1, 50, 99], 41
    labels = ["1  Eligibility\nSupported plot–crop pair",
              "2  Earliest start\nPrevious harvest + setup\n+ family recovery",
              "3  Sowing date\nEarliest start + wait\nNext allowed sowing week",
              "4  Harvest date\nSowing + maturity",
              "5  Area and output\nFraction × plot area\nYield × cultivated area",
              "6  Production event\nPlot, crop, dates and area"]
    for x, label in zip(xs, labels[:3]):
        _box(ax, x, 75, w, 17, label, face="blue_light", edge="blue")
    for x, label in zip(xs[::-1], labels[3:]):
        _box(ax, x, 98, w, 19, label, face="green_light", edge="green")
    for a, b in zip(xs[:-1], xs[1:]):
        _arrow(ax, (a + w, 83.5), (b, 83.5))
    _arrow(ax, (119.5, 92), (119.5, 98))
    _arrow(ax, (99, 107.5), (91, 107.5))
    _arrow(ax, (50, 107.5), (42, 107.5))
    # Rejection criteria are a single note, not a second layer of branches.
    _text(ax, 1, 122, "Skip unused or infeasible slots; require harvest within the horizon and positive output.")
    _panel(ax, 129, "(c)", "Illustrative schedule (not an optimization result)")
    # Compact schematic retains the original illustrative dates; no empirical data.
    x0, span = 20, 120
    schedules = [(0, 2, 13, "White maize", "blue"), (0, 18, 32, "Potato", "orange"),
                 (1, 7, 19, "Green pea", "teal"), (1, 24, 37, "White maize", "blue"),
                 (2, 1, 16, "Potato", "orange"), (2, 23, 41, "Cassava", "green")]
    for i, y in enumerate([137, 146, 155]):
        _text(ax, 1, y + 3, f"Plot {i+1}")
        ax.plot([x0, x0 + span], [y + 3, y + 3], color=COLORS["line"], lw=.5, zorder=0)
    for plot, start, end, label, color in schedules:
        x, w = x0 + span * start / 52, span * (end-start) / 52
        y = [137, 146, 155][plot]
        ax.add_patch(Rectangle((x, y), w, 6, facecolor=COLORS[f"{color}_light"],
                               edgecolor=COLORS[color], linewidth=.7))
        _text(ax, x + w/2, y + 3, label, ha="center")
    for week in [0, 13, 26, 39, 52]:
        _text(ax, x0 + span * week/52, 164, str(week), ha="center")
    _text(ax, 1, 164, "Week")
    return fig


def figure_variants(config):
    ecfg = config["experiment"]
    pop, gens = int(ecfg["nsga2"]["population"]), int(ecfg["nsga2"]["generations"])
    hcfg = ecfg["hybrid"]
    budget = max(1, math.ceil(pop * float(hcfg["refinement_fraction"])))
    explore = max(1, round(float(hcfg.get("exploration_share", .20)) * budget))
    fig, ax = _new_canvas(154)
    _panel(ax, 3, "(a)", "Shared evolutionary generation")
    labels = ["Population\n$P_t$", "Tournament\nselection", "Crossover\nand mutation", "Decode and\nevaluate offspring"]
    xs, w = [1, 38, 75, 112], 30
    for x, label in zip(xs, labels):
        _box(ax, x, 9, w, 14, label)
    for a, b in zip(xs[:-1], xs[1:]):
        _arrow(ax, (a+w, 16), (b, 16))
    _panel(ax, 30, "(b)", "Refinement allocation")
    rows = [(36, "NSGA-II", "No SLSQP calls\nRetain decoded cultivated areas", "grey_light", "muted"),
            (56, "NSGA-II\n+ full refinement", f"Refine every offspring\n{pop} calls per generation; {pop*gens:,} per run", "orange_light", "orange"),
            (76, "H-NSGA-II", f"Refine {budget} of {pop} offspring ({budget/pop:.0%})\n{budget*gens:,} calls per run", "teal_light", "teal")]
    for y, title, detail, face, edge in rows:
        _box(ax, 1, y, 40, 16, title, face=face, edge=edge, bold=True)
        _box(ax, 47, y, WIDTH_MM-48, 16, detail, face="white", edge=edge)
        _arrow(ax, (41, y+8), (47, y+8), color=edge)
    _text(ax, 1, 99, f"H-NSGA-II: random selection for the first {hcfg['warmup_generations']} generations; train after ≥{hcfg['min_training_samples']} gains.")
    _text(ax, 1, 105, f"Extra Trees ranks predicted gains: {budget-explore} ranked + {explore} random candidates per generation.")
    _text(ax, 1, 111, f"Observed gains update the predictor every {hcfg.get('retrain_every', 2)} generations; objectives are not predicted.")
    _box(ax, 1, 118, WIDTH_MM-2, 17,
         "Both refined variants: fix crops and dates; optimize areas with SLSQP.\nAccept successful refinements only if revenue is preserved and risk does not rise.")
    _panel(ax, 141, "(c)", "Shared survival using true objectives")
    _text(ax, 1, 148, "Parents + offspring → nondominated sorting → crowding distance → next population")
    return fig


def figure_workflow(snapshot, config):
    ecfg = config["experiment"]
    included = snapshot["included_complete_instances"]
    excluded = [i for i in ecfg["instances"] if i not in included]
    short = lambda items: ", ".join(i.split("_")[0] for i in items)
    fig, ax = _new_canvas(144)
    _panel(ax, 3, "1", "Empirical inputs")
    specs = [(1, "Productivity\nMunicipality, crop and yield"),
             (50, "Wholesale prices\nSIPSA data and forecasts"),
             (99, "Agronomic inputs\nCalendars and maturity")]
    for x, text in specs:
        _box(ax, x, 9, 41, 15, text, face="blue_light", edge="blue")
    _arrow(ax, (WIDTH_MM/2, 24), (WIDTH_MM/2, 29), color="blue")
    _panel(ax, 32, "2", "Data-informed synthetic benchmarks")
    _box(ax, 1, 38, WIDTH_MM-2, 17,
         f"{len(ecfg['instances'])} designed instances; {len(included)} meet the completeness rule\nIncluded: {short(included)} · Incomplete: {short(excluded) or 'none'}", face="panel")
    _arrow(ax, (WIDTH_MM/2, 55), (WIDTH_MM/2, 60), color="blue")
    _panel(ax, 63, "3", "Solution methods")
    specs = [(1, 69, f"SCIP ε-constraint\n{ecfg['exact']['points']} revenue thresholds", "blue_light", "blue"),
             (75, 69, "NSGA-II\nNo area refinement", "grey_light", "muted"),
             (1, 88, "NSGA-II + full refinement\nSLSQP for every offspring", "orange_light", "orange"),
             (75, 88, f"H-NSGA-II\nSLSQP for {ecfg['hybrid']['refinement_fraction']:.0%} of offspring", "teal_light", "teal")]
    for x, y, text, face, edge in specs:
        _box(ax, x, y, 66, 15, text, face=face, edge=edge)
    _text(ax, 1, 110, f"Evolutionary methods: matched initialization · {len(ecfg['seeds'])} seeds · {ecfg['nsga2']['population']} candidates · {ecfg['nsga2']['generations']} generations")
    _panel(ax, 118, "4", "Evaluation after all required runs are complete")
    _text(ax, 1, 125, r"Pareto quality: pooled reference, HV, IGD$^+$ and ε; paired Wilcoxon–Holm tests")
    _text(ax, 1, 132, "Computational effort: runtime, refinement calls, refinement time and useful-call rate")
    _text(ax, 1, 139, "Diagnostics: utilization, occupancy and idle weeks; SCIP status and optimality gap")
    return fig


def _publication_run():
    manuscript = ROOT / "paper" / "main.tex"
    if manuscript.exists():
        match = re.search(r"results/([A-Za-z0-9_.\\-]+)", manuscript.read_text(encoding="utf-8"))
        if match:
            return match.group(1).replace(r"\_", "_")
    return "publication_slsqp_test"


def _load_context(run):
    snapshot = json.loads((ROOT / "artifacts" / run / "paper_snapshot.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / snapshot.get("configuration_path", "configs/publication_slsqp.yaml")).read_text(encoding="utf-8"))
    return snapshot, config


def _save(fig, stem, paper_dir, artifact_dir):
    _validate_layout(fig)
    renderer = fig.canvas.get_renderer()
    artists = [a for ax in fig.axes for a in [*ax.texts, *ax.patches, *ax.lines]]
    content = Bbox.union([a.get_window_extent(renderer) for a in artists])
    # Crop vertical padding to one point, retaining the exact template width.
    crop = Bbox.from_extents(0, max(0, content.y0 - fig.dpi/72),
                            fig.bbox.width, min(fig.bbox.height, content.y1 + fig.dpi/72))
    crop_inches = crop.transformed(fig.dpi_scale_trans.inverted())
    for directory in (paper_dir, artifact_dir):
        directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, fmt, dpi in [(".pdf", "pdf", None), (".svg", "svg", None),
                              (".png", "png", 300), ("_600dpi.png", "png", 600)]:
        targets = [artifact_dir / f"{stem}{suffix}"]
        if fmt == "pdf":
            targets.append(paper_dir / f"{stem}.pdf")
        for target in targets:
            fig.savefig(target, format=fmt, dpi=dpi, facecolor="white", bbox_inches=crop_inches, pad_inches=0)
            outputs.append(target)
    plt.close(fig)
    return outputs


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root (for staging and tests)")
    parser.add_argument("--run", help="Publication snapshot identifier")
    parser.add_argument("--output-root", type=Path, help="Stage artwork without modifying project assets")
    args = parser.parse_args()
    ROOT = args.root.resolve()
    run = args.run or _publication_run()
    target = args.output_root.resolve() if args.output_root else ROOT
    _configure_style()
    snapshot, config = _load_context(run)
    figures = {"chromosome_encoding": figure_chromosome(),
               "metaheuristic_variants": figure_variants(config),
               "experimental_workflow": figure_workflow(snapshot, config)}
    for stem, fig in figures.items():
        for output in _save(fig, stem, target / "paper/generated/figures",
                            target / "artifacts" / run / "figures/methodology"):
            print(output)


if __name__ == "__main__":
    main()
