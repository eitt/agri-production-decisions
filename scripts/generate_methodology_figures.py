"""Generate publication-ready methodology figures from the implemented model and configuration."""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agriopt.instances import CATALOG  # noqa: E402


WIDTH_MM = 175.0
MM_TO_IN = 1.0 / 25.4
COLORS = {
    "ink": "#263238",
    "muted": "#607D8B",
    "line": "#90A4AE",
    "panel": "#F7F9FA",
    "blue": "#4477AA",
    "blue_light": "#DCE8F3",
    "orange": "#D9822B",
    "orange_light": "#F8E6D2",
    "green": "#228833",
    "green_light": "#DCEEDD",
    "teal": "#009988",
    "teal_light": "#D6EFEC",
    "grey": "#9E9E9E",
    "grey_light": "#EEEEEE",
    "red": "#B33A3A",
    "red_light": "#F4DEDE",
}


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 8,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _publication_run() -> str:
    manuscript = ROOT / "paper" / "main.tex"
    if manuscript.exists():
        match = re.search(r"results/([A-Za-z0-9_.\\-]+)", manuscript.read_text(encoding="utf-8"))
        if match:
            return match.group(1).replace(r"\_", "_")
    return "publication_slsqp_test"


def _load_context(run: str) -> tuple[dict, dict, dict[str, str]]:
    snapshot_path = ROOT / "artifacts" / run / "paper_snapshot.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Publication snapshot not found: {snapshot_path}")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    config_path = ROOT / snapshot.get("configuration_path", "configs/publication_slsqp.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Publication configuration not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    names_path = ROOT / "configs" / "product_names_en.yaml"
    if not names_path.exists():
        raise FileNotFoundError(f"English product-name mapping not found: {names_path}")
    names = yaml.safe_load(names_path.read_text(encoding="utf-8"))
    return snapshot, config, names


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def _box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    face: str = "white",
    edge: str = COLORS["line"],
    lw: float = 0.9,
    fontsize: float = 7.2,
    weight: str = "normal",
    align: str = "center",
    zorder: int = 2,
    rounding: float = 0.008,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={rounding}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        transform=ax.transAxes,
        clip_on=False,
        zorder=zorder,
    )
    ax.add_patch(patch)
    tx = x + (w / 2 if align == "center" else 0.012)
    text_artist = ax.text(
        tx,
        y + h / 2,
        text,
        transform=ax.transAxes,
        ha=align,
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        weight=weight,
        linespacing=1.18,
        zorder=zorder + 1,
    )
    patch._boxed_text = text_artist
    return patch


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["muted"],
    lw: float = 1.0,
    style: str = "-|>",
    connectionstyle: str = "arc3,rad=0",
    dashed: bool = False,
    zorder: int = 1,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=8,
        linewidth=lw,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=connectionstyle,
        transform=ax.transAxes,
        clip_on=False,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def _panel_label(ax, x: float, y: float, label: str, title: str) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top", fontsize=9.5, weight="bold", color=COLORS["ink"])
    ax.text(x + 0.043, y, title, transform=ax.transAxes, ha="left", va="top", fontsize=8.5, weight="bold", color=COLORS["ink"])


def _validate_box_text(fig: plt.Figure, padding_pt: float = 1.0) -> None:
    """Fail generation whenever boxed text exceeds its visible container."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    padding_px = fig.dpi * padding_pt / 72.0
    violations: list[str] = []
    for ax in fig.axes:
        for patch in ax.patches:
            text_artist = getattr(patch, "_boxed_text", None)
            if text_artist is None:
                continue
            patch_bbox = patch.get_window_extent(renderer)
            text_bbox = text_artist.get_window_extent(renderer)
            if (
                text_bbox.x0 < patch_bbox.x0 + padding_px
                or text_bbox.x1 > patch_bbox.x1 - padding_px
                or text_bbox.y0 < patch_bbox.y0 + padding_px
                or text_bbox.y1 > patch_bbox.y1 - padding_px
            ):
                violations.append(text_artist.get_text().replace("\n", " / "))
    if violations:
        details = "\n- ".join(violations)
        raise ValueError(f"Boxed text overflow detected:\n- {details}")


def _draw_matrix(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    values: list[list[str]],
    face: str,
    edge: str,
    note: str,
    highlight: tuple[int, int] = (0, 0),
) -> None:
    rows, cols = len(values), len(values[0])
    ax.text(x, y + h + 0.034, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.2, weight="bold", color=COLORS["ink"])
    cw, ch = w / cols, h / rows
    for r in range(rows):
        ax.text(x - 0.008, y + h - (r + 0.5) * ch, f"L{r + 1}", transform=ax.transAxes, ha="right", va="center", fontsize=6.8, color=COLORS["muted"])
        for c in range(cols):
            is_highlight = (r, c) == highlight
            rect = Rectangle(
                (x + c * cw, y + h - (r + 1) * ch),
                cw,
                ch,
                transform=ax.transAxes,
                facecolor="white" if not is_highlight else face,
                edgecolor=edge,
                linewidth=1.6 if is_highlight else 0.65,
            )
            ax.add_patch(rect)
            ax.text(x + (c + 0.5) * cw, y + h - (r + 0.5) * ch, values[r][c], transform=ax.transAxes, ha="center", va="center", fontsize=6.8, color=COLORS["ink"], weight="bold" if is_highlight else "normal")
    for c in range(cols):
        ax.text(x + (c + 0.5) * cw, y + h + 0.008, f"S{c + 1}", transform=ax.transAxes, ha="center", va="bottom", fontsize=6.6, color=COLORS["muted"])
    ax.text(x, y - 0.022, note, transform=ax.transAxes, ha="left", va="top", fontsize=6.6, color=COLORS["muted"])


def figure_chromosome(names: dict[str, str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(WIDTH_MM * MM_TO_IN, 210 * MM_TO_IN))
    ax.set_axis_off()
    _panel_label(ax, 0.02, 0.98, "(a)", "Three aligned L x S gene matrices (illustrative 3 lots x 4 slots)")

    crops = [["k2", "k5", "-1", "k1"], ["k3", "k2", "k7", "-1"], ["k5", "k5", "k1", "k4"]]
    waits = [["1", "0", "2", "1"], ["0", "2", "1", "0"], ["1", "3", "0", "2"]]
    areas = [["0.65", "0.80", "0.00", "0.35"], ["0.55", "0.70", "0.45", "0.00"], ["0.60", "0.40", "0.75", "0.50"]]
    _draw_matrix(ax, 0.055, 0.755, 0.245, 0.14, r"Crop index  $crops[l,s]$", crops, COLORS["blue_light"], COLORS["blue"], "-1 unused; k internal crop index")
    _draw_matrix(ax, 0.375, 0.755, 0.245, 0.14, r"Waiting weeks  $waits[l,s]$", waits, COLORS["orange_light"], COLORS["orange"], "Integer delay after earliest start")
    _draw_matrix(ax, 0.695, 0.755, 0.245, 0.14, r"Area fraction  $area\_frac[l,s]$", areas, COLORS["green_light"], COLORS["green"], "Continuous [0,1]; area = lot area x fraction")

    gene_text = "Highlighted slot\nk2  |  wait 1 week  |  fraction 0.65"
    _box(ax, 0.03, 0.65, 0.445, 0.06, gene_text, face=COLORS["panel"], edge=COLORS["ink"], fontsize=6.3, weight="bold")
    _arrow(ax, (0.475, 0.68), (0.525, 0.68))
    _box(ax, 0.525, 0.65, 0.445, 0.06, "Internal indices\nEnglish crop names appear after decoding", face="white", edge=COLORS["line"], fontsize=5.9)

    _panel_label(ax, 0.02, 0.625, "(b)", "Deterministic genotype-to-phenotype decoder")
    steps = [
        ("Eligibility", "Supported pair\naccepted"),
        ("Earliest start", "Previous harvest\n+ setup\n+ family recovery"),
        ("Planting", "Earliest + wait\nNext allowed\nsowing week"),
        ("Harvest", "Sow week\n+ maturity"),
        ("Quantities", "Area = lot area\nx fraction\nOutput = yield\nx area"),
        ("Event", "lot, crop\nsow, harvest\narea"),
    ]
    top_xs = [0.05, 0.365, 0.68]
    bottom_xs = [0.68, 0.365, 0.05]
    box_w = 0.27
    branch_w = 0.21

    for i, ((heading, detail), x) in enumerate(zip(steps[:3], top_xs)):
        face = COLORS["blue_light"] if i in {0, 2, 3} else COLORS["green_light"] if i >= 4 else COLORS["panel"]
        edge = COLORS["blue"] if i in {0, 2, 3} else COLORS["green"] if i >= 4 else COLORS["line"]
        _box(ax, x, 0.490, box_w, 0.100, f"{heading}\n{detail}", face=face, edge=edge, fontsize=6.5, weight="bold" if i == 0 else "normal")
        if i < 2:
            _arrow(ax, (x + box_w, 0.540), (top_xs[i + 1], 0.540))

    for offset, ((heading, detail), x) in enumerate(zip(steps[3:], bottom_xs), start=3):
        face = COLORS["blue_light"] if offset == 3 else COLORS["green_light"]
        edge = COLORS["blue"] if offset == 3 else COLORS["green"]
        _box(ax, x, 0.305, box_w, 0.100, f"{heading}\n{detail}", face=face, edge=edge, fontsize=6.5, weight="bold" if offset == 5 else "normal")
        if offset < 5:
            next_x = bottom_xs[offset - 2]
            _arrow(ax, (x, 0.355), (next_x + box_w, 0.355))

    ax.plot([0.95, 0.985, 0.985], [0.540, 0.540, 0.355], transform=ax.transAxes, color=COLORS["muted"], lw=1.0, clip_on=False)
    _arrow(ax, (0.985, 0.355), (0.95, 0.355), color=COLORS["muted"])

    branch_texts = [
        "skip:\nunused slot",
        "skip:\nineligible",
        "skip:\nno sow week",
        "skip:\nlate harvest",
        "skip:\nyield <= 0",
        "Optional demand\ncap inactive here",
    ]
    for x, text in zip(top_xs, branch_texts[:3]):
        branch_x = x + (box_w - branch_w) / 2
        _arrow(ax, (x + box_w / 2, 0.490), (x + box_w / 2, 0.473), color=COLORS["grey"], dashed=True)
        _box(ax, branch_x, 0.425, branch_w, 0.043, text, face=COLORS["grey_light"], edge=COLORS["grey"], fontsize=5.8)

    for x, text in zip(bottom_xs, branch_texts[3:]):
        branch_x = x + (box_w - branch_w) / 2
        _arrow(ax, (x + box_w / 2, 0.305), (x + box_w / 2, 0.288), color=COLORS["grey"], dashed=True)
        _box(ax, branch_x, 0.240, branch_w, 0.043, text, face=COLORS["grey_light"], edge=COLORS["grey"], fontsize=5.8)

    _panel_label(ax, 0.02, 0.205, "(c)", "Decoded phenotype: sowing and harvest dates emerge from the decoder")
    gantt_x0, gantt_x1 = 0.16, 0.96
    gantt_y = [0.145, 0.095, 0.045]
    weeks = 52
    for tick in [0, 13, 26, 39, 52]:
        x = gantt_x0 + (gantt_x1 - gantt_x0) * tick / weeks
        ax.plot([x, x], [0.028, 0.165], transform=ax.transAxes, color="#E3E7E9", lw=0.6, zorder=0)
        ax.text(x, 0.020, str(tick), transform=ax.transAxes, ha="center", va="top", fontsize=6.8, color=COLORS["muted"])
    ax.text((gantt_x0 + gantt_x1) / 2, 0.001, "Planning week", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color=COLORS["ink"])
    for i, y in enumerate(gantt_y, 1):
        ax.text(0.13, y, f"Lot {i}", transform=ax.transAxes, ha="right", va="center", fontsize=7.5, color=COLORS["ink"])
        ax.plot([gantt_x0, gantt_x1], [y, y], transform=ax.transAxes, color="#CFD8DC", lw=0.7, zorder=0)
    schedules = [
        (0, 2, 13, "White maize", COLORS["blue"]), (0, 18, 32, "Potato", COLORS["orange"]),
        (1, 7, 19, "Green pea", COLORS["teal"]), (1, 24, 37, "White maize", COLORS["blue"]),
        (2, 1, 16, "Potato", COLORS["orange"]), (2, 23, 41, "Cassava", COLORS["green"]),
    ]
    for lot, start, end, label, color in schedules:
        x = gantt_x0 + (gantt_x1 - gantt_x0) * start / weeks
        w = (gantt_x1 - gantt_x0) * (end - start) / weeks
        y = gantt_y[lot] - 0.020
        ax.add_patch(Rectangle((x, y), w, 0.040, transform=ax.transAxes, facecolor=color, edgecolor=COLORS["ink"], linewidth=0.5))
        ax.text(x + w / 2, gantt_y[lot], label, transform=ax.transAxes, ha="center", va="center", fontsize=7.8, color="white", weight="bold")
    ax.text(0.98, 0.177, "Illustrative encoding; not an empirical solution", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, style="italic", color=COLORS["muted"])
    return fig


def figure_variants(config: dict) -> plt.Figure:
    ecfg = config["experiment"]
    pop = int(ecfg["nsga2"]["population"])
    generations = int(ecfg["nsga2"]["generations"])
    hcfg = ecfg["hybrid"]
    budget = int(round(pop * float(hcfg["refinement_fraction"])))
    explore = max(1, int(round(0.20 * budget)))
    exploit = budget - explore
    fig, ax = plt.subplots(figsize=(WIDTH_MM * MM_TO_IN, 245 * MM_TO_IN))
    ax.set_axis_off()

    _panel_label(ax, 0.02, 0.985, "(a)", "Shared evolutionary generation")
    shared = ["Population\n$P_t$", "Binary\ntournament", "Crossover", "Mutation", "Decode\noffspring", "True income\nand risk"]
    sx = [0.02, 0.18, 0.34, 0.50, 0.66, 0.82]
    shared_w = 0.14
    for i, (label, x) in enumerate(zip(shared, sx)):
        _box(ax, x, 0.885, shared_w, 0.058, label, face=COLORS["panel"], edge=COLORS["ink"] if i in {0, 5} else COLORS["line"], fontsize=6.6, weight="bold" if i in {0, 5} else "normal")
        if i < len(shared) - 1:
            _arrow(ax, (x + shared_w, 0.914), (sx[i + 1], 0.914))
    ax.text(0.5, 0.848, "Shared decision semantics and variation mechanics; initialization details are deliberately omitted", transform=ax.transAxes, ha="center", va="center", fontsize=6.3, color=COLORS["muted"], style="italic")

    _panel_label(ax, 0.02, 0.820, "(b)", "Alternative allocation of conditional continuous area refinement")
    columns = [0.025, 0.35, 0.675]
    col_w = 0.30
    headers = [
        ("NSGA-II baseline", COLORS["grey_light"], COLORS["muted"]),
        ("NSGA-II + full SLSQP\nrefinement", COLORS["orange_light"], COLORS["orange"]),
        ("Surrogate-guided\nH-NSGA-II", COLORS["teal_light"], COLORS["teal"]),
    ]
    for x, (title, face, edge) in zip(columns, headers):
        _box(ax, x, 0.700, col_w, 0.062, title, face=face, edge=edge, fontsize=6.5, weight="bold")

    _box(ax, 0.04, 0.545, 0.27, 0.125, "No continuous\nmathematical refinement\n\nDecoded values remain\nthe final true objectives", face="white", edge=COLORS["muted"], fontsize=6.6)
    _box(ax, 0.04, 0.475, 0.27, 0.052, "0 refinement calls\nper generation", face=COLORS["grey_light"], edge=COLORS["grey"], fontsize=6.1, weight="bold")

    _box(ax, 0.365, 0.60, 0.27, 0.085, "All offspring\nConditional area problem\nSLSQP", face="white", edge=COLORS["orange"], fontsize=6.5, weight="bold")
    _arrow(ax, (0.50, 0.60), (0.50, 0.565), color=COLORS["orange"])
    _box(ax, 0.365, 0.47, 0.27, 0.09, "Accept only when:\nSLSQP succeeds\nincome is preserved\nrisk does not worsen", face=COLORS["orange_light"], edge=COLORS["orange"], fontsize=6.2)
    _box(ax, 0.365, 0.400, 0.27, 0.052, f"{pop} x {generations}\n= {pop * generations:,} potential calls", face="white", edge=COLORS["orange"], fontsize=6.0, weight="bold")

    _box(ax, 0.69, 0.628, 0.27, 0.057, f"Budget: {budget} of {pop}\n({100 * budget / pop:.0f}% of offspring)", face="white", edge=COLORS["teal"], fontsize=6.2, weight="bold")
    _box(ax, 0.69, 0.555, 0.27, 0.06, f"Warm-up: random budget\nfor {int(hcfg['warmup_generations'])} generations or until\n{int(hcfg['min_training_samples'])} realized gains exist", face=COLORS["teal_light"], edge=COLORS["teal"], fontsize=6.0)
    _box(ax, 0.69, 0.462, 0.27, 0.073, f"Active gate\n{exploit} predicted-gain candidates\n+ {explore} random candidates", face="white", edge=COLORS["teal"], fontsize=5.9, weight="bold")
    _box(ax, 0.69, 0.385, 0.27, 0.065, "Grouped features -> Extra Trees\nSchedule | resource use\nAgronomic-economic | solution state\nSearch progress", face=COLORS["teal_light"], edge=COLORS["teal"], fontsize=5.2)
    _box(ax, 0.69, 0.312, 0.27, 0.050, "Gate, not an\nobjective surrogate", face=COLORS["red_light"], edge=COLORS["red"], fontsize=6.0, weight="bold")

    _box(ax, 0.365, 0.285, 0.27, 0.095, "Continuous subproblem\nFix crop, order, and dates\nReoptimize cultivated areas\nto minimize quadratic risk\nAccepted result updates true values", face=COLORS["panel"], edge=COLORS["ink"], fontsize=5.6)
    _arrow(ax, (0.635, 0.34), (0.69, 0.34), color=COLORS["teal"], dashed=True)

    _box(ax, 0.69, 0.240, 0.27, 0.055, "Observed gain\nrelative risk improvement\n+ relative income improvement", face="white", edge=COLORS["teal"], fontsize=5.2)
    _arrow(ax, (0.825, 0.32), (0.825, 0.298), color=COLORS["teal"])
    ax.text(0.82, 0.222, "Append observations; retrain every 2 generations", transform=ax.transAxes, ha="center", va="center", fontsize=5.1, color=COLORS["teal"])

    _panel_label(ax, 0.02, 0.17, "(c)", "Shared Pareto survival mechanism")
    survival = ["Parents +\nprocessed offspring", "Fast nondominated\nsorting", "Crowding\ndistance", "Environmental\nselection", "$P_{t+1}$"]
    bx = [0.02, 0.215, 0.410, 0.605, 0.800]
    survival_w = 0.18
    for i, (label, x) in enumerate(zip(survival, bx)):
        _box(ax, x, 0.060, survival_w, 0.060, label, face=COLORS["blue_light"] if i == len(survival) - 1 else COLORS["panel"], edge=COLORS["blue"] if i == len(survival) - 1 else COLORS["line"], fontsize=5.5, weight="bold" if i in {0, 4} else "normal")
        if i < len(survival) - 1:
            _arrow(ax, (x + survival_w, 0.090), (bx[i + 1], 0.090))
    ax.text(0.5, 0.018, "Selection and archive membership use true decoded or SLSQP-refined objectives", transform=ax.transAxes, ha="center", va="center", fontsize=5.8, color=COLORS["ink"], weight="bold")
    return fig


def figure_workflow(snapshot: dict, config: dict) -> plt.Figure:
    ecfg = config["experiment"]
    included = set(snapshot["included_complete_instances"])
    fig, ax = plt.subplots(figsize=(WIDTH_MM * MM_TO_IN, 235 * MM_TO_IN))
    ax.set_axis_off()

    ax.text(0.02, 0.98, "1", transform=ax.transAxes, ha="left", va="top", fontsize=10, weight="bold", color=COLORS["blue"])
    ax.text(0.055, 0.98, "Materials and empirical inputs", transform=ax.transAxes, ha="left", va="top", fontsize=8.8, weight="bold", color=COLORS["ink"])
    inputs = [
        (0.03, "Productivity evidence", "Municipality | crop\nplanting month | yield"),
        (0.35, "Wholesale prices", "DANE SIPSA weekly series\nand 104-week forecast"),
        (0.67, "Agronomic parameters", "Maturity | botanical family\nsowing feasibility"),
    ]
    for x, head, detail in inputs:
        _box(ax, x, 0.865, 0.29, 0.075, f"{head}\n{detail}", face=COLORS["blue_light"], edge=COLORS["blue"], fontsize=7.0, weight="bold")
    _box(ax, 0.69, 0.808, 0.27, 0.042, "Optional demand cap\ninactive in reported experiment", face=COLORS["grey_light"], edge=COLORS["grey"], fontsize=5.6)
    _arrow(ax, (0.50, 0.815), (0.50, 0.775), color=COLORS["blue"])

    ax.text(0.02, 0.78, "2", transform=ax.transAxes, ha="left", va="top", fontsize=10, weight="bold", color=COLORS["blue"])
    ax.text(0.055, 0.78, "Configured benchmark ladder and publication-complete snapshot", transform=ax.transAxes, ha="left", va="top", fontsize=8.8, weight="bold", color=COLORS["ink"])
    catalog = CATALOG.copy()
    cell_w, cell_h = 0.232, 0.082
    for idx, row in catalog.iterrows():
        rr, cc = divmod(idx, 4)
        x, y = 0.025 + cc * 0.242, 0.655 - rr * 0.094
        complete = row["instance"] in included
        face = COLORS["blue_light"] if complete else COLORS["grey_light"]
        edge = COLORS["blue"] if complete else COLORS["grey"]
        status = "publication-complete" if complete else "configured; incomplete"
        label = f"{row['instance']}\n{int(row['farms'])} farms | {int(row['lots'])} lots\n{int(row['products'])} crops | {int(row['horizon_weeks'])} weeks\n{status}"
        _box(ax, x, y, cell_w, cell_h, label, face=face, edge=edge, fontsize=6.0, weight="bold" if complete else "normal")
    _box(ax, 0.10, 0.515, 0.80, 0.032, "Construction adds lot areas, municipalities, eligibility, yield tensors, forecast prices, and covariance", face=COLORS["panel"], edge=COLORS["line"], fontsize=5.8)
    _arrow(ax, (0.50, 0.515), (0.50, 0.490), color=COLORS["blue"])

    ax.text(0.02, 0.49, "3", transform=ax.transAxes, ha="left", va="top", fontsize=10, weight="bold", color=COLORS["blue"])
    ax.text(0.055, 0.49, "Solution methods", transform=ax.transAxes, ha="left", va="top", fontsize=8.8, weight="bold", color=COLORS["ink"])
    method_specs = [
        (0.025, "Exact epsilon-constraint\nSCIP", "I01-I04\n11 income thresholds", COLORS["blue_light"], COLORS["blue"]),
        (0.267, "NSGA-II", "No continuous\nrefinement", COLORS["grey_light"], COLORS["muted"]),
        (0.509, "NSGA-II + full SLSQP", "Every offspring\nrefined", COLORS["orange_light"], COLORS["orange"]),
        (0.751, "Surrogate-guided\nH-NSGA-II", "25% refinement\nbudget", COLORS["teal_light"], COLORS["teal"]),
    ]
    for x, head, detail, face, edge in method_specs:
        _box(ax, x, 0.385, 0.224, 0.072, f"{head}\n{detail}", face=face, edge=edge, fontsize=5.8, weight="bold")
    seeds = ecfg["seeds"]
    pop = int(ecfg["nsga2"]["population"])
    gens = int(ecfg["nsga2"]["generations"])
    ax.text(0.63, 0.365, f"Stochastic methods: matched seeds {min(seeds)}-{max(seeds)} | population {pop} | {gens} generations", transform=ax.transAxes, ha="center", va="center", fontsize=6.1, color=COLORS["ink"], weight="bold")
    _arrow(ax, (0.50, 0.350), (0.50, 0.325), color=COLORS["blue"])

    ax.text(0.02, 0.335, "4", transform=ax.transAxes, ha="left", va="top", fontsize=10, weight="bold", color=COLORS["blue"])
    ax.text(0.055, 0.335, "Controlled comparison", transform=ax.transAxes, ha="left", va="top", fontsize=8.8, weight="bold", color=COLORS["ink"])
    _box(ax, 0.12, 0.255, 0.76, 0.052, "Same instance and economic-agronomic inputs\nSame chromosome semantics, matched seeds, and matched evolutionary budgets", face=COLORS["panel"], edge=COLORS["ink"], fontsize=6.0, weight="bold")
    _box(ax, 0.20, 0.195, 0.60, 0.045, "Primary contrast\nNo refinement  vs  full refinement  vs  surrogate-gated refinement", face="white", edge=COLORS["teal"], fontsize=5.8, weight="bold")
    _arrow(ax, (0.50, 0.190), (0.50, 0.170), color=COLORS["blue"])

    ax.text(0.02, 0.170, "5", transform=ax.transAxes, ha="left", va="top", fontsize=10, weight="bold", color=COLORS["blue"])
    ax.text(0.055, 0.170, "Post-hoc evaluation and reporting filter", transform=ax.transAxes, ha="left", va="top", fontsize=8.8, weight="bold", color=COLORS["ink"])
    eval_specs = [
        (0.015, "Pareto quality", "Normalized HV | IGD+\nAdditive epsilon\nNondominated points"),
        (0.210, "Computational effort", "Runtime | calls\nCumulative\nrefinement time"),
        (0.405, "Refinement effectiveness", "Useful-refinement\nhit rate"),
        (0.600, "Operational diagnostics", "Space-time utilization\nLot temporal occupancy\nMean idle weeks"),
        (0.795, "Exact-reference evidence", "SCIP status | gap\nruntime"),
    ]
    for x, head, detail in eval_specs:
        _box(ax, x, 0.075, 0.190, 0.070, f"{head}\n{detail}", face=COLORS["green_light"], edge=COLORS["green"], fontsize=5.0, weight="bold")
    _box(ax, 0.16, 0.004, 0.68, 0.058, "Post-hoc diagnostics are not optimization objectives\nPublication filter includes an instance only when every required\nmethod and seed output is complete", face=COLORS["red_light"], edge=COLORS["red"], fontsize=5.2, weight="bold")
    return fig


def _save(fig: plt.Figure, stem: str, paper_dir: Path, artifact_dir: Path) -> list[Path]:
    _validate_box_text(fig)
    paper_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for ext in ("pdf", "svg", "png"):
        targets = [artifact_dir / f"{stem}.{ext}"]
        if ext == "pdf":
            targets.append(paper_dir / f"{stem}.pdf")
        for target in targets:
            kwargs = {"dpi": 300} if ext == "png" else {}
            fig.savefig(target, format=ext, facecolor="white", **kwargs)
            outputs.append(target)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=_publication_run(), help="Publication run used for snapshot status")
    args = parser.parse_args()
    _configure_style()
    snapshot, config, names = _load_context(args.run)
    paper_dir = ROOT / "paper" / "generated" / "figures"
    artifact_dir = ROOT / "artifacts" / args.run / "figures" / "methodology"
    figures = {
        "chromosome_encoding": figure_chromosome(names),
        "metaheuristic_variants": figure_variants(config),
        "experimental_workflow": figure_workflow(snapshot, config),
    }
    outputs: list[Path] = []
    for stem, fig in figures.items():
        outputs.extend(_save(fig, stem, paper_dir, artifact_dir))
    for output in outputs:
        print(output.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
