"""Render Figures 4 and 6 from frozen results; never launch experiments."""
from pathlib import Path
import argparse
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, MaxNLocator
import pandas as pd

WIDTH_MM = 34 * 12 / 72.27 * 25.4
METHODS = ['exact', 'nsga2', 'nsga2_qp', 'hnsga2']
LABELS = ['SCIP', 'NSGA-II', 'NSGA-II + full refinement', 'H-NSGA-II']
COLORS = ['#222222', '#4472C4', '#ED7D31', '#70AD47']
MARKERS = ['o', 's', '^', 'D']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.labelsize': 10, 'axes.titlesize': 10, 'xtick.labelsize': 9,
    'ytick.labelsize': 9, 'legend.fontsize': 9, 'pdf.fonttype': 42,
    'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})

def style(ax):
    ax.grid(axis='y', alpha=.2, linewidth=.6)
    ax.set_axisbelow(True)
    ax.tick_params(direction='out', length=3)
    ax.yaxis.set_major_locator(MaxNLocator(4))

def log_x(ax):
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.xaxis.set_minor_formatter(NullFormatter())

def legend(fig, methods):
    handles = [Line2D([], [], color=COLORS[i], marker=MARKERS[i],
        linestyle='none', markersize=6, label=LABELS[i]) for i in methods]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, .997),
        ncol=2, frameon=False, columnspacing=1.8, handletextpad=.5)

def save(fig, out, name):
    # Tight cropping trims only the outer canvas, not the panel gutters.
    for suffix, dpi in [('pdf', None), ('svg', None), ('png', 300), ('600dpi.png', 600)]:
        fig.savefig(out / f'{name}.{suffix}', dpi=dpi, bbox_inches='tight',
                    pad_inches=.025, facecolor='white')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run_id = (args.root / 'paper/generated/RUN_ID.txt').read_text().strip()
    source = args.root / 'artifacts' / run_id
    files = [source / 'method_summary.csv', source / 'seed_level_metrics.csv']
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    summary, seeds = [pd.read_csv(p) for p in files]
    instances = summary.instance.drop_duplicates().tolist()
    assert len(instances) == 4 and len(summary) == 16
    assert not summary[['runtime_median_s', 'hv_median']].isna().any().any()
    assert not summary.duplicated(['instance', 'method']).any()
    refined = seeds[seeds.method.isin(METHODS[2:])]
    assert len(refined) == 80
    assert refined.groupby(['instance', 'method']).size().eq(10).all()
    assert not refined[['refinement_time_s', 'normalized_hypervolume']].isna().any().any()
    assert refined.refinement_time_s.gt(0).all()
    args.out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 175 / 25.4))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.05, 1, 1],
        left=.14, right=.985, bottom=.075, top=.91, hspace=.68, wspace=.34)
    ax = fig.add_subplot(grid[0, :])
    for i, method in enumerate(METHODS):
        z = summary[summary.method.eq(method)].set_index('instance').loc[instances]
        ax.plot(range(4), z.runtime_median_s, color=COLORS[i], marker=MARKERS[i],
                markersize=5.5, linewidth=1.5)
    style(ax)
    ax.set_yscale('log')
    ax.set_xticks(range(4), [x.split('_')[0] for x in instances])
    ax.set_ylabel('Runtime (s, log scale)')
    ax.set_title('(a) Runtime by instance', loc='left', pad=7)
    ymin, ymax = summary.hv_median.min(), summary.hv_median.max()
    margin = (ymax - ymin) * .09
    for j, instance in enumerate(instances):
        ax = fig.add_subplot(grid[1 + j // 2, j % 2])
        z = summary[summary.instance.eq(instance)].set_index('method')
        for i, method in enumerate(METHODS):
            row = z.loc[method]
            ax.scatter(row.runtime_median_s, row.hv_median, color=COLORS[i],
                       marker=MARKERS[i], s=42, edgecolor='white', linewidth=.5, zorder=3)
        style(ax); log_x(ax)
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_title(f'({chr(98 + j)}) {instance.split("_")[0]}', loc='left', pad=7)
        if j % 2 == 0:
            ax.set_ylabel('Normalized HV')
        if j >= 2:
            ax.set_xlabel('Runtime (s, log scale)')
    legend(fig, range(4))
    save(fig, args.out, 'figure4_compute')

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_MM / 25.4, 108 / 25.4), sharey=True)
    for j, (ax, instance) in enumerate(zip(axes.flat, instances)):
        for i in [2, 3]:
            z = refined[refined.instance.eq(instance) & refined.method.eq(METHODS[i])]
            ax.scatter(z.refinement_time_s, z.normalized_hypervolume,
                       color=COLORS[i], marker=MARKERS[i], s=42, alpha=.85,
                       edgecolor='white', linewidth=.4, zorder=3)
        style(ax); log_x(ax)
        ax.set_title(f'({chr(97 + j)}) {instance.split("_")[0]}', loc='left', pad=7)
    fig.supxlabel('Cumulative refinement time (s, log scale)', y=.015, fontsize=10)
    fig.supylabel('Normalized hypervolume', x=.015, fontsize=10)
    fig.subplots_adjust(left=.14, right=.985, bottom=.14, top=.85, wspace=.25, hspace=.40)
    legend(fig, [2, 3])
    save(fig, args.out, 'surrogate_refinement_efficiency')
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report = {'run_id': run_id, 'source_sha256': before,
        'figure4_records': len(summary), 'figure6_points': len(refined),
        'width_mm_before_tight_crop': WIDTH_MM, 'font_pt': {'labels': 10, 'ticks': 9},
        'figure4_height_mm': 175, 'figure6_height_mm': 108,
        'transforms': 'logarithmic time axes only; original values unchanged',
        'filters': 'Figure 6: full refinement and H-NSGA-II; all 10 seeds per instance retained'}
    (args.out / 'readability_manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
