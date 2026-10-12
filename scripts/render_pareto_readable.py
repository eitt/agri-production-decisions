"""Figure 5: publication-size fronts from frozen results, without new runs."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import pandas as pd
from render_results_readable import WIDTH_MM, METHODS, LABELS, COLORS, MARKERS, style, legend, save

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run = (args.root / 'paper/generated/RUN_ID.txt').read_text().strip()
    sources = [args.root / 'artifacts' / run / name for name in
               ['publication_fronts_usd.csv', 'seed_level_metrics.csv']]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    fronts, metrics = [pd.read_csv(p) for p in sources]
    instances = metrics.instance.drop_duplicates().tolist()
    assert len(instances) == 4
    plt.rcParams.update({'font.size': 9, 'xtick.labelsize': 9,
                        'ytick.labelsize': 9, 'legend.fontsize': 9,
                        'axes.titlesize': 10})
    fig, axes = plt.subplots(2, 2, figsize=(.8 * WIDTH_MM / 25.4, 105 / 25.4))
    selections = []
    for j, (ax, instance) in enumerate(zip(axes.flat, instances)):
        for i, method in enumerate(METHODS):
            z = fronts[fronts.instance.eq(instance) & fronts.method.eq(method)]
            assert not z.empty
            if method == 'exact':
                seed = int(z.seed.iloc[0])
            else:
                m = metrics[metrics.instance.eq(instance) & metrics.method.eq(method)]
                seed = int(m.loc[(m.normalized_hypervolume - m.normalized_hypervolume.median()).abs().idxmin(), 'seed'])
            z = z[z.seed.eq(seed)].sort_values('income_kusd')
            assert not z[['income_kusd', 'risk_kusd2']].isna().any().any()
            ax.plot(z.income_kusd, z.risk_kusd2, color=COLORS[i],
                    marker=MARKERS[i], markersize=3, linewidth=1.1,
                    markevery=1 if method == 'exact' else max(1, len(z) // 12))
            selections.append({'instance': instance, 'method': method, 'seed': seed, 'points': len(z)})
        style(ax)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(formatter)
        ax.set_title(f'({chr(97 + j)}) {instance.split("_")[0]}', loc='left', y=1.13, pad=3)
    fig.supxlabel('Expected gross revenue (thousand USD)', y=.018, fontsize=10)
    fig.supylabel(r'Correlated price risk ((thousand USD)$^2$)', x=.015, fontsize=10)
    fig.subplots_adjust(left=.17, right=.985, bottom=.14, top=.80,
                        wspace=.40, hspace=.64)
    legend(fig, range(4))
    args.out.mkdir(parents=True, exist_ok=True)
    save(fig, args.out, 'pareto_fronts')
    assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report = {'run_id': run, 'source_sha256': hashes, 'selected_fronts': selections,
              'width_mm_before_crop': .8 * WIDTH_MM, 'height_mm_before_crop': 105,
              'aggregation': 'none; seed nearest method-instance median HV; exact archive unchanged',
              'scales': 'linear, original USD values; instance-specific limits'}
    (args.out / 'figure5_manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
