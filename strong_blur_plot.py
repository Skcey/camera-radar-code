import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np


def main(args):
    seeds = [json.loads((args.results / f'seed_{seed}.json').read_text())['metrics'] for seed in (42, 43, 44)]
    for seed, record in zip((42, 43, 44), seeds):
        reference = json.loads((args.results / 'rrpn' / f'seed_{seed}.json').read_text())
        record['rrpn_refined'] = reference['evaluation']['mini']['rrpn_refined']
    methods = ('rgb', 'radar_simple', 'image_refined', 'radar_refined', 'radar_missing50', 'rrpn_refined')
    labels = ('Camera\nonly', 'Direct\nradar merge', 'Image-only\nrefinement', 'Radar\nrescue',
              '50% radar\nmissing', 'RRPN\nrefinement')
    colors = ('#657E96', '#8D99A6', '#BA865C', '#08786B', '#87A798', '#7B6299')
    positions = np.arange(len(methods))
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 2.9), layout='constrained')
    panels = (('ap', '(a) mAP50–95'), ('ap50', '(b) AP50'))
    for axis, (key, title) in zip(axes, panels):
        values = np.array([[100 * record[method][key] for record in seeds] for method in methods])
        means = values.mean(axis=1)
        deviations = values.std(axis=1, ddof=1)
        top = means + deviations
        axis.bar(positions, means, color=colors, width=.64, yerr=deviations,
                 error_kw={'ecolor': '#28323C', 'elinewidth': 1, 'capsize': 3, 'capthick': 1},
                 zorder=3)
        for position, mean, upper in zip(positions, means, top):
            axis.annotate(f'{mean:.2f}', (position, upper), xytext=(0, 4),
                          textcoords='offset points', ha='center', va='bottom', fontsize=8)
        axis.set(xticks=positions, xticklabels=labels, ylabel='Score (%)',
                 ylim=(0, top.max() * 1.19))
        axis.set_title(title, loc='left', fontsize=10, fontweight='semibold', pad=8)
        axis.tick_params(axis='x', length=0, pad=5, labelsize=7)
        axis.tick_params(axis='y', labelsize=8)
        axis.grid(axis='y', color='#DCE2E8', linewidth=.7, zorder=0)
        axis.spines[['top', 'right']].set_visible(False)
    figure.savefig(args.out)
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--out', type=Path, default=Path('paper/figures/severe_comparison.pdf'))
    main(parser.parse_args())
