"""Build the external-validation table using recorded frozen-model metrics."""
import argparse
import json
from pathlib import Path
from statistics import mean


METHODS = {'rgb': '纯相机', 'radar_simple': '直接合并雷达框', 'image_refined': '同型纯图像精修', 'radar_refined': '雷达候选补检＋图像精修', 'radar_missing50': '雷达补检（半数帧缺雷达）'}


def main(args):
    datasets = [json.loads((args.results / filename).read_text(encoding='utf-8')) for filename in ('nuscenes_unseen.json', 'astyx.json')]
    for data in datasets:
        if data['sigma'] != 4 or data['target_data_training'] or set(data['seeds']) != {'42', '43', '44'}:
            raise ValueError(f"Unexpected external-validation protocol for {data['dataset']}")
    rows = []
    for method, label in METHODS.items():
        stats = [mean(seed[method][key] for seed in data['seeds'].values()) * 100 for data in datasets for key in ('ap', 'ap50')]
        rows.append(' & '.join([label, *(f'{value:.2f}' for value in stats)]) + r' \\')
        print(method, *(f'{value:.5f}' for value in stats))
    text = '\\newcommand{\\ExpansionRows}{%\n' + '\n'.join(rows) + '\n}\n'
    args.out.write_text(text, encoding='utf-8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path('results/expansion'))
    parser.add_argument('--out', type=Path, default=Path('paper/figures/expansion.tex'))
    main(parser.parse_args())
