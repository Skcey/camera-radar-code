"""Export the already measured sigma-4 evaluation and generate paper tables."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev


METHODS = {'rgb': '纯相机', 'radar_simple': '直接合并雷达框', 'image_refined': '同型纯图像精修', 'radar_refined': '雷达候选补检＋图像精修', 'radar_missing50': '雷达补检（半数帧缺雷达）'}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def row(*cells):
    return ' & '.join(cells) + r' \\'


def percentage(value):
    return f'{value * 100:.2f}'


def macro(name, lines):
    return rf'\newcommand{{\{name}}}{{%' + '\n' + '\n'.join(lines) + '\n}\n'


def main(args):
    seeds = (42, 43, 44)
    reports = []
    for seed in seeds:
        if args.extract:
            source = load(args.extract / f'rescue_{seed}/metrics.json')
            if source['seed'] != seed or '4' not in source['conditions'] or set(source['conditions']) != {'0', '2', '4'}:
                raise ValueError(f'Expected an original mixed-augmentation run for seed {seed}')
            report = {'seed': seed, 'blur_sigma_px': 4, 'weights': source['weights'], 'development_scenes': source['development_scenes'], 'refiner_score_scales': {name: value['scale'] for name, value in source['selection'].items()}, 'metrics': source['conditions']['4']}
            args.results.mkdir(parents=True, exist_ok=True)
            (args.results / f'seed_{seed}.json').write_text(json.dumps(report, indent=2), encoding='utf-8', newline='\n')
        report = load(args.results / f'seed_{seed}.json')
        if report['seed'] != seed or report['blur_sigma_px'] != 4:
            raise ValueError(f'Incorrect evaluation record for seed {seed}')
        reports.append(report)
    rows, class_rows, seed_rows = [], [], []
    for method, label in METHODS.items():
        records = [report['metrics'][method] for report in reports]
        aps = [record['ap'] for record in records]
        rows.append(row(label, *(percentage(mean(record[key] for record in records)) for key in ('ap', 'ap50', 'ap75', 'ar100'))))
        class_rows.append(row(label, *(percentage(mean(record['per_class'][name] for record in records)) for name in ('car', 'pedestrian', 'bicycle'))))
        if method in ('rgb', 'image_refined', 'radar_refined'):
            seed_rows.append(row(label, *(percentage(value) for value in aps), rf'${percentage(mean(aps))}\pm{percentage(stdev(aps))}$'))
    text = ''.join(macro(name, contents) for name, contents in (('SevereRows', rows), ('ClassRows', class_rows), ('SeedRows', seed_rows)))
    args.table.write_text(text, encoding='utf-8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', type=Path, help='Original runs directory for extracting the sigma-4 evaluation')
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--table', type=Path, default=Path('paper/figures/results.tex'))
    main(parser.parse_args())
