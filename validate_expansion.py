"""Frozen external evaluation at sigma-4, with no fitting on target scenes."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from experiment import Detector, Frames, ground_truth
from radar_rescue import RadarMetadata, Refiner, collect, measure, refine


class AstyxMetadata:
    def __init__(self, prepared):
        self.prepared = prepared

    def points(self, row):
        return np.load(self.prepared / row['radar']), row['focal']


def run_seed(args, seed, manifest, samples, annotations, metadata, categories):
    source = args.source_runs
    weight = source / ('rgb/model.pt' if seed == 42 else f'seed_{seed}/rgb/model.pt')
    fit = source / f'rescue_{seed}'
    model = Detector('rgb', None, manifest['classes']).to(args.device).eval()
    model.load_state_dict(torch.load(weight, map_location=args.device, weights_only=True))
    results = collect(model, samples, metadata, args.device)
    estimates = {'rgb': measure(results, [None] * len(results), 0, annotations, categories)}
    protocol = json.loads((fit / 'metrics.json').read_text())
    excluded = set(np.random.default_rng(2026).permutation(len(results))[:round(len(results) / 2)])
    for mode in ('image', 'radar'):
        scale = protocol['selection'][mode]['scale']
        head = Refiner(results[0][mode][1].shape[1]).to(args.device).eval()
        head.load_state_dict(torch.load(fit / f'{mode}_refiner.pt', map_location=args.device, weights_only=True))
        rescored = refine(head, results, mode, args.device)
        estimates[f'{mode}_refined'] = measure(results, rescored, scale, annotations, categories)
        if mode == 'radar':
            estimates['radar_missing50'] = measure(results, [item[:0] if i in excluded else item for i, item in enumerate(rescored)], scale, annotations, categories)
    simple = [torch.cat((r['radar'][0], torch.full((len(r['radar'][0]), 1), .01), r['radar'][2][:, None]), 1) for r in results]
    estimates['radar_simple'] = measure(results, simple, 1, annotations, categories)
    return estimates


def main(args):
    cv2.setNumThreads(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    manifest = json.loads((args.prepared / 'manifest.json').read_text())
    samples = Frames(args.prepared, manifest, 'val', blur=4)
    astyx = manifest['version'] == 'Astyx HiRes2019'
    categories = ['car'] if astyx else manifest['classes']
    metadata = AstyxMetadata(args.prepared) if astyx else RadarMetadata(args.raw, manifest['version'], manifest['size'])
    annotations = ground_truth(samples, categories)
    seeds = {}
    scenes = sorted({row['scene'] for row in samples.rows})
    labels = {name: sum(int(label[0]) == j for row in samples.rows for label in row['labels']) for j, name in enumerate(categories)}
    for seed in args.seeds:
        metrics = run_seed(args, seed, manifest, samples, annotations, metadata, categories)
        seeds[str(seed)] = metrics
        print(seed, {name: [round(100 * r['ap'], 3), round(100 * r['ap50'], 3)] for name, r in metrics.items()}, flush=True)
    domain = {'Astyx HiRes2019': 'cross-dataset', 'v1.0-mini': 'mini validation'}
    report = {'dataset': manifest['version'], 'domain': domain.get(manifest['version'], 'unseen validation scenes'), 'sigma': 4, 'frames': len(samples), 'scenes': None if astyx else scenes, 'classes': categories, 'labels': labels, 'target_data_training': False, 'astyx_unmatched_radar_fields': 'set to zero' if astyx else None, 'seeds': seeds}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--source-runs', type=Path, default=Path('runs'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=(42, 43, 44))
    parser.add_argument('--device', default='cuda:0')
    main(parser.parse_args())
