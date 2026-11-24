import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from experiment import Detector, Frames
from radar_rescue import RadarMetadata, Refiner, collect, final_detections, refine


DISPLAY_THRESHOLD = .004


def main(args):
    cv2.setNumThreads(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    manifest = json.loads((args.data / 'manifest.json').read_text())
    model = Detector('rgb', None, manifest['classes']).to(args.device).eval()
    model.load_state_dict(torch.load(args.weights, map_location=args.device, weights_only=True))
    dataset = Frames(args.data, manifest, 'val', blur=4)
    scenes = sorted({row['scene'] for row in dataset.rows})
    dataset.rows = [next(row for row in dataset.rows if row['scene'] == scene and len({item[0] for item in row['labels']}) >= 2) for scene in scenes]
    meta = RadarMetadata(args.raw, manifest['version'], manifest['size'])
    records = collect(model, dataset, meta, args.device)
    metrics = json.loads((args.run / 'metrics.json').read_text())
    methods = {}
    scales = {}
    for mode in ('image', 'radar'):
        dimension = records[0][mode][1].shape[1]
        head = Refiner(dimension).to(args.device).eval()
        head.load_state_dict(torch.load(args.run / f'{mode}_refiner.pt', map_location=args.device, weights_only=True))
        candidate = refine(head, records, mode, args.device)
        scale = metrics['selection'][mode]['scale']
        scales[mode] = scale
        methods[mode] = [final_detections(record['camera'], extra, scale)
                         for record, extra in zip(records, candidate)]
    figure, axes = plt.subplots(len(records), 4, squeeze=False, layout='constrained', figsize=(10, 1.65 * len(records)))
    titles = ('Ground truth', 'Camera', 'Image refinement', 'Radar rescue')
    colors = ('#f9c742', '#73ffc2', '#ff77ca')
    frames = []
    for i, record in enumerate(records):
        photo = dataset[i][0][:3].permute(1, 2, 0)
        gt = [(row[1:], int(row[0])) for row in record['target'].tolist()]
        sets = [gt]
        predictions = {
            'camera': record['camera'], 'image_refinement': methods['image'][i],
            'radar_rescue': methods['radar'][i],
        }
        for outputs in predictions.values():
            sets.append([(box[:4], int(box[5])) for box in outputs.tolist() if box[4] >= DISPLAY_THRESHOLD])
        if args.json_out:
            row = dataset.rows[i]
            camera_token = Path(row['image']).stem
            points, _ = meta.points(row)
            frames.append({
                'scene': row['scene'], 'sample_token': row['sample'],
                'camera_sample_data_token': camera_token,
                'camera_file': meta.data[camera_token]['filename'],
                'radar_file': meta.radar[row['sample']]['filename'],
                'projected_radar_points': len(points),
                'radar_candidates': len(record['radar'][0]),
                'ground_truth': record['target'].tolist(),
                'predictions': {name: boxes.tolist() for name, boxes in predictions.items()},
            })
        for j, ax in enumerate(axes[i]):
            ax.imshow(photo)
            for (x1, y1, x2, y2), category in sets[j]:
                ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color=colors[category], linewidth=.9))
            ax.set(xlim=(0, 640), ylim=(500, 140))
            ax.axis('off')
            if not i:
                ax.set_title(titles[j], fontsize=9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=185)
    plt.close(figure)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            'dataset': manifest['version'], 'split': 'val', 'classes': manifest['classes'],
            'blur_sigma_px': 4, 'image_size': manifest['size'], 'display_threshold': DISPLAY_THRESHOLD,
            'box_format': 'xyxy on the 640x640 preprocessed canvas',
            'ground_truth_columns': ['class_id', 'x1', 'y1', 'x2', 'y2'],
            'prediction_columns': ['x1', 'y1', 'x2', 'y2', 'score', 'class_id'],
            'camera_checkpoint': str(args.weights), 'refiner_directory': str(args.run),
            'fusion_score_scales': scales, 'frames': frames,
        }, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/prepared'))
    parser.add_argument('--raw', type=Path, default=Path('data/nuscenes'))
    parser.add_argument('--weights', type=Path, default=Path('runs/rgb/model.pt'))
    parser.add_argument('--run', type=Path, default=Path('runs/rescue_42'))
    parser.add_argument('--out', type=Path, default=Path('report_assets/rescue_examples.jpg'))
    parser.add_argument('--json-out', type=Path)
    parser.add_argument('--device', default='cuda:0')
    main(parser.parse_args())
