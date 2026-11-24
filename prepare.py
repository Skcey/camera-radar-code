import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.scripts import export_2d_annotations_as_json as export_2d
from nuscenes.utils.splits import create_splits_scenes

CLASSES = ('car', 'pedestrian', 'bicycle')


def class_id(name):
    if name == 'vehicle.car':
        return 0
    if name.startswith('human.pedestrian.'):
        return 1
    if name == 'vehicle.bicycle':
        return 2
    return None


def prepare(args):
    nusc = NuScenes(version=args.version, dataroot=str(args.root), verbose=False)
    export_2d.nusc = nusc
    args.out.mkdir(parents=True, exist_ok=True)
    scenes = create_splits_scenes()
    prefix = 'mini_' if args.version == 'v1.0-mini' else ''
    excluded = set(scenes['mini_train'] + scenes['mini_val']) if args.exclude_mini else set()
    split_by_scene = {name: split for split in args.splits for name in scenes[prefix + split] if name not in excluded}
    rows = {'train': [], 'val': []}
    counts = Counter()
    for sample in nusc.sample:
        scene = nusc.get('scene', sample['scene_token'])['name']
        if scene not in split_by_scene:
            continue
        split = split_by_scene[scene]
        cam = sample['data']['CAM_FRONT']
        radar = sample['data']['RADAR_FRONT']
        relative = nusc.get('sample_data', cam)['filename']
        radar_path = nusc.get('sample_data', radar)['filename']
        if not (args.root / relative).is_file() or not (args.root / radar_path).is_file():
            if args.allow_subset:
                continue
            raise FileNotFoundError(f'Missing camera or radar pair: {relative}, {radar_path}')
        pixels, depth, _ = nusc.explorer.map_pointcloud_to_image(radar, cam)
        image = cv2.imread(str(args.root / relative))
        height, width = image.shape[:2]
        scale = args.size / max(width, height)
        new_w, new_h = round(width * scale), round(height * scale)
        offset_x, offset_y = (args.size - new_w) // 2, (args.size - new_h) // 2
        sx, sy = new_w / width, new_h / height
        canvas = np.full((args.size, args.size, 3), 114, dtype=np.uint8)
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = cv2.resize(image, (new_w, new_h))
        px = np.clip(np.rint(pixels[0] * sx).astype(int), 0, new_w - 1) + offset_x
        py = np.clip(np.rint(pixels[1] * sy).astype(int), 0, new_h - 1) + offset_y
        distance = np.full((args.size, args.size), np.inf, dtype=np.float32)
        np.minimum.at(distance, (py, px), depth)
        y, x = np.nonzero(np.isfinite(distance))
        points = np.column_stack((x, y, distance[y, x])).astype(np.float32)
        labels = []
        for annotation in export_2d.get_2d_boxes(cam, ['1', '2', '3', '4']):
            category = class_id(annotation['category_name'])
            if category is None:
                continue
            x1, y1, x2, y2 = annotation['bbox_corners']
            box = [x1 * sx + offset_x, y1 * sy + offset_y, x2 * sx + offset_x, y2 * sy + offset_y]
            labels.append([category, *map(float, box)])
            counts[f'{split}:{CLASSES[category]}'] += 1
        image_name, point_name = f'{cam}.jpg', f'{cam}.npy'
        cv2.imwrite(str(args.out / image_name), canvas)
        np.save(args.out / point_name, points)
        rows[split].append({'image': image_name, 'radar': point_name, 'labels': labels, 'scene': scene, 'sample': sample['token']})
        if split == 'val' and labels and points.size and not (args.out / 'demo.jpg').exists():
            for category, x1, y1, x2, y2 in labels:
                cv2.rectangle(canvas, (round(x1), round(y1)), (round(x2), round(y2)), (90, 240, 90), 1)
                cv2.putText(canvas, CLASSES[category], (round(x1), round(y1) - 3), cv2.FONT_HERSHEY_SIMPLEX, .35, (90, 240, 90), 1)
            for x, y, _ in points:
                cv2.circle(canvas, (int(x), int(y)), 2, (35, 90, 255), -1)
            cv2.imwrite(str(args.out / 'demo.jpg'), canvas)
    if any(not rows[split] for split in args.splits):
        raise ValueError('Every selected split requires at least one matched scene')
    manifest = {'version': args.version, 'size': args.size, 'classes': CLASSES, **rows}
    (args.out / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    summary = {'version': args.version, 'size': args.size, 'labels': dict(counts)}
    for split, records in rows.items():
        summary[split] = {'frames': len(records), 'scenes': sorted({row['scene'] for row in records}), 'empty_radar_frames': sum(not np.load(args.out / row['radar']).size for row in records)}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('data/nuscenes'))
    parser.add_argument('--out', type=Path, default=Path('data/prepared'))
    parser.add_argument('--version', default='v1.0-mini')
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--allow-subset', action='store_true')
    parser.add_argument('--splits', nargs='+', choices=('train', 'val'), default=['train', 'val'])
    parser.add_argument('--exclude-mini', action='store_true')
    prepare(parser.parse_args())
