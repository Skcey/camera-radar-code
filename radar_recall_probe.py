"""Measure class-agnostic recall of camera and independent radar proposals."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from ultralytics.utils.nms import non_max_suppression

from experiment import Detector, Frames, collate


def radar_anchors(points, focal, size):
    points = points[(points[:, 2] >= 3) & (points[:, 2] <= 100)]
    boxes = []
    for width, height in ((1.8, 1.6), (.65, 1.75), (1.4, 1.55), (3.4, 1.6)):
        w, h = width * focal[0] / points[:, 2], height * focal[1] / points[:, 2]
        for offset in (0., -.22, .22):
            x, y = points[:, 0] + offset * w, points[:, 1] - .15 * h
            boxes.extend(np.column_stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
    return torch.from_numpy(np.asarray(boxes, dtype=np.float32)).reshape(-1, 4).clamp(0, size)


@torch.inference_mode()
def probe(args):
    cv2.setNumThreads(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    manifest = json.loads((args.prepared / 'manifest.json').read_text())
    version = args.raw / manifest['version']
    sample_data = {row['token']: row for row in json.loads((version / 'sample_data.json').read_text())}
    calibration = {row['token']: row['camera_intrinsic'] for row in json.loads((version / 'calibrated_sensor.json').read_text())}
    dataset = Frames(args.prepared, manifest, 'val', blur=args.blur)
    model = Detector('rgb', None, manifest['classes']).to(args.device).eval()
    model.load_state_dict(torch.load(args.weights, map_location=args.device, weights_only=True))
    loader = DataLoader(dataset, batch_size=16, collate_fn=collate, num_workers=2, pin_memory=True)
    groups = ('rgb100', 'rgb300', 'rgb300_low_threshold', 'rgb100_radar200', 'rgb300_radar_all')
    statistics = {group: {'proposals': [], 'iou': [], 'category': []} for group in groups}
    for images, labels, ids in loader:
        output = model(images.to(args.device, non_blocking=True))[0].cpu()
        predictions = non_max_suppression(output.clone(), nc=3, conf_thres=.001, iou_thres=.7, max_det=300, max_time_img=30)
        expanded = non_max_suppression(output, nc=3, conf_thres=.00001, iou_thres=.7, max_det=300, max_time_img=30)
        for image, target, predicted, low_score in zip(ids, labels, predictions, expanded):
            row = dataset.rows[image]
            cam = sample_data[Path(row['image']).stem]
            intrinsic = calibration[cam['calibrated_sensor_token']]
            scale = dataset.size / max(cam['width'], cam['height'])
            focal = (intrinsic[0][0] * scale, intrinsic[1][1] * scale)
            points = np.load(dataset.folder / row['radar'])
            radar = radar_anchors(points, focal, dataset.size)
            rgb = predicted[:, :4]
            proposals = (rgb[:100], rgb, low_score[:, :4], torch.cat((rgb[:100], radar[:200])), torch.cat((rgb, radar)))
            for group, boxes in zip(groups, proposals):
                record = statistics[group]
                record['proposals'].append(len(boxes))
                if len(target):
                    iou = box_iou(target[:, 1:], boxes).amax(dim=1) if len(boxes) else torch.zeros(len(target))
                    record['iou'].extend(iou.tolist())
                    record['category'].extend(target[:, 0].tolist())
    results = {}
    for key, record in statistics.items():
        iou, category = np.asarray(record['iou']), np.asarray(record['category'])
        subset = {'all': iou, **{name: iou[category == i] for i, name in enumerate(manifest['classes'])}}
        results[key] = {'mean_proposals': np.mean(record['proposals']), 'recall': {name: {str(threshold): float((values >= threshold).mean()) for threshold in (.50, .75)} for name, values in subset.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=Path('data/nuscenes'))
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    parser.add_argument('--weights', type=Path, default=Path('runs/rgb/model.pt'))
    parser.add_argument('--out', type=Path, default=Path('runs/radar_recall/proposal_recall.json'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--blur', type=float, default=0)
    probe(parser.parse_args())
