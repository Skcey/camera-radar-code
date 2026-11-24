"""Independent radar proposals with frozen image features and a learned refinement head."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path

import cv2
import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pycocotools.cocoeval import COCOeval
from pyquaternion import Quaternion
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms, box_iou, roi_align
from ultralytics.utils.nms import non_max_suppression

from experiment import Detector, Frames, collate, ground_truth


class RadarMetadata:
    def __init__(self, raw, version, size):
        folder = raw / version

        def table(name):
            return {r['token']: r for r in json.loads((folder / f'{name}.json').read_text())}

        self.data = table('sample_data')
        self.calibration = table('calibrated_sensor')
        self.pose = table('ego_pose')
        self.radar = {record['sample_token']: record for record in self.data.values()
                      if record['is_key_frame'] and '/RADAR_FRONT/' in record['filename']}
        self.base, self.size = raw, size

    @staticmethod
    def transform(cloud, record, inverse=False):
        rotation = Quaternion(record['rotation']).rotation_matrix
        shift = np.asarray(record['translation'])
        if inverse:
            cloud.translate(-shift)
            cloud.rotate(rotation.T)
        else:
            cloud.rotate(rotation)
            cloud.translate(shift)

    def points(self, row):
        camera = self.data[Path(row['image']).stem]
        radar = self.radar[row['sample']]
        calib = self.calibration[camera['calibrated_sensor_token']]
        cloud = RadarPointCloud.from_file(str(self.base / radar['filename']))
        for record in (self.calibration[radar['calibrated_sensor_token']], self.pose[radar['ego_pose_token']]):
            self.transform(cloud, record)
        for record in (self.pose[camera['ego_pose_token']], calib):
            self.transform(cloud, record, inverse=True)
        xyz = cloud.points[:3]
        depth = xyz[2]
        keep = (depth >= 3) & (depth <= 100)
        intrinsic = np.asarray(calib['camera_intrinsic'])
        projected = view_points(xyz[:, keep], intrinsic, normalize=True)[:2].T
        width, height = camera['width'], camera['height']
        scale = self.size / max(width, height)
        sx, sy = round(width * scale) / width, round(height * scale) / height
        offset = ((self.size - round(width * scale)) // 2, (self.size - round(height * scale)) // 2)
        visible = (projected[:, 0] > 1) & (projected[:, 0] < width - 1) & (projected[:, 1] > 1) & (projected[:, 1] < height - 1)
        uv = projected[visible] * (sx, sy) + offset
        rcs = np.clip(cloud.points[5, keep][visible], -30, 45)
        speed = np.clip(np.hypot(*cloud.points[8:10, keep][:, visible]), 0, 25)
        return np.column_stack((uv, depth[keep][visible], rcs, speed)), (intrinsic[0, 0] * sx, intrinsic[1, 1] * sy)


def proposals(points, focal, size):
    boxes, metadata, priors = [], [], []
    shapes = ((1.8, 1.6, 0), (3.4, 1.6, 0), (.65, 1.75, 1), (1.4, 1.55, 2))
    for width, height, prior in shapes:
        w, h = width * focal[0] / points[:, 2], height * focal[1] / points[:, 2]
        for shift in (0., -.22, .22):
            x, y = points[:, 0] + shift * w, points[:, 1] - .15 * h
            boxes.extend(np.column_stack((x - w/2, y - h/2, x + w/2, y + h/2)))
            metadata.extend(np.column_stack((np.ones(len(points)), points[:, 2]/100, points[:, 3]/45, points[:, 4]/25, np.full(len(points), width/4), np.full(len(points), height/2), np.full(len(points), shift))))
            priors.extend([prior] * len(points))
    return (torch.tensor(np.asarray(boxes).reshape(-1, 4), dtype=torch.float).clamp(0, size),
            torch.tensor(np.asarray(metadata).reshape(-1, 7), dtype=torch.float), torch.tensor(priors, dtype=torch.long))


def offsets(anchors, gt):
    awh = (anchors[:, 2:] - anchors[:, :2]).clamp(min=1)
    gwh = (gt[:, 2:] - gt[:, :2]).clamp(min=1)
    shift = ((gt[:, :2] + gt[:, 2:]) - (anchors[:, :2] + anchors[:, 2:])) / (2 * awh)
    return torch.cat((shift, (gwh / awh).log()), 1)


def decode(anchors, deltas):
    size = (anchors[:, 2:] - anchors[:, :2]).clamp(min=1)
    center = (anchors[:, :2] + anchors[:, 2:]) / 2 + deltas[:, :2].clamp(-1, 1) * size
    radius = size * deltas[:, 2:].clamp(-1.2, 1.2).exp() / 2
    return torch.cat((center - radius, center + radius), 1)


def assignment(boxes, labels):
    if not len(labels) or not len(boxes):
        return torch.full((len(boxes),), 3), torch.zeros_like(boxes), torch.zeros(len(boxes))
    iou, match = box_iou(boxes, labels[:, 1:]).max(dim=1)
    category = torch.where(iou >= .35, labels[match, 0].long(), 3)
    return category, offsets(boxes, labels[match, 1:]), iou


def roi_features(image, feature, boxes, radar, sharpness):
    xyxy = boxes.to(feature.device)
    image_pool = roi_align(image, [xyxy], 3, aligned=True).flatten(1)
    feature_pool = roi_align(feature, [xyxy], 2, spatial_scale=.125, aligned=True).flatten(1)
    geometry = torch.cat((boxes / image.shape[-1], radar, torch.full((len(boxes), 1), sharpness)), dim=1)
    return torch.cat((feature_pool, image_pool, geometry.to(feature.device)), dim=1).half().cpu()


@torch.inference_mode()
def collect(model, data, metadata, device, proposal_generator=proposals):
    data.augment = False
    records = []
    features = []
    hook = model.model[4].register_forward_hook(lambda _m, _i, out: features.append(out))
    loader = DataLoader(data, batch_size=12, collate_fn=collate, num_workers=2, pin_memory=True)
    try:
        for images, labels, ids in loader:
            gpu = images.to(device, non_blocking=True)
            feature_output = model(gpu)[0].cpu()
            feature = features.pop()
            predictions = non_max_suppression(feature_output.clone(), conf_thres=.001, nc=3, iou_thres=.7, max_det=300, max_time_img=30)
            dense = non_max_suppression(feature_output, conf_thres=.00001, nc=3, iou_thres=.7, max_det=300, max_time_img=30)
            for j, (targets, index, camera, extra) in enumerate(zip(labels, ids, predictions, dense)):
                record = data.rows[index]
                radar_points, focal = metadata.points(record)
                boxes, radar, priors = proposal_generator(radar_points, focal, data.size)
                gray = images[j, :3].mean(0).numpy()
                sharpness = np.clip(np.log10(1e-7 + cv2.Laplacian(gray, cv2.CV_32F).var()) / 6, -1, 0)
                radar_features = roi_features(gpu[j:j+1, :3], feature[j:j+1], boxes, radar, float(sharpness))
                image_features = roi_features(gpu[j:j+1, :3], feature[j:j+1], extra[:, :4], torch.zeros((len(extra), 7)), float(sharpness))
                records.append({'id': index, 'scene': record['scene'], 'camera': camera, 'dense': extra, 'radar': (boxes, radar_features, priors), 'image': (extra[:, :4], image_features, extra[:, 5].long()), 'target': targets})
    finally:
        hook.remove()
    return records


class Refiner(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.features = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 128), nn.SiLU(), nn.Dropout(.15), nn.Linear(128, 64), nn.SiLU())
        self.category = nn.Linear(64, 4)
        self.delta = nn.Linear(64, 4)
        self.quality = nn.Linear(64, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, x):
        hidden = self.features(x)
        return self.category(hidden), self.delta(hidden), self.quality(hidden).flatten()


def fit(records, mode, device, seed, *, zero_features=None, regress=True, quality_score=True):
    samples = [record[mode] for record in records]
    targets = [assignment(sample[0], record['target']) for sample, record in zip(samples, records)]
    features = torch.cat([x[1] for x in samples]).to(device)
    if zero_features is not None:
        features[:, zero_features] = 0
    categories, deltas, iou = (torch.cat(x).to(device) for x in zip(*targets))
    torch.manual_seed(seed)
    head = Refiner(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.001)
    positive = torch.nonzero(categories < 3, as_tuple=True)[0]
    negative = torch.nonzero(categories == 3, as_tuple=True)[0]
    assert len(positive) and len(negative)
    counts = torch.bincount(categories[positive], minlength=3).clamp(min=1).float()
    positive_weights = counts[categories[positive]].rsqrt()
    negative_weights = 1 + 8 * iou[negative]
    for epoch in range(16):
        head.train()
        for _ in range(65):
            a = positive[torch.multinomial(positive_weights, 128, replacement=True)]
            b = negative[torch.multinomial(negative_weights, 256, replacement=True)]
            ids = torch.cat((a, b))
            cls, box, quality = head(features[ids].float())
            loss = F.cross_entropy(cls, categories[ids])
            if regress:
                loss = loss + 2 * F.smooth_l1_loss(box[:128], deltas[a], beta=.15)
            if quality_score:
                # Keep the quality target aligned with the box used for inference.
                normalized = torch.tensor([0., 0., 1., 1.], device=device).expand(len(a), 4)
                reference = decode(normalized, deltas[a])
                refined = decode(normalized, box[:128].detach()) if regress else normalized
                iou_refined = box_iou(refined, reference).diag()
                expected = torch.cat((iou_refined, torch.zeros(len(b), device=device)))
                loss = loss + F.binary_cross_entropy_with_logits(quality, expected)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch in (0, 7, 15):
            print(mode, epoch + 1, round(loss.item(), 4), len(positive), len(negative), flush=True)
    return head.eval()


@torch.inference_mode()
def refine(head, records, mode, device, *, zero_features=None, regress=True, quality_score=True):
    predictions = []
    for record in records:
        boxes, feature, _ = record[mode]
        features = feature.float().to(device)
        if zero_features is not None:
            features[:, zero_features] = 0
        cls, box, quality = head(features)
        confidence, category = cls.softmax(1)[:, :3].max(dim=1)
        output = (decode(boxes, box.cpu()) if regress else boxes).clamp(0, 640)
        score = (confidence * quality.sigmoid().square() if quality_score else confidence).cpu()
        predictions.append(torch.cat((output, score[:, None], category.cpu()[:, None]), 1))
    return predictions


def final_detections(camera, additional, scale):
    if scale <= 0 or not len(additional):
        return camera
    additional = additional.clone()
    additional[:, 4] *= scale
    combined = torch.cat((camera, additional[additional[:, 4] >= .00001]))
    keep = batched_nms(combined[:, :4], combined[:, 4], combined[:, 5], .7)
    return combined[keep[:300]]


def measure(records, extra, scale, gt, classes):
    items = []
    for record, detections in zip(records, extra):
        combined = final_detections(record['camera'], detections, scale)
        for x1, y1, x2, y2, score, category in combined.tolist():
            items.append({'image_id': record['id'], 'category_id': int(category) + 1, 'bbox': [x1, y1, x2-x1, y2-y1], 'score': score})
    with redirect_stdout(io.StringIO()):
        evaluation = COCOeval(gt, gt.loadRes(items), 'bbox')
        evaluation.params.imgIds = [r['id'] for r in records]
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    per_class = {}
    for i, name in enumerate(classes):
        values = evaluation.eval['precision'][:, :, i, 0, -1]
        valid = values[values >= 0]
        per_class[name] = float(valid.mean()) if valid.size else None
    return {**dict(zip(('ap', 'ap50', 'ap75', 'ar1', 'ar10', 'ar100'), evaluation.stats[[0, 1, 2, 6, 7, 8]].tolist())), 'per_class': per_class}


def select(head, dev, mode, train_gt, classes, device, *,
           grid=(0., .02, .04, .08, .16, .32, .64, 1.), **options):
    refined = {blur: refine(head, records, mode, device, **options) for blur, records in dev.items()}
    sweep = {scale: {str(blur): measure(records, refined[blur], scale, train_gt, classes)['ap'] for blur, records in dev.items()} for scale in grid}
    selected = max(grid, key=lambda scale: np.mean(list(sweep[scale].values())))
    return selected, sweep


def main(args):
    cv2.setNumThreads(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    manifest = json.loads((args.data / 'manifest.json').read_text())
    classes = manifest['classes']
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = RadarMetadata(args.raw, manifest['version'], manifest['size'])
    model = Detector('rgb', None, classes).to(args.device).eval()
    model.load_state_dict(torch.load(args.weights, map_location=args.device, weights_only=True))
    train_gt = ground_truth(Frames(args.data, manifest, 'train'), classes)
    test_gt = ground_truth(Frames(args.data, manifest, 'val'), classes)
    train = {blur: collect(model, Frames(args.data, manifest, 'train', blur=blur), metadata, args.device) for blur in (0, 2, 4)}
    test = {blur: collect(model, Frames(args.data, manifest, 'val', blur=blur), metadata, args.device) for blur in (0, 2, 4)}
    dev_scenes = {'scene-0757', 'scene-1094'}
    development = {b: [r for r in recs if r['scene'] in dev_scenes] for b, recs in train.items()}
    first_fit = [r for records in train.values() for r in records if r['scene'] not in dev_scenes]
    full_fit = [r for records in train.values() for r in records]
    details = {'seed': args.seed, 'weights': str(args.weights), 'development_scenes': sorted(dev_scenes), 'selection': {}, 'conditions': {}}
    heads = {}
    for mode in ('image', 'radar'):
        head = fit(first_fit, mode, args.device, args.seed)
        strength, sweep = select(head, development, mode, train_gt, classes, args.device)
        print(mode, 'development scale', strength, flush=True)
        heads[mode] = fit(full_fit, mode, args.device, args.seed)
        details['selection'][mode] = {'scale': strength, 'training_sweep': sweep}
        torch.save(heads[mode].state_dict(), args.out / f'{mode}_refiner.pt')
    for blur, records in test.items():
        output = {'rgb': measure(records, [None] * len(records), 0, test_gt, classes)}
        missing = set(np.random.default_rng(2026).permutation(len(records))[:round(len(records) / 2)])
        for name, mode in (('image_refined', 'image'), ('radar_refined', 'radar')):
            data = refine(heads[mode], records, mode, args.device)
            strength = details['selection'][mode]['scale']
            output[name] = measure(records, data, strength, test_gt, classes)
            if mode == 'radar':
                dropped = [item[:0] if index in missing else item for index, item in enumerate(data)]
                output['radar_missing50'] = measure(records, dropped, strength, test_gt, classes)
        simple = [torch.cat((r['radar'][0], torch.full((len(r['radar'][0]), 1), .01), r['radar'][2][:, None]), 1) for r in records]
        output['radar_simple'] = measure(records, simple, 1, test_gt, classes)
        details['conditions'][str(blur)] = output
        print(blur, {name: round(100 * rec['ap'], 3) for name, rec in output.items()}, flush=True)
    (args.out / 'metrics.json').write_text(json.dumps(details, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/prepared'))
    parser.add_argument('--raw', type=Path, default=Path('data/nuscenes'))
    parser.add_argument('--weights', type=Path, default=Path('runs/rgb/model.pt'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    main(parser.parse_args())
