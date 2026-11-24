import argparse
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.nms import non_max_suppression


class Frames(Dataset):
    def __init__(self, folder, manifest, split, blur=0, missing=0):
        self.folder = folder
        self.rows = manifest[split]
        self.size = manifest['size']
        self.augment = split == 'train'
        self.blur = blur
        # Nested sets share one permutation across models and corruption rates.
        order = np.random.default_rng(2026).permutation(len(self.rows))
        self.removed = set(order[:round(len(order) * missing)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        record = self.rows[index]
        image = cv2.imread(str(self.folder / record['image']))
        if self.blur:
            image = cv2.GaussianBlur(image, (0, 0), self.blur)
        x = np.zeros((5, self.size, self.size), np.float32)
        x[:3] = image[:, :, ::-1].transpose(2, 0, 1) / 255
        if index not in self.removed:
            radar = np.load(self.folder / record['radar'])
            col, row = radar[:, :2].astype(int).T
            x[3, row, col] = 1
            x[4, row, col] = np.minimum(radar[:, 2], 100) / 100
        labels = np.asarray(record['labels'], np.float32).reshape(-1, 5)
        if self.augment and random.random() < .5:
            x = x[:, :, ::-1].copy()
            labels[:, [1, 3]] = self.size - labels[:, [3, 1]]
        return torch.from_numpy(x), torch.from_numpy(labels), index


def collate(frames):
    images, targets, ids = zip(*frames)
    return torch.stack(images), targets, ids


class Detector(DetectionModel):
    def __init__(self, variant, checkpoint, classes):
        self.variant = variant
        super().__init__('yolo11n.yaml', ch=5 if variant == 'early' else 3, nc=len(classes), verbose=False)
        if checkpoint is not None:
            source = YOLO(str(checkpoint)).model.float()
            self.load(source, verbose=False)
            if variant == 'early':
                original = source.model[0].conv.weight
                stem = self.model[0].conv.weight
                with torch.no_grad():
                    stem[:, :3].copy_(original)
                    stem[:, 3:].zero_()
        self.args = DEFAULT_CFG
        self.names = dict(enumerate(classes))
        if variant in ('gated', 'add', 'fixed'):
            channels = self.model[4].cv2.conv.out_channels
            self.radar_encoder = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.SiLU(), nn.Conv2d(16, channels, 3, padding=1))
            nn.init.normal_(self.radar_encoder[-1].weight, std=.005)
            nn.init.zeros_(self.radar_encoder[-1].bias)
            if variant in ('gated', 'fixed'):
                self.gate = nn.Conv2d(channels * 2, 1, 1)
                nn.init.zeros_(self.gate.weight)
                nn.init.zeros_(self.gate.bias)
                if variant == 'fixed':
                    self.gate.requires_grad_(False)

    def predict(self, x, *args, **kwargs):
        if x.shape[1] != 5 or self.variant == 'early':
            return super().predict(x, *args, **kwargs)
        if self.variant == 'rgb':
            return super().predict(x[:, :3], *args, **kwargs)
        radar = x[:, 3:]
        present = radar[:, :1].amax(dim=(2, 3), keepdim=True).bool()

        def fuse(_module, _inputs, image_features):
            radar_features = self.radar_encoder(nn.functional.adaptive_max_pool2d(radar, image_features.shape[-2:]))
            if self.variant in ('gated', 'fixed'):
                radar_features = radar_features * self.gate(torch.cat((image_features, radar_features), 1)).sigmoid()
            return image_features + radar_features * present

        hook = self.model[4].register_forward_hook(fuse)
        try:
            return super().predict(x[:, :3], *args, **kwargs)
        finally:
            hook.remove()


def training_batch(images, targets, device):
    labels = torch.cat(targets).to(device)
    index = torch.repeat_interleave(torch.arange(len(targets)), torch.tensor([len(t) for t in targets])).to(device)
    lo, hi = labels[:, 1:3], labels[:, 3:5]
    boxes = torch.cat(((lo + hi) / 2, hi - lo), dim=1) / images.shape[-1]
    return {'img': images.to(device, non_blocking=True), 'batch_idx': index, 'cls': labels[:, :1], 'bboxes': boxes}


def ground_truth(dataset, classes):
    gt = COCO()
    images, annotations = [], []
    for index, row in enumerate(dataset.rows):
        images.append({'id': index, 'file_name': row['image'], 'width': dataset.size, 'height': dataset.size})
        for category, x1, y1, x2, y2 in row['labels']:
            w, h = x2 - x1, y2 - y1
            annotations.append({'id': len(annotations) + 1, 'image_id': index, 'category_id': category + 1, 'bbox': [x1, y1, w, h], 'area': w * h, 'iscrowd': 0})
    gt.dataset = {'info': {}, 'images': images, 'annotations': annotations, 'categories': [{'id': i + 1, 'name': name} for i, name in enumerate(classes)]}
    with redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


@torch.inference_mode()
def evaluate(model, loader, gt, device, classes, image_ids=None):
    model.eval()
    predictions = []
    for images, _, ids in loader:
        output = model(images.to(device, non_blocking=True))[0].cpu()
        detections = non_max_suppression(output, conf_thres=.001, iou_thres=.7, nc=len(classes), max_det=300, max_time_img=30.0)
        for image_id, boxes in zip(ids, detections):
            for x1, y1, x2, y2, confidence, category in boxes.cpu().tolist():
                predictions.append({'image_id': image_id, 'category_id': int(category) + 1, 'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': confidence})
    if not predictions:
        return {'ap': 0, 'ap50': 0, 'per_class': {name: 0 for name in classes}}
    with redirect_stdout(io.StringIO()):
        metrics = COCOeval(gt, gt.loadRes(predictions), 'bbox')
        if image_ids is not None:
            metrics.params.imgIds = image_ids
        metrics.evaluate()
        metrics.accumulate()
        metrics.summarize()
    per_class = {}
    for i, name in enumerate(classes):
        precision = metrics.eval['precision'][:, :, i, 0, -1]
        valid = precision[precision >= 0]
        per_class[name] = float(valid.mean()) if valid.size else None
    return {'ap': float(metrics.stats[0]), 'ap50': float(metrics.stats[1]), 'per_class': per_class}


def main(args):
    cv2.setNumThreads(0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    manifest = json.loads((args.data / 'manifest.json').read_text())
    classes = manifest['classes']
    train_set = Frames(args.data, manifest, 'train')
    conditions = {'clean': {}, 'missing25': {'missing': .25}, 'missing50': {'missing': .5}, 'missing100': {'missing': 1}, 'blur2': {'blur': 2}, 'blur4': {'blur': 4}}
    val = {name: Frames(args.data, manifest, 'val', **kwargs) for name, kwargs in conditions.items()}
    gt = ground_truth(val['clean'], classes)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0, generator=torch.Generator().manual_seed(args.seed))
    val_loaders = {name: DataLoader(dataset, batch_size=args.batch, collate_fn=collate, num_workers=args.workers, pin_memory=True) for name, dataset in val.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / 'model.pt'
    model = Detector(args.variant, None if args.eval_only else args.weights, classes).to(device)
    if args.eval_only:
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.0005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr / 20)
        scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
        with (args.out / 'training.jsonl').open('w') as log:
            for epoch in range(args.epochs):
                started = time.monotonic()
                model.train()
                total = 0.
                for batch_index, (images, targets, _) in enumerate(train_loader):
                    optimizer.zero_grad(set_to_none=True)
                    batch = training_batch(images, targets, device)
                    with torch.autocast(device.type, enabled=args.amp):
                        loss_terms, _ = model(batch)
                        loss = loss_terms.sum()
                    if not math.isfinite(loss.item()):
                        raise FloatingPointError(f'Nonfinite loss at epoch {epoch + 1}, batch {batch_index}')
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    total += loss.detach().item()
                    if epoch == 0 and batch_index == 0:
                        print(f'First training step complete for {args.variant}', flush=True)
                scheduler.step()
                entry = {'epoch': epoch + 1, 'loss_per_frame': total / len(train_set), 'seconds': time.monotonic() - started}
                print(json.dumps(entry), flush=True)
                log.write(json.dumps(entry) + '\n')
                log.flush()
        torch.save(model.state_dict(), checkpoint)
    summary = {'variant': args.variant, 'train_epochs': args.epochs, 'seed': args.seed, 'train_frames': len(train_set), 'validation_frames': len(val['clean']), 'parameters': sum(p.numel() for p in model.parameters()), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None, 'torch': torch.__version__, 'conditions': {}}
    for name, loader in val_loaders.items():
        summary['conditions'][name] = evaluate(model, loader, gt, device, classes)
        print(name, summary['conditions'][name], flush=True)
    (args.out / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=('rgb', 'early', 'gated', 'add', 'fixed'))
    parser.add_argument('--data', type=Path, default=Path('data/prepared'))
    parser.add_argument('--weights', type=Path, default=Path('weights/yolo11n.pt'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    main(parser.parse_args())
