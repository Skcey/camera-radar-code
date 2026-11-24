"""Retest the published RRPN anchor stage with the project's frozen 2D detector."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from experiment import Detector, Frames, ground_truth
from radar_rescue import RadarMetadata, collect, fit, measure, refine, select


SOURCE_REVISION = "c9f4c291e53e64383158542ba79bf99d1cecebdc"
BASE_SIZE = 128
RATIOS = np.array([0.5, 1.0, 2.0])
SCALES = np.array([2, 4, 8])
# Original generation order: center, top, left, right.
LAYOUT = np.array([[0, 0], [0, -.5], [-.5, 0], [.5, 0]])
WIDTHS = np.round((BASE_SIZE + 1) / np.sqrt(RATIOS))
ASPECT_WH = np.column_stack((WIDTHS, np.round(WIDTHS * RATIOS)))
ANCHOR_WH = (ASPECT_WH[:, None, :] * SCALES[None, :, None]).reshape(-1, 2)
IMAGE_WH = np.array([1600, 900])


def anchor_boxes(pixel_points):
    """Match the four layouts, inclusive pixel arithmetic and beta=8 of RRPN."""
    xy, depth = pixel_points[:, :2], pixel_points[:, 2]
    wh = ANCHOR_WH[None, None, :, :] * (8 / depth[:, None, None, None])
    centers = xy[:, None, None, :] + LAYOUT[None, :, None, :] * wh
    radii = (wh - 1) / 2
    return np.concatenate((centers - radii, centers + radii), axis=-1).reshape(len(xy), 36, 4)


def rrpn_proposals(points, focal, size):
    scale = np.round(IMAGE_WH * (size / IMAGE_WH.max())) / IMAGE_WH
    padding = (size - np.round(IMAGE_WH * scale)) // 2
    pixel_points = points[:, :3].copy()
    pixel_points[:, :2] = (pixel_points[:, :2] - padding) / scale
    source_boxes = anchor_boxes(pixel_points).reshape(-1, 4)
    pixel_bounds = np.tile(IMAGE_WH - 1, 2)
    boxes = source_boxes.clip(0, pixel_bounds) * np.tile(scale, 2) + np.tile(padding, 2)
    keep = (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
    repeated = np.repeat(points, 36, axis=0)[keep]
    boxes = boxes[keep]
    unclipped = source_boxes[keep] * np.tile(scale, 2) + np.tile(padding, 2)
    wh = unclipped[:, 2:] - unclipped[:, :2]
    physical_wh = wh * repeated[:, 2, None] / np.asarray(focal)
    shift = ((unclipped[:, 0] + unclipped[:, 2]) / 2 - repeated[:, 0]) / wh[:, 0]
    features = np.column_stack((
        np.ones(len(boxes)), repeated[:, 2] / 100, repeated[:, 3] / 45,
        repeated[:, 4] / 25, physical_wh[:, 0] / 4, physical_wh[:, 1] / 2, shift,
    ))
    return (
        torch.as_tensor(boxes, dtype=torch.float32),
        torch.as_tensor(features, dtype=torch.float32),
        torch.zeros(len(boxes), dtype=torch.long),
    )


def evaluate(args):
    cv2.setNumThreads(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    manifest = json.loads((args.data / "manifest.json").read_text())
    classes = manifest["classes"]
    metadata = RadarMetadata(args.raw, manifest["version"], manifest["size"])
    camera_images = [row for row in metadata.data.values()
                     if row["is_key_frame"] and "/CAM_FRONT/" in row["filename"]]
    if not camera_images or any((row["width"], row["height"]) != tuple(IMAGE_WH)
                                for row in camera_images):
        raise ValueError("RRPN coordinates here require nuScenes 1600×900 front images.")
    model = Detector("rgb", None, classes).to(args.device).eval()
    model.load_state_dict(torch.load(args.weights, map_location=args.device, weights_only=True))
    train_gt = ground_truth(Frames(args.data, manifest, "train"), classes)
    blur_levels = (0, 2, 4)
    records = {}
    for blur in blur_levels:
        frames = Frames(args.data, manifest, "train", blur=blur)
        records[blur] = collect(model, frames, metadata, args.device, rrpn_proposals)
        print("collected training blur", blur, len(records[blur]), flush=True)
    dev_scenes = {"scene-0757", "scene-1094"}
    development = {blur: [r for r in batch if r["scene"] in dev_scenes]
                   for blur, batch in records.items()}
    first_fit = [r for batch in records.values() for r in batch if r["scene"] not in dev_scenes]
    full_fit = [r for batch in records.values() for r in batch]
    head = fit(first_fit, "radar", args.device, args.seed)
    strength, sweep = select(head, development, "radar", train_gt, classes, args.device)
    print("development scale", strength, flush=True)
    head = fit(full_fit, "radar", args.device, args.seed)
    results = {}
    for name, data, raw in (("mini", args.data, args.raw),
                            ("nuscenes_unseen", args.external_data, args.external_raw)):
        if data is None:
            continue
        evaluation_manifest = json.loads((data / "manifest.json").read_text())
        if evaluation_manifest["classes"] != classes:
            raise ValueError(f"Class mismatch for {name}")
        evaluation_metadata = (metadata if name == "mini" else
                               RadarMetadata(raw, evaluation_manifest["version"], evaluation_manifest["size"]))
        frames = Frames(data, evaluation_manifest, "val", blur=4)
        truth = ground_truth(frames, classes)
        validation = collect(model, frames, evaluation_metadata, args.device, rrpn_proposals)
        output = refine(head, validation, "radar", args.device)
        results[name] = {
            "frames": len(validation),
            "box_count_mean": float(np.mean([len(r["radar"][0]) for r in validation])),
            "rgb": measure(validation, [None] * len(validation), 0, truth, classes),
            "rrpn_refined": measure(validation, output, strength, truth, classes),
        }
        print(name, results[name], flush=True)
    payload = {
        "seed": args.seed, "weights": str(args.weights), "blur": 4,
        "anchor_source_revision": SOURCE_REVISION, "train_blur_levels": blur_levels,
        "development_scenes": sorted(dev_scenes), "scale": strength, "evaluation": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    torch.save(head.state_dict(), args.out / "rrpn_refiner.pt")
    (args.out / "development_sweep.json").write_text(json.dumps(sweep, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--raw", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--external-data", type=Path)
    parser.add_argument("--external-raw", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if (args.external_data is None) != (args.external_raw is None):
        parser.error("--external-data and --external-raw must be supplied together")
    evaluate(args)
