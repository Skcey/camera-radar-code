"""Module ablations using archived camera weights and strong-blur validation."""
import argparse
import json
from pathlib import Path

import cv2
import torch

from experiment import Detector, Frames, ground_truth
from radar_rescue import RadarMetadata, Refiner, collect, fit, measure, refine, select
from validate_expansion import AstyxMetadata


DEV_SCENES = {"scene-0757", "scene-1094"}
# Feature order: visual ROI, xyxy, radar flag/depth/RCS/speed/width/height/shift, sharpness.
VARIANTS = {
    "no_echo": {"zero_features": slice(-7, -4)},
    "no_visual_roi": {"zero_features": slice(None, -12)},
    "no_sharpness": {"zero_features": slice(-1, None)},
    "no_regression": {"regress": False},
    "no_quality": {"quality_score": False},
    "radar_geometry_only": {"zero_features": slice(-8, -1)},
}
GRID = (0., .02, .04, .08, .16, .32, .64, 1.)


def validation_scale(name, head, development, train_gt, classes, args, options):
    grid = GRID
    if name == "no_quality":
        # Confidence alone is on a larger scale than confidence times quality squared.
        grid = (0., .00125, .0025, .005, .01, *GRID[1:])
    scale, sweep = select(head, development, "radar", train_gt, classes, args.device,
                          grid=grid, **options)
    return scale, sweep, grid


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(args):
    cv2.setNumThreads(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    manifest = load(args.data / "manifest.json")
    classes = manifest["classes"]
    model = Detector("rgb", None, classes).to(args.device).eval()
    model.load_state_dict(torch.load(args.weights, map_location=args.device, weights_only=True))
    metadata = RadarMetadata(args.raw, manifest["version"], manifest["size"])
    levels = (0, 2, 4)
    training = {
        blur: collect(model, Frames(args.data, manifest, "train", blur=blur), metadata, args.device)
        for blur in levels
    }
    development = {blur: [r for r in records if r["scene"] in DEV_SCENES]
                   for blur, records in training.items()}
    fit_scenes = [r for records in training.values() for r in records if r["scene"] not in DEV_SCENES]
    all_training = [r for records in training.values() for r in records]
    train_gt = ground_truth(Frames(args.data, manifest, "train"), classes)
    args.out.mkdir(parents=True, exist_ok=True)
    heads = {}
    archived = load(args.rescue / "metrics.json")
    for name, mode in (("full", "radar"), ("image_geometry_only", "image")):
        head = Refiner(training[4][0][mode][1].shape[1]).to(args.device).eval()
        head.load_state_dict(torch.load(args.rescue / f"{mode}_refiner.pt",
                                       map_location=args.device, weights_only=True))
        heads[name] = (head, mode, archived["selection"][mode]["scale"], {})
    payload = {"seed": args.seed, "blur": 4, "development_scenes": sorted(DEV_SCENES),
               "calibration_grid": GRID, "train_frames": len(training[4]), "variants": {}, "datasets": {}}
    for name, options in VARIANTS.items():
        print("fitting", name, flush=True)
        head = fit(fit_scenes, "radar", args.device, args.seed, **options)
        scale, sweep, grid = validation_scale(name, head, development, train_gt, classes, args, options)
        head = fit(all_training, "radar", args.device, args.seed, **options)
        heads[name] = (head, "radar", scale, options)
        payload["variants"][name] = {"scale": scale, "grid": grid, "development_sweep": sweep}
        print("selected", name, scale, flush=True)
    del training, development, fit_scenes, all_training

    for dataset, path, radar in (
        ("mini", args.data, metadata),
        ("nuscenes_unseen", args.unseen, None),
        ("astyx", args.astyx, AstyxMetadata(args.astyx)),
    ):
        data_manifest = load(path / "manifest.json")
        if radar is None:
            radar = RadarMetadata(args.unseen_raw, data_manifest["version"], data_manifest["size"])
        frames = Frames(path, data_manifest, "val", blur=4)
        evaluation_classes = ["car"] if dataset == "astyx" else classes
        annotations = ground_truth(frames, evaluation_classes)
        records = collect(model, frames, radar, args.device)
        scores = {}
        for name, (head, mode, scale, options) in heads.items():
            predictions = refine(head, records, mode, args.device, **options)
            scores[name] = measure(records, predictions, scale, annotations, evaluation_classes)
            print(dataset, name, [round(scores[name][m] * 100, 3)
                                  for m in ("ap", "ap50", "ap75")], flush=True)
        payload["datasets"][dataset] = {"frames": len(records), "classes": evaluation_classes, "scores": scores}
        result = args.out / "metrics.json"
        temporary = result.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(result)
        del records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    for name in ("weights", "rescue", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name, default in (
        ("data", "data/prepared"), ("raw", "data/nuscenes"),
        ("unseen", "data/expansion/nuscenes_val_prepared"),
        ("unseen-raw", "data/expansion/nuscenes"),
        ("astyx", "data/expansion/astyx_prepared"),
    ):
        parser.add_argument(f"--{name}", type=Path, default=Path(default))
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())
