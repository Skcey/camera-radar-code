"""Prepare the Astyx front-camera/radar pairs for an external car-only check."""
import argparse
from itertools import product
import json
from pathlib import Path

import cv2
import numpy as np
from pyquaternion import Quaternion


def to_camera(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def project(points, camera):
    pixels = points @ camera.T
    return pixels[:, :2] / pixels[:, 2:3]


def car_box(obj, transform, camera, width, height):
    corners = np.asarray(list(product((-.5, .5), repeat=3))) * obj['dimension3d']
    rotation = Quaternion(obj['orientation_quat']).rotation_matrix
    corners = corners @ rotation.T + obj['center3d']
    camera_points = to_camera(corners, transform)
    visible = camera_points[camera_points[:, 2] > .01]
    if len(visible) < 3:
        return None
    hull = cv2.convexHull(project(visible, camera).astype(np.float32))
    image = np.array(((0, 0), (width, 0), (width, height), (0, height)), dtype=np.float32)
    area, intersection = cv2.intersectConvexConvex(hull, image)
    if area <= 0 or intersection is None:
        return None
    uv = intersection.reshape(-1, 2)
    return np.concatenate((uv.min(0), uv.max(0)))


def main(args):
    cv2.setNumThreads(0)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for filepath in sorted((args.raw / 'camera_front').glob('*.jpg')):
        token = filepath.stem
        image = cv2.imread(str(filepath))
        height, width = image.shape[:2]
        scale = args.size / max(width, height)
        new_w, new_h = round(width * scale), round(height * scale)
        sx, sy = new_w / width, new_h / height
        offset = np.array(((args.size - new_w) // 2, (args.size - new_h) // 2))
        calib = json.loads((args.raw / 'calibration' / f'{token}.json').read_text())['sensors']
        sensors = {record['sensor_uid']: record['calib_data'] for record in calib}
        camera = np.array(sensors['camera_front']['K'])
        to_cam = np.linalg.inv(sensors['camera_front']['T_to_ref_COS'])
        radar_transform = to_cam @ np.array(sensors['radar_6455']['T_to_ref_COS'])
        radar = np.loadtxt(args.raw / 'radar_6455' / f'{token}.txt', skiprows=2, ndmin=2)
        xyz = to_camera(radar[:, :3], radar_transform)
        xyz = xyz[(xyz[:, 2] >= 3) & (xyz[:, 2] <= 100)]
        xy = project(xyz, camera)
        visible = (xy[:, 0] > 1) & (xy[:, 0] < width - 1) & (xy[:, 1] > 1) & (xy[:, 1] < height - 1)
        points = np.column_stack((xy[visible] * (sx, sy) + offset, xyz[visible, 2], np.zeros((visible.sum(), 2)))).astype(np.float32)
        # Astyx exposes echo magnitude and radial velocity, not the two source-model units.
        labels = []
        objects = json.loads((args.raw / 'groundtruth_obj3d' / f'{token}.json').read_text())['objects']
        for obj in objects:
            if obj['classname'] != 'Car':
                continue
            box = car_box(obj, to_cam, camera, width, height)
            if box is not None:
                labels.append([0, *(box * [sx, sy, sx, sy] + np.tile(offset, 2)).tolist()])
        canvas = np.full((args.size, args.size, 3), 114, dtype=np.uint8)
        x, y = offset
        canvas[y:y+new_h, x:x+new_w] = cv2.resize(image, (new_w, new_h))
        cv2.imwrite(str(args.out / f'{token}.jpg'), canvas)
        np.save(args.out / f'{token}.npy', points)
        rows.append({'image': f'{token}.jpg', 'radar': f'{token}.npy', 'labels': labels, 'scene': 'astyx', 'sample': token, 'focal': [camera[0, 0]*sx, camera[1, 1]*sy]})
        if token == '000000':
            for _, x1, y1, x2, y2 in labels:
                cv2.rectangle(canvas, (round(x1), round(y1)), (round(x2), round(y2)), (44, 220, 90), 1)
            for px, py, *_ in points:
                cv2.circle(canvas, (round(px), round(py)), 1, (0, 105, 255), -1)
            cv2.imwrite(str(args.out / 'alignment_check.jpg'), canvas)
    if len(rows) != 546:
        raise ValueError(f'Expected 546 matched Astyx records, got {len(rows)}')
    (args.out / 'manifest.json').write_text(json.dumps({'version': 'Astyx HiRes2019', 'classes': ['car', 'pedestrian', 'bicycle'], 'size': args.size, 'train': [], 'val': rows}))
    summary = {'frames': len(rows), 'car_boxes': sum(len(row['labels']) for row in rows), 'missing_radar_frames': sum(not np.load(args.out / row['radar']).size for row in rows)}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=Path('data/expansion/astyx/radar_dataset_astyx-main/dataset_astyx_hires2019'))
    parser.add_argument('--out', type=Path, default=Path('data/expansion/astyx_prepared'))
    parser.add_argument('--size', type=int, default=640)
    main(parser.parse_args())
