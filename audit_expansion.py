"""Verify all added nuScenes scenes are complete, official validation and outside mini."""
import argparse
from collections import Counter
import json
from pathlib import Path

from nuscenes.utils.splits import create_splits_scenes


def main(args):
    manifest = json.loads((args.prepared / 'manifest.json').read_text())
    scene_table = json.loads((args.raw / 'v1.0-trainval/scene.json').read_text())
    expected = {item['name']: item['nbr_samples'] for item in scene_table}
    seen = Counter(row['scene'] for row in manifest['val'])
    splits = create_splits_scenes()
    mini = set(splits['mini_train'] + splits['mini_val'])
    if set(seen) & mini or not set(seen).issubset(splits['val']) or manifest['train']:
        raise ValueError('Expanded evaluation must contain only official validation scenes outside mini')
    if any(count != expected[scene] for scene, count in seen.items()):
        raise ValueError('Incomplete scene in the expanded validation set')
    audit = {'dataset': manifest['version'], 'scene_frames': dict(sorted(seen.items())), 'complete_scenes': len(seen), 'total_frames': sum(seen.values()), 'mini_scene_overlap': [], 'archive_parts': ['v1.0-trainval01_keyframes', 'v1.0-trainval02_keyframes']}
    args.out.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(audit['complete_scenes'], audit['total_frames'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
