"""Prepare only the 998 reviewed images, with an auditable scene-level split.

FireNet locations below were inspected in the two contact sheets in
reports/campus22_existing_data_split. They are visual scene inferences, NOT
publisher-certified capture sessions. Never reuse this mapping for unseen IDs.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

import campus22_schema as schema
from data_pipeline import UnionFind, duplicate_groups, split
sys.path.insert(0, str(schema.ROOT / 'training'))
from preflight import inspect_release

RANK = {'train': 0, 'val': 1, 'test': 2}
FIRE_SCENES = {
    'window_watermist': ('val', [0, 42, 54], 'Window/radiator setting; includes ambiguous nearby window scene.'),
    'yellow_lobby': ('val', [1, 40, 78], 'Same yellow wall, fire blanket, signs and extinguisher.'),
    'parquet_pair': ('test', [9, 100], 'Same two wall-mounted extinguishers over parquet.'),
    'library': ('test', [19, 35, 67], 'Library locations kept together conservatively.'),
    'grey_floor_corridors_gym': ('train', [4, 31, 39, 48, 50, 71, 82, 96, 121, 127, 135],
                                 'Grey-floor corridors and gym kept together; repeated corners 71/82/135 and gym 96/121.'),
    'portable_stands': ('train', [12, 94], 'Portable floor stands kept together despite different backgrounds.'),
    'doorframe_co2': ('train', [22, 26, 41], 'Narrow doorway settings kept together conservatively.'),
    'first_aid_watermist': ('train', [56, 63], 'First-aid/watermist locations kept together.'),
    'patterned_first_aid': ('train', [136], 'Patterned wallpaper and first-aid equipment.'),
}
PRIORITY = {'stairs', 'pothole', 'curb_step', 'glass_door', 'door', 'obstacle',
            'ramp', 'construction_barrier', 'trash_can', 'cone', 'person'}


def plan(source_rows):
    rows = deepcopy(source_rows)
    lookup = {f'firenet_{n:04d}': (scene, target, reason)
              for scene, (target, ids, reason) in FIRE_SCENES.items() for n in ids}
    fires = {r['id'] for r in rows if r['source'] == 'firenet'}
    if fires != set(lookup):
        raise ValueError('FireNet membership changed; inspect scenes before preparing another split')
    evidence = []
    for row in rows:
        row['source_group_before_scene_review'] = row['group']
        row['source_group_id_before_scene_review'] = row.get('group_id')
        row['resolved_split_before_scene_review'] = row['resolved_split']
        if row['source'] == 'firenet':
            scene, target, reason = lookup[row['id']]
            row['group'] = 'firenet:visual_scene:' + scene
            row['resolved_split'] = target
            row['group_basis'] = 'assistant_visual_scene_inference'
            row['session_known'] = False
            row['grouping_reason'] = reason
            row['grouping_limitations'] = 'Capture sessions are unknown; scene separation is inferred visually.'
            evidence.append({k: row[k] for k in ('id', 'sha256', 'group', 'resolved_split', 'grouping_reason')})
        elif row.get('source_video_info'):
            row['group'] = 'roadwork:video:' + row['source_video_info']['vid_id']
            row['group_basis'] = 'publisher_whole_video_conservative'

    # Preserve old transitive duplicate components as well as entire videos.
    uf = UnionFind(len(rows))
    for indexes in duplicate_groups(rows):
        for i in indexes[1:]:
            uf.union(indexes[0], i)
    old_groups = {}
    for i, row in enumerate(rows):
        if row['source'] != 'firenet' and row.get('group_id'):
            prior = old_groups.setdefault(row['group_id'], i)
            uf.union(prior, i)
    components = defaultdict(list)
    for i in range(len(rows)):
        components[uf.find(i)].append(i)
    for indexes in components.values():
        targets = {rows[i]['resolved_split'] for i in indexes}
        targets.update(rows[i].get('original_split') for i in indexes)
        target = max(targets & set(RANK), key=RANK.get)
        gid = 'reviewed-scene:' + min(rows[i]['id'] for i in indexes)
        for i in indexes:
            rows[i]['group'] = rows[i]['group_id'] = gid
            rows[i]['resolved_split'] = target
            rows[i]['split_resolution_policy'] = 'campus22-group-holdout-v1'

    classes = [{schema.CLASSES[b[0]] for b in schema.read_labels(r['label'])} for r in rows]
    counts = {s: Counter() for s in RANK}
    for row, names in zip(rows, classes):
        counts[row['resolved_split']].update(names)
    moves = []
    # Hold out whole reviewed components. Never move val/test back into train.
    desired = {'construction_barrier': 6, 'cone': 6}
    while any(counts['test'][name] < minimum for name, minimum in desired.items()):
        choices = []
        for indexes in components.values():
            source = rows[indexes[0]]['resolved_split']
            if source == 'test':
                continue
            change = Counter(name for i in indexes for name in classes[i])
            gain = sum(min(change[name], max(0, minimum-counts['test'][name]))
                       for name, minimum in desired.items())
            if not gain:
                continue
            if any(counts[source][name]-amount < (40 if source == 'train' and name in PRIORITY else 5)
                   for name, amount in change.items()):
                continue
            choices.append((source == 'train', len(indexes)/gain, rows[indexes[0]]['id'], indexes, change))
        if not choices:
            raise ValueError('Cannot hold out enough existing groups without depleting train/validation')
        _, _, _, indexes, change = min(choices, key=lambda x: x[:3])
        source = rows[indexes[0]]['resolved_split']
        counts[source].subtract(change)
        counts['test'].update(change)
        moves.append({'group': rows[indexes[0]]['group'], 'from': source, 'to': 'test',
                      'ids': [rows[i]['id'] for i in indexes]})
        for i in indexes:
            rows[i]['resolved_split'] = 'test'
    for row in rows:
        row['recommended_split'] = row['resolved_split']
    return rows, {'fire_scene_evidence': evidence, 'whole_group_holdouts': moves,
                  'images_per_class_by_split': {s: dict(c) for s, c in counts.items()},
                  'limitations': ['FireNet capture sessions are unavailable; grouping is inferred from visible scenes.',
                                 'Small pilot dataset; passing checks does not establish detection accuracy.']}


def prepare(manifest, output):
    manifest = Path(manifest); output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory')
    source_rows = schema.read_manifest(manifest)
    for row in source_rows:
        if row.get('taxonomy') != schema.TAXONOMY or row.get('review_status') != 'approved':
            raise ValueError('Only approved Campus22 rows may be prepared')
        if schema.sha(row['label']) != row.get('reviewed_label_sha256'):
            raise ValueError('Labels changed after review: ' + row['id'])
    rows, report = plan(source_rows)
    if not split(rows, output, schema=schema):
        raise ValueError('Split failed complete class coverage; inspect split_report.json')
    released = schema.read_manifest(output / 'manifest.jsonl')
    if {r['id'] for r in released} != {r['id'] for r in source_rows}:
        raise ValueError('Split omitted reviewed records; inspect split_report.json')
    strict = inspect_release(output / 'data.yaml', schema=schema)
    pilot = inspect_release(output / 'data.yaml', min_priority_images=40, schema=schema)
    report.update(created_at=datetime.now(timezone.utc).isoformat(),
                  source_manifest=str(manifest.resolve()), source_manifest_sha256=schema.sha(manifest),
                  released_manifest_sha256=schema.sha(output/'manifest.jsonl'),
                  images=len(released), boxes=sum(len(schema.read_labels(r['label'])) for r in released),
                  images_by_split=dict(Counter(r['split'] for r in released)),
                  pilot_minimum_priority_images=40, strict_preflight=strict, pilot_preflight=pilot,
                  training_started=False)
    schema.save_json(output / 'preparation_report.json', report)
    schema.save_json(output / 'preflight.json', pilot)
    if not pilot['ready']:
        raise ValueError('Pilot preflight failed: ' + '; '.join(pilot['errors']))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.manifest, args.output)
    print(json.dumps({k: result[k] for k in ('images', 'boxes', 'images_by_split', 'pilot_preflight')}, indent=2))
