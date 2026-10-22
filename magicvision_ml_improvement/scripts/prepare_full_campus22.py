"""Combine the historical and added images with corrections; never invent reviews."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import sys
import yaml

import campus22_schema as schema
from data_pipeline import duplicate_groups

RANK = {'train': 0, 'val': 1, 'test': 2}


def merge_rows(inventory, reviewed):
    originals = {r['id']: r for r in inventory}
    if len(originals) != len(inventory):
        raise ValueError('Repeated inventory IDs')
    overrides = {}
    for row in reviewed:
        if row['id'] in overrides or row['id'] not in originals:
            raise ValueError('Unknown or repeated reviewed ID')
        if row['sha256'] != originals[row['id']]['sha256']:
            raise ValueError('Reviewed image does not match inventory')
        if not (row.get('review_status') == 'approved' and row.get('coverage_complete')
                and row.get('provenance_reviewed') and row.get('reviewer')
                and row.get('reviewed_image_sha256') == row['sha256']
                and row.get('reviewed_label_sha256') == schema.sha(row['label'])):
            raise ValueError('Incomplete or outdated review: ' + row['id'])
        overrides[row['id']] = row
    merged = []
    for original in inventory:
        row = {**original, **overrides.get(original['id'], {})}
        row['annotation_usage'] = 'human_reviewed' if row['id'] in overrides else 'inherited_source_labels'
        row['input_image'] = row['image']
        row['input_label'] = row['label']
        merged.append(row)
    return merged


def select_records(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['sha256']].append(row)
    selected, excluded = [], []
    for same_image in grouped.values():
        reviewed = [r for r in same_image if r['annotation_usage'] == 'human_reviewed']
        candidates = reviewed or same_image
        signatures = {tuple(sorted(tuple(b) for b in schema.read_labels(r['label']))) for r in candidates}
        if len(signatures) > 1:
            excluded.extend({'id': r['id'], 'sha256': r['sha256'],
                             'reason': 'conflicting_annotations_on_identical_image'} for r in same_image)
            continue
        chosen = min(candidates, key=lambda r: r['id'])
        if not schema.read_labels(chosen['label']) and not chosen.get('verified_negative'):
            excluded.extend({'id': r['id'], 'sha256': r['sha256'],
                             'reason': 'empty_labels_without_negative_verification'} for r in same_image)
            continue
        selected.append(chosen)
        excluded.extend({'id': r['id'], 'sha256': r['sha256'], 'reason': 'duplicate_image',
                         'retained_id': chosen['id']} for r in same_image if r is not chosen)
    return selected, excluded


def assign_complete_groups(rows):
    # Keep full-inventory components, including unknown sessions, together.
    # Do NOT reuse a visual scene partition inspected only on 30 of 227 FireNet images.
    for row in rows:
        row['input_group'] = row['group']
        row['group'] = row.get('group_id', row['group'])
    components = duplicate_groups(rows)
    from data_pipeline import UnionFind
    uf = UnionFind(len(rows))
    for indexes in components:
        for i in indexes[1:]:
            uf.union(indexes[0], i)
    videos = {}
    for i, row in enumerate(rows):
        video = (row.get('source_video_info') or {}).get('vid_id')
        if video:
            uf.union(videos.setdefault(video, i), i)
    clusters = defaultdict(list)
    for i in range(len(rows)):
        clusters[uf.find(i)].append(i)
    for indexes in clusters.values():
        targets = {rows[i].get('resolved_split') for i in indexes}
        targets.update(rows[i].get('original_split') for i in indexes)
        target = max(targets & set(RANK), key=RANK.get)
        group = 'full-campus22:' + min(rows[i]['id'] for i in indexes)
        for i in indexes:
            rows[i]['group'] = rows[i]['group_id'] = group
            rows[i]['split'] = rows[i]['resolved_split'] = target
    # Reserve a few complete val groups for cone/barrier test coverage.
    counts = {s: Counter() for s in RANK}
    classes = [{schema.CLASSES[b[0]] for b in schema.read_labels(r['label'])} for r in rows]
    for row, names in zip(rows, classes):
        counts[row['split']].update(names)
    for wanted in ('construction_barrier', 'cone'):
        while counts['test'][wanted] < 6:
            options = []
            for indexes in clusters.values():
                if rows[indexes[0]]['split'] != 'val':
                    continue
                change = Counter(c for i in indexes for c in classes[i])
                if change[wanted] and all(counts['val'][c]-n >= 6 for c, n in change.items()):
                    options.append((len(indexes), rows[indexes[0]]['id'], indexes, change))
            if not options:
                break
            _, _, indexes, change = min(options, key=lambda item: item[:2])
            counts['val'].subtract(change)
            counts['test'].update(change)
            for i in indexes:
                rows[i]['split'] = rows[i]['resolved_split'] = 'test'


def prepare(manifest, reviewed_manifest, output, *, resume_copy=False):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        if not resume_copy or any(p.name not in {'images', 'labels'} for p in output.iterdir()):
            raise ValueError('Use a new empty version directory; only unfinished image/label copies can resume')
    inventory = schema.read_manifest(manifest)
    reviewed = schema.read_manifest(reviewed_manifest)
    rows = merge_rows(inventory, reviewed)
    if any(r.get('taxonomy') != schema.TAXONOMY for r in rows):
        raise ValueError('Expected 22-class labels')
    rows, excluded = select_records(rows)
    assign_complete_groups(rows)
    output.mkdir(parents=True, exist_ok=True)
    def materialize(row):
        if schema.sha(row['image']) != row['sha256']:
            raise ValueError('Image changed: ' + row['id'])
        name = row['sha256']
        image = output/'images'/row['split']/(name+Path(row['image']).suffix.lower())
        label = output/'labels'/row['split']/(name+'.txt')
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        if not image.exists() or schema.sha(image) != row['sha256']:
            shutil.copy2(row['image'], image)
        if not label.exists() or schema.sha(label) != schema.sha(row['label']):
            shutil.copy2(row['label'], label)
        row.update(image=str(image), label=str(label), input_label_sha256=schema.sha(label),
                   split_resolution_policy='campus22-group-holdout-v1')
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(materialize, rows), 1):
            if i % 2000 == 0:
                print(f'Preparadas {i}/{len(rows)} imagenes', flush=True)
    schema.write_manifest(output/'manifest.jsonl', rows)
    schema.write_manifest(output/'excluded.jsonl', excluded)
    (output/'data.yaml').write_text(yaml.safe_dump({
        'path': str(output), 'train': 'images/train', 'val': 'images/val', 'test': 'images/test',
        'names': dict(enumerate(schema.CLASSES))}, sort_keys=False), encoding='utf-8')
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'mode': 'all_existing_source_annotations_plus_human_corrections',
        'source_inventory': str(Path(manifest).resolve()), 'source_inventory_sha256': schema.sha(manifest),
        'reviewed_manifest': str(Path(reviewed_manifest).resolve()),
        'reviewed_manifest_sha256': schema.sha(reviewed_manifest),
        'manifest_sha256': schema.sha(output/'manifest.jsonl'),
        'yaml_sha256': schema.sha(output/'data.yaml'),
        'inventory_unique_images': len({r['sha256'] for r in inventory}),
        'images': len(rows), 'boxes': sum(len(schema.read_labels(r['label'])) for r in rows),
        'images_by_split': dict(Counter(r['split'] for r in rows)),
        'legacy_images': sum(r['source'] == 'legacy' for r in rows),
        'added_images': sum(r['source'] != 'legacy' for r in rows),
        'human_reviewed_images': sum(r['annotation_usage'] == 'human_reviewed' for r in rows),
        'inherited_annotation_images': sum(r['annotation_usage'] == 'inherited_source_labels' for r in rows),
        'excluded_unique_images': len({r['sha256'] for r in excluded}),
        'excluded_records_by_reason': dict(Counter(r['reason'] for r in excluded)),
        'limitations': [
            'Unreviewed inherited annotations can omit other Campus22 objects; they are not promoted to human-approved.',
            'FireNet unknown capture sessions remain one group; fire-extinguisher held-out performance cannot be certified.',
            'Training and metrics are experimental with inherited annotations, not a fully reviewed benchmark.'
        ],
        'training_started': False,
    }
    schema.save_json(output/'preparation_report.json', report)
    return report


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--reviewed-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-copy', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.reviewed_manifest, args.output, resume_copy=args.resume_copy), indent=2))
