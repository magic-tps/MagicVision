"""Acquire additional Open Images boxes without approving incomplete annotations.

Uses the existing train CSV cache, deterministic reservoir sampling and resumable
selection plans. All imported labels remain proposals for eleven-class review.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
import requests

from data_pipeline import CLASSES, ROOT, from_xyxy, record, save_json, sha, write_labels, write_manifest

_sessions=threading.local()


def download(url,destination,max_bytes):
    """Reuse one HTTP connection pool per worker; publish only complete files."""
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():return
    if not hasattr(_sessions,'http'):_sessions.http=requests.Session()
    partial=destination.with_suffix(destination.suffix+'.part')
    try:
        with _sessions.http.get(url,stream=True,timeout=(15,45)) as response:
            response.raise_for_status();total=0
            with partial.open('wb') as handle:
                for chunk in response.iter_content(128*1024):
                    total+=len(chunk)
                    if total>max_bytes:raise ValueError('Image exceeds size limit')
                    handle.write(chunk)
        partial.replace(destination)
    except Exception:
        if partial.exists():partial.unlink()
        raise

SOURCE_MAP = {
    'Person': 'person', 'Man': 'person', 'Woman': 'person', 'Boy': 'person', 'Girl': 'person',
    'Car': 'car', 'Motorcycle': 'motorcycle', 'Bus': 'bus', 'Bicycle': 'bicycle',
    'Stairs': 'stairs', 'Traffic sign': 'traffic_sign', 'Waste container': 'trash_bin',
    # Same explicit meanings as the legacy download_coco.py mapping.
    'Bench': 'obstacle', 'Fire hydrant': 'obstacle',
}
DEFAULT_QUOTAS = {'person': 800, 'car': 1000, 'motorcycle': 800, 'bus': 800,
                  'bicycle': 800, 'stairs': 800, 'traffic_sign': 1000,
                  'trash_bin': 800, 'obstacle': 800}
LICENSES = {'http://creativecommons.org/licenses/by/2.0',
            'https://creativecommons.org/licenses/by/2.0'}
REFERENCE = 'https://storage.googleapis.com/openimages/web/factsfigures_v7.html'


def read_rows(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def existing_records(manifests):
    ids, hashes = set(), set()
    for manifest in manifests:
        for row in read_rows(manifest):
            if row.get('source', '').startswith('openimages') and row.get('source_image_id'):
                ids.add(str(row['source_image_id']))
            if row.get('sha256'):
                hashes.add(row['sha256'])
    return ids, hashes


def real_box(row):
    return row.get('IsGroupOf') == '0' and row.get('IsDepiction') == '0'


def select_candidates(csv_path, names, quotas, excluded, seed=42):
    """Uniform deterministic reservoir over DISTINCT source images per class."""
    seen = {name: set() for name in quotas}
    reservoirs = {name: [] for name in quotas}
    rngs = {name: random.Random(f'{seed}:{name}') for name in quotas}
    limits = {name: max(goal + 100, int(goal * 1.3)) for name, goal in quotas.items() if goal}
    with Path(csv_path).open(encoding='utf-8', newline='') as handle:
        for number, row in enumerate(csv.DictReader(handle), 1):
            cls = SOURCE_MAP.get(names.get(row['LabelName']))
            image_id = row['ImageID']
            if cls not in limits or image_id in excluded or not real_box(row) or image_id in seen[cls]:
                continue
            seen[cls].add(image_id)
            items = reservoirs[cls]
            if len(items) < limits[cls]:
                items.append(image_id)
            else:
                slot = rngs[cls].randrange(len(seen[cls]))
                if slot < limits[cls]:
                    items[slot] = image_id
    for name, items in reservoirs.items():
        # Randomized but repeatable order also for small categories.
        rngs[name].shuffle(items)
    return reservoirs, {name: len(ids) for name, ids in seen.items()}


def prepare_plan(cache, output, quotas, excluded, exclusion_fingerprint, seed):
    marker = json.loads((cache / 'source.json').read_text())
    if marker.get('boxes') != 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv':
        raise ValueError('This expansion requires the official TRAIN box metadata cache')
    fingerprint = {key: {'size': (cache / (key + '.csv')).stat().st_size,
                        'mtime_ns': (cache / (key + '.csv')).stat().st_mtime_ns}
                   for key in ('classes', 'boxes', 'metadata')}
    params = {'quotas': quotas, 'seed': seed, 'excluded': exclusion_fingerprint,
              'source_map': SOURCE_MAP, 'cache_files': fingerprint}
    plan_path = output / 'selection_plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        if plan['parameters'] != params:
            raise ValueError('Selection inputs changed: choose a new output folder')
        return plan
    with (cache / 'classes.csv').open(encoding='utf-8', newline='') as handle:
        names = dict(csv.reader(handle))
    missing = set(SOURCE_MAP) - set(names.values())
    if missing:
        raise ValueError(f'Missing exact source categories: {sorted(missing)}')
    print('Selecting additional distinct image IDs from cached train boxes...', flush=True)
    candidates, availability = select_candidates(cache / 'boxes.csv', names, quotas, excluded, seed)
    selected = set().union(*map(set, candidates.values()))
    print(f'Collecting all compatible boxes for {len(selected)} candidates...', flush=True)
    annotations = defaultdict(list)
    with (cache / 'boxes.csv').open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['ImageID'] in selected and names.get(row['LabelName']) in SOURCE_MAP:
                annotations[row['ImageID']].append(row)
    metadata = {}
    with (cache / 'metadata.csv').open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['ImageID'] in selected:
                metadata[row['ImageID']] = row
    permitted = {image_id for image_id, info in metadata.items()
                 if info.get('License', '').rstrip('/') in LICENSES}
    plan = {'parameters': params, 'source_urls': marker, 'names': names,
            'available_new_images_per_class': availability,
            'candidates': {name: [i for i in ids if i in permitted] for name, ids in candidates.items()},
            'annotations': dict(annotations), 'metadata': metadata,
            'rejected_missing_or_incompatible_license': sorted(selected - permitted),
            'metadata_sha256': {key: sha(cache / (key + '.csv')) for key in ('classes', 'boxes', 'metadata')}}
    save_json(plan_path, plan)
    return plan


def fetch_record(image_id, output, plan):
    info = plan['metadata'][image_id]
    if info.get('License', '').rstrip('/') not in LICENSES:
        raise ValueError('Missing compatible per-image license')
    image = output / 'images' / (image_id + '.jpg')
    url = f'https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg'
    last = None
    for attempt in range(3):
        try:
            download(url, image, 20 * 1024 * 1024)
            with Image.open(image) as im:
                im.verify()
            break
        except Exception as error:
            last = error
            if attempt == 2:
                raise last
            time.sleep(0.3 * (attempt + 1))
    boxes, attrs, flags = [], [], []
    for ann in plan['annotations'].get(image_id, []):
        name = plan['names'][ann['LabelName']]
        cls = SOURCE_MAP[name]
        if not real_box(ann):
            flags.append('group_or_depiction_omitted')
            continue
        boxes.append(from_xyxy(CLASSES.index(cls), float(ann['XMin']), float(ann['YMin']),
                               float(ann['XMax']), float(ann['YMax'])))
        attrs.append({'class': cls, 'source_class': name, 'source_mid': ann['LabelName'],
                      'occluded': ann['IsOccluded'] == '1', 'truncated': ann['IsTruncated'] == '1',
                      'inside': ann.get('IsInside') == '1'})
    labels = output / 'labels' / (image_id + '.txt')
    write_labels(labels, boxes)
    return record(image, labels, 'openimages', source_image_id=image_id, source_split='train',
                  acquisition_batch=output.name, source_dataset_version='Open Images V6 train boxes',
                  download_url=url, license=info['License'].rstrip('/'), annotation_license='CC-BY-4.0',
                  license_verification='source_metadata_only_pending_provenance_review',
                  author=info.get('Author', ''), title=info.get('Title', ''),
                  original_url=info.get('OriginalURL', ''), landing_url=info.get('OriginalLandingURL', ''),
                  source_rotation=info.get('Rotation', ''), attributes=attrs, flags=sorted(set(flags)),
                  review_reason='Review all eleven classes and source hierarchy; boxes are not exhaustive across our taxonomy. Bench/hydrant are legacy obstacle subclasses; waste containers require definition review.')


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    quotas = dict(DEFAULT_QUOTAS)
    for item in args.quota or []:
        name, count = item.split('=', 1)
        if name not in DEFAULT_QUOTAS or int(count) < 0:
            raise ValueError(f'Unsupported quota: {item}')
        quotas[name] = int(count)
    manifests = args.exclude_manifest or sorted((ROOT / 'datasets').glob('*/manifest.jsonl'))
    manifests = [p.resolve() for p in manifests if p.resolve() != output / 'manifest.jsonl']
    legacy = ROOT / 'datasets/legacy_manifest.jsonl'
    if legacy.exists() and legacy.resolve() not in manifests:
        manifests.append(legacy.resolve())
    excluded_ids, excluded_hashes = existing_records(manifests)
    exclusion_fingerprint = hashlib.sha256('\n'.join(sorted(excluded_ids | excluded_hashes)).encode()).hexdigest()
    plan = prepare_plan(args.metadata_cache, output, quotas, excluded_ids, exclusion_fingerprint, args.seed)
    manifest = output / 'manifest.jsonl'
    rows = list(read_rows(manifest)) if manifest.exists() else []
    accepted_hashes = excluded_hashes | {r['sha256'] for r in rows}
    attempted = {r['source_image_id'] for r in rows}
    errors, duplicates = [], []
    counts = Counter()
    for row in rows:
        counts.update({a['class'] for a in row['attributes']})
    started = time.time()
    round_number = 0
    while True:
        selected = set()
        for cls, goal in quotas.items():
            missing = max(0, goal - counts[cls])
            selected.update([i for i in plan['candidates'][cls] if i not in attempted][:missing])
        if not selected:
            break
        attempted.update(selected)
        round_number += 1
        print(f'Round {round_number}: downloading {len(selected)} additional images with {args.workers} workers', flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(fetch_record, image_id, output, plan): image_id for image_id in sorted(selected)}
            for i, future in enumerate(as_completed(pending), 1):
                image_id = pending[future]
                try:
                    row = future.result()
                    if row['sha256'] in accepted_hashes:
                        duplicates.append({'source_image_id': image_id, 'sha256': row['sha256']})
                    else:
                        accepted_hashes.add(row['sha256'])
                        rows.append(row)
                        counts.update({a['class'] for a in row['attributes']})
                except Exception as error:
                    errors.append({'source_image_id': image_id, 'reason': f'{type(error).__name__}: {str(error)[:200]}'})
                if i % 100 == 0 or i == len(pending):
                    write_manifest(manifest, sorted(rows, key=lambda row: row['id']))
                    print(f'Accepted {len(rows)} unique new images; completed {i}/{len(selected)} in round; errors {len(errors)}', flush=True)
    write_manifest(manifest, sorted(rows, key=lambda row: row['id']))
    boxes = Counter()
    for row in rows:
        with Path(row['label']).open() as handle:
            boxes.update(CLASSES[int(line.split()[0])] for line in handle if line.strip())
    report = {'source': 'Open Images V6 TRAIN boxes; target split not assigned', 'reference': REFERENCE,
              'new_unique_images': len(rows), 'images_per_class': dict(counts), 'boxes_per_class': dict(boxes),
              'additional_quotas': quotas, 'remaining': {c: max(0, n - counts[c]) for c, n in quotas.items()},
              'source_mapping': SOURCE_MAP, 'available_new_images_per_class': plan['available_new_images_per_class'],
              'excluded_source_ids_before_download': len(excluded_ids), 'excluded_sha256_before_download': len(excluded_hashes),
              'exact_duplicates_excluded': duplicates, 'errors': errors,
              'license_rejected': plan['rejected_missing_or_incompatible_license'],
              'exclusion_manifests': [str(p) for p in manifests], 'metadata_sha256': plan['metadata_sha256'],
              'approved_for_training': 0, 'source_annotations_human_verified': True,
              'review_status': 'Eleven-class completeness and image provenance require review',
              'unsupported_classes': {'pothole': 'No exact boxable category', 'pole': 'No exact boxable category; Street light deliberately not mapped'},
              'source_license_note': 'Image metadata lists CC-BY-2.0; annotation license CC-BY-4.0. Google disclaims per-image license warranty; preserve and review original attribution.',
              'elapsed_download_seconds': round(time.time() - started, 2), 'seed': args.seed}
    save_json(output / 'download_report.json', report)
    print(json.dumps({k: report[k] for k in ('new_unique_images', 'images_per_class', 'remaining', 'approved_for_training')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-cache', type=Path, default=ROOT / 'datasets/openimages_train/metadata')
    parser.add_argument('--output', type=Path, default=ROOT / 'datasets/openimages_expansion_v3')
    parser.add_argument('--quota', action='append', help='class=count; override one default additional quota')
    parser.add_argument('--exclude-manifest', type=Path, action='append')
    parser.add_argument('--workers', type=int, choices=range(1, 13), default=12)
    parser.add_argument('--seed', type=int, default=42)
    run(parser.parse_args())
