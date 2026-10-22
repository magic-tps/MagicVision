"""Acquire exact campus classes using the existing official Open Images cache.

New labels use the isolated campus schema. Source labels remain incomplete and
pending review; no file in an existing dataset or deployment is overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from campus_schema import CLASSES, ROOT, from_xyxy, read_labels, record, save_json, sha, write_labels, write_manifest
from expand_openimages import LICENSES, download, existing_records, read_rows, real_box

SOURCE_MAP = {
    'Person': 'person', 'Man': 'person', 'Woman': 'person', 'Boy': 'person', 'Girl': 'person',
    'Car': 'car', 'Motorcycle': 'motorcycle', 'Bus': 'bus', 'Bicycle': 'bicycle',
    'Stairs': 'stairs', 'Traffic sign': 'traffic_sign', 'Waste container': 'trash_bin',
    'Bench': 'obstacle', 'Fire hydrant': 'obstacle',
    'Door': 'door', 'Chair': 'chair', 'Table': 'table_desk', 'Desk': 'table_desk',
    'Backpack': 'backpack',
}
DEFAULT_QUOTAS = {'door': 500, 'chair': 500, 'table_desk': 500, 'backpack': 500}
UNSUPPORTED = {
    'ramp': 'No exact boxable category', 'curb_step': 'No exact boxable category',
    'glass_door': 'Door does not establish the material of a door',
    'cable': 'No exact boxable category', 'wet_floor': 'No exact boxable category',
    'construction_barrier': 'Fence does not establish a construction barrier',
    'handrail': 'No exact boxable category', 'elevator': 'No exact boxable category',
    'cone': 'No exact traffic cone boxable category',
    'fire_extinguisher': 'No exact boxable category',
}


def hierarchy_review_flags(attributes):
    names_by_class = defaultdict(set)
    for attribute in attributes:
        names_by_class[attribute['class']].add(attribute['source_class'])
    return ['source_hierarchy_cooccurrence_review_overlap'] if any(
        len(names) > 1 for names in names_by_class.values()) else []


def refresh_review_flags(output):
    """Refresh review metadata without downloading or changing images/labels."""
    output = Path(output)
    rows = list(read_rows(output / 'manifest.jsonl'))
    counts = Counter()
    for row in rows:
        row['flags'] = sorted(set(row.get('flags', []) + hierarchy_review_flags(row['attributes'])))
        counts.update(row['flags'])
    write_manifest(output / 'manifest.jsonl', rows)
    report_path = output / 'download_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    report['images_by_review_flag'] = dict(counts)
    save_json(report_path, report)
    print(json.dumps({'images': len(rows), 'images_by_review_flag': dict(counts)}, indent=2))
    return counts


def select_candidates(csv_path, names, quotas, excluded, seed=42):
    """Reservoir sample distinct source images, ignoring crowd and depiction boxes."""
    seen = {name: set() for name in quotas}
    candidates = {name: [] for name in quotas}
    rngs = {name: random.Random(f'{seed}:{name}') for name in quotas}
    limits = {name: max(goal + 100, int(goal * 1.4)) for name, goal in quotas.items() if goal}
    mids = {mid: SOURCE_MAP[name] for mid, name in names.items()
            if name in SOURCE_MAP and SOURCE_MAP[name] in limits}
    with Path(csv_path).open(encoding='utf-8', newline='') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        image_col, label_col, group_col, depiction_col = (
            header.index(name) for name in ('ImageID', 'LabelName', 'IsGroupOf', 'IsDepiction'))
        for row in reader:
            cls = mids.get(row[label_col])
            if cls is None or row[group_col] != '0' or row[depiction_col] != '0':
                continue
            image_id = row[image_col]
            if image_id in excluded or image_id in seen[cls]:
                continue
            seen[cls].add(image_id)
            items = candidates[cls]
            if len(items) < limits[cls]:
                items.append(image_id)
            else:
                slot = rngs[cls].randrange(len(seen[cls]))
                if slot < limits[cls]:
                    items[slot] = image_id
    for cls, items in candidates.items():
        rngs[cls].shuffle(items)
    return candidates, {cls: len(images) for cls, images in seen.items()}


def prepare_plan(cache, output, quotas, excluded_ids, exclusion_fingerprint, seed):
    marker = json.loads((cache / 'source.json').read_text(encoding='utf-8'))
    if marker.get('boxes') != 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv':
        raise ValueError('Require existing official Open Images TRAIN box metadata')
    fingerprints = {key: {'size': (cache / f'{key}.csv').stat().st_size,
                          'mtime_ns': (cache / f'{key}.csv').stat().st_mtime_ns}
                    for key in ('classes', 'boxes', 'metadata')}
    parameters = {'quotas': quotas, 'source_map': SOURCE_MAP, 'seed': seed,
                  'excluded': exclusion_fingerprint, 'cache_files': fingerprints,
                  'classes': CLASSES}
    plan_path = output / 'selection_plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        if plan['parameters'] != parameters:
            raise ValueError('Inputs changed; choose another output directory')
        return plan
    with (cache / 'classes.csv').open(encoding='utf-8', newline='') as handle:
        names = dict(csv.reader(handle))
    missing = set(SOURCE_MAP) - set(names.values())
    if missing:
        raise ValueError(f'Exact source categories missing: {sorted(missing)}')
    print('Indexing cached train boxes for exact campus categories...', flush=True)
    candidates, availability = select_candidates(cache / 'boxes.csv', names, quotas, excluded_ids, seed)
    selected = set().union(*(set(ids) for ids in candidates.values()))
    print(f'Collecting compatible source boxes for {len(selected)} candidate images...', flush=True)
    annotations = defaultdict(list)
    with (cache / 'boxes.csv').open(encoding='utf-8', newline='') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        image_col, label_col = header.index('ImageID'), header.index('LabelName')
        allowed_mids = {mid for mid, name in names.items() if name in SOURCE_MAP}
        for values in reader:
            if values[image_col] in selected and values[label_col] in allowed_mids:
                annotations[values[image_col]].append(dict(zip(header, values)))
    print('Reading cached image license and attribution metadata...', flush=True)
    metadata = {}
    with (cache / 'metadata.csv').open(encoding='utf-8', newline='') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        image_col = header.index('ImageID')
        for values in reader:
            if values[image_col] in selected:
                metadata[values[image_col]] = dict(zip(header, values))
    permitted = {image_id for image_id, info in metadata.items()
                 if info.get('License', '').rstrip('/') in LICENSES}
    plan = {'parameters': parameters, 'source_urls': marker, 'names': names,
            'available_new_images_per_class': availability,
            'candidates': {cls: [image_id for image_id in ids if image_id in permitted]
                           for cls, ids in candidates.items()},
            'annotations': dict(annotations), 'metadata': metadata,
            'rejected_missing_or_incompatible_license': sorted(selected - permitted),
            'metadata_sha256': {key: sha(cache / f'{key}.csv') for key in ('classes', 'boxes', 'metadata')}}
    save_json(plan_path, plan)
    return plan


def fetch_record(image_id, output, plan):
    info = plan['metadata'][image_id]
    if info.get('License', '').rstrip('/') not in LICENSES:
        raise ValueError('Missing compatible image license')
    image = output / 'images' / f'{image_id}.jpg'
    url = f'https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg'
    for attempt in range(3):
        try:
            download(url, image, 20 * 1024 * 1024)
            with Image.open(image) as im:
                im.verify()
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(0.3 * (attempt + 1))
    boxes, attributes, flags = [], [], []
    for annotation in plan['annotations'].get(image_id, []):
        name = plan['names'][annotation['LabelName']]
        cls = SOURCE_MAP[name]
        if not real_box(annotation):
            flags.append('group_or_depiction_omitted')
            continue
        boxes.append(from_xyxy(CLASSES.index(cls), float(annotation['XMin']), float(annotation['YMin']),
                               float(annotation['XMax']), float(annotation['YMax'])))
        attributes.append({'class': cls, 'source_class': name, 'source_mid': annotation['LabelName'],
                           'occluded': annotation['IsOccluded'] == '1',
                           'truncated': annotation['IsTruncated'] == '1',
                           'inside': annotation.get('IsInside') == '1'})
        if cls == 'door':
            flags.append('door_material_unresolved_review_glass_door')
        if cls == 'backpack':
            flags.append('backpack_ground_relation_not_annotated')
    if not boxes:
        raise ValueError('No compatible real-object boxes')
    flags.extend(hierarchy_review_flags(attributes))
    label = output / 'labels' / f'{image_id}.txt'
    write_labels(label, boxes)
    return record(image, label, 'openimages_campus_v4', source_image_id=image_id, source_split='train',
                  source_dataset_version='Open Images V6 train boxes', acquisition_batch=output.name,
                  download_url=url, license=info['License'].rstrip('/'), annotation_license='CC-BY-4.0',
                  license_verification='source_metadata_only_pending_provenance_review',
                  provenance_reviewed=False, author=info.get('Author', ''), title=info.get('Title', ''),
                  original_url=info.get('OriginalURL', ''), landing_url=info.get('OriginalLandingURL', ''),
                  source_rotation=info.get('Rotation', ''), attributes=attributes, flags=sorted(set(flags)),
                  review_reason='Review all 25 classes, source hierarchy and attribution. Door material and backpack floor placement are unresolved. Campus provenance is not established by these generic public photographs.')


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    quotas = dict(DEFAULT_QUOTAS)
    for item in args.quota or []:
        cls, count = item.split('=', 1)
        if cls not in quotas or int(count) < 0:
            raise ValueError(f'Unsupported quota: {item}')
        quotas[cls] = int(count)
    manifests = [path.resolve() for path in args.exclude_manifest]
    if output / 'manifest.jsonl' in manifests:
        raise ValueError('Cannot exclude own output manifest')
    excluded_ids, excluded_hashes = existing_records(manifests)
    fingerprint = hashlib.sha256('\n'.join(sorted(excluded_ids | excluded_hashes)).encode()).hexdigest()
    plan = prepare_plan(args.metadata_cache, output, quotas, excluded_ids, fingerprint, args.seed)
    manifest = output / 'manifest.jsonl'
    rows = list(read_rows(manifest)) if manifest.exists() else []
    accepted_hashes = excluded_hashes | {row['sha256'] for row in rows}
    attempted = {row['source_image_id'] for row in rows}
    previous = output / 'download_report.json'
    old_report = json.loads(previous.read_text(encoding='utf-8')) if previous.exists() else {}
    errors, duplicates = [], old_report.get('exact_duplicates_excluded', [])
    counts = Counter()
    for row in rows:
        counts.update({CLASSES[box[0]] for box in read_labels(row['label'])})
    started = time.time()
    round_number = 0
    while True:
        selected = set()
        for cls, goal in quotas.items():
            missing = max(0, goal - counts[cls])
            selected.update([image_id for image_id in plan['candidates'][cls] if image_id not in attempted][:missing])
        if not selected:
            break
        attempted.update(selected)
        round_number += 1
        print(f'Round {round_number}: downloading {len(selected)} new images with {args.workers} workers', flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(fetch_record, image_id, output, plan): image_id for image_id in sorted(selected)}
            for number, future in enumerate(as_completed(pending), 1):
                image_id = pending[future]
                try:
                    row = future.result()
                    if row['sha256'] in accepted_hashes:
                        duplicates.append({'source_image_id': image_id, 'sha256': row['sha256']})
                    else:
                        accepted_hashes.add(row['sha256'])
                        rows.append(row)
                        counts.update({CLASSES[box[0]] for box in read_labels(row['label'])})
                except Exception as error:
                    errors.append({'source_image_id': image_id, 'reason': f'{type(error).__name__}: {str(error)[:200]}'})
                if number % 100 == 0 or number == len(pending):
                    write_manifest(manifest, sorted(rows, key=lambda row: row['id']))
                    print(f'Accepted {len(rows)} unique new images; processed {number}/{len(pending)}; errors {len(errors)}', flush=True)
    for row in rows:
        row['flags'] = sorted(set(row.get('flags', []) + hierarchy_review_flags(row['attributes'])))
    write_manifest(manifest, sorted(rows, key=lambda row: row['id']))
    boxes = Counter()
    for row in rows:
        boxes.update(CLASSES[box[0]] for box in read_labels(row['label']))
    report = {'source': 'Open Images V6 train boxes',
              'reference': 'https://storage.googleapis.com/openimages/web/download_v7.html',
              'new_unique_images': len(rows), 'images_per_class': dict(counts), 'boxes_per_class': dict(boxes),
              'additional_quotas': quotas, 'remaining': {cls: max(0, goal - counts[cls]) for cls, goal in quotas.items()},
              'source_mapping': SOURCE_MAP, 'available_new_images_per_class': plan['available_new_images_per_class'],
              'excluded_source_ids_before_download': len(excluded_ids),
              'excluded_sha256_before_download': len(excluded_hashes),
              'exact_duplicates_excluded': duplicates, 'errors': errors,
              'license_rejected': plan['rejected_missing_or_incompatible_license'],
              'exclusion_manifests': [str(path) for path in manifests],
              'metadata_sha256': plan['metadata_sha256'], 'approved_for_training': 0,
              'annotation_origin': 'source human-verified bounding boxes; 25-class completeness not reviewed',
              'review_status': 'needs_review', 'coverage_complete': False,
              'unsupported_requested_classes': UNSUPPORTED,
              'source_license_note': 'Per-image metadata: CC-BY-2.0; annotations: CC-BY-4.0. Original author and links retained; individual provenance review pending.',
              'domain_note': 'Generic public photographs; not represented as captures inside the user university.',
              'seed': args.seed, 'elapsed_download_seconds': round(time.time() - started, 2)}
    save_json(output / 'download_report.json', report)
    print(json.dumps({key: report[key] for key in ('new_unique_images', 'images_per_class', 'remaining')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-cache', type=Path, default=ROOT / 'datasets/openimages_train/metadata')
    parser.add_argument('--output', type=Path, default=ROOT / 'datasets/campus_openimages_v4')
    parser.add_argument('--exclude-manifest', type=Path, action='append',
                        default=[ROOT / 'datasets/expanded_v3/manifest.jsonl'])
    parser.add_argument('--quota', action='append')
    parser.add_argument('--workers', type=int, choices=range(1, 13), default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--refresh-review-flags', action='store_true',
                        help='Update review metadata in completed output; no network requests')
    args = parser.parse_args()
    if args.refresh_review_flags:
        refresh_review_flags(args.output)
    else:
        run(args)
