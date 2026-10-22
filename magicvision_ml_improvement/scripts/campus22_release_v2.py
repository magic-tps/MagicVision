"""Prepare the evidence-backed Campus22 v2 workset without inventing reviews.

This script may deduplicate byte-identical boxes inside one label and resolve
group-level split metadata. It never marks an annotation as human-approved.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import campus22_schema as schema
from data_pipeline import duplicate_groups, save_json, write_manifest
from training_path import ensure_imports  # noqa: F401


PRIORITY = [
    'stairs', 'pothole', 'curb_step', 'door', 'glass_door', 'ramp',
    'obstacle', 'construction_barrier', 'trash_can', 'cone',
    'person', 'car', 'motorcycle', 'bus', 'bicycle', 'pole',
    'traffic_sign', 'chair', 'table_desk', 'backpack', 'handrail',
    'fire_extinguisher',
]
PRIORITY_IDS = {schema.CLASSES.index(name): rank for rank, name in enumerate(PRIORITY)}
SPLIT_RANK = {'train': 0, 'val': 1, 'test': 2}
POLICY = 'campus22-group-holdout-v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def boxes_for(row):
    return row.get('_boxes') if '_boxes' in row else schema.read_labels(row['label'])


def component_id(rows, indexes):
    stable = '\n'.join(sorted(f"{rows[i]['id']}:{rows[i]['sha256']}" for i in indexes))
    return 'campus22-group-' + hashlib.sha256(stable.encode()).hexdigest()[:20]


def assign_groups(rows):
    """Assign each related component to one split, preferring held-out data."""
    components = duplicate_groups(rows)
    class_loads = {split: Counter() for split in SPLIT_RANK}
    image_loads = Counter()
    plans = []
    ratios = {'train': .7, 'val': .2, 'test': .1}
    prepared = []
    for indexes in components:
        classes = Counter(c for i in indexes for c in {b[0] for b in boxes_for(rows[i])})
        existing = sorted({rows[i].get('original_split') for i in indexes} & set(SPLIT_RANK),
                          key=SPLIT_RANK.get)
        prepared.append((indexes, classes, existing, component_id(rows, indexes)))
    for indexes, classes, existing, gid in sorted(prepared, key=lambda item: (-len(item[0]), item[3])):
        if existing:
            target = max(existing, key=SPLIT_RANK.get)
            reason = 'most_restrictive_source_holdout'
        else:
            target = min(
                ratios,
                key=lambda split: (
                    sum(class_loads[split][c] / ratios[split] for c in classes)
                    + image_loads[split] / ratios[split],
                    SPLIT_RANK[split],
                ),
            )
            reason = 'deterministic_group_balance'
        for index in indexes:
            rows[index]['group_id'] = gid
            rows[index]['resolved_split'] = target
            rows[index]['split_resolution_policy'] = POLICY
        image_loads[target] += len(indexes)
        class_loads[target].update(classes)
        plans.append({
            'group_id': gid,
            'record_ids': [rows[i]['id'] for i in indexes],
            'sources': sorted({rows[i]['source'] for i in indexes}),
            'original_splits': existing,
            'resolved_split': target,
            'reason': reason,
            'records': len(indexes),
            'conflict_before': len(existing) > 1,
            'leakage_after': False,
        })
    return plans


def classify_missing(row):
    required = {
        'image': row['image'], 'source': row['source'],
        'original_split': row.get('original_split'),
        'automatic_or_manual': 'manual', 'status': 'NEEDS_HUMAN_REVIEW',
    }
    image = Path(row['image']); label = Path(row['label'])
    if not image.exists() or not label.exists():
        return {**required, 'classification': 'D_FILE_ERROR', 'action': 'exclude_after_confirmation',
                'reason': 'Referenced image or label is missing'}
    if row.get('verified_negative') is True:
        return {**required, 'classification': 'A_VERIFIED_NEGATIVE', 'action': 'retain_as_negative',
                'reason': 'Existing explicit verified-negative evidence', 'automatic_or_manual': 'automatic',
                'status': 'RESOLVED'}
    source_label = row.get('source_25_label')
    if source_label and Path(source_label).exists():
        source_boxes = Path(source_label).read_text(encoding='utf-8-sig').strip()
        if source_boxes:
            return {**required, 'classification': 'B_SOURCE_ANNOTATION_AVAILABLE',
                    'action': 'recover_and_validate_source_annotation',
                    'reason': 'Non-empty source label exists and must be projected objectively',
                    'automatic_or_manual': 'automatic', 'status': 'RECOVERY_AVAILABLE'}
    if row.get('annotation_origin') == 'publisher_frame_classification_no_localization':
        reason = 'Publisher supplies frame-level class evidence but no bounding box coordinates'
    elif row.get('negative_candidate'):
        reason = 'Candidate negative lacks complete 22-class absence verification'
    else:
        reason = 'Empty label without objective localization or verified-negative evidence'
    return {**required, 'classification': 'C_POSSIBLE_OBJECT_WITHOUT_ANNOTATION',
            'action': 'review_boxes_or_mark_verified_negative', 'reason': reason}


def exact_duplicate_evidence(rows):
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row['sha256']].append(row)
    result = []
    for digest, members in sorted(by_hash.items()):
        if len(members) < 2:
            continue
        signatures = defaultdict(list)
        for row in members:
            signature = tuple(sorted(tuple(round(v, 8) for v in box) for box in boxes_for(row)))
            signatures[signature].append(row['id'])
        result.append({
            'sha256': digest, 'record_ids': [row['id'] for row in members],
            'same_annotation': len(signatures) == 1,
            'action': 'keep_one_logical_copy' if len(signatures) == 1 else 'NEEDS_HUMAN_RECONCILIATION',
            'annotation_variants': list(signatures.values()),
        })
    return result


def row_classes(row):
    ids = {box[0] for box in boxes_for(row)}
    ids.update(schema.CLASSES.index(name) for name in row.get('candidate_classes', [])
               if name in schema.CLASSES)
    return ids


def issue_for(row, duplicate_conflicts, conflict_groups):
    classes = row_classes(row)
    if not row['_boxes']:
        return 'missing_boxes_requires_visual_decision'
    if row['id'] in duplicate_conflicts:
        return 'exact_duplicate_has_different_annotations'
    if 13 in classes or 14 in classes:
        return 'door_glass_door_semantics_and_instance_geometry'
    if row['group_id'] in conflict_groups:
        return 'source_split_conflict_resolved_at_group_level; review_annotations'
    return 'complete_22_class_coverage_and_provenance_review'


def select_release_candidates(rows):
    """Select a compact reviewable subset with coverage in every planned split."""
    targets = {}
    critical = set(PRIORITY[:11])
    for split in SPLIT_RANK:
        for cid, name in enumerate(schema.CLASSES):
            if split == 'train':
                targets[(split, cid)] = 50 if name in critical else 30
            else:
                targets[(split, cid)] = 12 if name in critical else 8
    candidates = [row for row in rows if row['_boxes']]
    candidates.sort(key=lambda row: (
        min((PRIORITY_IDS[c] for c in row_classes(row)), default=len(PRIORITY)),
        row['resolved_split'], row['group_id'], row['id']))
    selected = []; selected_ids = set(); counts = Counter()
    for rank_name in PRIORITY:
        cid = schema.CLASSES.index(rank_name)
        for split in ('train', 'val', 'test'):
            need = targets[(split, cid)]
            for row in candidates:
                if counts[(split, cid)] >= need:
                    break
                if row['id'] in selected_ids or row['resolved_split'] != split or cid not in row_classes(row):
                    continue
                selected.append(row); selected_ids.add(row['id'])
                for present in row_classes(row):
                    counts[(split, present)] += 1
    return selected, counts, targets


def write_distribution(rows, path):
    counts = {(split, cid): [0, 0] for split in SPLIT_RANK for cid in range(len(schema.CLASSES))}
    for row in rows:
        boxes = row['_boxes']
        for cid in {box[0] for box in boxes}:
            counts[(row['resolved_split'], cid)][0] += 1
        for box in boxes:
            counts[(row['resolved_split'], box[0])][1] += 1
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['split', 'id', 'class', 'images', 'boxes'])
        for split in SPLIT_RANK:
            for cid, name in enumerate(schema.CLASSES):
                writer.writerow([split, cid, name, *counts[(split, cid)]])


def copy_work_labels(rows, output):
    labels = output / 'labels'; labels.mkdir(parents=True, exist_ok=True)
    corrected = []
    for index, row in enumerate(rows):
        original = row['_boxes']; clean = schema.dedup_boxes(original)
        destination = labels / f'{index:06d}.txt'; schema.write_labels(destination, clean)
        row['label'] = str(destination.resolve())
        row['_boxes'] = clean
        if len(clean) != len(original):
            row['automatic_correction'] = 'removed_exact_duplicate_box_geometry'
            corrected.append(row['id'])
    return corrected


def generate(output=None):
    output = Path(output or ROOT / 'datasets/campus22_work_v2')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty campus22 work version')
    active = json.loads((ROOT / 'configs/active_dataset.json').read_text(encoding='utf-8'))
    source_manifest = ROOT.parent / active['manifest']
    rows = schema.read_manifest(source_manifest)
    if any(row.get('taxonomy') != schema.TAXONOMY for row in rows):
        raise ValueError('Active manifest is not campus22')
    rows = [{**row, '_boxes': schema.read_labels(row['label'])} for row in rows]
    plans = assign_groups(rows)
    exact = exact_duplicate_evidence(rows)
    duplicate_conflicts = {rid for item in exact if not item['same_annotation'] for rid in item['record_ids']}
    conflict_groups = {plan['group_id'] for plan in plans if plan['conflict_before']}
    output.mkdir(parents=True)
    corrected = copy_work_labels(rows, output)
    for row in rows:
        row['review_priority'] = min((PRIORITY_IDS[c] for c in row_classes(row)), default=len(PRIORITY)) + 1
        row['problem_detected'] = issue_for(row, duplicate_conflicts, conflict_groups)
        row['recommended_split'] = row['resolved_split']
    def clean_rows(items):
        return [{key: value for key, value in row.items() if not key.startswith('_')} for row in items]
    write_manifest(output / 'manifest.jsonl', clean_rows(rows))
    shutil.copy2(ROOT / 'datasets/campus22_v1/class_mapping.json', output / 'class_mapping.json')
    missing = [{**classify_missing(row), 'id': row['id']} for row in rows
               if not row['_boxes']]
    selected, selected_counts, targets = select_release_candidates(rows)
    write_manifest(output / 'release_candidate_review_manifest.jsonl', clean_rows(selected))
    write_manifest(output / 'missing_boxes_review_manifest.jsonl',
                   clean_rows(sorted((row for row in rows if not row['_boxes']),
                                     key=lambda row: (row['review_priority'], row['id']))))
    save_json(ROOT / 'reports/campus22_missing_boxes_resolution.json', {
        'generated_at': now(), 'initial_empty_records': len(missing),
        'resolved_automatically': sum(item['status'] == 'RESOLVED' for item in missing),
        'recovery_available': sum(item['status'] == 'RECOVERY_AVAILABLE' for item in missing),
        'needs_human_review': sum(item['status'] == 'NEEDS_HUMAN_REVIEW' for item in missing),
        'records': missing,
    })
    conflicts = [plan for plan in plans if plan['conflict_before']]
    save_json(ROOT / 'reports/campus22_group_split_resolution.json', {
        'generated_at': now(), 'policy': POLICY, 'priority': ['test', 'val', 'train'],
        'conflicts_before': len(conflicts), 'conflicts_resolved_in_plan': len(conflicts),
        'leakage_after_planned_assignment': 0, 'note': 'Original split metadata is preserved.',
        'groups': conflicts,
    })
    save_json(ROOT / 'reports/campus22_exact_duplicates_resolution.json', {
        'generated_at': now(), 'groups': exact,
        'same_annotation_groups': sum(item['same_annotation'] for item in exact),
        'conflicting_annotation_groups': sum(not item['same_annotation'] for item in exact),
    })
    write_distribution(rows, output / 'distribution.csv')
    quality = {
        'taxonomy': schema.TAXONOMY, 'generated_at': now(), 'records': len(rows),
        'unique_images': len({row['sha256'] for row in rows}),
        'boxes': sum(len(row['_boxes']) for row in rows),
        'empty_records': len(missing), 'approved_complete_records': 0,
        'group_components': len(plans), 'split_conflicts_before': len(conflicts),
        'split_conflicts_after_plan': 0, 'automatic_duplicate_box_corrections': len(corrected),
        'exact_duplicate_image_groups': len(exact),
        'exact_duplicate_annotation_conflicts': sum(not item['same_annotation'] for item in exact),
        'release_candidate_review_records': len(selected), 'training_ready': False,
        'source_manifest_sha256': schema.sha(source_manifest),
        'work_manifest_sha256': schema.sha(output / 'manifest.jsonl'),
    }
    save_json(output / 'quality.json', quality)
    save_json(output / 'split_manifest.json', {
        'policy': POLICY, 'groups': plans,
        'candidate_targets': {f'{split}:{schema.CLASSES[cid]}': target
                              for (split, cid), target in targets.items()},
        'candidate_selected_counts': {f'{split}:{schema.CLASSES[cid]}': selected_counts[(split, cid)]
                                      for split in SPLIT_RANK for cid in range(len(schema.CLASSES))},
    })
    return quality


if __name__ == '__main__':
    # Importing training modules from scripts on Windows needs an explicit shim.
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'datasets/campus22_work_v2')
    args = parser.parse_args()
    print(json.dumps(generate(args.output), indent=2, ensure_ascii=False))
