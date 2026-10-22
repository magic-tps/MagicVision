"""Audit and combine acquisitions without approving labels or creating a split."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from PIL import Image
from campus_schema import (ROOT, CLASSES, TAXONOMY, read_labels, read_manifest,
                           write_manifest, save_json, sha, duplicate_groups)


def combine(manifests):
    rows = []; ids = set()
    for path in manifests:
        for source in read_manifest(path):
            row = dict(source)
            if row['id'] in ids:
                raise ValueError('Repeated record ID: '+row['id'])
            ids.add(row['id'])
            row['inventory_origin_manifest'] = str(Path(path).resolve())
            if row.get('taxonomy') != TAXONOMY:
                row['previous_review_status'] = row.get('review_status')
                row['previous_taxonomy'] = row.get('taxonomy', 'legacy-up-to-11')
                row['review_status'] = 'needs_review'
                row['coverage_complete'] = False
                row['review_reason'] = 'Recheck source provenance and missing annotations across all 25 campus classes'
            row['taxonomy'] = TAXONOMY
            rows.append(row)
    return rows


def audit(rows, baseline_hashes, verify_new_images=True, schema=None):
    names = schema.CLASSES if schema is not None else CLASSES
    parse_labels = schema.read_labels if schema is not None else read_labels
    classes = {name: {'id': i, 'image_hashes': set(), 'boxes': set(),
                    'new_hashes': set(), 'synthetic_hashes': set(), 'candidate_hashes': set()}
               for i, name in enumerate(names)}
    sources = {}; by_hash = defaultdict(list); errors = []; signatures = {}
    box_origins = Counter(); verified = 0; empty = 0; image_bytes = 0
    for row in rows:
        source = sources.setdefault(row['source'], {'records': 0, 'hashes': set(),
            'synthetic_hashes': set(), 'boxes': 0, 'unannotated_records': 0})
        source['records'] += 1; source['hashes'].add(row['sha256'])
        by_hash[row['sha256']].append(row)
        synthetic = bool(row.get('synthetic', False))
        if synthetic: source['synthetic_hashes'].add(row['sha256'])
        try:
            boxes = parse_labels(row['label'])
            signatures[row['id']] = tuple(sorted(tuple(round(v, 7) for v in b) for b in boxes))
            source['boxes'] += len(boxes)
            box_origins[row.get('label_origin', row.get('annotation_origin', 'source_labels_origin_not_explicit'))] += len(boxes)
            if not boxes:
                empty += 1; source['unannotated_records'] += 1
            for name in row.get('candidate_classes', []):
                if name not in classes: raise ValueError('Unknown candidate class '+name)
                classes[name]['candidate_hashes'].add(row['sha256'])
            for box in boxes:
                info = classes[names[box[0]]]
                info['image_hashes'].add(row['sha256'])
                # Identical labels on identical pixels are counted only once.
                info['boxes'].add((row['sha256'], *[round(v, 7) for v in box]))
                if row['sha256'] not in baseline_hashes: info['new_hashes'].add(row['sha256'])
                if synthetic: info['synthetic_hashes'].add(row['sha256'])
            path = Path(row['image'])
            if not path.is_file(): raise ValueError('Image missing')
            if len(by_hash[row['sha256']]) == 1: image_bytes += path.stat().st_size
            if verify_new_images and row['sha256'] not in baseline_hashes:
                if sha(path) != row['sha256']: raise ValueError('Image hash mismatch')
                with Image.open(path) as image:
                    image.load()
                    if image.size != (row['width'], row['height']): raise ValueError('Image dimensions mismatch')
                verified += 1
        except (ValueError, OSError, KeyError) as exc:
            errors.append({'id': row['id'], 'error': str(exc)})
    distribution = []
    for name, info in classes.items():
        distribution.append({'id': info['id'], 'class': name,
            'images_with_boxes': len(info['image_hashes']), 'boxes': len(info['boxes']),
            'new_images_with_boxes': len(info['new_hashes']),
            'synthetic_images_with_boxes': len(info['synthetic_hashes']),
            'real_images_with_boxes': len(info['image_hashes']-info['synthetic_hashes']),
            'unannotated_candidates': len(info['candidate_hashes']-info['image_hashes'])})
    source_report = {name: {k: (len(v) if isinstance(v, set) else v) for k,v in values.items()}
                     for name, values in sources.items()}
    duplicates = []
    for digest, variants in by_hash.items():
        if len(variants) < 2: continue
        labels = {signatures.get(r['id']) for r in variants}
        duplicates.append({'sha256': digest, 'ids': [r['id'] for r in variants],
            'label_conflict': len(labels)>1, 'source_splits': sorted({str(r.get('original_split')) for r in variants})})
    return {'records': len(rows), 'unique_images': len(by_hash),
        'new_unique_images': len(set(by_hash)-baseline_hashes),
        'synthetic_unique_images': len({r['sha256'] for r in rows if r.get('synthetic')}),
        'boxes_deduplicated_by_pixels_and_geometry': sum(r['boxes'] for r in distribution),
        'empty_label_records_not_necessarily_negatives': empty,
        'decoded_and_sha_verified_new_records': verified, 'unique_image_bytes': image_bytes,
        f'approved_complete_{len(names)}_class_records': sum(r.get('review_status')=='approved' and r.get('coverage_complete') is True for r in rows),
        'sources': source_report, 'classes': distribution, 'box_origins': dict(box_origins),
        'exact_duplicate_groups': duplicates, 'errors': errors}


def run(manifests, output, baseline, verify_new_images=True, report_path=None):
    rows = combine(manifests)
    old = read_manifest(baseline); hashes = {r['sha256'] for r in old}
    report = audit(rows, hashes, verify_new_images)
    report.update({'created_at': datetime.now(timezone.utc).isoformat(), 'taxonomy': TAXONOMY,
                   'baseline_manifest_sha256': sha(baseline), 'baseline_unique_images': len(hashes),
                   'input_manifests': [{'path':str(Path(p).resolve()), 'sha256':sha(p)} for p in manifests],
                   'baseline_manifest': str(Path(baseline).resolve()),
                   'verification_scope': ('All labels; new image hashes and decoding.' if verify_new_images else 'All labels; image existence only.') + ' Baseline pixels are not revalidated in this run; consult the baseline quality report.'})
    groups = duplicate_groups(rows)
    report['duplicate_or_source_groups'] = sum(len(g)>1 for g in groups)
    conflicts = []
    for group in groups:
        splits = {rows[i].get('original_split') for i in group} & {'train','val','test'}
        if len(splits)>1:
            conflicts.append({'ids':[rows[i]['id'] for i in group], 'splits':sorted(splits)})
    report['source_split_conflict_groups'] = len(conflicts)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    save_json(output/'quality.json', report)
    save_json(output/'source_split_conflicts.json', conflicts)
    with (output/'distribution.csv').open('w', encoding='utf-8', newline='') as handle:
        writer=csv.DictWriter(handle, fieldnames=list(report['classes'][0]));writer.writeheader();writer.writerows(report['classes'])
    if report['errors']:
        raise ValueError(f"{len(report['errors'])} invalid records; no manifest published; inspect quality.json")
    write_manifest(output/'manifest.jsonl', rows)
    (output/'classes.txt').write_text('\n'.join(CLASSES)+'\n', encoding='utf-8')
    report['manifest_sha256'] = sha(output/'manifest.jsonl')
    save_json(output/'quality.json', report)
    if report_path is not None:
        save_json(Path(report_path), report)
    print(json.dumps({k:v for k,v in report.items() if k not in {'classes','sources','input_manifests','box_origins','exact_duplicate_groups'}}, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifests', type=Path, nargs='+', required=True)
    parser.add_argument('--baseline', type=Path, default=ROOT/'datasets/expanded_v3/manifest.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT/'datasets/expanded_campus_v4')
    parser.add_argument('--report', type=Path, help='Optional additional report copy; quality.json is always saved in output')
    args=parser.parse_args();run(args.manifests,args.output,args.baseline,report_path=args.report)
