"""Create a separate 22-class candidate inventory, retaining historical sources."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import json
import campus_schema as source
import campus22_schema as target
from campus_inventory import audit


def prepare(manifest, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory to preserve edited labels')
    rows = source.read_manifest(manifest)
    if any(row.get('taxonomy') != source.TAXONOMY for row in rows):
        raise ValueError('Input must use the 25-class campus contract')
    output.mkdir(parents=True, exist_ok=True)
    retained = []; excluded = []; removed_boxes = 0
    for index, original in enumerate(rows):
        boxes = source.read_labels(original['label'])
        projected = target.project_boxes(boxes)
        candidates = [c for c in original.get('candidate_classes', []) if c not in target.EXCLUDED]
        removed_boxes += len(boxes)-len(projected)
        # Removed-only examples must not silently become negative training images.
        if not projected and not candidates and (boxes or original.get('candidate_classes')):
            excluded.append({'id':original['id'], 'sha256':original['sha256'], 'reason':'Only excluded categories annotated/candidate'})
            continue
        label = output/'labels'/f'{index:06d}.txt'
        target.write_labels(label, projected)
        row = dict(original)
        row.update(label=str(label), taxonomy=target.TAXONOMY,
                   source_25_label=original['label'], source_25_label_sha256=source.sha(original['label']),
                   source_25_manifest=str(Path(manifest).resolve()), candidate_classes=candidates,
                   previous_review_status=original.get('review_status'), review_status='needs_review',
                   coverage_complete=False, provenance_reviewed=False, verified_negative=False,
                   review_reason='Review all 22 retained classes and resolve source groups before training')
        for key in ('reviewer','reviewed_at','reviewed_image_sha256','reviewed_label_sha256','split'):
            row.pop(key, None)
        retained.append(row)
    report = audit(retained, {r['sha256'] for r in rows}, verify_new_images=False, schema=target)
    if report['errors']:
        source.save_json(output/'quality.json', report)
        raise ValueError('Projected dataset contains invalid records')
    groups = source.duplicate_groups(retained)
    conflicts = [{'ids':[retained[i]['id'] for i in g], 'splits':sorted({retained[i].get('original_split') for i in g} & {'train','val','test'})}
                 for g in groups if len({retained[i].get('original_split') for i in g} & {'train','val','test'}) > 1]
    source.write_manifest(output/'manifest.jsonl', retained)
    source.save_json(output/'excluded_records.json', excluded)
    source.save_json(output/'source_split_conflicts.json', conflicts)
    source.save_json(output/'class_mapping.json', {'source_taxonomy':source.TAXONOMY,'taxonomy':target.TAXONOMY,
        'excluded':sorted(target.EXCLUDED),'old_to_new':target.OLD_TO_NEW,'names':target.CLASSES})
    (output/'classes.txt').write_text('\n'.join(target.CLASSES)+'\n', encoding='utf-8')
    report.update(taxonomy=target.TAXONOMY, created_at=datetime.now(timezone.utc).isoformat(),
                  source_manifest_sha256=source.sha(manifest), manifest_sha256=source.sha(output/'manifest.jsonl'),
                  excluded_records=len(excluded), excluded_unique_images=len({r['sha256'] for r in excluded}-{r['sha256'] for r in retained}),
                  removed_box_records=removed_boxes, training_ready=False,
                  source_split_conflict_groups=len(conflicts), duplicate_or_source_groups=sum(len(g)>1 for g in groups),
                  verification_scope='All projected labels and image existence; pixel validation inherited from V5 quality report')
    source.save_json(output/'quality.json', report)
    with (output/'distribution.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(report['classes'][0]));writer.writeheader();writer.writerows(report['classes'])
    print(json.dumps({k:v for k,v in report.items() if k not in {'classes','sources','box_origins','exact_duplicate_groups'}},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=source.ROOT/'datasets/expanded_campus_v5/manifest.jsonl')
    parser.add_argument('--output',type=Path,default=source.ROOT/'datasets/campus22_v1')
    args=parser.parse_args();prepare(args.manifest,args.output)
