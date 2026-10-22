"""Verify a reviewed release on disk before loading a GPU or starting training."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from data_pipeline import CLASSES,read_manifest,read_labels,sha,save_json,duplicate_groups,dedup_boxes

EXTENSIONS={'.jpg','.jpeg','.png','.webp','.bmp'}


def inspect_release(dataset,min_priority_images=50,schema=None):
    names_expected = schema.CLASSES if schema is not None else CLASSES
    parse_labels = schema.read_labels if schema is not None else read_labels
    deduplicate = schema.dedup_boxes if schema is not None else dedup_boxes
    path=Path(dataset).resolve();root=path.parent;errors=[];counts={s:Counter() for s in ('train','val','test')};sources=Counter()
    result={'dataset':str(path),'ready':False,'errors':errors,'minimum_priority_images':min_priority_images,
            'note':'Readiness verifies review evidence and split integrity, not detection accuracy.'}
    if not path.exists():errors.append('Missing data.yaml');return result
    try:
        data=yaml.safe_load(path.read_text(encoding='utf-8'))
        names=data.get('names')
        if names!=dict(enumerate(names_expected)) and names!=names_expected:errors.append(f'Class IDs/names must match the {len(names_expected)}-class contract')
        configured=Path(data.get('path','.'))
        if not configured.is_absolute():configured=(root/configured).resolve()
        if configured!=root:errors.append('Release YAML path does not match its manifest directory')
        status=json.loads((root/'split_report.json').read_text(encoding='utf-8'))
        if not status.get('training_ready'):errors.append('Dataset not released: complete human review and all-class grouped split required')
        rows=read_manifest(root/'manifest.jsonl')
    except (ValueError,OSError,KeyError,TypeError) as exc:
        errors.append(f'Invalid release files: {exc}');return result
    result['records']=len(rows)
    seen_paths=set();independent=defaultdict(set)
    for row in rows:
        rid=row.get('id','unknown')
        try:
            subset=row['split'];image=Path(row['image']).resolve();label=Path(row['label']).resolve()
            if subset not in counts:raise ValueError('Unknown split')
            if row.get('original_split') and row['original_split']!=subset:
                if not (row.get('resolved_split') == subset and
                        row.get('split_resolution_policy') == 'campus22-group-holdout-v1'):
                    raise ValueError('Original holdout moved without group-level resolution evidence')
            expected=root/'images'/subset
            if not image.is_relative_to(expected) or not label.is_relative_to(root/'labels'/subset):raise ValueError('Image/label outside declared split')
            if image in seen_paths:raise ValueError('Duplicate manifest path')
            seen_paths.add(image)
            if not (row.get('review_status')=='approved' and row.get('coverage_complete') and row.get('provenance_reviewed') and row.get('reviewer') and row.get('reviewed_at')):
                raise ValueError('Incomplete annotation/provenance review evidence')
            if sha(image)!=row.get('sha256') or sha(image)!=row.get('reviewed_image_sha256'):raise ValueError('Image differs from reviewed content')
            if sha(label)!=row.get('reviewed_label_sha256'):raise ValueError('Label differs from reviewed content')
            boxes=parse_labels(label)
            if len(deduplicate(boxes))!=len(boxes):raise ValueError('Repeated boxes must be reconciled before release')
            if not boxes and not row.get('verified_negative'):raise ValueError('Unverified negative')
            classes={b[0] for b in boxes};counts[subset].update(names_expected[c] for c in classes);sources[row['source']]+=1
            if subset=='train':
                for c in classes:independent[names_expected[c]].add(row.get('derived_from',row['sha256']))
        except (ValueError,OSError,KeyError) as exc:errors.append(f'{rid}: {exc}')
    for subset in counts:
        if data.get(subset)!=f'images/{subset}':errors.append(f'{subset}: expected fixed reviewed images/{subset} path')
        directory=root/'images'/subset
        disk={p.resolve() for p in directory.rglob('*') if p.suffix.lower() in EXTENSIONS}
        declared={Path(r['image']).resolve() for r in rows if r.get('split')==subset}
        if disk!=declared:errors.append(f'{subset}: manifest and image folder disagree ({len(disk^declared)} paths)')
        missing=[c for c in names_expected if counts[subset][c]==0]
        if missing:errors.append(f'{subset}: no reviewed examples of {", ".join(missing)}')
    try:
        for group in duplicate_groups(rows):
            if len({rows[i]['split'] for i in group})>1:
                errors.append('Duplicate/near-duplicate/session cluster crosses partitions: '+', '.join(rows[i]['id'] for i in group[:5]))
    except (KeyError,ValueError) as exc:errors.append(f'Invalid grouping metadata: {exc}')
    priority_names = ('stairs', 'pothole', 'curb_step', 'glass_door', 'door', 'obstacle', 'ramp', 'construction_barrier', 'trash_can', 'cone', 'person') if 'trash_can' in names_expected else ('stairs', 'trash_bin')
    for name in priority_names:
        if len(independent[name])<min_priority_images:errors.append(f'{name}: requires {min_priority_images} distinct reviewed train scenes, found {len(independent[name])}')
    result.update(images_per_class_by_split={s:dict(c) for s,c in counts.items()},
                  independent_train_images={c:len(independent[c]) for c in names_expected},sources=dict(sources),ready=not errors)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=inspect_release(a.data);save_json(a.output,r);print(json.dumps(r,ensure_ascii=False));sys.exit(0 if r['ready'] else 2)
