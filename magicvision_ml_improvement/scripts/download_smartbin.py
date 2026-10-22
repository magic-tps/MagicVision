"""Import only the explicitly MIT-licensed legacy bin frames, never its crops/negatives."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
from PIL import Image
from data_pipeline import ROOT,record,read_labels,read_manifest,write_labels,write_manifest,save_json,sha
from expand_openimages import download

REPO='arudaev/smart-bin-detect'
REV='a3ff0d70b097d5bd4d8de3d850498d30b911d009'
BASE=f'https://huggingface.co/datasets/{REPO}/resolve/{REV}'


def map_labels(path):
    boxes=read_labels(path)
    if not boxes or any(b[0]!=0 for b in boxes):
        raise ValueError('Expected positive bin-only source labels')
    return [[10,*b[1:]] for b in boxes]


def fetch_frame(source,out):
    name=source['file']
    if Path(name).name!=name or ':' in name or '\\' in name:raise ValueError('Unsafe source filename')
    if source['licence']!='MIT':raise ValueError('Source license differs from the selected MIT subset')
    if source['legacy_split'] not in ('train','val','test'):raise ValueError('Unknown source split')
    image=out/'images'/name;source_label=out/'source_labels'/Path(name).with_suffix('.txt')
    label=out/'labels'/Path(name).with_suffix('.txt')
    image_url=f'{BASE}/legacy/images/{name}';label_url=f'{BASE}/legacy/labels/{source_label.name}'
    download(image_url,image,20*1024*1024);download(label_url,source_label,1024*1024)
    boxes=map_labels(source_label)
    if len(boxes)!=source['boxes']:raise ValueError('Source box count differs from manifest')
    with Image.open(image) as im:
        im.load()
        if im.size!=(source['width'],source['height']):raise ValueError('Source dimensions differ from manifest')
    write_labels(label,boxes)
    return record(image,label,'smartbin_legacy',group='smartbin:'+source['capture_cluster'],
        session_known=True,group_evidence='Publisher capture_cluster (100 clusters in 370 source frames)',
        original_split=source['legacy_split'],source_split=source['legacy_split'],source_image_id=name,
        source_url=f'https://huggingface.co/datasets/{REPO}',source_revision=REV,
        download_url=image_url,source_annotation_url=label_url,source_annotation=str(source_label.resolve()),
        source_annotation_sha256=sha(source_label),original_url=source['source_url'],
        original_archive_sha256=source['source_sha256'],license='MIT',
        author='arudaev/Painfully-Trivial contributors; annotator '+(source.get('annotator') or 'not supplied'),
        capture_date=source['capture_date'],label_origin=source['label_origin'],
        mapping={'bin':'trash_bin'},provenance_reviewed=False,
        review_reason='Public legacy bin boxes, one city/week. Check all eleven classes, geometry and provenance. No form-factor proposals imported. Preserve source holdouts and capture clusters.')


def run(out,exclude):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for source,target in [('README.md','README_source.md'),('legacy/manifest.json','legacy_manifest.json')]:
        download(f'{BASE}/{source}',out/'metadata'/target,10*1024*1024)
    manifest=json.loads((out/'metadata/legacy_manifest.json').read_text(encoding='utf-8'))
    if manifest['validator_classes']!=['bin']:raise ValueError('Source taxonomy changed')
    seen={r['sha256'] for path in exclude for r in read_manifest(path)}
    rows=[];errors=[];duplicates=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(fetch_frame,r,out):r['file'] for r in manifest['records']}
        for future in as_completed(futures):
            try:
                row=future.result()
                if row['sha256'] in seen:duplicates.append(row['source_image_id']);continue
                rows.append(row);seen.add(row['sha256'])
            except Exception as exc:errors.append({'file':futures[future],'error':str(exc)[:200]})
    rows.sort(key=lambda r:r['id']);write_manifest(out/'manifest.jsonl',rows)
    report={'source':REPO,'revision':REV,'license':'MIT legacy subset only','images':len(rows),
            'trash_bin_boxes':sum(len(read_labels(r['label'])) for r in rows),
            'capture_clusters':len({r['group'] for r in rows}),
            'by_source_split':dict(Counter(r['original_split'] for r in rows)),
            'errors':errors,'exact_duplicates_excluded':duplicates,'approved_for_training':0,
            'original_source_images':manifest['images'],'review_status':'needs_review'}
    save_json(out/'download_report.json',report);print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'datasets/smartbin_legacy_v3')
    p.add_argument('--exclude-manifest',type=Path,action='append',default=[])
    a=p.parse_args();run(a.output,a.exclude_manifest)
