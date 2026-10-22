"""Reproducible, review-gated data preparation. Never writes the original dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import yaml
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
APP = WORKSPACE / '01_app_magicvision_final/app_flutter_magicvision'
CLASSES = ['person', 'car', 'motorcycle', 'bus', 'bicycle', 'stairs', 'pothole', 'pole', 'traffic_sign', 'obstacle', 'trash_bin']
TARGETS = {'person': 1500, 'stairs': 1200, 'trash_bin': 1000, 'other_or_negative': 500}
OI_MAP = {'Person': 'person', 'Car': 'car', 'Motorcycle': 'motorcycle', 'Bus': 'bus', 'Bicycle': 'bicycle', 'Stairs': 'stairs', 'Traffic sign': 'traffic_sign', 'Waste container': 'trash_bin'}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_manifest(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_manifest(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def validate_box(box):
    if len(box) != 5:
        raise ValueError('YOLO box must have five values')
    cls, x, y, w, h = [float(v) for v in box]
    if not all(math.isfinite(v) for v in (cls, x, y, w, h)):
        raise ValueError('Nonfinite box')
    if not cls.is_integer() or not 0 <= cls < len(CLASSES):
        raise ValueError('Invalid class ID')
    if w <= 0 or h <= 0 or min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
        raise ValueError('Box outside image or empty')
    return [int(cls), x, y, w, h]


def from_xyxy(cls, left, top, right, bottom):
    return validate_box([cls, (left+right)/2, (top+bottom)/2, right-left, bottom-top])


def read_labels(path):
    return [validate_box(line.split()) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def dedup_boxes(boxes):
    result = []; seen = set()
    for box in boxes:
        box = validate_box(box)
        key = tuple(round(v, 6) for v in box)
        if key not in seen:
            seen.add(key); result.append(box)
    return result


def write_labels(path, boxes):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(f'{b[0]} ' + ' '.join(f'{v:.8f}' for v in b[1:]) + '\n' for b in dedup_boxes(boxes)), encoding='utf-8')


def record(image, label, source, group=None, **extra):
    with Image.open(image) as im:
        im.load(); width, height = im.size
        digest = str(imagehash.phash(ImageOps.exif_transpose(im).convert('RGB')))
    return {'id': f'{source}_{Path(image).stem}', 'image': str(Path(image).resolve()), 'label': str(Path(label).resolve()),
            'source': source, 'group': group or f'{source}:{Path(image).stem.split(".rf.")[0]}',
            'session_known': group is not None, 'width': width, 'height': height, 'sha256': sha(image), 'phash': digest,
            'review_status': 'needs_review', 'coverage_complete': False, 'verified_negative': False, **extra}


def baseline(output):
    dataset = APP / 'ml/datasets/dataset_final_magicvision'
    rows = []
    for split in ('train', 'val', 'test'):
        for image in sorted((dataset/'images'/split).iterdir()):
            if image.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
                continue
            label = dataset/'labels'/split/(image.stem+'.txt')
            rows.append(record(image, label, 'legacy', original_split=split,
                license_status='inherited_not_reverified', review_status='legacy10',
                review_reason='Review missing trash_bin annotations before eleven-class training'))
    write_manifest(output, rows)
    return len(rows)


def import_coco(annotation, images, mapping, output, source):
    data = json.loads(Path(annotation).read_text(encoding='utf-8'))
    names = {c['id']: c['name'] for c in data['categories']}
    anns = defaultdict(list)
    for ann in data['annotations']:
        anns[ann['image_id']].append(ann)
    rows = []; folder = Path(output).parent/'converted'/source
    for im in data['images']:
        boxes = []; flags = []
        for a in anns[im['id']]:
            name = names[a['category_id']]
            if name not in mapping:
                continue
            if a.get('iscrowd'):
                flags.append('crowd'); continue
            x,y,w,h = a['bbox']
            boxes.append(from_xyxy(CLASSES.index(mapping[name]),x/im['width'],y/im['height'],(x+w)/im['width'],(y+h)/im['height']))
        path = (Path(images)/im['file_name']).resolve()
        if not path.is_relative_to(Path(images).resolve()):
            raise ValueError('Image path escapes root')
        label = folder/(str(im['id'])+'.txt'); write_labels(label, boxes)
        rows.append(record(path,label,source,flags=flags,review_reason='Review all eleven classes and image license',
            source_image_id=im['id'], source_license_id=im.get('license'), source_url=im.get('flickr_url')))
    write_manifest(output, rows)


def import_yolo(images, labels, source_names, mapping, output, source):
    rows = []; folder=Path(output).parent/'converted'/source
    for image in sorted(Path(images).rglob('*')):
        if image.suffix.lower() not in {'.jpg','.jpeg','.png','.webp','.bmp'}:
            continue
        original=Path(labels)/image.relative_to(images).with_suffix('.txt')
        boxes=[]
        if original.exists():
            for line in original.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip(): continue
                values=line.split()
                index=float(values[0])
                if not index.is_integer() or not 0<=index<len(source_names): raise ValueError('Unknown source class ID')
                name=source_names[int(index)]
                if name in mapping:
                    boxes.append(validate_box([CLASSES.index(mapping[name]),*values[1:]]))
        label=folder/(sha(image)[:20]+'.txt');write_labels(label,boxes)
        rows.append(record(image,label,source,review_reason='Partial labels or missing annotations; verify all eleven classes and license'))
    write_manifest(output,rows)


def import_captures(folder, output):
    rows=[]
    for metadata in Path(folder).glob('*/metadata.json'):
        info=json.loads(metadata.read_text(encoding='utf-8'))
        if not info.get('consent_confirmed'): raise ValueError('Capture without consent flag')
        original=metadata.parent/'original.jpg'; sent=metadata.parent/'detector.jpg'
        if not original.exists() or not sent.exists(): raise ValueError('Incomplete capture pair')
        label=metadata.parent/'reviewed.txt'
        if not label.exists(): write_labels(label,[])
        session=info['session_id']; route=info['route_id']
        rows.append(record(sent,label,'campus',group=f'campus:{route}:{session}',original_image=str(original.resolve()),
            metadata=str(metadata.resolve()), review_reason='Manual annotation required; empty labels are NOT confirmed negatives'))
    write_manifest(output,rows)


class UnionFind:
    def __init__(self,n): self.parent=list(range(n))
    def find(self,x):
        while self.parent[x]!=x:
            self.parent[x]=self.parent[self.parent[x]];x=self.parent[x]
        return x
    def union(self,a,b): self.parent[self.find(b)]=self.find(a)


def duplicate_groups(rows, max_hamming=4):
    """Exact hashes + conservative pHash neighbours + collection groups, transitively."""
    uf=UnionFind(len(rows)); hashes={}; groups={}; bands=defaultdict(list)
    for i,row in enumerate(rows):
        for key,lookup in ((row['sha256'],hashes),(row['group'],groups)):
            if key in lookup: uf.union(i,lookup[key])
            else: lookup[key]=i
        value=int(row['phash'],16); candidates=set()
        # With distance <=4, at least one of five disjoint bands is unchanged.
        for band in range(5):
            key=(band,(value>>(band*13))&8191)
            candidates.update(bands[key]);bands[key].append(i)
        for j in candidates:
            if (value^int(rows[j]['phash'],16)).bit_count()<=max_hamming:
                uf.union(i,j)
    result=defaultdict(list)
    for i in range(len(rows)):result[uf.find(i)].append(i)
    return list(result.values())


def quality(rows, output, overlays=36):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    counts=Counter(); errors=[]; pending=[]; duplicates=[]; previews=[]; exact_labels={}; conflicts=[]
    for r in rows:
        try:
            with Image.open(r['image']) as im:
                im.load()
                if im.size!=(r['width'],r['height']):raise ValueError('Size changed')
            if sha(r['image'])!=r['sha256']:raise ValueError('Image changed')
            boxes=read_labels(r['label'])
            signature=tuple(sorted(tuple(round(v,6) for v in b) for b in boxes))
            previous=exact_labels.setdefault(r['sha256'],(r['id'],signature))
            if previous[1]!=signature:conflicts.append({'ids':[previous[0],r['id']],'reason':'same image, different annotations'})
            if len(dedup_boxes(boxes))!=len(boxes):duplicates.append(r['id'])
            counts.update(CLASSES[b[0]] for b in boxes)
            if not boxes and not r.get('verified_negative'):pending.append({'id':r['id'],'reason':'unverified empty label'})
            if r['review_status']!='approved' or not r.get('coverage_complete'):
                pending.append({'id':r['id'],'reason':r.get('review_reason','human review pending')})
            if len(previews)<overlays:
                with Image.open(r['image']) as im:
                    im=im.convert('RGB'); draw=ImageDraw.Draw(im)
                    for c,x,y,w,h in boxes:
                        coords=((x-w/2)*im.width,(y-h/2)*im.height,(x+w/2)*im.width,(y+h/2)*im.height)
                        draw.rectangle(coords,outline='red',width=3);draw.text(coords[:2],CLASSES[c],fill='red')
                    im.thumbnail((900,900));name=f'preview_{len(previews):03d}.jpg';im.save(out/name)
                    previews.append((r['id'],name))
        except (OSError,ValueError,KeyError) as e:errors.append({'id':r['id'],'error':str(e)})
    clusters=duplicate_groups(rows)
    duplicate_clusters=[[rows[i]['id'] for i in g] for g in clusters if len(g)>1]
    report={'images':len(rows),'annotations_by_class':dict(counts),'errors':errors,'duplicate_annotations':duplicates,
            'duplicate_or_session_clusters':duplicate_clusters,'review_queue':pending,
            'conflicting_exact_duplicates':conflicts,
            'approved_complete':sum(r['review_status']=='approved' and r.get('coverage_complete',False) for r in rows)}
    save_json(out/'quality.json',report)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Revisión MagicVision</title><h1>Cajas importadas: revisión humana pendiente</h1>'+''.join(f'<figure><img style="max-width:90%" src="{name}"><figcaption>{html.escape(id)}</figcaption></figure>' for id,name in previews),encoding='utf-8')
    return report


def split(rows, output, seed=42, schema=None):
    """Do not release partial annotations, duplicates or legacy-held-out images into train."""
    names = schema.CLASSES if schema is not None else CLASSES
    parse_labels = schema.read_labels if schema is not None else read_labels
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty version directory')
    clusters=duplicate_groups(rows); assignments={}; selected=[]; rejected=[]
    rng=random.Random(seed);rng.shuffle(clusters)
    total=sum(r['review_status']=='approved' and r.get('coverage_complete',False) for r in rows)
    loads=Counter(); classloads={s:Counter() for s in ('train','val','test')}
    ratios={'train':.7,'val':.2,'test':.1}
    for cluster in sorted(clusters,key=len,reverse=True):
        members=[rows[i] for i in cluster]
        eligible=[r for r in members if r['review_status']=='approved' and r.get('coverage_complete')]
        if not eligible: continue
        signatures=defaultdict(set)
        for r in eligible:
            signatures[r['sha256']].add(tuple(sorted(tuple(round(v,6) for v in b) for b in parse_labels(r['label']))))
        if any(len(values)>1 for values in signatures.values()):
            rejected.extend({'id':r['id'],'reason':'conflicting labels on exact duplicate; reconcile review first'} for r in eligible);continue
        resolved={r.get('resolved_split') for r in members}-{None}
        if resolved:
            if len(resolved)>1 or any(r.get('split_resolution_policy') != 'campus22-group-holdout-v1'
                                      for r in members):
                rejected.extend({'id':r['id'],'reason':'invalid group-level split resolution evidence'}
                                for r in eligible);continue
            target=next(iter(resolved))
        else:
            existing={r.get('original_split') for r in members}-{None}
            if len(existing)>1:
                rejected.extend({'id':r['id'],'reason':'cluster crosses original holdout splits'}
                                for r in eligible);continue
            if existing:
                target=next(iter(existing))
            else:
                classes=Counter(c for r in eligible for c in {b[0] for b in parse_labels(r['label'])})
                target=min(ratios,key=lambda s:sum(classloads[s][c]/ratios[s] for c in classes)+loads[s]/max(1,total*ratios[s]))
        seen=set()
        for r in eligible:
            if r['sha256'] in seen:continue
            if r.get('reviewed_image_sha256') and sha(r['image'])!=r['reviewed_image_sha256']:
                rejected.append({'id':r['id'],'reason':'image changed after review'});continue
            if r.get('reviewed_label_sha256') and sha(r['label'])!=r['reviewed_label_sha256']:
                rejected.append({'id':r['id'],'reason':'labels changed after review'});continue
            seen.add(r['sha256']); boxes=parse_labels(r['label'])
            if not boxes and not r.get('verified_negative'):
                rejected.append({'id':r['id'],'reason':'unverified negative'});continue
            name=r['source']+'_'+r['sha256'][:24]
            dst=out/'images'/target/(name+Path(r['image']).suffix.lower());dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(r['image'],dst)
            label_dst=out/'labels'/target/(name+'.txt');label_dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(r['label'],label_dst)
            selected.append({**r,'split':target,'image':str(dst.resolve()),'label':str((out/'labels'/target/(name+'.txt')).resolve())})
            loads[target]+=1;classloads[target].update({b[0] for b in boxes});assignments[r['id']]=target
    out.mkdir(parents=True,exist_ok=True)
    (out/'data.yaml').write_text(yaml.safe_dump({'path':str(out.resolve()),'train':'images/train','val':'images/val','test':'images/test','names':dict(enumerate(names))},sort_keys=False),encoding='utf-8')
    write_manifest(out/'manifest.jsonl',selected)
    missing={s:[names[i] for i in range(len(names)) if not classloads[s][i]] for s in ratios}
    ready=bool(selected) and not any(missing.values())
    save_json(out/'split_report.json',{'seed':seed,'ratios':ratios,'images':dict(loads),'missing_classes':missing,'rejected':rejected,'training_ready':ready})
    return ready


def preannotate(images, weights, output):
    from ultralytics import YOLO
    model=YOLO(str(weights));out=Path(output);rows=[]
    for result in model.predict(source=str(images),conf=.15,save=False,stream=True,verbose=False):
        boxes=[]
        for box in result.boxes:
            label=model.names[int(box.cls.item())]
            if label in CLASSES:boxes.append([CLASSES.index(label),*box.xywhn[0].tolist()])
        dest=out/'proposals'/(Path(result.path).stem+'.txt');write_labels(dest,boxes)
        rows.append(record(result.path,dest,'preannotation',review_reason='MODEL PROPOSALS ONLY; stairs, occlusions and missing classes need human review',model_sha256=sha(weights)))
    write_manifest(out/'review_queue.jsonl',rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('baseline');p.add_argument('--output',type=Path,default=ROOT/'datasets/legacy_manifest.jsonl')
    p=sub.add_parser('convert');p.add_argument('--format',choices=['coco','yolo'],required=True);p.add_argument('--annotations',required=True);p.add_argument('--images',required=True);p.add_argument('--mapping',required=True);p.add_argument('--source-names');p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('import-captures');p.add_argument('--folder',required=True);p.add_argument('--output',required=True)
    for command in ('quality','split'):
        p=sub.add_parser(command);p.add_argument('--manifest',nargs='+',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('preannotate');p.add_argument('--images',required=True);p.add_argument('--weights',required=True);p.add_argument('--output',required=True)
    a=parser.parse_args()
    if a.command=='baseline': print('Indexed:',baseline(a.output))
    elif a.command=='convert':
        mapping=json.loads(Path(a.mapping).read_text())
        if any(v not in CLASSES for v in mapping.values()):raise ValueError('Unknown target class')
        if a.format=='coco':import_coco(a.annotations,a.images,mapping,a.output,a.source)
        else:import_yolo(a.images,a.annotations,Path(a.source_names).read_text().splitlines(),mapping,a.output,a.source)
    elif a.command=='import-captures':import_captures(a.folder,a.output)
    elif a.command in ('quality','split'):
        rows=[r for path in a.manifest for r in read_manifest(path)]
        if a.command=='quality':
            r=quality(rows,a.output);print(json.dumps({k:r[k] for k in ['images','annotations_by_class','approved_complete']}))
        else:print('Training ready:',split(rows,a.output))
    else:preannotate(a.images,a.weights,a.output)


if __name__=='__main__':main()
