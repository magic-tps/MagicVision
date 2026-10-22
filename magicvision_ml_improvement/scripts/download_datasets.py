"""Download licensed Open Images subsets; preserve attribution, quarantine partial labels."""
from __future__ import annotations
import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
from PIL import Image
from data_pipeline import ROOT, CLASSES, OI_MAP, TARGETS, from_xyxy, record, write_labels, write_manifest, save_json, sha

URLS={
    'classes':'https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv',
    'boxes':'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv',
    'metadata':'https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv',
}


def download(url, dest, max_bytes):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():return
    temp=dest.with_suffix(dest.suffix+'.part')
    try:
        with requests.get(url,stream=True,timeout=(15,45)) as response:
            response.raise_for_status();total=0
            with temp.open('wb') as f:
                for block in response.iter_content(1024*128):
                    total+=len(block)
                    if total>max_bytes:raise ValueError('Download exceeds size budget')
                    f.write(block)
        temp.replace(dest)
    except Exception:
        if temp.exists():temp.unlink()
        raise


def run(output, people, stairs, bins, split='validation', negative_candidates=0, metadata_cache=None):
    output=Path(output);cache=Path(metadata_cache) if metadata_cache else output/'metadata'; metadata_checksums={}
    urls=dict(URLS)
    if split=='train':
        urls['boxes']='https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv'
        urls['metadata']='https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv'
    marker=cache/'source.json'
    if marker.exists() and json.loads(marker.read_text())!=urls:
        raise ValueError('Metadata cache belongs to a different source split')
    save_json(marker,urls)
    for key,url in urls.items():
        print('Metadata:',key,flush=True)
        dest=cache/(key+'.csv');download(url,dest,3*1024*1024*1024);metadata_checksums[key]=sha(dest)
    with (cache/'classes.csv').open(encoding='utf-8') as f:names=dict(csv.reader(f))
    desired={name:cid for cid,name in names.items() if name in OI_MAP}
    for name in ('Person','Stairs','Waste container'):
        if name not in desired:raise ValueError(f'Missing source class {name}')
    selected=set(); available=Counter();goals={'Stairs':stairs,'Waste container':bins,'Person':people}
    if negative_candidates:
        if 'Flowerpot' not in names.values():raise ValueError('Flowerpot category not found')
        goals['Flowerpot']=negative_candidates
    selected_per_class={name:set() for name in goals}
    seen_per_class={name:set() for name in goals}
    with (cache/'boxes.csv').open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            name=names.get(row['LabelName'])
            if name in goals and row['IsGroupOf']=='0' and row['IsDepiction']=='0':
                seen_per_class[name].add(row['ImageID'])
                if len(selected_per_class[name])<goals[name]:selected_per_class[name].add(row['ImageID'])
    for name,ids in selected_per_class.items():
        selected.update(ids);available[name]=len(seen_per_class[name])
    annotations=defaultdict(list)
    with (cache/'boxes.csv').open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['ImageID'] in selected and row['LabelName'] in desired.values():annotations[row['ImageID']].append(row)
    with (cache/'metadata.csv').open(encoding='utf-8') as f:metadata={r['ImageID']:r for r in csv.DictReader(f) if r['ImageID'] in selected}
    errors=[]; skipped=[]
    def fetch(id):
        info=metadata.get(id,{})
        license=info.get('License','').rstrip('/')
        if license not in ('https://creativecommons.org/licenses/by/2.0','http://creativecommons.org/licenses/by/2.0'):
            return None,{'id':id,'reason':'No compatible per-image license in source metadata'}
        try:
            image=output/'images'/(id+'.jpg');url=f'https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg'
            download(url,image,20*1024*1024)
            with Image.open(image) as im:im.verify()
            boxes=[];attrs=[];flags=[]
            for a in annotations[id]:
                label=OI_MAP[names[a['LabelName']]]
                if a['IsGroupOf']=='1' or a['IsDepiction']=='1':
                    flags.append('group_or_depiction_omitted');continue
                boxes.append(from_xyxy(CLASSES.index(label),float(a['XMin']),float(a['YMin']),float(a['XMax']),float(a['YMax'])))
                attrs.append({'class':label,'occluded':a['IsOccluded']=='1','truncated':a['IsTruncated']=='1'})
            labelpath=output/'labels'/(id+'.txt');write_labels(labelpath,boxes)
            row=record(image,labelpath,'openimages',source_image_id=id,source_split=split,download_url=url,
                license=license,annotation_license='CC-BY-4.0',author=info.get('Author',''),title=info.get('Title',''),
                original_url=info.get('OriginalURL',''),landing_url=info.get('OriginalLandingURL',''),
                source_rotation=info.get('Rotation',''),attributes=attrs,flags=flags,
                negative_candidate=id in selected_per_class.get('Flowerpot',set()),
                review_reason='Human source boxes; review eleven-class completeness, bin definition, stairs and occlusions. Source license metadata retained; provenance requires review.')
            return row,None
        except Exception as e:return None,{'id':id,'reason':type(e).__name__+': '+str(e)[:180]}
    rows=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,(row,error) in enumerate(pool.map(fetch,sorted(selected)),1):
            if row:rows.append(row)
            else:errors.append(error)
            if i%100==0:print(f'Downloaded {len(rows)}/{len(selected)}; errors {len(errors)}',flush=True)
    write_manifest(output/'manifest.jsonl',rows)
    counts=Counter()
    for row in rows:counts.update({a['class'] for a in row['attributes']})
    report={'source':f'Open Images {split} subset (not our final holdout)','license_reference':'https://storage.googleapis.com/openimages/web/factsfigures_v7.html',
            'available_source_images':available,'downloaded_unique_images':len(rows),'images_per_class':dict(counts),
            'targets':TARGETS,'remaining':{c:max(0,n-counts[c]) for c,n in TARGETS.items()},
            'approved_for_training':0,'errors':errors,'metadata_sha256':metadata_checksums,
            'negative_candidates_pending_review':sum(r['negative_candidate'] for r in rows),
            'license_caveat':'Open Images disclaims license warranty; preserve per-image attribution and review provenance. No source without CC-BY-2.0 metadata downloaded.'}
    save_json(output/'download_report.json',report);print(json.dumps({k:report[k] for k in ('downloaded_unique_images','images_per_class','remaining','approved_for_training')},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'datasets/openimages');p.add_argument('--people',type=int,default=1500);p.add_argument('--stairs',type=int,default=1200);p.add_argument('--bins',type=int,default=1000);p.add_argument('--split',choices=['validation','train'],default='validation');p.add_argument('--negative-candidates',type=int,default=0);p.add_argument('--metadata-cache',type=Path)
    a=p.parse_args();run(a.output,a.people,a.stairs,a.bins,a.split,a.negative_candidates,a.metadata_cache)
