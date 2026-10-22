"""Acquire publisher annotations for campus hazards; never approves training data."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import re
from pathlib import Path
import tarfile
import time
import zipfile
from xml.etree import ElementTree as ET

import numpy as np
import requests
from PIL import Image
from campus_schema import CLASSES, ROOT, from_xyxy, record, read_labels, sha, save_json, write_labels, write_manifest
from http_range_zip import RangeReader

OUT = ROOT / 'datasets/campus_hazards_v4'
AUTO_REPO = 'autoexpert-cvpr2026-workshop/dataset-public'
AUTO_REV = '51f8e92ce458e12e43938d7e308d749c1900740a'
CABLE_REPO = 'pablomalvido/cables_segmentation'
CABLE_REV = '2f0d86585d7d2279f126c6da0b68d107a5073a91'
AUTO_MAP = {'Temporary_Construction_Barriers':20, 'Cones':23, 'Car':1,
            'Pedestrian':0, 'Bus':3, 'Motorcycle':2}


def fetch(url, path, limit=100_000_000):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    part = path.with_suffix(path.suffix + '.part')
    for attempt in range(3):
        try:
            with requests.get(url, timeout=(20, 90), stream=True) as response:
                response.raise_for_status()
                if int(response.headers.get('Content-Length', 0)) > limit:
                    raise ValueError('File exceeds declared download budget')
                with part.open('wb') as handle:
                    for chunk in response.iter_content(256*1024):
                        handle.write(chunk)
                        if handle.tell() > limit:
                            raise ValueError('Download exceeds budget')
            part.replace(path)
            return path
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def get_json(url, path):
    return json.loads(fetch(url, path).read_text(encoding='utf-8'))


def finish(directory, rows, **extra):
    write_manifest(directory/'manifest.jsonl', rows)
    boxes = Counter(); images = Counter()
    for row in rows:
        labels = read_labels(row['label'])
        boxes.update(CLASSES[b[0]] for b in labels)
        images.update(set(CLASSES[b[0]] for b in labels))
    result = {'images':len(rows), 'unique_images':len({r['sha256'] for r in rows}),
              'boxes_per_class':dict(boxes), 'images_per_class':dict(images),
              'training_approved':0, 'coverage_complete':False, **extra}
    save_json(directory/'download_report.json', result)
    print(json.dumps(result, ensure_ascii=True), flush=True)


def auto_boxes(objects, width, height):
    boxes = objects.get('bbox', []); categories = objects.get('categories', [])
    if len(boxes) != len(categories):
        raise ValueError('Category/box count mismatch')
    result = []
    for coords, category in zip(boxes, categories):
        if category not in AUTO_MAP:
            continue
        x, y, w, h = map(float, coords)
        result.append(from_xyxy(AUTO_MAP[category], x/width, y/height, (x+w)/width, (y+h)/height))
    return result


def autoexpert():
    output = OUT/'autoexpert'; base = f'https://huggingface.co/datasets/{AUTO_REPO}/resolve/{AUTO_REV}'
    fetch(base+'/README.md', output/'metadata/README.md')
    candidates = []
    for split in ('train','val'):
        meta = fetch(base+f'/{split}/metadata.jsonl', output/f'metadata/{split}.jsonl')
        for line in meta.read_text(encoding='utf-8').splitlines():
            item=json.loads(line)
            if {'Temporary_Construction_Barriers','Cones'} & set(item.get('objects',{}).get('categories',[])):
                candidates.append((split,item))
    def one(pair):
        split,item=pair
        filename=Path(item['file_name']).name
        image=fetch(base+f'/{split}/'+filename, output/'images'/split/filename)
        with Image.open(image) as picture:
            width,height=picture.size
        boxes=auto_boxes(item['objects'],width,height)
        label=output/'labels'/split/(Path(filename).stem+'.txt')
        write_labels(label,boxes)
        return record(image,label,'autoexpert_pandaset',group='pandaset:'+str(item['seq_id']),
            original_split=split, source_url=f'https://huggingface.co/datasets/{AUTO_REPO}',
            download_url=base+f'/{split}/'+filename, source_revision=AUTO_REV,
            license='CC-BY-NC-SA-4.0', license_url='https://creativecommons.org/licenses/by-nc-sa/4.0/',
            attribution='AutoExpert authors; PandaSet (Hesai and Scale)',
            annotation_origin='publisher_human_2d_boxes', source_annotation= str(output/f'metadata/{split}.jsonl'),
            source_frame=item.get('frame_idx'), source_camera=item.get('camera_name'),
            annotation_scope='Federated target-only in train; campus classes still incomplete',
            synthetic=False)
    with ThreadPoolExecutor(max_workers=8) as executor:
        rows=list(executor.map(one,candidates))
    finish(output,rows,source_candidates=len(candidates),preserved_splits=True,
           caveat='Cones definition includes short guidance poles; verify semantic fit per box before training.')


def voc_boxes(xml_data, names):
    if b'<!DOCTYPE' in xml_data or b'<!ENTITY' in xml_data:
        raise ValueError('Unsafe XML declaration')
    root=ET.fromstring(xml_data)
    width=int(root.findtext('size/width')); height=int(root.findtext('size/height'))
    if min(width,height)<=0:
        raise ValueError('Invalid image dimensions')
    boxes=[]
    for obj in root.findall('object'):
        name=obj.findtext('name')
        if name not in names:
            continue
        bounds=obj.find('bndbox')
        x1,y1,x2,y2=[float(bounds.findtext(v)) for v in ('xmin','ymin','xmax','ymax')]
        boxes.append(from_xyxy(names[name],x1/width,y1/height,x2/width,y2/height))
    return root.findtext('filename'), width,height,boxes


def fire_boxes(xml_data):
    return voc_boxes(xml_data, {'Fire_Extinguisher':24})


def firenet(limit):
    output=OUT/'firenet'
    catalog=get_json('https://api.figshare.com/v2/articles/9137798',output/'metadata/catalog.json')
    if catalog['license']['name'] != 'CC BY-NC 4.0':
        raise ValueError('Publisher license changed: review catalog')
    files={f['name']:f for f in catalog['files']}
    annotation=files['FireNet_ObjectDetection.zip']; source=files['FireNet_Images.zip']
    archive=fetch(annotation['download_url'],output/'metadata/annotations.zip')
    if hashlib.md5(archive.read_bytes()).hexdigest()!=annotation['computed_md5']:
        raise ValueError('Annotation archive MD5 mismatch')
    candidates=[]; failures=[]
    with zipfile.ZipFile(archive) as labels:
        for name in sorted(labels.namelist()):
            if not name.endswith('.xml'):continue
            try:
                item=fire_boxes(labels.read(name))
                if item[3]:candidates.append((name,item))
            except ValueError as error:failures.append({'annotation':name,'reason':str(error)})
        # Even coverage across source image order; grouped conservatively below.
        chosen=[candidates[i] for i in np.linspace(0,len(candidates)-1,min(limit,len(candidates)),dtype=int)]
        rows=[]
        with RangeReader(source['download_url'],source['size'],block_size=1024*1024) as remote:
            with zipfile.ZipFile(remote) as images:
                lookup={Path(n).name:n for n in images.namelist() if n.lower().endswith(('.jpg','.png'))}
                for n,(annotation_name,(filename,width,height,boxes)) in enumerate(chosen):
                    image=output/'images'/filename; image.parent.mkdir(parents=True,exist_ok=True)
                    if not image.exists():image.write_bytes(images.read(lookup[filename]))
                    with Image.open(image) as picture:
                        if picture.size != (width,height):
                            failures.append({'image':filename,'reason':'Dimensions differ from annotation'});continue
                    label=output/'labels'/(Path(filename).stem+'.txt');write_labels(label,boxes)
                    raw=output/'source_annotations'/annotation_name;raw.parent.mkdir(parents=True,exist_ok=True);raw.write_bytes(labels.read(annotation_name))
                    rows.append(record(image,label,'firenet',group='firenet:unknown_session',
                        source_url='https://rdr.ucl.ac.uk/articles/dataset/FireNet/9137798',
                        download_url=source['download_url']+'#'+lookup[filename],
                        source_annotation=str(raw.resolve()),source_annotation_sha256=sha(raw),
                        license='CC-BY-NC-4.0',license_url=catalog['license']['url'],
                        attribution='J. Boehm et al., FireNet, UCL (2019)', synthetic=False,
                        session_known=False,
                        grouping_reason='Capture-session metadata absent; source grouped conservatively until review'))
                    if (n+1)%25==0:
                        write_manifest(output/'manifest.jsonl',rows);print('FireNet imported',n+1,flush=True)
        finish(output,rows,source_positive_images=len(candidates),quarantined=failures,
               remote_archive_bytes_read=remote.bytes_downloaded)


def mask_boxes(mask, cls=18):
    """Bounding extent of semantic cable foreground; no invented instance separation."""
    foreground=np.asarray(mask.convert('L'))>=128
    yy,xx=np.where(foreground)
    if not len(xx):return []
    height,width=foreground.shape
    return [from_xyxy(cls,float(xx.min())/width,float(yy.min())/height,
                     float(xx.max()+1)/width,float(yy.max()+1)/height)]


def cables():
    output=OUT/'cables_synthetic'
    catalog=get_json('https://metax.fairdata.fi/v3/datasets/c48afa1d-1dad-4477-9c45-62ed47c5f4de',output/'metadata/fairdata.json')
    licenses=catalog['access_rights']['license']
    if not any('CC-BY-4.0' in str(v) for v in licenses):raise ValueError('License changed')
    tree=get_json(f'https://api.github.com/repos/{CABLE_REPO}/git/trees/{CABLE_REV}?recursive=1',output/'metadata/tree.json')
    paths={x['path'] for x in tree['tree'] if x['type']=='blob'}
    base=f'https://raw.githubusercontent.com/{CABLE_REPO}/{CABLE_REV}'
    fetch(base+'/README.md',output/'metadata/README.md')
    all_images=sorted(p for p in paths if p.startswith('synthetic_images/') and '/images/' in p and p.endswith('.png'))
    missing=[n for n in all_images if n.replace('/images/','/masks/') not in paths]
    images=[n for n in all_images if n not in missing]
    def one(name):
        mask_name=name.replace('/images/','/masks/')
        if mask_name not in paths:raise ValueError('Mask not found: '+mask_name)
        category=name.split('/')[1];stem=Path(name).stem
        image=fetch(base+'/'+name,output/'images'/category/(stem+'.png'))
        mask_path=fetch(base+'/'+mask_name,output/'masks'/category/(stem+'.png'))
        with Image.open(mask_path) as mask,Image.open(image) as picture:
            if mask.size!=picture.size:raise ValueError('Image/mask dimensions differ')
            boxes=mask_boxes(mask)
        if not boxes:return None
        label=output/'labels'/category/(stem+'.txt');write_labels(label,boxes)
        return record(image,label,'malvido_synthetic_cables',group='malvido_synthetic:'+category,
            source_url='https://doi.org/10.23729/93af7b3a-0f99-418b-9769-3ab8f345909a',
            download_url=base+'/'+name,source_revision=CABLE_REV,
            source_annotation=str(mask_path.resolve()),source_annotation_sha256=sha(mask_path),
            license='CC-BY-4.0',license_url='https://creativecommons.org/licenses/by/4.0/',
            attribution='Pablo Malvido Fresnillo et al., Tampere University, 2024',synthetic=True,
            session_known=False,
            annotation_origin='publisher_semantic_mask_to_extent',
            annotation_scope='Single bounding extent of all cable pixels; review cable instances and suitability for detection')
    with ThreadPoolExecutor(max_workers=8) as executor:rows=[r for r in executor.map(one,images) if r]
    finish(output,rows,synthetic_images=len(rows),real_images=0,missing_masks=missing)


def cones():
    output=OUT/'cones_hf';repo='Junhan0518/Traffic_Cone';rev='a06fc558f85adb8d07b4c36d07660acf8ecfc9ee'
    metadata=get_json(f'https://huggingface.co/api/datasets/{repo}/revision/{rev}',output/'metadata/repository.json')
    if metadata.get('cardData',{}).get('license')!='apache-2.0':raise ValueError('License changed')
    base=f'https://huggingface.co/datasets/{repo}/resolve/{rev}'
    fetch(base+'/README.md',output/'metadata/README.md')
    archive=fetch(base+'/cone_dataset.tar.gz',output/'archives/cone_dataset.tar.gz')
    rows=[];failures=[]
    with tarfile.open(archive) as source:
        members={n.name:n for n in source.getmembers() if n.isfile()}
        for name in sorted(members):
            if not name.endswith('.xml'):continue
            try:
                raw=source.extractfile(members[name]).read()
                filename,width,height,boxes=voc_boxes(raw,{'trafficcone':23})
                if not boxes:continue
                # Author XML filename omits the extension; pair by annotation stem.
                image_name='cone_dataset/'+Path(name).stem+'.jpg'
                if image_name not in members:raise ValueError('Image missing')
                image=output/'images'/(Path(name).stem+'.jpg');image.parent.mkdir(parents=True,exist_ok=True)
                image.write_bytes(source.extractfile(members[image_name]).read())
                with Image.open(image) as picture:
                    if picture.size!=(width,height):raise ValueError('Dimensions differ')
                label=output/'labels'/(Path(name).stem+'.txt');write_labels(label,boxes)
                original=output/'source_annotations'/Path(name).name;original.parent.mkdir(parents=True,exist_ok=True);original.write_bytes(raw)
                rows.append(record(image,label,'junhan_traffic_cone',group='junhan_cone:unknown_session',
                    source_url=f'https://huggingface.co/datasets/{repo}',download_url=base+'/cone_dataset.tar.gz#'+image_name,
                    source_revision=rev,license='Apache-2.0',license_url='https://www.apache.org/licenses/LICENSE-2.0',
                    attribution='Junhan0518, Traffic_Cone dataset',synthetic=False,session_known=False,
                    source_annotation=str(original.resolve()),source_annotation_sha256=sha(original),
                    annotation_origin='publisher_voc_boxes',
                    review_reason='Publisher declares Apache2; original individual image authors are not documented. Verify provenance and all 25 categories.'))
            except (ValueError,KeyError) as error:failures.append({'annotation':name,'reason':str(error)})
    finish(output,rows,quarantined=failures,archive_sha256=sha(archive),
           source_image_files=sum(n.endswith('.jpg') for n in members),source_xml_files=sum(n.endswith('.xml') for n in members),
           unannotated_images_excluded=True)


def wet_floor():
    """Download classified synthetic scenes separately; do not invent detector boxes."""
    output=OUT/'wet_floor_synthetic';repo='Jsohal174/warehouse-safety-hazard-dataset';rev='b23808ec287ad4aa377f661f9cbda840e76865e1'
    metadata=get_json(f'https://huggingface.co/api/datasets/{repo}/revision/{rev}',output/'metadata/repository.json')
    if metadata.get('cardData',{}).get('license')!='cc-by-4.0':raise ValueError('License changed')
    base=f'https://huggingface.co/datasets/{repo}/resolve/{rev}'
    fetch(base+'/README.md',output/'metadata/README.md')
    annotation=fetch(base+'/metadata.jsonl',output/'metadata/source.jsonl')
    source_rows=[json.loads(v) for v in annotation.read_text(encoding='utf-8').splitlines() if v.strip()]
    by_name={Path(r['image']).stem:r for r in source_rows}
    names=sorted(r['rfilename'] for r in metadata['siblings'] if re.match(r'^(train|test)/spill/[^/]+\.jpg$',r['rfilename']))
    def one(name):
        split=name.split('/')[0];stem=Path(name).stem
        image=fetch(base+'/'+name,output/'images'/split/(stem+'.jpg'))
        item=by_name.get(stem,{})
        # Empty labels are an editing scaffold, never verified negatives.
        label=output/'labels_pending'/split/(stem+'.txt')
        if not label.exists():write_labels(label,[])
        row=record(image,label,'warehouse_synthetic_spill',
            group='warehouse_synthetic:frame:'+str(item.get('source_frame','unknown')),
            original_split=split,source_url=f'https://huggingface.co/datasets/{repo}',download_url=base+'/'+name,
            source_revision=rev,license='CC-BY-4.0',license_url='https://creativecommons.org/licenses/by/4.0/',
            attribution='Jsohal174, Warehouse Safety Hazard Detection Dataset',synthetic=True,
            candidate_classes=['wet_floor'],annotation_status='image_class_only_no_boxes',
            label_available=False,source_description=item.get('description'),source_frame=item.get('source_frame'),
            source_subtype=item.get('subtype'),source_annotation=str(annotation.resolve()),
            review_reason='Synthetic overhead scene; draw boxes, check that surface is wet rather than dried residue, annotate all campus classes.')
        row['id']='warehouse_synthetic_spill_'+split+'_'+stem
        return row
    with ThreadPoolExecutor(max_workers=8) as executor:rows=list(executor.map(one,names))
    write_manifest(output/'candidate_manifest.jsonl',rows)
    report={'images':len(rows),'unique_images':len({r['sha256'] for r in rows}),'synthetic':True,
            'detector_boxes':0,'training_approved':0,'pending_bbox_annotation':len(rows),
            'manifest':'candidate_manifest.jsonl','class':'wet_floor'}
    save_json(output/'download_report.json',report);print(json.dumps(report),flush=True)


def inspect_liquids():
    output=OUT/'piap_liquid'
    archive=fetch('https://cloud.piap.pl/index.php/s/ApiXNzt4ZUUSRks/download',output/'archives/Liquid_dataset.zip')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        report={'source_url':'https://cloud.piap.pl/index.php/s/ApiXNzt4ZUUSRks',
            'archive_sha256':sha(archive),'members':len(names),
            'metadata_files':[n for n in names if n.lower().endswith(('.json','.txt','.md','.csv','.xml'))],
            'license_files':[n for n in names if 'license' in n.lower() or 'readme' in n.lower()]}
        save_json(output/'metadata/inspection.json',report)
        print(json.dumps(report),flush=True)
        for name in report['metadata_files']:
            if z.getinfo(name).file_size<10_000_000:
                target=output/'metadata'/Path(name).name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('source',choices=['autoexpert','firenet','cables','cones','wet_floor','inspect_liquids'])
    parser.add_argument('--limit',type=int,default=350);args=parser.parse_args()
    if args.source=='firenet':firenet(args.limit)
    else:globals()[args.source]()

if __name__=='__main__':main()
