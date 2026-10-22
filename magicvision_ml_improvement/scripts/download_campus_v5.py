"""Additional real campus hazards and roadwork boxes, with unchanged 25-class IDs."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import io
import json
from pathlib import Path
import random
import re
import zipfile
import zlib
from PIL import Image
from campus_schema import ROOT, CLASSES, record, from_xyxy, read_labels, write_labels, write_manifest, read_manifest, save_json, sha
from download_campus_access import read_member

RESEARCH=ROOT/'datasets/campus_v5_research'
OUT=ROOT/'datasets/campus_additions_v5'
ROAD_MAP={'Cone':23,'Barricade':20,'Barrier':20,'Worker':0,'Police Officer':0,'Temporary Traffic Control Sign':8}


def road_boxes(annotations,names,width,height):
    boxes=[]
    for ann in annotations:
        name=names[ann['category_id']]
        if name not in ROAD_MAP or ann.get('iscrowd'):continue
        x,y,w,h=ann['bbox']
        boxes.append(from_xyxy(ROAD_MAP[name],x/width,y/height,(x+w)/width,(y+h)/height))
    return boxes


def select_road(items,quotas,seed=42):
    """Quota selection favors different source sequences before extra frames."""
    items=list(items);random.Random(seed).shuffle(items)
    used=set();groups=Counter();counts=Counter();selected=[]
    for cap in (1,2,4,8):
        for item in items:
            if item['key'] in used or groups[item['group']]>=cap:continue
            classes={b[0] for b in item['boxes']}
            if not any(counts[c]<v and c in classes for c,v in quotas.items()):continue
            selected.append(item);used.add(item['key']);groups[item['group']]+=1;counts.update(classes)
    return selected


def download_checked(url,size,member,path,dimensions=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    good=path.exists() and path.stat().st_size==member['file_size']
    if good:good=zlib.crc32(path.read_bytes())&0xffffffff==member['CRC']
    if not good:
        content=read_member(url,size,member)
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            if dimensions and image.size!=dimensions:raise ValueError('Source image/annotation dimensions differ')
        part=path.with_suffix(path.suffix+'.part');part.write_bytes(content);part.replace(path)
    if dimensions:
        with Image.open(path) as image:
            if image.size!=dimensions:raise ValueError('Cached image/annotation dimensions differ')


def finish(out,rows,errors,**extra):
    rows.sort(key=lambda r:r['id'])
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate source IDs')
    write_manifest(out/'manifest.jsonl',rows)
    hashes=defaultdict(set);boxes=Counter();candidates=defaultdict(set)
    for row in rows:
        labels=read_labels(row['label']);boxes.update(CLASSES[b[0]] for b in labels)
        for cls in {b[0] for b in labels}:hashes[CLASSES[cls]].add(row['sha256'])
        for cls in row.get('candidate_classes',[]):candidates[cls].add(row['sha256'])
    previous={r['sha256'] for r in read_manifest(ROOT/'datasets/expanded_campus_v4/manifest.jsonl')}
    report={'records':len(rows),'unique_images':len({r['sha256'] for r in rows}),
        'new_unique_vs_v4':len({r['sha256'] for r in rows}-previous),
        'images_with_boxes_by_class':{k:len(v) for k,v in hashes.items()},'boxes_by_class':dict(boxes),
        'candidate_images_by_class':{k:len(v) for k,v in candidates.items()},'errors':errors,
        'approved_complete_25':0,'synthetic':False,**extra}
    save_json(out/'download_report.json',report)
    print(json.dumps(report,ensure_ascii=True),flush=True)


def roadwork(goal=1000,workers=4):
    out=OUT/'roadwork';out.mkdir(parents=True,exist_ok=True)
    catalog=json.loads((RESEARCH/'roadwork_catalog.json').read_text(encoding='utf-8'))
    archive=next(f for f in catalog['files'] if f['name']=='images.zip')
    index=json.loads((RESEARCH/'roadwork_zip_index.json').read_text(encoding='utf-8'))
    lookup={Path(name).name:name for name in index if name.lower().endswith('.jpg')}
    if len(lookup)!=sum(n.lower().endswith('.jpg') for n in index):raise ValueError('Ambiguous image filenames')
    readme=(RESEARCH/'roadwork_source_readme.md').read_text(encoding='utf-8')
    if 'The ROADWork dataset is licensed under' not in readme or 'Open Data Commons Attribution' not in readme:
        raise ValueError('Missing explicit dataset license evidence')
    annotations_path=RESEARCH/'roadwork_annotations.zip'
    annotation_digest=sha(annotations_path)
    selected=[];errors=[];available={};target={}
    with zipfile.ZipFile(annotations_path) as z:
        for split,fraction in (('train',.8),('val',.2)):
            member=f'annotations/instances_{split}_gps_split.json'
            data=json.loads(z.read(member));names={c['id']:c['name'] for c in data['categories']}
            annotations=defaultdict(list)
            for ann in data['annotations']:annotations[ann['image_id']].append(ann)
            items=[]
            for image in data['images']:
                try:
                    boxes=road_boxes(annotations[image['id']],names,image['width'],image['height'])
                    if not ({20,23}&{b[0] for b in boxes}):continue
                    filename=image['file_name']
                    if Path(filename).name!=filename:raise ValueError('Unexpected image filename')
                    video=image.get('video_info',{})
                    group=f"roadwork:{image.get('city_name','unknown')}:{video.get('vid_id',filename.split('_')[0])}:{video.get('seq_id','unknown')}"
                    items.append({'key':filename,'split':split,'image':image,'boxes':boxes,'group':group,
                                  'source_annotations':annotations[image['id']],'annotation_member':member})
                except (ValueError,KeyError) as exc:errors.append({'split':split,'id':image['id'],'error':str(exc)})
            quota={20:round(goal*fraction),23:round(goal*fraction)};target[split]=quota
            available[split]=dict(Counter(c for x in items for c in {b[0] for b in x['boxes']}))
            selected.extend(select_road(items,quota))
    save_json(out/'selection.json',{'selected':len(selected),'target':target,'available':available})
    def work(item):
        split=item['split'];filename=item['key'];image=out/'images'/split/filename
        member=lookup[filename];info=item['image']
        download_checked(archive['download_url'],archive['size'],index[member],image,(info['width'],info['height']))
        label=out/'labels'/split/(image.stem+'.txt');write_labels(label,item['boxes'])
        source=out/'source_annotations'/split/(image.stem+'.json')
        save_json(source,{'image':info,'annotations':item['source_annotations'],'class_mapping':ROAD_MAP})
        return record(image,label,'roadwork_v5',group=item['group'],session_known=False,
            group_basis='publisher_city_video_sequence',original_split=split,
            source_url='https://kilthub.cmu.edu/articles/dataset/ROADWork_Data/26093197',
            download_url=archive['download_url']+'#'+member,source_archive_md5=archive['computed_md5'],
            source_annotation=str(source.resolve()),source_annotation_sha256=sha(source),
            source_annotations_archive_sha256=annotation_digest,source_annotation_member=item['annotation_member'],
            source_image_id=info['id'],source_image_filename=filename,source_video_info=info.get('video_info'),
            source_gps=info.get('gps'),source_city=info.get('city_name'),
            license='ODC-By-1.0 declared explicitly by dataset authors',license_url='https://github.com/anuragxel/roadwork-dataset#-license',
            source_catalog_rights_statement=catalog['license'],source_license_evidence=str((RESEARCH/'roadwork_source_readme.md').resolve()),
            attribution='Anurag Ghosh et al., ROADWork, ICCV 2025 / Carnegie Mellon University',
            annotation_origin='publisher_manual_instance_boxes',synthetic=False,
            review_reason='Complete all 25 classes; review Barrier/Barricade as temporary construction barriers. Driving-view images; not local campus captures.')
    rows=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(work,x):x for x in selected}
        for i,future in enumerate(as_completed(futures),1):
            try:rows.append(future.result())
            except Exception as exc:errors.append({'image':futures[future]['key'],'error':str(exc)})
            if i%50==0:
                write_manifest(out/'manifest.jsonl',rows);print(f'ROADWork {i}/{len(selected)} processed; {len(rows)} imported',flush=True)
    finish(out,rows,errors,selection_target=target,available=available,source_splits_preserved=True,
           source_groups=len({r['group'] for r in rows}))


def sample_hazard_runs(labels,environments,quota=200,stride=150,max_per_run=30):
    """Sample spaced frames across contiguous label/environment runs, not duplicates."""
    mapping={2:'cable',3:'cone',18:'wet_floor'}
    runs=[];current=[];previous=None
    for frame,lab in sorted(labels.items()):
        key=(lab,environments[frame])
        if previous is None or key!=previous[1] or frame!=previous[0]+1:
            if current:runs.append(current)
            current=[]
        current.append((frame,lab,environments[frame]));previous=(frame,key)
    if current:runs.append(current)
    by_class=defaultdict(list)
    for run in runs:
        if run[0][1] not in mapping:continue
        frames=run[stride//2::stride][:max_per_run]
        by_class[mapping[run[0][1]]].append([(f,lab,env,run[0][0],run[-1][0]) for f,lab,env in frames])
    selected=[]
    for cls,queues in by_class.items():
        count=0
        while count<quota and any(queues):
            for queue in queues:
                if queue and count<quota:selected.append((cls,*queue.pop(0)));count+=1
    return selected


def hazards(workers=4):
    out=OUT/'hazards_robots';out.mkdir(parents=True,exist_ok=True)
    catalog=json.loads((RESEARCH/'hazards_catalog.json').read_text(encoding='utf-8'))
    if catalog['metadata']['license']['id']!='cc-by-4.0':raise ValueError('Unexpected dataset license')
    metadata=RESEARCH/'hazards_metadata.zip';selection=[];files={};indices={};metadata_digest=sha(metadata)
    with zipfile.ZipFile(metadata) as z:
        for partition,prefix,quota in (('train','training_mixed',200),('test','test',75)):
            files[partition]=next(f for f in catalog['files'] if f['key']==prefix+'_set.zip')
            index=json.loads((RESEARCH/(prefix+'_set_zip_index.json')).read_text(encoding='utf-8'))
            # Names are resolved from actual ZIP members, not inferred frame ordering.
            images={int(Path(n).stem.split('_')[0]):n for n in index if re.fullmatch(r'\d+_512_512\.jpg',Path(n).name)}
            if not images:raise ValueError('Unknown hazard image naming; inspect ZIP before mapping labels')
            indices[partition]=(index,images)
            labels={int(r['frame_id']):int(r['label']) for r in csv.DictReader(io.StringIO(z.read(f'metadata/{prefix}_frames_labels.csv').decode()))}
            envs={int(r['frame_id']):int(r['env']) for r in csv.DictReader(io.StringIO(z.read(f'metadata/{prefix}_set_frames_envs.csv').decode()))}
            selection.extend((partition,*x) for x in sample_hazard_runs(labels,envs,quota=quota))
    save_json(out/'selection.json',{'items':selection,'stride_frames':150,'max_per_contiguous_run':30})
    def work(item):
        split,cls,frame,source_class,env,first,last=item
        index,images=indices[split];member=images[frame];file=files[split]
        image=out/'images'/split/(f'{frame:08d}'+Path(member).suffix.lower())
        download_checked(file['links']['self'],file['size'],index[member],image)
        label=out/'labels'/split/(image.stem+'.txt')
        if not label.exists():write_labels(label,[])
        row=record(image,label,'hazards_robots_v5',group=f'hazards_robots:environment:{env}',session_known=False,
            group_basis='same_physical_corridor_environment_conservative',original_split=split,
            source_url='https://zenodo.org/records/7859211',download_url=file['links']['self']+'#'+member,
            source_archive_checksum=file['checksum'],source_annotation=str(metadata.resolve()),source_annotation_sha256=metadata_digest,
            source_frame=frame,source_class=source_class,source_environment=env,source_contiguous_run=[first,last],
            source_sampling_stride=150,license='CC-BY-4.0',license_url='https://creativecommons.org/licenses/by/4.0/',
            attribution='Dario Mantegazza et al., Hazards & Robots, Data in Brief 2023',synthetic=False,
            candidate_classes=[cls],annotation_status='source_frame_class_only_no_boxes',
            annotation_origin='publisher_frame_classification_no_localization',verified_negative=False,
            review_reason='Real university corridor recorded by low robot camera. Class is frame-level; draw all visible objects. Repeated environments must remain grouped.')
        row['id']=f'hazards_robots_v5_{split}_{frame:08d}';return row
    rows=[];errors=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(work,x):x for x in selection}
        for i,future in enumerate(as_completed(futures),1):
            try:rows.append(future.result())
            except Exception as exc:errors.append({'item':futures[future],'error':str(exc)})
            if i%50==0:
                write_manifest(out/'manifest.jsonl',rows);print(f'Hazards & Robots {i}/{len(selection)} processed',flush=True)
    finish(out,rows,errors,selected=len(selection),stride_frames=150,max_per_run=30,
           warning='Frame classification only, no detection ground truth; limited physical corridors and source splits retained')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',choices=['roadwork','hazards'])
    parser.add_argument('--goal',type=int,default=1000);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.source=='roadwork':roadwork(args.goal,args.workers)
    else:hazards(args.workers)
