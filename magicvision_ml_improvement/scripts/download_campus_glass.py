"""Acquire real glass-door semantic masks; derived boxes remain review candidates."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from scipy import ndimage
from campus_schema import ROOT,record,write_labels,write_manifest,read_manifest,save_json,sha,from_xyxy
from download_rare_sources import fetch

REPO='rdyzakya/Trans10K-v2'
REV='3445e93f1e507ea6c2dd36f06badfd5e818b699f'
BASE=f'https://huggingface.co/datasets/{REPO}/resolve/{REV}'
PRIMARY='https://github.com/xieenze/Trans2Seg'
BASE_TERMS='https://github.com/xieenze/Segment_Transparent_Objects'


def mask_boxes(mask,min_pixels=16):
    """Source class7 RGB120=glass door. Connected components are NOT instances."""
    rgb=np.asarray(mask.convert('RGB'))
    foreground=np.all(rgb==[120,120,120],axis=-1)
    regions,n=ndimage.label(foreground,structure=np.ones((3,3)))
    slices=ndimage.find_objects(regions);height,width=foreground.shape
    boxes=[];ignored=0
    areas=np.bincount(regions.ravel())
    for index,region in enumerate(slices,1):
        if region is None:continue
        y,x=region
        if areas[index]<min_pixels or x.stop-x.start<2 or y.stop-y.start<2:
            ignored+=1;continue
        boxes.append(from_xyxy(14,x.start/width,y.start/height,x.stop/width,y.stop/height))
    return boxes,ignored


def run(output,limits):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    tree_path=fetch(f'https://huggingface.co/api/datasets/{REPO}/tree/{REV}/data',out/'metadata/tree.json')
    fetch(f'{BASE}/README.md',out/'metadata/mirror_README.md')
    fetch('https://raw.githubusercontent.com/xieenze/Trans2Seg/master/README.md',out/'metadata/primary_README.md')
    fetch('https://raw.githubusercontent.com/xieenze/Segment_Transparent_Objects/master/README.md',out/'metadata/base_dataset_terms.md')
    entries=json.loads(tree_path.read_text(encoding='utf-8'))
    shards=[e for e in entries if e['path'].endswith('.parquet')]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(fetch,f"{BASE}/{e['path']}",out/'archives'/Path(e['path']).name,e['size'],e['lfs']['oid']) for e in shards]
        for f in futures:f.result()
    seen={r['sha256'] for r in read_manifest(ROOT/'datasets/expanded_v3/manifest.jsonl')}
    rows=[];errors=[];count=Counter();boxcounts=Counter();duplicates=[];small=0;examined=Counter()
    # Source partitions are retained. Do not use held-out masks to select models.
    for entry in shards:
        subset=Path(entry['path']).name.split('-')[0];subset='val' if subset=='validation' else subset
        path=out/'archives'/Path(entry['path']).name;row_index=-1
        if subset not in limits:raise ValueError('Unknown source partition')
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
            if count[subset]>=limits[subset]:break
            for source in batch.to_pylist():
                row_index+=1
                if count[subset]>=limits[subset]:break
                examined[subset]+=1
                key=f'{path.stem}_{row_index:05d}'
                try:
                    raw_mask=source['mask']['bytes'];raw_image=source['image']['bytes']
                    if not raw_mask or not raw_image:raise ValueError('Missing embedded image/mask')
                    if max(len(raw_image),len(raw_mask))>32_000_000:raise ValueError('Oversized image/mask')
                    with Image.open(io.BytesIO(raw_mask)) as mask:
                        mask.load();boxes,ignored=mask_boxes(mask);mask_size=mask.size
                    small+=ignored
                    if not boxes:continue
                    with Image.open(io.BytesIO(raw_image)) as im:
                        im.load()
                        if im.size!=mask_size:raise ValueError('Image and mask dimensions differ')
                        suffix='.'+{'JPEG':'jpg','PNG':'png'}.get(im.format,'unsupported')
                    if suffix=='.unsupported':raise ValueError('Unexpected image encoding')
                    image=out/'images'/subset/(key+suffix);label=out/'labels'/subset/(key+'.txt')
                    mask_path=out/'masks'/subset/(key+'.png')
                    image.parent.mkdir(parents=True,exist_ok=True);mask_path.parent.mkdir(parents=True,exist_ok=True)
                    image.write_bytes(raw_image);mask_path.write_bytes(raw_mask);write_labels(label,boxes)
                    row=record(image,label,'trans10k_glass',group='trans10k:'+key,session_known=False,
                        original_split=subset,source_split=subset,source_url=PRIMARY,download_url=f"{BASE}/{entry['path']}",
                        source_revision=REV,source_image_id=key,source_image_path=source['image'].get('path'),
                        source_archive_sha256=entry['lfs']['oid'],source_mask=str(mask_path.resolve()),source_mask_sha256=sha(mask_path),
                        license='Academic research: Apache-2.0 per base project; commercial use requires author clarification',
                        license_url=BASE_TERMS,mirror_license='Apache-2.0 declared by mirror',
                        author='Enze Xie et al.; Hugging Face mirror rdyzakya',
                        label_origin='derived_boxes_from_human_semantic_mask',
                        mapping={'RGB 120,120,120 / semantic door':'glass_door'},
                        review_reason='Review connected components: one door may be fragmented or several doors connected. Complete all 25 classes and verify source conditions. Semantic regions are not instance ground truth.')
                    if row['sha256'] in seen:duplicates.append(key);continue
                    seen.add(row['sha256']);rows.append(row);count[subset]+=1;boxcounts[subset]+=len(boxes)
                except (ValueError,KeyError,OSError,TypeError) as exc:errors.append({'id':key,'error':str(exc)})
        write_manifest(out/'manifest.jsonl',rows)
        print(f'Imported glass-door candidates: {dict(count)}',flush=True)
    report={'source':REPO,'source_revision':REV,'images':len(rows),'images_by_source_split':dict(count),
            'glass_door_boxes_by_split':dict(boxcounts),'quotas':limits,'remaining':{k:max(0,v-count[k]) for k,v in limits.items()},
            'source_rows_examined':dict(examined),'ignored_tiny_components':small,'exact_duplicates_excluded':duplicates,
            'errors':errors,'annotations':'Derived boxes from semantic masks; instances need correction',
            'license_scope':'Academic research; base project requests contact for commercial use',
            'approved_for_training':0,'review_status':'needs_review'}
    save_json(out/'download_report.json',report);print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'datasets/campus_glass_v4')
    args=p.parse_args();run(args.output,{'train':300,'val':100,'test':100})
