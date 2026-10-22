"""Evaluate real predictions with fixed matching, sweeps and per-class error counts."""
from __future__ import annotations
import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from data_pipeline import CLASSES, read_labels, save_json, sha


def xyxy(box):
    x,y,w,h=box
    return [x-w/2,y-h/2,x+w/2,y+h/2]


def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0


def match(predictions,truth,threshold=.5):
    """Descending confidence, same-class maximum IoU, one-to-one; duplicates are FP."""
    used=set();result=[]
    for p in sorted(predictions,key=lambda x:x['confidence'],reverse=True):
        candidates=[(iou(p['bbox'],g['bbox']),j) for j,g in enumerate(truth) if j not in used and p['class_id']==g['class_id']]
        best=max(candidates,default=(0,-1))
        success=best[0]>=threshold
        if success:used.add(best[1])
        result.append((p,success))
    return result,used


def average_precision(scored,total):
    if total==0:return None
    if not scored:return 0.
    scored=sorted(scored,key=lambda p:p[0],reverse=True);tp=0;points=[]
    for rank,(_,correct) in enumerate(scored,1):
        tp+=int(correct);points.append((tp/total,tp/rank))
    return sum(max((p for r,p in points if r>=t/100),default=0) for t in range(101))/101


def metrics(records,confidence=.30,names=None):
    names=CLASSES if names is None else names
    background=len(names)
    counts=defaultdict(Counter);ap=defaultdict(list);totals=Counter();matrix=[[0]*(background+1) for _ in range(background+1)]
    for row in records:
        truth=row['truth'];pred=[p for p in row['predictions'] if p['confidence']>=confidence]
        totals.update(g['class_id'] for g in truth)
        matched,used=match(pred,truth)
        for p,ok in matched:counts[p['class_id']]['tp' if ok else 'fp']+=1
        for j,g in enumerate(truth):
            if j not in used:counts[g['class_id']]['fn']+=1
        # Diagnostic confusion matrix matches geometry irrespective of class.
        free=set(range(len(truth)))
        for p in sorted(pred,key=lambda x:x['confidence'],reverse=True):
            score,j=max(((iou(p['bbox'],truth[j]['bbox']),j) for j in free),default=(0,-1))
            if score>=.5:matrix[p['class_id']][truth[j]['class_id']]+=1;free.remove(j)
            else:matrix[p['class_id']][background]+=1
        for j in free:matrix[background][truth[j]['class_id']]+=1
        for k in range(10):
            m,_=match(row['predictions'],truth,.5+k*.05)
            for p,ok in m:ap[(p['class_id'],k)].append((p['confidence'],ok))
    classes={}
    for c,name in enumerate(names):
        tp,fp,fn=(counts[c][key] for key in ('tp','fp','fn'))
        precision=tp/(tp+fp) if tp+fp else None;recall=tp/(tp+fn) if tp+fn else None
        aps=[average_precision(ap[(c,k)],totals[c]) for k in range(10)]
        classes[name]={'tp':tp,'fp':fp,'fn':fn,'support':totals[c],'precision':precision,'recall':recall,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,'ap50':aps[0],
            'ap50_95':sum(aps)/10 if aps[0] is not None else None,'fp_per_image':fp/len(records) if records else None}
    return {'images':len(records),'confidence':confidence,'iou_match':.5,'classes':classes,
        'map50':statistics.mean(x['ap50'] for x in classes.values() if x['ap50'] is not None) if totals else None,
        'map50_95':statistics.mean(x['ap50_95'] for x in classes.values() if x['ap50_95'] is not None) if totals else None,
        'confusion_matrix_predicted_rows_true_columns':matrix,'matrix_labels':names+['background'],
        'ap_protocol':'101-point interpolated AP at IoU .50:.05:.95; predictions cached above configured floor; not directly identical to Ultralytics historical metrics'}


def evaluate(weights,dataset,output,imgsz=640,limit=0,compression='source',floor=.001,split='val',schema=None):
    names=CLASSES if schema is None else schema.CLASSES
    parse_labels=read_labels if schema is None else schema.read_labels
    if not math.isfinite(floor) or not 0<floor<=.15:raise ValueError('Prediction floor must be in (0, 0.15] to compare all four thresholds')
    import torch,ultralytics,yaml,psutil
    from PIL import Image,ImageOps
    from ultralytics import YOLO
    out=Path(output)
    if (out/'predictions.jsonl').exists():raise ValueError('Use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    data=yaml.safe_load(Path(dataset).read_text());root=Path(data['path'])
    if not root.is_absolute():root=(Path(dataset).parent/root).resolve()
    directory=root/data[split]
    images=sorted(p for p in directory.rglob('*') if p.suffix.lower() in {'.jpg','.png','.jpeg','.webp','.bmp'})
    if limit:images=images[:limit]
    if not images:raise ValueError('No evaluation images')
    model=YOLO(str(weights));device=0 if torch.cuda.is_available() else 'cpu'
    if device==0:torch.cuda.reset_peak_memory_stats()
    records=[];latencies=[];sizes=[]; process=psutil.Process()
    model.predict(Image.new('RGB',(640,480)),imgsz=imgsz,device=device,verbose=False,conf=floor,max_det=100)
    with (out/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for n,image in enumerate(images,1):
            label=root/'labels'/split/image.relative_to(directory).with_suffix('.txt')
            truth=[{'class_id':b[0],'bbox':xyxy(b[1:])} for b in parse_labels(label)]
            source=image;sent_size=image.stat().st_size
            if compression=='app640':
                import io
                with Image.open(image) as im:
                    im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((640,640),Image.Resampling.BILINEAR)
                    buffer=io.BytesIO();im.save(buffer,format='JPEG',quality=80);sent_size=buffer.tell();buffer.seek(0);source=Image.open(buffer).convert('RGB')
            if device==0:torch.cuda.synchronize()
            started=time.perf_counter()
            result=model.predict(source,imgsz=imgsz,conf=floor,iou=.65,max_det=100,device=device,verbose=False,save=False)[0]
            if device==0:torch.cuda.synchronize()
            elapsed=(time.perf_counter()-started)*1000;latencies.append(elapsed);sizes.append(sent_size)
            preds=[]
            for b in result.boxes:
                name=model.names[int(b.cls.item())]
                if schema is not None and name=='trash_bin':name='trash_can'
                if name not in names:raise ValueError('Model class mismatch')
                preds.append({'class_id':names.index(name),'confidence':float(b.conf.item()),'bbox':b.xyxyn[0].tolist()})
            row={'image':str(image),'sha256':sha(image),'truth':truth,'predictions':preds,'latency_ms':elapsed,'input_bytes':sent_size}
            records.append(row);f.write(json.dumps(row)+'\n')
            if n%100==0:print(f'Evaluated {n}/{len(images)}',flush=True)
    for threshold in (.15,.20,.25,.30):save_json(out/f'metrics_{threshold:.2f}.json',metrics(records,threshold,names=names))
    info={'checkpoint':str(Path(weights).resolve()),'checkpoint_sha256':sha(weights),'model_bytes':Path(weights).stat().st_size,
        'model_names':model.names,'ultralytics':ultralytics.__version__,'torch':torch.__version__,'device':str(device),
        'imgsz':imgsz,'compression':compression,'compression_note':'app640 uses PIL approximation; use exact captured detector.jpg for phone evidence',
        'conf_floor':floor,'iou_parameter':.65,'end2end':getattr(model.model,'end2end',None),
        'nms_note':'For end-to-end YOLO26, iou may not affect default NMS-free output; no unmeasured NMS improvement claimed',
        'images':len(records),'split':split,'partial_subset':bool(limit),'latency_mean_ms':statistics.mean(latencies),
        'latency_p95_ms':sorted(latencies)[math.ceil(.95*len(latencies))-1],
        'input_bytes_mean':statistics.mean(sizes),'gpu_peak_bytes':torch.cuda.max_memory_allocated() if device==0 else None,
        'process_rss_bytes':process.memory_info().rss,'network_latency_ms':None,'subgroup_metrics':'pending verified campus scene metadata; no meters inferred from boxes',
        'trash_bin_baseline':'not supported' if 'trash_bin' not in model.names.values() else 'supported class'}
    save_json(out/'run.json',info);print(json.dumps(info))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--weights',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--imgsz',type=int,choices=[640,960],default=640);p.add_argument('--limit',type=int,default=0);p.add_argument('--compression',choices=['source','app640'],default='source');p.add_argument('--floor',type=float,default=.001);p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--final-test',action='store_true')
    a=p.parse_args()
    if a.split=='test' and not a.final_test:p.error('Final test requires --final-test after selecting the candidate on validation')
    evaluate(a.weights,a.data,a.output,a.imgsz,a.limit,a.compression,a.floor,a.split)


if __name__=='__main__':main()
