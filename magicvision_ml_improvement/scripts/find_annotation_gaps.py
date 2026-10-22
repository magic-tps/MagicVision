"""Prioritize possible missing boxes with model proposals, never approve labels."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import sys

from data_pipeline import CLASSES,read_manifest,read_labels,save_json,sha
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation'))
from evaluate_models import iou,xyxy


def possible_gaps(predictions,truth,match_iou=.50):
    suggestions=[]
    for p in predictions:
        same=[iou(p['bbox'],g['bbox']) for g in truth if p['class_id']==g['class_id']]
        if max(same,default=0)>=match_iou:continue
        other=[g['class_id'] for g in truth if g['class_id']!=p['class_id'] and iou(p['bbox'],g['bbox'])>=match_iou]
        suggestions.append({**p,'status':'unverified_model_proposal',
            'reason':'overlapping_different_class' if other else 'possible_missing_or_misaligned_box',
            'overlapping_classes':[CLASSES[c] for c in sorted(set(other))]})
    return suggestions


def select_diverse(rows,limit):
    """Round robin classes, one SHA once; keeps source annotations untouched."""
    by_class={c:[] for c in range(len(CLASSES))};seen=set();chosen=[]
    for r in rows:
        for c in {b[0] for b in read_labels(r['label'])}:by_class[c].append(r)
    order=sorted(by_class,key=lambda c:len(by_class[c]))
    positions=Counter()
    while len(chosen)<limit:
        count=len(chosen)
        for c in order:
            while positions[c]<len(by_class[c]) and by_class[c][positions[c]]['sha256'] in seen:positions[c]+=1
            if positions[c]<len(by_class[c]) and len(chosen)<limit:
                row=by_class[c][positions[c]];positions[c]+=1
                chosen.append(row);seen.add(row['sha256'])
        if len(chosen)==count:break
    return chosen


def run(manifests,weights,output,limit=220,confidence=.35):
    from ultralytics import YOLO
    import torch
    if not 0<confidence<1 or limit<1:raise ValueError('Invalid confidence or sample limit')
    if Path(output).exists():raise ValueError('Use a new report file')
    rows=select_diverse([r for m in manifests for r in read_manifest(m)],limit)
    model=YOLO(str(weights));version=sha(weights);reports=[];counts=Counter()
    device=0 if torch.cuda.is_available() else 'cpu'
    for index,r in enumerate(rows,1):
        truth=[{'class_id':b[0],'bbox':xyxy(b[1:])} for b in read_labels(r['label'])]
        result=model.predict(r['image'],imgsz=640,conf=confidence,device=device,max_det=100,verbose=False)[0]
        pred=[]
        for b in result.boxes:
            name=model.names[int(b.cls.item())]
            if name in CLASSES:pred.append({'class_id':CLASSES.index(name),'bbox':b.xyxyn[0].tolist(),'confidence':float(b.conf.item())})
        suggestions=possible_gaps(pred,truth)
        counts.update(CLASSES[s['class_id']] for s in suggestions)
        reports.append({'id':r['id'],'sha256':r['sha256'],'image':r['image'],'label':r['label'],
                        'label_sha256':sha(r['label']),'suggestions':suggestions})
        if index%50==0:print(f'Checked {index}/{len(rows)}',flush=True)
    report={'model_sha256':version,'model_classes':model.names,'images_checked':len(rows),
            'images_with_suggestions':sum(bool(r['suggestions']) for r in reports),
            'suggestions_by_class':dict(counts),'confidence':confidence,'iou_match':.5,
            'labels_modified':False,'approved_images':0,
            'warning':'Model proposals can be false positives. Empty suggestions do NOT prove annotation completeness. Old model cannot propose trash_bin. Review manually; this is not a measurement of annotation errors or model accuracy.',
            'records':reports}
    save_json(output,report)
    print({k:v for k,v in report.items() if k!='records'})
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',nargs='+',required=True);p.add_argument('--weights',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=220);p.add_argument('--confidence',type=float,default=.35)
    a=p.parse_args();run(a.manifest,a.weights,a.output,a.limit,a.confidence)
