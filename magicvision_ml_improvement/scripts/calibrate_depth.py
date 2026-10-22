"""Fit log-affine camera calibration on measured rows and evaluate held-out rows.

Never captures or invents ground truth. A generated profile remains unvalidated
until a reviewer supplies an explicit acceptance decision and measurement evidence.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

RANGES = [('0-1', 0, 1), ('1-2', 1, 2), ('2-4', 2, 4), ('4-6', 4, 6), ('>6', 6, float('inf'))]


def metrics(predicted, real):
    p=np.asarray(predicted,dtype=float);r=np.asarray(real,dtype=float)
    if not len(r):return {'n':0,'MAE':None,'RMSE':None,'MAPE_percent':None,'bias':None,'median_absolute_error':None}
    error=p-r
    return {'n':len(r),'MAE':float(np.abs(error).mean()),'RMSE':float(np.sqrt(np.mean(error**2))),
        'MAPE_percent':float(np.mean(np.abs(error)/r)*100),'bias':float(error.mean()),
        'median_absolute_error':float(np.median(np.abs(error)))}


def calibrate(csv_path, camera_id, model_sha256, width, height, decision=None):
    from campus22_schema import CLASSES
    rows=list(csv.DictReader(Path(csv_path).open(encoding='utf-8-sig',newline='')))
    if not rows:raise ValueError('No real measured rows; empty template cannot calibrate')
    for row in rows:
        if row['class'] not in CLASSES or row['camera_id']!=camera_id or row['split'] not in ('calibration','validation'):
            raise ValueError('Invalid class/camera/split')
        if not row.get('session_id') or not row.get('frame_id') or not row.get('measurement_evidence'):
            raise ValueError('Real session, frame and measurement evidence required')
        if int(row['image_width'])!=width or int(row['image_height'])!=height:
            raise ValueError('Use a separate profile for each inference geometry')
        for key in ('predicted_m','real_m'):
            row[key]=float(row[key])
            if not np.isfinite(row[key]) or row[key]<=0:raise ValueError('Positive finite measured distances required')
    train=[r for r in rows if r['split']=='calibration'];val=[r for r in rows if r['split']=='validation']
    if len(train)<3 or len(val)<3:raise ValueError('At least 3 calibration and 3 held-out measurements required for a diagnostic fit, not sufficient proof of reliability')
    if {r['session_id'] for r in train}&{r['session_id'] for r in val}:raise ValueError('Calibration/validation session leakage')
    if len({r['frame_id'] for r in rows})!=len(rows):raise ValueError('Repeated calibration frame')
    p=np.log([r['predicted_m'] for r in train]);gt=np.log([r['real_m'] for r in train])
    if np.ptp(p)<.1:raise ValueError('Calibration distances lack variation')
    a,b=np.linalg.lstsq(np.column_stack((p,np.ones(len(p)))),gt,rcond=None)[0]
    if not np.isfinite(a+b) or a<=0:raise ValueError('Non-monotonic or invalid calibration')
    corrected=np.exp(a*np.log([r['predicted_m'] for r in val])+b);real=np.array([r['real_m'] for r in val])
    if not np.all(np.isfinite(corrected)):raise ValueError('Calibration overflow')
    digest=hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    report={'source_csv_sha256':digest,'camera_id':camera_id,'calibration_rows':len(train),'validation_rows':len(val),
        'before':metrics([r['predicted_m'] for r in val],real),'after':metrics(corrected,real),
        'by_class':{name:metrics(corrected[[r['class']==name for r in val]],real[[r['class']==name for r in val]]) for name in CLASSES},
        'by_range':{name:metrics(corrected[(real>lo)&(real<=hi)],real[(real>lo)&(real<=hi)]) for name,lo,hi in RANGES}}
    approved=False
    if decision:
        approval=json.loads(Path(decision).read_text(encoding='utf-8'))
        required=('reviewer','reviewed_at','acceptance_criteria','measurement_evidence')
        if not all(approval.get(k) for k in required) or approval.get('source_csv_sha256')!=digest or approval.get('approved') is not True:
            raise ValueError('Approval must identify this measurement CSV and acceptance criteria')
        approved=True
    profile={'validated':approved,'camera_id':camera_id,'model_sha256':model_sha256,'policy_version':'detection-depth-v1',
        'image_width':width,'image_height':height,'log_scale':float(a),'log_bias':float(b),'source_csv_sha256':digest,
        'validation':report['after'],'acceptance_decision':str(decision) if decision else None}
    return profile,report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True);p.add_argument('--camera-id',required=True)
    p.add_argument('--model-sha256',required=True);p.add_argument('--width',type=int,required=True);p.add_argument('--height',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--decision',type=Path)
    args=p.parse_args();profile,report=calibrate(args.csv,args.camera_id,args.model_sha256,args.width,args.height,args.decision)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(profile,indent=2),encoding='utf-8')
    args.output.with_suffix('.metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
