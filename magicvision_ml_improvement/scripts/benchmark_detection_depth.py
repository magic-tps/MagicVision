"""Measured local inference only: legacy detector + official depth, no campus accuracy claim."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys
import time
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];APP=ROOT.parent/'01_app_magicvision_final/app_flutter_magicvision'
sys.path.insert(0,str(APP/'backend/sagemaker'))
from detection_depth import UltralyticsDepthEstimator,FrameIdentity,DetectionDepthFusion,RiskEngine
from campus22_schema import read_manifest,read_labels,save_json,sha


def main():
    import torch,ultralytics
    from ultralytics import YOLO
    device='cuda' if torch.cuda.is_available() else 'cpu'
    weights=APP/'runs/detect/modelo_final_yolo26m_80e_640_b8/weights/best.pt'
    detector=YOLO(str(weights));depth=UltralyticsDepthEstimator(ROOT/'models/yolo26m-depth.pt',device=device)
    rows=read_manifest(ROOT/'datasets/campus22_v1/manifest.jsonl');selected=[]
    for cls in (5,13,14):
        selected.extend([r for r in rows if cls in {b[0] for b in read_labels(r['label'])}][:2])
    samples=[]
    for i,row in enumerate(selected):
        with Image.open(row['image']) as image:image=image.convert('RGB');image.thumbnail((640,640))
        identity=FrameIdentity.create(image,str(i),row['id'],'2026-09-22T00:00:00+00:00','dataset_diagnostic')
        samples.append((image,identity))
    def detect(image):
        start=time.perf_counter();result=detector.predict(image,imgsz=640,device=device,conf=.3,verbose=False,save=False)[0]
        objects=[{'class_id':int(b.cls[0]),'class_name':detector.names[int(b.cls[0])],'confidence':float(b.conf[0]),
                  'bbox':dict(zip(('left','top','right','bottom'),map(float,b.xyxyn[0].tolist())))} for b in result.boxes]
        return objects,(time.perf_counter()-start)*1000
    detect(samples[0][0]);depth.estimate(*samples[0])
    results={}
    for mode in ('sequential','concurrent'):
        if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        entries=[]
        with ThreadPoolExecutor(max_workers=2) as pool:
            for image,identity in samples:
                start=time.perf_counter()
                if mode=='sequential':
                    objects,detection_ms=detect(image);estimated=depth.estimate(image,identity)
                else:
                    a=pool.submit(detect,image);b=pool.submit(depth.estimate,image,identity)
                    objects,detection_ms=a.result();estimated=b.result()
                fusion_start=time.perf_counter();objects,diagnostics=DetectionDepthFusion().fuse(objects,estimated,identity)
                objects=[RiskEngine().assess(o) for o in objects];fusion_ms=(time.perf_counter()-fusion_start)*1000
                if device=='cuda':torch.cuda.synchronize()
                entries.append({'frame_id':identity.frame_id,'width':image.width,'height':image.height,
                    'detection_ms':detection_ms,'depth_ms':estimated.inference_ms,'fusion_ms':fusion_ms,
                    'total_ms':(time.perf_counter()-start)*1000,'depth_min':float(np.nanmin(estimated.depth_map)),
                    'depth_max':float(np.nanmax(estimated.depth_map)),'diagnostics':diagnostics,'detections':objects})
        average=float(np.mean([r['total_ms'] for r in entries]));results[mode]={'samples':entries,'mean_total_ms':average,
            'processing_fps_reciprocal':1000/average,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
            'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else None}
    report={'scope':'Local six-image diagnostic, legacy 10-class detector; no campus22 checkpoint, no camera calibration, no HTTP latency',
        'torch':torch.__version__,'ultralytics':ultralytics.__version__,'device':device,
        'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,'detector_sha256':sha(weights),
        'depth_sha256':depth.model_sha256,'metric_error':None,'training_performed':False,'results':results,
        'parallel_enabled_in_production':False}
    save_json(ROOT/'reports/detection_depth_benchmark.json',report)
    print(json.dumps({mode:{k:v for k,v in r.items() if k!='samples'} for mode,r in results.items()},indent=2))


if __name__=='__main__':main()
