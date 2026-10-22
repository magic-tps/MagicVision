"""Profile a training batch only for an approved campus22 release; no epochs."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import campus22_schema as schema
from preflight import inspect_release


def probe(data,weights,output,imgsz=640, *, min_priority_images=50, existing_labels=False):
    if existing_labels:
        from existing_labels import inspect_existing
        status=inspect_existing(data)
    else:
        status=inspect_release(data,schema=schema,min_priority_images=min_priority_images)
    if not status['ready']:
        schema.save_json(output,{'performed':False,'reason':'Dataset review incomplete','preflight':status})
        return False
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.autobatch import check_train_batch_size
    if not torch.cuda.is_available():raise ValueError('CUDA required for GPU batch profiling')
    base=YOLO(str(weights),task='detect')
    if 'yolo26m' not in str(base.model.yaml.get('yaml_file','')):raise ValueError('YOLO26m required')
    model=DetectionModel(base.model.yaml,nc=22,verbose=False)
    # Ultralytics load() updates the model in place and may return None.
    model.load(base.model)
    model=model.cuda()
    rows=[r for r in schema.read_manifest(Path(data).parent/'manifest.jsonl') if r['split']=='train']
    max_objects=max(len(schema.read_labels(r['label'])) for r in rows)
    estimate=check_train_batch_size(model,imgsz=imgsz,amp=True,batch=.6,max_num_obj=max_objects,dataset_size=len(rows))
    schema.save_json(output,{'performed':True,'suggested_batch':estimate,'memory_fraction':.6,'imgsz':imgsz,
        'gpu':torch.cuda.get_device_name(0),'vram_bytes':torch.cuda.get_device_properties(0).total_memory,
        'torch':torch.__version__,'manifest_sha256':schema.sha(Path(data).parent/'manifest.jsonl'),
        'weights_sha256':schema.sha(weights),'note':'Profiler estimate, not a guaranteed maximum. Confirm during initial training; restart with smaller batch after OOM.'})
    return True


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--weights',type=Path,default=schema.ROOT/'models/yolo26m.pt');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--imgsz',type=int,choices=[640,768,960],default=640)
    a=p.parse_args();sys.exit(0 if probe(a.data,a.weights,a.output,a.imgsz) else 2)
