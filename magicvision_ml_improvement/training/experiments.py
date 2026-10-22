"""Hardware audit, explicit 10→11-class transfer inspection and gated training."""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from data_pipeline import ROOT,APP,CLASSES,read_manifest,read_labels,save_json,sha
from preflight import inspect_release


def hardware():
    import torch,ultralytics,psutil
    cuda=torch.cuda.is_available()
    return {'torch':torch.__version__,'ultralytics':ultralytics.__version__,'cuda_runtime':torch.version.cuda,
        'cuda_available':cuda,'gpu':torch.cuda.get_device_name(0) if cuda else None,
        'vram_bytes':torch.cuda.get_device_properties(0).total_memory if cuda else 0,
        'ram_available_bytes':psutil.virtual_memory().available,'disk_free_bytes':shutil.disk_usage(ROOT).free}


def transfer(weights,output,classes=CLASSES):
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    import torch
    source=YOLO(str(weights));target=DetectionModel(source.model.yaml,nc=len(classes),verbose=False)
    old=source.model.float().state_dict();new=target.state_dict()
    copied={k:v for k,v in old.items() if k in new and v.shape==new[k].shape}
    result=target.load_state_dict(copied,strict=False)
    missing=list(result.missing_keys)
    save_json(output,{'source_sha256':sha(weights),'old_classes':source.names,'new_classes':classes,'transferred_keys':list(copied),
        'initialized_keys':missing,'source_only_keys':[k for k in old if k not in new],
        'changed_shapes':{k:{'old':list(old[k].shape),'new':list(new[k].shape)} for k in new if k in old and old[k].shape!=new[k].shape},
        'trained':False,'note':'Shape-compatible backbone/head tensors transferred; changed class outputs remain initialized. This is a transfer probe, not a trained model.'})
    print(f'Transfer probe: {len(copied)} tensors copied; {len(missing)} initialized')


def train(weights,dataset,experiment):
    import yaml
    readiness=inspect_release(dataset)
    if not readiness['ready']:
        raise ValueError('Training preflight failed: '+ '; '.join(readiness['errors'][:8]))
    from ultralytics import YOLO
    info=hardware();data_path=Path(dataset)
    report_path=data_path.parent/'split_report.json'
    if not report_path.exists() or not json.loads(report_path.read_text()).get('training_ready'):
        raise ValueError('Dataset not released: complete human review and all-class grouped split required')
    rows=read_manifest(data_path.parent/'manifest.jsonl')
    if not all(r['review_status']=='approved' and r.get('coverage_complete') and r.get('reviewer') and r.get('reviewed_at') and r.get('provenance_reviewed') for r in rows):raise ValueError('Review identity/date and provenance approval required for every image')
    independent={name:set() for name in CLASSES}
    for r in rows:
        if r['split']=='train':
            for c in {b[0] for b in read_labels(r['label'])}:independent[CLASSES[c]].add(r.get('derived_from',r['sha256']))
    images={name:len(ids) for name,ids in independent.items()}
    if images['stairs']<50 or images['trash_bin']<50:raise ValueError('Pilot minimum: 50 distinct reviewed training images per new/rare priority class; not a reliability guarantee')
    if not info['cuda_available']:raise ValueError('CUDA unavailable; configuration prepared, no long CPU training started')
    config=yaml.safe_load((ROOT/'training/experiment_config.yaml').read_text())
    settings=config['experiments'][experiment];imgsz=settings['imgsz']
    gib=info['vram_bytes']/1024**3
    if imgsz==960 and gib<7.5:raise ValueError('960 experiment needs at least 8 GiB nominal VRAM and batch probe')
    run=ROOT/'training/runs'/f'{experiment}_{int(time.time())}'
    run.mkdir(parents=True)
    save_json(run/'environment.json',{**info,'weights_sha256':sha(weights),'dataset_manifest_sha256':sha(data_path.parent/'manifest.jsonl'),'training_images_per_class':images})
    transfer(weights,run/'transfer.json')
    model=YOLO(str(weights));started=time.perf_counter()
    # Fixed optimizer and seed make E1/E2 comparable; batch selected for measured GPU capacity.
    kwargs={**config['common'],**settings,'data':str(data_path.resolve()),'device':0,'batch':2 if imgsz==960 else 4,
        'project':str(run),'name':'train','exist_ok':False}
    save_json(run/'arguments.json',kwargs)
    try:model.train(**kwargs)
    finally:save_json(run/'duration.json',{'seconds':time.perf_counter()-started})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['hardware','transfer','train']);p.add_argument('--weights',type=Path,default=APP/'runs/detect/modelo_final_yolo26m_80e_640_b8/weights/best.pt');p.add_argument('--data',type=Path,default=ROOT/'datasets/magicvision_v2/data.yaml');p.add_argument('--experiment',choices=['E1','E2','E3'],default='E1')
    a=p.parse_args()
    if a.command=='hardware':r=hardware();save_json(ROOT/'reports/hardware.json',r);print(json.dumps(r))
    elif a.command=='transfer':transfer(a.weights,ROOT/'reports/transfer_probe.json')
    else:train(a.weights,a.data,a.experiment)
