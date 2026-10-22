"""Release/check/train the 22-class campus dataset using existing YOLO26 weights."""
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import campus22_schema as schema
from data_pipeline import split, APP
from preflight import inspect_release
from experiments import hardware
from training_progress import TrainingProgress


def release(manifest, output):
    rows=schema.read_manifest(manifest)
    if any(r.get('taxonomy') != schema.TAXONOMY for r in rows):
        raise ValueError('Use the reviewed campus22 manifest, not a 25-class inventory')
    # Existing splitter preserves source partitions and groups and rejects conflicts.
    split(rows,output,schema=schema)
    result=inspect_release(Path(output)/'data.yaml',schema=schema)
    result['source_inventory_records']=len(rows)
    result['source_approved_complete_records']=sum(r.get('review_status')=='approved' and r.get('coverage_complete') is True for r in rows)
    schema.save_json(Path(output)/'preflight.json',result)
    return result


def check(dataset, *, pilot=False, existing_labels=False):
    if existing_labels:
        if pilot:
            raise ValueError('Choose pilot or existing-labels mode, not both')
        from existing_labels import inspect_existing
        return inspect_existing(dataset)
    result = inspect_release(dataset, schema=schema, min_priority_images=40 if pilot else 50)
    if pilot:
        # Only the explicitly prepared fixed-data pilot can use the lower sample minimum.
        try:
            root = Path(dataset).resolve().parent
            evidence = json.loads((root / 'preparation_report.json').read_text(encoding='utf-8'))
            if evidence['released_manifest_sha256'] != schema.sha(root / 'manifest.jsonl'):
                raise ValueError('Pilot manifest changed since preparation')
            if evidence.get('pilot_minimum_priority_images') != 40:
                raise ValueError('Pilot sample policy is missing')
            result['limitations'] = evidence['limitations']
        except (OSError, ValueError, KeyError) as exc:
            result['errors'].append('Invalid pilot preparation evidence: ' + str(exc))
            result['ready'] = False
        result['evaluation_scope'] = 'fixed-data pilot with visually inferred FireNet scenes'
    return result


def train(dataset, weights, report, resume=None, *, pilot=False, existing_labels=False):
    if pilot and existing_labels:
        raise ValueError('Choose pilot or existing-labels mode, not both')
    print('Comprobando etiquetas, aprobaciones y separacion del dataset...', flush=True)
    readiness=(check(dataset, existing_labels=True) if existing_labels
               else check(dataset, pilot=True) if pilot else check(dataset))
    schema.save_json(report,readiness)
    if not readiness['ready']:
        return readiness
    if existing_labels:
        print(f'Dataset completo: {readiness["records"]} imagenes | '
              f'Anteriores: {readiness["legacy_images"]} | Nuevas: {readiness["added_images"]}', flush=True)
        print(f'Distribucion: {readiness["images_by_split"]} | '
              f'Revisadas: {readiness["human_reviewed_images"]} | '
              f'Etiquetas originales: {readiness["inherited_annotation_images"]}', flush=True)
        print('Modo experimental: las etiquetas originales pueden estar incompletas para las 22 clases.', flush=True)
        print(f'Clases ausentes en evaluacion: {readiness["missing_classes_by_split"]}', flush=True)
    from ultralytics import YOLO
    model=YOLO(str(resume or weights), task='detect')
    architecture=str(model.model.yaml.get('yaml_file','')).lower()
    if 'yolo26m' not in architecture or model.task != 'detect':
        raise ValueError('Expected YOLO26m detection checkpoint with verifiable architecture')
    info=hardware()
    if not info['cuda_available']:
        raise ValueError('CUDA unavailable; no long CPU training started')
    print(f'GPU: {info.get("gpu", "CUDA disponible")}', flush=True)
    config=yaml.safe_load((schema.ROOT/'training/campus22_yolo26.yaml').read_text(encoding='utf-8'))
    if resume:
        if [model.names[i] for i in range(len(model.names))] != schema.CLASSES:
            raise ValueError('Resume requires a campus22 detection checkpoint')
        evidence = json.loads(Path(resume).parents[1].joinpath('magicvision_resume.json').read_text(encoding='utf-8'))
        if evidence['manifest_sha256'] != schema.sha(Path(dataset).parent/'manifest.jsonl'):
            raise ValueError('Reviewed dataset changed since training; start a new experiment')
        summary = TrainingProgress().run(model, resume=True)
        return {**readiness, 'training_summary': summary}
    from batch_probe import probe
    batch_report = Path(report).with_name('campus22_batch_probe.json')
    probe_options = {'existing_labels': True} if existing_labels else {'min_priority_images': 40} if pilot else {}
    print('Calculando el batch para la GPU. El entrenamiento aun no ha comenzado...', flush=True)
    if not probe(dataset, weights, batch_report, config['train']['imgsz'], **probe_options):
        blocked = {**readiness, 'ready': False, 'training_started': False,
                   'errors': [*readiness.get('errors', []),
                              'Batch profiling rejected training; see ' + str(batch_report)]}
        schema.save_json(report, blocked)
        return blocked
    proposed = json.loads(batch_report.read_text(encoding='utf-8'))['suggested_batch']
    config['train']['batch'] = min(config['train']['batch'], proposed)
    run_name='campus22_yolo26m_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    kwargs={**config['train'],'data':str(Path(dataset).resolve()),'device':0,
            'project':str(schema.ROOT/'training/runs'),'name':run_name,'exist_ok':False,'pretrained':True}
    def record_resume(trainer):
        schema.save_json(Path(trainer.save_dir)/'magicvision_resume.json', {
            'manifest_sha256':schema.sha(Path(dataset).parent/'manifest.jsonl'), 'taxonomy':schema.TAXONOMY,
            'pilot': pilot, 'existing_labels': existing_labels, 'preflight': readiness})
    model.add_callback('on_pretrain_routine_end', record_resume)
    schema.save_json(Path(report).with_name(run_name+'_environment.json'),{
        **info,'weights_sha256':schema.sha(weights),'architecture':architecture,
        'manifest_sha256':schema.sha(Path(dataset).parent/'manifest.jsonl'),'arguments':kwargs,
        'pilot': pilot, 'existing_labels': existing_labels, 'preflight': readiness})
    # Ultralytics reconstructs the detection head to match data.yaml's 22 names.
    summary = TrainingProgress().run(model, **kwargs)
    return {**readiness, 'training_summary': summary}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['release','check','train'])
    parser.add_argument('--manifest',type=Path,default=schema.ROOT/'datasets/campus22_v1/manifest.jsonl')
    parser.add_argument('--output',type=Path,default=schema.ROOT/'datasets/campus22_release_depth_v1')
    parser.add_argument('--data',type=Path,default=schema.ROOT/'datasets/campus22_release_depth_v1/data.yaml')
    parser.add_argument('--weights',type=Path,default=schema.ROOT/'models/yolo26m.pt')
    parser.add_argument('--resume',type=Path,help='Existing campus22 last.pt; reviewed manifest must match')
    parser.add_argument('--pilot',action='store_true',help='Use the documented fixed-data pilot (40 priority images, scene-based FireNet grouping)')
    parser.add_argument('--existing-labels',action='store_true',help='Use all existing source annotations plus human corrections; experimental evaluation')
    parser.add_argument('--report',type=Path,default=schema.ROOT/'reports/campus22_training_preflight.json')
    args=parser.parse_args()
    if args.command=='release':result=release(args.manifest,args.output)
    elif args.command=='check':result=check(args.data,pilot=args.pilot,existing_labels=args.existing_labels);schema.save_json(args.report,result)
    else:result=train(args.data,args.weights,args.report,args.resume,pilot=args.pilot,existing_labels=args.existing_labels)
    if args.command != 'train' or not result['ready']:
        print(json.dumps(result,ensure_ascii=False,indent=2))
    sys.exit(0 if result['ready'] else 2)
