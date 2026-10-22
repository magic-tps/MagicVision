"""Compare exact reserved frames at 640/768/960 only after release approval."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
import campus22_schema as schema
from preflight import inspect_release
from evaluate_models import evaluate


def compare(data,old,new,output,sizes=(640,),split='val'):
    readiness=inspect_release(data,schema=schema)
    if not readiness['ready']:raise ValueError('Reviewed release required: '+ '; '.join(readiness['errors']))
    from ultralytics import YOLO
    model=YOLO(str(new),task='detect')
    if [model.names[i] for i in range(len(model.names))]!=schema.CLASSES:raise ValueError('New checkpoint must have all campus22 classes')
    del model
    results=[];expected=None
    for size in sizes:
        for tag,weights in [('previous',old),('campus22',new)]:
            target=Path(output)/f'{tag}_{size}'
            evaluate(weights,data,target,imgsz=size,split=split,schema=schema)
            rows=[json.loads(line) for line in (target/'predictions.jsonl').read_text().splitlines()]
            hashes=[r['sha256'] for r in rows]
            if expected is not None and hashes!=expected:raise ValueError('Evaluation images changed between models')
            expected=hashes
            metrics=json.loads((target/'metrics_0.30.json').read_text())
            matrix=metrics['confusion_matrix_predicted_rows_true_columns'];ids=[13,14,22]
            # Other-class errors remain visible in the full 23x23 matrix.
            schema.save_json(target/'door_glass_door_confusion.json',{'labels':['door','glass_door','background'],
                'matrix':[[matrix[i][j] for j in ids] for i in ids],'full_matrix':'metrics_0.30.json'})
            results.append({'model':tag,'imgsz':size,'metrics':metrics,'runtime':json.loads((target/'run.json').read_text())})
    schema.save_json(Path(output)/'comparison.json',{'same_reserved_image_sha256':expected,'split':split,'results':results,
        'note':'Unsupported classes in the previous detector are coverage gaps. No automatic deployment or claim of improvement.'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sizes',type=int,nargs='+',choices=[640,768,960],default=[640]);p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--final-test',action='store_true')
    a=p.parse_args()
    if a.split=='test' and not a.final_test:p.error('Select model/thresholds on validation before --final-test')
    compare(a.data,a.old,a.new,a.output,a.sizes,a.split)
