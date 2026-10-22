"""Evaluate cached predictions only for human-verified scene tags; never infer meters."""
import argparse
import json
from pathlib import Path
from evaluate_models import metrics,save_json

TAGS=('person_near','person_far','person_occluded','stairs_up','stairs_down',
      'stairs_partial','bin_small','bin_large','multi_object')


def evaluate_subgroups(predictions,manifest,output,confidence=.30):
    records=[json.loads(line) for line in Path(predictions).read_text().splitlines() if line.strip()]
    rows=[json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    tags={};excluded=[]
    for row in rows:
        if not row.get('scene_tags'):continue
        if row.get('scene_tags_verified') is not True or not row.get('tag_evidence'):
            excluded.append(row['id']);continue
        tags.setdefault(row['sha256'],set()).update(set(row['scene_tags'])&set(TAGS))
    report={'confidence':confidence,'excluded_unverified_tags':excluded,'groups':{},
            'protocol':'Scene-level subsets: all annotated objects within each tagged scene are scored. Groups may overlap. Distance tags require recorded human evidence, never bbox area.'}
    for tag in TAGS:
        subset=[r for r in records if tag in tags.get(r['sha256'],set())]
        report['groups'][tag]={'status':'evaluated' if subset else 'pending_verified_examples',**metrics(subset,confidence)}
    save_json(output,report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--confidence',type=float,default=.30)
    a=p.parse_args();evaluate_subgroups(a.predictions,a.manifest,a.output,a.confidence)
