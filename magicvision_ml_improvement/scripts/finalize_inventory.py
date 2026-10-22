"""Recompute acquisition counts, review gates and balanced inspection previews."""
import csv
import json
from collections import Counter,defaultdict
from data_pipeline import ROOT,CLASSES,read_manifest,read_labels,quality,split,save_json,write_manifest,duplicate_groups


def main():
    legacy=read_manifest(ROOT/'datasets/legacy_manifest.jsonl')
    sources={name:read_manifest(ROOT/'datasets'/name/'manifest.jsonl') for name in
             ('openimages','openimages_train','openimages_negative_candidates')}
    new=[r for rows in sources.values() for r in rows];allrows=legacy+new
    oldhash={r['sha256'] for r in legacy};classes=defaultdict(set);newboxes=Counter();oldboxes=Counter();approved=defaultdict(set)
    for r in legacy:oldboxes.update(CLASSES[b[0]] for b in read_labels(r['label']))
    for r in new:
        labels=read_labels(r['label']);newboxes.update(CLASSES[b[0]] for b in labels)
        if r['sha256'] not in oldhash:classes[r['sha256']].update(CLASSES[b[0]] for b in labels)
        if r['review_status']=='approved' and r.get('coverage_complete'):
            for c in {b[0] for b in labels}:approved[CLASSES[c]].add(r['sha256'])
    images=Counter(c for labels in classes.values() for c in labels)
    negative={r['sha256'] for r in new if r.get('negative_candidate') and r['sha256'] not in oldhash}
    preview=[];seen=set()
    # Inspect different priority categories, not only the first person images.
    for c in (5,10,0):
        picked=0
        for r in new:
            if r['sha256'] not in seen and c in {b[0] for b in read_labels(r['label'])}:
                preview.append(r);seen.add(r['sha256']);picked+=1
                if picked==6:break
    for r in sources['openimages_negative_candidates'][:6]:
        if r['sha256'] not in seen:preview.append(r);seen.add(r['sha256'])
    ordered=preview+[r for r in allrows if r not in preview]
    report=quality(ordered,ROOT/'reports/quality_final',overlays=len(preview))
    clusters=duplicate_groups(allrows)
    cross=[g for g in clusters if len({allrows[i].get('original_split') for i in g}-{None})>1]
    inventory={'legacy_records':len(legacy),'new_records':len(new),
      'new_unique_sha256':len({r['sha256'] for r in new}),
      'new_exact_overlap_legacy':len({r['sha256'] for r in new}&oldhash),
      'new_unique_sha256_excluding_legacy':len(classes),
      'images_per_class_excluding_exact_legacy_duplicates':dict(images),
      'source_counts':{name:len(rows) for name,rows in sources.items()},
      'annotation_boxes_imported':dict(newboxes),'annotation_boxes_legacy':dict(oldboxes),
      'candidate_negatives_unique':len(negative),'verified_negatives':sum(r.get('verified_negative',False) for r in new),
      'approved_new':sum(r['review_status']=='approved' and r.get('coverage_complete',False) for r in new),
      'legacy_cross_split_clusters':len(cross),'quality_errors':len(report['errors']),
      'conflicting_exact_duplicates':len(report['conflicting_exact_duplicates']),
      'duplicate_or_session_clusters':len(report['duplicate_or_session_clusters']),
      'note':'Acquired candidates, not a released training dataset. pHash/sessions may reveal additional near-duplicates. Class counts overlap.'}
    save_json(ROOT/'reports/dataset_inventory.json',inventory)
    write_manifest(ROOT/'datasets/combined_manifest.jsonl',allrows)
    with (ROOT/'reports/dataset_distribution.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['id','class','legacy_boxes','new_imported_boxes','new_unique_images_excluding_exact_legacy','approved_new_images'])
        for i,c in enumerate(CLASSES):writer.writerow([i,c,oldboxes[c],newboxes[c],images[c],len(approved[c])])
    destination=ROOT/'datasets/magicvision_v2'
    if not destination.exists() or not any(destination.iterdir()):
        print('Training ready:',split(allrows,destination))
    print(json.dumps(inventory,ensure_ascii=False))


if __name__=='__main__':main()
