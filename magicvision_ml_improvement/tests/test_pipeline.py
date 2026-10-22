import json
import sys
from pathlib import Path
import pytest
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'evaluation'))
from data_pipeline import CLASSES, validate_box, from_xyxy, dedup_boxes, import_coco, read_manifest, quality, split, write_labels, duplicate_groups
from evaluate_models import match,metrics


@pytest.mark.parametrize('box',[[11,.5,.5,.2,.2],[10.5,.5,.5,.2,.2],[0,.1,.1,.5,.5],[0,.5,.5,0,.1],[0,float('nan'),.5,.1,.1]])
def test_invalid_box_rejected(box):
    with pytest.raises(ValueError):validate_box(box)


def test_ids_geometry_and_duplicates():
    assert CLASSES.index('trash_bin')==10 and CLASSES.index('obstacle')==9
    box=from_xyxy(10,.2,.3,.6,.9)
    assert box[1:]==pytest.approx([.4,.6,.4,.6])
    assert len(dedup_boxes([box,box]))==1


def test_coco_maps_explicitly_and_quarantines_missing_labels(tmp_path):
    Image.new('RGB',(100,200)).save(tmp_path/'scene.jpg')
    annotation={'images':[{'id':1,'file_name':'scene.jpg','width':100,'height':200}],
      'categories':[{'id':100,'name':'bin'},{'id':101,'name':'unknown'}],
      'annotations':[{'image_id':1,'category_id':100,'bbox':[20,60,40,80]},
       {'image_id':1,'category_id':101,'bbox':[0,0,10,10]}]}
    (tmp_path/'input.json').write_text(json.dumps(annotation))
    dest=tmp_path/'manifest.jsonl'
    import_coco(tmp_path/'input.json',tmp_path,{'bin':'trash_bin'},dest,'fixture')
    row=read_manifest(dest)[0]
    assert row['review_status']=='needs_review' and not row['coverage_complete']
    assert Path(row['label']).read_text().startswith('10 ')
    assert len(Path(row['label']).read_text().splitlines())==1
    assert not split([row],tmp_path/'split')
    assert read_manifest(tmp_path/'split/manifest.jsonl')==[]


def test_session_and_visual_duplicates_stay_together():
    rows=[{'sha256':'a','group':'route1','phash':'0000000000000000'},
          {'sha256':'b','group':'route1','phash':'ffffffffffffffff'},
          {'sha256':'c','group':'route2','phash':'0000000000000001'}]
    assert duplicate_groups(rows)==[[0,1,2]]


def test_quality_flags_unverified_negatives(tmp_path):
    from data_pipeline import record
    image=tmp_path/'empty.jpg';Image.new('RGB',(40,60)).save(image)
    label=tmp_path/'empty.txt';write_labels(label,[])
    r=quality([record(image,label,'test')],tmp_path/'report')
    assert r['approved_complete']==0 and r['review_queue']
    assert (tmp_path/'report/index.html').exists()


def test_duplicate_predictions_and_wrong_classes_are_not_true_positives():
    truth=[{'class_id':10,'bbox':[.1,.1,.9,.9]}]
    predictions=[{'class_id':10,'confidence':.9,'bbox':[.1,.1,.9,.9]},
                 {'class_id':10,'confidence':.8,'bbox':[.1,.1,.9,.9]},
                 {'class_id':9,'confidence':.95,'bbox':[.1,.1,.9,.9]}]
    matched,used=match(predictions,truth)
    assert sum(correct for _,correct in matched)==1 and used=={0}
    result=metrics([{'truth':truth,'predictions':predictions}])
    assert result['classes']['trash_bin']['tp']==1
    assert result['classes']['trash_bin']['fp']==1
    assert result['classes']['obstacle']['fp']==1
    assert result['classes']['stairs']['recall'] is None


def test_all_classes_represented_and_holdout_never_moves(tmp_path):
    from data_pipeline import record
    rows=[]
    for s,color in [('train','red'),('val','green'),('test','blue')]:
        image=tmp_path/f'{s}.jpg';Image.new('RGB',(40,60),color).save(image)
        label=tmp_path/f'{s}.txt';write_labels(label,[[c,.5,.5,.2,.2] for c in range(11)])
        row=record(image,label,'test',group=s,original_split=s,review_status='approved',coverage_complete=True)
        # Flat-color pHashes coincide: simulate distinct hashes to test split constraints separately.
        row['phash']={'train':'0'*16,'val':'f'*16,'test':'a'*16}[s];rows.append(row)
    assert split(rows,tmp_path/'release')
    assert {r['original_split']:r['split'] for r in read_manifest(tmp_path/'release/manifest.jsonl')}=={'train':'train','val':'val','test':'test'}
    rows[1]['phash']=rows[0]['phash']
    assert not split(rows,tmp_path/'rejected')
    assert json.loads((tmp_path/'rejected/split_report.json').read_text())['rejected']


def test_audited_group_resolution_keeps_conflicting_source_rows_together(tmp_path):
    from data_pipeline import record
    rows=[]
    for subset,color,phash in [('train','red','0'*16),('val','green','f'*16),('test','blue','a'*16)]:
        image=tmp_path/f'base_{subset}.png';Image.new('RGB',(30,30),color).save(image)
        label=tmp_path/f'base_{subset}.txt';write_labels(label,[[c,.5,.5,.1,.1] for c in range(11)])
        row=record(image,label,'fixture',group=f'base_{subset}',original_split=subset,
                   review_status='approved',coverage_complete=True)
        row['phash']=phash;rows.append(row)
    for index,(subset,color) in enumerate([('train',(1,2,3)),('test',(4,5,6))]):
        image=tmp_path/f'conflict_{index}.png';Image.new('RGB',(31,31),color).save(image)
        label=tmp_path/f'conflict_{index}.txt';write_labels(label,[[0,.5,.5,.2,.2]])
        row=record(image,label,'fixture',group='shared',original_split=subset,
                   review_status='approved',coverage_complete=True,
                   resolved_split='test',split_resolution_policy='campus22-group-holdout-v1')
        row['phash']=('1' if index==0 else '2')*16;rows.append(row)
    assert split(rows,tmp_path/'resolved')
    released=read_manifest(tmp_path/'resolved/manifest.jsonl')
    shared=[row for row in released if row['group']=='shared']
    assert len(shared)==2 and {row['split'] for row in shared}=={'test'}


def test_conflicting_labels_are_not_silently_deduplicated(tmp_path):
    from data_pipeline import record
    image=tmp_path/'same.jpg';Image.new('RGB',(40,60)).save(image)
    rows=[]
    for c in (0,10):
        label=tmp_path/f'{c}.txt';write_labels(label,[[c,.5,.5,.2,.2]])
        rows.append(record(image,label,f'source{c}',review_status='approved',coverage_complete=True))
    assert quality(rows,tmp_path/'quality')['conflicting_exact_duplicates']
    assert not split(rows,tmp_path/'split')
    assert not read_manifest(tmp_path/'split/manifest.jsonl')


def test_subgroups_require_verified_evidence(tmp_path):
    from evaluate_subgroups import evaluate_subgroups
    prediction={'sha256':'abc','truth':[{'class_id':0,'bbox':[0,0,1,1]}],'predictions':[]}
    (tmp_path/'p.jsonl').write_text(json.dumps(prediction)+'\n')
    row={'id':'one','sha256':'abc','scene_tags':['person_far']}
    (tmp_path/'m.jsonl').write_text(json.dumps(row)+'\n')
    r=evaluate_subgroups(tmp_path/'p.jsonl',tmp_path/'m.jsonl',tmp_path/'out.json')
    assert r['groups']['person_far']['images']==0
    row.update(scene_tags_verified=True,tag_evidence='Measured along marked campus corridor; protocol A')
    (tmp_path/'m.jsonl').write_text(json.dumps(row)+'\n')
    r=evaluate_subgroups(tmp_path/'p.jsonl',tmp_path/'m.jsonl',tmp_path/'out.json')
    assert r['groups']['person_far']['classes']['person']['fn']==1


def test_augmentation_changes_train_only_and_preserves_labels(tmp_path):
    from augment_train import augment
    from data_pipeline import record
    rows=[]
    for s,color in [('train','red'),('val','green'),('test','blue')]:
        image=tmp_path/f'{s}.jpg';Image.new('RGB',(40,60),color).save(image)
        label=tmp_path/f'{s}.txt';write_labels(label,[[c,.5,.5,.2,.2] for c in range(11)])
        row=record(image,label,'test',group=s,original_split=s,review_status='approved',coverage_complete=True)
        row['phash']={'train':'0'*16,'val':'f'*16,'test':'a'*16}[s];rows.append(row)
    release=tmp_path/'release';assert split(rows,release)
    assert augment(release,tmp_path/'augmented')==1
    result=read_manifest(tmp_path/'augmented/manifest.jsonl')
    derived=[r for r in result if r.get('derived_from')]
    assert len(derived)==1 and derived[0]['split']=='train'
    assert derived[0]['group']=='train'
    for r in result:
        if r['split']!='train':assert r['sha256']==next(x['sha256'] for x in rows if x['original_split']==r['split'])
