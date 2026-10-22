import json
import sys
from pathlib import Path
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from data_pipeline import record,write_labels,split,sha,read_manifest,write_manifest
from preflight import inspect_release


def fixture_release(tmp_path):
    rows=[]
    for subset,color in [('train','red'),('val','green'),('test','blue')]:
        image=tmp_path/f'{subset}.jpg';Image.new('RGB',(40,60),color).save(image)
        label=tmp_path/f'{subset}.txt';write_labels(label,[[c,.5,.5,.2,.2] for c in range(11)])
        row=record(image,label,'fixture',group=subset,original_split=subset,review_status='approved',coverage_complete=True,
                   reviewer='test_fixture_only',reviewed_at='2026-09-20T00:00:00Z',provenance_reviewed=True,
                   reviewed_image_sha256=sha(image),reviewed_label_sha256=sha(label))
        row['phash']={'train':'0'*16,'val':'f'*16,'test':'a'*16}[subset];rows.append(row)
    destination=tmp_path/'release';assert split(rows,destination)
    return destination


def test_valid_content_can_pass_and_label_tampering_blocks(tmp_path):
    release=fixture_release(tmp_path)
    assert inspect_release(release/'data.yaml',min_priority_images=1)['ready']
    label=next((release/'labels/train').glob('*.txt'));label.write_text('0 .5 .5 .1 .1\n')
    r=inspect_release(release/'data.yaml',min_priority_images=1)
    assert not r['ready'] and any('Label differs' in e for e in r['errors'])


def test_orphan_training_image_is_not_silently_trained(tmp_path):
    release=fixture_release(tmp_path)
    Image.new('RGB',(40,60),'black').save(release/'images/train/unreviewed.jpg')
    r=inspect_release(release/'data.yaml',min_priority_images=1)
    assert not r['ready'] and any('folder disagree' in e for e in r['errors'])


def test_manual_flag_without_review_hash_is_insufficient(tmp_path):
    release=fixture_release(tmp_path);rows=read_manifest(release/'manifest.jsonl')
    del rows[0]['reviewed_image_sha256'];write_manifest(release/'manifest.jsonl',rows)
    assert not inspect_release(release/'data.yaml',min_priority_images=1)['ready']


def test_photometry_retains_review_integrity_and_independent_scene_count(tmp_path):
    from augment_train import augment
    release=fixture_release(tmp_path);out=tmp_path/'augmented'
    assert augment(release,out)==1
    report=inspect_release(out/'data.yaml',min_priority_images=1)
    assert report['ready'],report['errors']
    assert report['independent_train_images']['stairs']==1
    assert report['images_per_class_by_split']['train']['stairs']==2
    rows=read_manifest(out/'manifest.jsonl')
    for subset in ('val','test'):
        original=next(r for r in read_manifest(release/'manifest.jsonl') if r['split']==subset)
        copied=next(r for r in rows if r['split']==subset)
        assert sha(original['image'])==sha(copied['image'])
