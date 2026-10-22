import sys
from pathlib import Path
import pytest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import campus_schema as campus
import data_pipeline as legacy
from review_dataset import export,decide,apply,render


def test_new_schema_preserves_old_ids_and_is_not_used_by_active_pipeline(tmp_path):
    assert campus.CLASSES[:11]==legacy.CLASSES
    assert campus.CLASSES[24]=='fire_extinguisher'
    label=tmp_path/'box.txt';campus.write_labels(label,[[24,.5,.5,.2,.2]])
    assert campus.read_labels(label)[0][0]==24
    with pytest.raises(ValueError):legacy.read_labels(label)
    with pytest.raises(ValueError):campus.validate_box([25,.5,.5,.2,.2])


def test_25class_review_editor_and_applied_hashes(tmp_path):
    image=tmp_path/'img.png';Image.new('RGB',(40,60),'red').save(image)
    label=tmp_path/'img.txt';campus.write_labels(label,[[24,.5,.5,.2,.2]])
    row=campus.record(image,label,'fixture');manifest=tmp_path/'manifest.jsonl'
    campus.write_manifest(manifest,[row]);package=tmp_path/'review'
    info=export([manifest],package,limit=1,taxonomy='campus-v4')
    assert len(info['classes'])==25
    assert 'fire_extinguisher' in (package/'index.html').read_text(encoding='utf-8')
    campus.write_labels(package/info['items'][0]['label'],[[23,.5,.5,.3,.3]])
    render(package)
    decide(package,row['id'],'approve','test_fixture_only','Fixture coverage check',complete=True,provenance_reviewed=True)
    out=tmp_path/'applied';apply(package,out)
    reviewed=campus.read_manifest(out/'manifest.jsonl')[0]
    assert campus.read_labels(reviewed['label'])[0][0]==23
    assert campus.sha(reviewed['label'])==reviewed['reviewed_label_sha256']
    assert not row['coverage_complete']


def test_unannotated_candidates_enter_class_queue_without_inventing_boxes(tmp_path):
    image=tmp_path/'lift.png';Image.new('RGB',(40,60),'blue').save(image)
    label=tmp_path/'lift.txt';campus.write_labels(label,[])
    row=campus.record(image,label,'candidate',candidate_classes=['elevator'],annotation_status='unannotated')
    manifest=tmp_path/'manifest.jsonl';campus.write_manifest(manifest,[row])
    package=tmp_path/'review'
    info=export([manifest],package,limit=1,classes=['elevator'],selection='balanced',taxonomy='campus-v4')
    assert info['rows_exported']==1
    assert campus.read_labels(package/info['items'][0]['label'])==[]
    assert info['items'][0]['provenance']['annotation_status']=='unannotated'
    with pytest.raises(ValueError,match='verified-negative'):
        decide(package,row['id'],'approve','fixture','Fixture only',complete=True,provenance_reviewed=True)
