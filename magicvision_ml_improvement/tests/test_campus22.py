import json
from pathlib import Path
import sys
import pytest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
import campus_schema as old
import campus22_schema as new
from prepare_campus22 import prepare
from review_dataset import schema_for, export
import campus22


def row(tmp_path, name, boxes, candidates=()):
    image=tmp_path/(name+'.png');Image.new('RGB',(30,30),color=(len(name)*17,10,20)).save(image)
    label=tmp_path/(name+'.txt');old.write_labels(label,boxes)
    return old.record(image,label,name,candidate_classes=list(candidates))


def test_contract_remaps_tail_without_changing_first_eighteen():
    assert len(new.CLASSES)==22
    # Campus22 explicitly renames ID 10; historic sources retain trash_bin.
    assert new.CLASSES[:10] == old.CLASSES[:10]
    assert new.CLASSES[10] == 'trash_can' and old.CLASSES[10] == 'trash_bin'
    assert new.CLASSES[11:18] == old.CLASSES[11:18]
    boxes=[[i,.5,.5,.2,.2] for i in range(25)]
    actual=new.project_boxes(boxes)
    assert [b[0] for b in actual]==list(range(22))
    assert new.CLASSES[18:]==['construction_barrier','handrail','cone','fire_extinguisher']
    assert not new.EXCLUDED.intersection(new.CLASSES)
    with pytest.raises(ValueError):new.validate_box([22,.5,.5,.2,.2])


def test_projection_drops_removed_only_rows_and_preserves_mixed_labels(tmp_path):
    a=row(tmp_path,'cable',[[18,.5,.5,.2,.2]])
    b=row(tmp_path,'mixed',[[18,.5,.5,.2,.2],[20,.4,.4,.1,.1]])
    c=row(tmp_path,'elevator',[],['elevator'])
    d=row(tmp_path,'negative_candidate',[])
    source=tmp_path/'source.jsonl';old.write_manifest(source,[a,b,c,d]);before=old.sha(source)
    q=prepare(source,tmp_path/'active')
    kept=old.read_manifest(tmp_path/'active/manifest.jsonl')
    assert len(kept)==2 and q['excluded_records']==2
    assert new.read_labels(kept[0]['label'])==[[18,.4,.4,.1,.1]]
    assert all(not r['verified_negative'] and not r['coverage_complete'] for r in kept)
    assert old.sha(source)==before and len(old.read_labels(b['label']))==2
    with pytest.raises(ValueError):prepare(source,tmp_path/'active')


def test_review_editor_uses_22_names(tmp_path):
    a=row(tmp_path,'cone',[[23,.5,.5,.2,.2]])
    source=tmp_path/'source.jsonl';old.write_manifest(source,[a]);prepare(source,tmp_path/'active')
    info=export([tmp_path/'active/manifest.jsonl'],tmp_path/'review',limit=1,taxonomy='campus22')
    assert schema_for(info) is new
    assert new.read_labels(tmp_path/'review'/info['items'][0]['label'])[0][0]==20
    assert not (tmp_path/'review/decisions.jsonl').read_text().strip()


def test_unreviewed_data_cannot_start_yolo26_training(tmp_path,monkeypatch):
    a=row(tmp_path,'cone',[[23,.5,.5,.2,.2]])
    source=tmp_path/'source.jsonl';old.write_manifest(source,[a]);prepare(source,tmp_path/'active')
    status=campus22.release(tmp_path/'active/manifest.jsonl',tmp_path/'release')
    assert not status['ready'] and status['source_approved_complete_records']==0
    monkeypatch.setattr(campus22,'hardware',lambda:pytest.fail('Must reject before GPU use'))
    result=campus22.train(tmp_path/'release/data.yaml',tmp_path/'missing.pt',tmp_path/'check.json')
    assert not result['ready'] and (tmp_path/'check.json').exists()


def test_reviewed_22_class_release_passes_and_rejects_label_tampering(tmp_path):
    from data_pipeline import split
    from preflight import inspect_release
    rows=[]
    for subset,color,phash in [('train','red','0'*16),('val','green','f'*16),('test','blue','a'*16)]:
        image=tmp_path/(subset+'.png');Image.new('RGB',(30,30),color).save(image)
        label=tmp_path/(subset+'.txt');new.write_labels(label,[[c,.5,.5,.1,.1] for c in range(22)])
        item=old.record(image,label,'fixture',group=subset,original_split=subset,taxonomy=new.TAXONOMY,
            review_status='approved',coverage_complete=True,provenance_reviewed=True,reviewer='unit_test_fixture_only',
            reviewed_at='2026-09-22T00:00:00Z',reviewed_image_sha256=old.sha(image),reviewed_label_sha256=old.sha(label))
        item['phash']=phash;rows.append(item)
    destination=tmp_path/'release'
    assert split(rows,destination,schema=new)
    result=inspect_release(destination/'data.yaml',min_priority_images=1,schema=new)
    assert result['ready'],result['errors']
    label=next((destination/'labels/train').glob('*.txt'));label.write_text('18 .5 .5 .2 .2\n')
    result=inspect_release(destination/'data.yaml',min_priority_images=1,schema=new)
    assert not result['ready'] and any('Label differs' in error for error in result['errors'])


def test_rejected_batch_probe_cannot_report_training_success(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import batch_probe

    monkeypatch.setattr(campus22, 'check', lambda _: {'ready': True, 'errors': []})
    monkeypatch.setattr(campus22, 'hardware', lambda: {'cuda_available': True})
    model = SimpleNamespace(
        model=SimpleNamespace(yaml={'yaml_file': 'yolo26m.yaml'}), task='detect',
        train=lambda **_: pytest.fail('Rejected batch must not start training'))
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=lambda *a, **kw: model))
    monkeypatch.setattr(batch_probe, 'probe', lambda *a: False)
    report = tmp_path / 'preflight.json'
    result = campus22.train(tmp_path / 'data.yaml', tmp_path / 'weights.pt', report)
    assert result['ready'] is False
    assert result['training_started'] is False
    assert 'Batch profiling rejected' in result['errors'][0]
    assert json.loads(report.read_text()) == result
