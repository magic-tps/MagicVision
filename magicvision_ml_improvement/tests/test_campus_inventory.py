import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from PIL import Image
from campus_schema import record, write_labels, write_manifest
from campus_inventory import audit, combine


def test_versioned_inventory_only_writes_requested_report(tmp_path, monkeypatch):
    import campus_inventory
    image=tmp_path/'image.png';Image.new('RGB',(20,20)).save(image)
    label=tmp_path/'label.txt';write_labels(label,[[23,.5,.5,.5,.5]])
    manifest=tmp_path/'source.jsonl';write_manifest(manifest,[record(image,label,'cone')])
    monkeypatch.setattr(campus_inventory,'ROOT',tmp_path)
    historical=tmp_path/'reports/campus_v4_inventory.json'
    historical.parent.mkdir();historical.write_text('historical',encoding='utf-8')
    report=tmp_path/'reports/campus_v5_inventory.json'
    campus_inventory.run([manifest],tmp_path/'v5',manifest,report_path=report)
    assert historical.read_text(encoding='utf-8')=='historical'
    assert report.exists() and (tmp_path/'v5/quality.json').exists()


def test_conflicting_duplicate_annotations_are_preserved_not_counted_as_new_pixels(tmp_path):
    image=tmp_path/'image.png';Image.new('RGB',(20,20)).save(image)
    one=tmp_path/'one.txt';two=tmp_path/'two.txt'
    write_labels(one,[[11,.5,.5,.5,.5]])
    write_labels(two,[[11,.5,.5,.5,.5],[12,.5,.5,.2,.2]])
    rows=[record(image,one,'a'),record(image,two,'b')]
    report=audit(rows,set())
    assert report['unique_images']==1 and report['records']==2
    assert report['boxes_deduplicated_by_pixels_and_geometry']==2
    assert report['exact_duplicate_groups'][0]['label_conflict']
    assert report['classes'][11]['images_with_boxes']==1


def test_candidates_and_legacy_approval_do_not_become_25class_training_data(tmp_path):
    image=tmp_path/'image.png';Image.new('RGB',(20,20)).save(image)
    label=tmp_path/'empty.txt';write_labels(label,[])
    row=record(image,label,'candidate',candidate_classes=['elevator'],taxonomy='legacy11',
               review_status='approved',coverage_complete=True)
    manifest=tmp_path/'source.jsonl';write_manifest(manifest,[row])
    rows=combine([manifest]);report=audit(rows,set())
    assert report['classes'][22]['unannotated_candidates']==1
    assert report['classes'][22]['images_with_boxes']==0
    assert report['approved_complete_25_class_records']==0
    assert report['errors']==[]
