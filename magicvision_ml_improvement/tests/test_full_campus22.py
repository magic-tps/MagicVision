from copy import deepcopy
from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'training'))
import campus22_schema as schema
from data_pipeline import record
from prepare_full_campus22 import merge_rows, select_records, prepare
from existing_labels import inspect_existing
from preflight import inspect_release


def inventory(tmp_path):
    rows = []
    for subset, color, phash in [('train', 'red', '0'*16), ('val', 'green', 'f'*16), ('test', 'blue', 'a'*16)]:
        image=tmp_path/f'{subset}.png'; Image.new('RGB',(40,40),color).save(image)
        label=tmp_path/f'{subset}.txt';schema.write_labels(label,[[c,.5,.5,.2,.2] for c in range(22)])
        row=record(image,label,'legacy' if subset=='train' else 'added_source',group=subset,
                   original_split=subset,taxonomy=schema.TAXONOMY)
        row.update(resolved_split=subset, group_id=subset, phash=phash, source_video_info=None)
        rows.append(row)
    return rows


def reviewed_copy(row, tmp_path):
    result = deepcopy(row)
    label=tmp_path/'corrected.txt'
    schema.write_labels(label,[[c,.4,.4,.1,.1] for c in range(22)])
    result.update(label=str(label), review_status='approved', coverage_complete=True,
                  provenance_reviewed=True, reviewer='test fixture', reviewed_at='2026-09-23T00:00:00Z',
                  reviewed_image_sha256=row['sha256'], reviewed_label_sha256=schema.sha(label))
    return result


def test_full_release_keeps_old_new_and_corrections_without_inventing_reviews(tmp_path):
    rows=inventory(tmp_path); corrected=reviewed_copy(rows[0],tmp_path)
    original_labels={r['id']:Path(r['label']).read_bytes() for r in rows}
    source=tmp_path/'source.jsonl';reviews=tmp_path/'reviews.jsonl'
    schema.write_manifest(source,rows);schema.write_manifest(reviews,[corrected])
    out=tmp_path/'full';report=prepare(source,reviews,out)
    assert report['images']==3 and report['legacy_images']==1 and report['added_images']==2
    released=schema.read_manifest(out/'manifest.jsonl')
    assert {r['id'] for r in released}=={r['id'] for r in rows}
    byid={r['id']:r for r in released}
    assert schema.sha(byid[corrected['id']]['label'])==corrected['reviewed_label_sha256']
    for row in rows[1:]:
        assert byid[row['id']]['review_status']==row['review_status']
        assert not byid[row['id']]['coverage_complete']
    assert all(Path(r['label']).read_bytes()==original_labels[r['id']] for r in rows)
    assert inspect_existing(out/'data.yaml')['ready']
    # Experimental acceptance must not weaken the original human-review gate.
    assert not inspect_release(out/'data.yaml',min_priority_images=1,schema=schema)['ready']
    Path(released[0]['label']).write_text('0 .5 .5 .2 .2\n')
    assert not inspect_existing(out/'data.yaml')['ready']


def test_empty_and_conflicting_duplicate_labels_are_excluded(tmp_path):
    rows=inventory(tmp_path)
    rows=merge_rows(rows,[])
    Path(rows[1]['label']).write_text('')
    duplicate={**rows[0], 'id':'conflicting', 'label':str(tmp_path/'conflict.txt')}
    schema.write_labels(duplicate['label'],[[0,.5,.5,.1,.1]])
    selected,excluded=select_records(rows+[duplicate])
    assert [r['id'] for r in selected]==[rows[2]['id']]
    assert {r['reason'] for r in excluded}=={'empty_labels_without_negative_verification',
                                           'conflicting_annotations_on_identical_image'}


def test_outdated_review_cannot_override_source(tmp_path):
    rows=inventory(tmp_path);review=reviewed_copy(rows[0],tmp_path)
    Path(review['label']).write_text('0 .5 .5 .2 .2\n')
    with pytest.raises(ValueError,match='outdated review'):
        merge_rows(rows,[review])
