import csv
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import download_campus_openimages as campus


def test_candidates_exclude_existing_groups_and_depictions(tmp_path):
    source = tmp_path / 'boxes.csv'
    with source.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['ImageID', 'LabelName', 'IsGroupOf', 'IsDepiction'])
        writer.writeheader()
        for image_id, group, depiction in [('old', '0', '0'), ('real', '0', '0'),
                                          ('group', '1', '0'), ('drawing', '0', '1'), ('real', '0', '0')]:
            writer.writerow(dict(ImageID=image_id, LabelName='chair', IsGroupOf=group, IsDepiction=depiction))
    selected, available = campus.select_candidates(source, {'chair': 'Chair'}, {'chair': 20}, {'old'})
    assert selected == {'chair': ['real']} and available == {'chair': 1}


def test_taxonomy_does_not_invent_risks_or_material():
    assert campus.SOURCE_MAP['Door'] == 'door'
    assert campus.SOURCE_MAP['Backpack'] == 'backpack'
    assert campus.SOURCE_MAP['Desk'] == campus.SOURCE_MAP['Table'] == 'table_desk'
    for name in ('Glass', 'Handbag', 'Fence', 'Street light'):
        assert name not in campus.SOURCE_MAP
    assert campus.CLASSES[13] == 'door' and campus.CLASSES[17] == 'backpack'


def test_import_retains_old_and_new_labels_with_pending_review(tmp_path, monkeypatch):
    def fixture(url, image, max_bytes):
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (80, 60), 'red').save(image)
    monkeypatch.setattr(campus, 'download', fixture)
    base = dict(XMin='0.1', YMin='0.2', XMax='0.6', YMax='0.8', IsGroupOf='0',
                IsDepiction='0', IsOccluded='0', IsTruncated='0')
    plan = {'metadata': {'abc': {'License': 'https://creativecommons.org/licenses/by/2.0/',
                                 'Author': 'Fixture author'}},
            'names': {'d': 'Door', 'p': 'Person', 'b': 'Backpack'},
            'annotations': {'abc': [dict(base, LabelName=mid) for mid in ('d', 'p', 'b')]}}
    row = campus.fetch_record('abc', tmp_path, plan)
    assert {box[0] for box in campus.read_labels(row['label'])} == {0, 13, 17}
    assert row['review_status'] == 'needs_review' and row['coverage_complete'] is False
    assert row['provenance_reviewed'] is False and row['author'] == 'Fixture author'
    assert 'door_material_unresolved_review_glass_door' in row['flags']
    assert 'backpack_ground_relation_not_annotated' in row['flags']


def test_license_missing_prevents_download(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(campus, 'download', lambda *args: calls.append(args))
    with pytest.raises(ValueError, match='license'):
        campus.fetch_record('a', tmp_path, {'metadata': {'a': {'License': ''}}})
    assert calls == []


def test_source_hierarchy_overlap_is_flagged_without_deleting_boxes():
    attrs = [{'class': 'table_desk', 'source_class': name} for name in ('Table', 'Desk')]
    assert campus.hierarchy_review_flags(attrs) == ['source_hierarchy_cooccurrence_review_overlap']
    assert campus.hierarchy_review_flags(attrs[:1]) == []


def test_exact_table_desk_boxes_are_deduplicated_with_hierarchy_recorded(tmp_path, monkeypatch):
    def fixture(url, image, max_bytes):
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (80, 60), 'blue').save(image)
    monkeypatch.setattr(campus, 'download', fixture)
    base = dict(XMin='0.1', YMin='0.2', XMax='0.6', YMax='0.8', IsGroupOf='0',
                IsDepiction='0', IsOccluded='0', IsTruncated='0')
    plan = {'metadata': {'abc': {'License': 'https://creativecommons.org/licenses/by/2.0/'}},
            'names': {'t': 'Table', 'd': 'Desk'},
            'annotations': {'abc': [dict(base, LabelName=mid) for mid in ('t', 'd')]}}
    row = campus.fetch_record('abc', tmp_path, plan)
    assert len(campus.read_labels(row['label'])) == 1
    assert len(row['attributes']) == 2
    assert 'source_hierarchy_cooccurrence_review_overlap' in row['flags']
