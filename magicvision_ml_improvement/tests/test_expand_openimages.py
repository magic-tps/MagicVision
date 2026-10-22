import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import expand_openimages as expansion


def test_candidate_selection_excludes_existing_groups_and_depictions(tmp_path):
    source = tmp_path / 'boxes.csv'
    with source.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['ImageID', 'LabelName', 'IsGroupOf', 'IsDepiction'])
        writer.writeheader()
        for image_id, group, depiction in [('old', '0', '0'), ('real', '0', '0'),
                                             ('crowd', '1', '0'), ('drawing', '0', '1'),
                                             ('real', '0', '0')]:
            writer.writerow(dict(ImageID=image_id, LabelName='person_mid', IsGroupOf=group, IsDepiction=depiction))
    selected, available = expansion.select_candidates(source, {'person_mid': 'Person'}, {'person': 10}, {'old'})
    assert selected == {'person': ['real']}
    assert available == {'person': 1}


def test_existing_hash_exclusion_covers_other_sources(tmp_path):
    source = tmp_path / 'manifest.jsonl'
    source.write_text('\n'.join(json.dumps(row) for row in [
        {'source': 'openimages', 'source_image_id': 'abc', 'sha256': 'hash1'},
        {'source': 'legacy', 'source_image_id': 'def', 'sha256': 'hash2'}]))
    ids, hashes = expansion.existing_records([source])
    assert ids == {'abc'} and hashes == {'hash1', 'hash2'}


def test_unlicensed_image_rejected_before_network(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(expansion, 'download', lambda *args: called.append(args))
    with pytest.raises(ValueError, match='license'):
        expansion.fetch_record('id', tmp_path, {'metadata': {'id': {'License': ''}}})
    assert called == []


def test_exact_taxonomy_has_no_false_pole_mapping():
    assert 'Street light' not in expansion.SOURCE_MAP
    assert expansion.SOURCE_MAP['Bench'] == 'obstacle'
    assert expansion.SOURCE_MAP['Fire hydrant'] == 'obstacle'
    assert expansion.SOURCE_MAP['Waste container'] == 'trash_bin'


def test_import_preserves_source_licence_and_pending_review(tmp_path, monkeypatch):
    from PIL import Image
    def download_fixture(url, destination, limit):
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (80, 60), 'red').save(destination)
    monkeypatch.setattr(expansion, 'download', download_fixture)
    plan = {'metadata': {'abc': {'License': 'https://creativecommons.org/licenses/by/2.0/',
                                 'Author': 'Test author', 'OriginalLandingURL': 'https://example.org/photo'}},
            'names': {'bin': 'Waste container'}, 'annotations': {'abc': [
                {'LabelName': 'bin', 'XMin': '0.1', 'YMin': '0.2', 'XMax': '0.6', 'YMax': '0.8',
                 'IsGroupOf': '0', 'IsDepiction': '0', 'IsOccluded': '1', 'IsTruncated': '0'}]}}
    row = expansion.fetch_record('abc', tmp_path, plan)
    assert row['review_status'] == 'needs_review' and row['coverage_complete'] is False
    assert row['author'] == 'Test author' and row['annotation_license'] == 'CC-BY-4.0'
    assert row['attributes'][0]['source_class'] == 'Waste container'
    assert row['attributes'][0]['occluded'] is True
    assert Path(row['label']).read_text().startswith('10 ')
