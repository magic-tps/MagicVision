from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import campus22
import campus22_schema as schema
from prepare_reviewed_campus22 import FIRE_SCENES, plan


def test_scene_split_keeps_repeats_and_entire_videos_together(monkeypatch):
    rows = []
    boxes = {}

    def add(name, source, subset, classes, video=None):
        digest = hashlib.sha256(name.encode()).hexdigest()
        rows.append({'id': name, 'source': source, 'group': name, 'group_id': name,
                     'resolved_split': subset, 'sha256': digest, 'phash': digest[:16],
                     'label': name, 'source_video_info': {'vid_id': video} if video else None})
        boxes[name] = [[c, .5, .5, .1, .1] for c in classes]

    for _, ids, _ in FIRE_SCENES.values():
        for n in ids:
            add(f'firenet_{n:04d}', 'firenet', 'train', [21])
            rows[-1]['group'] = rows[-1]['group_id'] = 'firenet:unknown_session'
    for subset, count in [('train', 60), ('val', 12), ('test', 10)]:
        classes = list(range(21)) if subset != 'test' else [c for c in range(21) if c not in (18, 20)]
        for i in range(count):
            add(f'{subset}_{i}', 'fixture', subset, classes)
    add('video_seq1', 'roadwork_v5', 'train', [18, 20], 'same_video')
    add('video_seq2', 'roadwork_v5', 'val', [18, 20], 'same_video')
    original = deepcopy(rows)
    monkeypatch.setattr(schema, 'read_labels', lambda path: boxes[path])
    result, report = plan(rows)
    assert rows == original
    byid = {r['id']: r for r in result}
    for _, ids, _ in FIRE_SCENES.values():
        assert len({byid[f'firenet_{n:04d}']['group'] for n in ids}) == 1
        assert len({byid[f'firenet_{n:04d}']['resolved_split'] for n in ids}) == 1
    assert byid['video_seq1']['group'] == byid['video_seq2']['group']
    assert byid['video_seq1']['resolved_split'] != 'train'
    for subset in ('train', 'val', 'test'):
        assert all(report['images_per_class_by_split'][subset].get(name, 0) > 0 for name in schema.CLASSES)
    assert report['whole_group_holdouts']
    for row in result:
        if row['resolved_split_before_scene_review'] in ('val', 'test'):
            assert row['resolved_split'] != 'train'
    assert all(byid[f'firenet_{n:04d}']['session_known'] is False
               for _, ids, _ in FIRE_SCENES.values() for n in ids)


def test_unknown_fire_images_require_new_visual_evidence():
    with pytest.raises(ValueError, match='membership changed'):
        plan([{'id': 'firenet_unseen', 'source': 'firenet'}])


def test_pilot_does_not_accept_missing_or_changed_preparation_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(campus22, 'inspect_release', lambda *a, **kw: {'ready': True, 'errors': []})
    data = tmp_path / 'data.yaml'
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('original\n')
    assert not campus22.check(data, pilot=True)['ready']
    report = {'released_manifest_sha256': schema.sha(manifest), 'pilot_minimum_priority_images': 40,
              'limitations': ['Scene inference; unknown capture sessions.']}
    (tmp_path / 'preparation_report.json').write_text(json.dumps(report))
    assert campus22.check(data, pilot=True)['ready']
    manifest.write_text('changed\n')
    result = campus22.check(data, pilot=True)
    assert not result['ready']
    assert 'manifest changed' in result['errors'][0]
