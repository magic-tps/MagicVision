import json
from pathlib import Path
import sys
from contextlib import contextmanager
from threading import Thread
from urllib.request import Request, urlopen

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from data_pipeline import record, read_manifest, sha, write_labels, write_manifest
from review_dataset import apply, create_server, decide, export, render


def setup_package(tmp_path, boxes=None, count=1):
    source = tmp_path / 'source'; source.mkdir()
    rows = []
    for index in range(count):
        image = source / f'{index}.png'; Image.new('RGB', (80, 90), (index, 40, 50)).save(image)
        label = source / f'{index}.txt'
        write_labels(label, [[0, .5, .5, .2, .2]] if boxes is None else boxes)
        rows.append(record(image, label, 'fixture', group='session_A', original_split='val'))
    manifest = source / 'manifest.jsonl'; write_manifest(manifest, rows)
    package = tmp_path / 'package'; info = export([manifest], package, limit=count)
    return rows, package, info


def approve(package, row, **extra):
    return decide(package, row['id'], 'approve', 'Human reviewer', 'Reviewed every class and source',
                  complete=True, provenance_reviewed=True, **extra)


def test_export_never_approves_and_has_editor(tmp_path):
    rows, package, info = setup_package(tmp_path)
    assert info['rows_exported'] == 1
    assert read_manifest(package / 'decisions.jsonl') == []
    assert read_manifest(package / 'source_manifest.jsonl')[0]['review_status'] == 'needs_review'
    page = (package / 'index.html').read_text(encoding='utf-8')
    assert '<canvas' in page and 'showSaveFilePicker' in page
    assert 'data-action="approve"' in page and 'data-action="negative"' in page
    assert '/api/decision' in page and '/api/labels' in page
    assert 'https://' not in page  # all editing resources stay local
    assert sha(package / info['items'][0]['image']) == rows[0]['sha256']


def test_corrected_labels_preserve_source_group_holdout_and_pending_rows(tmp_path):
    rows, package, info = setup_package(tmp_path, count=2)
    item = next(i for i in info['items'] if i['id'] == rows[0]['id'])
    original_label = Path(rows[0]['label']).read_bytes()
    write_labels(package / item['label'], [[10, .5, .6, .2, .3]])
    render(package)
    approval = approve(package, rows[0])
    out = tmp_path / 'reviewed'; report = apply(package, out)
    result = {r['id']: r for r in read_manifest(out / 'manifest.jsonl')}
    reviewed = result[rows[0]['id']]
    assert report['decisions'] == {'approve': 1} and not report['training_ready']
    assert reviewed['group'] == 'session_A' and reviewed['original_split'] == 'val'
    assert reviewed['review_status'] == 'approved' and reviewed['coverage_complete']
    assert Path(reviewed['label']).read_text().startswith('10 ')
    assert sha(reviewed['label']) == approval['reviewed_label_sha256']
    assert Path(rows[0]['label']).read_bytes() == original_label
    assert result[rows[1]['id']]['review_status'] == 'needs_review'


@pytest.mark.parametrize('change', ['packaged_label', 'source_label', 'packaged_image', 'source_image', 'manifest'])
def test_changed_files_invalidate_decision_before_any_output(tmp_path, change):
    rows, package, info = setup_package(tmp_path)
    approve(package, rows[0]); item = info['items'][0]
    paths = {'packaged_label': package/item['label'], 'source_label': Path(rows[0]['label']),
             'packaged_image': package/item['image'], 'source_image': Path(rows[0]['image']),
             'manifest': package/'source_manifest.jsonl'}
    with paths[change].open('ab') as handle:
        handle.write(b'\n')
    out = tmp_path / 'reviewed'
    with pytest.raises(ValueError, match='changed'):
        apply(package, out)
    assert not out.exists()


def test_empty_annotation_is_not_an_approved_negative(tmp_path):
    rows, package, _ = setup_package(tmp_path, boxes=[])
    with pytest.raises(ValueError, match='verified-negative'):
        approve(package, rows[0])
    approve(package, rows[0], verified_negative=True)
    apply(package, tmp_path/'reviewed')
    row = read_manifest(tmp_path/'reviewed/manifest.jsonl')[0]
    assert row['verified_negative'] and row['coverage_complete']


def test_explicit_completeness_and_provenance_required(tmp_path):
    rows, package, _ = setup_package(tmp_path)
    with pytest.raises(ValueError, match='coverage and provenance'):
        decide(package, rows[0]['id'], 'approve', 'reviewer', 'note')
    with pytest.raises(ValueError, match='identity'):
        decide(package, rows[0]['id'], 'approve', '', 'note', complete=True, provenance_reviewed=True)
    with pytest.raises(ValueError, match='Positive boxes'):
        approve(package, rows[0], verified_negative=True)


def test_reject_is_not_eligible_and_review_history_is_preserved(tmp_path):
    rows, package, _ = setup_package(tmp_path)
    approve(package, rows[0])
    decide(package, rows[0]['id'], 'reject', 'reviewer', 'Incorrect source label')
    assert len(read_manifest(package/'decisions_audit.jsonl')) == 2
    apply(package, tmp_path/'reviewed')
    row = read_manifest(tmp_path/'reviewed/manifest.jsonl')[0]
    assert row['review_status'] == 'rejected' and not row['coverage_complete']


def test_edited_decision_cannot_bypass_hash_or_completion_checks(tmp_path):
    rows, package, _ = setup_package(tmp_path)
    approval = approve(package, rows[0]); approval['coverage_complete'] = False
    write_manifest(package/'decisions.jsonl', [approval])
    with pytest.raises(ValueError, match='complete coverage'):
        apply(package, tmp_path/'reviewed')


def test_reviewed_manifest_snapshot_does_not_accept_duplicate_ids(tmp_path):
    rows, package, _ = setup_package(tmp_path)
    manifest = package/'source_manifest.jsonl'
    with pytest.raises(ValueError, match='Duplicate record IDs'):
        export([manifest, manifest], tmp_path/'other')


def test_package_reexport_never_overwrites_existing_review(tmp_path):
    _, package, _ = setup_package(tmp_path)
    with pytest.raises(ValueError, match='new empty'):
        export([package/'source_manifest.jsonl'], package)


def test_balanced_selection_represents_all_classes_without_duplicate_images(tmp_path):
    rows=[]
    for c in range(11):
        for i in range(2):
            image=tmp_path/f'{c}_{i}.png';Image.new('RGB',(40,50),(c*20,i*60,90)).save(image)
            label=image.with_suffix('.txt');write_labels(label,[[c,.5,.5,.2,.2]])
            rows.append(record(image,label,'fixture',group=f'{c}_{i}'))
    manifest=tmp_path/'manifest.jsonl';write_manifest(manifest,rows)
    info=export([manifest],tmp_path/'balanced',limit=11,selection='balanced')
    chosen={item['id'] for item in info['items']}
    from data_pipeline import read_labels
    classes={read_labels(r['label'])[0][0] for r in rows if r['id'] in chosen}
    assert classes==set(range(11))
    assert len({item['source_image_sha256'] for item in info['items']})==11
    assert info['rows_total']==22


def test_manifest_order_preserves_priority_queue_order(tmp_path):
    rows=[]
    for name,cls,color in [('first',0,'red'),('second',10,'blue')]:
        image=tmp_path/f'{name}.png';Image.new('RGB',(40,50),color).save(image)
        label=image.with_suffix('.txt');write_labels(label,[[cls,.5,.5,.2,.2]])
        rows.append(record(image,label,'fixture',group=name))
    manifest=tmp_path/'ordered.jsonl';write_manifest(manifest,rows)
    info=export([manifest],tmp_path/'ordered',limit=2,selection='manifest-order')
    assert [item['id'] for item in info['items']]==[row['id'] for row in rows]


@contextmanager
def running_editor(package):
    server = create_server(package, 0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_http_reopen_and_restart_restore_saved_boxes_and_review(tmp_path):
    rows, package, info = setup_package(tmp_path)
    original_page = (package / 'index.html').read_bytes()
    original_label = Path(rows[0]['label']).read_bytes()
    boxes = [[10, .5, .6, .2, .3]]

    def post(base, endpoint, **values):
        body = dict(package_id=info['package_id'], id=rows[0]['id'], **values)
        request = Request(base + endpoint, json.dumps(body).encode(),
                          {'Content-Type': 'application/json'})
        with urlopen(request) as response:
            return json.load(response)

    def reopen(base):
        # Even a cached HTML timestamp must not hide changes stored on disk.
        request = Request(base + '/index.html', headers={'If-Modified-Since': 'Wed, 01 Jan 2031 00:00:00 GMT'})
        with urlopen(request) as response:
            assert response.status == 200
            assert response.headers['Cache-Control'] == 'no-store'
            page = response.read().decode()
        assert 'data-boxes="[[10, 0.5, 0.6, 0.2, 0.3]]"' in page
        assert 'Decisión registrada: approve' in page
        assert 'estado: approved' in page
        assert 'value="Human reviewer"' in page
        assert 'value="Checked &quot;all&quot; &amp; source"' in page

    with running_editor(package) as base:
        assert post(base, '/api/labels', boxes=boxes)['saved']
        post(base, '/api/decision', action='approve', reviewer='Human reviewer', note='Checked "all" & source')
        reopen(base)
    with running_editor(package) as base:
        reopen(base)
        post(base, '/api/labels', boxes=[[0, .5, .5, .2, .2]])
        with urlopen(base + '/') as response:
            page = response.read().decode()
        assert 'estado: approved' not in page
        assert 'cambiaron desde la revisión' in page
    assert (package / 'index.html').read_bytes() == original_page
    assert Path(rows[0]['label']).read_bytes() == original_label
    assert len(read_manifest(package / 'decisions_audit.jsonl')) == 1
