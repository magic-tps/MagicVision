"""Acquire official CVAT accessibility boxes by verified selective ZIP ranges.

The 8 GB public archive stays remote. Every imported image remains unreviewed
against the 25-class campus taxonomy; source height/width estimates are retained
only as publisher attributes, never metric ground truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import struct
import threading
import time
from xml.etree import ElementTree as ET
import zipfile
import zlib

import requests
from PIL import Image

from campus_schema import CLASSES, ROOT, from_xyxy, read_labels, record, save_json, sha, write_labels, write_manifest
from http_range_zip import RangeReader

CATALOG = 'https://zenodo.org/api/records/6382090'
SOURCE_PAGE = 'https://zenodo.org/records/6382090'
CLASS_MAP = {'step': 12, 'stair': 5, 'ramp': 11, 'grab_bar': 21}
OUTPUT = ROOT / 'datasets/campus_access_v4/wheelmap'
_local = threading.local()


def session():
    if not hasattr(_local, 'session'):
        _local.session = requests.Session()
        _local.session.headers['User-Agent'] = 'MagicVision-academic-dataset-import/1.0'
    return _local.session


def checked_range(url, start, stop, size):
    if start < 0 or stop < start or stop >= size or stop-start > 30_000_000:
        raise ValueError('Invalid or unbounded ZIP byte range')
    for attempt in range(5):
        try:
            with session().get(url, headers={'Range': f'bytes={start}-{stop}'},
                               timeout=(20, 90), stream=True) as response:
                response.raise_for_status()
                expected = f'bytes {start}-{stop}/{size}'
                if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                    raise ValueError('Publisher did not return the requested ZIP range')
                content = bytearray()
                for part in response.iter_content(131072):
                    content.extend(part)
                    if len(content) > stop-start+1:
                        raise ValueError('ZIP response exceeds requested range')
                if len(content) != stop-start+1:
                    raise ValueError('Truncated ZIP byte range')
                return bytes(content)
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(min(20, 2 ** attempt))


def decode_member(raw, member):
    """Validate a ZIP local header, decompressed size and CRC without extraction."""
    if len(raw) < 30:
        raise ValueError('Incomplete ZIP local header')
    fields = struct.unpack('<4s5H3I2H', raw[:30])
    if fields[0] != b'PK\x03\x04' or fields[2] & 1:
        raise ValueError('Invalid or encrypted ZIP member')
    if fields[3] != member['compress_type']:
        raise ValueError('ZIP compression metadata mismatch')
    offset = 30 + fields[-2] + fields[-1]
    payload = raw[offset:offset+member['compress_size']]
    if len(payload) != member['compress_size'] or member['file_size'] > 25_000_000:
        raise ValueError('Invalid ZIP payload size')
    if member['compress_type'] == zipfile.ZIP_STORED:
        result = payload
    elif member['compress_type'] == zipfile.ZIP_DEFLATED:
        decoder = zlib.decompressobj(-15)
        result = decoder.decompress(payload, member['file_size'] + 1)
        if not decoder.eof or decoder.unconsumed_tail:
            raise ValueError('Invalid or oversized compressed ZIP member')
    else:
        raise ValueError('Unsupported ZIP compression')
    if len(result) != member['file_size'] or zlib.crc32(result) & 0xffffffff != member['CRC']:
        raise ValueError('ZIP size or CRC mismatch')
    return result


def read_member(url, size, member):
    # Normal ZIP names/extras fit here; unusually large header extras are read
    # separately, still bounded and never interpreted as filesystem paths.
    start = member['header_offset']
    stop = min(size-1, start + 30 + 65535 + 65535 + member['compress_size'] - 1)
    return decode_member(checked_range(url, start, stop, size), member)


def parse_annotations(content):
    if len(content) > 10_000_000 or b'<!DOCTYPE' in content or b'<!ENTITY' in content:
        raise ValueError('Invalid CVAT XML document')
    xml = ET.fromstring(content)
    images, errors = [], []
    for element in xml.findall('image'):
        try:
            width, height = int(element.attrib['width']), int(element.attrib['height'])
            if min(width, height) <= 0:
                raise ValueError('Invalid CVAT dimensions')
            name = element.attrib['name']
            if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or ':' in name:
                raise ValueError('Unsafe image member name')
            boxes, originals = [], []
            for box in element.findall('box'):
                category = box.attrib['label']
                if category not in CLASS_MAP:
                    continue
                coords = [float(box.attrib[key]) for key in ('xtl', 'ytl', 'xbr', 'ybr')]
                boxes.append(from_xyxy(CLASS_MAP[category], coords[0]/width, coords[1]/height,
                                       coords[2]/width, coords[3]/height))
                originals.append({'label': category, 'xyxy_pixels': coords,
                    'attributes': {attr.attrib['name']: attr.text for attr in box.findall('attribute')},
                    'occluded': box.attrib.get('occluded'), 'source': box.attrib.get('source')})
            if boxes:
                images.append({'name': name, 'width': width, 'height': height, 'boxes': boxes,
                    'source_annotations': originals, 'task_id': element.attrib.get('task_id'),
                    'source_image_id': element.attrib.get('id'), 'source_subset': element.attrib.get('subset')})
        except (KeyError, ValueError) as error:
            errors.append({'name': element.attrib.get('name'), 'reason': str(error)})
    return images, errors


def select_images(images, quotas, seed=42):
    """Rarest requested class first, count overlapping classes toward quotas."""
    shuffled = sorted(images, key=lambda row: row['name'])
    random.Random(seed).shuffle(shuffled)
    counts, selected, names = Counter(), [], set()
    for cls in sorted(quotas, key=lambda cls: sum(any(b[0] == cls for b in r['boxes']) for r in shuffled)):
        for row in shuffled:
            if counts[cls] >= quotas[cls]:
                break
            labels = {b[0] for b in row['boxes']}
            if cls not in labels or row['name'] in names:
                continue
            names.add(row['name']); selected.append(row); counts.update(labels)
    return selected


def acquire(args):
    out = Path(args.output); metadata = out / 'metadata'; metadata.mkdir(parents=True, exist_ok=True)
    catalog_path = metadata / 'zenodo_record.json'
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    else:
        response = session().get(CATALOG, timeout=(20, 60)); response.raise_for_status()
        catalog = response.json(); save_json(catalog_path, catalog)
    if catalog['metadata']['license']['id'] != 'cc-by-4.0':
        raise ValueError('Source license changed; inspect before importing')
    file = next(f for f in catalog['files'] if f['key'] == 'wm_barriers_data.zip')
    url, size = file['links']['self'], file['size']
    index_path, xml_path = metadata / 'archive_index.json', metadata / 'wm_annotations.xml'
    if not index_path.exists() or not xml_path.exists():
        with RangeReader(url, size) as remote:
            with zipfile.ZipFile(remote) as archive:
                index = {info.filename: {key: getattr(info, key) for key in
                    ('filename', 'header_offset', 'file_size', 'compress_size', 'compress_type', 'CRC')}
                    for info in archive.infolist()}
                save_json(index_path, index)
                xml_path.write_bytes(archive.read('wm_annotations.xml'))
    index = json.loads(index_path.read_text(encoding='utf-8'))
    candidates, errors = parse_annotations(xml_path.read_bytes())
    annotation_hash = sha(xml_path)
    quotas = {11: args.ramps, 12: args.steps, 21: args.handrails}
    chosen = select_images(candidates, quotas)
    save_json(metadata / 'selection.json', {'seed':42, 'quotas': {CLASSES[k]:v for k,v in quotas.items()},
        'selected_images':len(chosen), 'names':[r['name'] for r in chosen]})
    print(f'Selected {len(chosen)} distinct image names; {len(errors)} malformed annotations quarantined.', flush=True)

    def work(item):
        stem = 'wheelmap_' + PurePosixPath(item['name']).stem
        image, label = out / 'images' / (stem + '.jpg'), out / 'labels' / (stem + '.txt')
        image.parent.mkdir(parents=True, exist_ok=True)
        member = index[item['name']]
        valid_cached = image.exists() and image.stat().st_size == member['file_size']
        if valid_cached:
            valid_cached = zlib.crc32(image.read_bytes()) & 0xffffffff == member['CRC']
        if not valid_cached:
            image_bytes = read_member(url, size, member)
            with Image.open(io.BytesIO(image_bytes)) as im:
                im.load()
                if im.size != (item['width'], item['height']):
                    raise ValueError('CVAT and downloaded image dimensions disagree')
            part = image.with_suffix('.jpg.part'); part.write_bytes(image_bytes); part.replace(image)
        write_labels(label, item['boxes'])
        return record(image, label, 'wheelmap_accessibility_barriers_zenodo6382090',
            group=f'wheelmap_annotation_task_{item["task_id"]}',
            group_basis='publisher_annotation_task_conservative_group_not_verified_capture_session',
            source_url=SOURCE_PAGE, download_url=url, source_member=item['name'],
            source_image_id=item['source_image_id'], source_annotation_task_id=item['task_id'],
            source_subset=item['source_subset'], license='CC-BY-4.0',
            attribution='Jakob Stolberg et al., Image Dataset of Accessibility Barriers, HIIG, 2022, doi:10.5281/zenodo.6382090',
            annotation_type='publisher_manual_bounding_boxes_remapped_to_campus_v4',
            source_annotations=item['source_annotations'], source_annotations_sha256=annotation_hash,
            source_archive_checksum=file['checksum'], image_crc32=f'{member["CRC"]:08x}',
            annotation_caveat='Step height and ramp width attributes are visual estimates, not measured distances. Only visible increments and sloped surfaces are boxed.',
            class_mapping=CLASS_MAP, source_session_known=False, session_known=False)

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, item): item for item in chosen}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as error:
                errors.append({'name':futures[future]['name'], 'reason':str(error)})
            if len(rows) and len(rows) % 25 == 0:
                write_manifest(out / 'manifest.jsonl', sorted(rows, key=lambda r:r['id']))
                print(f'Imported {len(rows)}/{len(chosen)}', flush=True)
    rows.sort(key=lambda r:r['id']); write_manifest(out / 'manifest.jsonl', rows)
    counts, boxes = Counter(), Counter()
    for row in rows:
        annotations = read_labels(row['label'])
        counts.update(set(CLASSES[int(b[0])] for b in annotations))
        boxes.update(CLASSES[int(b[0])] for b in annotations)
    report = {'source_url':SOURCE_PAGE, 'license':'CC-BY-4.0', 'source_class_mapping':CLASS_MAP,
        'images':len(rows), 'unique_image_sha256':len({r['sha256'] for r in rows}),
        'images_per_class':dict(counts), 'boxes_per_class':dict(boxes),
        'selected':len(chosen), 'requested_image_quotas':{CLASSES[k]:v for k,v in quotas.items()},
        'source_positive_images':len(candidates), 'errors':errors, 'coverage_complete':False,
        'approved_images':0, 'source_split':'default_unsplit_preserved_as_metadata',
        'download_method':'verified_HTTP_ranges_local_ZIP_header_uncompressed_size_CRC32',
        'scope':'only visible surfaces; not a complete 25-class campus annotation; no training performed'}
    save_json(out / 'download_report.json', report)
    save_json(ROOT / 'reports/campus_access_wheelmap.json', report)
    print(json.dumps({k:v for k,v in report.items() if k!='errors'}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--ramps', type=int, default=150)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--handrails', type=int, default=400)
    parser.add_argument('--workers', type=int, default=4)
    acquire(parser.parse_args())
