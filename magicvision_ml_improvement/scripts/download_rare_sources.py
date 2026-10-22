"""Download and quarantine official RDD2022 D40 and HOT-OSM pole annotations.

No row is training-approved. Importing boxes is not an eleven-class audit.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import time
import struct
import zlib
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET

import requests
from PIL import Image

from data_pipeline import CLASSES, ROOT, from_xyxy, record, save_json, sha, write_labels, write_manifest

HF_REPO = 'hotosm/streetlevel-poles'
HF_REV = '0352f0b2b067439ca3467994cb9938a42310a09e'
RDD_BASE = 'https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022'
RDD_CATALOG = 'https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547'


def fetch(url, path, expected_size=None, expected_hash=None):
    """Resume public files, verify known content hashes, and publish atomically."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (expected_size is None or path.stat().st_size == expected_size):
        if expected_hash is None or sha(path) == expected_hash:
            return path
    part = path.with_suffix(path.suffix + '.part')
    for attempt in range(4):
        try:
            start = part.stat().st_size if part.exists() else 0
            headers = {'Range': f'bytes={start}-'} if start else {}
            with requests.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    if not response.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
                        raise ValueError('Invalid resume range')
                else:
                    start = 0
                total = response.headers.get('Content-Length')
                if total and int(total) + start > 2_000_000_000:
                    raise ValueError('File exceeds per-file download limit')
                with part.open('ab' if start else 'wb') as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
                        if handle.tell() > 2_000_000_000:
                            raise ValueError('File exceeds download limit')
            if expected_size is not None and part.stat().st_size != expected_size:
                raise ValueError('Downloaded size differs from catalog')
            if expected_hash is not None and sha(part) != expected_hash:
                raise ValueError('Downloaded SHA256 differs from catalog')
            part.replace(path)
            print(f'Downloaded {path.name}: {path.stat().st_size:,} bytes', flush=True)
            return path
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)
    raise RuntimeError('Download failed')


def safe_member(name):
    """Archive names may never become arbitrary filesystem paths."""
    value = PurePosixPath(name.replace('\\', '/'))
    if value.is_absolute() or '..' in value.parts or ':' in name:
        raise ValueError('Unsafe archive path')
    return value


def convert_rdd_xml(xml_bytes):
    if len(xml_bytes) > 1_000_000 or b'<!DOCTYPE' in xml_bytes or b'<!ENTITY' in xml_bytes:
        raise ValueError('Unexpected XML declaration or size')
    root = ET.fromstring(xml_bytes)
    width = int(root.findtext('size/width')); height = int(root.findtext('size/height'))
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image dimensions')
    boxes = []
    for obj in root.findall('object'):
        if obj.findtext('name', '').strip() != 'D40':
            continue
        b = obj.find('bndbox')
        coords = [float(b.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
        # Preserve the publisher's box coordinates without silently clipping errors.
        boxes.append(from_xyxy(6, coords[0]/width, coords[1]/height, coords[2]/width, coords[3]/height))
    return width, height, boxes


def convert_pole_boxes(objects, width, height):
    if len(objects['bbox']) != len(objects['category']):
        raise ValueError('Unequal category and box counts')
    boxes = []
    for category, coords in zip(objects['category'], objects['bbox']):
        if int(category) == 1:  # transmission towers are not a pole
            continue
        if int(category) != 0:
            raise ValueError('Unknown HOT-OSM class')
        x, y, w, h = map(float, coords)
        boxes.append(from_xyxy(7, x/width, y/height, (x+w)/width, (y+h)/height))
    return boxes


def import_rdd(archive, country, out, download_url=None):
    rows = []; errors = []; omitted = 0
    with zipfile.ZipFile(archive) as zf:
        members = {str(safe_member(x.filename)): x for x in zf.infolist()}
        for name, info in sorted(members.items()):
            if not name.endswith('.xml') or '/train/annotations/xmls/' not in name:
                continue
            try:
                if info.file_size > 1_000_000:
                    raise ValueError('Oversized annotation')
                xml = zf.read(info); width, height, boxes = convert_rdd_xml(xml)
                if not boxes:
                    omitted += 1; continue
                image_member = name.replace('/annotations/xmls/', '/images/').removesuffix('.xml') + '.jpg'
                image_info = members[image_member]
                if image_info.file_size > 32_000_000:
                    raise ValueError('Oversized image')
                stem = Path(name).stem
                image = out/'images'/country/(stem+'.jpg')
                label = out/'labels'/country/(stem+'.txt')
                original = out/'source_annotations'/country/(stem+'.xml')
                image.parent.mkdir(parents=True, exist_ok=True); original.parent.mkdir(parents=True, exist_ok=True)
                raw = zf.read(image_info)
                with Image.open(io.BytesIO(raw)) as im:
                    im.load()
                    if im.size != (width, height):
                        raise ValueError('XML and image dimensions differ')
                image.write_bytes(raw); original.write_bytes(xml); write_labels(label, boxes)
                match = re.search(r'(\d+)$', stem)
                # Consecutive IDs can share a drive. This grouping is conservative and
                # explicitly inferred; it is not a claim that route metadata exists.
                group = f'rdd2022:{country}:index_block_{int(match.group(1))//100}' if match else f'rdd2022:{country}:{stem}'
                rows.append(record(image, label, 'rdd2022', group=group, session_known=False,
                    group_evidence='inferred country plus 100 consecutive filename IDs; not verified sessions',
                    source_image_id=stem, source_split='train', original_split='train', country=country,
                    source_url=RDD_CATALOG, download_url=download_url or f'{RDD_BASE}/RDD2022_{country}.zip',
                    source_archive_member=image_member, source_annotation=str(original.resolve()),
                    source_annotation_sha256=sha(original), source_archive_sha256=sha_cached(archive),
                    license='https://creativecommons.org/licenses/by/4.0/',
                    author='Deeksha Arya, Hiroya Maeda, Yoshihide Sekimoto et al. (RDD2022)',
                    mapping={'D40':'pothole'}, provenance_reviewed=False,
                    review_reason='D40 source boxes only. Review all eleven classes, road-to-campus domain and source provenance. Other cracks excluded.'))
            except (ValueError, KeyError, OSError, ET.ParseError) as error:
                errors.append({'member': name, 'error': str(error)})
    write_manifest(out/f'manifest_{country}.jsonl', rows)
    return rows, {'country':country, 'imported_images':len(rows), 'pothole_boxes':sum(len(convert_rdd_xml(Path(r['source_annotation']).read_bytes())[2]) for r in rows), 'omitted_without_D40':omitted, 'errors':errors}


_HASHES = {}
def sha_cached(path):
    key = str(Path(path).resolve())
    if key not in _HASHES:
        _HASHES[key] = sha(path)
    return _HASHES[key]


def fetch_figshare_country(metadata,country,path):
    """The public Figshare ZIP stores country ZIPs without compression: range only one."""
    from http_range_zip import RangeReader
    bundle=next(f for f in metadata['files'] if f['name'].endswith('.zip'))
    reader=RangeReader(bundle['download_url'],bundle['size'])
    with zipfile.ZipFile(reader) as archive:
        info=archive.getinfo(f'RDD2022/{country}.zip')
        if info.compress_type!=zipfile.ZIP_STORED or info.file_size>2_000_000_000:
            raise ValueError('Unexpected country archive layout or size')
        reader.seek(info.header_offset);header=reader.read(30)
        values=struct.unpack('<IHHHHHIIIHH',header)
        if values[0]!=0x04034b50:raise ValueError('Invalid local ZIP header')
        first=info.header_offset+30+values[-2]+values[-1];last=first+info.file_size-1
    reader.close();path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def crc_file(file):
        crc=0
        with file.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):crc=zlib.crc32(block,crc)
        return crc&0xffffffff
    if path.exists() and path.stat().st_size==info.file_size and crc_file(path)==info.CRC:return path
    partial=path.with_suffix('.zip.part')
    for attempt in range(4):
        start=partial.stat().st_size if partial.exists() else 0
        if start==info.file_size:break
        if start>info.file_size:raise ValueError('Country partial file larger than expected')
        try:
            with requests.get(bundle['download_url'],headers={'Range':f'bytes={first+start}-{last}'},stream=True,timeout=(20,90)) as response:
                if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {first+start}-{last}/{bundle["size"]}':
                    raise ValueError('Publisher did not return the requested country byte range')
                with partial.open('ab') as stream:
                    for block in response.iter_content(1024*1024):
                        stream.write(block)
                        if stream.tell()>info.file_size:raise ValueError('Country download exceeds member size')
            break
        except requests.RequestException:
            if attempt==3:raise
            time.sleep(attempt+1)
    if partial.stat().st_size!=info.file_size or crc_file(partial)!=info.CRC:raise ValueError('Country ZIP size/CRC differs from official bundle')
    partial.replace(path)
    save_json(path.with_suffix('.provenance.json'),{'url':bundle['download_url'],'bundle_size':bundle['size'],
              'member':info.filename,'byte_range':[first,last],'crc32':info.CRC,'sha256':sha(path)})
    print(f'Figshare country range complete: {country}, {info.file_size:,} bytes (CRC verified)',flush=True)
    return path


def rdd(out, countries):
    meta_path = fetch('https://api.figshare.com/v2/articles/21431547', out/'metadata/figshare_article.json')
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    if metadata['license']['name'] != 'CC BY 4.0':
        raise ValueError('Dataset license changed; review first')
    archives = []
    def country_archive(country):
        path=out/'archives'/f'RDD2022_{country}.zip'
        provenance=path.with_suffix('.provenance.json')
        if path.exists() and provenance.exists():
            saved=json.loads(provenance.read_text(encoding='utf-8'))
            if sha(path)!=saved['sha256']:
                raise ValueError('Cached country ZIP differs from its verified provenance')
            return path,saved['url']+'#member='+saved['member']
        try:return fetch(f'{RDD_BASE}/RDD2022_{country}.zip',path),f'{RDD_BASE}/RDD2022_{country}.zip'
        except requests.HTTPError as error:
            if error.response.status_code not in (403,404):raise
            print(f'Country URL unavailable ({error.response.status_code}); using publisher Figshare for {country}',flush=True)
            bundle=next(f for f in metadata['files'] if f['name'].endswith('.zip'))
            return fetch_figshare_country(metadata,country,path),bundle['download_url']+'#member=RDD2022/'+country+'.zip'
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [(country, pool.submit(country_archive,country)) for country in countries]
        for country, future in futures:
            archive,url=future.result();archives.append((country,archive,url))
    rows = []; reports = []
    for country, archive,url in archives:
        new_rows, report = import_rdd(archive, country, out,url); rows += new_rows; reports.append(report)
        print(json.dumps(report), flush=True)
    write_manifest(out/'manifest.jsonl', rows)
    save_json(out/'download_report.json', {'images':len(rows), 'reports':reports, 'review_status':'needs_review', 'no_unannotated_test_images_imported':True})


def poles(out):
    import pyarrow.parquet as pq
    base = f'https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REV}'
    fetch(f'{base}/README.md', out/'metadata/README_source.md')
    tree_path = fetch(f'https://huggingface.co/api/datasets/{HF_REPO}/tree/{HF_REV}?recursive=true', out/'metadata/tree.json')
    entries = json.loads(tree_path.read_text(encoding='utf-8'))
    shards = [e for e in entries if e['path'].endswith('.parquet') and e['path'].startswith('data/')]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [(e, pool.submit(fetch, f"{base}/{e['path']}", out/'archives'/e['path'], e['size'], e['lfs']['oid'])) for e in shards]
        for entry, future in futures:
            future.result()
    rows = []; errors = []; omitted = 0; source_counts = Counter(); box_count = 0
    for entry in shards:
        shard = out/'archives'/entry['path']; split = PurePosixPath(entry['path']).parts[1]
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=32):
            for source in batch.to_pylist():
                try:
                    boxes = convert_pole_boxes(source['objects'], source['width'], source['height'])
                    if not boxes:
                        omitted += 1; continue
                    stem = re.sub(r'[^A-Za-z0-9_.-]', '_', source['image_id'])
                    image = out/'images'/split/(stem+'.jpg'); label = out/'labels'/split/(stem+'.txt')
                    image.parent.mkdir(parents=True, exist_ok=True)
                    raw = source['image']['bytes']
                    if raw is None or len(raw) > 32_000_000:
                        raise ValueError('Missing or oversized embedded image')
                    with Image.open(io.BytesIO(raw)) as im:
                        im.load()
                        if im.size != (source['width'], source['height']):
                            raise ValueError('Parquet and image dimensions differ')
                    image.write_bytes(raw); write_labels(label, boxes)
                    # .rf. derivatives must stay grouped. Sequence identity remains unknown.
                    group = f"hotosm:{source['image_id'].split('.rf.')[0]}"
                    rows.append(record(image, label, 'hotosm_poles', group=group, session_known=False,
                        source_image_id=source['image_id'], source_split=split, original_split=split,
                        source_url=f'https://huggingface.co/datasets/{HF_REPO}', source_revision=HF_REV,
                        source_collection=source.get('source'), original_url=source.get('source_url'),
                        panoramax_id=source.get('panoramax_id'), source_archive=entry['path'], source_archive_sha256=entry['lfs']['oid'],
                        license='https://creativecommons.org/licenses/by-sa/4.0/',
                        author='HOT-OSM / UC Berkeley; individual positive-image creators not supplied in source card',
                        provenance_reviewed=False, mapping={'pole':'pole'},
                        review_reason='Utility pole source boxes only; towers excluded. Review visible shaft boundaries versus attachments, all eleven classes and per-image provenance. No precise coordinates copied into manifest.'))
                    source_counts[split] += 1; box_count += len(boxes)
                except (KeyError, ValueError, OSError) as error:
                    errors.append({'image_id':source.get('image_id'), 'error':str(error)})
    write_manifest(out/'manifest.jsonl', rows)
    report = {'images':len(rows), 'pole_boxes':box_count, 'by_source_split':dict(source_counts), 'omitted_without_pole':omitted, 'errors':errors,
              'license':'CC-BY-SA-4.0', 'revision':HF_REV, 'review_status':'needs_review'}
    save_json(out/'download_report.json', report); print(json.dumps(report), flush=True)


def self_test():
    import unittest
    class ConversionTests(unittest.TestCase):
        def test_rdd_only_maps_potholes(self):
            xml = b'<annotation><size><width>100</width><height>50</height></size><object><name>D40</name><bndbox><xmin>10</xmin><ymin>5</ymin><xmax>30</xmax><ymax>25</ymax></bndbox></object><object><name>D20</name></object></annotation>'
            w, h, boxes = convert_rdd_xml(xml)
            self.assertEqual((w,h),(100,50)); self.assertEqual(boxes, [[6,.2,.3,.19999999999999998,.4]])
        def test_poles_exclude_towers(self):
            result = convert_pole_boxes({'bbox':[[10,5,20,40],[0,0,100,100]],'category':[0,1]},100,100)
            self.assertEqual(len(result),1); self.assertEqual(result[0][0],7)
        def test_invalid_box_rejected(self):
            with self.assertRaises(ValueError):
                convert_pole_boxes({'bbox':[[90,5,20,40]],'category':[0]},100,100)
        def test_archive_paths(self):
            for value in ('../evil.xml', '/root/evil', 'C:/evil', 'a/../../evil'):
                with self.assertRaises(ValueError): safe_member(value)
            self.assertEqual(str(safe_member('India/train/images/a.jpg')),'India/train/images/a.jpg')
        def test_xml_entity_rejected(self):
            with self.assertRaises(ValueError): convert_rdd_xml(b'<!DOCTYPE x><annotation/>')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ConversionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=['rdd','poles','self-test'])
    parser.add_argument('--output', type=Path, default=ROOT/'datasets/rare_sources_v3')
    parser.add_argument('--countries', nargs='+', choices=['India','Japan','United_States'], default=['India','Japan','United_States'])
    args = parser.parse_args()
    if args.source == 'self-test': self_test()
    elif args.source == 'rdd': rdd(args.output/'rdd2022', args.countries)
    else: poles(args.output/'hotosm_poles')


if __name__ == '__main__': main()
