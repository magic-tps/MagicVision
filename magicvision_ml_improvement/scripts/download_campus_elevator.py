"""Download individually licensed Commons elevator photos as unannotated candidates.

Category membership is a retrieval hint, not a detection annotation. The output
contains no invented boxes and cannot be used as verified negative training data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import json
from pathlib import Path
import random
import re
import threading
import time
from urllib.parse import urlparse

import requests
from PIL import Image

from campus_schema import ROOT, record, save_json, write_labels, write_manifest, sha

API = 'https://commons.wikimedia.org/w/api.php'
CATEGORIES = ['Category:Elevator doors', 'Category:Closed elevator doors', 'Category:Open elevator doors']
OUTPUT = ROOT / 'datasets/campus_elevator_v4'
_local = threading.local()


def session():
    if not hasattr(_local, 'session'):
        _local.session = requests.Session()
        _local.session.headers['User-Agent'] = 'MagicVisionUniversityDataset/1.0 (academic image collection; Commons API)'
    return _local.session


def plain(value):
    return html.unescape(re.sub('<[^>]+>', '', value)).strip()


def license_details(info):
    """Accept explicit CC BY/SA, CC0 or public-domain declarations only."""
    ext = info.get('extmetadata', {})
    values = {key: item.get('value', '') for key, item in ext.items()}
    short = plain(values.get('LicenseShortName', ''))
    license_url = values.get('LicenseUrl', '')
    parsed = urlparse(license_url)
    normalized = re.sub(r'\s+', ' ', short).upper()
    if 'NC' in normalized or 'ND' in normalized:
        raise ValueError('Restricted or ambiguous Commons license')
    if normalized.startswith(('CC BY ', 'CC BY-SA ')):
        if parsed.hostname not in {'creativecommons.org', 'www.creativecommons.org'}:
            raise ValueError('Missing canonical Creative Commons license URL')
        kind = 'by-sa' if normalized.startswith('CC BY-SA ') else 'by'
        if not re.match(rf'^/licenses/{kind}/[1-4]\.0(?:/|$)', parsed.path):
            raise ValueError('Commons license name/URL disagree')
    elif normalized in {'CC0', 'PUBLIC DOMAIN', 'PUBLIC DOMAIN MARK'}:
        if normalized == 'CC0':
            if parsed.hostname != 'creativecommons.org' or not parsed.path.startswith('/publicdomain/zero/'):
                raise ValueError('Missing CC0 declaration')
        elif values.get('Copyrighted', '').lower() not in {'false', 'no'}:
            raise ValueError('Public domain status is not explicit')
    else:
        raise ValueError('No accepted explicit individual image license')
    author = plain(values.get('Artist', ''))
    if not author:
        raise ValueError('Missing author attribution')
    return {'license':short, 'license_url':license_url,
        'author':author, 'author_html':values.get('Artist', ''),
        'usage_terms':plain(values.get('UsageTerms', '')),
        'attribution_required':values.get('AttributionRequired', ''),
        'restrictions':plain(values.get('Restrictions', '')),
        'capture_date':plain(values.get('DateTimeOriginal', '')),
        'description':plain(values.get('ImageDescription', ''))}


def valid_photo(info):
    return (info.get('mime') in {'image/jpeg', 'image/png'}
        and 0 < info.get('size', 0) <= 20_000_000
        and min(info.get('width', 0), info.get('height', 0)) >= 320
        and urlparse(info.get('url', '')).hostname == 'upload.wikimedia.org')


def request_json(params):
    for attempt in range(4):
        try:
            response = session().get(API, params=params, timeout=(20, 60))
            response.raise_for_status()
            body = response.json()
            if body.get('error'):
                raise ValueError(f'Commons API error: {body["error"].get("code")}')
            return body
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def load_category(category, folder):
    path = folder / (category.split(':', 1)[-1].replace(' ', '_') + '.json')
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    pages, continuation = [], {}
    for _ in range(20):
        params = {'action':'query','format':'json','generator':'categorymembers',
            'gcmtitle':category,'gcmtype':'file','gcmlimit':50,'prop':'imageinfo',
            'iiprop':'url|extmetadata|mime|size|sha1|timestamp', **continuation}
        data = request_json(params)
        pages.extend(data.get('query',{}).get('pages',{}).values())
        continuation = data.get('continue')
        if not continuation:
            break
        time.sleep(.4)
    save_json(path, pages)
    return pages


def download_photo(info, target):
    expected = info['sha1'].lower()
    if target.exists():
        with target.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha1').hexdigest() == expected:
                return {'download_url':info['url'], 'download_variant':'original_sha1_verified'}
    sidecar=target.with_suffix('.download.json')
    if target.exists() and sidecar.exists():
        prior=json.loads(sidecar.read_text(encoding='utf-8'))
        if sha(target)==prior.get('local_sha256'):
            return prior
    thumbnail=bool(info.get('thumburl'))
    url=info['thumburl'] if thumbnail else info['url']
    dimensions=(info['thumbwidth'],info['thumbheight']) if thumbnail else (info['width'],info['height'])
    part = target.with_suffix(target.suffix + '.part')
    for attempt in range(4):
        try:
            with session().get(url, stream=True, timeout=(20,90)) as response:
                if response.status_code==429:
                    # Stop this request; the caller stops the batch after rate limits.
                    response.raise_for_status()
                response.raise_for_status()
                with part.open('wb') as handle:
                    for chunk in response.iter_content(131072):
                        handle.write(chunk)
                        if handle.tell() > 20_000_000:
                            raise ValueError('Commons photo exceeds size limit')
            with part.open('rb') as handle:
                if not thumbnail and hashlib.file_digest(handle,'sha1').hexdigest() != expected:
                    raise ValueError('Original image SHA1 differs from Commons metadata')
            with Image.open(part) as image:
                image.load()
                if image.size != dimensions:
                    raise ValueError('Commons image dimensions mismatch')
            part.replace(target)
            result={'download_url':url, 'download_variant':'standard_thumbnail' if thumbnail else 'original_sha1_verified',
                    'local_sha256':sha(target), 'width':dimensions[0], 'height':dimensions[1]}
            save_json(sidecar,result)
            return result
        except requests.RequestException as error:
            if getattr(error, 'response', None) is not None and error.response.status_code==429:
                raise
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def acquire(args):
    out = Path(args.output); metadata = out / 'metadata'; metadata.mkdir(parents=True,exist_ok=True)
    pages = {}
    for category in CATEGORIES:
        for page in load_category(category, metadata):
            if page['pageid'] not in pages:
                pages[page['pageid']] = dict(page, retrieved_categories=[])
            pages[page['pageid']]['retrieved_categories'].append(category)
    candidates, excluded = [], []
    for page in pages.values():
        try:
            info = page['imageinfo'][0]
            if not valid_photo(info):
                raise ValueError('Not a supported bounded photograph')
            candidates.append((page, info, license_details(info)))
        except (KeyError, ValueError) as error:
            excluded.append({'page_id':page['pageid'], 'reason':str(error)})
    candidates.sort(key=lambda item:item[0]['pageid'])
    random.Random(42).shuffle(candidates)
    # Separate open/closed categories first, then the wider parent category.
    candidates.sort(key=lambda item:len(item[0]['retrieved_categories']), reverse=True)
    # Wikimedia recommends standard thumbnail widths for bounded automated use.
    # Preserve cached originals and record the different provenance of thumbnails.
    if getattr(args,'thumbnails',False):
        for start in range(0,len(candidates),40):
            batch=candidates[start:start+40]
            path=metadata/f'thumbnails_960_{start:03d}.json'
            if path.exists():body=json.loads(path.read_text(encoding='utf-8'))
            else:
                body=request_json({'action':'query','format':'json','prop':'imageinfo',
                    'pageids':'|'.join(str(p['pageid']) for p,_,_ in batch),
                    'iiprop':'url|size','iiurlwidth':960})
                save_json(path,body)
                time.sleep(.5)
            by_id=body.get('query',{}).get('pages',{})
            for page,info,_ in batch:
                thumb=by_id.get(str(page['pageid']),{}).get('imageinfo',[{}])[0]
                for key in ('thumburl','thumbwidth','thumbheight'):
                    if key in thumb:info[key]=thumb[key]
    print(f'{len(candidates)} licensed photo candidates; target {args.limit}; {len(excluded)} excluded',flush=True)
    images, labels = out / 'images', out / 'labels'
    images.mkdir(parents=True,exist_ok=True); labels.mkdir(parents=True,exist_ok=True)
    rows, errors, seen = [], [], set()

    def work(item):
        page, info, provenance = item
        stem = f'commons_elevator_{page["pageid"]}'
        image = images / (stem + ('.jpg' if info['mime']=='image/jpeg' else '.png'))
        label = labels / (stem + '.txt')
        downloaded=download_photo(info,image)
        if not label.exists():write_labels(label,[])
        # Author/day is a conservative grouping hint, not a known capture session.
        author_hash = hashlib.sha256(provenance['author'].encode()).hexdigest()[:16]
        day = provenance['capture_date'][:10]
        group = f'commons_author_day_{author_hash}_{day}' if re.fullmatch(r'\d{4}-\d{2}-\d{2}',day) else f'commons_image_{page["pageid"]}'
        return record(image,label,'wikimedia_commons_elevator_candidates',group=group,
            session_known=False, group_basis='author_capture_day_hint_unverified',
            source_url=info['descriptionurl'], download_url=downloaded['download_url'],
            download_variant=downloaded['download_variant'], source_title=page['title'],
            source_image_id=page['pageid'], source_revision_timestamp=info.get('timestamp'),
            source_image_sha1=info['sha1'], source_categories=page['retrieved_categories'],
            candidate_classes=['elevator'], candidate_class_ids=[22],
            annotation_status='unannotated', annotation_type='none', verified_negative=False,
            attribution=provenance['author'] + ' — ' + page['title'] + ' — ' + info['descriptionurl'],
            review_reason='Draw all visible objects of the 25-class taxonomy; category retrieval is not a positive box or verified negative',
            **provenance)

    # Small sequential batches keep requests bounded and resume against original hashes.
    for start in range(0,len(candidates),args.workers):
        if len(rows)>=args.limit:
            break
        batch = candidates[start:start+min(args.workers,args.limit-len(rows))]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(work,item):item for item in batch}
            for future in as_completed(futures):
                try:
                    row = future.result()
                    if row['sha256'] not in seen:
                        rows.append(row); seen.add(row['sha256'])
                except Exception as error:
                    errors.append({'page_id':futures[future][0]['pageid'],'reason':str(error)})
        write_manifest(out/'manifest.jsonl',sorted(rows,key=lambda row:row['id']))
        print(f'Downloaded {len(rows)} unannotated candidates',flush=True)
        if sum('429' in e['reason'] for e in errors)>=3:
            print('Stopping acquisition after repeated HTTP429 rate limits',flush=True)
            break
        time.sleep(1)
    report = {'images':len(rows),'unique_image_sha256':len(seen),'candidate_images_per_class':{'elevator':len(rows)},
        'boxes_per_class':{'elevator':0},'annotated_images':0,'annotation_status':'unannotated',
        'approved_images':0,'verified_negative_images':0,'coverage_complete':False,
        'requested':args.limit,'catalog_pages':len(pages),'source_categories':CATEGORIES,
        'licenses':dict(Counter(row['license'] for row in rows)), 'excluded':excluded,'errors':errors,
        'source_api':API,'verification':'Cached original SHA1 verified; standard thumbnails verified by decoded dimensions with local SHA256, not compared to original SHA1',
        'download_variants':dict(Counter(row.get('download_variant') for row in rows)),
        'limitations':'Category membership may include details or irrelevant images. Human selection and complete boxes are required before training.'}
    save_json(out/'download_report.json',report)
    save_json(ROOT/'reports/campus_access_elevator_candidates.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in {'errors','excluded'}},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--limit',type=int,default=100)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--thumbnails',action='store_true',help='Use API-provided standard 960px thumbnails')
    acquire(parser.parse_args())
