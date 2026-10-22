"""Versioned 25-class acquisition schema. Does not change the active 11-class model.

Acquisitions are candidates: public boxes do not establish complete campus labels.
"""
from pathlib import Path
import math
from data_pipeline import (ROOT, CLASSES as BASE_CLASSES, record as base_record,
                           sha, save_json, write_manifest, read_manifest, duplicate_groups)

CLASSES = BASE_CLASSES + ['ramp', 'curb_step', 'door', 'glass_door', 'chair',
    'table_desk', 'backpack', 'cable', 'wet_floor', 'construction_barrier',
    'handrail', 'elevator', 'cone', 'fire_extinguisher']
TAXONOMY = 'magicvision-campus-v4-25'


def validate_box(box):
    if len(box) != 5:
        raise ValueError('YOLO box must have five values')
    cls, x, y, w, h = map(float, box)
    if not all(math.isfinite(v) for v in (cls, x, y, w, h)):
        raise ValueError('Nonfinite box')
    if not cls.is_integer() or not 0 <= cls < len(CLASSES):
        raise ValueError('Class ID outside campus-v4 taxonomy')
    if w <= 0 or h <= 0 or min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
        raise ValueError('Box outside image or empty')
    return [int(cls), x, y, w, h]


def from_xyxy(cls, left, top, right, bottom):
    return validate_box([cls, (left+right)/2, (top+bottom)/2, right-left, bottom-top])


def read_labels(path):
    return [validate_box(line.split()) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def dedup_boxes(boxes):
    result = []; seen = set()
    for box in boxes:
        box = validate_box(box)
        key = tuple(round(v, 6) for v in box)
        if key not in seen:
            seen.add(key); result.append(box)
    return result


def write_labels(path, boxes):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(f'{b[0]} ' + ' '.join(f'{v:.8f}' for v in b[1:]) + '\n'
                           for b in dedup_boxes(boxes)), encoding='utf-8')


def record(image, label, source, group=None, **extra):
    extra.setdefault('taxonomy', TAXONOMY)
    extra.setdefault('review_status', 'needs_review')
    extra.setdefault('coverage_complete', False)
    extra.setdefault('provenance_reviewed', False)
    extra.setdefault('review_reason', 'Review source provenance and all 25 campus categories before training')
    return base_record(image, label, source, group=group, **extra)
