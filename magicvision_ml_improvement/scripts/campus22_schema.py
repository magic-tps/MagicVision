"""Active campus contract: exclude cable, wet_floor and elevator; IDs 0..21."""
from pathlib import Path
import campus_schema as source
from data_pipeline import ROOT, read_manifest, write_manifest, save_json, sha, duplicate_groups

EXCLUDED = {'cable', 'wet_floor', 'elevator'}
CLASSES = [('trash_can' if name == 'trash_bin' else name) for name in source.CLASSES if name not in EXCLUDED]
TAXONOMY = 'magicvision-campus22-v1'
OLD_TO_NEW = {i: CLASSES.index('trash_can' if name == 'trash_bin' else name) for i, name in enumerate(source.CLASSES) if name not in EXCLUDED}


def validate_box(box):
    value = source.validate_box(box)
    if value[0] >= len(CLASSES):
        raise ValueError('Class ID outside campus22 taxonomy')
    return value


def read_labels(path):
    return [validate_box(line.split()) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def dedup_boxes(boxes):
    return source.dedup_boxes([validate_box(box) for box in boxes])


def write_labels(path, boxes):
    source.write_labels(path, [validate_box(box) for box in boxes])


def project_boxes(boxes):
    result = []
    for box in boxes:
        box = source.validate_box(box)
        if box[0] in OLD_TO_NEW:
            result.append([OLD_TO_NEW[box[0]], *box[1:]])
    return result
