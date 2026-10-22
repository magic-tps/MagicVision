"""Integrity checks for explicitly using existing annotations without claiming review."""
from collections import Counter
import json
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import campus22_schema as schema
from data_pipeline import duplicate_groups


def inspect_existing(dataset):
    root = Path(dataset).resolve().parent
    errors = []
    result = {'dataset': str(Path(dataset).resolve()), 'ready': False, 'errors': errors,
              'evaluation_scope': 'experimental_existing_annotations', 'fully_human_reviewed': False}
    try:
        report = json.loads((root/'preparation_report.json').read_text(encoding='utf-8'))
        if report['mode'] != 'all_existing_source_annotations_plus_human_corrections':
            raise ValueError('Not a complete existing-data preparation')
        if schema.sha(root/'manifest.jsonl') != report['manifest_sha256']:
            raise ValueError('Manifest changed since full-data preparation')
        if schema.sha(dataset) != report['yaml_sha256']:
            raise ValueError('YAML changed since full-data preparation')
        data = yaml.safe_load(Path(dataset).read_text(encoding='utf-8'))
        if data['names'] != dict(enumerate(schema.CLASSES)) or Path(data['path']).resolve() != root:
            raise ValueError('Wrong dataset path or class contract')
        rows = schema.read_manifest(root/'manifest.jsonl')
        if len(rows) != report['images'] or len({r['sha256'] for r in rows}) != len(rows):
            raise ValueError('Manifest count changed or duplicate images')
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate record IDs')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))
        return result
    counts = {s: Counter() for s in ('train', 'val', 'test')}
    paths = {s: set() for s in counts}
    rank = {'train': 0, 'val': 1, 'test': 2}
    print(f'Verificando {len(rows)} imagenes del dataset completo...', flush=True)
    for index, row in enumerate(rows, 1):
        try:
            subset = row['split']
            image = Path(row['image']).resolve(); label = Path(row['label']).resolve()
            if subset not in counts or row['resolved_split'] != subset:
                raise ValueError('Invalid split')
            if row.get('original_split') and rank[subset] < rank[row['original_split']]:
                raise ValueError('Original held-out image moved into an earlier split')
            if not image.is_relative_to(root/'images'/subset) or not label.is_relative_to(root/'labels'/subset):
                raise ValueError('File outside split directory')
            if schema.sha(image) != row['sha256'] or schema.sha(label) != row['input_label_sha256']:
                raise ValueError('Image or label differs from prepared content')
            boxes = schema.read_labels(label)
            if not boxes and not row.get('verified_negative'):
                raise ValueError('Unverified empty label')
            if len(schema.dedup_boxes(boxes)) != len(boxes):
                raise ValueError('Repeated boxes')
            if row['annotation_usage'] == 'human_reviewed':
                if not (row.get('review_status') == 'approved' and row.get('coverage_complete')
                        and row.get('provenance_reviewed') and row.get('reviewer')
                        and row.get('reviewed_at') and row.get('reviewed_image_sha256') == row['sha256']
                        and row.get('reviewed_label_sha256') == row['input_label_sha256']):
                    raise ValueError('Human correction lost its review evidence')
            elif row['annotation_usage'] != 'inherited_source_labels':
                raise ValueError('Unknown annotation usage')
            counts[subset].update({schema.CLASSES[b[0]] for b in boxes})
            paths[subset].add(image)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f'{row.get("id")}: {exc}')
        if index % 5000 == 0:
            print(f'Integridad: {index}/{len(rows)} imagenes comprobadas', flush=True)
    for subset in counts:
        if data[subset] != f'images/{subset}':
            errors.append(f'{subset}: unexpected image path')
        disk = {p.resolve() for p in (root/'images'/subset).rglob('*')
                if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}}
        if disk != paths[subset]:
            errors.append(f'{subset}: files differ from manifest')
    missing = {s: [name for name in schema.CLASSES if not counts[s][name]] for s in counts}
    if missing['train']:
        errors.append('Classes absent from training: ' + ', '.join(missing['train']))
    for cluster in duplicate_groups(rows):
        if len({rows[i]['split'] for i in cluster}) > 1:
            errors.append('Related image group crosses partitions: ' + rows[cluster[0]]['id'])
    result.update(ready=not errors, records=len(rows), images_by_split=dict(Counter(r['split'] for r in rows)),
                  legacy_images=report['legacy_images'], added_images=report['added_images'],
                  human_reviewed_images=report['human_reviewed_images'],
                  inherited_annotation_images=report['inherited_annotation_images'],
                  missing_classes_by_split=missing, all_classes_evaluable=not any(missing.values()),
                  images_per_class_by_split={s: dict(c) for s, c in counts.items()},
                  limitations=report['limitations'])
    return result
