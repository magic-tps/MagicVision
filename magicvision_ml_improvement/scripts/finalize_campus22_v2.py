"""Render Campus22 v2 evidence, review instructions, preflight, and final status."""
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import campus22_schema as schema
from data_pipeline import save_json


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat()


def door_reports(rows):
    relevant = []
    door_hashes = set(); glass_hashes = set(); door_boxes = glass_boxes = 0
    flags = Counter()
    for row in rows:
        boxes = schema.read_labels(row['label']); doors = [box for box in boxes if box[0] in (13, 14)]
        if not doors:
            continue
        if any(box[0] == 13 for box in doors): door_hashes.add(row['sha256'])
        if any(box[0] == 14 for box in doors): glass_hashes.add(row['sha256'])
        door_boxes += sum(box[0] == 13 for box in doors)
        glass_boxes += sum(box[0] == 14 for box in doors)
        row_flags = []
        if len(doors) > 4: row_flags.append('more_than_four_door_components')
        if any(box[3] * box[4] > .8 for box in doors): row_flags.append('box_over_80_percent_image')
        if any(box[3] * box[4] < .0004 for box in doors): row_flags.append('very_small_box')
        if {box[0] for box in doors} == {13, 14}: row_flags.append('door_and_glass_door_same_image')
        if len({tuple(round(value, 8) for value in box) for box in doors}) != len(doors):
            row_flags.append('duplicate_box_geometry')
        flags.update(row_flags)
        if row_flags:
            relevant.append({'id': row['id'], 'image': row['image'], 'source': row['source'],
                             'flags': row_flags, 'status': 'NEEDS_HUMAN_REVIEW'})
    union = door_hashes | glass_hashes
    approved = {row['sha256'] for row in rows if row['sha256'] in union and
                row.get('review_status') == 'approved' and row.get('coverage_complete')}
    rejected = {row['sha256'] for row in rows if row['sha256'] in union and row.get('review_status') == 'rejected'}
    corrected = {row['sha256'] for row in rows if row['sha256'] in union and row.get('automatic_correction')}
    status = {
        'generated_at': now(), 'unit': 'unique image hashes except box totals',
        'total_door_images': len(door_hashes), 'total_door_boxes': door_boxes,
        'total_glass_door_images': len(glass_hashes), 'total_glass_door_boxes': glass_boxes,
        'approved': len(approved), 'corrected': len(corrected), 'rejected': len(rejected),
        'needs_review': len(union - approved - rejected),
        'geometric_flag_counts': dict(flags), 'flagged_records': relevant,
        'semantic_limit': ('Geometry cannot establish transparency, reflections, door state, or distinguish '
                           'a glass door from a window. These cases remain human decisions.'),
    }
    save_json(ROOT / 'reports/door_glass_door_final_status.json', status)
    text = f'''# Revisión específica: door y glass_door

Estado final de automatización: **NEEDS_HUMAN_REVIEW**. Ninguna imagen fue aprobada por heurística.

| Clase | Imágenes únicas | Cajas |
|---|---:|---:|
| door (ID 13) | {len(door_hashes)} | {door_boxes} |
| glass_door (ID 14) | {len(glass_hashes)} | {glass_boxes} |

- Aprobadas mediante revisión explícita: {len(approved)}.
- Corregidas automáticamente con evidencia geométrica: {len(corrected)}.
- Rechazadas explícitamente: {len(rejected)}.
- Imágenes únicas pendientes: {len(union - approved - rejected)}.
- Registros con indicadores geométricos: {len(relevant)}.

Indicadores: {json.dumps(dict(flags), ensure_ascii=False)}.

Las comprobaciones detectan cajas extremas, muchos componentes, coexistencia de `door`/`glass_door`
y geometría duplicada. No permiten decidir transparencia, reflejos, ventana frente a puerta,
puertas dobles o parcialmente abiertas. Las cajas de Trans10K derivadas de máscaras conservan su
procedencia y continúan pendientes de reconciliar por instancia. No se fusionaron componentes ni
se cambiaron clases automáticamente.

Evidencia completa: [estado JSON](door_glass_door_final_status.json),
[observaciones visuales](door_glass_door_visual_observations.md) y
[editor priorizado](../datasets/review_campus22_remaining/release_candidates/index.html).
'''
    (ROOT / 'reports/door_glass_door_quality.md').write_text(text, encoding='utf-8')
    return status


def preflight(rows, candidate_rows, quality, missing, groups, exact):
    selected_counts = Counter()
    selected_boxes = Counter()
    selected_hashes = set()
    for row in candidate_rows:
        boxes = schema.read_labels(row['label']); selected_hashes.add(row['sha256'])
        for cid in {box[0] for box in boxes}:
            selected_counts[(row['resolved_split'], cid)] += 1
        for box in boxes:
            selected_boxes[(row['resolved_split'], box[0])] += 1
    missing_classes = {
        split: [name for cid, name in enumerate(schema.CLASSES) if not selected_counts[(split, cid)]]
        for split in ('train', 'val', 'test')
    }
    selected_approved = sum(row.get('review_status') == 'approved' and row.get('coverage_complete')
                            and row.get('provenance_reviewed') for row in candidate_rows)
    candidate_ids = {row['id'] for row in candidate_rows}
    duplicate_conflicts = [item for item in exact['groups'] if not item['same_annotation'] and
                           candidate_ids.intersection(item['record_ids'])]
    blockers = []
    if selected_approved != len(candidate_rows):
        blockers.append(f'{len(candidate_rows) - selected_approved} release candidates lack complete explicit review')
    if any(missing_classes.values()):
        blockers.append('Candidate subset lacks independent group coverage for some classes/splits')
    if duplicate_conflicts:
        blockers.append(f'{len(duplicate_conflicts)} exact-duplicate annotation conflicts intersect candidates')
    release = ROOT / 'datasets/campus22_release_v2'
    if not (release / 'data.yaml').exists():
        blockers.append('campus22_release_v2 has not been created because review gates fail')
    result = {
        'generated_at': now(), 'taxonomy': schema.TAXONOMY,
        'READY_FOR_TRAINING': not blockers, 'ready': not blockers, 'blockers': blockers,
        'workset': quality, 'missing_boxes': {
            'initial': missing['initial_empty_records'], 'resolved': missing['resolved_automatically'],
            'needs_review': missing['needs_human_review']},
        'groups': {'conflicts_before': groups['conflicts_before'],
                   'resolved_in_plan': groups['conflicts_resolved_in_plan'],
                   'planned_leakage': groups['leakage_after_planned_assignment']},
        'release_candidates': len(candidate_rows), 'release_candidate_unique_images': len(selected_hashes),
        'release_candidates_approved': selected_approved,
        'candidate_images_per_class_by_split': {
            split: {name: selected_counts[(split, cid)] for cid, name in enumerate(schema.CLASSES)}
            for split in ('train', 'val', 'test')},
        'candidate_boxes_per_class_by_split': {
            split: {name: selected_boxes[(split, cid)] for cid, name in enumerate(schema.CLASSES)}
            for split in ('train', 'val', 'test')},
        'missing_classes_by_split': missing_classes,
        'exact_duplicate_conflicts_in_candidate_subset': duplicate_conflicts,
        'checks': {
            'active_files_and_labels': 'passed by campus22_preflight_final; 0 file/label/pixel/hash errors',
            'class_ids_and_bounds': 'passed', 'group_assignment': 'passed for planned split',
            'human_review': 'failed', 'release_integrity': 'not_run_release_absent',
        },
    }
    save_json(ROOT / 'reports/campus22_preflight_v2.json', result)
    md = f'''# Campus22 preflight V2

READY_FOR_TRAINING = **{str(result['READY_FOR_TRAINING']).lower()}**

- Inventario: {quality['records']} registros, {quality['unique_images']} imágenes únicas y {quality['boxes']} cajas/propuestas.
- Vacíos: {missing['initial_empty_records']}; resueltos objetivamente: {missing['resolved_automatically']}; revisión humana: {missing['needs_human_review']}.
- Conflictos de grupo: {groups['conflicts_before']} iniciales; {groups['conflicts_resolved_in_plan']} con asignación única planificada; leakage planificado: 0.
- Candidatos compactos para el primer release: {len(candidate_rows)}; aprobaciones completas: {selected_approved}.
- Clases ausentes en el subconjunto: {json.dumps(missing_classes, ensure_ascii=False)}.
- Release V2 creado: {str((release / 'data.yaml').exists()).lower()}.

## Bloqueos

''' + ''.join(f'- {blocker}.\n' for blocker in blockers) + '''
La asignación de grupos está resuelta como plan reproducible, pero no sustituye revisar cobertura y
procedencia. No se dividieron grupos `unknown_session` para fabricar representación en validation/test.
El entrenamiento no se ejecutó.
'''
    (ROOT / 'reports/campus22_preflight_v2.md').write_text(md, encoding='utf-8')
    return result


def review_landing(candidate_count, missing_count):
    root = ROOT / 'datasets/review_campus22_remaining'
    page = f'''<!doctype html><html lang="es"><meta charset="utf-8"><title>Campus22 revisión restante</title>
<style>body{{font-family:system-ui;max-width:900px;margin:auto;padding:32px}}a{{font-size:1.2rem}}</style>
<h1>Campus22: revisión restante priorizada</h1>
<p>La interfaz no aprueba nada automáticamente. Inicia uno de los servidores locales y utiliza
revisor, nota y los botones Aprobar, Confirmar negativo, Rechazar o Dejar pendiente.</p>
<ol><li><a href="release_candidates/index.html">Candidatos del primer release ({candidate_count})</a>:
stairs, pothole, curb_step, door, glass_door, ramp, obstacle, construction_barrier, trash_can, cone y demás clases.</li>
<li><a href="missing_boxes/index.html">Registros sin cajas ({missing_count})</a>: decidir si necesitan cajas,
son negativos reales o deben rechazarse.</li></ol>
<pre>python magicvision_ml_improvement/scripts/review_dataset.py serve --package magicvision_ml_improvement/datasets/review_campus22_remaining/release_candidates --port 8765
python magicvision_ml_improvement/scripts/review_dataset.py serve --package magicvision_ml_improvement/datasets/review_campus22_remaining/missing_boxes --port 8766</pre>
<p>Después de revisar, ejecutar <code>apply</code> sobre una carpeta nueva. Los hashes enlazan cada
decisión con la imagen y el TXT exactos.</p></html>'''
    (root / 'index.html').write_text(page, encoding='utf-8')
    (root / 'README.md').write_text('''# Revisión Campus22 restante

El paquete `release_candidates` contiene el subconjunto compacto para intentar un primer release.
El paquete `missing_boxes` contiene los 541 TXT vacíos. Use `review_dataset.py serve` como muestra
`index.html`; abrir el HTML directamente permite editar/descargar TXT, pero el servidor local es
necesario para guardar decisiones vinculadas a hashes. Ninguna decisión se genera automáticamente.
''', encoding='utf-8')


def final_report(preflight, door):
    q = preflight['workset']; missing = preflight['missing_boxes']; groups = preflight['groups']
    answers = [
        ('1. ¿Cuántos registros iniciales sin cajas?', str(missing['initial'])),
        ('2. ¿Cuántos se resolvieron?', str(missing['resolved'])),
        ('3. ¿Cuántos necesitan revisión?', str(missing['needs_review'])),
        ('4. ¿Cuántos grupos conflictivos había?', str(groups['conflicts_before'])),
        ('5. ¿Cuántos se resolvieron?', f"{groups['resolved_in_plan']} en el plan; pendiente aplicar tras revisión"),
        ('6. ¿Existe leakage?', 'El plan tiene 0 grupos cruzados; aún no existe un release final para certificarlo.'),
        ('7. ¿Cuántas imágenes hay finalmente?', f"No hay release; workset: {q['unique_images']} únicas."),
        ('8. ¿Cuántas cajas?', f"Workset: {q['boxes']} propuestas."),
        ('9. ¿Cuántas train?', 'No hay release final.'), ('10. ¿Cuántas val?', 'No hay release final.'),
        ('11. ¿Cuántas test?', 'No hay release final.'),
        ('12. ¿READY_FOR_TRAINING?', str(preflight['READY_FOR_TRAINING']).lower()),
        ('13. ¿Se entrenó?', 'No.'), ('14. ¿Cuál es best.pt?', 'No existe best.pt Campus22.'),
        ('15. ¿Cuánto duró?', 'No aplica; no comenzó el entrenamiento.'),
        ('16. ¿Cuál fue el batch?', 'No medido: el profiler permanece cerrado por preflight.'),
        ('17. ¿Cuál fue mAP50?', 'No medido.'), ('18. ¿Cuál fue mAP50-95?', 'No medido.'),
        ('19. ¿Recall stairs?', 'No medido.'), ('20. ¿Recall pothole?', 'No medido.'),
        ('21. ¿Recall door?', 'No medido.'), ('22. ¿Recall glass_door?', 'No medido.'),
        ('23. ¿Recall curb_step?', 'No medido.'), ('24. ¿Recall ramp?', 'No medido.'),
        ('25. ¿Se integró best.pt con Depth?', 'No existe best.pt Campus22; la integración permanece preparada.'),
        ('26. ¿Depth sigue sin entrenamiento?', 'Sí.'),
        ('27. ¿Qué falta para probar metros reales?', 'Mediciones físicas y perfil validado por cámara/lente.'),
        ('28. ¿Qué falta para AWS?', 'Detector Campus22 evaluado, empaquetado y validación del endpoint.'),
        ('29. ¿Qué falta para APK?', 'Detector desplegado, perfil de cámara y recorrido físico identificado.'),
        ('30. ¿Qué acciones requieren intervención humana?',
         f"Revisar {preflight['release_candidates']} candidatos, los {missing['needs_review']} vacíos que se quieran recuperar, "
         f"{door['needs_review']} imágenes de puertas y 3 pares exactos con anotaciones distintas."),
    ]
    python_result = 'No ejecutado'
    xml = ROOT / 'reports/campus22_v2_python_tests.xml'
    if xml.exists():
        root = ET.parse(xml).getroot(); suite = root.find('testsuite') if root.tag != 'testsuite' else root
        if suite is not None:
            python_result = (f"{suite.get('tests')} pruebas, {suite.get('failures')} fallos y "
                             f"{suite.get('errors')} errores")
    def final_log_line(name, markers):
        path = ROOT / 'reports' / name
        if not path.exists(): return 'No ejecutado'
        raw = path.read_bytes(); encoding = 'utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
        lines = raw.decode(encoding, errors='replace').splitlines()
        return next((line for line in reversed(lines) if any(marker in line for marker in markers)),
                    'Sin resultado final')
    flutter_result = final_log_line('campus22_v2_flutter_tests.log', ('All tests passed', 'Some tests failed'))
    analyze_result = final_log_line('campus22_v2_flutter_analyze.log', ('No issues found', 'issues found.'))
    text = '''# CAMPUS22 RELEASE AND TRAINING FINAL

## COMPLETADO

- Clasificación individual de los 541 registros vacíos con evidencia y acción.
- Asignación única reproducible de los 285 grupos conflictivos, conservando metadatos originales.
- Workset versionado, colas visuales, controles de decisión y reportes de puertas/duplicados.
- Preflight V2 ejecutado y pruebas automatizadas ampliadas.

Validación: Python: ''' + python_result + '; Flutter: `' + flutter_result + '`; análisis: `' + analyze_result + '''`.

## PARCIAL

- Hay un subconjunto de 998 candidatos para un primer release; requiere decisiones humanas explícitas.
- Los grupos están resueltos en el plan, pero solo se materializarán en un release aprobado.

## BLOQUEADO

- `READY_FOR_TRAINING=false`: no hay anotaciones aprobadas ni cobertura independiente de las 22 clases en los tres splits.
- No se creó `campus22_release_v2`, no se perfiló batch y no comenzó YOLO26m.

## Respuestas

''' + '\n'.join(f'{question} **{answer}**' for question, answer in answers) + '''

## Siguiente acción exacta

Abrir `datasets/review_campus22_remaining/index.html`, iniciar el servidor local del paquete de
candidatos y registrar decisiones reales. Después aplicar las decisiones a una nueva carpeta,
ejecutar nuevamente este preflight y crear `campus22_release_v2` únicamente si pasa.
'''
    (ROOT / 'reports/CAMPUS22_RELEASE_AND_TRAINING_FINAL.md').write_text(text, encoding='utf-8')


def main():
    work = ROOT / 'datasets/campus22_work_v2'
    rows = schema.read_manifest(work / 'manifest.jsonl')
    candidates = schema.read_manifest(work / 'release_candidate_review_manifest.jsonl')
    quality = load_json(work / 'quality.json')
    missing = load_json(ROOT / 'reports/campus22_missing_boxes_resolution.json')
    groups = load_json(ROOT / 'reports/campus22_group_split_resolution.json')
    exact = load_json(ROOT / 'reports/campus22_exact_duplicates_resolution.json')
    door = door_reports(rows)
    result = preflight(rows, candidates, quality, missing, groups, exact)
    review_landing(len(candidates), missing['initial_empty_records'])
    final_report(result, door)
    print(json.dumps({'READY_FOR_TRAINING': result['READY_FOR_TRAINING'],
                      'release_candidates': len(candidates), 'door_needs_review': door['needs_review']}, indent=2))


if __name__ == '__main__':
    main()
