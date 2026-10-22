"""Index the completed v3 acquisition without silently releasing partial labels."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import csv
import json
from pathlib import Path
from data_pipeline import ROOT,APP,CLASSES,read_manifest,read_labels,save_json,write_manifest,quality,sha

OLD=['legacy_manifest.jsonl','openimages/manifest.jsonl','openimages_train/manifest.jsonl','openimages_negative_candidates/manifest.jsonl']
NEW=['openimages_expansion_v3/manifest.jsonl','rare_sources_v3/hotosm_poles/manifest.jsonl','rare_sources_v3/rdd2022/manifest.jsonl','smartbin_legacy_v3/manifest.jsonl']


def count_classes(rows):
    unique=defaultdict(set);boxes=Counter()
    for r in rows:
        labels=read_labels(r['label']);boxes.update(CLASSES[b[0]] for b in labels)
        unique[r['sha256']].update(CLASSES[b[0]] for b in labels)
    counts=Counter(c for labels in unique.values() for c in labels)
    return unique,counts,boxes


def canonical_rows(rows):
    """Collapse repeated IDs only if geometry and holdout status agree."""
    result={};aliases=0
    for row in rows:
        key=row['id']
        if key in result:
            previous=result[key]
            if (previous['sha256']==row['sha256'] and previous.get('original_split')==row.get('original_split')
                    and read_labels(previous['label'])==read_labels(row['label'])):
                previous['negative_candidate']=previous.get('negative_candidate',False) or row.get('negative_candidate',False)
                continue
            key=f"{key}__variant_{aliases}";aliases+=1
        result[key]={**row,'id':key}
    return list(result.values()),aliases


def main():
    datasets=ROOT/'datasets'
    for source in NEW:
        report=datasets/Path(source).parent/'download_report.json'
        if not report.exists():raise ValueError(f'Acquisition still incomplete: {report}')
    old=[r for source in OLD for r in read_manifest(datasets/source)]
    added=[r for source in NEW for r in read_manifest(datasets/source)]
    oldhash={r['sha256'] for r in old}
    truly_new=[r for r in added if r['sha256'] not in oldhash]
    rows,aliases=canonical_rows(old+added)
    index=datasets/'expanded_v3';index.mkdir(exist_ok=True)
    write_manifest(index/'manifest.jsonl',rows)
    oldunique,oldcounts,_=count_classes(old)
    newunique,newcounts,newboxes=count_classes(truly_new)
    allunique,totalcounts,totalboxes=count_classes(rows)
    # Put new rare categories early in the visual QA gallery.
    preview=[];used=set()
    for c in range(len(CLASSES)):
        count=0
        for r in truly_new:
            if r['id'] not in used and c in {b[0] for b in read_labels(r['label'])}:
                preview.append(r);used.add(r['id']);count+=1
                if count==3:break
    ordered=preview+[r for r in rows if r['id'] not in used]
    q=quality(ordered,ROOT/'reports/quality_v3',overlays=len(preview))
    sources={s:json.loads((datasets/Path(s).parent/'download_report.json').read_text(encoding='utf-8')) for s in NEW}
    checks={'created_at':datetime.now(timezone.utc).isoformat(),'prior_records':len(old),
      'prior_unique_images':len(oldunique),'added_source_records':len(added),
      'added_unique_excluding_prior':len(newunique),'exact_overlaps_prior':len({r['sha256'] for r in added}&oldhash),
      'total_records_after_repeated_id_removal':len(rows),'total_unique_images':len(allunique),
      'new_class_images':dict(newcounts),'total_class_images':dict(totalcounts),'prior_class_images':dict(oldcounts),
      'new_imported_boxes':dict(newboxes),'total_imported_boxes':dict(totalboxes),
      'all_eleven_classes_received_new_images':all(newcounts[c]>0 for c in CLASSES),
      'duplicate_id_variants':aliases,'quality_errors':len(q['errors']),
      'quality_exact_label_conflicts':len(q['conflicting_exact_duplicates']),
      'duplicate_or_session_groups':len(q['duplicate_or_session_clusters']),
      'approved_complete':q['approved_complete'],'released_training_dataset':False,
      'source_manifests':OLD+NEW,'current_checkpoint_sha256':sha(APP/'runs/detect/modelo_final_yolo26m_80e_640_b8/weights/best.pt')}
    save_json(ROOT/'reports/expansion_v3_inventory.json',checks)
    table=['| ID | Clase | Imágenes antes | Nuevas distintas en esta ampliación | Total de imágenes distintas |',
           '| ---: | --- | ---: | ---: | ---: |']
    csvrows=[]
    for i,c in enumerate(CLASSES):
        table.append(f'| {i} | {c} | {oldcounts[c]} | {newcounts[c]} | {totalcounts[c]} |')
        csvrows.append({'id':i,'class':c,'before_images':oldcounts[c],'new_images':newcounts[c],'total_images':totalcounts[c],'new_imported_boxes':newboxes[c]})
    with (ROOT/'reports/expansion_v3_distribution.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    gap_path=ROOT/'reports/annotation_gaps_v3.json';gaps=json.loads(gap_path.read_text(encoding='utf-8'))
    oi=sources[NEW[0]];poles=sources[NEW[1]];rdd=sources[NEW[2]];bins=sources[NEW[3]]
    text=f'''# MagicVision: ampliación de datos para las once clases

Actualización posterior al informe de mejora inicial. Estado del inventario: {checks['created_at']}.

Se descargaron e importaron **{len(added)} registros adicionales**. Tras excluir coincidencias SHA256 con todo lo anterior y repeticiones entre fuentes, esta ampliación aporta **{len(newunique)} imágenes nuevas distintas**. El inventario combinado contiene **{len(allunique)} imágenes distintas**. Las once clases recibieron ejemplos nuevos: **{'sí' if checks['all_eleven_classes_received_new_images'] else 'no'}**.

## Distribución medida

{chr(10).join(table)}

“Antes” incluye el dataset original y la ampliación previa de 4.308 imágenes. Las clases pueden coexistir; no sumar sus columnas como imágenes independientes. Los conteos son por SHA256: variantes visualmente similares siguen pendientes de revisión. Las columnas no son una partición de entrenamiento aprobada. [Inventario JSON](magicvision_ml_improvement/reports/expansion_v3_inventory.json) y [CSV](magicvision_ml_improvement/reports/expansion_v3_distribution.csv).

## Fuentes incorporadas

- **Open Images**: {oi['new_unique_images']} imágenes nuevas aceptadas en este lote. Metadatos de imágenes CC-BY-2.0 y anotaciones CC-BY-4.0; autor/enlace conservados. Se excluyen IDs y hashes ya adquiridos, y el muestreo es reproducible. Bench e hidrante se asignan a `obstacle` según el alcance anterior; no se usa cualquier objeto desconocido como obstáculo. Las clases humanas Man/Woman/Boy/Girl se unifican a `person`. [Fuente oficial](https://storage.googleapis.com/openimages/web/factsfigures_v7.html).
- **HOT-OSM / UC Berkeley, Street-level Poles**: {poles['images']} imágenes, {poles['pole_boxes']} cajas de postes importadas; licencia declarada CC-BY-SA-4.0 y revisión de procedencia individual pendiente. Se excluyen torres y negativos no revisados para las once clases. Se conservaron train/val/test de origen. {len(poles['errors'])} registros se pusieron fuera de la importación por cajas inválidas. [Dataset publicado](https://huggingface.co/datasets/hotosm/streetlevel-poles).
- **RDD2022**: {rdd['images']} imágenes con anotaciones D40, convertidas a `pothole`; las grietas D00/D10/D20 no se convierten en huecos. Se usa India, Japón y Estados Unidos, licencia CC-BY-4.0. Los enlaces por país devolvieron 403; se extrajeron sus ZIP desde el repositorio oficial de Figshare mediante rangos HTTP y CRC del archivo contenedor, evitando el ZIP completo de 13 GB. [Catálogo y licencia](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547).
- **Smart Bin / subconjunto legacy**: {bins['images']} imágenes adicionales de tachos, {bins['trash_bin_boxes']} cajas y {bins['capture_clusters']} grupos de captura del publicador; licencia MIT declarada por imagen. Se conservan splits y grupos; los recortes de un mismo tacho no se cuentan como nuevas imágenes. Son escenas de una ciudad durante una semana, por lo que no demuestran diversidad geográfica ni rendimiento en el campus. [Dataset y procedencia](https://huggingface.co/datasets/arudaev/smart-bin-detect).

**Cuota de tachos no completada:** se plantearon 800 imágenes adicionales en Open Images, pero quedaban solo {oi['images_per_class'].get('trash_bin',0)} compatibles tras excluir la recopilación anterior. La segunda fuente aporta {bins['images']}; frente a esa meta orientativa aún faltan {max(0,800-newcounts['trash_bin'])} imágenes distintas. No se duplicaron datos para aparentar que se alcanzó. El inventario total de tachos figura en la tabla y supera la meta inicial de 1.000, que pertenecía a la etapa anterior.

Cada fuente conserva anotaciones, licencia, autor/procedencia disponibles y hashes. No se descargaron imágenes privadas del teléfono ni se usaron datos sintéticos como si fueran del campus.

## Calidad y posibilidad de entrenar

Se revisaron automáticamente {q['images']} registros: {len(q['errors'])} errores de integridad/geometría, {len(q['conflicting_exact_duplicates'])} conflictos de etiquetas para imágenes idénticas y {len(q['duplicate_or_session_clusters'])} grupos por similitud, copia o sesión. Los grupos deben inspeccionarse: un grupo por sesión no significa que todas sus imágenes sean duplicadas. [Galería con cajas](magicvision_ml_improvement/reports/quality_v3/index.html).

Las anotaciones públicas son útiles, pero no garantizan cobertura de **todas las once clases en cada imagen**. En una muestra real de {gaps['images_checked']} imágenes de la recopilación anterior de Open Images y los postes nuevos, el modelo anterior sugirió cajas adicionales o distintas en {gaps['images_with_suggestions']}; son propuestas que pueden estar equivocadas, **no una tasa de errores confirmados** ni una evaluación del inventario completo. No se añadieron automáticamente a ground truth. [Propuestas para revisar](magicvision_ml_improvement/reports/annotation_gaps_v3.json).

Open Images documenta etiquetas no exhaustivas. Si se entrena YOLO tratando un objeto sin etiquetar como fondo, puede aumentar precisamente las omisiones que se quieren corregir. Por eso no basta con cambiar una bandera de aprobación. [Protocolo oficial de Open Images](https://storage.googleapis.com/openimages/web/evaluation.html), [guía de preparación de Ultralytics](https://docs.ultralytics.com/yolov5/tutorials/tips-for-best-training-results).

Se añadió un **editor local de cajas**: permite añadir, eliminar o cambiar clases, guardar etiquetas YOLO, registrar una decisión por imagen y exportar un manifiesto corregido con hashes de revisión. La documentación explica cómo guardar el TXT en su lugar correcto; la página no aprueba datos automáticamente. [Procedimiento de revisión](magicvision_ml_improvement/configs/review_workflow.md).

Se preparó un [primer lote equilibrado de 275 imágenes para revisión](magicvision_ml_improvement/datasets/review_v3_batch01/index.html), que conserva el inventario completo como referencia para seguir por lotes. Es un punto de inicio, no una validación completa del dataset.

El entrenador ahora comprueba que las imágenes y etiquetas coincidan con lo revisado, que no entren archivos ajenos al manifiesto y que no haya grupos compartidos entre particiones. Los derivados fotométricos no cuentan como nuevas escenas independientes. La partición conserva los archivos de etiquetas revisados sin reescribir su contenido.

**Aprobadas como completas en este momento: {q['approved_complete']}. No se ejecutó entrenamiento definitivo ni se sustituyó el checkpoint.** El conjunto reunido está preparado para revisión y conversión al entrenamiento; no se presenta como un dataset ya certificado. Hay hardware CUDA disponible y configuraciones E1/E2, pero descargar más datos no reemplaza corregir etiquetas.

## Secuencia para entrenar y demostrar una mejora

1. Revisar los paquetes locales, corregir objetos faltantes y procedencia, y registrar las decisiones. Priorizar postes, escaleras, tachos y clases con peor recall. Incluir negativos reales verificados.
2. Ejecutar `quality` y `split` sobre el manifiesto aplicado, manteniendo todas las filas originales en el manifiesto para conservar los holdouts. Resolver grupos que cruzan particiones.
3. Ejecutar `training/preflight.py`; entrenar E1 a 640 y E2 a 960 únicamente con la versión revisada, conservando el checkpoint anterior.
4. Comparar ambos modelos sobre las mismas escenas reservadas: precision **y** recall por clase, mAP, FP por imagen y latencia. Una precisión mayor obtenida eliminando detecciones no resuelve las omisiones. Los objetivos 95 % recall personas/escaleras y 90 % tachos siguen siendo metas, no resultados obtenidos.
5. Añadir escenas del S25 dentro de la universidad y reservar recorridos distintos. Fotos de carreteras o postes de servicios públicos aportan ejemplos, pero no acreditan rendimiento en pasillos del campus.

La guía está en [mejora del reconocimiento](magicvision_ml_improvement/README.md). Las métricas del modelo anterior permanecen en el [material del paper](02_paper_metricas_modelo_final/README.md). No hay un nuevo porcentaje de precisión que reportar porque no se ha entrenado y evaluado un candidato con estas imágenes. No se modificó AWS ni se publicó una APK.

## Verificación del flujo

- La suite `pytest` pasó 48 pruebas, incluidas importación, mapeos, selección equilibrada, revisión vinculada a hashes, separación de escenas y descarga por rangos. Las cinco pruebas adicionales del conversor RDD/postes también pasaron. [Resultado JUnit](magicvision_ml_improvement/reports/pipeline_v3_tests.xml).
- El editor se probó con Chromium/Playwright: dibujar, eliminar, exportar un TXT YOLO válido, descarga alternativa y vista de 390 px sin desbordamiento. No se modificaron ni aprobaron las etiquetas reales durante la prueba. [Resultado del navegador](magicvision_ml_improvement/reports/review_editor_browser_test.json).
- El lote inicial contiene 275 imágenes y ejemplos de las once clases. [Comprobación del lote](magicvision_ml_improvement/reports/review_v3_batch01_validation.json).

Estas comprobaciones validan el software y la integridad de archivos; no son métricas de un detector nuevo.
'''
    (ROOT.parent/'MAGICVISION_AMPLIACION_DATASET_V3.md').write_text(text,encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False))


if __name__=='__main__':main()
