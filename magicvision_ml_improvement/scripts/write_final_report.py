"""Build the Spanish delivery report exclusively from completed local evidence."""
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from data_pipeline import ROOT,APP,CLASSES,sha,save_json


def main():
    def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
    inventory=read('reports/dataset_inventory.json');quality=read('reports/quality_final/quality.json')
    transfer=read('reports/transfer_probe.json');hardware=read('reports/hardware.json')
    split=read('datasets/magicvision_v2/split_report.json')
    integration=read('reports/local_integration_final/summary.json')
    model=APP/'runs/detect/modelo_final_yolo26m_80e_640_b8/weights/best.pt'
    digest=sha(model)
    assert digest=='704361e5746e23160b95976d342484d250a7931fabbbe798dd1d4e48d269367f'
    def log_text(name):
        raw=(ROOT/'reports'/name).read_bytes()
        return raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')
    flutter_log=log_text('flutter_tests.log')
    assert 'All tests passed!' in flutter_log
    assert 'No issues found!' in log_text('flutter_analyze.log')
    assert 'Dataset not released' in log_text('training_gate.log')
    flutter_passed=int(re.findall(r'\+(\d+)\s+~\d+: All tests passed!',flutter_log)[-1])
    test_counts={}
    for name in ('pipeline','backend'):
        suites=ET.parse(ROOT/f'reports/{name}_tests.xml').getroot().findall('testsuite')
        assert all(int(s.attrib.get('failures',0))==0 and int(s.attrib.get('errors',0))==0 for s in suites)
        test_counts[name]=sum(int(s.attrib['tests'])-int(s.attrib.get('skipped',0)) for s in suites)
    changes=[]
    for original in (ROOT/'original_sources').rglob('*'):
        if not original.is_file():continue
        relative=original.relative_to(ROOT/'original_sources');current=APP/relative
        if current.exists() and sha(current)!=sha(original):
            changes.append({'path':str(relative),'before_sha256':sha(original),'after_sha256':sha(current)})
    save_json(ROOT/'reports/changed_files.json',changes)
    artifact=ROOT/'models/baseline_704361e5_640.tar.gz'
    save_json(ROOT/'reports/release_status.json',{'candidate_ready':False,'reason':'Reviewed complete eleven-class dataset unavailable',
      'operational_checkpoint_sha256':digest,'rollback_artifact_sha256':sha(artifact),
      'production_deployed':False,'apk_published':False,'installed_phone_apk_verified':False,
      'cloud_architecture_verified_remotely':False,'new_training_runs':[]})
    counts=inventory['images_per_class_excluding_exact_legacy_duplicates']
    table=['| ID | Clase | Cajas originales (train+val+test) | Cajas nuevas importadas | Imágenes nuevas, SHA único |',
           '| ---: | --- | ---: | ---: | ---: |']
    for c,name in enumerate(CLASSES):
        table.append(f"| {c} | {name} | {inventory['annotation_boxes_legacy'].get(name,0)} | {inventory['annotation_boxes_imported'].get(name,0)} | {counts.get(name,0)} |")
    original=['| Clase | Cajas train | Imágenes train | Cajas val | Cajas test |','| --- | ---: | ---: | ---: | ---: |']
    with (ROOT.parent/'MAGICVISION_DISTRIBUCION_DATASET.csv').open(encoding='utf-8-sig') as f:distribution=list(csv.DictReader(f))
    for name in CLASSES[:-1]:
        rows={r['split']:r for r in distribution if r['clase']==name}
        original.append(f"| {name} | {rows['train']['anotaciones']} | {rows['train']['imagenes_con_clase']} | {rows['val']['anotaciones']} | {rows['test']['anotaciones']} |")
    evaluation=['| Ejecución / 1.020 imágenes val | mAP50 | mAP50–95 | Recall persona | Recall escalera (n=10) | Latencia media / p95 ms | JPEG/archivo medio bytes |',
                '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    sweeps=['| Entrada / inferencia | Confianza | Recall personas | FP personas/imagen | Recall escaleras | FP escaleras/imagen |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    for folder in ('E0_640','E0_960','E0_640_appjpeg'):
        run=read(f'reports/{folder}/run.json');m=read(f'reports/{folder}/metrics_0.30.json')
        evaluation.append(f"| {folder} | {m['map50']:.4f} | {m['map50_95']:.4f} | {m['classes']['person']['recall']:.4f} | {m['classes']['stairs']['recall']:.4f} | {run['latency_mean_ms']:.1f} / {run['latency_p95_ms']:.1f} | {run['input_bytes_mean']:.0f} |")
        for t in (.15,.20,.25,.30):
            m=read(f'reports/{folder}/metrics_{t:.2f}.json');p=m['classes']['person'];s=m['classes']['stairs']
            sweeps.append(f"| {folder} | {t:.2f} | {p['recall']:.4f} | {p['fp_per_image']:.4f} | {s['recall']:.4f} | {s['fp_per_image']:.4f} |")
    source_counts='; '.join(f'{name}: {n}' for name,n in inventory['source_counts'].items())
    text=f'''# MagicVision: informe de mejora de datos, detección e integración

Fecha de trabajo: 20/09/2026, America/Lima. Evidencia local; algunos registros usan UTC del 21/09.

Se implementó y probó el pipeline de datos, se descargaron **{inventory['new_records']} registros nuevos**, se ejecutaron tres evaluaciones completas del checkpoint actual y se corrigieron pérdidas de información entre detección y descripción. **No se entrenó ni desplegó un detector nuevo de once clases**: faltan revisión completa de anotaciones y capturas del S25. El soporte de `trash_bin` está preparado y probado con contratos; los pesos operativos originales permanecen intactos.

## 1. Diagnóstico del repositorio

La app está en `01_app_magicvision_final/app_flutter_magicvision`. Hay backend FastAPI/Lambda y servidor compatible con SageMaker; no se verificó remotamente que sean la infraestructura de la APK instalada. No había teléfono en `adb devices` ni endpoint real identificado. `main.dart` instancia `UsbCameraSource`, mientras el usuario reporta cámara interna. Se conserva esa selección hasta identificar la versión real; no se atribuyen estas correcciones a la APK del teléfono.

Se respaldaron 33 fuentes antes de los cambios en `magicvision_ml_improvement/original_sources`. Esta raíz no es un repositorio Git. El checkpoint anterior, dataset original y entrega AWS empaquetada se conservaron. Auditoría: [audit_initial.json](magicvision_ml_improvement/reports/audit_initial.json).

## 2. Modelo anterior

YOLO26m, Ultralytics 8.4.56, diez clases. Checkpoint de {model.stat().st_size:,} bytes, SHA256 `{digest}`. Se volvió a verificar ese hash al generar este informe. El contenedor existente usa PyTorch 2.7.1+cu118/Python 3.11; el entorno local evaluado usa {hardware['torch']}. No se trasladó el detector al teléfono ni se cambió automáticamente de arquitectura.

El historial guardado tiene precision 0.68426, recall 0.55655, mAP50 0.58750 y mAP50–95 0.35306 en época 50, y termina en 75. Es evidencia histórica; no son métricas de la APK instalada ni «68 % de exactitud».

## 3. Distribución original

5.104 imágenes: 3.572 train, 1.020 val, 512 test. Una caja no equivale a una imagen.

{chr(10).join(original)}

Solo dieciséis imágenes train contienen escaleras y no existe `trash_bin`. Es una limitación de diversidad que la resolución del S25 no puede corregir por sí sola.

## 4. Fuentes y licencias

Se verificaron en Open Images las categorías Person (`/m/01g317`), Stairs (`/m/01lynh`) y Waste container (`/m/0bjyj5`), además de Flowerpot para candidatos a negativos difíciles. El descargador exige metadatos de licencia individual CC-BY-2.0 y conserva autor, título, enlaces y hashes; anotaciones CC-BY-4.0. Open Images no garantiza la licencia de cada imagen: revisar procedencia antes de liberar. Fuentes: [descargas oficiales](https://storage.googleapis.com/openimages/web/download_v7.html) y [condiciones y atribución](https://storage.googleapis.com/openimages/web/factsfigures_v7.html).

COCO ya participa en el dataset original; se conserva su descargador y se implementó conversión con mapeo explícito. No se descargó otro lote COCO al cubrir la meta de personas con Open Images. El proyecto público [Stair Escalator](https://universe.roboflow.com/stairsescalators/stair_escalator), que lista CC-BY-4.0, queda como fuente opcional pendiente de exportación autorizada y revisión de etiquetas. No se eludieron credenciales. No hay imágenes nuevas del campus ni del Samsung.

## 5. Imágenes realmente incorporadas

Registros descargados: {source_counts}. Total: **{inventory['new_records']}**; SHA256 distintos: **{inventory['new_unique_sha256']}**; coincidencias exactas con el dataset original: **{inventory['new_exact_overlap_legacy']}**. Después de excluir esas coincidencias quedan **{inventory['new_unique_sha256_excluding_legacy']} imágenes nuevas por SHA256**. Las imágenes parecidas pueden representar una misma escena: todavía deben revisarse los grupos visuales.

| Categoría | Meta orientativa | Nuevas candidatas por SHA256 | Falta para recopilar la meta | Aprobadas para entrenamiento |
| --- | ---: | ---: | ---: | ---: |
| Personas | 1500 | {counts.get('person',0)} | {max(0,1500-counts.get('person',0))} | 0 |
| Escaleras | 1200 | {counts.get('stairs',0)} | {max(0,1200-counts.get('stairs',0))} | 0 |
| Tachos / Waste container pendiente de definición | 1000 | {counts.get('trash_bin',0)} | {max(0,1000-counts.get('trash_bin',0))} | 0 |
| Candidatos a negativos con macetas | 500 | {inventory['candidate_negatives_unique']} | {max(0,500-inventory['candidate_negatives_unique'])} | 0 |

Las categorías se superponen; no se suman como imágenes independientes. Los {inventory['candidate_negatives_unique']} candidatos con macetas **no son negativos verificados**. «Incorporadas» significa descargadas e indexadas en la carpeta de mejora, no añadidas al entrenamiento operativo. La cuota de anotaciones revisadas sigue pendiente. Evidencia: [inventario](magicvision_ml_improvement/reports/dataset_inventory.json).

## 6. Distribución resultante de la recopilación

{chr(10).join(table)}

Las cajas nuevas cuentan registros importados, incluso aquellos que pueden duplicar una imagen de otra fuente. Las columnas no describen un split final aprobado. `trash_bin` conserva ID 10; los IDs 0–9 permanecen iguales. [CSV completo](magicvision_ml_improvement/reports/dataset_distribution.csv).

## 7. Calidad y cobertura de las anotaciones

El control sobre {quality['images']} registros encontró {inventory['quality_errors']} errores de archivo/geometría, {len(quality['duplicate_annotations'])} registros con cajas repetidas, {inventory['conflicting_exact_duplicates']} conflictos de etiquetas sobre duplicados exactos y {inventory['duplicate_or_session_clusters']} grupos por coincidencia visual, exacta o sesión. Son controles automáticos, no certificación semántica.

Se inspeccionó visualmente una muestra de ocho imágenes de escaleras/tachos, sin aprobarlas como ground truth; [observaciones](magicvision_ml_improvement/reports/source_sample_review.md). La [galería final](magicvision_ml_improvement/reports/quality_final/index.html) permite examinar cajas por categorías prioritarias, y `quality.json` contiene la cola completa. Se ejecutó preanotación real de dieciséis imágenes con el modelo anterior: cero aprobaciones automáticas, sin capacidad de preanotar `trash_bin`.

**{inventory['approved_new']} imágenes nuevas aprobadas**, cero negativos verificados. Las originales también requieren comprobar objetos de la clase nueva antes de entrenar con once categorías. La revisión exige cobertura de todas las clases, cajas consistentes, responsable, fecha y procedencia. Las categorías desconocidas nunca se convierten indiscriminadamente a `obstacle`.

## 8. División sin contaminación

Se implementó agrupación transitiva por sesión/recorrido, SHA256 y pHash con distancia ≤ 4, seguida de asignación orientativa 70/20/10 con balance de clases. Se respetan particiones históricas; los grupos que cruzan holdouts se rechazan hasta revisión. Se encontraron {inventory['legacy_cross_split_clusters']} grupos candidatos que cruzan particiones originales: requieren inspección; la similitud visual no prueba por sí sola que sean duplicados.

La separación se ejecutó y produjo `training_ready: {str(split['training_ready']).lower()}` en `datasets/magicvision_v2/split_report.json`. No libera datos incompletos ni crea una división ficticia. La independencia por sesión de las fuentes públicas es desconocida; deduplicar no sustituye reservar recorridos diferentes del campus. El test original no se usó para elegir parámetros ni se ejecutó como evaluación final.

## 9. Configuración de entrenamiento

Preparados E1 (640), E2 (960) y E3 (refinamiento del ganador de validación). Hasta 100 épocas, paciencia 20, AdamW, semilla 42, batch 4 a 640 y 2 a 960, AMP y aumentos moderados. Geometría con actualización de cajas; sin inversión vertical. El módulo fotométrico adicional aplica brillo/contraste/desenfoque/JPEG solo a train y conserva grupos. Se probó en fixtures, no sobre datos reales sin revisar.

Hardware detectado: {hardware['gpu']}, {hardware['vram_bytes']/1024**3:.2f} GiB VRAM, CUDA disponible. El chequeo de 960 exige 8 GiB nominales y aún requiere probar el batch real; no se promete que quepa todo entrenamiento. La transferencia ejecutada copió **{len(transfer['transferred_keys'])} tensores** e inicializó **{len(transfer['initialized_keys'])}** incompatibles por tamaño al pasar a once clases. [Detalle de transferencia](magicvision_ml_improvement/reports/transfer_probe.json).

El intento controlado de iniciar entrenamiento fue rechazado por el requisito de dataset revisado. No se ejecutaron épocas E1/E2/E3 ni se produjo un checkpoint nuevo. La causa no fue falta de GPU, sino anotaciones completas y verificadas insuficientes. El mínimo piloto de cincuenta escenas distintas de escaleras y tachos no es un objetivo de calidad ni permite contar aumentos como imágenes nuevas.

## 10. Experimentos ejecutados

Tres ejecuciones reales sobre las mismas 1.020 imágenes val y el mismo checkpoint; 3.060 inferencias de evaluación más calentamientos. Métricas de esta tabla a confianza 0.30:

{chr(10).join(evaluation)}

`E0_960` usa los pesos anteriores con otro tamaño de inferencia; **no es E2 entrenado**. `E0_640_appjpeg` aproxima en PIL el escalado a 640/JPEG80; no utiliza capturas reales del teléfono. El original es la imagen del dataset tal como se guarda, no necesariamente un archivo original de cámara.

Las predicciones se obtuvieron desde confianza 0.001, lo que permite comparar umbrales inferiores a 0.30 sin intentar recuperar cajas descartadas desde Flutter. Correspondencia por clase e IoU ≥ 0.50, una predicción por anotación, orden de confianza descendente; duplicados son FP. AP interpolada en 101 puntos e IoU 0.50:0.05:0.95. No se presenta como reproducción exacta del evaluador histórico. [Protocolo y métricas de validación de Ultralytics](https://docs.ultralytics.com/modes/val/).

## 11. Comparación y selección de checkpoint

No hay candidato entrenado para comparar contra E0. Aumentar inferencia a 960 con estos pesos no mejoró el recall de personas/escaleras ni el mAP global en esta validación. La reducción/JPEG aproximados disminuyeron recall de personas de 0.6064 a 0.5454: indicio medido de pérdida en procesamiento, pendiente de confirmar con los JPEG exactos del S25.

El checkpoint anterior no tiene clase tacho. Ceros o ausencia de soporte en esa categoría no evalúan un detector de tachos; tampoco se equipara automáticamente `obstacle` con `trash_bin`. Se conserva el checkpoint original. Metas de ingeniería 95 % recall personas/escaleras y 90 % tachos no están demostradas. Diez cajas val de escaleras no bastan para generalizar al campus.

## 12. Resolución y umbrales

Se mantienen operativamente **640 / JPEG80 / confianza 0.30**, al no contar con una validación del campus que justifique una sustitución. El soporte técnico permite probar 960 y verticales mayores de forma coherente, después de verificar el servidor. Comparación medida:

{chr(10).join(sweeps)}

No se seleccionó el umbral que genera más cajas. [Comparación por las once categorías](magicvision_ml_improvement/reports/comparison.csv) y [lectura de resultados](magicvision_ml_improvement/reports/comparison.md). El parámetro IoU 0.65 se registra, pero YOLO26 tiene salida end-to-end sin NMS convencional; no se atribuye a ese valor una mejora no medida. Faltan escenas verificadas de multitudes y metadatos de subgrupos del campus.

## 13. Latencia y falsas alarmas

Los tiempos de la tabla son locales del predictor con GPU e incluyen su preprocesamiento. Hubo trabajo concurrente de CPU/disco: no usar las diferencias como benchmark aislado, ni afirmar que 960 es más rápido que 640 en AWS. Se registraron tamaño de modelo, memoria CUDA, RSS del proceso, media/p95 y bytes por imagen en cada `run.json`.

La prueba local de integración completó {integration['invocations']} invocaciones reales y dejó {integration['temporary_images_remaining']} imágenes temporales. No mide transferencia AWS real ni costos productivos. La app registra tiempo HTTP, tiempo comunicado por servidor y total para medirlos en el teléfono. Los límites aceptables de FP/imagen, avisos molestos por recorrido y latencia total deben fijarse con sesiones reales; los valores de validación están disponibles sin inventar tolerancias ya aprobadas.

## 14. Flutter, CERM, API y generador

- Contrato de once IDs, correspondencia ID/nombre estricta, `model_version` SHA256 opcional para compatibilidad, dimensiones y cajas normalizadas validadas. El empaquetador rechaza pesos de diez clases con etiquetas de once.
- Se eliminaron filtros adicionales 0.65/0.55 del resumen que omitían detecciones ya aceptadas por la API. Se representan hasta once tipos. Una detección no se convierte por ello en certeza ni en peligro inmediato.
- CERM comunica objetos y su posición incluso en riesgo bajo. Sin detecciones informa incertidumbre; no afirma que el camino está libre. El aviso prioritario se elige por el objeto que activa el riesgo. `trash_bin` se interpreta espacialmente como obstáculo, con soporte nominal de voz; sin pesos entrenados todavía no habrá detecciones reales de esa clase.
- Se anuncia presencia de escalera sin inferir subida/bajada a partir de la clase. Las reglas usan posiciones/proporciones de caja como heurísticas, no distancia física medida ni movimiento demostrado.
- YOLO y VLM comparten el mismo JPEG procesado, ID, instante y resultado asociado. El límite existente de antigüedad de captura es 15 s, configurable con máximo 30 s; comienza al capturar. Se invalida al cambiar captura, desconectar o superar la antigüedad, y se vuelve a comprobar antes de entregar voz. Una generación lenta puede descartarse: medir en el S25 antes de ajustar ese límite.
- Se registran detecciones, CERM, prompt, texto visual y mensaje final generado con el mismo ID; «generado» no acredita reproducción audible. El VLM real no se ejecutó en el teléfono en esta sesión.
- Archivo de investigación optativo con consentimiento explícito, sesión/recorrido anónimos y límite de cien capturas. Conserva píxeles originales y JPEG enviado; elimina metadatos privados del original y EXIF del JPEG enviado, y guarda orientación por separado. Distingue dimensiones almacenadas de las orientadas. Una prueba detectó que el decodificador JPEG elimina la etiqueta de orientación al aplicarla; se corrigió leyendo EXIF antes de decodificar. Está desactivado por defecto.
- API/servidor y configuración de captura admiten hasta 1280×1280, mantienen JPEG, máximo de 2 MiB y controles existentes. La app verifica que las dimensiones de respuesta coincidan con la captura enviada. Se verifica orientación EXIF y referencia geométrica. El valor por defecto de la app sigue en 640. No se cambió la fuente de cámara ni la inferencia en nube.

Los cambios en fuentes respaldadas están identificados por hashes en [changed_files.json](magicvision_ml_improvement/reports/changed_files.json). También se añadieron archivo de capturas, pruebas, herramientas de datos/evaluación, y se extendieron empaquetado e integración local.

## 15. Pruebas ejecutadas

| Verificación | Resultado | Alcance |
| --- | --- | --- |
| Pipeline Python | {test_counts['pipeline']} aprobadas | Geometría, conversión, duplicados, particiones, correspondencia, subgrupos y aumentos solo train |
| Backend | {test_counts['backend']} aprobadas | API sync/async, errores, límites, once IDs, dimensiones y EXIF |
| Flutter | {flutter_passed} aprobadas, 1 omitida en suite normal | Parsers, CERM, mensajes, geometría, capturas y sincronización; la integración requiere activación explícita |
| `flutter analyze --no-pub` | Sin incidencias | Análisis estático de la app final |
| Integración activada aparte | Aprobada | Checkpoint real en CUDA → servidor actual → HTTPS/FastAPI → Flutter → CERM |
| Preanotación | 16 imágenes procesadas | Propuestas reales, pendientes de revisión |
| Evaluación E0 | 3 × 1.020 imágenes | Pesos reales, resoluciones/compresión y cuatro umbrales |

Evidencias en `reports/pipeline_tests.xml`, `backend_tests.xml`, `flutter_tests.log`, `flutter_analyze.log` e [integración final](magicvision_ml_improvement/reports/local_integration_final/summary.json). La integración usa transporte/almacenamiento AWS simulados y una imagen del dataset; no acredita AWS, cámara física, VLM real ni TTS audible. Los fixtures unitarios no son evidencia de precisión visual.

## 16. Preparación de despliegue y rollback

Se preparó localmente `magicvision_ml_improvement/models/baseline_704361e5_640.tar.gz`, SHA256 `{sha(artifact)}`, con checkpoint anterior, diez etiquetas y configuración para conservar/recuperar el modelo. No hay artefacto nuevo validado de once clases.

Antes de un despliegue autorizado: identificar endpoint e infraestructura reales; conservar referencia al artefacto, versiones de Lambda/contenedor/configuración; evaluar el candidato en un entorno aislado, contrato, autenticación, límites, latencia y costo con evidencia real. Actualizar y verificar validadores/servidor antes de aumentar imágenes desde Flutter. El empaquetador comprueba etiquetas contra pesos y no sobrescribe artefactos existentes.

Rollback: restaurar las referencias/versiones de servicio anteriores y parámetros de imagen compatibles, además del checkpoint previo. Los 33 respaldos son fuentes de referencia para revertir esos archivos; conservar cualquier trabajo posterior antes de restaurar. El artefacto del modelo por sí solo no revierte un backend modificado. Procedimiento y comandos en [README](magicvision_ml_improvement/README.md). **No se modificó AWS productivo ni se publicó una APK.**

## 17. Limitaciones y continuación necesaria

1. Identificar APK, cámara, endpoint y hash de modelo que usa el Samsung. No se puede probar una mejora en ese teléfono sin capturas ni conexión.
2. Revisar la cola de anotaciones, licencias/procedencia, duplicados y cobertura de once clases. Resolver los conflictos; confirmar que Waste container corresponde al alcance definido de tacho. Las cuotas descargadas no sustituyen esta revisión.
3. Incorporar recorridos reales y reservar sesiones distintas para validación y prueba. Completar subgrupos verificados (personas cerca/lejos/ocultas, escaleras por perspectiva, tachos por tamaño, varios objetos). [Informe de subgrupos](magicvision_ml_improvement/reports/campus_subgroups.json) los marca pendientes; no inventa distancias.
4. Liberar un dataset revisado; ejecutar E1/E2 y posteriormente E3 si la validación lo justifica. Comparar todas las clases, recall, FP/imagen, fallos por recorrido y latencia. No sustituir el modelo si hay degradaciones inaceptables.
5. Validar el flujo completo en S25 y AWS de pruebas, incluyendo descripción y voz reales. Solo después preparar el reemplazo productivo para aprobación.

El [diagnóstico de detección](MAGICVISION_DIAGNOSTICO_DETECCION.md) explica las limitaciones del reconocimiento. El catálogo de escenarios se conserva en `MAGICVISION_INPUTS_OUTPUTS_CAMPUS.xlsx`; sus ejemplos no deben tratarse como resultados medidos de la aplicación. Este informe distingue la implementación verificada de los resultados que siguen pendientes. La guía oficial de [pruebas del modelo](https://docs.ultralytics.com/guides/model-testing/) fundamenta evaluar escenas representativas antes de atribuir mejoras al uso real.
'''
    (ROOT.parent/'MAGICVISION_MEJORA_MODELO_REPORTE.md').write_text(text,encoding='utf-8')
    print('Report written; original checkpoint hash verified; production untouched.')


if __name__=='__main__':main()
