# Revisar la ampliación y preparar el entrenamiento

Los datos están en `datasets/expanded_v3/manifest.jsonl`. El paquete inicial está en `datasets/review_v3_batch01/index.html`: abrirlo con Edge o Chrome. Incluye un muestreo equilibrado de las once clases; no sustituye la revisión del resto de imágenes. Los originales permanecen intactos.

## Corregir una imagen

1. Abrir la imagen completa y comprobar **todas las once clases**, no solo la clase que motivó su descarga. El ID de cada registro aparece encima de la imagen.
2. En el editor, elegir clase y arrastrar para dibujar una caja. Cambiar una clase con el selector de su caja; para moverla, eliminarla y dibujarla de nuevo.
3. Pulsar **Guardar etiquetas corregidas** y reemplazar el archivo indicado dentro de `labels/` del paquete. Si el navegador descarga el TXT, moverlo a ese directorio conservando el nombre: guardar en Descargas no modifica el paquete.
4. Ejecutar `refresh` y volver a abrir la página para comprobar las cajas guardadas. Cerrar/reabrir sin guardar pierde los cambios del navegador.
5. Registrar una decisión por imagen después de revisar geometría, objetos omitidos y procedencia. Si hay dudas, usar `defer`; si no sirve, `reject`.

Comandos PowerShell desde `D:\Escritorio\Tesis`:

```powershell
python magicvision_ml_improvement/scripts/review_dataset.py refresh --package magicvision_ml_improvement/datasets/review_v3_batch01
# Reemplazar ID_REAL, NOMBRE y la nota por los de la revisión realizada.
python magicvision_ml_improvement/scripts/review_dataset.py decide --package magicvision_ml_improvement/datasets/review_v3_batch01 --id ID_REAL --action approve --reviewer NOMBRE --note "Revisé las once clases, corregí las cajas y comprobé la fuente" --complete --provenance-reviewed
```

No ejecutar el ejemplo como una aprobación masiva. La decisión queda vinculada al SHA256 de imagen y etiquetas. Si se edita después, repetir `refresh` y registrar la decisión de nuevo. Una etiqueta vacía requiere además `--verified-negative`, solo después de verificar que no hay ninguno de los once tipos de objeto. El editor no interpreta el silencio como aprobación.

Usar el criterio de [annotation_policy.md](annotation_policy.md). `trash_bin` es ID 10; no duplicar la misma caja como `obstacle`. `stairs` cubre el tramo visible según la política, no cada peldaño como objeto independiente. D40 de RDD2022 se importó como `pothole`; las grietas no se convirtieron en huecos. Los postes de servicios públicos requieren verificar el límite del poste respecto a cables, transformadores y torres. Open Images puede tener cajas jerárquicas superpuestas de Person/Man/Woman/Boy/Girl: conservar una caja coherente por persona.

La procedencia disponible está junto a cada imagen. La licencia del dataset se conserva; eso no garantiza por sí mismo la trazabilidad individual de todos los ejemplos. Si falta información necesaria, dejar la imagen pendiente o rechazarla. Las sugerencias automáticas en `reports/annotation_gaps_v3.json` son ayudas y pueden equivocarse.

## Aplicar decisiones y continuar por lotes

```powershell
python magicvision_ml_improvement/scripts/review_dataset.py apply --package magicvision_ml_improvement/datasets/review_v3_batch01 --output magicvision_ml_improvement/datasets/reviewed_v3_batch01
python magicvision_ml_improvement/scripts/review_dataset.py export --manifest magicvision_ml_improvement/datasets/reviewed_v3_batch01/manifest.jsonl --output magicvision_ml_improvement/datasets/review_v3_batch02 --limit 275 --selection balanced
```

Cada lote siguiente debe partir del **manifiesto aplicado anterior**, para acumular correcciones. No concatenar manifiestos completos de varios lotes ni borrar filas pendientes: contienen grupos y particiones históricas necesarios para detectar fugas. Conservar `decisions_audit.jsonl`. Los grupos cruzados y conflictos de cajas para copias idénticas requieren reconciliación documentada; no mover un holdout original a entrenamiento.

`balanced` alterna clases y evita hashes repetidos dentro del lote; no balancea automáticamente el conjunto final ni confirma etiquetas. `--classes stairs trash_bin` limita el muestreo a imágenes con esas clases anotadas; aun así hay que comprobar todas las clases visibles. `rare-first` prioriza las menos representadas.

## Liberar una versión y entrenar

Sustituir `reviewed_v3_batch01` por el último lote aplicado. Usar directorios nuevos para cada versión:

```powershell
python magicvision_ml_improvement/scripts/data_pipeline.py quality --manifest magicvision_ml_improvement/datasets/reviewed_v3_batch01/manifest.jsonl --output magicvision_ml_improvement/reports/reviewed_v3_quality
python magicvision_ml_improvement/scripts/data_pipeline.py split --manifest magicvision_ml_improvement/datasets/reviewed_v3_batch01/manifest.jsonl --output magicvision_ml_improvement/datasets/magicvision_v3_released
python magicvision_ml_improvement/training/preflight.py --data magicvision_ml_improvement/datasets/magicvision_v3_released/data.yaml --output magicvision_ml_improvement/reports/v3_preflight.json
# Ejecutar solo cuando preflight termine con ready=true / código 0.
python magicvision_ml_improvement/training/experiments.py train --experiment E1 --data magicvision_ml_improvement/datasets/magicvision_v3_released/data.yaml
python magicvision_ml_improvement/training/experiments.py train --experiment E2 --data magicvision_ml_improvement/datasets/magicvision_v3_released/data.yaml
```

El control compara los archivos con su revisión, verifica que no haya imágenes extra en las carpetas, exige las once clases en train/val/test y al menos cincuenta escenas distintas de escaleras y tachos en train. Ese mínimo es una condición piloto del proyecto, no un requisito de YOLO ni garantía de rendimiento. Una revisión parcial puede no alcanzar las cantidades y la diversidad necesarias.

Elegir modelo y umbrales con validación. Mantener test sin usar hasta la comparación final y añadir recorridos reservados del S25 en la universidad. Reportar precision y recall por clase, falsos positivos, mAP y latencia. No publicar un porcentaje nuevo hasta medirlo. El checkpoint y el despliegue actuales se conservan.
