# Reporte de ejecución: Detection + Depth V1

Fecha: 22/09/2026. Evidencia local; no certifica la APK instalada ni un despliegue AWS.

## Completado

- Auditoría de los 24.773 registros: archivos, etiquetas, hashes, píxeles, dimensiones,
  IDs, duplicados/pHash y agrupaciones. Inventario: 24.770 imágenes únicas, 82.541 cajas/propuestas.
- Contrato canónico de 22 clases compartido por dataset, backend y Flutter; ID 10 trash_can,
  IDs finales 18/19/20/21 correctos; exclusión de cables, derrames y ascensores conservada.
- Revisión geométrica y muestra visual de puertas; editor nuevo de 100 imágenes sin aprobaciones.
- Descarga de YOLO26m Detection base y YOLO26m Depth oficiales, con hashes registrados.
- Ejecución real de Depth en GPU, fusión por ROI, validación de mapa e identidad de captura.
- Implementación y pruebas de riesgo, seguimiento, suavizado, cooldown y priorización.
- API v2 síncrona/asíncrona compatible con v1, identidad de frame y hash verificados.
- Flutter acepta campus22 y Depth; CERM comparte política; modo caminar optativo con botón,
  parada por ciclo de vida/desconexión/OCR, una solicitud en vuelo y descarte por caducidad.
- Herramientas de entrenamiento con preflight/resume/perfilador y evaluación comparativa;
  script de calibración con sesiones reservadas y métricas, sin perfiles aprobados ficticios.
- Benchmark real local secuencial/concurrente de seis imágenes con el detector anterior.

## Parcial

La integración completa está implementada y probada con fixtures en API/Flutter, pero no puede
probarse con un detector campus22 real inexistente. El benchmark usa el checkpoint anterior de
diez clases y no mide calidad ni red. Rolling median y EMA están implementados y comprobados
numéricamente, sin comparación ganadora sobre recorridos reales. El protocolo físico y el CSV
están preparados; no hay capturas nuevas, distancias medidas ni resultados de batería.
El filtro lingüístico del VLM reduce afirmaciones no sustentadas, pero no demuestra ausencia de
alucinaciones. Las alertas de distancia dependen del canal estructurado.

## Bloqueado: evidencia y siguiente acción

| Elemento | Motivo y archivo de evidencia | Siguiente acción concreta |
|---|---|---|
| Release y entrenamiento campus22 | 0 imágenes aprobadas; 541 registros sin cajas; 285 grupos con particiones de origen distintas. `campus22_preflight_final.json`, `campus22_training_preflight.json` | Corregir cajas/cobertura y procedencia en el editor, aplicar decisiones y resolver grupos; crear un release nuevo y volver a preflight |
| Batch máximo y entrenamiento 640/768/960 | `campus22_batch_probe.json`: performed=false por release sin revisión; batch=4 es configuración inicial, no máximo medido | Ejecutar perfilador sobre release aprobado y medir primeras iteraciones; comparar resoluciones con mismo holdout |
| Nuevo best.pt, mAP y recall | No se iniciaron épocas: gate ready=false antes de GPU. `campus22_training_preflight.log` | Entrenar al aprobar release; guardar best/last y evaluar todas las clases, especialmente puertas/escaleras |
| Metros fiables y error de Depth | No hay medidas reales ni perfiles validados; plantilla vacía en `datasets/real_world_test/measurements_template.csv` | Medir objetos con cámara/lente real, separar sesiones de calibración/validación y aprobar perfil contra criterios documentados |
| Prueba final en el S25/USB | ADB no mostró dispositivos conectados; no se inspeccionó APK actual | Conectar teléfono, instalar una compilación identificada y ejecutar el protocolo físico con acompañamiento |
| AWS real | No hay checkpoint campus22 para alojar; no se reemplazó el endpoint | Tras evaluación, empaquetar detector22 + Depth y perfil, verificar imagen/driver y probar endpoint antes de adoptarlo |

## Respuestas a las 27 preguntas

1. **¿Dataset listo?** No. READY_FOR_TRAINING=false.
2. **¿Cuántas imágenes?** 24.770 únicas; 24.773 registros.
3. **¿Cuántas cajas?** 82.541 cajas/propuestas, no instancias humanas confirmadas.
4. **¿División train/val/test?** No hay split final liberado. El intento canónico
   `datasets/campus22_release_depth_v1` tiene cero imágenes en cada partición.
5. **¿Leakage resuelto?** No. Hay 285 grupos con particiones de origen incompatibles;
   no se ocultaron mediante una división aleatoria.
6. **¿Se entrenó YOLO26m?** No en esta mejora. Se ejecutó el comando y el preflight lo detuvo.
7. **¿Cuál es best.pt?** No existe un best.pt campus22. El anterior está en
   `01_app_magicvision_final/app_flutter_magicvision/runs/detect/modelo_final_yolo26m_80e_640_b8/weights/best.pt`;
   tiene diez clases y SHA256 `704361e5746e23160b95976d342484d250a7931fabbbe798dd1d4e48d269367f`.
8. **¿mAP nuevo?** No medido. No se atribuyen métricas históricas al nuevo contrato.
9. **¿Recall por clase nuevo?** No medido; evaluador preparado para las 22 clases.
10. **¿Recall stairs?** No medido para el nuevo detector.
11. **¿Recall door?** No medido; el anterior no tiene esta clase.
12. **¿Recall glass_door?** No medido; el anterior no tiene esta clase.
13. **¿Recall pothole?** No medido para el nuevo detector.
14. **¿Recall curb_step?** No medido; el anterior no tiene esta clase.
15. **¿Se integró Depth?** Sí en código y prueba GPU real; API/Flutter probados con fixtures.
    Falta la prueba completa con detector22 entrenado y cámara real.
16. **¿Se entrenó Depth?** No. Se utilizó el checkpoint oficial preentrenado.
17. **¿Cómo se calcula distancia?** ROI por clase, filtro de inválidos, recorte de extremos/MAD,
    p25 inicial y p50 registrado, corrección log-afín únicamente con perfil válido y suavizado temporal.
18. **¿Error métrico?** Desconocido: no hay ground truth físico; MAE/RMSE no se inventan.
19. **¿Tracking?** Asociación IoU por clase, IDs por sesión, histories de caja/confianza/distancia,
    timeout y descarte de frames fuera de orden; mediana o EMA. El servidor no comparte trackers entre usuarios.
20. **¿Cooldown?** Siete segundos iniciales por track/clase; aumentos de riesgo y reducción de
    distancia pueden anticipar aviso; máximo un crítico y un secundario por ciclo.
21. **¿Tiempo Detection?** Promedios medidos en tabla inferior, exclusivamente detector anterior.
22. **¿Tiempo Depth?** Promedios medidos en tabla inferior.
23. **¿Tiempo Fusion?** Promedios medidos en tabla inferior.
24. **¿Total?** 124,10 ms secuencial y 96,21 ms concurrente en esta muestra local.
25. **¿FPS posible?** Recíprocos de procesamiento 8,06/10,39; no son FPS end-to-end del celular.
26. **¿Qué falta?** Revisión/release, entrenamiento/evaluación, calibración, prueba física y validación de despliegue.
27. **¿Limitaciones?** Etiquetado incompleto, objetos no soportados por modelo anterior, vidrio/fondo,
    escala monocular, oclusiones/cambios de cámara, asociación IoU, latencia/red/batería y VLM generativo.

## Benchmark medido

Seis imágenes reales del inventario, detector anterior de diez clases + Depth m, GPU RTX 2070
Max-Q, PyTorch 2.11.0+cu128, Ultralytics 8.4.159. Sin HTTP, calibración ni referencia física.
Los tiempos concurrentes individuales incluyen contención/esperas; su suma no es el tiempo total.
La memoria es el máximo asignado/reservado por PyTorch, no toda la memoria física del proceso.

| Ejecución | Detection ms | Depth ms | Fusion ms | Total ms | Recíproco FPS | GPU asignada MiB | GPU reservada MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| sequential | 79.07 | 32.50 | 6.56 | 124.10 | 8.06 | 282.74 | 310.00 |
| concurrent | 89.12 | 73.74 | 6.40 | 96.21 | 10.39 | 388.91 | 422.00 |

No se activó concurrencia en producción. Son seis muestras diagnósticas; no un benchmark
estadístico definitivo ni una medición del nuevo detector campus22.

## Pruebas de código

Python: 203 pruebas, 0 fallos, 0 errores. XML: `detection_depth_python.xml`.

Flutter: `No final result`. Log completo: `detection_depth_flutter.log`.

Las fixtures sintéticas verifican lógica y contratos: no se agregaron al dataset ni son evidencia
de precisión. Se conservaron las pruebas previas; se actualizaron los contratos de ID 10 y los
mensajes que antes esperaban metros derivados de geometría. Los mensajes ahora siguen la política
central y una caja pequeña lateral puede recibir riesgo bajo en vez del aviso medio genérico anterior.
Las pruebas de OCR, imágenes, captura, cancelación, HTTP asíncrono y renderizado se ejecutan juntas.

## Impacto de nube

Modo caminar desactivado inicialmente. Objetivos 1/2/3/5 análisis por segundo equivalen a
60/120/180/300 solicitudes de inferencia por minuto si se alcanzan, más polling/S3 en modo async.
La preparación del endpoint se reutiliza por sesión. No se midió batería/red/coste monetario,
no se consultaron precios y no se incrementó uso real de AWS. No se enviaron imágenes a terceros.

## Evidencia y documentación

- `campus22_preflight_final.json/.md`: auditoría completa.
- `door_glass_door_quality.md`, `door_glass_door_visual_observations.md`: puertas.
- `detection_depth_benchmark.json`: todas las muestras y tiempos reales.
- `yolo26_official_models.json`: procedencia y hashes de modelos descargados.
- `campus22_training_preflight.json`, `campus22_batch_probe.json`: gates ejecutados.
- `detection_depth_python.xml`, `detection_depth_flutter.log`, `detection_depth_flutter_analyze.log`: pruebas.
- `detection_depth_docker_build.log`: intento de construcción, revisar resultado final.
- [Funcionamiento de la aplicación](../../01_app_magicvision_final/app_flutter_magicvision/ARQUITECTURA_APLICACION.md) y [comandos de voz](../../01_app_magicvision_final/app_flutter_magicvision/GUIA_COMANDOS.md).

No hay base experimental todavía para afirmar que la precisión será muy superior o que la
distancia monocular es fiable para navegar por el campus.


## Verificación de continuación

Los resultados siguientes corresponden a la nueva ejecución local.

- Python: 204 pruebas; 0 fallos; 0 errores. `continuation_python_tests.xml`.
- `continuation_flutter_verified.log`: 00:15 +69 ~1: All tests passed!
- `continuation_flutter_analyze.log`: No issues found! (ran in 137.7s)
- Auditoría: 24773 registros, 0 errores de archivos/etiquetas, 0 errores de píxeles/hash, 0 aprobaciones completas y 285 grupos con particiones incompatibles. `campus22_preflight_final.json`.
- Depth ejecutado: 2026-09-23T04:18:30.658060+00:00; NVIDIA GeForce RTX 2070 with Max-Q Design; mapa [426, 640]; fracción finita y positiva 100.00%. Una imagen real, sin entrenamiento ni calibración. `continuation_depth_smoke.json` y `scripts/smoke_depth.py` (desde la raíz del módulo).
- Se corrigió el lanzador para devolver bloqueo si el perfilado de batch rechaza el entrenamiento; prueba de regresión incluida.
- Entrenamiento campus22 pendiente: `campus22_training_preflight.json` conserva los motivos concretos. No hay nuevo best.pt ni métricas del detector.
- Revisar las cajas de `datasets/review_campus22_doors_v2/index.html`, registrar decisiones reales y aplicar el manifiesto revisado antes de crear un release nuevo.
