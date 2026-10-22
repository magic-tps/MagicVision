# Análisis del repositorio y plan de migración

## Fase 1: inventario previo

El workspace no contiene `.git` ni instrucciones AGENTS.md. Se creó `../../respaldo_pre_migracion_cloud` con el código/configuración anterior. No se copiaron ni borraron datasets, pesos de entrenamiento o assets del VLM.

| área | Implementación encontrada | Intervención |
|---|---|---|
| App Flutter | `lib/main.dart`, unas 8 mil líneas antes de migrar | Sustituir únicamente captura/inferencia y enlaces de voz/preview |
| YOLO por foto | `lib/core/services/object_detection_service.dart` | Reemplazar intérpretes, tensores, letterbox, parser y NMS por interfaz |
| YOLO continuo | `YoloDetector`, `_processFrame` y `_dangerDetector` en `main.dart` | Retirar inferencia y bucle; conservar entidades/funciones de HUD |
| Asset YOLO | `assets/models/magicvision_detector.tflite` | Trasladar a `ml/legacy_exports`; quitar de pubspec |
| Exportación | `ml/scripts/export_tflite.py`, notebook y guía de entrenamiento | Exportador en respaldo; notebook empaqueta `.pt` para S3 |
| Cámara | `CameraController`, `availableCameras`, preview y `takePicture` en `main.dart` | Extraer fuentes; UVC por defecto; conservar interna como opción explícita |
| Android | MainActivity vacía, sin USB Host | Puente USB separado, manifest, filtro y dependencia UVCAndroid |
| CERM | `risk_assessment_service.dart`, `detection_result.dart`, `analysis_result.dart` | Mapper de entrada; sin cambios de reglas; nuevo nombre de fuente |
| USCE | No existe símbolo/módulo con ese nombre | Preservar `MobileVlmClient`, prompts y funciones de contexto semántico |
| OCR | `TextRecognizer`, `_captureAndReadText` en `main.dart` | Mantener ML Kit, adaptar captura y borrar temporal |
| Voz | `_setupVoiceOutput`, `_setupSpeechCommands`, `_listenForCommands`, `_speak` | Mantener STT/TTS; consultar disponibilidad de audio USB |
| GPS/SOS/contactos/historial | Geolocator, Geocoding, SharedPreferences, URL Launcher en `main.dart` | Conservar lógica; historial recibe AnalysisResult cloud |
| Backend | No existía | API multipart, contrato validado, llamada IAM, servidor YOLO y plantilla AWS |
| Pruebas | `test/widget_test.dart`, 24 pruebas antes | Sustituir comprobación de asset retirado; añadir pruebas de migración |
| Documentación | README, arquitectura, comandos, ML, LEEME raíz | Actualizar flujo y separar evidencia histórica |
| Tesis | `02_paper_metricas_modelo_final`, Excel CERM y generador | Conservar evidencia histórica; advertir alcance anterior |

Dependencias conservadas: `camera`, `image`, `http`, `permission_handler`, `google_mlkit_text_recognition`, `speech_to_text`, `flutter_tts`, `geolocator`, `geocoding`, `shared_preferences`, `url_launcher`, `path_provider`, `share_plus`, `cactus`. Se elimina `tflite_flutter`; `http_parser` y `uuid` pasan a dependencias directas.

## Fase 2: plan A-G

A. **Cámara USB**: `CameraSource`, `CameraFrame`, `UsbCameraSource`, `InternalCameraSource`, `UsbCameraBridge.kt`, filtros USB, textura, permisos, captura discreta y ciclo de vida.

B. **Inferencia AWS**: `InferenceImageConfig`, modelos `BoundingBox`/`Detection`/`DetectionResponse`, interfaz `ObjectDetectionService`, `AwsObjectDetectionService`, procesador JPEG en isolate y coordinador `VisionAnalysisService`. API en `backend/lambda`, servidor en `backend/sagemaker`, empaquetado `.pt` y plantilla SAM.

C. **Adaptador CERM**: `CermDetectionMapper` construye el `CermInput` compatible con `assess(List<DetectionResult>)`. Mantener duplicados, coordenadas, scores y reglas. Actualizar fuente en UI e historial sin reinterpretar registros antiguos.

D. **Audio USB**: `AudioInputSource`, `InternalMicrophoneSource`, `UsbMicrophoneSource`, enumeración nativa mediante AudioManager. Mantener reconocimiento existente; ruta final administrada por Android. No abrir una grabación concurrente que compita con STT.

E. **Logging**: JSONL con metadatos por intento, errores diferenciados, rotación, debug de imágenes optativo y limpieza de capturas OCR/VLM.

F. **Seguridad**: HTTPS estricto en el cliente de inferencia, `.env.example`, exclusión de `.env`, credenciales AWS solo mediante rol del backend, permisos `InvokeEndpoint` limitados al recurso. API de laboratorio restringida por CIDR y frecuencia. El soporte HTTP LAN preexistente del VLM no se cambia; el cliente de inferencia rechaza HTTP aunque el manifest permita ese otro uso.

G. **Testing**: analizar y ejecutar pruebas existentes después de los grupos de cambios; probar JSON, bbox, confidence, mapper, vacío, HTTP, timeout, exclusión y desconexión/respuesta tardía. Compilar Android y verificar manualmente hardware/despliegue según PRUEBAS_HARDWARE.md.

## Fase 3: decisiones de integración

Se eliminó el fallback mock del flujo de inferencia: antes un fallo podía anunciar camino libre. Se retiró la fábrica de resultados simulados de AnalysisResult. Una lista vacía válida mantiene la regla CERM existente.

La app toma capturas bajo demanda. **Ver peligro** prepara el modo; las solicitudes de descripción/peligro ejecutan una captura. No hay promesa de vigilancia continua ni alertas automáticas entre capturas. Se conservaron funciones de clasificación/dirección/distancia y sus pruebas.

Las fuentes antiguas `tflite`/`tflite_float32` se siguen interpretando solo para leer historial previo. El VLM local se conserva por estar fuera de la migración YOLO.

La copia inicial de Kotlin usaba cachés incrementales incompatibles con Pub en C: y el proyecto en D:. Se desactivó la compilación incremental Kotlin y se limitó el número de workers para verificar Android en este workspace.

El Excel CERM y su generador describen la arquitectura previa, incluyendo los problemas de fallback e inferencia local. No deben regenerarse o citarse como especificación completa de esta migración sin actualizar ese estudio; las reglas CERM sí se mantienen. Las referencias de líneas de ese libro son históricas.

Consulta VALIDACION.md para los resultados efectivamente ejecutados y las comprobaciones físicas pendientes.
