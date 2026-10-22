# Arquitectura de MagicVision

```text
Usuario (botón/voz)
  -> DetectorPage
  -> VisionAnalysisService.analyzeCurrentFrame() [exclusión de solicitudes]
       -> CameraSource / UsbCameraSource
            -> MethodChannel + EventChannel
            -> UsbCameraBridge / CameraHelper / Android USB Host
       -> InferenceImageProcessor (isolate: orientación, resize, JPEG)
       -> ObjectDetectionService / AwsObjectDetectionService (HTTPS multipart)
            -> API -> rol IAM -> SageMaker -> YOLO26m completo
       <- DetectionResponse [validación de id, dimensiones, confianza y bbox]
       -> CermDetectionMapper -> CermInput -> RiskAssessmentService
       <- AnalysisResult
  -> UI + TTS + historial existente
  -> ExperimentalLogger (metadatos; imágenes desactivadas por defecto)
```

## Límites de responsabilidad

`ObjectDetectionService` es la interfaz abstracta. `AwsObjectDetectionService` es la implementación principal; la UI no conoce protocolos de SageMaker ni carga modelos YOLO. Un request posee UUID, timeout total de red y cliente HTTP cancelable. No hay reintentos automáticos que dupliquen inferencias.

`CameraSource` separa inicialización, estado, preview, capturas y cierre. `UsbCameraSource` usa eventos nativos de conexión/desconexión. `InternalCameraSource` conserva una alternativa explícita para laboratorio; no se activa automáticamente si falta la Arducam. `frames` emite capturas solicitadas, no un flujo que suba vídeo.

El puente nativo abre una cámara UVC con `com.herohan:UVCAndroid:1.0.13`. USBMonitor utiliza `UsbManager` para permisos. Se comprueba la clase de vídeo en dispositivos e interfaces; no se inventa un VID/PID basándose en el sensor IMX219. La textura de preview permanece local. Un callback copia únicamente el frame solicitado; la codificación ocurre en un executor. Cada sesión y captura se invalidan al cerrar, desconectar o agotar su espera. `onStop` libera la cámara; `onStart` permite reabrirla.

## CERM y contexto semántico

CERM es el nombre de tesis para `RiskAssessmentService`. Recibe detecciones normalizadas y mantiene reglas de zona central/inferior, área aparente, múltiples peligros, niveles alto/medio/bajo y plantillas de recomendación. El cambio de fuente a `aws_sagemaker` no altera las reglas.

La dirección egocéntrica y la distancia aproximada pertenecen a funciones separadas de `main.dart`, conservadas para HUD y resúmenes semánticos. No se añade distancia física calibrada al CERM. Tampoco existe una clase llamada USCE: el contexto visual real se encuentra en `MobileVlmClient` y las funciones que anclan su texto a las detecciones.

OCR y VLM reciben archivos temporales porque sus APIs requieren rutas. Se eliminan en `finally`. El detector cloud trabaja en memoria. La voz, preferencias, contactos, SOS, geolocalización, navegación e historial conservan sus implementaciones anteriores.

## Estados

`cameraDisconnected -> cameraConnecting -> cameraReady -> capturing -> uploading -> processing -> resultReady`.
Cualquier operación puede terminar en `error` o `cameraDisconnected`. Una desconexión invalida la respuesta pendiente. `processing` indica recepción/validación y CERM; el protocolo de una sola respuesta no permite observar el instante exacto de inicio de cómputo remoto. Durante la espera se presenta "Analizando entorno".

No se confunden errores con listas vacías. Los mensajes de conexión se anuncian al cambiar de estado. Solo se admite un análisis activo. Las detecciones vencen tras la ventana de frescura existente y no se reutilizan para crear una recomendación nueva sin captura.

## Datos experimentales

Archivo privado de la app: `applicationSupportDirectory/experiments/inference.jsonl`. Incluye `request_id`, `timestamp` UTC, `camera_source`, `original_resolution`, `inference_resolution`, `jpeg_size_bytes`, `network_latency_ms`, `cloud_inference_ms`, `total_processing_ms`, `detections_count`, `risk_level`, `recommendation`; añade `status` y `error_code` si corresponde. Las métricas que no llegaron a calcularse se registran como null.

`original_resolution` es la resolución del frame capturado, no los 8 MP nominales del sensor. `network_latency_ms` mide la ida/vuelta HTTP completa e incluye backend e inferencia; no es latencia pura de transporte. `cloud_inference_ms` procede de SageMaker. `total_processing_ms` incluye captura, compresión, HTTP y CERM, pero excluye TTS y escritura del registro. Rotación de metadatos: dos archivos de aproximadamente 10 MB. `SAVE_DEBUG_IMAGES=true` permite conservar el JPEG optimizado para pruebas; no se activa por defecto.

## Fuentes técnicas

- [USB Host Android](https://developer.android.com/develop/connectivity/usb/host).
- [UVCAndroid y sus APIs](https://github.com/shiyinghan/UVCAndroid).
- [Política de audio USB Android](https://source.android.com/docs/core/audio/usb).
- [Entrada externa del reconocedor](https://developer.android.com/reference/android/speech/RecognizerIntent#EXTRA_AUDIO_SOURCE): no expuesta por la integración STT actual; su soporte depende del servicio de reconocimiento.
- [Contrato de contenedores SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html).
- [Invocación del endpoint](https://docs.aws.amazon.com/boto3/latest/reference/services/sagemaker-runtime/client/invoke_endpoint.html).
- [YOLO26](https://docs.ultralytics.com/models/yolo26/).
