# Validación funcional de Flutter, API y YOLO26m Async

Fecha: 9 de septiembre de 2026. Este informe corresponde a la integración posterior a la migración UVC y a las pruebas iniciales del contenedor. No acredita un despliegue en AWS ni una prueba con Arducam conectada.

## Cambio funcional

La app y la API estaban separadas del endpoint Async: el cliente esperaba un resultado inmediato y la Lambda invocaba el endpoint síncrono. Ahora el flujo admite preparación, creación de trabajos, consulta, cancelación y caducidad.

1. Flutter solicita preparar el servicio. La API envía un JPEG sintético a SageMaker Async y devuelve un job_id aleatorio. La app espera hasta 15 minutos con llamadas HTTP cortas, mostrando preparación y permitiendo cancelar.
2. Solo cuando el trabajo de preparación termina se captura la escena. La imagen se orienta/reduce sin estirar y se envía por HTTPS como JPEG.
3. La API devuelve HTTP 202 y un identificador de trabajo; las consultas devuelven 202 mientras esté pendiente y 200 con las detecciones validadas cuando finaliza.
4. El cliente exige correlación, identidad estable del trabajo, geometría normalizada y las dimensiones de la captura. CERM conserva sus reglas. El resultado caduca a los 15 segundos desde el inicio de la captura, incluso si el retraso ocurre al guardar el registro o antes de narrar.
5. Cancelar por botón o «cancelar análisis», desconectar o cerrar la app invalida la respuesta. El backend guarda un marcador separado para que una consulta concurrente no reactive un trabajo cancelado. Cancelar no detiene una inferencia que SageMaker ya esté ejecutando.

## Evidencia ejecutada

| Comprobación | Resultado |
|---|---|
| Flutter analyze, código de aplicación | Sin incidencias |
| Flutter test | 54 pruebas pasan; la prueba que necesita servicios externos se omite en la ejecución ordinaria |
| Prueba integrada con modelo real | Pasa, ejecutada explícitamente por run_integration.py |
| Python: backend y scripts AWS | 75 pruebas pasan; dos avisos de deprecación de Starlette/TestClient |
| Lambda en Linux x86_64 | Imagen construida, dependencias importadas y ZIP generado en el runtime Lambda Python 3.11 |
| CloudFormation API | cfn-lint sin errores |
| Scripts PowerShell | 22 archivos pasan el parser de PowerShell |
| APK Android arm64 actualizado | Compilado correctamente; assembleDebug terminó en 395,5 s |
| AWS real | Pendiente: AWS CLI/sesión/región no disponibles |
| Android/USB/UAC/TTS audible | Pendiente: adb devices no muestra un teléfono |

La prueba integrada usa el código de API y Dart de la app, HTTPS con certificado de prueba confiado explícitamente y el checkpoint completo en Docker CUDA. Solo el transporte S3/SageMaker y la fuente de cámara se simulan. No hay respuestas de detección inventadas ni sustitución del modelo. Los archivos de transporte de prueba están en backend/testing y no se incluyen en el paquete Lambda.

JPEG real: ml/datasets/dataset_final_magicvision/images/test/coco_filtered_000000000872.jpg. Resultado:

- Imagen procesada: 621×640, JPEG de 140780 bytes.
- Una persona, confianza 0.922624; caja normalizada (0.245579, 0.196847, 0.686740, 0.955771).
- CERM: riesgo medio; «Se detecta una persona cerca. Avanza lentamente y manten distancia. Accion sugerida: Avanza con precaucion.»
- Preparación: 19276 ms. Antigüedad de la captura al completar: 4841 ms. Tiempo del modelo en esa petición: 2219.54 ms.
- Dos invocaciones: preparación y escena; cero imágenes/salidas del modelo restantes en el transporte simulado al finalizar.

Estas medidas pertenecen a una ejecución local en la GPU del equipo; no son latencias de AWS, un benchmark ni una evaluación de precisión. El texto de narración fue verificado; no se escuchó el TTS en un teléfono.

Evidencia generada en backend/artifacts/local-integration/: flutter-result.json, summary.json y model.log. La imagen de modelo utilizada conserva el ID sha256:43efadf11b5fc71f0d0f86ed89e5abbcc834e9ceaf7dec2d377fb7664589e265. El archivo model.tar.gz mantiene su SHA-256 1af51ad50198e77ba31b1538fddf1ba15d0567d291fdf11dee3a22eff25fa780.

El ZIP Linux backend/artifacts/api/api.zip tiene SHA-256 1f8f578ad3fff5a98b438aac1ea1c5be0f920042b475f18a15daf7a5bb96090c. Su prueba registra hashes de fuentes en package-tested.json. Contiene la API y dependencias, sin checkpoint, archivos de prueba ni bibliotecas Windows.

APK generado: build/app/outputs/flutter-apk/app-debug.apk, 1642656323 bytes. Se inspeccionó el archivo: contiene las bibliotecas UVC y el kernel Dart actualizado; no contiene pesos .pt/.onnx de YOLO, magicvision_detector.tflite ni tflite_flutter. Los modelos internos de OCR y los assets del VLM conservado explican que siga siendo grande. Es un APK de comprobación sin API_BASE_URL: se debe recompilar con la URL de AWS real para usar inferencia cloud.

SHA-256 del APK: c035a848e520be4b984fba741c04863f8f4ddf0c385048eb77ba5e8e092092e6. Los registros locales son integration_flutter_analyze.log, integration_flutter_tests.log, integration_python_tests.log, integration_real_model.log, integration_api_package.log e integration_android_build.log.

## Despliegue preparado

backend/api-template.yaml utiliza un rol Lambda existente y un endpoint Async existente. Configura multipart binario, HTTPS, acceso limitado al CIDR público autorizado, límites de solicitudes y logs sin cuerpos de imágenes. No crea roles IAM. scripts/aws/package-api.ps1 empaqueta en Linux; deploy-api.ps1 verifica rol, endpoint, artefacto, configuración, stack y /health. Solo después genera backend/artifacts/flutter-cloud.json con la URL real para compilar/ejecutar Flutter.

Los trabajos usan objetos temporales privados. Se eliminan las imágenes al consumir/cancelar, y se configuran reglas de expiración de un día para prefijos exclusivos de la app, incluidas versiones anteriores. S3 elimina de forma eventual. Se conservan las reglas ajenas, el modelo, el ZIP y los experimentos CLI. Una regla existente incompatible o un stack distinto bloquea la modificación automática.

scripts/aws/test-api.ps1 reutiliza la prueba Dart con el JPEG del dataset contra la API desplegada, usando certificados públicos y sin credenciales AWS en Flutter. Esa prueba todavía no se ejecutó porque no existe una URL de despliegue verificada. La guía completa está en [aws-deployment.md](aws-deployment.md).

## Qué falta para dar por terminado el sistema

1. Instalar AWS CLI v2 desde la sesión de Windows y configurar la sesión del laboratorio y su región asignada fuera del repositorio. La instalación/extracción mediante msiexec fue rechazada por la revisión automática con «blocked by policy»; no se reintentó por otra vía.
2. Verificar STS, cuota regional, roles SageMaker/Lambda existentes, permisos S3/ECR/API y CIDR de acceso. Antes de crear el endpoint GPU se deben mostrar los datos reales de región, instancia y nombre y recibir la confirmación solicitada por el usuario.
3. Desplegar, ejecutar test-api.ps1, verificar logs y el ciclo 0→1→0 en AWS; construir el APK con flutter-cloud.json real. El APK de comprobación sin API_BASE_URL informa que el servicio no está configurado.
4. Conectar teléfono y Arducam por OTG, instalar el APK configurado y ejecutar [PRUEBAS_HARDWARE.md](PRUEBAS_HARDWARE.md): preview, orientación, reconexión, voz, OCR, TTS, historial y audio USB/fallback. No se ha determinado por una prueba física qué micrófono usa el reconocedor Android.

El respaldo previo a esta integración está en D:/Escritorio/Respaldo_MagicVision_integracion_async_20260909. Los modelos, datasets y reglas de riesgo no se sustituyeron.
