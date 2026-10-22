# Validación de la migración

Fecha: 9 de septiembre de 2026. Workspace Windows, Flutter 3.35.3 / Dart 3.9.2.

Actualización posterior: se preparó el despliegue Async y se probó el contenedor real en CPU/CUDA. Ver [VALIDACION_AWS_ASYNC.md](VALIDACION_AWS_ASYNC.md) para esa evidencia y los bloqueos de AWS; el informe de abajo conserva los resultados de la migración Flutter/UVC inicial.

Estado más reciente: [VALIDACION_FUNCIONAL_ASYNC.md](VALIDACION_FUNCIONAL_ASYNC.md), con integración Flutter/API Async, prueba HTTPS con el modelo real, cancelación y caducidad. Las instrucciones del despliegue síncrono de abajo son históricas.

## Comprobaciones ejecutadas

| Comprobación | Resultado |
|---|---|
| `flutter analyze --no-pub` | Sin incidencias |
| `flutter test --no-pub` | 44 pruebas pasan: 23 conservadas y 21 nuevas |
| API Python / `pytest backend -q` | 11 pruebas pasan |
| `cfn-lint -t backend/template.yaml` | Plantilla SAM/CloudFormation válida estáticamente |
| Servidor SageMaker con el checkpoint real | Smoke test local: carga completa, labels coherentes, `/ping`, rechazo de MIME inválido y `/invocations` con JPEG sintético 640x480 |
| `flutter build apk --debug --target-platform android-arm64` | APK de depuración compilado |
| Auditoría del APK | Bibliotecas UVC presentes; sin `magicvision_detector.tflite`, `.pt` ni `.onnx` de YOLO |
| Comparación CERM contra respaldo | `RiskAssessmentService` idéntico salvo el valor predeterminado de `source` |
| Dependencias Flutter | No contiene `tflite_flutter` ni sus intérpretes de YOLO |

Las pruebas nuevas cubren JSON, tipos, bbox y normalización, confianza, mapper y duplicados, escena vacía, compresión, frame inválido, HTTP, timeout, HTTPS, errores de red, exclusión de requests, desconexión durante inferencia y durante logging, respuesta tardía, reconexión, permiso USB rechazado y fallback de audio.

ML Kit conserva sus modelos internos `.tflite` de OCR dentro del APK. Son parte de la funcionalidad OCR existente, no del detector YOLO. El APK sigue siendo grande por los assets del VLM Cactus/LFM que se conservaron; el APK final auditado mide 1,642,648,278 bytes.

La configuración de compilación Kotlin desactiva las cachés incrementales que fallaban por tener Pub en C: y el proyecto en D:. Las advertencias de Java 8/operaciones sin comprobación provienen de plugins existentes. Las pruebas Python muestran dos avisos de deprecación de Starlette/TestClient, sin fallos.

## Artefacto de modelo

`backend/artifacts/model.tar.gz` contiene exclusivamente `best.pt`, `labels.txt` y `config.json`.

SHA-256:

```text
1af51ad50198e77ba31b1538fddf1ba15d0567d291fdf11dee3a22eff25fa780
```

El smoke test inicial se ejecutó en el equipo de desarrollo con Ultralytics 8.4.56 y PyTorch 2.11.0+cu128, usando CPU. El 18 de septiembre se repitió el contrato con PyTorch 2.7.1+cpu y torchvision 0.22.1+cpu para comprobar la compatibilidad de `best.pt` antes de cambiar la receta a cu118. Esto verifica el código de serving y el checkpoint; no acredita un despliegue real ni la ejecución CUDA de la nueva imagen. El JPEG sintético no constituye una evaluación de precisión del modelo.

## Comprobaciones externas pendientes

- Desplegar los recursos en una cuenta AWS autorizada, proporcionar los parámetros de `backend/template.yaml` y configurar `API_BASE_URL`.
- La construcción y prueba Docker se completaron en la etapa posterior Async; queda pendiente verificar el contenedor en una instancia AWS autorizada.
- Conectar un teléfono y la Arducam. ADB no mostraba dispositivos disponibles.
- Verificar preview, formatos UVC, montaje/orientación, permisos, ciclos de reconexión y funcionamiento de OCR/TTS/SOS en hardware.
- Confirmar acústicamente la entrada efectiva del STT. Android administra la ruta USB; detectar un micrófono no demuestra que el reconocedor lo haya seleccionado.
- Medir latencia real, red, consumo y calidad del flujo completo. Las métricas del entrenamiento anterior no equivalen a estas mediciones.

Lista de pruebas: [PRUEBAS_HARDWARE.md](PRUEBAS_HARDWARE.md). Despliegue: [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md).

## Respaldo y evidencia histórica

El código y la configuración anteriores están en `../../respaldo_pre_migracion_cloud` respecto de la raíz de la app. El export YOLO antiguo está en `ml/legacy_exports/`, fuera de los assets Flutter. Los datasets, checkpoints de entrenamiento, material de paper y Excel CERM originales se conservaron.

Los logs locales `analysis_validation.log`, `test_validation.log`, `backend_test_validation.log` y `build_validation_final.log` sirven como evidencia de esta ejecución y están ignorados por Git.
