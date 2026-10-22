# Validación de la preparación AWS Async

Fecha: 9 de septiembre de 2026. Esta evidencia corresponde al flujo nuevo de `scripts/aws/`; no sustituye la validación previa de Flutter/UVC.

Actualización del 18 de septiembre de 2026: la imagen descrita abajo con CUDA 12.6 fue una validación local histórica y falló al iniciar en SageMaker con el driver 470.x (`CUDA driver version 11040`). La receta vigente usa PyTorch 2.7.1, torchvision 0.22.1 y cu118. El checkpoint y el contrato de serving pasaron en PyTorch 2.7.1 CPU; la nueva imagen cu118 y su ejecución GPU en la T4 siguen pendientes de reconstrucción y prueba en AWS.

## Estado comprobado

| Comprobación | Resultado |
|---|---|
| Inventario del repositorio | Completado; modelo, clases, serving y plantilla síncrona localizados |
| AWS CLI | Bloqueado: `aws` no se encuentra en PATH ni en la instalación habitual de AWS CLI v2 |
| STS / cuenta autenticada | No verificable mientras falte la CLI |
| Región del laboratorio | No determinada; no se eligió una región por defecto |
| Cuota / rol / disponibilidad GPU en AWS | Pendientes de sesión y región verificadas |
| Docker | CLI 29.5.3; motor Linux x86_64 iniciado correctamente |
| Paquete del modelo | Verificado contra checkpoint, clases y parámetros originales |
| Scripts PowerShell | Sintaxis válida |
| Pruebas Python | 47 pasan: 36 nuevas + 11 del backend existente |
| Docker CUDA Linux amd64 (histórico) | Construcción local completada; PyTorch 2.10.0, CUDA 12.6, Ultralytics 8.4.56; incompatible con el driver SageMaker reportado |
| Receta cu118 vigente | PyTorch 2.7.1 y torchvision 0.22.1; checkpoint y contrato validados en CPU, reconstrucción/prueba T4 pendientes |
| `/ping` y `/invocations` en contenedor real | Pasan en CPU y CUDA con el checkpoint completo montado, JSON y dimensiones válidos |
| CUDA local | Pasa en NVIDIA RTX 2070 Max-Q, driver 572.83; logs confirman device=cuda |
| S3 / ECR / SageMaker / Auto Scaling / CloudWatch | No se ejecutaron operaciones de despliegue |

Las pruebas offline validan los JSON de operaciones AWS contra los esquemas de botocore, errores/normalización del servidor, selección CPU/CUDA, idempotencia, rechazo de cambios de instancia, ausencia de cuota, barrera de confirmación, negativa a sobrescribir modelos y procesamiento de éxito/fallo Async. No acreditan permisos, capacidad ni inferencia en AWS.

Se mantienen dos advertencias de deprecación Starlette/TestClient del backend anterior; no son fallos.

## Modelo

`backend/artifacts/model.tar.gz` contiene exactamente `best.pt`, `labels.txt`, `config.json` sin carpetas intermedias. El `.pt` empaquetado coincide por SHA-256 con el checkpoint entrenado.

SHA-256 del archivo comprimido:

```text
1af51ad50198e77ba31b1538fddf1ba15d0567d291fdf11dee3a22eff25fa780
```

No se entrenó, convirtió ni sustituyó el modelo. El test sintético solo demuestra funcionamiento del contrato; no mide precisión.

## Contenedor real histórico: CPU y GPU local con cu126

Imagen local: `magicvision-yolo26m:latest`, Linux amd64, con manifest sin attestation adicional. ID/digest local informado por Docker:

```text
sha256:43efadf11b5fc71f0d0f86ed89e5abbcc834e9ceaf7dec2d377fb7664589e265
```

Ambas ejecuciones completas de `test-docker.ps1` terminaron correctamente. Se probó primero `MV_LOCAL_DEVICE=cpu` y después `cuda`, usando la misma imagen. Se verificaron los hashes de `serve.py` y `requirements.txt` dentro del contenedor contra el workspace, y que PyTorch incluye CUDA 12.6. El modelo procede del montaje, no se incluye en la imagen Docker.

El JPEG sintético mide 640x480. Respuesta CUDA obtenida:

```json
{
  "detections": [],
  "image_height": 480,
  "image_width": 640,
  "processing_time_ms": 3329.58
}
```

La primera petición CPU reportó `2778.19 ms`. Son peticiones iniciales aisladas en el equipo de desarrollo, con inicialización y carga del entorno; **no son un benchmark CPU/GPU ni latencia AWS**. No se ha medido todavía el cold start de SageMaker ni el comportamiento 0→1→0. `startup_time_ms` del log mide la carga/movimiento del modelo después de importar frameworks, no todo el arranque del proceso o de una instancia.

Evidencia local ignorada por Git en `backend/artifacts/aws/`: `docker-tested-cpu.json`, `docker-tested-cuda.json`, `local-result-cpu.json`, `local-result-cuda.json`, `local-cpu.log`, `local-cuda.log`. `docker-tested.json` conserva la prueba CUDA más reciente.

Se corrigió un error de consola Windows al imprimir Unicode en los logs. La salida ahora usa UTF-8 y la limpieza del contenedor se ejecuta aunque falle el logging. Se retiraron exclusivamente los contenedores efímeros de estas pruebas; no se eliminaron recursos AWS ni otros contenedores del usuario.

## Bloqueo reproducible

Comandos intentados: `aws --version`, `aws configure get region`, `aws sts get-caller-identity --output json --no-cli-pager`.

PowerShell informa: `El término 'aws' no se reconoce como nombre de un cmdlet, función, archivo de script o programa ejecutable.`

`scripts/aws/check-environment.ps1` también se detiene correctamente con `Executable missing: aws. Install/configure it before continuing.` El reporte queda en `backend/artifacts/aws/last-error.txt`. No existe todavía un error de permisos emitido por AWS: no se logró contactar con la cuenta mediante la CLI.

Para continuar: instalar AWS CLI v2, iniciar/configurar la sesión del laboratorio fuera del repositorio, seleccionar su región asignada y repetir `check-environment.ps1`. No entregar ni guardar claves en el proyecto.

## Recursos propuestos, todavía no creados

- Bucket: se resolverá como `magicvision-ACCOUNT_ID-REGION` con STS y región real.
- ECR URI: pendiente de cuenta/región; repositorio propuesto `magicvision-yolo26m`.
- SageMaker Model: `magicvision-yolo26m`.
- EndpointConfig y endpoint: `magicvision-yolo26m-async`.
- Instancia propuesta: `ml.g4dn.xlarge`, sin cuota/disponibilidad aún verificadas.
- Región: pendiente, sin valor inventado.

No se ha solicitado confirmación de creación porque aún faltan los datos reales y las comprobaciones previas. La creación del endpoint exige mostrar región, instancia y nombre, y recibir confirmación explícita, según la solicitud del usuario.

## Archivos y alcance

Guía operativa: [aws-deployment.md](aws-deployment.md). Configuración central: [config.ps1](../scripts/aws/config.ps1). Orquestación: [deploy.ps1](../deploy.ps1). La integración posterior de Flutter/API con trabajos, consulta de resultados, cancelación y caducidad se documenta en [VALIDACION_FUNCIONAL_ASYNC.md](VALIDACION_FUNCIONAL_ASYNC.md). Los resultados de arriba corresponden a la primera etapa del contenedor y scripts.

Respaldo del serving y documentos modificados inicialmente: `D:/Escritorio/Tesis/respaldo_pre_async_aws_20260909`. Los logs locales `docker_async_build_final.log`, `docker_async_test_cpu.log`, `docker_async_test_cuda.log` y `async_tests.log` están ignorados por Git; las pruebas no ejecutan AWS ni requieren credenciales. Los 19 archivos PowerShell de entrada/configuración pasan el parser de PowerShell.
