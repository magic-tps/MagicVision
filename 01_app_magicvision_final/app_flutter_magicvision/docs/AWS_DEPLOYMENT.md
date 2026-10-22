# API y despliegue AWS

> Esta guía conserva el despliegue **síncrono** anterior. Para la nueva preparación de SageMaker Async, CUDA y escalado a cero, usar [aws-deployment.md](aws-deployment.md) y `scripts/aws/`. La Lambda descrita aquí usa `InvokeEndpoint`; no funciona apuntándola directamente al endpoint Async.

## Contrato

`POST /v1/inference`, multipart con `image` de tipo `image/jpeg`; `request_id` UUID opcional, `timestamp` y `device_id` opcionales. La app siempre genera UUID y timestamp. No envía identificadores personales por defecto.

```json
{"request_id":"550e8400-e29b-41d4-a716-446655440000","processing_time_ms":145,"image_width":640,"image_height":480,"detections":[{"class_id":0,"label":"person","confidence":0.87,"bbox":{"left":0.25,"top":0.18,"right":0.61,"bottom":0.93}}]}
```

Coordenadas normalizadas respecto de la imagen enviada, no del tensor letterbox. Scores finitos 0..1; cajas con área positiva. Backend verifica correspondencia de las 10 clases con labels. `200` puede contener cero detecciones. Errores: 413 tamaño, 415 formato, 422 entrada inválida, 502 respuesta de modelo inválida, 503 servicio no disponible, 504 timeout. No se sustituyen errores por detecciones inventadas. La app rechaza JSON, dimensiones o correlación incorrectos.

Límites: JPEG 2 MiB, dimensiones 1280x720, respuesta 1 MiB. Lambda limita el cuerpo completo antes del parser multipart. No persiste imágenes ni utiliza DynamoDB/RDS para almacenar modelos.

## Artefactos

```powershell
python backend/package_model.py --weights runs/detect/modelo_final_yolo26m_80e_640_b8/weights/best.pt
```

Produce `backend/artifacts/model.tar.gz` con `best.pt`, `labels.txt`, `config.json`; no entrena ni convierte. El script rechaza sobrescribir un artefacto existente: usar otro `--output` para una nueva versión. SageMaker extrae S3 en `/opt/ml/model`.

El contenedor carga una vez el checkpoint completo con Ultralytics. Atiende `GET /ping` y `POST /invocations` en 8080. No usa TFLite. Para esta plantilla síncrona CPU se construye con `TORCH_FLAVOR=cpu` y `MODEL_DEVICE=cpu`. La receta ahora también permite CUDA para el flujo Async descrito en la guía nueva; requiere comprobación GPU real.

## Despliegue revisable

Requiere Docker activo, AWS CLI y SAM CLI en la máquina de despliegue, región/cuenta autorizadas, bucket S3 privado y repositorio ECR. No ejecutar estos pasos hasta disponer de esos recursos y aprobar el coste de un endpoint persistente.

1. Construir y publicar en ECR la imagen de `backend/sagemaker/Dockerfile`:

```powershell
docker build --platform linux/amd64 --build-arg TORCH_FLAVOR=cpu -t magicvision-inference backend/sagemaker
# Autenticarse con AWS CLI/SSO o rol desde la estación de despliegue.
# Etiquetar y publicar en el repositorio ECR autorizado.
```

2. Subir `model.tar.gz` al bucket privado, cifrado. Proporcionar en la plantilla la URL S3 y el ARN del mismo objeto. El ejemplo IAM supone cifrado administrado S3; para una clave KMS propia se requiere permiso específico de descifrado.
3. Desde `backend`, ejecutar `sam build --use-container` y después `sam deploy --guided` en la cuenta autorizada. `CodeUri: lambda/` evita empaquetar pesos, entorno virtual o datos dentro de Lambda.
4. Parámetros: `ModelDataUrl`, `ModelObjectArn`, `InferenceImageUri`, `AllowedClientCidr`, `InstanceType`. El CIDR debe ser la IP pública de salida del laboratorio/red de pruebas. No usar /0. La plantilla no modifica autenticación de la app; para movilidad pública se debe integrar el mecanismo de autenticación elegido en lugar de ampliar indiscriminadamente el acceso.
5. Usar la salida `ApiBaseUrl` como `API_BASE_URL` en `.env`, recompilar y probar desde la red autorizada.

La función tiene permiso exclusivamente `sagemaker:InvokeEndpoint` sobre el endpoint creado. SageMaker tiene lectura del objeto S3 y permisos ECR/logs; no hay recursos de entrenamiento. El endpoint permanece provisionado y factura hasta que se elimine. No se desplegó automáticamente desde este workspace.

## Desarrollo y pruebas

```powershell
python -m venv backend/.venv
backend/.venv/Scripts/python -m pip install -r backend/requirements-test.txt
backend/.venv/Scripts/python -m pytest backend -q
# Con un rol AWS de desarrollo y SAGEMAKER_ENDPOINT_NAME configurado:
backend/.venv/Scripts/python -m uvicorn api:app --app-dir backend/lambda --host 127.0.0.1 --port 8000
```

El servidor local se expone solo en loopback. La app exige HTTPS: para pruebas con teléfono se necesita un proxy TLS válido y control de acceso. Nunca resolverlo desactivando la validación de certificados ni incluyendo claves IAM en Dart.

Fuentes: [contrato SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html), [InvokeEndpoint](https://docs.aws.amazon.com/boto3/latest/reference/services/sagemaker-runtime/client/invoke_endpoint.html), [archivos multipart FastAPI](https://fastapi.tiangolo.com/tutorial/request-files/), [YOLO26](https://docs.ultralytics.com/models/yolo26/).
