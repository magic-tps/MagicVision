"""Create a verified handoff from existing artifacts; no AWS calls or model changes."""
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    destination = args.dest.resolve()
    archive_path = destination.with_suffix('.zip')
    checksum_path = archive_path.with_suffix('.zip.sha256')
    if any(p.exists() for p in (destination, archive_path, checksum_path)):
        parser.error('Choose a new destination; existing deliveries are never overwritten.')

    model = project / 'backend/artifacts/model.tar.gz'
    expected_model = '1af51ad50198e77ba31b1538fddf1ba15d0567d291fdf11dee3a22eff25fa780'
    assert sha256(model) == expected_model, 'Model archive differs from validated artifact'
    with tarfile.open(model, 'r:gz') as package:
        members = package.getmembers()
        assert len(members) == 3
        assert {p.name for p in members} == {'best.pt', 'labels.txt', 'config.json'}
        assert all(p.isfile() for p in members)
        with package.extractfile('best.pt') as stream:
            weights_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert weights_hash == sha256(project / 'runs/detect/modelo_final_yolo26m_80e_640_b8/weights/best.pt')
        labels = package.extractfile('labels.txt').read().decode('utf-8').splitlines()
        assert labels == (project / 'ml/configs/classes.txt').read_text(encoding='utf-8').splitlines()
        config = json.load(package.extractfile('config.json'))
        assert config == {'imgsz': 640, 'confidence': 0.30, 'iou': 0.65, 'max_det': 100}

    api = project / 'backend/artifacts/api/api.zip'
    api_proof = json.loads((api.parent / 'package-tested.json').read_text(encoding='utf-8'))
    assert sha256(api) == api_proof['sha256']
    for name, expected in api_proof['sources'].items():
        assert sha256(project / 'backend/lambda' / name) == expected, name
    with ZipFile(api) as package:
        assert package.testzip() is None
        for name in ('api.py', 'async_jobs.py', 'schema.py'):
            assert package.read(name) == (project / 'backend/lambda' / name).read_bytes()
        assert not any(p.endswith(('.pt', '.tflite', '.dll', '.pyd')) for p in package.namelist())

    docker_proof = json.loads((project / 'backend/artifacts/aws/docker-tested-cuda.json').read_text(encoding='utf-8'))
    assert docker_proof['model_sha256'] == expected_model
    for name, expected in docker_proof['sources'].items():
        assert sha256(project / 'backend/sagemaker' / name) == expected, name

    sources = {'model.tar.gz': model}
    for directory in ('backend/sagemaker', 'backend/lambda'):
        for source in sorted((project / directory).rglob('*')):
            if source.is_file() and '__pycache__' not in source.parts and source.suffix != '.pyc':
                sources[source.relative_to(project).as_posix()] = source
    for name in ('backend/api-template.yaml', 'backend/artifacts/api/api.zip',
                 'backend/artifacts/api/package-tested.json'):
        sources[name] = project / name
    sources['evidencia/docker-tested-cuda.json'] = project / 'backend/artifacts/aws/docker-tested-cuda.json'
    for name, source in sources.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha256(target) == sha256(source), name

    write(destination / 'LEEME_ENTREGA.md', r'''
# Entrega MagicVision para primera prueba en AWS

Fecha: 10 de septiembre de 2026.

Este paquete contiene el servidor completo y el modelo ya armado para que el responsable de AWS construya la imagen, la publique en ECR, suba el modelo a S3 y despliegue SageMaker Async. También incluye la API intermedia existente y su ZIP Linux, porque aún falta desplegar esa parte para conectar la app.

## Contenido

| Ruta | Uso |
|---|---|
| `backend/sagemaker/` | Código completo del contenedor: Dockerfile, serve.py, requirements.txt, .dockerignore y verificador local |
| `model.tar.gz` | Artefacto listo para S3; best.pt, labels.txt y config.json directamente en la raíz |
| `backend/lambda/` | Código y receta de empaquetado de la API intermedia |
| `backend/artifacts/api/api.zip` | API y dependencias Linux para Lambda Python 3.11 x86_64 |
| `backend/api-template.yaml` | CloudFormation de API Gateway + Lambda, usando endpoint y rol existentes |
| `API_INTERMEDIA.md` | Estado real, configuración y diferencias de timeout de la API actual |
| `configuracion/` | Ejemplos de parámetros acordados, con nombres y bucket por completar |
| `evidencia/` | Integridad comprobada en esta entrega y referencia de la prueba CUDA local previa |
| `MENSAJE_PARA_ENVIAR.txt` | Borrador de respuesta para acompañar el ZIP |
| `SHA256SUMS.txt` | Huellas SHA-256 de todos los archivos de la entrega, excepto el propio manifiesto |

No se necesitan el dataset, el entrenamiento ni el proyecto Flutter para construir este contenedor. Los pesos se proporcionan por separado de la imagen Docker.

## Construir la imagen

Ejecutar desde esta carpeta extraída, con Docker en modo Linux:

```powershell
docker build --platform linux/amd64 --provenance=false --build-arg TORCH_FLAVOR=cu126 -t magicvision-yolo26m:entrega ./backend/sagemaker
tar -tzf model.tar.gz
Get-FileHash model.tar.gz -Algorithm SHA256
```

El hash esperado de model.tar.gz es:

```text
1af51ad50198e77ba31b1538fddf1ba15d0567d291fdf11dee3a22eff25fa780
```

El Dockerfile fija PyTorch 2.10.0, torchvision 0.25.0, CUDA cu126 y Ultralytics 8.4.56. Sirve HTTP en el puerto 8080, con GET /ping y POST /invocations. SageMaker debe montar el modelo en /opt/ml/model; configurar MODEL_PATH=/opt/ml/model/best.pt y MODEL_DEVICE=cuda en el modelo de SageMaker. Esta selección exige GPU y falla si CUDA no está disponible.

## Parámetros de la primera prueba solicitada

| Parámetro | Valor solicitado |
|---|---|
| Instancia | ml.g4dn.xlarge |
| Modalidad | Asynchronous Inference |
| Instancias iniciales | 1 |
| MaxConcurrentInvocationsPerInstance | 2 |
| Auto Scaling | MinCapacity=0, MaxCapacity=2 |
| Timeout de procesamiento | InvocationTimeoutSeconds=60 en cada invocación de prueba |
| Entrada y salida | image/jpeg y application/json |

Los JSON de configuracion/ son ejemplos que el desplegador debe completar. El timeout de procesamiento se pasa a InvokeEndpointAsync; no es un campo de EndpointConfig ni el tiempo máximo de arranque en frío. RequestTTLSeconds controla la espera en cola por separado. [Referencia de AWS](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpointAsync.html).

Configurar target tracking con ApproximateBacklogSizePerInstance y una política/alarma HasBacklogWithoutCapacity que reactive desde cero ante una solicitud. Los límites 0–2 por sí solos no completan esa configuración. [Auto Scaling Async en AWS](https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference-autoscale.html).

El servidor actual tiene dos hilos HTTP y protege model.predict con un lock: puede recibir dos solicitudes, pero ejecuta las predicciones de una en una por proceso. No se ha medido rendimiento con concurrencia 2 ni con dos instancias en AWS.

Los scripts históricos del proyecto usaban concurrencia 1, escalado 0–1 y una invocación de prueba con timeout de 300 segundos. No se incluyen como automatización de esta entrega porque no reflejan los valores solicitados. La API incluida también tiene tiempos propios: consultar API_INTERMEDIA.md antes de desplegarla.

## Modelo y validación

Parámetros preservados: imgsz=640, confidence=0.30, iou=0.65 y max_det=100. Clases, en orden: person, car, motorcycle, bus, bicycle, stairs, pothole, pole, traffic_sign, obstacle.

El contenedor acepta JPEG de hasta 2 MiB y dimensiones de hasta 1280×720, verificadas también después de aplicar la orientación EXIF. Devuelve image_width, image_height, processing_time_ms y detections; cada detección contiene class_id, label, confidence y bbox con left/top/right/bottom normalizados entre 0 y 1.

En esta entrega se verifica que el peso empaquetado coincide con el checkpoint entrenado, que las clases/configuración son las originales y que el código del servidor y de la API coincide con sus artefactos previamente probados. No se vuelve a entrenar ni convertir el modelo.

La evidencia del 9 de septiembre registra /ping e inferencia en Docker CPU y CUDA local (RTX 2070 Max-Q, CUDA 12.6). La integración local anterior usó el modelo real y transporte AWS simulado. Sigue pendiente comprobar la T4 de ml.g4dn.xlarge, la latencia fría/caliente y el ciclo de escalado en AWS. Esta entrega no crea recursos AWS ni acredita un endpoint publicado.

## Datos que necesitamos al terminar el despliegue

Nombre del endpoint SageMaker, región, URI del modelo en S3 y prefijos de entrada/salida/fallo. Si también se despliega la API, compartir su URL HTTPS (ApiBaseUrl de CloudFormation) y el CIDR autorizado. La app se configura con esa URL mediante API_BASE_URL; no utiliza directamente el endpoint de SageMaker.
''')

    write(destination / 'API_INTERMEDIA.md', r'''
# API intermedia: estado y entrega

Según el código y la evidencia disponible al preparar esta entrega, la API está implementada y probada localmente, pero no existe una URL de despliegue verificada. Para completar App → API → SageMaker hay que incluir su despliegue en AWS. El endpoint SageMaker por sí solo permite probar el modelo, pero no conecta la app actual.

## Componentes incluidos

- FastAPI + Mangum en backend/lambda/; no carga YOLO.
- ZIP Linux ya construido: backend/artifacts/api/api.zip.
- Plantilla backend/api-template.yaml: Lambda Python 3.11 x86_64, API Gateway REST regional con HTTPS y multipart binario.
- Flujo: prepara el modelo, sube el JPEG a S3, invoca SageMaker Async, devuelve job_id y permite consultar/cancelar.

## Configuración de despliegue

Subir api.zip a una clave propia del bucket del proyecto y proporcionar a CloudFormation estos parámetros:

| Parámetro | Valor que completa el desplegador |
|---|---|
| CodeBucket | Bucket que almacenará tanto el ZIP como los trabajos temporales de la app |
| CodeKey | Clave S3 de api.zip, por ejemplo api/entrega-20260910/api.zip |
| EndpointName | Nombre del endpoint Async existente, en la misma región |
| ExecutionRoleArn | ARN de un rol Lambda existente con permisos suficientes |
| AllowedClientCidr | CIDR IPv4 público autorizado para acceder a la API, por ejemplo la salida del laboratorio con /32 |

La plantilla establece INFERENCE_MODE=async, ASYNC_BUCKET=CodeBucket y SAGEMAKER_ENDPOINT_NAME=EndpointName. El endpoint debe escribir async-output/ y async-failure/ en ese mismo bucket: la API valida esas ubicaciones. Handler: api.handler. La salida ApiBaseUrl es la URL que se entrega al equipo de Flutter para configurar API_BASE_URL y recompilar la app.

El rol Lambda debe confiar en lambda.amazonaws.com, permitir sagemaker:InvokeEndpointAsync sobre el endpoint y acceso S3 a los trabajos: ListBucket; GetObject/DeleteObject en app-jobs/, async-input/app/, async-output/ y async-failure/; PutObject en app-jobs/ y async-input/app/. También necesita permisos para escribir sus logs de CloudWatch. La plantilla no crea el rol ni le adjunta políticas.

Configurar retención de un día para app-jobs/, async-input/app/, async-output/app/ y async-failure/app/, incluidas versiones anteriores si el bucket las conserva. Esas reglas no están en api-template.yaml; deben añadirse al bucket preservando las demás reglas. La limpieza de la API elimina imágenes/salidas al consumir o cancelar; la retención cubre trabajos abandonados y su eliminación es eventual. Verificar en la prueba real el prefijo exacto usado por las salidas de fallo.

La API está restringida por CIDR público; no implementa cuentas individuales. El teléfono debe salir por una red autorizada. GET /health solo comprueba configuración; la prueba completa debe incluir un trabajo de preparación y una imagen real autorizada.

## Tiempos actuales: diferencia con los 60 segundos solicitados

Se incluye el código y ZIP ya validados, sin cambiar su comportamiento:

| Límite | Valor en el código actual |
|---|---|
| SageMaker InvocationTimeoutSeconds para una escena | 30 s |
| SageMaker InvocationTimeoutSeconds para preparación | 300 s |
| RequestTTLSeconds de escena / preparación | 60 s / 900 s |
| Caducidad de escena en API / app | 30 s / 15 s |
| Timeout Lambda / integración API Gateway | 28 s / 29 s |
| Timeout HTTP de la app | 10 s por llamada |

Por tanto, el JSON de invocación de prueba incluido usa 60 s, pero el ZIP de la API conserva 30/300 s. Si el acuerdo exige 60 s también en todas las llamadas de la API, hay que ajustar InvocationTimeoutSeconds en backend/lambda/async_jobs.py y reconstruir api.zip. No basta con cambiar EndpointConfig. La caducidad de escenas y la espera de preparación son límites independientes.

La receta para reconstruir la API después de un ajuste es backend/lambda/Dockerfile.package; debe ejecutarse en Linux amd64. El contenedor empaquetador escribe /output/api.zip, por lo que /output debe montarse en la carpeta de artefactos. No instalar dependencias Windows dentro del ZIP Lambda.

## Contrato para la conexión

| Operación | Resultado |
|---|---|
| POST /v1/inference/warmup (formulario request_id UUID) | 202 con job_id |
| POST /v1/inference (multipart request_id e image JPEG) | 202 con job_id |
| GET /v1/inference/{job_id} | 202 pendiente, 200 listo/detecciones, 410 cancelado/caducado |
| DELETE /v1/inference/{job_id} | 200 cancelled; descarta el resultado, sin detener una inferencia ya iniciada |

No se comparten claves AWS con Flutter. Se necesitan el nombre/región del endpoint y la URL HTTPS de esta API para cerrar la integración.
''')

    write(destination / 'MENSAJE_PARA_ENVIAR.txt', '''
Hola,

Te compartimos la entrega con backend/sagemaker/ completo (Dockerfile, servidor y requirements), y model.tar.gz ya armado con best.pt, labels.txt y config.json en la raíz. Incluimos un LEEME, ejemplos de configuración y hashes para verificar los archivos.

Sobre la API intermedia: está implementada y probada localmente, pero todavía no tenemos una URL desplegada y verificada. Para completar el flujo de la app, hay que sumar API Gateway + Lambda al despliegue en AWS. En el mismo ZIP incluimos el código, api.zip para Lambda Linux y la plantilla backend/api-template.yaml. Al finalizar necesitamos el nombre y la región del endpoint SageMaker y, si despliegas también la API, su URL HTTPS para configurar Flutter.

La guía recoge ml.g4dn.xlarge, Async Inference, concurrencia 2, escalado 0–2 e InvocationTimeoutSeconds=60 para la prueba. Dejamos explícita una diferencia del código actual: la API validada usa 30 s para escenas y 300 s para preparación; si los 60 s deben aplicarse también a la API, hay que ajustar ese parámetro y reconstruir su ZIP. El servidor recibe dos solicitudes, pero serializa las predicciones por proceso.

El contenedor CUDA 12.6 fue probado en nuestra GPU local. Falta validar la T4 de AWS y medir latencia fría/caliente y escalado. Podemos revisar juntos los resultados cuando esté disponible el despliegue.

Saludos,
Equipo MagicVision
''')

    endpoint_config = {
        'EndpointConfigName': 'magicvision-yolo26m-async',
        'ProductionVariants': [{'VariantName': 'AllTraffic', 'ModelName': 'REEMPLAZAR_NOMBRE_MODELO',
            'InitialInstanceCount': 1, 'InstanceType': 'ml.g4dn.xlarge', 'InitialVariantWeight': 1.0,
            'ModelDataDownloadTimeoutInSeconds': 600, 'ContainerStartupHealthCheckTimeoutInSeconds': 600}],
        'AsyncInferenceConfig': {'ClientConfig': {'MaxConcurrentInvocationsPerInstance': 2},
            'OutputConfig': {'S3OutputPath': 's3://REEMPLAZAR_BUCKET/async-output/',
                             'S3FailurePath': 's3://REEMPLAZAR_BUCKET/async-failure/'}}}
    invocation = {'EndpointName': 'REEMPLAZAR_ENDPOINT',
        'InputLocation': 's3://REEMPLAZAR_BUCKET/async-input/prueba.jpg',
        'ContentType': 'image/jpeg', 'Accept': 'application/json',
        'RequestTTLSeconds': 900, 'InvocationTimeoutSeconds': 60}
    scaling = {'ServiceNamespace': 'sagemaker',
        'ResourceId': 'endpoint/REEMPLAZAR_ENDPOINT/variant/AllTraffic',
        'ScalableDimension': 'sagemaker:variant:DesiredInstanceCount', 'MinCapacity': 0, 'MaxCapacity': 2}
    for name, value in [('endpoint-config.example.json', endpoint_config),
                        ('invoke-async.example.json', invocation), ('scalable-target.example.json', scaling)]:
        write(destination / 'configuracion' / name, json.dumps(value, indent=2))

    report = {'date': '2026-09-10', 'model_sha256': expected_model,
        'checkpoint_sha256': weights_hash, 'model_members': ['best.pt', 'labels.txt', 'config.json'],
        'labels': labels, 'model_config': config, 'api_zip_sha256': sha256(api),
        'checks': {'model_matches_original_checkpoint': True, 'labels_and_config_match': True,
            'serving_sources_match_previous_cuda_validation': True,
            'api_zip_matches_previous_validation': True, 'api_sources_match_zip': True,
            'api_zip_crc_valid': True, 'copied_sources_byte_identical': True},
        'aws_deployed_by_this_packaging': False, 'new_inference_run': False,
        'api_current_invocation_timeouts_seconds': {'scene': 30, 'warmup': 300},
        'requested_test_invocation_timeout_seconds': 60}
    write(destination / 'evidencia/integridad_entrega.json', json.dumps(report, indent=2))
    files = sorted(p for p in destination.rglob('*') if p.is_file())
    write(destination / 'SHA256SUMS.txt', '\n'.join(
        f'{sha256(p)}  {p.relative_to(destination).as_posix()}' for p in files))
    files = sorted(p for p in destination.rglob('*') if p.is_file())
    with ZipFile(archive_path, 'x', compression=ZIP_DEFLATED, compresslevel=6) as output:
        for path in files:
            method = ZIP_STORED if path.name.endswith(('.zip', '.tar.gz')) else ZIP_DEFLATED
            output.write(path, f'{destination.name}/{path.relative_to(destination).as_posix()}', compress_type=method)
    with ZipFile(archive_path) as output:
        assert output.testzip() is None
        assert len(output.infolist()) == len(files)
        for path in files:
            with output.open(f'{destination.name}/{path.relative_to(destination).as_posix()}') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == sha256(path)
    write(checksum_path, f'{sha256(archive_path)}  {archive_path.name}')
    print(json.dumps({'folder': str(destination), 'zip': str(archive_path),
        'files': len(files), 'bytes': archive_path.stat().st_size,
        'sha256': sha256(archive_path), 'integrity': 'passed'}, indent=2))


if __name__ == '__main__':
    main()
