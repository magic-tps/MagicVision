# MagicVision Dataset

MagicVision detecta 10 clases urbanas:

`person`, `car`, `motorcycle`, `bus`, `bicycle`, `stairs`, `pothole`, `pole`, `traffic_sign`, `obstacle`.

## Estructura Esperada

```text
ml/datasets/dataset_final_magicvision/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Las etiquetas deben estar en formato YOLO: `class_id x_center y_center width height`, con valores normalizados entre 0 y 1.

Esta version limpia del workspace conserva solo el dataset final dividido en `train`, `val` y `test`. Las carpetas crudas/intermedias y `images/all` fueron eliminadas para evitar duplicados y confusion; si necesitas reconstruir el dataset desde cero, vuelve a descargar/procesar las fuentes antes de ejecutar `merge_datasets.py` y `split_dataset.py`.

## Fuentes

### COCO

COCO 2017 sirve para arrancar con:

- `person`
- `car`
- `motorcycle`
- `bus`
- `bicycle`

Script:

```bash
python ml/scripts/download_coco.py --download-images
```

COCO completo pesa bastante. Si trabajas en laptop con poco espacio, descarga primero solo anotaciones o ejecuta el proceso en Google Colab.

### BDD100K

BDD100K es util para contexto de trafico urbano: vehiculos, personas, bicicletas, motos, buses, escenas de dia/noche y condiciones variadas.

BDD100K suele requerir cuenta y aceptar terminos de uso. No se incluyen credenciales. Descargalo desde la pagina oficial y luego convierte sus anotaciones a YOLO o usa una version ya convertida cuya licencia puedas usar.

### Roboflow Universe

Roboflow Universe puede ayudar con clases que COCO no cubre bien:

- `pothole`
- `stairs`
- `obstacle`
- `traffic_sign`
- `pole`

Busca manualmente datasets reales con terminos como:

- `pothole detection`
- `stairs detection`
- `obstacle detection visually impaired`
- `traffic sign detection`
- `pole detection street`

Revisa licencia, numero de imagenes, calidad de bounding boxes y clases antes de usarlo. Luego descarga con:

```bash
set ROBOFLOW_API_KEY=tu_api_key
python ml/scripts/download_roboflow.py --dataset workspace/project/version
```

Tambien puedes descargar manualmente desde Roboflow en formato YOLOv8 y copiarlo a `ml/datasets/processed/roboflow/`.

### Kaggle

Kaggle es alternativa para potholes, escaleras, obstaculos y mirrors de datasets de trafico. Configura credenciales:

```bash
pip install kaggle
# coloca kaggle.json en ~/.kaggle/kaggle.json
python ml/scripts/download_kaggle.py --dataset owner/dataset-slug
```

No uses un dataset solo por estar disponible: valida licencia, clases, formato y procedencia.

## Data Propia de Lima / Peru

Esta seccion es obligatoria para la tesis. Los datasets internacionales no representan completamente la realidad urbana peruana: veredas irregulares, huecos, postes mal ubicados, comercio ambulatorio, paraderos, rampas improvisadas, entradas de mercados, baja iluminacion y senalizacion local.

Recolecta fotos o videos en:

- veredas
- cruces peatonales
- avenidas
- paraderos
- escaleras
- rampas
- huecos
- postes
- senales
- obstaculos temporales
- zonas con baja iluminacion
- entradas de edificios
- mercados
- universidades

Recomendaciones:

- Captura en horizontal y vertical.
- Incluye dia, noche, sombra y lluvia si es posible.
- Evita rostros identificables cuando no tengas consentimiento.
- Etiqueta con bounding boxes ajustadas.
- Balancea clases: no entrenes solo vehiculos si tu tesis necesita huecos y obstaculos.

Para extraer frames:

```bash
python ml/scripts/extract_frames.py --input ml/datasets/raw/videos --output ml/datasets/raw/frames --every_seconds 2
```

Etiqueta los frames con CVAT, Label Studio, Roboflow Annotate o makesense.ai y exporta en YOLO.

## Unificar, Dividir y Validar

```bash
python ml/scripts/merge_datasets.py
python ml/scripts/split_dataset.py --train 0.7 --val 0.2 --test 0.1
python ml/scripts/validate_dataset.py --dataset ml/datasets/dataset_final_magicvision
```

Antes de entrenar, revisa:

- imagenes sin label
- labels sin imagen
- clases fuera de rango
- cajas invalidas
- labels vacios
- extensiones no soportadas
- duplicados simples por nombre

## Mejora Continua

Despues de probar en campo:

- guarda falsos positivos y falsos negativos
- agrega mas ejemplos de Lima
- reetiqueta cajas deficientes
- balancea clases dificiles como `pothole`, `stairs`, `pole` y `obstacle`
- evalua por clase, no solo mAP global
- prueba el modelo en celular antes de asumir que la metrica de Colab es suficiente
