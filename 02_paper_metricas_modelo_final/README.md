# Metricas y figuras para paper

Carpeta armada a partir del experimento:

`01_app_magicvision_final\app_flutter_magicvision\runs\detect\modelo_final_yolo26m_80e_640_b8`

Configuracion principal:

- Modelo: `yolo26m.pt`
- Tarea: deteccion (`detect`)
- Epocas: 80
- Batch: 8
- Tamano de imagen: 640
- Dataset: `01_app_magicvision_final\app_flutter_magicvision\ml\datasets\dataset_final_magicvision`
- Clases: `person`, `car`, `motorcycle`, `bus`, `bicycle`, `stairs`, `pothole`, `pole`, `traffic_sign`, `obstacle`

## Metricas principales

Los valores reportados por el usuario corresponden a la fila mas cercana en la epoca 50 del archivo `results.csv`.

| Metrica | Valor para paper | Valor exacto en epoca 50 |
|---|---:|---:|
| mAP50 | 0.587 | 0.58750 |
| mAP50-95 | 0.353 | 0.35306 |
| Precision | 0.683 | 0.68426 |
| Recall | 0.558 | 0.55655 |

La epoca 50 tambien es la mejor por `mAP50-95` dentro de este entrenamiento.

## Contenido

- `datos/results_modelo_final_yolo26m_80e_640_b8.csv`: historial completo de metricas por epoca.
- `datos/args_modelo_final_yolo26m_80e_640_b8.yaml`: parametros del entrenamiento final.
- `datos/metricas_para_paper.csv`: valores resumidos para tabla del paper.
- `datos/metricas_epoch50_exactas.csv`: fila exacta usada como respaldo.
- `datos/mejores_epocas.csv`: mejores epocas por `mAP50` y `mAP50-95`.
- `figuras/`: imagenes copiadas y renombradas para insertar en el paper.

## Figuras recomendadas

- `01_curvas_entrenamiento_resultados.png`: evolucion de perdidas y metricas.
- `02_matriz_confusion.png`: matriz de confusion absoluta.
- `03_matriz_confusion_normalizada.png`: matriz de confusion normalizada.
- `04_curva_precision_recall.png`: curva Precision-Recall.
- `05_curva_f1_confianza.png`: F1 segun umbral de confianza.
- `06_curva_precision_confianza.png`: precision segun umbral de confianza.
- `07_curva_recall_confianza.png`: recall segun umbral de confianza.
- `08_distribucion_etiquetas.jpg`: distribucion de clases y cajas del dataset.
- `10_validacion_batch0_predicciones.jpg`, `12_validacion_batch1_predicciones.jpg`, `14_validacion_batch2_predicciones.jpg`: ejemplos visuales de prediccion.
