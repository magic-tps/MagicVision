# Comparación medida del checkpoint operativo

Mismo checkpoint e imágenes de validación. No se ha usado el conjunto test para seleccionar parámetros.
Las métricas usan correspondencia por clase, IoU ≥ 0.50 y asignación uno a uno en orden de confianza. AP se integra en 101 puntos; no es una reproducción exacta del evaluador histórico.
La compresión app640 es una aproximación PIL, no una captura del S25. Las latencias locales incluyen preprocesamiento del predictor; hubo trabajo concurrente de CPU/disco, por lo que no demuestran rendimiento aislado ni latencia AWS.

| Experimento | Umbral | Clase | Precision | Recall | FN | FP/imagen |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| E0_640 | 0.15 | person | 0.694 | 0.684 | 668 | 0.6255 |
| E0_640 | 0.15 | stairs | 0.318 | 0.700 | 3 | 0.0147 |
| E0_640 | 0.15 | obstacle | 0.379 | 0.424 | 34 | 0.0402 |
| E0_640 | 0.20 | person | 0.745 | 0.655 | 730 | 0.4647 |
| E0_640 | 0.20 | stairs | 0.368 | 0.700 | 3 | 0.0118 |
| E0_640 | 0.20 | obstacle | 0.418 | 0.390 | 36 | 0.0314 |
| E0_640 | 0.25 | person | 0.788 | 0.625 | 792 | 0.3490 |
| E0_640 | 0.25 | stairs | 0.375 | 0.600 | 4 | 0.0098 |
| E0_640 | 0.25 | obstacle | 0.426 | 0.390 | 36 | 0.0304 |
| E0_640 | 0.30 | person | 0.824 | 0.606 | 832 | 0.2686 |
| E0_640 | 0.30 | stairs | 0.357 | 0.500 | 5 | 0.0088 |
| E0_640 | 0.30 | obstacle | 0.435 | 0.339 | 39 | 0.0255 |
| E0_640_appjpeg | 0.15 | person | 0.729 | 0.630 | 782 | 0.4843 |
| E0_640_appjpeg | 0.15 | stairs | 0.261 | 0.600 | 4 | 0.0167 |
| E0_640_appjpeg | 0.15 | obstacle | 0.373 | 0.373 | 37 | 0.0363 |
| E0_640_appjpeg | 0.20 | person | 0.777 | 0.598 | 850 | 0.3559 |
| E0_640_appjpeg | 0.20 | stairs | 0.353 | 0.600 | 4 | 0.0108 |
| E0_640_appjpeg | 0.20 | obstacle | 0.420 | 0.356 | 38 | 0.0284 |
| E0_640_appjpeg | 0.25 | person | 0.810 | 0.568 | 914 | 0.2765 |
| E0_640_appjpeg | 0.25 | stairs | 0.357 | 0.500 | 5 | 0.0088 |
| E0_640_appjpeg | 0.25 | obstacle | 0.500 | 0.356 | 38 | 0.0206 |
| E0_640_appjpeg | 0.30 | person | 0.841 | 0.545 | 961 | 0.2137 |
| E0_640_appjpeg | 0.30 | stairs | 0.357 | 0.500 | 5 | 0.0088 |
| E0_640_appjpeg | 0.30 | obstacle | 0.513 | 0.339 | 39 | 0.0186 |
| E0_960 | 0.15 | person | 0.664 | 0.682 | 672 | 0.7157 |
| E0_960 | 0.15 | stairs | 0.294 | 0.500 | 5 | 0.0118 |
| E0_960 | 0.15 | obstacle | 0.333 | 0.356 | 38 | 0.0412 |
| E0_960 | 0.20 | person | 0.718 | 0.660 | 718 | 0.5363 |
| E0_960 | 0.20 | stairs | 0.357 | 0.500 | 5 | 0.0088 |
| E0_960 | 0.20 | obstacle | 0.404 | 0.356 | 38 | 0.0304 |
| E0_960 | 0.25 | person | 0.758 | 0.632 | 779 | 0.4186 |
| E0_960 | 0.25 | stairs | 0.364 | 0.400 | 6 | 0.0069 |
| E0_960 | 0.25 | obstacle | 0.435 | 0.339 | 39 | 0.0255 |
| E0_960 | 0.30 | person | 0.789 | 0.600 | 846 | 0.3324 |
| E0_960 | 0.30 | stairs | 0.364 | 0.400 | 6 | 0.0069 |
| E0_960 | 0.30 | obstacle | 0.556 | 0.339 | 39 | 0.0157 |

Decisión: conservar checkpoint y valores operativos 640 / 0.30. Las metas de recall no están alcanzadas en esta validación; falta el lote independiente del campus y límites aceptados de falsas alarmas/latencia. E1/E2/E3 no se presentan como ejecutados.
La clase trash_bin no existe en este checkpoint: no confundir sus ceros/sin soporte con una evaluación de un detector nuevo. Las 10 anotaciones de escaleras en val son insuficientes para generalizar al campus.