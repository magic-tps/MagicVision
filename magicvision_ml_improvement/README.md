# Mejorar el reconocimiento de MagicVision

Esta carpeta reúne las herramientas para ampliar las imágenes de entrenamiento, revisar qué objetos aparecen en ellas y comprobar si un modelo nuevo reconoce mejor el entorno universitario.

La mejora busca reducir omisiones de elementos como personas, escaleras y tachos, además de incorporar categorías como puertas y rampas. Añadir imágenes es solo una parte del trabajo: también hay que revisar sus etiquetas y medir el resultado.

## Cómo se prepara una mejora

1. Reunir imágenes y registrar su procedencia y condiciones de uso.
2. Revisar las etiquetas para que describan los objetos visibles.
3. Separar los ejemplos usados para aprender de los reservados para evaluar.
4. Entrenar un modelo y comparar sus aciertos y errores con los del anterior.
5. Probar el resultado en la app antes de reemplazar el servicio que utiliza.

Las imágenes repetidas o tomadas en la misma sesión deben mantenerse juntas al repartir los datos. Así se evita evaluar el modelo con escenas casi iguales a las que ya vio al aprender.

## Trabajo actual

El detector de referencia tiene diez categorías. Para la ampliación a 22 se preparó un conjunto experimental de **24.226 imágenes**, con 998 revisiones humanas incorporadas y etiquetas de origen para el resto.

Hay ensayos locales interrumpidos. Estos archivos no acreditan un modelo ampliado terminado ni un reemplazo del detector de la app. Además, las etiquetas heredadas pueden estar incompletas y los extintores no tienen ejemplos independientes de validación y prueba en este conjunto.

La [guía de entrenamiento](../COMO_ENTRENAR_CAMPUS22.md) explica cómo comprobar los archivos, iniciar una ejecución y encontrar los resultados.

## Organización

| Carpeta | Para qué sirve |
| --- | --- |
| `configs/` | Define las categorías y las reglas para revisar imágenes. |
| `scripts/` | Descarga, organiza y revisa los datos; prepara informes. |
| `training/` | Comprueba los datos e inicia los entrenamientos. |
| `evaluation/` | Compara resultados y analiza errores. |
| `reports/` | Conserva evidencia e informes fechados de los experimentos. |
| `tests/` | Comprueba el funcionamiento de las herramientas. |

Los datasets y los modelos se guardan localmente y están excluidos de Git. Hay que restaurarlos para reproducir los experimentos. Las dependencias de Python se indican en [requirements.txt](requirements.txt); PyTorch debe instalarse con una versión compatible con la GPU.

Para revisar las etiquetas, consulta el [procedimiento de revisión](configs/review_workflow.md) y los [criterios de anotación](configs/annotation_policy.md). Los informes de `reports/` describen su fecha de ejecución; algunos corresponden a etapas anteriores.

[Volver a la presentación del proyecto](../README.md).
