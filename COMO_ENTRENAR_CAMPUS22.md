# Entrenar el modelo de 22 categorías

El entrenamiento permite que el detector aprenda a reconocer las categorías previstas para el campus, como personas, escaleras, tachos, puertas y rampas. Se realiza en la computadora y sus resultados se revisan antes de usarlos en la app.

## Antes de empezar

Necesitas Python, las dependencias de [requirements.txt](magicvision_ml_improvement/requirements.txt) y una instalación de PyTorch compatible con la GPU. También deben estar disponibles los siguientes archivos locales, que no se incluyen en Git:

- Las imágenes y etiquetas de `magicvision_ml_improvement/datasets/campus22_full_existing_v1/`.
- El modelo inicial `magicvision_ml_improvement/models/yolo26m.pt`.

El conjunto preparado contiene **24.226 imágenes**: 16.570 para aprender, 5.238 para comparar el rendimiento durante el entrenamiento y 2.418 para la evaluación final. Incorpora 998 imágenes revisadas por personas; las demás conservan sus etiquetas de origen, que pueden estar incompletas.

## Comprobar los archivos

Abre PowerShell en la raíz del proyecto y ejecuta:

```powershell
.\ENTRENAR_CAMPUS22.cmd --check
```

Esta opción revisa los datos sin iniciar el entrenamiento. Si muestra un error, hay que resolverlo antes de continuar.

## Iniciar el entrenamiento

Haz doble clic en `ENTRENAR_CAMPUS22.cmd` o ejecuta:

```powershell
.\ENTRENAR_CAMPUS22.cmd
```

El programa comprueba los archivos y la GPU antes de empezar. Está configurado para un máximo de 100 pasadas por los datos y puede terminar antes si deja de mejorar. La consola muestra el avance, el tiempo transcurrido y una estimación del tiempo restante.

## Encontrar los resultados

Cada ejecución crea una carpeta dentro de `magicvision_ml_improvement/training/runs/`. Allí se guardan el resumen de tiempos y, cuando se generan, el modelo seleccionado (`weights/best.pt`) y el último estado (`weights/last.pt`).

Los resultados de este conjunto son **experimentales**: no todas las imágenes están completamente revisadas y los extintores carecen de ejemplos independientes en los grupos de validación y prueba.

Al terminar, corresponde evaluar el modelo con las imágenes reservadas para la prueba final. El entrenamiento no publica el modelo ni reemplaza el que utiliza la aplicación.
