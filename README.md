# MagicVision

MagicVision es un proyecto de tesis que desarrolla una aplicación Android de asistencia visual. Permite consultar qué aparece frente a la cámara, escuchar una descripción del entorno y leer textos en voz alta.

La idea es facilitar el acceso a información del entorno mediante botones y comandos de voz.

## Qué permite hacer

- Consultar los objetos reconocidos en una imagen y escuchar una orientación sobre posibles obstáculos.
- Pedir una descripción más detallada de una escena.
- Leer textos de carteles, documentos u otros elementos visibles.
- Consultar la ubicación, abrir rutas, gestionar contactos y usar un contacto SOS.
- Crear recordatorios y revisar el historial de comandos y resultados.

La disponibilidad de estas funciones depende de la configuración, los permisos y los servicios del teléfono.

## Cómo funciona

Al pedir «qué ves», la app toma una fotografía y la envía al servicio de reconocimiento. Con los objetos encontrados, prepara una respuesta que se muestra en pantalla y se lee en voz alta. El análisis de objetos necesita conexión con el servicio configurado.

La lectura de texto se solicita por separado. También existe una opción para ampliar la descripción visual de la captura, que depende del modelo visual disponible.

## Qué hay en este repositorio

| Carpeta | Contenido |
| --- | --- |
| [Aplicación](01_app_magicvision_final/app_flutter_magicvision/README.md) | App Android, servicio de reconocimiento e instrucciones de ejecución. |
| [Mejora del reconocimiento](magicvision_ml_improvement/README.md) | Herramientas para preparar imágenes, revisar etiquetas y entrenar nuevos modelos. |
| [Material del paper](02_paper_metricas_modelo_final/README.md) | Tablas, resultados y figuras del experimento de referencia. |

## Estado del proyecto

El detector de referencia reconoce diez categorías. El proyecto incluye una ampliación experimental a 22 categorías para entornos universitarios. Preparar más imágenes o iniciar un entrenamiento no cambia automáticamente el modelo que utiliza la app.

Las cifras del paper pertenecen al experimento de referencia. La ampliación requiere completar el entrenamiento, evaluar sus resultados y comprobar el funcionamiento en el teléfono y en el servicio remoto.

La app puede omitir objetos o interpretar una escena de forma incompleta. El [diagnóstico de detección](MAGICVISION_DIAGNOSTICO_DETECCION.md) explica estas limitaciones con más detalle.

## Para empezar

Consulta la [guía de la aplicación](01_app_magicvision_final/app_flutter_magicvision/README.md) para ejecutarla o la [guía de entrenamiento](COMO_ENTRENAR_CAMPUS22.md) para trabajar con las 22 categorías.

Los datasets, los pesos de los modelos y las configuraciones privadas se conservan por separado. Al clonar el repositorio, hay que restaurar los archivos necesarios para la tarea que se quiera realizar.
