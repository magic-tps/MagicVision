# Inspección visual de la muestra de puertas

Se inspeccionó `door_glass_door_sample.jpg`, 16 ejemplos seleccionados por heurísticas.
Esto no representa una revisión completa ni aprueba sus etiquetas.

- Open Images, IDs terminados en `396fca7def3db14c`, `64e92764acdc519c` y
  `66db7e2ddb478bf8`: puertas con paneles acristalados etiquetadas como door.
  Revisar clasificación y definición de instancias en accesos con varios paneles.
- `3ff3af7e69742066`: entradas lejanas detrás de una cerca; cajas pequeñas que
  requieren verificar visibilidad, no descarte automático por tamaño.
- `285c33df583022d1`: collage de puertas; revisar pertinencia para escenas del campus.
- `bf87b83607a69736` y `cd1c44747f5249eb`: primeros planos de herrajes/superficies;
  comprobar que se delimita la puerta y no solo una pieza.
- Trans10K, sufijos `00386`, `00407`, `00479`, `00538`, `00602`: divisiones opacas,
  barras o letras separan una puerta en varios componentes en las máscaras. Las
  cajas derivadas necesitan reconciliación por instancia visible.

No se borraron imágenes, cambiaron cajas ni registraron aprobaciones humanas.
Siguiente acción: corregir estas muestras en el editor y revisar todas las clases
presentes, conservando procedencia y hash de las etiquetas aprobadas.
