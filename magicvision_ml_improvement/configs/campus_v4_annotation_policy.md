# Criterios de anotación de la ampliación universitaria V4

Contrato de adquisición: `magicvision-campus-v4-25`, en `classes_campus_v4.txt`.
Los IDs 0–10 se mantienen. Este contrato no cambia las clases del modelo desplegado.

| ID | Clase | Qué anotar y qué revisar |
|---:|---|---|
| 11 | ramp | Superficie inclinada transitable; no deducir pendiente ni accesibilidad exacta. |
| 12 | curb_step | Desnivel visible, borde de vereda o escalón aislado. Conservar subtipo de origen. Escaleras completas siguen en `stairs`. |
| 13 | door | Puerta sin evidencia suficiente para `glass_door`. La clase Door de Open Images aún mezcla materiales. |
| 14 | glass_door | Puerta de vidrio reconocible como puerta; no ventana ni pared de vidrio. Reasignar la caja de door, no duplicarla sobre el mismo objeto. |
| 15 | chair | Silla, incluyendo las parcialmente ocultas. No convertir automáticamente bancas. |
| 16 | table_desk | Mesa o escritorio. Revisar solapamientos de Table/Desk de la fuente, conservar una caja por objeto. |
| 17 | backpack | Mochila visible. Estar en el suelo es contexto adicional, no un atributo garantizado por la clase. |
| 18 | cable | Cable visible con caja ajustada al tramo; revisar separación de cables en máscaras semánticas y registrar oclusión. |
| 19 | wet_floor | Región visible de líquido o derrame sobre una superficie transitable. No un cartel de piso mojado, reflejo o residuo seco. |
| 20 | construction_barrier | Barrera temporal de obra; no cerca permanente, baranda o cualquier muro. |
| 21 | handrail | Pasamanos o barra de apoyo para escalera/rampa. La fuente Wheelmap no cubre todos los tipos de barandas horizontales. |
| 22 | elevator | Acceso/cabina reconocible de ascensor. Un botón o señal aislados no acreditan la presencia visible del ascensor completo. |
| 23 | cone | Cono de señalización; revisar las cajas de PandaSet porque su definición de Cones incluye algunos postes cortos. |
| 24 | fire_extinguisher | Extintor visible. Su pictograma, gabinete vacío o señal no sustituyen el objeto. |

Objetos diversos en el suelo se revisan contra el criterio existente de `obstacle`, sin inventar una clase ilimitada de «cualquier peligro». Una detección de rampa, silla o mochila no determina por sí sola peligro ni distancia.

Antes de aprobar una imagen, revisar todas las 25 categorías, no solo la que motivó su descarga. Una fuente con cajas de extintores puede dejar sin etiquetar las personas, puertas y sillas del mismo fotograma. Un TXT vacío de un candidato pendiente tampoco es un negativo confirmado.

Las cajas derivadas de máscaras de vidrio y cables son propuestas geométricas. Componentes separados pueden pertenecer al mismo objeto y una región conectada puede juntar varios. Conservar la máscara, corregir instancias y límites, y registrar la decisión humana. Las fotos clasificadas como derrame o ascensor todavía necesitan cajas.

Conservar autor, licencia, enlace, imagen original/variante, anotación original, hashes y grupo. Las fuentes sintéticas se identifican como `synthetic=true` y no deben usarse para afirmar rendimiento en fotos reales del campus.

Separar sesiones, ubicaciones y grupos de imágenes parecidas antes de dividir train/val/test. Las particiones de origen y sus conflictos se conservan; no constituyen una división final de MagicVision. Los grupos por tarea de anotación o fuente completa son conservadores y necesitan revisión antes de crear una división útil.

El editor `review_dataset.py --taxonomy campus-v4` admite los 25 IDs. El entrenamiento y `preflight.py` anteriores siguen siendo de 11 clases: no se les debe pasar esta ampliación. Tras completar revisión y división, hace falta una configuración de entrenamiento de 25 clases y evaluar en sesiones reales reservadas antes de actualizar modelo, etiquetas de la app, API o mensajes.
