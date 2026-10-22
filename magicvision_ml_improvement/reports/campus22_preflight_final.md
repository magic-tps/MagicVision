# Auditoría final campus22

READY_FOR_TRAINING = **false**

- Registros: 24773; imágenes únicas: 24770.
- Cajas/propuestas: 82541.
- Errores de etiquetas/archivos: 0; errores de píxeles/hash: 0.
- Imágenes sin cajas: 541 (no son negativos aprobados).
- Aprobaciones completas: 0.
- Grupos con particiones de origen diferentes: 285. No se resolvió leakage mediante barajado.
- Cero imágenes liberadas en train/val/test; las 22 clases están ausentes del release aprobado.

Se comprobaron existencia, SHA256, decodificación, dimensiones, etiquetas, IDs, límites, duplicados exactos y pHash recalculado. Se agruparon secuencias/origen cuando esos metadatos existen; no se infirieron sesiones independientes de fuentes sin metadatos. Las anotaciones sospechosas son una cola de revisión, no errores semánticos confirmados.

La corrección automática fue el nombre de ID 10 a trash_can, conservando ID y cajas. No se borraron datos ni se inventaron anotaciones. Falta completar revisión de las 22 clases y resolver grupos antes de liberar un split.

Evidencia detallada: [JSON](campus22_preflight_final.json), [revisión de puertas](door_glass_door_quality.md).
