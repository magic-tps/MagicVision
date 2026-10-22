# Criterios de anotación y liberación

IDs 0–9 se conservan; `trash_bin` es 10. `classes_v2.txt` solo corresponde a candidatos de once clases: no sustituir `labels.txt` del modelo operativo de diez clases.

- Persona: caja ajustada a la parte visible, con criterio consistente de oclusión. No inferir identidad.
- Escalera: caja del tramo visible, incluyendo el primer escalón si aparece. Etiquetar presencia; sentido ascendente/descendente requiere metadatos revisados, no se deduce del nombre `stairs`.
- Tacho: recipiente destinado a recibir residuos, incluidos tachos de reciclaje. Excluir macetas, cajas, barriles y basura suelta. «Waste container» de Open Images exige revisión porque puede incluir contenedores que exceden esta definición.
- Obstáculo: conservar el alcance explícito anterior (por ejemplo banca, hidrante, barrera); no convertir cualquier categoría desconocida ni tachos a `obstacle`.
- Negativo: debe revisarse respecto de las once clases. Una etiqueta vacía o incompleta no es un negativo confiable.

Los manifiestos contienen `review_status: needs_review` o `legacy10`. Tras revisar imagen y TODAS las clases, corregir el `.txt` YOLO y registrar `review_status: approved`, `coverage_complete: true`, `reviewer`, `reviewed_at` y, para negativos, `verified_negative: true`. No aprobar por puntuación del detector ni automáticamente las preanotaciones.

Añadir en cada registro `scene_tags` y `scene_tags_verified: true` cuando esté verificado: `person_near`, `person_far`, `person_occluded`, `stairs_up`, `stairs_down`, `stairs_partial`, `bin_small`, `bin_large`, `multi_object`. Para cercanía/lejanía documentar en `tag_evidence` cómo se obtuvo la categoría; los metros solo son válidos si se midieron. No deducir distancia física de la caja.

Guardar `group` por sesión/recorrido del campus; el particionador conserva el grupo completo y agrupa coincidencias exactas y pHash cercano. En fuentes públicas sin sesión, documentar `session_known: false`. El pHash reduce posibles fugas, pero no prueba independencia y puede agrupar imágenes distintas: revisar sus grupos.

Las imágenes públicas deben conservar licencia individual, autor y enlace de origen. Open Images advierte que no garantiza el estado legal de cada imagen; revisar procedencia antes de liberar. No hay acceso automático a Roboflow sin la clave/exportación y los términos del proyecto correspondientes.

Tras revisar procedencia, condiciones de uso y consentimiento si corresponde, registrar `provenance_reviewed: true`. El entrenador exige también responsable y fecha; aprobar la geometría de las cajas no sustituye esta revisión. Los derivados mantienen estos campos y su `derived_from`; no cuentan como nuevas escenas para el mínimo piloto.

Usar [review_workflow.md](review_workflow.md) para registrar estas decisiones y los hashes de imagen/etiquetas. El entrenador comprueba `reviewed_image_sha256` y `reviewed_label_sha256`; editar solo las banderas del manifiesto no completa una revisión. Las modificaciones posteriores invalidan los hashes y requieren revisar de nuevo.
