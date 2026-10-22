# Revisión específica: door y glass_door

Estado final de automatización: **NEEDS_HUMAN_REVIEW**. Ninguna imagen fue aprobada por heurística.

| Clase | Imágenes únicas | Cajas |
|---|---:|---:|
| door (ID 13) | 511 | 715 |
| glass_door (ID 14) | 500 | 2060 |

- Aprobadas mediante revisión explícita: 0.
- Corregidas automáticamente con evidencia geométrica: 0.
- Rechazadas explícitamente: 0.
- Imágenes únicas pendientes: 1011.
- Registros con indicadores geométricos: 161.

Indicadores: {"more_than_four_door_components": 131, "very_small_box": 66, "box_over_80_percent_image": 11}.

Las comprobaciones detectan cajas extremas, muchos componentes, coexistencia de `door`/`glass_door`
y geometría duplicada. No permiten decidir transparencia, reflejos, ventana frente a puerta,
puertas dobles o parcialmente abiertas. Las cajas de Trans10K derivadas de máscaras conservan su
procedencia y continúan pendientes de reconciliar por instancia. No se fusionaron componentes ni
se cambiaron clases automáticamente.

Evidencia completa: [estado JSON](door_glass_door_final_status.json),
[observaciones visuales](door_glass_door_visual_observations.md) y
[editor priorizado](../datasets/review_campus22_remaining/release_candidates/index.html).
