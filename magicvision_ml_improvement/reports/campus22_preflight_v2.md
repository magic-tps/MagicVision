# Campus22 preflight V2

READY_FOR_TRAINING = **false**

- Inventario: 24773 registros, 24770 imágenes únicas y 82541 cajas/propuestas.
- Vacíos: 541; resueltos objetivamente: 0; revisión humana: 541.
- Conflictos de grupo: 285 iniciales; 285 con asignación única planificada; leakage planificado: 0.
- Candidatos compactos para el primer release: 998; aprobaciones completas: 0.
- Clases ausentes en el subconjunto: {"train": [], "val": ["fire_extinguisher"], "test": ["construction_barrier", "cone", "fire_extinguisher"]}.
- Release V2 creado: false.

## Bloqueos

- 998 release candidates lack complete explicit review.
- Candidate subset lacks independent group coverage for some classes/splits.
- campus22_release_v2 has not been created because review gates fail.

La asignación de grupos está resuelta como plan reproducible, pero no sustituye revisar cobertura y
procedencia. No se dividieron grupos `unknown_session` para fabricar representación en validation/test.
El entrenamiento no se ejecutó.
