# CAMPUS22 RELEASE AND TRAINING FINAL

## COMPLETADO

- Clasificación individual de los 541 registros vacíos con evidencia y acción.
- Asignación única reproducible de los 285 grupos conflictivos, conservando metadatos originales.
- Workset versionado, colas visuales, controles de decisión y reportes de puertas/duplicados.
- Preflight V2 ejecutado y pruebas automatizadas ampliadas.

Validación: Python: 211 pruebas, 0 fallos y 0 errores; Flutter: `00:13 +69 ~1: All tests passed!`; análisis: `No issues found! (ran in 42.6s)`.

## PARCIAL

- Hay un subconjunto de 998 candidatos para un primer release; requiere decisiones humanas explícitas.
- Los grupos están resueltos en el plan, pero solo se materializarán en un release aprobado.

## BLOQUEADO

- `READY_FOR_TRAINING=false`: no hay anotaciones aprobadas ni cobertura independiente de las 22 clases en los tres splits.
- No se creó `campus22_release_v2`, no se perfiló batch y no comenzó YOLO26m.

## Respuestas

1. ¿Cuántos registros iniciales sin cajas? **541**
2. ¿Cuántos se resolvieron? **0**
3. ¿Cuántos necesitan revisión? **541**
4. ¿Cuántos grupos conflictivos había? **285**
5. ¿Cuántos se resolvieron? **285 en el plan; pendiente aplicar tras revisión**
6. ¿Existe leakage? **El plan tiene 0 grupos cruzados; aún no existe un release final para certificarlo.**
7. ¿Cuántas imágenes hay finalmente? **No hay release; workset: 24770 únicas.**
8. ¿Cuántas cajas? **Workset: 82541 propuestas.**
9. ¿Cuántas train? **No hay release final.**
10. ¿Cuántas val? **No hay release final.**
11. ¿Cuántas test? **No hay release final.**
12. ¿READY_FOR_TRAINING? **false**
13. ¿Se entrenó? **No.**
14. ¿Cuál es best.pt? **No existe best.pt Campus22.**
15. ¿Cuánto duró? **No aplica; no comenzó el entrenamiento.**
16. ¿Cuál fue el batch? **No medido: el profiler permanece cerrado por preflight.**
17. ¿Cuál fue mAP50? **No medido.**
18. ¿Cuál fue mAP50-95? **No medido.**
19. ¿Recall stairs? **No medido.**
20. ¿Recall pothole? **No medido.**
21. ¿Recall door? **No medido.**
22. ¿Recall glass_door? **No medido.**
23. ¿Recall curb_step? **No medido.**
24. ¿Recall ramp? **No medido.**
25. ¿Se integró best.pt con Depth? **No existe best.pt Campus22; la integración permanece preparada.**
26. ¿Depth sigue sin entrenamiento? **Sí.**
27. ¿Qué falta para probar metros reales? **Mediciones físicas y perfil validado por cámara/lente.**
28. ¿Qué falta para AWS? **Detector Campus22 evaluado, empaquetado y validación del endpoint.**
29. ¿Qué falta para APK? **Detector desplegado, perfil de cámara y recorrido físico identificado.**
30. ¿Qué acciones requieren intervención humana? **Revisar 998 candidatos, los 541 vacíos que se quieran recuperar, 1011 imágenes de puertas y 3 pares exactos con anotaciones distintas.**

## Siguiente acción exacta

Abrir `datasets/review_campus22_remaining/index.html`, iniciar el servidor local del paquete de
candidatos y registrar decisiones reales. Después aplicar las decisiones a una nueva carpeta,
ejecutar nuevamente este preflight y crear `campus22_release_v2` únicamente si pasa.
