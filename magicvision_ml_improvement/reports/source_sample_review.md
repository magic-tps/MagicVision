# Inspección exploratoria de la fuente

Se inspeccionó visualmente el mosaico `source_sample.jpg`: ocho imágenes reales descargadas, cuatro seleccionadas por la etiqueta pública Stairs y cuatro por Waste container. Es una comprobación exploratoria de contenido, no una auditoría completa de cajas ni una aprobación para entrenar.

- `0001787dfadcf6bc`, `00162d431bdb61b5` y `0027cc0afe9b2a05` muestran recipientes para residuos de aspectos diferentes; no basta para asegurar la cobertura de todos los objetos presentes.
- `00019e09fe61747c` contiene puestos y recipientes pequeños dentro de una escena amplia: necesita inspección a resolución completa para confirmar cada caja y su clase.
- `00043633834f5a1c` y `0009399f91f86b40` muestran estructuras de escaleras con perspectivas distintas, incluyendo una escalera vista desde abajo. No representan necesariamente el primer escalón de un recorrido peatonal dentro de la universidad.
- `0003a351caf34972` y `0007ac467f3e0260` necesitan especial revisión de orientación, extensión del tramo visible y oclusiones. El mosaico no permite certificar esos límites.

Conclusión de esta muestra: hay contenido pertinente y variedad visual, pero se requiere revisión de anotaciones y complementarlo con recorridos reales del campus. Ninguna de estas ocho imágenes fue marcada como `approved`; no se infiere el sentido de subida/bajada como etiqueta del detector a partir de esta inspección.
