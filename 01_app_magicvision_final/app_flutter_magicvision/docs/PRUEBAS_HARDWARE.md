# Validación con teléfono, Arducam y AWS

Estas comprobaciones requieren el dispositivo físico y una API desplegada. Las pruebas unitarias y una compilación no certifican compatibilidad UVC/UAC del modelo concreto.

| Prueba | Resultado esperado |
|---|---|
| Arrancar sin Arducam | Cámara desconectada; voz, contactos/SOS e historial siguen disponibles |
| Conectar OTG | Un permiso USB; preview local tras abrir UVC |
| Denegar permiso y reintentar | Error accesible y botón Reconectar; no bucle de diálogos |
| Desconectar/reconectar repetidamente | Libera interfaz/superficie; reabre sin reiniciar app |
| Pasar a segundo plano y volver | Cámara liberada/reabierta; no se narra una respuesta antigua |
| Pulsar Que ves varias veces | Una única captura/inferencia activa |
| Endpoint en cero instancias | Anuncia preparación, mantiene UI disponible y captura solo después de estar listo |
| Cancelar durante preparación/consulta | Botón y «cancelar análisis» descartan el trabajo; no se narra una respuesta posterior |
| Escena tarda más de 15 segundos | Caduca y solicita nuevo análisis; no usa el resultado viejo |
| Analizar una escena | JPEG reducido, mismo UUID en API/log, CERM/TTS/historial |
| Mover cámara mientras llega respuesta | Las cajas corresponden a la captura, no al preview posterior |
| Desconectar durante captura/subida | Descarta respuesta; error, nunca camino libre simulado |
| Modo avión, API lenta, HTTP 4xx/5xx | Error accesible, interfaz operativa, se puede reintentar |
| Captura sin objetos | Respuesta válida con lista vacía; regla original CERM |
| OCR y VLM | Captura desde UVC, consumo del archivo temporal y eliminación al terminar |
| Micrófonos L/R | Comprobar enumeración Android y reconocimiento aislando acústicamente teléfono/USB |
| Desenchufar audio durante STT | Android recupera entrada predeterminada; comandos funcionan |
| SAVE_DEBUG_IMAGES=false | Solo JSONL; no JPEG persistentes de inferencia |
| SAVE_DEBUG_IMAGES=true | Se conserva JPEG reducido para experimento explícito |

El puente utiliza el preview UVC reducido negociado por la biblioteca, inicialmente 640x480. Verificar la orientación física de montaje y que el formato anunciado por la cámara sea compatible. El sensor IMX219 no determina VID/PID, formato UVC ni compatibilidad del firmware del adaptador.

Android puede reconocer un micrófono UAC y enrutarlo automáticamente. `speech_to_text` no expone selección explícita de AudioDeviceInfo ni `EXTRA_AUDIO_SOURCE`; por tanto la enumeración USB no confirma el micrófono efectivo. Si el servicio de reconocimiento no usa ese dispositivo, se conserva el STT con la ruta predeterminada; no se fuerza una entrada abriendo otro AudioRecord. El fallback interno y la ruta real deben comprobarse en el modelo de teléfono/servicio de reconocimiento utilizado.

Registrar teléfono, versión Android, VID/PID, firmware/formatos UVC, orientación de montaje, red, región, instancia SageMaker, versión de imagen ECR y checksum del modelo. Las bibliotecas nativas UVC/Cactus también necesitan validación de páginas de 16 KiB para distribución en dispositivos que las exijan.
