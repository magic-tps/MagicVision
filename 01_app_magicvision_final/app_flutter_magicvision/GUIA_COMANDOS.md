# Guia de comandos de MagicVision

Esta guia resume las frases que puedes decirle a la app. No tienes que repetirlas exactamente igual, pero conviene usar frases parecidas para que el reconocimiento de voz entienda mejor.

## Control de voz

Usa estos comandos cuando la app esta hablando o cuando quieres controlar la sesion:

- `alto`
- `cancelar`
- `silencio`
- `para`
- `detente`
- `repite`
- `comandos`
- `que puedes hacer`

`alto`, `cancelar` y `silencio` cortan la voz de la app. `repite` vuelve a decir la ultima lectura o respuesta importante.

## Vision rapida con YOLO26m

Este modo captura una imagen de la Arducam UVC y utiliza YOLO26m completo mediante inferencia cloud en AWS SageMaker. Requiere conexión a la API; no realiza vigilancia continua.

- `que ves`
- `que hay`
- `describe`
- `reconocer entorno`
- `reconocer alrededor`
- `ver peligro`
- `peligro`
- `bache`
- `hueco`
- `grieta`

Ejemplo de respuesta:

`Se detecta una persona cerca. Avanza lentamente y manten distancia. Accion sugerida: Avanza con precaucion.`

El modelo reconoce estas clases:

- `person`
- `car`
- `motorcycle`
- `bus`
- `bicycle`
- `stairs`
- `pothole`
- `pole`
- `traffic_sign`
- `obstacle`

Cuando presionas `Que ves`, la app captura una imagen, envía un JPEG reducido a la API, recibe detecciones de SageMaker y evalua riesgo, muestra objetos detectados, recomendacion, accion sugerida, confianza y fuente.

## Descripcion visual con LFM

Este modo da una descripcion mas completa de la foto actual. MagicVision usa dos caminos:

- Si existe `MAGICVISION_VLM_ENDPOINT`, usa `LiquidAI/LFM2.5-VL-450M` por endpoint compatible con OpenAI.
- Si no hay endpoint, usa el VLM local fuerte `lfm2-vl-1.6b`, incluido en la app.

La app usa YOLO26m como fuente confiable de detecciones y riesgo. El VLM agrega contexto visual, pero su respuesta queda anclada a lo que YOLO confirma para evitar objetos inventados. El modelo local 1.6B es pesado; la primera preparacion puede tardar varios minutos.

Comandos:

- `analiza imagen`
- `analiza la imagen`
- `modelo visual`
- `modelo de vision`
- `vision inteligente`
- `estado modelo visual`
- `descargar modelo visual`

Usa `analiza imagen` cuando quieras una descripcion mas completa de la foto actual. Usa `estado modelo visual` para comprobar si LFM esta listo. Usa `descargar modelo visual` si quieres forzar la preparacion del modelo local.

Flujo esperado:

1. La app responde rapido con YOLO26m y la recomendacion de riesgo.
2. Si LFM esta listo, captura la foto y genera una descripcion completa.
3. Si LFM aun no esta listo, inicia la preparacion del modelo local fuerte y mantiene la respuesta rapida como respaldo.

Si la descripcion visual tarda mas de unos segundos o no responde, la app evita leer errores tecnicos y conserva la descripcion rapida de respaldo. Esto mantiene la respuesta util para caminar o revisar el entorno sin esperar demasiado.

## Lectura de texto

Usa estos comandos para OCR:

- `leer`
- `lectura`
- `leer ahora`
- `lee ahora`
- `capturar`
- `escanea`

Flujo recomendado:

1. Di `leer`.
2. Apunta al texto.
3. Di `leer ahora`.

## Ubicacion y rutas

Comandos para saber donde estas o iniciar una ruta:

- `dime donde estoy`
- `mi ubicacion`
- `donde estoy`
- `quiero ir a una farmacia`
- `guiame a Plaza de Armas`
- `como llegar al hospital`
- `abrir Maps`
- `abrir ruta en Maps`
- `cancelar ruta`

La app intenta darte rumbo y distancia. Para navegacion peatonal exacta, usa `abrir Maps`.

## Hora y fecha

- `que hora es`
- `hora`
- `que fecha es`
- `que dia es`

## Recordatorios y temporizadores

- `recuerdame tomar agua en diez minutos`
- `recordatorio medicina en una hora`
- `temporizador en cinco minutos`
- `alarma en diez minutos`
- `mis recordatorios`
- `recordatorios pendientes`
- `cancelar recordatorios`
- `que tengo guardado`

## Contactos y SOS

Primero guarda contactos:

- `quiero guardar contacto`
- `agrega contacto mama numero nueve ocho siete seis cinco cuatro tres dos uno`
- `agrega contacto papa numero 987654321`
- `mis contactos`
- `contacto SOS`

Flujo por voz recomendado para Peru:

1. Di `quiero guardar contacto`.
2. La app pregunta el nombre. Responde, por ejemplo: `mama`.
3. La app pregunta el numero. Di solo los 9 digitos del celular peruano, separados: `nueve ocho siete seis cinco cuatro tres dos uno`.
4. La app agrega automaticamente `+51`.
5. La app pregunta si sera SOS. Responde `si` o `no`.

Luego puedes llamar o escribir:

- `llamar a mama`
- `llamar a mama por WhatsApp`
- `mensaje a mama`
- `SOS`

`SOS` intenta usar tu contacto de emergencia si lo tienes guardado.

## Historial

- `historial`
- `ultimos comandos`
- `que tengo guardado`

El historial ayuda a revisar que frases entendio la app.

## Consejos para que entienda mejor

- Di comandos cortos: `analiza imagen`, `que ves`, `leer ahora`.
- Espera medio segundo despues de que la app termine de hablar.
- Evita hablar encima de la voz de la app salvo para decir `alto`, `cancelar` o `silencio`.
- Si la descripcion visual tarda, usa `que ves` para una respuesta rapida. La primera preparacion del VLM local puede tardar por el tamano del modelo 1.6B.
- Si el reconocedor de voz falla mucho, instala o activa el idioma espanol offline en el teclado/reconocimiento de voz del telefono.
