# Aplicación MagicVision

MagicVision permite consultar el entorno mediante una cámara y escuchar la respuesta en español. Reúne reconocimiento de objetos, descripción de escenas, lectura de texto y comandos de voz en una aplicación Android.

## Uso básico

1. Conecta la cámara USB prevista para esta configuración y concede los permisos que solicite Android.
2. Pulsa el botón de análisis o di «qué ves» para consultar la imagen actual.
3. Espera la respuesta en pantalla y por voz. Puedes cancelar el análisis mientras está en curso.
4. Para leer un texto, di «leer», apunta la cámara y después di «leer ahora».

También puedes pedir una descripción visual más completa, consultar tu ubicación, gestionar contactos y crear recordatorios. La [guía de comandos](GUIA_COMANDOS.md) reúne las frases disponibles.

El análisis de objetos utiliza un servicio remoto y necesita conexión. La descripción visual adicional depende de la disponibilidad de su propio modelo. La configuración predeterminada usa cámara USB; la cámara interna se puede seleccionar al compilar con `CAMERA_SOURCE=internal`.

## Ejecutar el proyecto

Se necesita Flutter, un entorno de desarrollo Android y acceso al servicio de reconocimiento. Desde esta carpeta:

```powershell
Copy-Item .env.example .env
flutter pub get
flutter run --dart-define-from-file=.env
```

Antes de ejecutar, completa `API_BASE_URL` en `.env` con la dirección del servicio. Sin un servicio configurado, la app no podrá analizar objetos.

Para compilar con el modelo visual local, restaura todos sus archivos en `assets/mobilevlm/lfm2-vl-1.6b/`. Los pesos se distribuyen por separado del repositorio. El script `scripts/install_mobilevlm_model.ps1 -DownloadOnly` permite preparar la descarga en `.mobilevlm-cache/`; después hay que copiar el contenido completo del modelo a la carpeta de assets.

## Estado y comprobaciones

El detector de referencia reconoce diez categorías. La ampliación a 22 categorías se desarrolla en la [carpeta de mejora del reconocimiento](../../magicvision_ml_improvement/README.md).

Las pruebas locales no confirman por sí solas el funcionamiento de una APK instalada, la cámara física o el servicio remoto. La [guía de pruebas con hardware](docs/PRUEBAS_HARDWARE.md) describe las comprobaciones pendientes en esos componentes.

Para mantener o desplegar el sistema, consulta el [funcionamiento interno](ARQUITECTURA_APLICACION.md), la [guía del servicio AWS](docs/aws-deployment.md) y la [validación funcional](docs/VALIDACION_FUNCIONAL_ASYNC.md).

[Volver a la presentación del proyecto](../../README.md).
