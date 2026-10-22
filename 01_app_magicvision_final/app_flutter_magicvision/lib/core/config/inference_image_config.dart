class InferenceImageConfig {
  const InferenceImageConfig({
    this.maxWidth = 640,
    this.maxHeight = 640,
    this.jpegQuality = 80,
    this.timeout = const Duration(seconds: 10),
    this.warmupTimeout = const Duration(minutes: 15),
    this.maxCaptureAge = const Duration(seconds: 15),
  });

  factory InferenceImageConfig.environment() => const InferenceImageConfig(
    maxWidth: int.fromEnvironment('INFERENCE_MAX_WIDTH', defaultValue: 640),
    maxHeight: int.fromEnvironment('INFERENCE_MAX_HEIGHT', defaultValue: 640),
    jpegQuality: int.fromEnvironment('JPEG_QUALITY', defaultValue: 80),
    timeout: Duration(
      seconds: int.fromEnvironment('API_TIMEOUT_SECONDS', defaultValue: 10),
    ),
    warmupTimeout: Duration(
      seconds: int.fromEnvironment(
        'API_WARMUP_TIMEOUT_SECONDS',
        defaultValue: 900,
      ),
    ),
    maxCaptureAge: Duration(
      seconds: int.fromEnvironment('MAX_CAPTURE_AGE_SECONDS', defaultValue: 15),
    ),
  );

  final int maxWidth;
  final int maxHeight;
  final int jpegQuality;
  final Duration timeout;
  final Duration warmupTimeout;
  final Duration maxCaptureAge;
  static const apiBaseUrl = String.fromEnvironment('API_BASE_URL');
  static const saveDebugImages = bool.fromEnvironment(
    'SAVE_DEBUG_IMAGES',
    defaultValue: false,
  );
  // Reserved for a future explicit scheduler; no automatic uploads in this version.
  static const continuousAnalysis = false;

  void validate() {
    if (maxWidth <= 0 ||
        maxHeight <= 0 ||
        maxWidth > 1280 ||
        maxHeight > 1280 ||
        jpegQuality < 70 ||
        jpegQuality > 95 ||
        timeout <= Duration.zero ||
        warmupTimeout <= Duration.zero ||
        maxCaptureAge <= Duration.zero ||
        maxCaptureAge > const Duration(seconds: 30)) {
      throw ArgumentError('Configuracion de imagen o timeout invalida');
    }
  }
}
