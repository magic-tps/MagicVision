import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:uuid/uuid.dart';
import '../../camera/camera_source.dart';
import '../config/inference_image_config.dart';
import '../models/analysis_result.dart';
import '../models/detection_response.dart';
import 'aws_object_detection_service.dart';
import 'cerm_detection_mapper.dart';
import 'experimental_logger.dart';
import 'inference_exception.dart';
import 'inference_image_processor.dart';
import 'inference_readiness_service.dart';
import 'object_detection_service.dart';
import 'risk_assessment_service.dart';
import 'capture_archive.dart';
import 'frame_detection_service.dart';

class AnalyzedCapture {
  const AnalyzedCapture({
    required this.image,
    required this.response,
    required this.analysis,
    required this.capturedAt,
  });
  final PreparedInferenceImage image;
  final DetectionResponse response;
  final AnalysisResult analysis;
  final DateTime capturedAt;
  String get id => response.requestId;
}

enum InferenceState {
  cameraDisconnected,
  cameraConnecting,
  cameraReady,
  preparing,
  capturing,
  uploading,
  processing,
  resultReady,
  error,
  cancelled,
}

class VisionAnalysisService extends ChangeNotifier {
  VisionAnalysisService({
    required this.camera,
    ObjectDetectionService? objectDetectionService,
    RiskAssessmentService? riskAssessmentService,
    InferenceImageConfig? config,
    ExperimentalLogger? logger,
    CaptureArchive? captureArchive,
  }) : _detector =
           objectDetectionService ?? AwsObjectDetectionService(config: config),
       _risk = riskAssessmentService ?? RiskAssessmentService(),
       _images = InferenceImageProcessor(
         config ?? InferenceImageConfig.environment(),
       ),
       _config = config ?? InferenceImageConfig.environment(),
       _logger = logger ?? FileExperimentalLogger(),
       _captureArchive = captureArchive ?? const CaptureArchive() {
    camera.addListener(_cameraChanged);
  }
  final CameraSource camera;
  final ObjectDetectionService _detector;
  final RiskAssessmentService _risk;
  final InferenceImageProcessor _images;
  final InferenceImageConfig _config;
  final ExperimentalLogger _logger;
  final CaptureArchive _captureArchive;
  AnalyzedCapture? lastCapture;
  InferenceState state = InferenceState.cameraDisconnected;
  String? errorMessage;
  DetectionResponse? lastResponse;
  PreparedInferenceImage? lastImage;
  bool isInferenceRunning = false;
  bool _disposed = false;
  bool _serviceReady = false;
  int _generation = 0;
  AnalysisResult? _lastAssessment;
  Stopwatch? _resultAge;

  bool isResultCurrent(AnalysisResult result) =>
      !_disposed &&
      camera.status == CameraStatus.ready &&
      identical(result, _lastAssessment) &&
      lastResponse != null &&
      _resultAge != null &&
      _resultAge!.elapsed <= _config.maxCaptureAge;

  void _setState(InferenceState value) {
    if (_disposed) return;
    state = value;
    notifyListeners();
  }

  Future<void> initialize() => camera.initialize();
  void _cameraChanged() {
    if (camera.status != CameraStatus.ready) {
      _serviceReady = false;
      _generation++;
      _detector.cancel();
      lastResponse = null;
      lastImage = null;
      lastCapture = null;
    }
    errorMessage = camera.errorMessage;
    _setState(switch (camera.status) {
      CameraStatus.ready =>
        isInferenceRunning ? state : InferenceState.cameraReady,
      CameraStatus.connecting => InferenceState.cameraConnecting,
      CameraStatus.error => InferenceState.error,
      CameraStatus.disconnected => InferenceState.cameraDisconnected,
    });
  }

  Future<AnalysisResult?> analyzeCurrentFrame() async {
    if (_disposed || isInferenceRunning) return null;
    isInferenceRunning = true;
    lastResponse = null;
    lastImage = null;
    lastCapture = null;
    errorMessage = null;
    _lastAssessment = null;
    final generation = _generation;
    final id = const Uuid().v4();
    final total = Stopwatch()..start();
    final network = Stopwatch();
    final captureAge = Stopwatch();
    final preparation = Stopwatch();
    final event = <String, Object?>{
      'request_id': id,
      'timestamp': DateTime.now().toUtc().toIso8601String(),
      'camera_source': camera.name,
      'original_resolution': null,
      'inference_resolution': null,
      'jpeg_size_bytes': null,
      'network_latency_ms': null,
      'cloud_inference_ms': null,
      'total_processing_ms': null,
      'detections_count': null,
      'risk_level': null,
      'recommendation': null,
    };
    PreparedInferenceImage? image;
    AnalysisResult? assessment;
    DateTime? capturedAt;
    try {
      _checkCurrent(generation);
      final detector = _detector;
      if (detector is InferenceReadinessService && !_serviceReady) {
        _setState(InferenceState.preparing);
        preparation.start();
        await (detector as InferenceReadinessService).ensureReady();
        preparation.stop();
        _checkCurrent(generation);
        _serviceReady = true;
      }
      event['service_preparation_ms'] = preparation.elapsedMilliseconds;
      _setState(InferenceState.capturing);
      captureAge.start();
      capturedAt = DateTime.now().toUtc();
      final bytes = await camera.captureFrame();
      _checkCurrent(generation);
      image = await _images.prepare(bytes);
      _checkCurrent(generation);
      event.addAll({
        'original_resolution': {
          'width': image.originalWidth,
          'height': image.originalHeight,
        },
        'inference_resolution': {'width': image.width, 'height': image.height},
        'jpeg_size_bytes': image.bytes.length,
      });
      _setState(InferenceState.uploading);
      network.start();
      void onProcessing() {
        if (!_disposed && generation == _generation) {
          _setState(InferenceState.processing);
        }
      }

      final response = _detector is FrameDetectionService
          ? await (_detector as FrameDetectionService).detectFrame(
              image.bytes,
              requestId: id,
              capturedAt: capturedAt,
              cameraId:
                  const String.fromEnvironment('CAMERA_PROFILE_ID').isNotEmpty
                  ? const String.fromEnvironment('CAMERA_PROFILE_ID')
                  : camera.name,
              onProcessing: onProcessing,
            )
          : await _detector.detect(
              image.bytes,
              requestId: id,
              onProcessing: onProcessing,
            );
      network.stop();
      _checkCurrent(generation);
      _checkAge(captureAge);
      if (response.requestId != id ||
          response.imageWidth != image.width ||
          response.imageHeight != image.height) {
        throw const InferenceException(
          'invalid_response',
          'Respuesta incompatible con la captura',
        );
      }
      _setState(InferenceState.processing);
      final input = const CermDetectionMapper().map(response);
      final result = _risk.assess(input.detections, source: 'aws_sagemaker');
      lastResponse = response;
      lastImage = image;
      try {
        await _captureArchive.record(
          captureId: id,
          original: bytes,
          detector: image.bytes,
          metadata: {
            'captured_at': capturedAt.toIso8601String(),
            'camera_source': camera.name,
            'original_resolution': event['original_resolution'],
            'original_resolution_reference': 'after_exif_orientation',
            'stored_original_resolution': {
              'width': image.encodedOriginalWidth,
              'height': image.encodedOriginalHeight,
            },
            'inference_resolution': event['inference_resolution'],
            'orientation': image.originalHeight > image.originalWidth
                ? 'portrait'
                : 'landscape',
            'exif_orientation': image.exifOrientation,
            'model_version': response.modelVersion,
            'frame_id': response.frameId,
            'camera_profile_id': response.cameraId,
            'depth_model_sha256': response.depthModelSha256,
            'pixel_sha256': response.pixelSha256,
            'performance': response.performance,
            'detections': response.detections.map((d) => d.toJson()).toList(),
          },
        );
      } catch (_) {
        debugPrint('No se pudo archivar la captura de investigacion');
      }
      event.addAll({
        // HTTP round trip includes backend overhead and cloud computation.
        'network_latency_ms': network.elapsedMilliseconds,
        'cloud_inference_ms': response.processingTimeMs,
        'detections_count': input.objectCount, 'risk_level': result.riskLevel,
        'recommendation': result.recommendation, 'status': 'success',
        'model_version': response.modelVersion,
        'captured_at': capturedAt.toIso8601String(),
        'detections': response.detections.map((d) => d.toJson()).toList(),
        'cerm_description': result.description,
        'cerm_speech_text': result.speechText,
      });
      assessment = result;
    } catch (error) {
      _serviceReady = false;
      final failure = error is InferenceException
          ? error
          : const InferenceException(
              'unexpected',
              'No se pudo analizar la imagen',
            );
      errorMessage = failure.message;
      event.addAll({'status': 'error', 'error_code': failure.code});
      if (!_disposed) {
        _setState(
          failure.code == 'cancelled'
              ? InferenceState.cancelled
              : camera.status == CameraStatus.disconnected
              ? InferenceState.cameraDisconnected
              : InferenceState.error,
        );
      }
      return null;
    } finally {
      network.stop();
      total.stop();
      event['total_processing_ms'] = total.elapsedMilliseconds;
      event['capture_age_ms'] = captureAge.elapsedMilliseconds;
      if (network.elapsedMicroseconds > 0) {
        event['network_latency_ms'] = network.elapsedMilliseconds;
      }
      try {
        await _logger.record(event, debugImage: image?.bytes);
      } catch (_) {
        debugPrint('No se pudo escribir el registro experimental');
      }
      isInferenceRunning = false;
      if (!_disposed) notifyListeners();
    }
    // Persistence is asynchronous too: a detach during logging must invalidate
    // delivery to the UI/TTS just like a detach during HTTP.
    if (_disposed ||
        generation != _generation ||
        camera.status != CameraStatus.ready ||
        captureAge.elapsed > _config.maxCaptureAge) {
      if (!_disposed &&
          generation == _generation &&
          camera.status == CameraStatus.ready) {
        lastResponse = null;
        lastImage = null;
        errorMessage = 'La captura caduco. Analiza de nuevo';
        _setState(InferenceState.error);
      }
      return null;
    }
    _lastAssessment = assessment;
    _resultAge = captureAge;
    lastCapture = AnalyzedCapture(
      image: image,
      response: lastResponse!,
      analysis: assessment,
      capturedAt: capturedAt,
    );
    _setState(InferenceState.resultReady);
    return assessment;
  }

  void _checkCurrent(int generation) {
    if (_disposed || camera.status != CameraStatus.ready) {
      throw const InferenceException(
        'camera_disconnected',
        'Camara desconectada',
      );
    }
    if (generation != _generation) {
      throw const InferenceException('cancelled', 'Analisis cancelado');
    }
  }

  void _checkAge(Stopwatch age) {
    if (age.elapsed > _config.maxCaptureAge) {
      throw const InferenceException(
        'stale_response',
        'La captura caduco. Analiza de nuevo',
      );
    }
  }

  void cancelAnalysis() {
    if (!isInferenceRunning || _disposed) return;
    _generation++;
    _detector.cancel();
    lastResponse = null;
    lastImage = null;
    lastCapture = null;
    errorMessage = 'Analisis cancelado';
    _setState(InferenceState.cancelled);
  }

  Future<bool> recordVisualResult(
    AnalyzedCapture capture, {
    required String prompt,
    required String text,
    required String model,
    String? speechText,
  }) async {
    if (!isResultCurrent(capture.analysis)) return false;
    try {
      await _logger.record({
        'event': 'visual_description',
        'request_id': capture.id,
        'captured_at': capture.capturedAt.toIso8601String(),
        'timestamp': DateTime.now().toUtc().toIso8601String(),
        'model_version': capture.response.modelVersion,
        'generator': model,
        'prompt': prompt,
        'description': text,
        if (speechText != null) 'final_message': speechText,
        'delivery_status': 'generated_not_audio_confirmed',
      });
    } catch (_) {
      debugPrint('No se pudo registrar la descripcion visual');
    }
    return isResultCurrent(capture.analysis);
  }

  @override
  void dispose() {
    _disposed = true;
    lastCapture = null;
    lastImage = null;
    lastResponse = null;
    _generation++;
    camera.removeListener(_cameraChanged);
    _detector.dispose();
    super.dispose();
  }
}
