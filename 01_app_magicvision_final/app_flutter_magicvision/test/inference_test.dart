import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:image/image.dart' as img;
import 'package:mi_app/camera/camera_frame.dart';
import 'package:mi_app/camera/camera_source.dart';
import 'package:mi_app/core/config/inference_image_config.dart';
import 'package:mi_app/core/models/detection_response.dart';
import 'package:mi_app/core/services/aws_object_detection_service.dart';
import 'package:mi_app/core/services/cerm_detection_mapper.dart';
import 'package:mi_app/core/services/experimental_logger.dart';
import 'package:mi_app/core/services/inference_exception.dart';
import 'package:mi_app/core/services/inference_image_processor.dart';
import 'package:mi_app/core/services/object_detection_service.dart';
import 'package:mi_app/core/services/risk_assessment_service.dart';
import 'package:mi_app/core/services/vision_analysis_service.dart';

Map<String, dynamic> detection({double confidence = .87}) => {
  'class_id': 6,
  'label': 'pothole',
  'confidence': confidence,
  'bbox': {'left': .25, 'top': .5, 'right': .61, 'bottom': .93},
};
Map<String, dynamic> response({
  String id = 'test-request',
  List<Map<String, dynamic>>? detections,
}) => {
  'request_id': id,
  'processing_time_ms': 145,
  'image_width': 16,
  'image_height': 12,
  'detections': detections ?? [detection()],
};
Uint8List jpeg() =>
    Uint8List.fromList(img.encodeJpg(img.Image(width: 16, height: 12)));
Matcher failure(String code) =>
    isA<InferenceException>().having((e) => e.code, 'code', code);

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test('DetectionResponse.fromJson round trips typed detections', () {
    final json = response();
    final parsed = DetectionResponse.fromJson(json);
    expect(parsed.toJson(), json);
    expect(parsed.detections.single.classId, 6);
  });
  test('BoundingBox explicitly normalizes pixels', () {
    final bbox = BoundingBox.fromPixels(
      left: 160,
      top: 90,
      right: 480,
      bottom: 360,
      width: 640,
      height: 480,
    );
    expect(bbox.toJson(), {
      'left': .25,
      'top': .1875,
      'right': .75,
      'bottom': .75,
    });
  });
  test(
    'BoundingBox rejects unnormalized inverted empty and nonfinite boxes',
    () {
      for (final box in [
        {'left': -.1, 'top': 0, 'right': 1, 'bottom': 1},
        {'left': 0, 'top': 0, 'right': 640, 'bottom': 480},
        {'left': .7, 'top': 0, 'right': .2, 'bottom': 1},
        {'left': 0, 'top': .5, 'right': 1, 'bottom': .5},
        {'left': double.nan, 'top': 0, 'right': 1, 'bottom': 1},
      ]) {
        expect(() => BoundingBox.fromJson(box), throwsFormatException);
      }
    },
  );
  test('confidence validates endpoints and rejects nonfinite values', () {
    for (final score in [0.0, .3, 1.0]) {
      expect(
        Detection.fromJson(detection(confidence: score)).confidence,
        score,
      );
    }
    for (final score in [-.1, 1.1, double.infinity, double.nan]) {
      expect(
        () => Detection.fromJson(detection(confidence: score)),
        throwsFormatException,
      );
    }
  });
  test('invalid JSON shapes and processing metadata are rejected', () {
    for (final json in [
      <String, dynamic>{},
      {...response(), 'detections': null},
      {...response(), 'image_width': 0},
      {...response(), 'processing_time_ms': -1},
      {
        ...response(),
        'detections': [null],
      },
    ]) {
      expect(() => DetectionResponse.fromJson(json), throwsFormatException);
    }
  });
  test(
    'CermDetectionMapper preserves duplicates geometry confidence and risk',
    () {
      final mapped = const CermDetectionMapper().map(
        DetectionResponse.fromJson(
          response(detections: [detection(), detection(confidence: .6)]),
        ),
      );
      expect(mapped.objectCount, 2);
      expect(
        mapped.detections.first.boundingBox,
        const Rect.fromLTRB(.25, .5, .61, .93),
      );
      expect(mapped.detections.last.confidence, .6);
      final assessment = RiskAssessmentService().assess(mapped.detections);
      expect(assessment.riskLevel, 'alto');
      expect(assessment.detectedObjects, ['pothole x2']);
    },
  );
  test('valid empty response reaches unchanged CERM empty-scene rule', () {
    final mapped = const CermDetectionMapper().map(
      DetectionResponse.fromJson(response(detections: [])),
    );
    expect(mapped.objectCount, 0);
    expect(RiskAssessmentService().assess(mapped.detections).riskLevel, 'bajo');
  });
  test('image resize preserves aspect ratio and JPEG bounds', () async {
    final bytes = Uint8List.fromList(
      img.encodeJpg(img.Image(width: 1920, height: 1080)),
    );
    final prepared = await const InferenceImageProcessor(
      InferenceImageConfig(),
    ).prepare(bytes);
    expect([prepared.width, prepared.height], [640, 360]);
    expect([prepared.originalWidth, prepared.originalHeight], [1920, 1080]);
    expect(img.decodeJpg(prepared.bytes), isNotNull);
  });
  test('invalid and empty frame fail before upload', () async {
    final processor = const InferenceImageProcessor(InferenceImageConfig());
    await expectLater(
      processor.prepare(Uint8List(0)),
      throwsA(failure('empty_image')),
    );
    await expectLater(
      processor.prepare(Uint8List.fromList([1, 2])),
      throwsA(failure('invalid_frame')),
    );
  });
  test('defaults disable continuous analysis and saved images', () {
    expect(InferenceImageConfig.continuousAnalysis, isFalse);
    expect(InferenceImageConfig.saveDebugImages, isFalse);
    expect(const InferenceImageConfig().timeout, const Duration(seconds: 10));
  });
  test('HTTP sends multipart JPEG with a single correlated request', () async {
    final service = AwsObjectDetectionService(
      baseUrl: 'https://example.test/api/',
      clientFactory: () => MockClient((request) async {
        expect(request.url.path, '/api/v1/inference');
        expect(
          request.headers['content-type'],
          startsWith('multipart/form-data'),
        );
        expect(
          latin1.decode(request.bodyBytes),
          contains('name="image"; filename="frame.jpg"'),
        );
        expect(latin1.decode(request.bodyBytes), contains('name="request_id"'));
        return http.Response(
          jsonEncode(response()),
          200,
          headers: {'content-type': 'application/json'},
        );
      }),
    );
    final result = await service.detect(jpeg(), requestId: 'test-request');
    expect(result.requestId, 'test-request');
    service.dispose();
  });
  test(
    'HTTP 4xx 5xx redirects invalid JSON and wrong request ID fail',
    () async {
      for (final status in [400, 401, 429, 500, 503, 302]) {
        final service = AwsObjectDetectionService(
          baseUrl: 'https://example.test',
          clientFactory: () =>
              MockClient((_) async => http.Response('{}', status)),
        );
        await expectLater(
          service.detect(jpeg()),
          throwsA(failure('http_$status')),
        );
        service.dispose();
      }
      for (final body in [
        'not json',
        '{}',
        jsonEncode(response(id: 'another-request')),
      ]) {
        final service = AwsObjectDetectionService(
          baseUrl: 'https://example.test',
          clientFactory: () => MockClient(
            (_) async => http.Response(
              body,
              200,
              headers: {'content-type': 'application/json'},
            ),
          ),
        );
        await expectLater(
          service.detect(jpeg(), requestId: 'test-request'),
          throwsA(isA<InferenceException>()),
        );
        service.dispose();
      }
    },
  );
  test(
    'timeout covers a pending HTTP request and network errors are distinct',
    () async {
      final delayed = Completer<http.Response>();
      final service = AwsObjectDetectionService(
        baseUrl: 'https://example.test',
        config: const InferenceImageConfig(timeout: Duration(milliseconds: 20)),
        clientFactory: () => MockClient((_) => delayed.future),
      );
      await expectLater(service.detect(jpeg()), throwsA(failure('timeout')));
      delayed.complete(http.Response('{}', 200));
      service.dispose();
      final offline = AwsObjectDetectionService(
        baseUrl: 'https://example.test',
        clientFactory: () =>
            MockClient((_) async => throw const SocketException('offline')),
      );
      await expectLater(offline.detect(jpeg()), throwsA(failure('network')));
      offline.dispose();
    },
  );
  test('HTTPS required and empty image does not create HTTP client', () async {
    final service = AwsObjectDetectionService(
      baseUrl: 'http://example.test',
      clientFactory: () => throw StateError('must not send'),
    );
    await expectLater(
      service.detect(jpeg()),
      throwsA(failure('configuration')),
    );
    service.dispose();
  });
  test(
    'camera disconnected creates an error log without invented CERM result',
    () async {
      final camera = FakeCamera(ready: false);
      final detector = FakeDetector();
      final logger = MemoryLogger();
      final service = VisionAnalysisService(
        camera: camera,
        objectDetectionService: detector,
        logger: logger,
      );
      expect(await service.analyzeCurrentFrame(), isNull);
      expect(detector.calls, 0);
      expect(service.state, InferenceState.cameraDisconnected);
      expect(logger.events.single['risk_level'], isNull);
      expect(logger.events.single['error_code'], 'camera_disconnected');
      service.dispose();
      camera.dispose();
    },
  );
  test(
    'one request at a time and disconnect invalidates late response; reconnect works',
    () async {
      final camera = FakeCamera();
      final detector = FakeDetector()..pending = Completer<DetectionResponse>();
      final logger = MemoryLogger();
      final service = VisionAnalysisService(
        camera: camera,
        objectDetectionService: detector,
        logger: logger,
      );
      final first = service.analyzeCurrentFrame();
      await detector.started.future;
      expect(await service.analyzeCurrentFrame(), isNull);
      expect(detector.calls, 1);
      camera.setReady(false);
      detector.pending!.complete(
        DetectionResponse.fromJson(response(id: detector.requestId!)),
      );
      expect(await first, isNull);
      expect(service.lastResponse, isNull);
      camera.setReady(true);
      detector.pending = null;
      final result = await service.analyzeCurrentFrame();
      expect(result?.riskLevel, 'alto');
      expect(result?.source, 'aws_sagemaker');
      expect(logger.events.last['detections_count'], 1);
      expect(logger.events.last['jpeg_size_bytes'], greaterThan(0));
      expect(service.state, InferenceState.resultReady);
      service.dispose();
      camera.dispose();
    },
  );
  test('disconnect during logging also invalidates delivery to TTS', () async {
    final camera = FakeCamera();
    final logger = BlockingLogger();
    final service = VisionAnalysisService(
      camera: camera,
      objectDetectionService: FakeDetector(),
      logger: logger,
    );
    final result = service.analyzeCurrentFrame();
    await logger.started.future;
    camera.setReady(false);
    logger.release.complete();
    expect(await result, isNull);
    expect(service.lastResponse, isNull);
    service.dispose();
    camera.dispose();
  });
}

class BlockingLogger implements ExperimentalLogger {
  final started = Completer<void>();
  final release = Completer<void>();
  @override
  Future<void> record(
    Map<String, Object?> event, {
    Uint8List? debugImage,
  }) async {
    started.complete();
    await release.future;
  }
}

class FakeCamera extends CameraSource {
  FakeCamera({bool ready = true}) : _ready = ready;
  bool _ready;
  void setReady(bool value) {
    _ready = value;
    notifyListeners();
  }

  @override
  String get name => 'test_uvc';
  @override
  CameraStatus get status =>
      _ready ? CameraStatus.ready : CameraStatus.disconnected;
  @override
  String? get errorMessage => null;
  @override
  Stream<CameraFrame> get frames => const Stream.empty();
  @override
  Future<void> initialize() async {}
  @override
  Future<Uint8List> captureFrame() async => jpeg();
  @override
  Widget buildPreview() => const SizedBox();
  @override
  Future<void> close() async {}
}

class FakeDetector implements ObjectDetectionService {
  int calls = 0;
  String? requestId;
  Completer<DetectionResponse>? pending;
  final started = Completer<void>();
  @override
  Future<DetectionResponse> detect(
    Uint8List bytes, {
    String? requestId,
    void Function()? onProcessing,
  }) async {
    calls++;
    this.requestId = requestId;
    if (!started.isCompleted) started.complete();
    onProcessing?.call();
    return pending == null
        ? DetectionResponse.fromJson(response(id: requestId!))
        : pending!.future;
  }

  @override
  void cancel() {}
  @override
  void dispose() {}
}

class MemoryLogger implements ExperimentalLogger {
  final events = <Map<String, Object?>>[];
  @override
  Future<void> record(
    Map<String, Object?> event, {
    Uint8List? debugImage,
  }) async {
    events.add(Map.of(event));
  }
}
