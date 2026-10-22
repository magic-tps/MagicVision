import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:mi_app/core/config/inference_image_config.dart';
import 'package:mi_app/core/services/aws_object_detection_service.dart';
import 'package:mi_app/core/services/inference_readiness_service.dart';
import 'package:mi_app/core/services/vision_analysis_service.dart';
import 'inference_test.dart' as fixtures;

const job = '84600650-837f-437c-9b1e-cd60c4786e79';
http.Response jsonReply(Object body, [int status = 200]) => http.Response(
  jsonEncode(body),
  status,
  headers: {'content-type': 'application/json'},
);
Map<String, Object> pending(String requestId, [String jobId = job]) => {
  'request_id': requestId,
  'job_id': jobId,
  'status': 'processing',
  'poll_after_ms': 100,
};
String multipartId(http.Request request) => RegExp(
  r'name="request_id"\r\n\r\n([^\r]+)',
).firstMatch(latin1.decode(request.bodyBytes))!.group(1)!;

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test(
    'delivered result expires before delayed narration and is invalidated by a new capture',
    () async {
      final camera = CountingCamera();
      final service = VisionAnalysisService(
        camera: camera,
        objectDetectionService: fixtures.FakeDetector(),
        logger: fixtures.MemoryLogger(),
        config: const InferenceImageConfig(maxCaptureAge: Duration(seconds: 1)),
      );
      addTearDown(service.dispose);
      addTearDown(camera.dispose);
      final first = await service.analyzeCurrentFrame();
      expect(first, isNotNull);
      expect(service.isResultCurrent(first!), isTrue);
      final second = await service.analyzeCurrentFrame();
      expect(service.isResultCurrent(first), isFalse);
      expect(service.isResultCurrent(second!), isTrue);
      await Future<void>.delayed(const Duration(milliseconds: 1010));
      expect(service.isResultCurrent(second), isFalse);
    },
  );
  test(
    'async submit and polls deliver only the correlated final response',
    () async {
      final methods = <String>[];
      final service = AwsObjectDetectionService(
        baseUrl: 'https://example.test/prod',
        clientFactory: () => MockClient((request) async {
          methods.add(request.method);
          if (request.method == 'POST') return jsonReply(pending('scene'), 202);
          expect(request.url.path, '/prod/v1/inference/$job');
          return methods.length == 2
              ? jsonReply(pending('scene'), 202)
              : jsonReply(fixtures.response(id: 'scene'));
        }),
      );
      addTearDown(service.dispose);
      final result = await service.detect(fixtures.jpeg(), requestId: 'scene');
      expect(result.detections.single.label, 'pothole');
      expect(methods, ['POST', 'GET', 'GET']);
    },
  );
  test('warmup carries no camera image and waits for ready', () async {
    String? id;
    final service = AwsObjectDetectionService(
      baseUrl: 'https://example.test',
      clientFactory: () => MockClient((request) async {
        if (request.method == 'POST') {
          expect(request.url.path, '/v1/inference/warmup');
          expect(
            latin1.decode(request.bodyBytes),
            isNot(contains('name="image"')),
          );
          id = multipartId(request);
          return jsonReply(pending(id!), 202);
        }
        return jsonReply({'request_id': id, 'status': 'ready'});
      }),
    );
    addTearDown(service.dispose);
    await service.ensureReady();
  });
  test(
    'cancel during polling sends DELETE and prevents subsequent polls',
    () async {
      final submitted = Completer<void>();
      final deleted = Completer<void>();
      var gets = 0;
      final service = AwsObjectDetectionService(
        baseUrl: 'https://example.test',
        clientFactory: () => MockClient((request) async {
          if (request.method == 'DELETE') {
            deleted.complete();
            return jsonReply({'status': 'cancelled'});
          }
          if (request.method == 'GET') gets++;
          if (!submitted.isCompleted) submitted.complete();
          return jsonReply(pending('scene'), 202);
        }),
      );
      addTearDown(service.dispose);
      final action = service.detect(fixtures.jpeg(), requestId: 'scene');
      final assertion = expectLater(
        action,
        throwsA(fixtures.failure('cancelled')),
      );
      await submitted.future;
      // Let the submission resolve and register its job before cancelling the delay.
      await Future<void>.delayed(const Duration(milliseconds: 20));
      service.cancel();
      await assertion;
      await deleted.future.timeout(const Duration(seconds: 1));
      await Future<void>.delayed(const Duration(milliseconds: 120));
      expect(gets, 0);
    },
  );
  test('changed job during polling is rejected', () async {
    final service = AwsObjectDetectionService(
      baseUrl: 'https://example.test',
      clientFactory: () => MockClient(
        (request) async => request.method == 'DELETE'
            ? jsonReply({'status': 'cancelled'})
            : jsonReply(
                pending(
                  'scene',
                  request.method == 'POST'
                      ? job
                      : '84600650-837f-437c-9b1e-cd60c4786e70',
                ),
                202,
              ),
      ),
    );
    addTearDown(service.dispose);
    await expectLater(
      service.detect(fixtures.jpeg(), requestId: 'scene'),
      throwsA(fixtures.failure('invalid_response')),
    );
  });
  test('expired server capture is rejected', () async {
    final service = AwsObjectDetectionService(
      baseUrl: 'https://example.test',
      clientFactory: () =>
          MockClient((request) async => jsonReply({'detail': 'expired'}, 410)),
    );
    addTearDown(service.dispose);
    await expectLater(
      service.detect(fixtures.jpeg()),
      throwsA(fixtures.failure('stale_response')),
    );
  });
  test(
    'camera captures only after service is ready, then CERM receives the scene',
    () async {
      final camera = CountingCamera();
      final detector = ReadinessDetector();
      final service = VisionAnalysisService(
        camera: camera,
        objectDetectionService: detector,
        logger: fixtures.MemoryLogger(),
      );
      addTearDown(service.dispose);
      addTearDown(camera.dispose);
      final action = service.analyzeCurrentFrame();
      await detector.preparing.future;
      expect(service.state, InferenceState.preparing);
      expect(camera.captures, 0);
      detector.ready.complete();
      expect((await action)?.riskLevel, 'alto');
      expect(camera.captures, 1);
    },
  );
  test(
    'cancelling preparation prevents capture even if readiness arrives later',
    () async {
      final camera = CountingCamera();
      final detector = ReadinessDetector();
      final service = VisionAnalysisService(
        camera: camera,
        objectDetectionService: detector,
        logger: fixtures.MemoryLogger(),
      );
      addTearDown(service.dispose);
      addTearDown(camera.dispose);
      final action = service.analyzeCurrentFrame();
      await detector.preparing.future;
      service.cancelAnalysis();
      detector.ready.complete();
      expect(await action, isNull);
      expect(camera.captures, 0);
      expect(service.state, InferenceState.cancelled);
    },
  );
  test(
    'scene that expires during persistence cannot reach narration',
    () async {
      final camera = CountingCamera();
      final logger = fixtures.BlockingLogger();
      final service = VisionAnalysisService(
        camera: camera,
        objectDetectionService: fixtures.FakeDetector(),
        logger: logger,
        config: const InferenceImageConfig(
          maxCaptureAge: Duration(milliseconds: 300),
        ),
      );
      addTearDown(service.dispose);
      addTearDown(camera.dispose);
      final action = service.analyzeCurrentFrame();
      await logger.started.future;
      await Future<void>.delayed(const Duration(milliseconds: 310));
      logger.release.complete();
      expect(await action, isNull);
      expect(service.lastResponse, isNull);
      expect(service.state, InferenceState.error);
    },
  );
}

class CountingCamera extends fixtures.FakeCamera {
  int captures = 0;
  @override
  Future<Uint8List> captureFrame() async {
    captures++;
    return super.captureFrame();
  }
}

class ReadinessDetector extends fixtures.FakeDetector
    implements InferenceReadinessService {
  final ready = Completer<void>();
  final preparing = Completer<void>();
  @override
  Future<void> ensureReady() async {
    preparing.complete();
    await ready.future;
  }
}
