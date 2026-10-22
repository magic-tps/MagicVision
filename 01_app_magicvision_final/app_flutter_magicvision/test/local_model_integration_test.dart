// Opt-in test driven by backend/testing/run_integration.py. No AWS or USB claim.
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/io_client.dart';
import 'package:mi_app/core/services/aws_object_detection_service.dart';
import 'package:mi_app/core/services/vision_analysis_service.dart';
import 'inference_test.dart' as fixtures;

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  const url = String.fromEnvironment('LOCAL_INTEGRATION_URL');
  test(
    'real JPEG -> HTTPS async API -> full YOLO26m -> Dart -> CERM speech',
    () async {
      const certificate = String.fromEnvironment('LOCAL_INTEGRATION_CERT');
      final context = SecurityContext(withTrustedRoots: certificate.isEmpty);
      if (certificate.isNotEmpty) context.setTrustedCertificates(certificate);
      final camera = DatasetCamera(
        File(
          const String.fromEnvironment('LOCAL_INTEGRATION_IMAGE'),
        ).readAsBytesSync(),
      );
      final logger = fixtures.MemoryLogger();
      final detector = AwsObjectDetectionService(
        baseUrl: url,
        clientFactory: () => IOClient(
          HttpOverrides.runWithHttpOverrides(
            () => HttpClient(context: context),
            RealHttpOverrides(),
          ),
        ),
      );
      final service = VisionAnalysisService(
        camera: camera,
        objectDetectionService: detector,
        logger: logger,
      );
      addTearDown(service.dispose);
      addTearDown(camera.dispose);
      final states = <String>[];
      service.addListener(() => states.add(service.state.name));
      final result = await service.analyzeCurrentFrame();
      expect(result, isNotNull, reason: service.errorMessage);
      expect(service.lastResponse!.detections, isNotEmpty);
      expect(
        service.lastResponse!.detections.any((d) => d.label == 'person'),
        isTrue,
      );
      expect(result!.isMock, isFalse);
      expect(result.speechText, isNotEmpty);
      expect(
        states.indexOf('preparing'),
        lessThan(states.indexOf('capturing')),
      );
      File(
        const String.fromEnvironment('LOCAL_INTEGRATION_REPORT'),
      ).writeAsStringSync(
        const JsonEncoder.withIndent('  ').convert({
          'scope': const String.fromEnvironment(
            'INTEGRATION_SCOPE',
            defaultValue:
                'Local real model and production API/Dart; simulated S3/SageMaker transport and camera; no audible TTS test',
          ),
          'response': service.lastResponse!.toJson(),
          'assessment': result.toHistoryJson(),
          'speech_text': result.speechText,
          'states': states,
          'metrics': logger.events,
        }),
      );
    },
    skip: url.isEmpty
        ? 'Run backend/testing/run_integration.py to start the real model and TLS API'
        : false,
    timeout: const Timeout(Duration(minutes: 17)),
  );
}

class DatasetCamera extends fixtures.FakeCamera {
  DatasetCamera(this.bytes);
  final Uint8List bytes;
  @override
  Future<Uint8List> captureFrame() async => bytes;
}

class RealHttpOverrides extends HttpOverrides {}
