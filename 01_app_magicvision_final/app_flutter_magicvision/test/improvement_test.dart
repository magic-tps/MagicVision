import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:mi_app/core/models/detection_response.dart';
import 'package:mi_app/core/models/detection_result.dart';
import 'package:mi_app/core/config/inference_image_config.dart';
import 'package:mi_app/core/services/inference_image_processor.dart';
import 'package:mi_app/core/services/capture_archive.dart';
import 'package:mi_app/core/services/risk_assessment_service.dart';
import 'package:mi_app/core/services/vision_analysis_service.dart';
import 'inference_test.dart'
    show FakeCamera, FakeDetector, MemoryLogger, detection, response;

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test(
    'EXIF capture retains stored dimensions and oriented detector reference',
    () async {
      final source = img.Image(width: 120, height: 80);
      source.exif.imageIfd.orientation = 6;
      source.exif.gpsIfd[1] = img.IfdValueAscii('N');
      final bytes = Uint8List.fromList(img.encodeJpg(source));
      expect(img.decodeJpgExif(bytes)!.gpsIfd.isEmpty, isFalse);
      final result = await const InferenceImageProcessor(
        InferenceImageConfig(),
      ).prepare(bytes);
      expect(result.exifOrientation, 6);
      expect([result.originalWidth, result.originalHeight], [80, 120]);
      expect(
        [result.encodedOriginalWidth, result.encodedOriginalHeight],
        [120, 80],
      );
      expect([result.width, result.height], [80, 120]);
      expect(img.decodeJpg(result.bytes)!.exif.isEmpty, isTrue);
      final stripped = img.decodeJpg(stripPrivateJpegMetadata(bytes))!;
      expect([stripped.width, stripped.height], [120, 80]);
      expect(stripped.exif.imageIfd.orientation, isNull);
    },
  );
  test(
    'capture removes private JPEG metadata without changing image pixels',
    () {
      final jpeg = img.encodeJpg(img.Image(width: 16, height: 12));
      final metadata = utf8.encode('Exif GPS private-coordinate');
      final segmentLength = metadata.length + 2;
      final tagged = Uint8List.fromList([
        ...jpeg.sublist(0, 2),
        0xff,
        0xe1,
        segmentLength >> 8,
        segmentLength & 0xff,
        ...metadata,
        ...jpeg.sublist(2),
      ]);
      final stripped = stripPrivateJpegMetadata(tagged);
      expect(stripped, jpeg);
      expect(img.decodeJpg(stripped)!.width, 16);
      expect(
        () => stripPrivateJpegMetadata(Uint8List.fromList([0, 1])),
        throwsFormatException,
      );
    },
  );
  test('campus22 class IDs and model version roundtrip', () {
    for (var id = 0; id < detectionClassNames.length; id++) {
      final json = {
        ...detection(),
        'class_id': id,
        'label': detectionClassNames[id],
      };
      expect(Detection.fromJson(json).classId, id);
    }
    expect(
      () => Detection.fromJson({...detection(), 'class_id': 10}),
      throwsFormatException,
    );
    final data = {...response(), 'model_version': 'hash-v2'};
    expect(DetectionResponse.fromJson(data).toJson(), data);
  });

  test(
    'detected distant person remains audible without asserting clear path',
    () {
      final result = RiskAssessmentService().assess([
        const DetectionResult(
          label: 'person',
          confidence: .35,
          boundingBox: Rect.fromLTRB(.1, .1, .2, .3),
        ),
      ]);
      expect(result.riskLevel, 'bajo');
      expect(
        result.speechText.toLowerCase(),
        contains('persona a la izquierda'),
      );
      expect(result.speechText, isNot(contains('El camino parece libre')));
      final empty = RiskAssessmentService().assess([]);
      expect(empty.confidence, 0);
      expect(empty.speechText, contains('no confirma'));
    },
  );

  test(
    'trash bin is high risk in walking zone and not automatically high at side',
    () {
      final service = RiskAssessmentService();
      for (final (box, risk) in [
        (const Rect.fromLTRB(.4, .6, .6, .9), 'alto'),
        (const Rect.fromLTRB(.05, .1, .2, .3), 'bajo'),
      ]) {
        final result = service.assess([
          DetectionResult(label: 'trash_bin', confidence: .8, boundingBox: box),
        ]);
        expect(result.riskLevel, risk);
        expect(result.speechText.toLowerCase(), contains('tacho de basura'));
      }
    },
  );

  test('risk message follows the triggering detection', () {
    final result = RiskAssessmentService().assess([
      const DetectionResult(
        label: 'pothole',
        confidence: .9,
        boundingBox: Rect.fromLTRB(.1, .1, .2, .2),
      ),
      const DetectionResult(
        label: 'car',
        confidence: .9,
        boundingBox: Rect.fromLTRB(.3, .3, .7, .8),
      ),
    ]);
    expect(result.riskLevel, 'alto');
    expect(result.description, startsWith('Cuidado. Auto al frente.'));
  });

  test(
    'portrait and landscape preserve geometry at candidate resolution',
    () async {
      for (final (w, h, expected) in [
        (1080, 1920, [540, 960]),
        (1920, 1080, [960, 540]),
      ]) {
        final bytes = Uint8List.fromList(
          img.encodeJpg(img.Image(width: w, height: h)),
        );
        final result = await const InferenceImageProcessor(
          InferenceImageConfig(maxWidth: 960, maxHeight: 960),
        ).prepare(bytes);
        expect([result.width, result.height], expected);
      }
    },
  );

  test(
    'snapshot uses exact inference image and is invalidated on disconnect',
    () async {
      final camera = FakeCamera();
      final logger = MemoryLogger();
      final service = VisionAnalysisService(
        camera: camera,
        objectDetectionService: FakeDetector(),
        logger: logger,
      );
      final result = await service.analyzeCurrentFrame();
      final capture = service.lastCapture!;
      expect(identical(capture.image, service.lastImage), isTrue);
      expect(identical(capture.response, service.lastResponse), isTrue);
      expect(identical(capture.analysis, result), isTrue);
      expect(service.isResultCurrent(capture.analysis), isTrue);
      expect(
        await service.recordVisualResult(
          capture,
          prompt: 'Describe',
          text: 'Persona',
          model: 'test-vlm',
        ),
        isTrue,
      );
      expect(logger.events.last['request_id'], capture.id);
      expect(logger.events.last['event'], 'visual_description');
      camera.setReady(false);
      expect(service.lastCapture, isNull);
      expect(service.isResultCurrent(capture.analysis), isFalse);
      final eventCount = logger.events.length;
      expect(
        await service.recordVisualResult(
          capture,
          prompt: 'Describe',
          text: 'Vieja',
          model: 'test-vlm',
        ),
        isFalse,
      );
      expect(logger.events.length, eventCount);
      service.dispose();
      camera.dispose();
    },
  );

  test(
    'archive requires consent and stores paired images without location fields',
    () async {
      final root = await Directory.systemTemp.createTemp('mv-capture-test');
      addTearDown(() => root.delete(recursive: true));
      final bytes = Uint8List.fromList(
        img.encodeJpg(img.Image(width: 16, height: 12)),
      );
      final noConsent = CaptureArchive(
        enabled: true,
        directory: root,
        sessionId: 'session',
        routeId: 'route',
      );
      await noConsent.record(
        captureId: 'test',
        original: bytes,
        detector: bytes,
        metadata: {},
      );
      expect(await root.list().length, 0);
      final archive = CaptureArchive(
        enabled: true,
        consentConfirmed: true,
        directory: root,
        sessionId: 'session',
        routeId: 'route',
      );
      await archive.record(
        captureId: 'test',
        original: bytes,
        detector: bytes,
        metadata: {'latitude': -12.0, 'captured_at': 'now'},
      );
      final data = jsonDecode(
        await File('${root.path}/test/metadata.json').readAsString(),
      );
      expect(data.containsKey('latitude'), isFalse);
      expect(data['session_id'], 'session');
      expect(await File('${root.path}/test/detector.jpg').readAsBytes(), bytes);
    },
  );
}
