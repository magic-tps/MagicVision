import 'dart:async';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mi_app/audio/audio_input_source.dart';
import 'package:mi_app/camera/camera_source.dart';
import 'package:mi_app/camera/usb_camera_source.dart';
import 'package:mi_app/core/services/inference_exception.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final messenger =
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
  const camera = MethodChannel('magicvision/usb_camera');
  const events = MethodChannel('magicvision/usb_camera_events');
  const audio = MethodChannel('magicvision/audio_input');
  setUp(() {
    messenger.setMockMethodCallHandler(events, (_) async => null);
  });
  tearDown(() {
    messenger.setMockMethodCallHandler(camera, null);
    messenger.setMockMethodCallHandler(events, null);
    messenger.setMockMethodCallHandler(audio, null);
  });
  test(
    'USB source handles permission denial and explicit reconnection',
    () async {
      var denied = true;
      messenger.setMockMethodCallHandler(camera, (call) async {
        if (call.method == 'initialize') {
          if (denied) {
            throw PlatformException(
              code: 'permission_denied',
              message: 'Permiso USB rechazado',
            );
          }
          return {
            'state': 'cameraReady',
            'texture_id': 1,
            'width': 640,
            'height': 480,
          };
        }
        if (call.method == 'captureFrame') return Uint8List.fromList([1, 2, 3]);
        return null;
      });
      final source = UsbCameraSource();
      await source.initialize();
      expect(source.status, CameraStatus.error);
      await expectLater(
        source.captureFrame(),
        throwsA(isA<InferenceException>()),
      );
      denied = false;
      await source.initialize();
      expect(source.status, CameraStatus.ready);
      final frame = source.frames.first;
      expect(await source.captureFrame(), [1, 2, 3]);
      expect((await frame).source, 'arducam_uvc');
      await source.close();
      source.dispose();
    },
  );
  test('USB late capture is rejected after a detach event', () async {
    final capture = Completer<Uint8List>();
    messenger.setMockMethodCallHandler(camera, (call) async {
      if (call.method == 'initialize') return {'state': 'cameraReady'};
      if (call.method == 'captureFrame') return capture.future;
      return null;
    });
    final source = UsbCameraSource();
    await source.initialize();
    final pending = source.captureFrame();
    await messenger.handlePlatformMessage(
      'magicvision/usb_camera_events',
      const StandardMethodCodec().encodeSuccessEnvelope({
        'state': 'cameraDisconnected',
      }),
      (_) {},
    );
    capture.complete(Uint8List.fromList([1, 2]));
    await expectLater(
      pending,
      throwsA(
        isA<InferenceException>().having(
          (e) => e.code,
          'code',
          'camera_disconnected',
        ),
      ),
    );
    expect(source.status, CameraStatus.disconnected);
    await source.close();
    source.dispose();
  });
  test(
    'USB audio presence never claims explicit SpeechRecognizer selection',
    () async {
      messenger.setMockMethodCallHandler(
        audio,
        (_) async => {'usb_available': true},
      );
      final input = await UsbMicrophoneSource().prepare();
      expect(input.usbAvailable, isTrue);
      expect(input.explicitSelection, isFalse);
    },
  );
  test(
    'missing USB audio or native plugin preserves the default microphone',
    () async {
      messenger.setMockMethodCallHandler(
        audio,
        (_) async => {'usb_available': false},
      );
      expect((await UsbMicrophoneSource().prepare()).routing, 'system_default');
      messenger.setMockMethodCallHandler(
        audio,
        (_) async => throw PlatformException(code: 'unavailable'),
      );
      expect((await UsbMicrophoneSource().prepare()).routing, 'system_default');
    },
  );
}
