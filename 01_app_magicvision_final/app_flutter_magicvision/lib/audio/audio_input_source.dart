import 'package:flutter/services.dart';

class AudioInputStatus {
  const AudioInputStatus({
    required this.usbAvailable,
    required this.routing,
    required this.explicitSelection,
  });
  final bool usbAvailable;
  final String routing;
  final bool explicitSelection;
}

abstract class AudioInputSource {
  Future<AudioInputStatus> prepare();
}

/// SpeechToText owns SpeechRecognizer. Android chooses its recording device.
/// Do not open an competing AudioRecord or change the TTS communication route.
class InternalMicrophoneSource implements AudioInputSource {
  @override
  Future<AudioInputStatus> prepare() async => const AudioInputStatus(
    usbAvailable: false,
    routing: 'system_default',
    explicitSelection: false,
  );
}

class UsbMicrophoneSource implements AudioInputSource {
  UsbMicrophoneSource({MethodChannel? channel, AudioInputSource? fallback})
    : _channel = channel ?? const MethodChannel('magicvision/audio_input'),
      _fallback = fallback ?? InternalMicrophoneSource();
  final MethodChannel _channel;
  final AudioInputSource _fallback;
  @override
  Future<AudioInputStatus> prepare() async {
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'inputDevices',
      );
      if (result?['usb_available'] == true) {
        // Android USB audio policy normally routes the recognizer automatically.
        // Presence alone is not proof that this recognizer selected that input.
        return const AudioInputStatus(
          usbAvailable: true,
          routing: 'android_usb_policy',
          explicitSelection: false,
        );
      }
    } on PlatformException {
      /* Continue with the existing recognizer. */
    } on MissingPluginException {
      /* Tests and non-Android platforms. */
    }
    return _fallback.prepare();
  }
}
