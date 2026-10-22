import 'dart:typed_data';

class CameraFrame {
  const CameraFrame({
    required this.bytes,
    required this.timestamp,
    required this.source,
  });
  final Uint8List bytes;
  final DateTime timestamp;
  final String source;
}
