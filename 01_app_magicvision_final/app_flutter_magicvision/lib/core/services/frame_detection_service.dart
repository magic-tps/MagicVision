import 'dart:typed_data';
import '../models/detection_response.dart';

abstract interface class FrameDetectionService {
  Future<DetectionResponse> detectFrame(
    Uint8List bytes, {
    required String requestId,
    required DateTime capturedAt,
    required String cameraId,
    void Function()? onProcessing,
  });
}
