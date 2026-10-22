import 'dart:typed_data';
import '../models/detection_response.dart';

abstract class ObjectDetectionService {
  Future<DetectionResponse> detect(
    Uint8List imageBytes, {
    String? requestId,
    void Function()? onProcessing,
  });
  void cancel();
  void dispose();
}
