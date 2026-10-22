import 'dart:ui';
import '../models/detection_response.dart';
import '../models/detection_result.dart';

class CermInput {
  CermInput(List<DetectionResult> detections)
    : detections = List.unmodifiable(detections);
  final List<DetectionResult> detections;
  int get objectCount => detections.length;
}

class CermDetectionMapper {
  const CermDetectionMapper();
  CermInput map(DetectionResponse response) => CermInput(
    response.detections
        .map(
          (d) => DetectionResult(
            label: d.label,
            confidence: d.confidence,
            distanceM: d.distanceM,
            boundingBox: Rect.fromLTRB(
              d.bbox.left,
              d.bbox.top,
              d.bbox.right,
              d.bbox.bottom,
            ),
          ),
        )
        .toList(),
  );
}
