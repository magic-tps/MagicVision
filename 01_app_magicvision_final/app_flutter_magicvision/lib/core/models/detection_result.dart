import 'dart:ui';

class DetectionResult {
  const DetectionResult({
    required this.label,
    required this.confidence,
    required this.boundingBox,
    this.distanceM,
  });

  final String label;
  final double confidence;
  final double? distanceM;

  /// Normalized rectangle in camera/image coordinates: left/top/right/bottom
  /// are expected to be in the 0..1 range.
  final Rect boundingBox;

  double get centerX => boundingBox.center.dx;
  double get centerY => boundingBox.center.dy;
  double get width => boundingBox.width;
  double get height => boundingBox.height;
  double get area => width * height;

  bool get isCenterArea => centerX >= 0.35 && centerX <= 0.65;
  bool get isBottomArea => centerY >= 0.55 || boundingBox.bottom >= 0.68;
  bool get isDangerZone => isCenterArea && isBottomArea;
}
