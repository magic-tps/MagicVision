import '../config/campus_contract.dart';

double _number(Object? value, String name) {
  if (value is! num || !value.isFinite) throw FormatException('$name invalido');
  return value.toDouble();
}

const detectionClassNames = campusClassNames;

int _integer(Object? value, String name, {int min = 0}) {
  if (value is! int || value < min) throw FormatException('$name invalido');
  return value;
}

String _text(Object? value, String name) {
  if (value is! String || value.trim().isEmpty) {
    throw FormatException('$name invalido');
  }
  return value;
}

class BoundingBox {
  BoundingBox({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  }) {
    if ([left, top, right, bottom].any((v) => !v.isFinite || v < 0 || v > 1) ||
        right <= left ||
        bottom <= top) {
      throw const FormatException(
        'bbox debe ser normalizada y tener area positiva',
      );
    }
  }
  final double left, top, right, bottom;

  factory BoundingBox.fromJson(Map<String, dynamic> json) => BoundingBox(
    left: _number(json['left'], 'left'),
    top: _number(json['top'], 'top'),
    right: _number(json['right'], 'right'),
    bottom: _number(json['bottom'], 'bottom'),
  );

  /// Explicit pixel conversion, never guess the coordinate system of API JSON.
  factory BoundingBox.fromPixels({
    required double left,
    required double top,
    required double right,
    required double bottom,
    required int width,
    required int height,
  }) {
    if (width <= 0 || height <= 0) {
      throw const FormatException('Resolucion invalida');
    }
    return BoundingBox(
      left: left / width,
      top: top / height,
      right: right / width,
      bottom: bottom / height,
    );
  }

  Map<String, dynamic> toJson() => {
    'left': left,
    'top': top,
    'right': right,
    'bottom': bottom,
  };
}

class Detection {
  Detection({
    required this.classId,
    required this.label,
    required this.confidence,
    required this.bbox,
    this.distanceM,
    this.distanceRawM,
    this.depthStatus,
  }) {
    if (classId < 0 ||
        classId >= detectionClassNames.length ||
        (label != detectionClassNames[classId] &&
            !(classId == 10 && label == 'trash_bin')) ||
        label.trim().isEmpty ||
        !confidence.isFinite ||
        confidence < 0 ||
        confidence > 1) {
      throw const FormatException('Deteccion invalida');
    }
    for (final value in [distanceM, distanceRawM]) {
      if (value != null && (!value.isFinite || value <= 0 || value > 150)) {
        throw const FormatException('Distancia invalida');
      }
    }
    if (distanceM != null && depthStatus != 'calibrated') {
      throw const FormatException('Distancia sin calibracion');
    }
  }
  final int classId;
  final String label;
  final double confidence;
  final BoundingBox bbox;
  final double? distanceM, distanceRawM;
  final String? depthStatus;
  factory Detection.fromJson(Map<String, dynamic> json) => Detection(
    classId: _integer(json['class_id'], 'class_id'),
    label: _text(json['label'], 'label'),
    confidence: _number(json['confidence'], 'confidence'),
    bbox: BoundingBox.fromJson(Map<String, dynamic>.from(json['bbox'] as Map)),
    distanceM: json['distance_m'] == null
        ? null
        : _number(json['distance_m'], 'distance_m'),
    distanceRawM: json['distance_raw_m'] == null
        ? null
        : _number(json['distance_raw_m'], 'distance_raw_m'),
    depthStatus: json['depth_status'] as String?,
  );
  Map<String, dynamic> toJson() => {
    'class_id': classId,
    'label': label,
    'confidence': confidence,
    'bbox': bbox.toJson(),
    if (depthStatus != null) 'depth_status': depthStatus,
    if (depthStatus != null) 'class_name': label,
    if (depthStatus != null) 'distance_m': distanceM,
    if (depthStatus != null) 'distance_raw_m': distanceRawM,
  };
}

class DetectionResponse {
  DetectionResponse({
    required this.requestId,
    required this.processingTimeMs,
    required this.imageWidth,
    required this.imageHeight,
    required List<Detection> detections,
    this.modelVersion,
    this.frameId,
    this.timestamp,
    this.cameraId,
    this.performance,
    this.depthModelSha256,
    this.pixelSha256,
  }) : detections = List.unmodifiable(detections) {
    if (requestId.trim().isEmpty ||
        !processingTimeMs.isFinite ||
        processingTimeMs < 0 ||
        imageWidth <= 0 ||
        imageHeight <= 0) {
      throw const FormatException('Respuesta invalida');
    }
  }
  final String requestId;
  final double processingTimeMs;
  final int imageWidth, imageHeight;
  final List<Detection> detections;
  final String? modelVersion;
  final String? frameId, timestamp, cameraId;
  final Map<String, dynamic>? performance;
  final String? depthModelSha256, pixelSha256;
  factory DetectionResponse.fromJson(Map<String, dynamic> json) {
    try {
      if (json['schema_version'] == 2) {
        for (final key in ['depth_model_sha256','pixel_sha256']) {
          if (json[key] is! String || !RegExp(r'^[a-f0-9]{64}$').hasMatch(json[key] as String)) throw const FormatException('Hash invalido');
        }
        if (json['detections'] is! List || (json['detections'] as List).length > 300) throw const FormatException('Cantidad de detecciones invalida');
        for (final item in json['detections'] as List) {
          if (item is! Map || item['class_name'] != item['label'] || item['label'] == 'trash_bin') throw const FormatException('Contrato v2 invalido');
        }
        final perf = json['performance'];
        if (perf is! Map) throw const FormatException('Latencias ausentes');
        for (final key in ['detection_ms','depth_ms','fusion_ms','total_ms']) {
          if (_number(perf[key], key) < 0) throw const FormatException('Latencia invalida');
        }
      }
      if (json['schema_version'] != null &&
          (json['schema_version'] != 2 || json['taxonomy'] != campusTaxonomy)) {
        throw const FormatException('Contrato desconocido');
      }
      return DetectionResponse(
        requestId: _text(json['request_id'], 'request_id'),
        depthModelSha256: json['depth_model_sha256'] as String?,
        pixelSha256: json['pixel_sha256'] as String?,
        processingTimeMs: _number(
          json['processing_time_ms'] ?? json['performance']?['total_ms'],
          'processing_time_ms',
        ),
        imageWidth: _integer(json['image_width'], 'image_width', min: 1),
        imageHeight: _integer(json['image_height'], 'image_height', min: 1),
        detections: (json['detections'] as List)
            .map((v) => Detection.fromJson(Map<String, dynamic>.from(v as Map)))
            .toList(),
        modelVersion: json['model_version'] == null
            ? null
            : _text(json['model_version'], 'model_version'),
        frameId: json['schema_version'] == 2
            ? _text(json['frame_id'], 'frame_id')
            : null,
        timestamp: json['schema_version'] == 2
            ? _text(json['timestamp'], 'timestamp')
            : null,
        cameraId: json['schema_version'] == 2
            ? _text(json['camera_id'], 'camera_id')
            : null,
        performance: json['performance'] == null
            ? null
            : Map<String, dynamic>.from(json['performance'] as Map),
      );
    } on TypeError {
      throw const FormatException('Estructura JSON invalida');
    }
  }
  Map<String, dynamic> toJson() => {
    'request_id': requestId,
    'processing_time_ms': processingTimeMs,
    'image_width': imageWidth,
    'image_height': imageHeight,
    'detections': detections.map((v) => v.toJson()).toList(),
    if (modelVersion != null) 'model_version': modelVersion,
    if (frameId != null) ...{
      'schema_version': 2, 'taxonomy': campusTaxonomy, 'frame_id': frameId,
      'timestamp': timestamp, 'camera_id': cameraId, 'performance': performance,
      'depth_model_sha256': depthModelSha256, 'pixel_sha256': pixelSha256,
    },
  };
}
