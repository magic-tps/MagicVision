class AnalysisResult {
  const AnalysisResult({
    required this.riskLevel,
    required this.description,
    required this.recommendation,
    required this.suggestedAction,
    required this.detectedObjects,
    required this.confidence,
    required this.timestamp,
    required this.source,
  });

  final String riskLevel;
  final String description;
  final String recommendation;
  final String suggestedAction;
  final List<String> detectedObjects;
  final double confidence;
  final DateTime timestamp;
  final String source;

  bool get isMock => source == 'mock';

  String get sourceText {
    return switch (source) {
      'mock' => 'Demo mock',
      'aws_sagemaker' => 'YOLO26m AWS SageMaker',
      'tflite_float32' => 'YOLO26m local (historico)',
      'tflite' => 'TFLite',
      _ => source,
    };
  }

  String get detectedObjectsText {
    if (detectedObjects.isEmpty) return 'sin objetos peligrosos';
    return detectedObjects.join(', ');
  }

  String get speechText {
    return '$description $recommendation Accion sugerida: $suggestedAction.';
  }

  String get displayText {
    final safeConfidence = confidence.isFinite ? confidence : 0.0;
    final percent = (safeConfidence * 100).clamp(0, 100).toStringAsFixed(0);
    return [
      description,
      'Riesgo: $riskLevel',
      'Objetos: $detectedObjectsText',
      'Recomendacion: $recommendation',
      'Accion: $suggestedAction',
      'Confianza: $percent%',
      'Fuente: $sourceText',
    ].join('\n');
  }

  Map<String, Object?> toHistoryJson() {
    return {
      'type': 'environment',
      'description': description,
      'riskLevel': riskLevel,
      'detectedObjects': detectedObjects,
      'recommendation': recommendation,
      'suggestedAction': suggestedAction,
      'confidence': confidence,
      'timestamp': timestamp.toIso8601String(),
      'source': source,
    };
  }
}
