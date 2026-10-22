import '../models/analysis_result.dart';
import '../models/detection_result.dart';
import '../models/detection_response.dart';
import 'walking_assist.dart';

/// CERM and walking narration share the same 22-class risk policy.
class RiskAssessmentService {
  AnalysisResult assess(
    List<DetectionResult> detections, {
    String source = 'aws_sagemaker',
  }) {
    final engine = CampusRiskEngine();
    final hazards = detections.map((d) {
      final name = campusName(d.label);
      final id = detectionClassNames.indexOf(name);
      if (id < 0) throw FormatException('Clase fuera del contrato: $name');
      return engine.assess(
        TrackedHazard(
          Detection(
            classId: id,
            label: name,
            confidence: d.confidence,
            bbox: BoundingBox(
              left: d.boundingBox.left,
              top: d.boundingBox.top,
              right: d.boundingBox.right,
              bottom: d.boundingBox.bottom,
            ),
          ),
          distance: d.distanceM,
        ),
      );
    }).toList()..sort((a, b) => b.score.compareTo(a.score));
    final rank = hazards.isEmpty ? 0 : riskRank(hazards.first.risk);
    final counts = <String, int>{};
    for (final d in detections) {
      final name = campusName(d.label);
      counts[name] = (counts[name] ?? 0) + 1;
    }
    return AnalysisResult(
      riskLevel: rank >= 2
          ? 'alto'
          : rank == 1
          ? 'medio'
          : 'bajo',
      description: hazards.isEmpty
          ? 'No se detectaron objetos en esta captura. Esto no confirma que el camino este libre.'
          : hazards.take(5).map((h) => h.message).join(' '),
      recommendation: rank >= 2
          ? 'Detente y verifica el entorno antes de avanzar.'
          : 'Verifica el entorno; la deteccion no confirma un paso libre.',
      suggestedAction: rank >= 2 ? 'Detente y verifica' : 'Verifica el entorno',
      detectedObjects: counts.entries
          .map((e) => e.value == 1 ? e.key : '${e.key} x${e.value}')
          .toList(),
      confidence: hazards.isEmpty
          ? 0
          : hazards
                    .take(5)
                    .map((h) => h.detection.confidence)
                    .reduce((a, b) => a + b) /
                hazards.take(5).length,
      timestamp: DateTime.now(),
      source: source,
    );
  }
}
