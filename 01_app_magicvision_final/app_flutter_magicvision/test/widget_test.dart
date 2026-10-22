import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart';
import 'package:mi_app/main.dart';

void main() {
  test('uses LFM2.5 vision model by default', () {
    expect(mobileVlmPreferredModelName, 'lfm2.5-vl-450m');
    expect(mobileVlmLocalModelName, 'lfm2-vl-1.6b');
    expect(mobileVlmModelId, 'LiquidAI/LFM2.5-VL-450M');
    expect(mobileVlmLocalModelId, 'LiquidAI/LFM2-VL-1.6B');
    expect(
      mobileVlmReadableModelName(mobileVlmPreferredModelName),
      'LFM 2.5 VL 450M',
    );
    expect(
      mobileVlmReadableModelName(mobileVlmLocalModelName),
      'LFM 2 VL 1.6B',
    );
    expect(mobileVlmBundledModelSpecs['lfm2-vl-1.6b']?.fileCount, 694);
    expect(mobileVlmBundledModelSpecs['lfm2-vl-1.6b']?.totalBytes, 2122555588);
  });

  test('formats long numbers for speech as digits', () {
    expect(
      formatLongNumbersForSpeech('Codigo 4560 listo'),
      'Codigo cuatro mil quinientos sesenta listo',
    );
    expect(
      formatLongNumbersForSpeech('Folio 999999 listo'),
      'Folio novecientos noventa y nueve mil novecientos noventa y nueve listo',
    );
    expect(
      formatLongNumbersForSpeech('Telefono 987-654-321 y monto 25'),
      'Telefono 9 8 7 6 5 4 3 2 1 y monto 25',
    );
  });

  test('parses spoken reminder durations', () {
    expect(
      parseSpanishDuration('temporizador en 5 minutos'),
      const Duration(minutes: 5),
    );
    expect(
      parseSpanishDuration('recuerdame medicina en diez minutos'),
      const Duration(minutes: 10),
    );
    expect(
      parseSpanishDuration('alarma en una hora'),
      const Duration(hours: 1),
    );
  });

  test('extracts reminder message', () {
    expect(
      extractReminderMessage('recuerdame tomar medicina en diez minutos'),
      'tomar medicina',
    );
  });

  test('recognizes voice stop commands', () {
    expect(isStopSpeakingCommand('alto'), isTrue);
    expect(isStopSpeakingCommand('cancelar'), isTrue);
    expect(isStopSpeakingCommand('para de hablar'), isTrue);
    expect(isStopSpeakingCommand('cancelar ruta'), isFalse);
  });

  test(
    'analysis cancellation phrases do not cancel navigation or voice alone',
    () {
      for (final phrase in [
        'cancelar analisis',
        'cancela el analisis',
        'por favor cancelar el analisis',
      ]) {
        expect(isCancelAnalysisCommand(phrase), isTrue);
      }
      for (final phrase in [
        'cancelar ruta',
        'cancelar',
        'silencio',
        'analiza imagen',
      ]) {
        expect(isCancelAnalysisCommand(phrase), isFalse);
      }
    },
  );

  test('keeps voice command restarts responsive', () {
    expect(
      speechCommandCooldown,
      lessThanOrEqualTo(const Duration(seconds: 1)),
    );
    expect(speechRestartDelay, lessThanOrEqualTo(const Duration(seconds: 5)));
    expect(
      speechAfterResponseRestartDelay,
      lessThanOrEqualTo(const Duration(seconds: 5)),
    );
  });

  test('keeps MobileVLM user wait bounded', () {
    expect(
      mobileVlmBackgroundResultWindow,
      lessThanOrEqualTo(const Duration(seconds: 4)),
    );
    expect(mobileVlmMaxTokens, lessThanOrEqualTo(128));
  });

  test('recognizes natural MobileVLM voice aliases', () {
    expect(isMobileVlmStatusCommandForSpeech('estado mobile vlm'), isTrue);
    expect(isMobileVlmStatusCommandForSpeech('estado modelo visual'), isTrue);
    expect(isMobileVlmStatusCommandForSpeech('estado externo'), isFalse);
    expect(isMobileVlmStatusCommandForSpeech('que modelo tienes'), isTrue);
    expect(isMobileVlmStatusCommandForSpeech('cual modelo usas'), isTrue);
    expect(isMobileVlmAnalyzeCommandForSpeech('analiza la imagen'), isTrue);
    expect(isMobileVlmAnalyzeCommandForSpeech('modelo de vision'), isTrue);
    expect(isMobileVlmAnalyzeCommandForSpeech('ve ele eme'), isTrue);
    expect(isMobileVlmAnalyzeCommandForSpeech('be ele eme'), isTrue);
  });

  test('explains unclear speech recognition errors', () {
    expect(
      speechRecognitionStatusMessage('error_no_match'),
      contains('No te escuche claro'),
    );
    expect(
      speechRecognitionStatusMessage('error_language_unavailable'),
      contains('idioma espanol'),
    );
  });

  test('parses voice contact command', () {
    final parsed = parseVoiceContactCommand(
      'agrega contacto mama numero nueve ocho siete seis cinco cuatro tres dos uno sos',
    );

    expect(parsed, isNotNull);
    expect(parsed!.name, 'mama');
    expect(parsed.phone, '+51987654321');
    expect(parsed.isEmergency, isTrue);
  });

  test('summarizes danger position and approximate distance', () {
    final detection = Detection(
      label: 'bache',
      confidence: 0.9,
      sourceSize: const Size(640, 480),
      rect: const Rect.fromLTWH(255, 260, 140, 120),
    );

    expect(
      summarizeDetectionsForSpeech([detection]),
      'Peligro al frente, mas adelante: bache.',
    );
  });

  test('summarizes left and right danger directions', () {
    final left = Detection(
      label: 'grieta',
      confidence: 0.8,
      sourceSize: const Size(640, 480),
      rect: const Rect.fromLTWH(40, 240, 100, 80),
    );
    final right = Detection(
      label: 'alcantarilla',
      confidence: 0.8,
      sourceSize: const Size(640, 480),
      rect: const Rect.fromLTWH(500, 240, 100, 80),
    );

    expect(dangerDirectionForSpeech(left), 'a la izquierda');
    expect(dangerDirectionForSpeech(right), 'a la derecha');
  });

  test('classifies safety labels by risk', () {
    expect(riskLevelForLabel('bache'), RiskLevel.danger);
    expect(riskLevelForLabel('grieta'), RiskLevel.danger);
    expect(riskLevelForLabel('alcantarilla'), RiskLevel.danger);
    expect(riskLevelForLabel('pothole'), RiskLevel.danger);
    expect(riskLevelForLabel('crack'), RiskLevel.danger);
    expect(riskLevelForLabel('open_manhole'), RiskLevel.danger);
    expect(riskLevelForLabel('auto'), RiskLevel.caution);
    expect(riskLevelForLabel('manzana'), RiskLevel.info);
  });

  test('finds only nearby floor dangers for alerts', () {
    final near = Detection(
      label: 'grieta',
      confidence: 0.9,
      sourceSize: const Size(640, 480),
      rect: const Rect.fromLTWH(220, 300, 180, 120),
    );
    final object = Detection(
      label: 'manzana',
      confidence: 0.9,
      sourceSize: const Size(640, 480),
      rect: const Rect.fromLTWH(260, 280, 120, 110),
    );

    expect(nearestNearbyDanger([object, near]), near);
    expect(
      summarizeDetectionsForSpeech([object]),
      contains('No detecto un riesgo alto inmediato'),
    );
  });

  test('summarizes full scene with position and distance', () {
    final bache = Detection(
      label: 'bache',
      confidence: 0.9,
      sourceSize: const Size(640, 480),
      rect: const Rect.fromLTWH(255, 260, 140, 120),
    );
    final person = Detection(
      label: 'persona',
      confidence: 0.8,
      sourceSize: const Size(640, 480),
      rect: const Rect.fromLTWH(500, 180, 100, 160),
    );
    final apple = Detection(
      label: 'manzana',
      confidence: 0.7,
      sourceSize: const Size(640, 480),
      rect: const Rect.fromLTWH(40, 240, 100, 80),
    );

    expect(
      summarizeSceneForSpeech([apple, person, bache]),
      'Veo un bache al frente, mas adelante, peligro; una persona a la derecha, mas adelante, precaucion; y una manzana a la izquierda, mas adelante.',
    );
  });

  test('summarizes two people as a natural group', () {
    final right = Detection(
      label: 'persona',
      confidence: 0.8,
      sourceSize: const Size(640, 480),
      rect: const Rect.fromLTWH(500, 170, 65, 65),
    );
    final left = Detection(
      label: 'persona',
      confidence: 0.85,
      sourceSize: const Size(640, 480),
      rect: const Rect.fromLTWH(40, 120, 140, 140),
    );

    expect(
      summarizeSceneForSpeech([right, left]),
      'Veo dos personas. Una esta a la izquierda, mas adelante y la otra esta a la derecha, mas adelante.',
    );
  });

  test(
    'keeps people already accepted by cloud detection in the description',
    () {
      final right = Detection(
        label: 'persona',
        confidence: 0.4,
        sourceSize: const Size(640, 480),
        rect: const Rect.fromLTWH(500, 170, 65, 65),
      );
      final left = Detection(
        label: 'persona',
        confidence: 0.45,
        sourceSize: const Size(640, 480),
        rect: const Rect.fromLTWH(40, 120, 140, 140),
      );

      expect(
        summarizeSceneForSpeech([right, left]),
        startsWith('Veo dos personas.'),
      );
    },
  );

  test('normalizes MobileVLM local output', () {
    expect(
      extractMobileVlmText({
        'choices': [
          {
            'message': {'content': 'Veo dos personas cerca.'},
          },
        ],
      }),
      'Veo dos personas cerca.',
    );
    expect(
      normalizeMobileVlmDescription('Respuesta: Hay una persona al frente.'),
      'Hay una persona al frente.',
    );
  });

  test('recognizes split MobileVLM asset chunks', () {
    expect(
      mobileVlmChunkBasePath('embed_tokens_per_layer.weights.part000'),
      'embed_tokens_per_layer.weights',
    );
    expect(mobileVlmChunkBasePath('config.txt'), isNull);
  });

  test('makes navigation prompts more precise and natural', () {
    expect(
      cleanNavigationDestination('la farmacia por favor a pie'),
      'la farmacia',
    );
    expect(navigationDistanceForSpeech(934), '930 metros');
    expect(navigationDistanceForSpeech(1250), '1.3 kilometros');

    expect(
      navigationCueForSpeech(
        targetName: 'la farmacia',
        direction: 'hacia el noreste',
        distanceMeters: 1250,
        accuracyMeters: 18,
        opening: true,
      ),
      contains('Te dare avisos de rumbo y distancia'),
    );
    expect(
      navigationCueForSpeech(
        targetName: 'la farmacia',
        direction: 'a la derecha',
        distanceMeters: 420,
        accuracyMeters: 95,
        movingAway: true,
      ),
      contains('Te estas alejando un poco'),
    );
  });

  test('explains missing local MobileVLM model folder', () {
    const status = MobileVlmModelStatus(
      isPlatformSupported: true,
      modelName: 'lfm2-vl-1.6b',
      modelId: 'LiquidAI/LFM2-VL-1.6B',
      modelPath: null,
      searchedPaths: [
        '/storage/emulated/0/Android/data/app/files/models/lfm2-vl-1.6b',
      ],
    );

    expect(status.isReady, isFalse);
    expect(status.speechMessage, contains('sin internet'));
    expect(status.speechMessage, contains('LFM 2 VL 1.6B'));
    expect(status.speechMessage, isNot(contains('lfm2-vl-1.6b')));
    expect(status.speechMessage, contains('descargar modelo visual'));
  });

  test('does not call MobileVLM usable until runtime is loaded', () {
    const status = MobileVlmModelStatus(
      isPlatformSupported: true,
      modelName: 'lfm2-vl-1.6b',
      modelId: 'LiquidAI/LFM2-VL-1.6B',
      modelPath:
          '/data/user/0/com.example.mi_app/app_flutter/models/lfm2-vl-1.6b',
      searchedPaths: [
        '/data/user/0/com.example.mi_app/app_flutter/models/lfm2-vl-1.6b',
      ],
    );

    expect(status.isReady, isTrue);
    expect(status.speechMessage, contains('Encontre los archivos'));
    expect(status.speechMessage, contains('Falta probar'));
  });

  testWidgets('renders detector app', (tester) async {
    await tester.pumpWidget(const YoloRealtimeApp());

    expect(find.text('Peligro'), findsWidgets);
    expect(find.text('Leer'), findsWidgets);
    expect(find.text('Leer ahora'), findsNothing);
  });
}
