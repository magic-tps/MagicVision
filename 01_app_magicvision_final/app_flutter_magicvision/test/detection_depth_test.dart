import 'dart:async';
import 'package:flutter_test/flutter_test.dart';
import 'package:mi_app/core/models/detection_response.dart';
import 'package:mi_app/core/services/walking_assist.dart';
import 'package:mi_app/main.dart'
    show groundedMobileVlmSpeech;

Detection sample({int id = 14, double? distance}) => Detection(
  classId: id,
  label: detectionClassNames[id],
  confidence: .9,
  bbox: BoundingBox(left: .2, top: .1, right: .8, bottom: .95),
  distanceM: distance,
  depthStatus: distance == null ? null : 'calibrated',
);

void main() {
  test('all 22 classes have explicit priority translation and risk', () {
    expect(detectionClassNames.length, 22);
    expect(detectionClassNames[10], 'trash_can');
    expect(detectionClassNames.sublist(18), [
      'construction_barrier',
      'handrail',
      'cone',
      'fire_extinguisher',
    ]);
    for (var id = 0; id < 22; id++) {
      final h = CampusRiskEngine().assess(TrackedHazard(sample(id: id)));
      expect(h.message, isNotEmpty);
      expect(h.risk, isNotEmpty);
    }
    expect(() => sample(distance: double.nan), throwsFormatException);
  });
  test(
    'glass door precedes close contextual extinguisher and emits rounded calibrated distance',
    () {
      final risk = CampusRiskEngine();
      final door = risk.assess(
        TrackedHazard(sample(distance: 1.83), distance: 1.83),
      );
      final extinguisher = risk.assess(
        TrackedHazard(sample(id: 21, distance: .8), distance: .8),
      );
      expect(door.risk, 'high');
      expect(door.message, contains('aproximadamente 2 metros al frente'));
      expect(extinguisher.risk, 'low');
      expect(door.score, greaterThan(extinguisher.score));
      expect(
        risk.assess(TrackedHazard(sample())).message,
        isNot(contains('metro')),
      );
    },
  );
  test(
    'tracker preserves identity median approach and clears missing distances',
    () {
      final tracker = CampusTracker();
      TrackedHazard? h;
      for (var i = 0; i < 4; i++) {
        h = tracker.update([
          sample(distance: [3.8, 3.2, 2.6, 2.1][i]),
        ], i * 1000).single;
        expect(h.trackId, 1);
      }
      expect(h!.distance, closeTo(2.9, .0001));
      expect(h.approaching, isTrue);
      expect(tracker.update([sample()], 4000).single.distance, isNull);
      expect(() => tracker.update([], 4000), throwsFormatException);
      expect(tracker.update([sample()], 8001).single.trackId, 2);
    },
  );
  test(
    'cooldown permits escalation and at most one critical plus one secondary',
    () {
      final manager = CampusAlertManager(), risk = CampusRiskEngine();
      TrackedHazard h(double d, {int id = 14, int track = 1}) => risk.assess(
        TrackedHazard(
          sample(id: id, distance: d),
          trackId: track,
          distance: d,
        ),
      );
      expect(manager.select([h(3)], 0).length, 1);
      expect(manager.select([h(2.9)], 1000), isEmpty);
      expect(manager.select([h(1.8)], 2000).length, 1);
      final selected = manager.select([
        h(.8),
        h(.7, id: 5, track: 2),
        h(.6, id: 21, track: 3),
      ], 3000);
      expect(selected.length, 2);
      expect(selected.where((h) => h.risk == 'critical').length, 1);
    },
  );
  test('walking stop cancels delivery of an in-flight response', () async {
    final pending = Completer<(List<Detection>, DateTime)?>();
    var calls = 0, spoken = 0, cancelled = 0;
    final loop = WalkingAssistLoop(
      capture: () {
        calls++;
        return pending.future;
      },
      deliver: (_) async {
        spoken++;
      },
      cancel: () {
        cancelled++;
      },
      available: () => true,
    );
    loop.start();
    loop.start();
    expect(calls, 1);
    loop.stop();
    pending.complete(([sample()], DateTime.now()));
    await Future<void>.delayed(Duration.zero);
    expect(spoken, 0);
    expect(cancelled, 1);
    expect(loop.running, isFalse);
  });
  test('VLM cannot provide metric claims or a clear path', () {
    for (final text in [
      'Hay una puerta a dos metros al frente.',
      'El camino esta libre para caminar.',
      'La puerta de vidrio esta cerrada.',
    ]) {
      expect(
        groundedMobileVlmSpeech(
          vlmDescription: text,
          analysis: null,
          detections: [],
        ),
        isNot(contains(text)),
      );
    }
  });
}
