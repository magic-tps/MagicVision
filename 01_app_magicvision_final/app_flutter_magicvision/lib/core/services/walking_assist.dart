import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;
import '../config/campus_contract.dart';
import '../models/detection_response.dart';

final Map<String, dynamic> campusPolicy =
    jsonDecode(campusPolicyJson) as Map<String, dynamic>;
String campusName(String name) => name == 'trash_bin' ? 'trash_can' : name;
String campusDirection(double x) => x < campusPolicy['direction']['left_max']
    ? 'left'
    : x > campusPolicy['direction']['right_min']
    ? 'right'
    : 'center';
String campusPriority(String name) =>
    (campusPolicy['priorities'] as Map).entries
            .firstWhere((e) => (e.value as List).contains(campusName(name)))
            .key
        as String;
String campusSpanish(String name) =>
    campusPolicy['names_es'][campusName(name)] as String;

class TrackedHazard {
  TrackedHazard(
    this.detection, {
    this.trackId,
    this.distance,
    this.approaching = false,
  });
  final Detection detection;
  final int? trackId;
  final double? distance;
  final bool approaching;
  late String direction, risk, message;
  late double score;
  late bool inPath;
  int? distanceApprox;
}

class CampusRiskEngine {
  TrackedHazard assess(TrackedHazard h) {
    final d = h.detection, b = d.bbox;
    final name = campusName(d.label), priority = campusPriority(d.label);
    final area = (b.right - b.left) * (b.bottom - b.top);
    h.direction = campusDirection((b.left + b.right) / 2);
    h.inPath =
        h.direction == 'center' &&
        b.bottom >= campusPolicy['geometry']['bottom_min'];
    h.risk = 'low';
    if (priority != 'contextual') {
      if (h.distance != null) {
        for (final level in ['critical', 'high', 'medium']) {
          if (h.distance! < campusPolicy['distance_thresholds_m'][level]) {
            h.risk = level;
            break;
          }
        }
        if (h.direction != 'center' && h.risk == 'critical') h.risk = 'high';
      } else if (name == 'stairs' ||
          (h.inPath && ['critical', 'high'].contains(priority))) {
        h.risk = 'high';
      } else if (priority == 'critical' ||
          area >= campusPolicy['geometry']['near_area']) {
        h.risk = 'medium';
      }
      if (['door', 'glass_door'].contains(name) &&
          h.direction == 'center' &&
          area >= campusPolicy['geometry']['large_area'] &&
          ['low', 'medium'].contains(h.risk)) {
        h.risk = 'high';
      }
    }
    h.distanceApprox = h.distance == null
        ? null
        : h.distance! < 1
        ? 0
        : (h.distance! + .5).floor();
    h.score =
        riskRank(h.risk) * 100.0 +
        {'contextual': 0, 'medium': 10, 'high': 20, 'critical': 30}[priority]!;
    h.score +=
        (h.direction == 'center' ? 20 : 0) +
        (h.approaching ? 10 : 0) +
        d.confidence +
        (h.distance == null ? 0 : 1 / (1 + h.distance!));
    final dist = h.distanceApprox == null
        ? ''
        : h.distanceApprox == 0
        ? ' a menos de un metro'
        : ' a aproximadamente ${h.distanceApprox} ${h.distanceApprox == 1 ? 'metro' : 'metros'}';
    h.message =
        '${riskRank(h.risk) >= 2 ? 'Cuidado. ' : ''}${campusSpanish(name)}$dist ${{'left': 'a la izquierda', 'center': 'al frente', 'right': 'a la derecha'}[h.direction]}.';
    return h;
  }
}

int riskRank(String risk) =>
    const {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}[risk]!;
double _median(List<double> values) {
  final s = [...values]..sort();
  final n = s.length;
  return n.isOdd ? s[n ~/ 2] : (s[n ~/ 2 - 1] + s[n ~/ 2]) / 2;
}

double _iou(BoundingBox a, BoundingBox b) {
  final intersection =
      math.max(0.0, math.min(a.right, b.right) - math.max(a.left, b.left)) *
      math.max(0.0, math.min(a.bottom, b.bottom) - math.max(a.top, b.top));
  return intersection /
      ((a.right - a.left) * (a.bottom - a.top) +
          (b.right - b.left) * (b.bottom - b.top) -
          intersection);
}

class ObjectTrack {
  ObjectTrack(this.id, this.classId, this.firstSeen, this.lastSeen);
  final int id, classId, firstSeen;
  int lastSeen;
  int? lastAlertTime;
  final bboxHistory = <BoundingBox>[];
  final confidenceHistory = <double>[];
  final distanceHistory = <(int, double)>[];
  double? ema;
}

class CampusTracker {
  final tracks = <int, ObjectTrack>{};
  int _next = 1;
  int? _lastTime;
  void reset() {
    tracks.clear();
    _lastTime = null; /* IDs never reused in a session. */
  }

  List<TrackedHazard> update(List<Detection> objects, int time) {
    if (_lastTime != null && time <= _lastTime!) {
      throw const FormatException('Fotograma repetido o fuera de orden');
    }
    _lastTime = time;
    final cfg = campusPolicy['tracking'];
    tracks.removeWhere((_, t) => time - t.lastSeen > cfg['max_gap_ms']);
    final pairs = <(double, int, int)>[];
    for (var i = 0; i < objects.length; i++) {
      for (final t in tracks.values) {
        if (t.classId == objects[i].classId) {
          pairs.add((_iou(objects[i].bbox, t.bboxHistory.last), i, t.id));
        }
      }
    }
    pairs.sort((a, b) => b.$1.compareTo(a.$1));
    final assigned = <int, int>{}, used = <int>{};
    for (final p in pairs) {
      if (p.$1 >= cfg['iou_threshold'] &&
          !assigned.containsKey(p.$2) &&
          used.add(p.$3)) {
        assigned[p.$2] = p.$3;
      }
    }
    final result = <TrackedHazard>[];
    for (var i = 0; i < objects.length; i++) {
      final d = objects[i], id = assigned[i] ?? _next++;
      final t = tracks.putIfAbsent(
        id,
        () => ObjectTrack(id, d.classId, time, time),
      );
      t.lastSeen = time;
      t.bboxHistory.add(d.bbox);
      t.confidenceHistory.add(d.confidence);
      if (t.bboxHistory.length > cfg['history_size']) {
        t.bboxHistory.removeAt(0);
        t.confidenceHistory.removeAt(0);
      }
      double? smooth;
      bool approaching = false;
      if (d.distanceM == null) {
        t.distanceHistory.clear();
        t.ema = null;
      } else {
        t.distanceHistory.add((time, d.distanceM!));
        if (t.distanceHistory.length > cfg['history_size']) {
          t.distanceHistory.removeAt(0);
        }
        final alpha = (cfg['ema_alpha'] as num).toDouble();
        t.ema = t.ema == null
            ? d.distanceM!
            : alpha * d.distanceM! + (1 - alpha) * t.ema!;
        smooth = cfg['smoothing'] == 'rolling_median'
            ? _median(t.distanceHistory.map((v) => v.$2).toList())
            : t.ema;
        approaching =
            t.distanceHistory.length >= cfg['approaching_min_samples'] &&
            time - t.distanceHistory.first.$1 >=
                cfg['approaching_min_span_ms'] &&
            t.distanceHistory.first.$2 - t.distanceHistory.last.$2 >=
                cfg['approaching_drop_m'];
      }
      result.add(
        TrackedHazard(
          d,
          trackId: id,
          distance: smooth,
          approaching: approaching,
        ),
      );
    }
    return result;
  }
}

class CampusAlertManager {
  final _last = <(int?, int), (int, int, double?)>{};
  void reset() => _last.clear();
  List<TrackedHazard> select(List<TrackedHazard> objects, int time) {
    final cfg = campusPolicy['alerts'];
    _last.removeWhere((_, v) => time - v.$1 > 60000);
    final selected = <TrackedHazard>[];
    for (final h in [...objects]..sort((a, b) => b.score.compareTo(a.score))) {
      final key = (h.trackId, h.detection.classId), old = _last[key];
      final changed =
          old == null ||
          riskRank(h.risk) > old.$2 ||
          (h.distance != null &&
              old.$3 != null &&
              old.$3! - h.distance! >= cfg['distance_change_m']);
      if (old != null && !changed && time - old.$1 < cfg['cooldown_ms']) {
        continue;
      }
      if (h.risk == 'critical' && selected.any((v) => v.risk == 'critical')) {
        continue;
      }
      selected.add(h);
      _last[key] = (time, riskRank(h.risk), h.distance);
      if (selected.length >= cfg['max_per_cycle']) break;
    }
    return selected;
  }
}

/// One capture at a time. A stop invalidates delivery even if HTTP completes later.
class WalkingAssistLoop {
  WalkingAssistLoop({
    required this.capture,
    required this.deliver,
    required this.cancel,
    required this.available,
    this.intervalMs = 1000,
    this.maxAgeMs = 3000,
    this.onStopped,
  });
  final Future<(List<Detection>, DateTime)?> Function() capture;
  final Future<void> Function(List<TrackedHazard>) deliver;
  final void Function() cancel;
  final bool Function() available;
  final void Function()? onStopped;
  final int intervalMs, maxAgeMs;
  final tracker = CampusTracker();
  final alerts = CampusAlertManager();
  final risk = CampusRiskEngine();
  Timer? _timer;
  bool running = false, _busy = false;
  int _generation = 0, _errors = 0;
  void start() {
    if (running) return;
    if (intervalMs < 200) throw ArgumentError('Intervalo minimo 200 ms');
    running = true;
    _errors = 0;
    _generation++;
    _tick();
  }

  void stop() {
    running = false;
    _generation++;
    _timer?.cancel();
    cancel();
    tracker.reset();
    alerts.reset();
    onStopped?.call();
  }

  Future<void> _tick() async {
    if (!running || _busy) return;
    if (!available()) {
      stop();
      return;
    }
    _busy = true;
    final gen = _generation;
    final watch = Stopwatch()..start();
    try {
      final result = await capture();
      if (!running || gen != _generation) return;
      if (result == null) {
        throw StateError('Captura no disponible');
      }
      final now = DateTime.now();
      final age = now.difference(result.$2).inMilliseconds;
      if (age < 0 || age > maxAgeMs) {
        throw StateError('Captura caducada');
      }
      final hazards = tracker
          .update(result.$1, result.$2.millisecondsSinceEpoch)
          .map(risk.assess)
          .toList();
      final selected = alerts.select(hazards, now.millisecondsSinceEpoch);
      if (selected.isNotEmpty) {
        for (final h in selected) {
          tracker.tracks[h.trackId]?.lastAlertTime = now.millisecondsSinceEpoch;
        }
        await deliver(selected);
      }
      _errors = 0;
    } catch (_) {
      if (running &&
          gen == _generation &&
          ++_errors >= campusPolicy['walking']['stop_after_errors']) {
        stop();
      }
    } finally {
      _busy = false;
      if (running) {
        _timer = Timer(
          Duration(
            milliseconds: math.max(0, intervalMs - watch.elapsedMilliseconds),
          ),
          _tick,
        );
      }
    }
  }
}
