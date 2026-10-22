import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:uuid/uuid.dart';
import '../config/inference_image_config.dart';
import '../models/detection_response.dart';
import 'inference_exception.dart';
import 'inference_readiness_service.dart';
import 'object_detection_service.dart';
import 'frame_detection_service.dart';

class AwsObjectDetectionService
    implements
        ObjectDetectionService,
        InferenceReadinessService,
        FrameDetectionService {
  AwsObjectDetectionService({
    String? baseUrl,
    InferenceImageConfig? config,
    http.Client Function()? clientFactory,
    this.depthEnabled = const bool.fromEnvironment('ENABLE_DETECTION_DEPTH'),
  }) : _baseUrl = baseUrl ?? InferenceImageConfig.apiBaseUrl,
       _config = config ?? InferenceImageConfig.environment(),
       _clientFactory = clientFactory ?? http.Client.new;
  final String _baseUrl;
  final bool depthEnabled;
  final InferenceImageConfig _config;
  final http.Client Function() _clientFactory;
  http.Client? _activeClient;
  Completer<void>? _cancelled;
  bool _disposed = false;
  int _generation = 0;
  Uri? _jobUri;

  Uri _uri([String suffix = '']) {
    final uri = Uri.tryParse(_baseUrl);
    if (uri == null ||
        uri.scheme != 'https' ||
        uri.host.isEmpty ||
        uri.userInfo.isNotEmpty ||
        uri.hasQuery ||
        uri.hasFragment) {
      throw const InferenceException(
        'configuration',
        'Servicio de analisis no configurado',
      );
    }
    final path = uri.path.replaceFirst(RegExp(r'/+$'), '');
    return uri.replace(path: '$path/v1/inference$suffix');
  }

  void _check(int generation) {
    if (_disposed || generation != _generation) {
      throw const InferenceException('cancelled', 'Analisis cancelado');
    }
  }

  Future<T> _run<T>(
    Duration budget,
    Future<T> Function(http.Client, int) action,
  ) async {
    if (_disposed) {
      throw const InferenceException('disposed', 'Servicio cerrado');
    }
    if (_activeClient != null) {
      throw const InferenceException('busy', 'Analisis en curso');
    }
    _config.validate();
    _uri();
    final client = _clientFactory();
    final generation = _generation;
    final cancelled = Completer<void>();
    _activeClient = client;
    _cancelled = cancelled;
    bool completed = false;
    try {
      final value = await Future.any<T>([
        action(client, generation),
        cancelled.future.then<T>(
          (_) =>
              throw const InferenceException('cancelled', 'Analisis cancelado'),
        ),
      ]).timeout(budget);
      _check(generation);
      completed = true;
      return value;
    } on TimeoutException {
      throw const InferenceException(
        'timeout',
        'Tiempo de espera agotado. Intenta analizar de nuevo',
      );
    } on SocketException {
      throw const InferenceException('network', 'Sin conexion a internet');
    } on http.ClientException {
      _check(generation);
      throw const InferenceException(
        'network',
        'Sin conexion a internet o servicio inaccesible',
      );
    } on FormatException {
      throw const InferenceException(
        'invalid_response',
        'Respuesta de analisis invalida',
      );
    } finally {
      if (!completed) {
        _generation++;
        final job = _jobUri;
        if (job != null) unawaited(_cancelJob(job));
      }
      client.close();
      _activeClient = null;
      _cancelled = null;
      _jobUri = null;
    }
  }

  Future<(int, Map<String, dynamic>)> _send(
    http.Client client,
    http.BaseRequest request,
    int generation,
  ) async {
    _check(generation);
    request.followRedirects = false;
    request.headers['Accept'] = 'application/json';
    return await (() async {
      final response = await client.send(request);
      final bytes = BytesBuilder(copy: false);
      await for (final part in response.stream) {
        _check(generation);
        bytes.add(part);
        if (bytes.length > 1024 * 1024) {
          throw const FormatException('Respuesta demasiado grande');
        }
      }
      _check(generation);
      if (response.statusCode == 410) {
        throw const InferenceException(
          'stale_response',
          'La captura caduco. Analiza de nuevo',
        );
      }
      if (response.statusCode != 200 && response.statusCode != 202) {
        final code = response.statusCode;
        throw InferenceException(
          'http_$code',
          code >= 500
              ? 'Servicio de analisis no disponible'
              : 'No se pudo analizar la imagen',
        );
      }
      final type = (response.headers['content-type'] ?? '')
          .split(';')
          .first
          .trim()
          .toLowerCase();
      if (type != 'application/json') {
        throw const FormatException('Content-Type invalido');
      }
      final body = jsonDecode(utf8.decode(bytes.takeBytes()));
      if (body is! Map<String, dynamic>) {
        throw const FormatException('JSON invalido');
      }
      return (response.statusCode, body);
    })().timeout(_config.timeout);
  }

  Future<Map<String, dynamic>> _resolve(
    http.Client client,
    int generation,
    String requestId,
    (int, Map<String, dynamic>) response,
  ) async {
    String? expectedJob;
    while (true) {
      _check(generation);
      final (status, body) = response;
      if (body['request_id'] != requestId) {
        throw const InferenceException(
          'stale_response',
          'Respuesta de otra captura descartada',
        );
      }
      if (status == 200) return body;
      final job = body['job_id'];
      final delay = body['poll_after_ms'];
      if (body['status'] != 'processing' ||
          job is! String ||
          !RegExp(
            r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$',
          ).hasMatch(job) ||
          (expectedJob != null && expectedJob != job) ||
          delay is! int ||
          delay < 100 ||
          delay > 5000) {
        throw const FormatException(
          'Identificador o estado de trabajo invalido',
        );
      }
      expectedJob = job;
      final jobUri = _uri('/$job');
      _jobUri = jobUri;
      await Future<void>.delayed(Duration(milliseconds: delay));
      _check(generation);
      response = await _send(client, http.Request('GET', jobUri), generation);
    }
  }

  @override
  Future<void> ensureReady() =>
      _run<void>(_config.warmupTimeout, (client, generation) async {
        final id = const Uuid().v4();
        final request = http.MultipartRequest('POST', _uri('/warmup'))
          ..fields['request_id'] = id;
        final body = await _resolve(
          client,
          generation,
          id,
          await _send(client, request, generation),
        );
        if (body['status'] != 'ready') {
          throw const FormatException('Servicio no preparado');
        }
      });

  @override
  Future<DetectionResponse> detect(
    Uint8List imageBytes, {
    String? requestId,
    void Function()? onProcessing,
  }) => detectFrame(
    imageBytes,
    requestId: requestId ?? const Uuid().v4(),
    capturedAt: DateTime.now().toUtc(),
    cameraId: 'unspecified',
    onProcessing: onProcessing,
  );

  @override
  Future<DetectionResponse> detectFrame(
    Uint8List imageBytes, {
    required String requestId,
    required DateTime capturedAt,
    required String cameraId,
    void Function()? onProcessing,
  }) async {
    if (imageBytes.isEmpty) {
      throw const InferenceException('empty_image', 'Imagen vacia');
    }
    return _run<DetectionResponse>(_config.maxCaptureAge, (
      client,
      generation,
    ) async {
      final id = requestId;
      final stamp = capturedAt.toUtc().toIso8601String();
      final uri = _uri();
      final request =
          http.MultipartRequest(
              'POST',
              depthEnabled
                  ? uri.replace(
                      path: uri.path.replaceFirst(
                        '/v1/inference',
                        '/v2/inference',
                      ),
                    )
                  : uri,
            )
            ..fields['request_id'] = id
            ..fields['timestamp'] = stamp
            ..files.add(
              http.MultipartFile.fromBytes(
                'image',
                imageBytes,
                filename: 'frame.jpg',
                contentType: MediaType('image', 'jpeg'),
              ),
            );
      if (depthEnabled) {
        request.fields.addAll({'frame_id': id, 'camera_id': cameraId});
      }
      final response = await _send(client, request, generation);
      _check(generation);
      onProcessing?.call();
      final result = DetectionResponse.fromJson(
        await _resolve(client, generation, id, response),
      );
      if (depthEnabled &&
          (result.frameId != id ||
              result.timestamp != stamp ||
              result.cameraId != cameraId)) {
        throw const FormatException('Respuesta de otro fotograma');
      }
      return result;
    });
  }

  Future<void> _cancelJob(Uri uri) async {
    final client = _clientFactory();
    try {
      final request = http.Request('DELETE', uri)..followRedirects = false;
      final response = await client
          .send(request)
          .timeout(const Duration(seconds: 2));
      await response.stream.drain<void>().timeout(const Duration(seconds: 2));
    } catch (_) {
      // A cancelled/orphan job also expires in the backend. Never deliver its result.
    } finally {
      client.close();
    }
  }

  @override
  void cancel() {
    _generation++;
    final cancelled = _cancelled;
    if (cancelled != null && !cancelled.isCompleted) cancelled.complete();
    _activeClient?.close();
  }

  @override
  void dispose() {
    _disposed = true;
    cancel();
  }
}
