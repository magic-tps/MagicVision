import 'dart:async';
import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';
import '../core/services/inference_exception.dart';
import 'camera_frame.dart';
import 'camera_source.dart';

class UsbCameraSource extends CameraSource {
  UsbCameraSource({MethodChannel? channel, EventChannel? events})
    : _channel = channel ?? const MethodChannel('magicvision/usb_camera'),
      _events = events ?? const EventChannel('magicvision/usb_camera_events');
  final MethodChannel _channel;
  final EventChannel _events;
  final _frames = StreamController<CameraFrame>.broadcast();
  StreamSubscription<dynamic>? _subscription;
  CameraStatus _status = CameraStatus.disconnected;
  String? _error;
  int? _texture;
  double _width = 640, _height = 480;
  bool _closed = false;
  bool _capturing = false;
  int _generation = 0;
  @override
  String get name => 'arducam_uvc';
  @override
  CameraStatus get status => _status;
  @override
  String? get errorMessage => _error;
  @override
  Stream<CameraFrame> get frames => _frames.stream;

  @override
  Future<void> initialize() async {
    if (_closed) return;
    _subscription ??= _events.receiveBroadcastStream().listen(
      _event,
      onError: (_) {
        _setStatus(CameraStatus.error, 'No se pudo conectar la camara USB');
      },
    );
    _setStatus(CameraStatus.connecting);
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'initialize',
      );
      if (result != null && !_closed) _event(result);
    } on PlatformException catch (e) {
      _setStatus(CameraStatus.error, e.message ?? 'Permiso USB rechazado');
    } on MissingPluginException {
      _setStatus(CameraStatus.error, 'Camara USB disponible solo en Android');
    }
  }

  void _event(dynamic event) {
    if (_closed || event is! Map) return;
    _texture = event['texture_id'] as int? ?? _texture;
    _width = (event['width'] as num?)?.toDouble() ?? _width;
    _height = (event['height'] as num?)?.toDouble() ?? _height;
    final state = switch (event['state']) {
      'cameraReady' => CameraStatus.ready,
      'cameraConnecting' => CameraStatus.connecting,
      'error' => CameraStatus.error,
      _ => CameraStatus.disconnected,
    };
    _setStatus(state, event['message'] as String?);
  }

  void _setStatus(CameraStatus value, [String? message]) {
    if (_closed) return;
    if (value != CameraStatus.ready) _generation++;
    _status = value;
    _error = message;
    notifyListeners();
  }

  @override
  Future<Uint8List> captureFrame() async {
    if (_closed || _status != CameraStatus.ready) {
      throw const InferenceException(
        'camera_disconnected',
        'Camara desconectada',
      );
    }
    if (_capturing) {
      throw const InferenceException('camera_busy', 'Captura en curso');
    }
    final generation = _generation;
    _capturing = true;
    try {
      final bytes = await _channel
          .invokeMethod<Uint8List>('captureFrame')
          .timeout(const Duration(seconds: 5));
      if (_closed ||
          _status != CameraStatus.ready ||
          generation != _generation) {
        throw const InferenceException(
          'camera_disconnected',
          'Camara desconectada',
        );
      }
      if (bytes == null || bytes.isEmpty) {
        throw const InferenceException('invalid_frame', 'Frame invalido');
      }
      _frames.add(
        CameraFrame(
          bytes: bytes,
          timestamp: DateTime.now().toUtc(),
          source: name,
        ),
      );
      return bytes;
    } on PlatformException catch (e) {
      throw InferenceException(
        e.code,
        e.message ?? 'No se pudo capturar la imagen',
      );
    } on TimeoutException {
      throw const InferenceException(
        'capture_timeout',
        'La camara no entrego una imagen',
      );
    } finally {
      _capturing = false;
    }
  }

  @override
  Widget buildPreview() => _texture == null || status != CameraStatus.ready
      ? const SizedBox.expand()
      : FittedBox(
          fit: BoxFit.cover,
          child: SizedBox(
            width: _width,
            height: _height,
            child: Texture(textureId: _texture!),
          ),
        );

  @override
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    _generation++;
    await _subscription?.cancel();
    try {
      await _channel.invokeMethod<void>('dispose');
    } on PlatformException {
      /* Activity may already be detached. */
    } on MissingPluginException {
      /* Non-Android test host. */
    }
    await _frames.close();
  }

  @override
  void dispose() {
    unawaited(close());
    super.dispose();
  }
}
