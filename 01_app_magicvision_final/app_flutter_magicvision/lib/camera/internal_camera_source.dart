import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/widgets.dart';
import '../core/services/inference_exception.dart';
import 'camera_frame.dart';
import 'camera_source.dart';

/// Optional explicit source for laboratory comparisons; never an automatic USB fallback.
class InternalCameraSource extends CameraSource {
  CameraController? _controller;
  final _frames = StreamController<CameraFrame>.broadcast();
  CameraStatus _status = CameraStatus.disconnected;
  String? _error;
  bool _closed = false;
  @override
  String get name => 'internal_camera';
  @override
  CameraStatus get status => _status;
  @override
  String? get errorMessage => _error;
  @override
  Stream<CameraFrame> get frames => _frames.stream;
  @override
  Future<void> initialize() async {
    if (_closed || _status == CameraStatus.ready) return;
    _status = CameraStatus.connecting;
    notifyListeners();
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) throw StateError('Camara no disponible');
      final controller = CameraController(
        cameras.firstWhere(
          (c) => c.lensDirection == CameraLensDirection.back,
          orElse: () => cameras.first,
        ),
        ResolutionPreset.high,
        enableAudio: false,
      );
      _controller = controller;
      await controller.initialize();
      if (_closed) {
        await controller.dispose();
        return;
      }
      _status = CameraStatus.ready;
    } catch (_) {
      _status = CameraStatus.error;
      _error = 'No se pudo abrir la camara interna';
    }
    if (!_closed) notifyListeners();
  }

  @override
  Future<Uint8List> captureFrame() async {
    final controller = _controller;
    if (_closed || status != CameraStatus.ready || controller == null) {
      throw const InferenceException(
        'camera_disconnected',
        'Camara desconectada',
      );
    }
    final picture = await controller.takePicture();
    final file = File(picture.path);
    try {
      final bytes = await file.readAsBytes();
      if (!_closed) {
        _frames.add(
          CameraFrame(
            bytes: bytes,
            timestamp: DateTime.now().toUtc(),
            source: name,
          ),
        );
      }
      return bytes;
    } finally {
      if (await file.exists()) await file.delete();
    }
  }

  @override
  Widget buildPreview() => _controller == null || status != CameraStatus.ready
      ? const SizedBox.expand()
      : CameraPreview(_controller!);
  @override
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    await _controller?.dispose();
    await _frames.close();
  }

  @override
  void dispose() {
    unawaited(close());
    super.dispose();
  }
}
