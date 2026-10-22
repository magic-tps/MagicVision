import 'dart:typed_data';
import 'package:flutter/widgets.dart';
import 'camera_frame.dart';

enum CameraStatus { disconnected, connecting, ready, error }

abstract class CameraSource extends ChangeNotifier {
  String get name;
  CameraStatus get status;
  String? get errorMessage;
  Stream<CameraFrame> get frames;
  Future<void> initialize();
  Future<Uint8List> captureFrame();
  Widget buildPreview();
  Future<void> close();
}
