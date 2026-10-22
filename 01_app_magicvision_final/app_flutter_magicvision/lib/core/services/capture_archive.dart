import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:path_provider/path_provider.dart';

/// Explicit research capture only. No location, contacts or device identifiers.
class CaptureArchive {
  const CaptureArchive({
    this.enabled = const bool.fromEnvironment(
      'CAPTURE_DATASET',
      defaultValue: false,
    ),
    this.consentConfirmed = const bool.fromEnvironment(
      'CAPTURE_CONSENT',
      defaultValue: false,
    ),
    this.sessionId = const String.fromEnvironment('CAPTURE_SESSION_ID'),
    this.routeId = const String.fromEnvironment('CAPTURE_ROUTE_ID'),
    this.maxCaptures = 100,
    this.directory,
  });
  final bool enabled;
  final bool consentConfirmed;
  final String sessionId;
  final String routeId;
  final int maxCaptures;
  final Directory? directory;

  bool get canRecord =>
      enabled &&
      consentConfirmed &&
      RegExp(r'^[a-zA-Z0-9_-]{1,64}$').hasMatch(sessionId) &&
      RegExp(r'^[a-zA-Z0-9_-]{1,64}$').hasMatch(routeId);

  Future<void> record({
    required String captureId,
    required Uint8List original,
    required Uint8List detector,
    required Map<String, Object?> metadata,
  }) async {
    if (!canRecord) return;
    if (!RegExp(r'^[a-zA-Z0-9-]{1,64}$').hasMatch(captureId)) {
      throw const FormatException('Invalid capture identifier');
    }
    final root =
        directory ??
        Directory(
          '${(await getApplicationSupportDirectory()).path}/dataset_captures',
        );
    await root.create(recursive: true);
    final entries = await root
        .list()
        .where((entry) => entry is Directory)
        .length;
    if (entries >= maxCaptures) return;
    final capture = Directory('${root.path}/$captureId');
    if (await capture.exists()) return;
    await capture.create();
    await File(
      '${capture.path}/original.jpg',
    ).writeAsBytes(stripPrivateJpegMetadata(original), flush: true);
    await File(
      '${capture.path}/detector.jpg',
    ).writeAsBytes(detector, flush: true);
    // Whitelist fields: callers cannot accidentally archive GPS or personal data.
    final allowed = <String, Object?>{
      for (final key in [
        'captured_at',
        'camera_source',
        'original_resolution',
        'original_resolution_reference',
        'stored_original_resolution',
        'inference_resolution',
        'orientation',
        'exif_orientation',
        'model_version',
        'frame_id',
        'camera_profile_id',
        'depth_model_sha256',
        'pixel_sha256',
        'performance',
        'detections',
      ])
        if (metadata.containsKey(key)) key: metadata[key],
      'capture_id': captureId,
      'session_id': sessionId,
      'route_id': routeId,
      'consent_confirmed': true,
      'annotation_status': 'unreviewed',
      'original_metadata_removed': true,
    };
    await File(
      '${capture.path}/metadata.json',
    ).writeAsString(jsonEncode(allowed), flush: true);
  }
}

/// Remove EXIF/XMP/IPTC/comments (including GPS), preserving compressed pixels.
/// Orientation is stored separately in the whitelisted capture metadata.
Uint8List stripPrivateJpegMetadata(Uint8List bytes) {
  if (bytes.length < 4 || bytes[0] != 0xff || bytes[1] != 0xd8) {
    throw const FormatException('JPEG required for research capture');
  }
  final output = BytesBuilder()..add(bytes.sublist(0, 2));
  var offset = 2;
  while (offset < bytes.length) {
    final start = offset;
    if (bytes[offset++] != 0xff) {
      throw const FormatException('Invalid JPEG marker');
    }
    while (offset < bytes.length && bytes[offset] == 0xff) {
      offset++;
    }
    if (offset >= bytes.length) throw const FormatException('Truncated JPEG');
    final marker = bytes[offset++];
    if (marker == 0xda || marker == 0xd9) {
      output.add(bytes.sublist(start));
      return output.takeBytes();
    }
    if (offset + 2 > bytes.length) {
      throw const FormatException('Truncated JPEG segment');
    }
    final length = (bytes[offset] << 8) | bytes[offset + 1];
    if (length < 2 || offset + length > bytes.length) {
      throw const FormatException('Invalid JPEG segment');
    }
    if (!(marker >= 0xe1 && marker <= 0xef) && marker != 0xfe) {
      output.add(bytes.sublist(start, offset + length));
    }
    offset += length;
  }
  throw const FormatException('Missing JPEG scan');
}
