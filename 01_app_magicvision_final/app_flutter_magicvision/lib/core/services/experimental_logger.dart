import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:path_provider/path_provider.dart';
import '../config/inference_image_config.dart';

abstract class ExperimentalLogger {
  Future<void> record(Map<String, Object?> event, {Uint8List? debugImage});
}

class FileExperimentalLogger implements ExperimentalLogger {
  FileExperimentalLogger({
    this.saveDebugImages = InferenceImageConfig.saveDebugImages,
  });
  final bool saveDebugImages;
  @override
  Future<void> record(
    Map<String, Object?> event, {
    Uint8List? debugImage,
  }) async {
    final directory = Directory(
      '${(await getApplicationSupportDirectory()).path}/experiments',
    );
    await directory.create(recursive: true);
    final log = File('${directory.path}/inference.jsonl');
    // Bound default metadata retention to two files, ~10 MB each.
    if (await log.exists() && await log.length() > 10 * 1024 * 1024) {
      final old = File('${directory.path}/inference.previous.jsonl');
      if (await old.exists()) await old.delete();
      await log.rename(old.path);
    }
    await log.writeAsString(
      '${jsonEncode(event)}\n',
      mode: FileMode.append,
      flush: true,
    );
    if (saveDebugImages && debugImage != null) {
      final id = event['request_id'].toString().replaceAll(
        RegExp('[^a-zA-Z0-9-]'),
        '',
      );
      await File(
        '${directory.path}/$id.jpg',
      ).writeAsBytes(debugImage, flush: true);
    }
  }
}
