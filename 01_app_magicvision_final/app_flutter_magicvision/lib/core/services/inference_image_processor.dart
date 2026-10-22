import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import '../config/inference_image_config.dart';
import 'inference_exception.dart';

class PreparedInferenceImage {
  const PreparedInferenceImage(
    this.bytes,
    this.originalWidth,
    this.originalHeight,
    this.width,
    this.height, {
    this.exifOrientation = 1,
  });
  final Uint8List bytes;
  final int originalWidth, originalHeight, width, height;
  final int exifOrientation;
  // originalWidth/Height retain the existing oriented-coordinate convention.
  int get encodedOriginalWidth => exifOrientation >= 5 && exifOrientation <= 8
      ? originalHeight
      : originalWidth;
  int get encodedOriginalHeight => exifOrientation >= 5 && exifOrientation <= 8
      ? originalWidth
      : originalHeight;
}

class InferenceImageProcessor {
  const InferenceImageProcessor(this.config);
  final InferenceImageConfig config;
  Future<PreparedInferenceImage> prepare(Uint8List bytes) {
    config.validate();
    return compute(_prepare, (bytes, config));
  }
}

PreparedInferenceImage _prepare((Uint8List, InferenceImageConfig) input) {
  final (bytes, config) = input;
  if (bytes.isEmpty) {
    throw const InferenceException('empty_image', 'Imagen vacia');
  }
  img.Image? decoded;
  var exifOrientation = 1;
  try {
    final isJpeg = bytes.length >= 2 && bytes[0] == 0xff && bytes[1] == 0xd8;
    // image's JPEG decoder bakes rotation and clears its tag. Read the source
    // tag before decoding so the archived pixel reference stays recoverable.
    if (isJpeg) {
      exifOrientation = img.decodeJpgExif(bytes)?.imageIfd.orientation ?? 1;
    }
    decoded = img.decodeImage(bytes);
    if (!isJpeg) {
      exifOrientation = decoded?.exif.imageIfd.orientation ?? 1;
    }
  } catch (_) {
    throw const InferenceException('invalid_frame', 'Frame invalido');
  }
  if (decoded == null) {
    throw const InferenceException('invalid_frame', 'Frame invalido');
  }
  final oriented = img.bakeOrientation(decoded);
  final scale = math.min(
    1.0,
    math.min(
      config.maxWidth / oriented.width,
      config.maxHeight / oriented.height,
    ),
  );
  final resized = img.copyResize(
    oriented,
    width: math.max(1, (oriented.width * scale).round()),
    height: math.max(1, (oriented.height * scale).round()),
    interpolation: img.Interpolation.linear,
  );
  // bakeOrientation/copyResize retain EXIF, which can contain GPS. The
  // detector receives pixels in the oriented reference, never source EXIF.
  resized.exif = img.ExifData();
  return PreparedInferenceImage(
    Uint8List.fromList(img.encodeJpg(resized, quality: config.jpegQuality)),
    oriented.width,
    oriented.height,
    resized.width,
    resized.height,
    exifOrientation: exifOrientation,
  );
}
