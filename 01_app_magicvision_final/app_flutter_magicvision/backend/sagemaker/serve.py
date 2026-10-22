"""SageMaker full-checkpoint hosting: one model per process, inference only."""
import io
import base64
import hashlib
import json
import logging
import math
import os
import subprocess
import threading
import time
from pathlib import Path
from uuid import UUID

from flask import Flask, jsonify, request
from PIL import Image, ImageOps, JpegImagePlugin, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

log = logging.getLogger('magicvision.serving')
logging.basicConfig(level=logging.INFO, format='%(message)s')


def event(name, **fields):
    log.info(json.dumps({'event': name, **fields}, allow_nan=False))


def load_model():
    import torch
    import torchvision
    import ultralytics
    from ultralytics import YOLO

    started = time.perf_counter()
    model_path = Path(os.getenv('MODEL_PATH', str(
        Path(os.getenv('SM_MODEL_DIR', '/opt/ml/model')) / 'best.pt')))
    model_dir = model_path.parent
    config = json.loads((model_dir / 'config.json').read_text(encoding='utf-8'))
    if (not isinstance(config.get('imgsz'), int) or not 320 <= config['imgsz'] <= 1280
            or not 0 < config.get('confidence', 0) < 1
            or not 0 < config.get('iou', 0) <= 1
            or not isinstance(config.get('max_det'), int) or not 1 <= config['max_det'] <= 300):
        raise ValueError('Invalid bounded inference configuration')
    labels = (model_dir / 'labels.txt').read_text(encoding='utf-8').splitlines()
    selection = os.getenv('MODEL_DEVICE', 'auto').lower()
    if selection not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('MODEL_DEVICE must be auto, cpu or cuda')
    try:
        driver = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version',
            '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        driver_info = driver.stdout.strip() if driver.returncode == 0 else 'unavailable'
    except (OSError, subprocess.TimeoutExpired):
        driver_info = 'unavailable'
    cuda = torch.cuda.is_available()
    event('runtime', torch=torch.__version__, torchvision=torchvision.__version__,
          ultralytics=ultralytics.__version__, cuda_runtime=torch.version.cuda,
          cudnn=torch.backends.cudnn.version(), cuda_available=cuda,
          requested_device=selection, gpu_driver=driver_info)
    if selection == 'cuda' and not cuda:
        raise RuntimeError('CUDA requested but unavailable; refusing silent CPU fallback')
    device = 'cuda' if cuda and selection != 'cpu' else 'cpu'
    event('startup', device=device)
    if not model_path.is_file():
        raise FileNotFoundError('Mounted MODEL_PATH checkpoint is missing')
    with model_path.open('rb') as model_file:
        config['model_version'] = hashlib.file_digest(model_file, 'sha256').hexdigest()
    model = YOLO(str(model_path))
    if [model.names[i] for i in range(len(model.names))] != labels:
        raise RuntimeError('labels.txt does not match the checkpoint')
    model.to(device)
    # Exercise the real kernels before reporting readiness to SageMaker.
    model.predict(Image.new('RGB', (640, 480)), imgsz=config['imgsz'],
        conf=config['confidence'], iou=config['iou'], max_det=config['max_det'],
        device=device, verbose=False, save=False)
    if device == 'cuda':
        torch.cuda.synchronize()
    event('model_loaded', device=device, classes=len(labels),
          startup_time_ms=round((time.perf_counter() - started) * 1000, 2))
    return model, labels, config, device


def create_app(model, labels, config, device, depth_pipeline=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
    lock = threading.Lock()

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.name), error.code

    @app.get('/ping')
    def ping():
        return jsonify(status='ok', device=device), 200

    @app.post('/invocations')
    def invoke():
        started = time.perf_counter()
        status = 'error'
        inference_ms = None
        count = None
        request_id = None
        attributes = request.headers.get('X-Amzn-SageMaker-Custom-Attributes', '')
        if attributes.startswith('request_id='):
            try:
                request_id = str(UUID(attributes.removeprefix('request_id=')))
            except ValueError:
                pass
        try:
            if request.mimetype != 'image/jpeg':
                return jsonify(error='JPEG required'), 415
            try:
                # Decode JPEG directly; avoid Ultralytics Image.open auto-install hooks.
                with JpegImagePlugin.JpegImageFile(io.BytesIO(request.get_data())) as decoded:
                    if decoded.format != 'JPEG' or decoded.width > 1280 or decoded.height > 1280:
                        return jsonify(error='Invalid image size or format'), 422
                    image = ImageOps.exif_transpose(decoded).convert('RGB')
            except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
                return jsonify(error='Invalid JPEG'), 422
            if image.width > 1280 or image.height > 1280:
                return jsonify(error='Invalid oriented image size'), 422
            if attributes.startswith('magicvision_v2='):
                if depth_pipeline is None:
                    return jsonify(error='Campus22 Detection + Depth is not configured'), 503
                from detection_depth import FrameIdentity
                try:
                    if len(attributes) > 1024:
                        raise ValueError('Oversized metadata')
                    meta = json.loads(base64.urlsafe_b64decode(attributes.split('=', 1)[1]))
                    request_id = str(UUID(meta['request_id']))
                    UUID(meta['frame_id'])
                    identity = FrameIdentity.create(image, **meta)
                except (ValueError, TypeError, KeyError):
                    return jsonify(error='Invalid frame metadata'), 422
                with lock:
                    output = depth_pipeline.analyze(image, identity)
                count = len(output['detections'])
                status = 'success'
                return jsonify(output)
            with lock:
                infer_started = time.perf_counter()
                result = model.predict(image, imgsz=config['imgsz'], conf=config['confidence'],
                    iou=config['iou'], max_det=config['max_det'], device=device,
                    verbose=False, save=False)[0]
                inference_ms = round((time.perf_counter() - infer_started) * 1000, 2)
            detections = []
            for box in result.boxes:
                coords = [float(v) for v in box.xyxyn[0].tolist()]
                confidence = float(box.conf[0])
                class_value = float(box.cls[0])
                if not all(math.isfinite(v) for v in [*coords, confidence, class_value]):
                    raise ValueError('Non-finite model output')
                if not 0 <= confidence <= 1 or not class_value.is_integer():
                    raise ValueError('Invalid model score/class')
                class_id = int(class_value)
                if not 0 <= class_id < len(labels):
                    raise ValueError('Invalid model class')
                left, top, right, bottom = [min(1.0, max(0.0, v)) for v in coords]
                if right <= left or bottom <= top:
                    continue
                detections.append({'class_id': class_id, 'label': labels[class_id],
                    'confidence': confidence, 'bbox': {
                        'left': left, 'top': top, 'right': right, 'bottom': bottom}})
            count = len(detections)
            status = 'success'
            return jsonify(processing_time_ms=round((time.perf_counter() - started) * 1000, 2),
                image_width=image.width, image_height=image.height, detections=detections,
                **({'model_version': config['model_version']} if config.get('model_version') else {}))
        except HTTPException:
            raise
        except Exception as error:
            event('inference_error', error_type=type(error).__name__, device=device)
            return jsonify(error='Inference failed'), 500
        finally:
            event('request', request_id=request_id, status=status, device=device, inference_time_ms=inference_ms,
                  processing_time_ms=round((time.perf_counter() - started) * 1000, 2),
                  detections_count=count)

    return app


if __name__ == '__main__':
    from waitress import serve
    try:
        loaded = load_model()
        pipeline = None
        if os.getenv('ENABLE_DETECTION_DEPTH', 'false').lower() == 'true':
            from detection_depth import DetectionDepthPipeline, UltralyticsDepthEstimator
            model, labels, config, device = loaded
            depth = UltralyticsDepthEstimator(os.environ['DEPTH_MODEL_PATH'], device=device)
            profile_path = os.getenv('DEPTH_CALIBRATION_PATH')
            profile = json.loads(Path(profile_path).read_text(encoding='utf-8')) if profile_path else None
            pipeline = DetectionDepthPipeline(model, depth, config, device, calibration=profile)
        app = create_app(*loaded, depth_pipeline=pipeline)
    except Exception as error:
        event('startup_error', error_type=type(error).__name__)
        raise
    serve(app, host='0.0.0.0', port=8080, threads=2)
