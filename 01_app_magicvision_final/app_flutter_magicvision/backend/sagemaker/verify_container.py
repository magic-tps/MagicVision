"""Check the packaged checkpoint and serving contract in the selected runtime."""
import argparse
import hashlib
import io
import json
import os
import platform
import runpy
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import torch
    import torchvision
    import ultralytics
    with tempfile.TemporaryDirectory(prefix='magicvision-model-') as temporary:
        with tarfile.open(args.artifact) as archive:
            members = archive.getmembers()
            assert len(members) == 3
            assert {m.name for m in members} == {'best.pt', 'labels.txt', 'config.json'}
            for member in members:
                assert member.isfile()
                with archive.extractfile(member) as data:
                    Path(temporary, member.name).write_bytes(data.read())
        os.environ['MODEL_PATH'] = str(Path(temporary, 'best.pt'))
        os.environ['MODEL_DEVICE'] = args.device
        module = runpy.run_path(str(Path(__file__).with_name('serve.py')))
        started = time.perf_counter()
        client = module['create_app'](*module['load_model']()).test_client()
        startup_ms = round((time.perf_counter() - started) * 1000, 2)
        ping = client.get('/ping')
        assert ping.status_code == 200 and ping.json['device'] == args.device
        assert client.post('/invocations', data=b'bad', content_type='text/plain').status_code == 415
        assert client.post('/invocations', data=b'bad', content_type='image/jpeg').status_code == 422
        assert client.post('/invocations', data=b'x' * (2 * 1024 * 1024 + 1), content_type='image/jpeg').status_code == 413
        image = io.BytesIO()
        Image.new('RGB', (640, 480)).save(image, 'JPEG')
        latencies = []
        for _ in range(3):
            response = client.post('/invocations', data=image.getvalue(), content_type='image/jpeg')
            assert response.status_code == 200, response.data
            result = response.get_json()
            assert (result['image_width'], result['image_height']) == (640, 480)
            assert isinstance(result['detections'], list)
            assert result['processing_time_ms'] >= 0
            latencies.append(result['processing_time_ms'])
        with args.artifact.open('rb') as stream:
            model_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        report = {
            'date': datetime.now(timezone.utc).isoformat(),
            'serving_contract': 'passed', 'model_sha256': model_hash,
            'environment': {'platform': platform.platform(), 'python': platform.python_version(),
                'torch': torch.__version__, 'torchvision': torchvision.__version__,
                'ultralytics': ultralytics.__version__, 'cuda_runtime': torch.version.cuda,
                'cudnn': torch.backends.cudnn.version(), 'device': args.device,
                'gpu': torch.cuda.get_device_name(0) if args.device == 'cuda' else None},
            'checks': {'checkpoint_loaded': True, 'labels_matched': True,
                'startup_inference_passed': True, 'ping_passed': True,
                'jpeg_invocations_passed': True, 'invalid_requests_rejected': True},
            'startup_including_warmup_ms': startup_ms,
            'local_synthetic_processing_ms': latencies,
            'synthetic_image': {'width': 640, 'height': 480, 'detections_count': len(result['detections'])},
            'aws_validated': False,
            'note': 'Local smoke timings only; not SageMaker latency or accuracy validation.'}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
