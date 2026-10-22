"""Run pretrained Depth on one real inventory frame; no training or metric claims."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / '01_app_magicvision_final/app_flutter_magicvision/backend/sagemaker'))


def main():
    # Keep this diagnostic's settings inside the workspace, preserving overrides.
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'test_deps/ultralytics_smoke'))
    import numpy as np
    import torch
    import ultralytics
    from PIL import Image
    from detection_depth import UltralyticsDepthEstimator, FrameIdentity

    active = json.loads((ROOT / 'configs/active_dataset.json').read_text(encoding='utf-8'))
    with (ROOT.parent / active['manifest']).open(encoding='utf-8') as stream:
        row = json.loads(next(stream))
    with Image.open(row['image']) as source:
        image = source.convert('RGB')
    identity = FrameIdentity.create(image, 'continuation', row['id'],
                                    datetime.now(timezone.utc).isoformat(), 'dataset-diagnostic')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    estimator = UltralyticsDepthEstimator(ROOT / 'models/yolo26m-depth.pt', device=device)
    result = estimator.estimate(image, identity)
    if result.identity != identity or result.depth_map.shape != (image.height, image.width):
        raise ValueError('Depth frame identity or resolution mismatch')
    valid = np.isfinite(result.depth_map) & (result.depth_map > 0)
    if not valid.any():
        raise ValueError('Depth returned no finite positive values')
    report = {
        'executed_at': datetime.now(timezone.utc).isoformat(),
        'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
        'device': device, 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
        'model_sha256': result.model_sha256, 'frame_id': result.identity.frame_id,
        'shape': list(result.depth_map.shape), 'finite_positive_fraction': float(valid.mean()),
        'depth_ms': result.inference_ms, 'training_performed': False, 'calibrated': False,
        'scope': 'One real dataset frame; cold inference, not metric accuracy or throughput',
    }
    (ROOT / 'reports/continuation_depth_smoke.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
