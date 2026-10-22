"""Package an existing checkpoint for S3; never train or convert the model."""
import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=Path, required=True)
parser.add_argument('--labels', type=Path, default=Path('ml/configs/classes.txt'))
parser.add_argument('--output', type=Path, default=Path('backend/artifacts/model.tar.gz'))
parser.add_argument('--imgsz', type=int, choices=[640, 960], default=640)
parser.add_argument('--confidence', type=float, default=0.30)
args = parser.parse_args()
if not args.weights.is_file() or args.weights.suffix != '.pt':
    parser.error('An existing full .pt checkpoint is required')
if not args.labels.is_file():
    parser.error('Labels file is required')
if args.output.exists():
    parser.error('Output exists; choose a new output name')
if not 0 < args.confidence < 1:
    parser.error('confidence must be between 0 and 1')
from ultralytics import YOLO
labels = args.labels.read_text(encoding='utf-8').splitlines()
model = YOLO(str(args.weights))
if [model.names[i] for i in range(len(model.names))] != labels:
    parser.error('Labels do not match the checkpoint; do not label ten-class weights as eleven-class')
args.output.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(args.output, 'w:gz') as archive:
    archive.add(args.weights, arcname='best.pt')
    archive.add(args.labels, arcname='labels.txt')
    data = json.dumps({'imgsz': args.imgsz, 'confidence': args.confidence, 'iou': 0.65, 'max_det': 100,
        'model_version': hashlib.sha256(args.weights.read_bytes()).hexdigest()}).encode()
    info = tarfile.TarInfo('config.json')
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))
print(f'{args.output}: sha256={hashlib.sha256(args.output.read_bytes()).hexdigest()}')
