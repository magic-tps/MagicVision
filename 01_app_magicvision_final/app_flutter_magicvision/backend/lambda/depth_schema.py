"""Versioned same-frame response validation; v1 remains wire compatible."""
import base64
import json
import math
from datetime import datetime
from uuid import UUID
from campus_contract import LABELS, TAXONOMY
from schema import BoundingBox


def frame_attributes(metadata):
    validate_metadata(metadata)
    return 'magicvision_v2=' + base64.urlsafe_b64encode(json.dumps(metadata, separators=(',', ':')).encode()).decode()


def validate_metadata(meta):
    for key in ('request_id', 'frame_id'):
        UUID(meta[key])
    if not isinstance(meta['camera_id'], str) or not 1 <= len(meta['camera_id']) <= 128:
        raise ValueError('Invalid camera_id')
    if len(meta['timestamp']) > 64 or datetime.fromisoformat(meta['timestamp'].replace('Z', '+00:00')).tzinfo is None:
        raise ValueError('Timestamp requires timezone')


def validate_response(body, metadata, size):
    validate_metadata(metadata)
    if not isinstance(body, dict):
        raise ValueError('Response must be an object')
    if body.get('schema_version') != 2 or body.get('taxonomy') != TAXONOMY:
        raise ValueError('Campus22 v2 response required')
    if any(body.get(k) != v for k, v in metadata.items()):
        raise ValueError('Frame correlation mismatch')
    if (body.get('image_width'), body.get('image_height')) != tuple(size):
        raise ValueError('Image dimensions changed')
    for key in ('pixel_sha256', 'depth_model_sha256'):
        if len(body.get(key, '')) != 64 or any(c not in '0123456789abcdef' for c in body[key]):
            raise ValueError('Invalid frame/model digest')
    objects = body['detections']
    if not isinstance(objects, list) or len(objects) > 300:
        raise ValueError('Invalid detection count')
    for obj in objects:
        if not isinstance(obj, dict):
            raise ValueError('Detection must be an object')
        cid = obj['class_id']
        if type(cid) is not int or not 0 <= cid < len(LABELS) or obj['class_name'] != LABELS[cid] or obj['label'] != LABELS[cid]:
            raise ValueError('Class mismatch')
        BoundingBox.model_validate(obj['bbox'])
        confidence = obj['confidence']
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Invalid confidence')
        for key in ('distance_raw_m', 'distance_m', 'distance_smoothed_m'):
            value = obj[key]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 150):
                raise ValueError('Invalid distance')
        if obj['distance_m'] is not None and obj.get('depth_status') != 'calibrated':
            raise ValueError('Metric distance requires calibration')
        if obj['direction'] not in ('left', 'center', 'right') or obj['risk'] not in ('low', 'medium', 'high', 'critical'):
            raise ValueError('Invalid risk/direction')
    for key in ('detection_ms', 'depth_ms', 'fusion_ms', 'total_ms'):
        value = body['performance'][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid measured latency')
    return body
