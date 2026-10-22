import io
import json
from uuid import uuid4

import pytest
from botocore.exceptions import ClientError, ReadTimeoutError
from fastapi.testclient import TestClient
from PIL import Image

from api import MAX_REQUEST_BYTES, create_app


def jpeg(size=(640, 480)):
    buffer = io.BytesIO()
    Image.new('RGB', size).save(buffer, 'JPEG')
    return buffer.getvalue()


def output(detections=None):
    return {'processing_time_ms': 145, 'image_width': 640, 'image_height': 480,
            'detections': detections or []}


class Runtime:
    def __init__(self, body=None, error=None):
        self.body = output() if body is None else body
        self.error = error
        self.calls = []

    def invoke_endpoint(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return {'Body': io.BytesIO(json.dumps(self.body).encode())}


def test_multipart_round_trip_and_iam_endpoint_call():
    runtime = Runtime()
    client = TestClient(create_app(runtime, 'test-endpoint'))
    request_id = str(uuid4())
    result = client.post('/v1/inference', data={'request_id': request_id}, files={'image': ('frame.jpg', jpeg(), 'image/jpeg')})
    assert result.status_code == 200
    assert result.json()['request_id'] == request_id
    assert result.json()['detections'] == []
    assert runtime.calls[0]['EndpointName'] == 'test-endpoint'
    assert runtime.calls[0]['ContentType'] == 'image/jpeg'
    assert runtime.calls[0]['CustomAttributes'] == f'request_id={request_id}'


@pytest.mark.parametrize('error,status', [
    (ClientError({'Error': {'Code': 'ModelError', 'Message': 'private details'}}, 'InvokeEndpoint'), 503),
    (ReadTimeoutError(endpoint_url='https://private'), 504),
])
def test_cloud_failure_is_not_an_empty_scene(error, status):
    client = TestClient(create_app(Runtime(error=error), 'test'))
    result = client.post('/v1/inference', files={'image': ('frame.jpg', jpeg(), 'image/jpeg')})
    assert result.status_code == status
    assert 'detections' not in result.json()
    assert 'private' not in result.text


@pytest.mark.parametrize('body', [{}, [], {**output(), 'image_width': 0}, {**output(), 'image_width': 1280},
    output([{'class_id': 6, 'label': 'pothole', 'confidence': 1.1,
             'bbox': {'left': .2, 'top': .5, 'right': .6, 'bottom': .9}}])])
def test_invalid_model_response(body):
    client = TestClient(create_app(Runtime(body), 'test'))
    result = client.post('/v1/inference', files={'image': ('frame.jpg', jpeg(), 'image/jpeg')})
    assert result.status_code == 502


def test_invalid_image_and_oversized_request_never_invoke():
    runtime = Runtime()
    client = TestClient(create_app(runtime, 'test'))
    for data, kind, status in [(b'', 'image/jpeg', 415), (b'junk', 'image/jpeg', 422),
                              (jpeg((1920, 1080)), 'image/jpeg', 422), (jpeg(), 'image/png', 415)]:
        assert client.post('/v1/inference', files={'image': ('frame.jpg', data, kind)}).status_code == status
    assert client.post('/v1/inference', content=b'x' * (MAX_REQUEST_BYTES + 1)).status_code == 413
    assert runtime.calls == []


def test_multiple_detections_and_invalid_bbox():
    box = {'class_id': 6, 'label': 'pothole', 'confidence': .8,
           'bbox': {'left': .2, 'top': .5, 'right': .6, 'bottom': .9}}
    runtime = Runtime(output([box, box]))
    client = TestClient(create_app(runtime, 'test'))
    result = client.post('/v1/inference', files={'image': ('frame.jpg', jpeg(), 'image/jpeg')})
    assert len(result.json()['detections']) == 2
    box['bbox']['right'] = 600
    assert client.post('/v1/inference', files={'image': ('frame.jpg', jpeg(), 'image/jpeg')}).status_code == 502


def test_invalid_request_id_and_missing_image():
    runtime = Runtime()
    client = TestClient(create_app(runtime, 'test'))
    assert client.post('/v1/inference').status_code == 422
    assert client.post('/v1/inference', data={'request_id': 'bad'}, files={'image': ('frame.jpg', jpeg(), 'image/jpeg')}).status_code == 422
    assert not runtime.calls
