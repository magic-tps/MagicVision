import io
import json
from uuid import uuid4

import botocore.session
import pytest
from botocore.validate import validate_parameters
from fastapi.testclient import TestClient
from PIL import Image

from api import create_app
from async_jobs import AsyncJobs
from testing.transport import MemoryS3, TestAsyncRuntime


def jpeg():
    data = io.BytesIO()
    Image.new('RGB', (640, 480)).save(data, 'JPEG')
    return data.getvalue()


def model_output():
    return {'processing_time_ms': 12, 'image_width': 640, 'image_height': 480,
            'detections': [{'class_id': 0, 'label': 'person', 'confidence': .8,
                'bbox': {'left': .2, 'top': .3, 'right': .6, 'bottom': .9}}]}


@pytest.fixture
def setup():
    s3 = MemoryS3()
    runtime = TestAsyncRuntime(s3)
    now = [1000.0]
    jobs = AsyncJobs(s3, runtime, 'private-test-bucket', 'test-endpoint', clock=lambda: now[0])
    client = TestClient(create_app(endpoint='test-endpoint', jobs=jobs, mode='async'))
    return client, jobs, runtime, s3, now


def submit(setup, warmup=False):
    client, jobs, runtime, s3, now = setup
    request_id = str(uuid4())
    response = client.post('/v1/inference/warmup' if warmup else '/v1/inference',
        data={'request_id': request_id},
        files=None if warmup else {'image': ('frame.jpg', jpeg(), 'image/jpeg')})
    assert response.status_code == 202, response.text
    body = response.json()
    assert body['request_id'] == request_id
    assert 'OutputLocation' not in body and 's3://' not in response.text
    validate_parameters(runtime.calls[-1], botocore.session.get_session().get_service_model(
        'sagemaker-runtime').operation_model('InvokeEndpointAsync').input_shape)
    stored = jobs.read(jobs.key(body['job_id']))
    reply = {'OutputLocation': 's3://private-test-bucket/' + stored['output_key'],
             'FailureLocation': 's3://private-test-bucket/' + stored['failure_key']}
    return body, stored, reply


def test_submit_poll_complete_and_cleanup(setup):
    client, jobs, runtime, s3, now = setup
    ticket, stored, reply = submit(setup)
    path = '/v1/inference/' + ticket['job_id']
    pending = client.get(path)
    assert pending.status_code == 202
    assert pending.headers['cache-control'] == 'no-store'
    runtime.complete(reply, model_output())
    done = client.get(path)
    assert done.status_code == 200
    assert done.json()['detections'][0]['label'] == 'person'
    assert done.json()['request_id'] == ticket['request_id']
    assert stored['input_key'] not in s3.objects
    assert stored['output_key'] not in s3.objects
    assert client.get(path).json() == done.json()


def test_warmup_does_not_return_scene_detections(setup):
    client, jobs, runtime, s3, now = setup
    ticket, stored, reply = submit(setup, warmup=True)
    assert stored['expires_at'] == now[0] + 900
    runtime.complete(reply, model_output())
    done = client.get('/v1/inference/' + ticket['job_id'])
    assert done.json() == {'request_id': ticket['request_id'], 'status': 'ready'}


def test_expired_capture_never_reaches_cerm(setup):
    client, jobs, runtime, s3, now = setup
    ticket, stored, reply = submit(setup)
    runtime.complete(reply, model_output())
    now[0] += 31
    result = client.get('/v1/inference/' + ticket['job_id'])
    assert result.status_code == 410
    assert 'detections' not in result.json()
    assert stored['input_key'] not in s3.objects


def test_cancel_marker_wins_over_late_concurrent_completion(setup):
    client, jobs, runtime, s3, now = setup
    ticket, stored, reply = submit(setup)
    path = '/v1/inference/' + ticket['job_id']
    assert client.delete(path).status_code == 200
    # A poll that already read the old job can write later; marker must still win.
    jobs.write({**stored, 'status': 'completed', 'result': model_output()})
    runtime.complete(reply, model_output())
    assert client.get(path).status_code == 410


@pytest.mark.parametrize('bad', [
    {}, {'error': 'private model failure'},
    {**model_output(), 'image_width': 100},
    {**model_output(), 'detections': [{'class_id': 0, 'label': 'wrong', 'confidence': .8,
      'bbox': {'left': .2, 'top': .3, 'right': .6, 'bottom': .9}}]},
])
def test_corrupt_output_is_not_success(setup, bad):
    client, jobs, runtime, s3, now = setup
    ticket, stored, reply = submit(setup)
    runtime.complete(reply, bad)
    result = client.get('/v1/inference/' + ticket['job_id'])
    assert result.status_code == 502
    assert 'detections' not in result.json()


def test_async_failure_is_reported_without_sensitive_payload(setup):
    client, jobs, runtime, s3, now = setup
    ticket, stored, reply = submit(setup)
    runtime.complete(reply, {'private': 'sensitive details'}, failure=True)
    result = client.get('/v1/inference/' + ticket['job_id'])
    assert result.status_code == 503
    assert 'sensitive' not in result.text


def test_unknown_and_invalid_job_are_not_found(setup):
    client = setup[0]
    assert client.get('/v1/inference/' + str(uuid4())).status_code == 404
    assert client.get('/v1/inference/invalid').status_code == 404


def test_output_location_cannot_escape_project_bucket(setup):
    jobs = setup[1]
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        jobs.output_key('s3://other-bucket/async-output/app/result.out')


def test_synchronous_backend_supports_readiness_without_changing_inference():
    client = TestClient(create_app(endpoint='existing', mode='sync'))
    request_id = str(uuid4())
    assert client.post('/v1/inference/warmup', data={'request_id': request_id}).json() == {
        'request_id': request_id, 'status': 'ready'}


def test_lambda_gateway_base64_multipart_reaches_async_job_and_poll(setup):
    import base64
    from types import SimpleNamespace
    import httpx
    from mangum import Mangum
    client, jobs, runtime, s3, now = setup
    handler = Mangum(create_app(endpoint='test-endpoint', jobs=jobs, mode='async'), lifespan='off')
    request_id = str(uuid4())
    request = httpx.Request('POST', 'https://example.test/v1/inference',
        data={'request_id': request_id}, files={'image': ('scene.jpg', jpeg(), 'image/jpeg')})
    payload = request.read()
    event = {'resource': '/{proxy+}', 'path': '/v1/inference', 'httpMethod': 'POST',
        'headers': dict(request.headers), 'multiValueHeaders': {}, 'isBase64Encoded': True,
        'body': base64.b64encode(payload).decode(), 'queryStringParameters': None,
        'requestContext': {'stage': 'prod', 'path': '/prod/v1/inference',
            'identity': {'sourceIp': '127.0.0.1'}, 'requestId': str(uuid4())}}
    reply = handler(event, SimpleNamespace())
    assert reply['statusCode'] == 202
    ticket = json.loads(reply['body'])
    assert ticket['request_id'] == request_id
    event.update(path='/v1/inference/' + ticket['job_id'], httpMethod='GET', body=None,
                 isBase64Encoded=False, headers={'accept': 'application/json'})
    assert handler(event, SimpleNamespace())['statusCode'] == 202
