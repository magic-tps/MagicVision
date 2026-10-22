"""Offline safety/contract tests: no AWS credentials, Docker or network used."""
import copy
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import botocore.session
import pytest
from botocore.validate import validate_parameters
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime
import workflow
from runtime import Aws, CommandFailure, Stop, valid_response


@pytest.fixture
def flow(tmp_path):
    return workflow.Workflow({'root': str(tmp_path), 'region': 'us-west-2',
        'account': '123456789012', 'bucket': 'magicvision-123456789012-us-west-2',
        'model': 'magicvision-yolo26m', 'endpoint': 'magicvision-yolo26m-async',
        'endpoint_config': 'magicvision-yolo26m-async', 'instance': 'ml.g4dn.xlarge',
        'repository': 'magicvision-yolo26m', 'ecr_uri': '123456789012.dkr.ecr.us-west-2.amazonaws.com/magicvision-yolo26m',
        'registry': '123456789012.dkr.ecr.us-west-2.amazonaws.com',
        'role_arn': '', 'torch_flavor': 'cu118', 'test_image': ''})


class FakeAws:
    """Validate every emitted JSON against real SDK operation shapes before faking responses."""
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []
        self.session = botocore.session.get_session()

    def call(self, service, operation, payload=None, **kwargs):
        model = self.session.get_service_model('s3' if service == 's3api' else service)
        method = next(name for name in model.operation_names
                      if ''.join(c for c in name if c.isalnum()).lower() == operation.replace('-', '').lower())
        shape = model.operation_model(method).input_shape
        if shape is not None:
            # get-object/put-object Body/filename are CLI positional extras.
            validate_parameters(payload or {}, shape)
        self.calls.append((service, operation, payload))
        response = self.responses.get((service, operation), {})
        return response(payload) if callable(response) else copy.deepcopy(response)


def output():
    return {'processing_time_ms': 10.5, 'image_width': 640, 'image_height': 480,
            'detections': [{'class_id': 0, 'label': 'person', 'confidence': .9,
                'bbox': {'left': .1, 'top': .2, 'right': .3, 'bottom': .4}}]}


@pytest.mark.parametrize('mutation', [
    lambda p: p['detections'][0].update(confidence=float('nan')),
    lambda p: p['detections'][0].update(label='wrong'),
    lambda p: p['detections'][0]['bbox'].update(left=-.1),
    lambda p: p['detections'][0]['bbox'].update(left=.8),
    lambda p: p.update(image_width=641),
    lambda p: p.update(detections={}),
    lambda p: p.update(processing_time_ms=float('inf')),
])
def test_reject_corrupt_model_response(mutation):
    body = output()
    mutation(body)
    with pytest.raises(Stop):
        valid_response(body, ['person'], (640, 480))


def test_empty_detections_are_valid():
    body = output()
    body['detections'] = []
    assert valid_response(body, ['person'], (640, 480)) == body


def test_access_denied_never_becomes_resource_absent(monkeypatch):
    def denied(*args, **kwargs):
        raise CommandFailure(['aws', 's3api', 'head-bucket'], '(403) Forbidden AccessDenied', 254)
    monkeypatch.setattr(runtime, 'run', denied)
    with pytest.raises(CommandFailure):
        Aws('us-west-2').call('s3api', 'head-bucket', {'Bucket': 'example'}, missing=runtime.MISSING_S3)


def test_missing_region_does_not_default(monkeypatch):
    monkeypatch.delenv('AWS_REGION', raising=False)
    monkeypatch.delenv('AWS_DEFAULT_REGION', raising=False)
    def absent(*args, **kwargs):
        raise CommandFailure(['aws', 'configure', 'get', 'region'], '', 1)
    monkeypatch.setattr(runtime, 'run', absent)
    with pytest.raises(Stop, match='assigned region'):
        runtime.discover_region()


def test_describe_defaults_do_not_break_idempotence():
    assert workflow.contains({'Image': 'x', 'Mode': 'SingleModel'}, {'Image': 'x'})
    assert not workflow.contains({'Image': 'other'}, {'Image': 'x'})


def test_no_endpoint_creation_without_explicit_confirmation(flow, monkeypatch):
    flow.aws = FakeAws({('sagemaker', 'describe-endpoint-config'): flow.endpoint_configuration(),
                        ('sagemaker', 'describe-endpoint'): None})
    monkeypatch.setattr(flow, 'check_quota', lambda: None)
    monkeypatch.setattr('builtins.input', lambda _: '')
    with pytest.raises(Stop, match='confirmation'):
        flow.create_endpoint()
    assert not any(operation == 'create-endpoint' for _, operation, _ in flow.aws.calls)


def test_changed_remote_instance_stops_before_confirmation(flow, monkeypatch):
    config = flow.endpoint_configuration()
    config['ProductionVariants'][0]['InstanceType'] = 'ml.p5.48xlarge'
    flow.aws = FakeAws({('sagemaker', 'describe-endpoint-config'): config})
    monkeypatch.setattr('builtins.input', lambda _: pytest.fail('must not ask to create changed config'))
    with pytest.raises(Stop, match='differs'):
        flow.create_endpoint()


def test_existing_endpoint_is_reused_without_creation(flow, monkeypatch):
    flow.aws = FakeAws({('sagemaker', 'describe-endpoint-config'): flow.endpoint_configuration(),
        ('sagemaker', 'describe-endpoint'): {'EndpointConfigName': flow.c['endpoint_config']}})
    monkeypatch.setattr(flow, 'wait_endpoint', lambda: {})
    flow.create_endpoint()
    assert not any(operation == 'create-endpoint' for _, operation, _ in flow.aws.calls)


def test_endpoint_config_is_async_and_idempotent(flow, monkeypatch):
    desired = flow.endpoint_configuration()
    flow.aws = FakeAws({('sagemaker', 'describe-endpoint-config'): {**desired, 'CreationTime': 'ignored'}})
    monkeypatch.setattr(flow, 'check_quota', lambda: None)
    flow.create_endpoint_config()
    assert not any(op == 'create-endpoint-config' for _, op, _ in flow.aws.calls)
    flow.aws.responses[('sagemaker', 'describe-endpoint-config')] = None
    flow.create_endpoint_config()
    payload = flow.aws.calls[-1][2]
    assert payload['AsyncInferenceConfig']['ClientConfig']['MaxConcurrentInvocationsPerInstance'] == 1
    assert payload['ProductionVariants'][0]['InitialInstanceCount'] == 1


def test_autoscaling_payloads_and_wake_policy(flow, monkeypatch):
    flow.aws = FakeAws({
        ('application-autoscaling', 'put-scaling-policy'): {'PolicyARN': 'arn:aws:autoscaling:us-west-2:123456789012:scalingPolicy:example'},
        ('application-autoscaling', 'describe-scalable-targets'): {'ScalableTargets': [{'MinCapacity': 0, 'MaxCapacity': 1}]}})
    monkeypatch.setattr(flow, 'wait_endpoint', lambda: {})
    flow.configure_autoscaling()
    register = next(p for _, op, p in flow.aws.calls if op == 'register-scalable-target')
    assert (register['MinCapacity'], register['MaxCapacity']) == (0, 1)
    alarm = next(p for _, op, p in flow.aws.calls if op == 'put-metric-alarm')
    assert alarm['MetricName'] == 'HasBacklogWithoutCapacity'


def test_ecr_payload_is_valid_and_repository_reused(flow):
    flow.aws = FakeAws({('ecr', 'describe-repositories'): None})
    flow.create_ecr()
    assert flow.aws.calls[-1][2]['imageTagMutability'] == 'IMMUTABLE'
    flow.aws.calls.clear()
    flow.aws.responses[('ecr', 'describe-repositories')] = {'repositories': [{}]}
    flow.create_ecr()
    assert len(flow.aws.calls) == 1


def test_s3_payloads_keep_public_access_blocked(flow):
    flow.aws = FakeAws({('s3api', 'head-bucket'): None,
        ('s3api', 'get-bucket-location'): {'LocationConstraint': 'us-west-2'},
        ('s3api', 'get-bucket-encryption'): {'ServerSideEncryptionConfiguration': {'Rules': [
            {'ApplyServerSideEncryptionByDefault': {'SSEAlgorithm': 'AES256'}}]}},
        ('s3api', 'head-object'): None})
    flow.create_s3()
    block = next(p for _, op, p in flow.aws.calls if op == 'put-public-access-block')
    assert all(block['PublicAccessBlockConfiguration'].values())


def test_model_upload_refuses_overwriting_existing_object(flow):
    flow.archive.parent.mkdir(parents=True)
    flow.archive.write_bytes(b'full-model')
    flow.aws = FakeAws({('s3api', 'head-object'): {'Metadata': {'sha256': 'another'}}})
    with pytest.raises(Stop, match='No overwrite'):
        flow.upload_model()
    assert not any(op == 'put-object' for _, op, _ in flow.aws.calls)


def test_zero_quota_does_not_change_instance(flow):
    flow.aws = FakeAws({('service-quotas', 'list-service-quotas'): {'Quotas': [
        {'QuotaName': 'ml.g4dn.xlarge for endpoint usage', 'Value': 0, 'QuotaCode': 'L-test'}]}})
    with pytest.raises(Stop, match='No usable endpoint quota'):
        flow.check_quota()
    assert flow.c['instance'] == 'ml.g4dn.xlarge'


def test_role_discovery_stops_when_ambiguous(flow):
    flow.aws = FakeAws({('iam', 'list-roles'): {'Roles': []}})
    with pytest.raises(Stop, match='MV_EXECUTION_ROLE_ARN'):
        flow.resolve_role()
    assert (flow.state / 'execution-role-policy.json').exists()
    assert not any(op.startswith('create-') for _, op, _ in flow.aws.calls)


def test_destroy_requires_confirmation(flow, monkeypatch):
    flow.aws = FakeAws({('sagemaker', 'describe-endpoint'): {'EndpointName': flow.c['endpoint']}})
    monkeypatch.setattr('builtins.input', lambda _: 'no')
    with pytest.raises(Stop, match='not confirmed'):
        flow.destroy_endpoint()
    assert not any(op.startswith(('delete-', 'deregister-')) for _, op, _ in flow.aws.calls)


def test_direct_upload_requires_successful_docker_proof(flow, monkeypatch):
    monkeypatch.setattr(flow, 'validate_archive', lambda: None)
    with pytest.raises(Stop, match='test-docker'):
        flow.require_docker_test()


@pytest.fixture
def serving():
    source = Path(__file__).resolve().parents[3] / 'backend/sagemaker/serve.py'
    spec = importlib.util.spec_from_file_location('test_serving_module', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Model:
    def __init__(self, score=.8, error=None):
        self.calls = 0
        self.score = score
        self.error = error

    def predict(self, *args, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        box = SimpleNamespace(xyxyn=[SimpleNamespace(tolist=lambda: [.1, .2, .4, .8])],
                              cls=[0], conf=[self.score])
        return [SimpleNamespace(boxes=[box])]


def jpeg():
    output = io.BytesIO()
    Image.new('RGB', (640, 480)).save(output, 'JPEG')
    return output.getvalue()


def test_server_reuses_model_and_returns_valid_json(serving):
    model = Model()
    client = serving.create_app(model, ['person'], workflow.CONFIG, 'cpu').test_client()
    assert client.get('/ping').json == {'status': 'ok', 'device': 'cpu'}
    for _ in range(2):
        response = client.post('/invocations', data=jpeg(), content_type='image/jpeg')
        assert response.status_code == 200
        valid_response(response.json, ['person'], (640, 480))
    assert model.calls == 2


@pytest.mark.parametrize('data,mime,status', [(b'bad', 'text/plain', 415),
    (b'bad', 'image/jpeg', 422), (b'x' * (2 * 1024 * 1024 + 1), 'image/jpeg', 413)],
    ids=['mime', 'invalid-jpeg', 'oversized'])
def test_server_errors_are_json(serving, data, mime, status):
    client = serving.create_app(Model(), ['person'], workflow.CONFIG, 'cpu').test_client()
    response = client.post('/invocations', data=data, content_type=mime)
    assert response.status_code == status
    assert response.is_json and 'error' in response.json


@pytest.mark.parametrize('model', [Model(score=float('nan')), Model(error=RuntimeError('GPU error'))])
def test_inference_failure_is_not_a_safe_empty_scene(serving, model):
    client = serving.create_app(model, ['person'], workflow.CONFIG, 'cpu').test_client()
    response = client.post('/invocations', data=jpeg(), content_type='image/jpeg')
    assert response.status_code == 500
    assert 'detections' not in response.json


def stub_runtime(monkeypatch, serving, available, load):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        __version__='2.7.1+cu118', version=SimpleNamespace(cuda='11.8'),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 90100)),
        cuda=SimpleNamespace(is_available=lambda: available, synchronize=lambda: None)))
    monkeypatch.setitem(sys.modules, 'torchvision', SimpleNamespace(__version__='0.22.1+cu118'))
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(__version__='8.4.56', YOLO=load))
    monkeypatch.setattr(serving.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='T4, 470.82'))


@pytest.mark.parametrize('available,selected,expected', [(False, 'auto', 'cpu'),
    (True, 'auto', 'cuda'), (True, 'cpu', 'cpu')])
def test_device_selection_uses_cuda_when_available(serving, tmp_path, monkeypatch, available, selected, expected):
    (tmp_path / 'best.pt').write_bytes(b'trusted-placeholder')
    (tmp_path / 'labels.txt').write_text('person\n')
    (tmp_path / 'config.json').write_text(json.dumps(workflow.CONFIG))
    monkeypatch.setenv('MODEL_PATH', str(tmp_path / 'best.pt'))
    monkeypatch.setenv('MODEL_DEVICE', selected)
    model = Model()
    model.names = {0: 'person'}
    model.to = lambda device: None
    loads = []
    def load(path):
        loads.append(path)
        return model
    stub_runtime(monkeypatch, serving, available, load)
    loaded = serving.load_model()
    assert loaded[3] == expected
    assert len(loads) == 1
    assert model.calls == 1


def test_gpu_request_does_not_silently_use_cpu(serving, tmp_path, monkeypatch):
    (tmp_path / 'labels.txt').write_text('person\n')
    (tmp_path / 'config.json').write_text(json.dumps(workflow.CONFIG))
    monkeypatch.setenv('MODEL_PATH', str(tmp_path / 'best.pt'))
    monkeypatch.setenv('MODEL_DEVICE', 'cuda')
    stub_runtime(monkeypatch, serving, False, lambda path: pytest.fail('no load without CUDA'))
    with pytest.raises(RuntimeError, match='CUDA requested'):
        serving.load_model()


def test_model_references_validated_s3_and_ecr_digest(flow, monkeypatch):
    flow.archive.parent.mkdir(parents=True)
    flow.archive.write_bytes(b'full-checkpoint')
    uri = flow.c['ecr_uri'] + '@sha256:' + 'a' * 64
    runtime.save_json(flow.state / 'ecr-pushed.json', {'uri': uri, 'image_id': 'tested-image'})
    monkeypatch.setattr(flow, 'resolve_role', lambda: 'arn:aws:iam::123456789012:role/LabRole')
    monkeypatch.setattr(flow, 'require_docker_test', lambda: {'image_id': 'tested-image'})
    flow.aws = FakeAws({('s3api', 'head-object'): {'Metadata': {'sha256': runtime.digest(flow.archive)}},
                        ('sagemaker', 'describe-model'): None})
    flow.create_model()
    payload = next(p for _, op, p in flow.aws.calls if op == 'create-model')
    assert payload['PrimaryContainer']['Image'] == uri
    assert payload['PrimaryContainer']['Environment']['MODEL_DEVICE'] == 'cuda'
    assert payload['EnableNetworkIsolation'] is True
    flow.aws.responses[('sagemaker', 'describe-model')] = payload
    before = len([1 for _, op, _ in flow.aws.calls if op == 'create-model'])
    flow.create_model()
    assert len([1 for _, op, _ in flow.aws.calls if op == 'create-model']) == before


@pytest.mark.parametrize('failure', [False, True], ids=['valid-output', 'aws-failure'])
def test_async_round_trip_checks_output_and_failure(flow, monkeypatch, failure):
    image = flow.root / 'test.jpg'
    image.write_bytes(jpeg())
    flow.labels_path.parent.mkdir(parents=True)
    flow.labels_path.write_text('person\n')
    monkeypatch.setattr(flow, 'wait_endpoint', lambda: {})
    monkeypatch.setattr(flow, 'prepare_test_image', lambda: (image, (640, 480)))
    checked = []
    monkeypatch.setattr(flow, 'cloudwatch', lambda inference_id: checked.append(inference_id))
    responses = {
        ('sagemaker-runtime', 'invoke-endpoint-async'): lambda p: {
            'InferenceId': p['InferenceId'], 'OutputLocation': f's3://{flow.bucket}/async-output/result.json',
            'FailureLocation': f's3://{flow.bucket}/async-failure/result.json'},
        ('s3api', 'head-object'): lambda p: None if failure and 'async-output' in p['Key'] else {'ContentLength': 400},
    }
    fake = FakeAws(responses)
    original = fake.call
    def call(service, operation, payload=None, **kwargs):
        result = original(service, operation, payload, **kwargs)
        if operation == 'get-object':
            runtime.save_json(kwargs['extra'][0], {'error': 'GPU failed'} if failure else output())
        return result
    fake.call = call
    flow.aws = fake
    if failure:
        with pytest.raises(Stop, match='AWS asynchronous failure'):
            flow.test_async_inference()
        assert not checked
    else:
        flow.test_async_inference()
        assert len(checked) == 1
    payload = next(p for _, op, p in fake.calls if op == 'invoke-endpoint-async')
    assert payload['InputLocation'].startswith(f's3://{flow.bucket}/async-input/')
    assert payload['CustomAttributes'] == 'request_id=' + payload['InferenceId']
    assert payload['ContentType'] == 'image/jpeg'


def test_startup_rejects_kernel_failure(serving, tmp_path, monkeypatch):
    (tmp_path / 'best.pt').write_bytes(b'trusted-placeholder')
    (tmp_path / 'labels.txt').write_text('person\n')
    (tmp_path / 'config.json').write_text(json.dumps(workflow.CONFIG))
    monkeypatch.setenv('MODEL_PATH', str(tmp_path / 'best.pt'))
    monkeypatch.setenv('MODEL_DEVICE', 'cuda')
    model = Model(error=RuntimeError('CUDA kernel unavailable'))
    model.names = {0: 'person'}
    model.to = lambda device: None
    stub_runtime(monkeypatch, serving, True, lambda path: model)
    with pytest.raises(RuntimeError, match='CUDA kernel unavailable'):
        serving.load_model()
