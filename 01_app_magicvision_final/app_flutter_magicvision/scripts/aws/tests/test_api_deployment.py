import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from test_workflow import flow, FakeAws  # Reuse offline AWS shape validation and fixture.
from api_deployment import api_settings, configure_app_retention, deploy_api, source_hashes, write_api_policy
from runtime import Stop, digest, save_json


def settings(flow):
    flow.c.update(api_stack='magicvision-async-api',
        api_role_arn='arn:aws:iam::123456789012:role/lab/LambdaRole', allowed_client_cidr='203.0.113.10/32')


@pytest.mark.parametrize('cidr', ['', '0.0.0.0/0', '::/0', '203.0.113.5/24', '999.0.0.0/24'])
def test_api_rejects_unbounded_or_invalid_cidr(flow, cidr):
    settings(flow)
    flow.c['allowed_client_cidr'] = cidr
    with pytest.raises(Stop):
        api_settings(flow)


def test_api_requires_role_in_authenticated_account(flow):
    settings(flow)
    flow.c['api_role_arn'] = 'arn:aws:iam::999999999999:role/LambdaRole'
    with pytest.raises(Stop, match='existing Lambda'):
        api_settings(flow)


def test_app_lifecycle_preserves_other_rules_and_is_idempotent(flow):
    original = {'ID': 'existing-experiments', 'Status': 'Enabled', 'Filter': {'Prefix': 'reports/'},
                'Expiration': {'Days': 365}}
    flow.aws = FakeAws({('s3api', 'get-bucket-lifecycle-configuration'): {'Rules': [original]}})
    configure_app_retention(flow)
    written = flow.aws.calls[-1][2]['LifecycleConfiguration']
    assert written['Rules'][0] == original
    prefixes = {r['Filter']['Prefix'] for r in written['Rules'][1:]}
    assert prefixes == {'app-jobs/', 'async-input/app/', 'async-output/app/', 'async-failure/app/'}
    assert all(r['Expiration']['Days'] == 1 and r['NoncurrentVersionExpiration']['NoncurrentDays'] == 1
               for r in written['Rules'][1:])
    flow.aws = FakeAws({('s3api', 'get-bucket-lifecycle-configuration'): written})
    configure_app_retention(flow)
    assert not any(op == 'put-bucket-lifecycle-configuration' for _, op, _ in flow.aws.calls)


def test_lifecycle_conflict_does_not_overwrite_user_rule(flow):
    flow.aws = FakeAws({('s3api', 'get-bucket-lifecycle-configuration'): {'Rules': [
        {'ID': 'magicvision-temporary-app-jobs', 'Expiration': {'Days': 90}}]}})
    with pytest.raises(Stop, match='differs'):
        configure_app_retention(flow)
    assert len(flow.aws.calls) == 1


def test_proposed_role_has_async_permission_but_no_model_write_or_iam_admin(flow):
    settings(flow)
    write_api_policy(flow, flow.c['api_stack'])
    policy = json.loads((flow.state / 'proposed-api-role-policy.json').read_text())
    assert policy['Statement'][0]['Action'] == 'sagemaker:InvokeEndpointAsync'
    assert 'iam:' not in json.dumps(policy)
    assert 'models/' not in json.dumps(policy)
    assert not flow.aws.__dict__.get('calls')


def setup_deploy(flow):
    settings(flow)
    (flow.root / 'backend/lambda').mkdir(parents=True)
    template = Path(__file__).resolve().parents[3] / 'backend/api-template.yaml'
    (flow.root / 'backend/api-template.yaml').write_text(template.read_text())
    artifact = flow.artifacts / 'api/api.zip'
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'local-verified-artifact-fixture')
    sha = digest(artifact)
    save_json(artifact.parent / 'package-tested.json', {'sha256': sha, 'sources': source_hashes(flow)})
    values = {'CodeBucket': flow.bucket, 'CodeKey': f'api/{sha}/api.zip', 'EndpointName': flow.c['endpoint'],
        'ExecutionRoleArn': flow.c['api_role_arn'], 'AllowedClientCidr': flow.c['allowed_client_cidr']}
    stack = {'StackStatus': 'CREATE_COMPLETE', 'Parameters': [
        {'ParameterKey': k, 'ParameterValue': v} for k, v in values.items()],
        'Outputs': [{'OutputKey': 'ApiBaseUrl', 'OutputValue': 'https://api-id.execute-api.us-west-2.amazonaws.com/prod'}]}
    responses = {
        ('iam', 'get-role'): {'Role': {'Arn': flow.c['api_role_arn'], 'AssumeRolePolicyDocument': {
            'Statement': [{'Effect': 'Allow', 'Principal': {'Service': 'lambda.amazonaws.com'}, 'Action': 'sts:AssumeRole'}]}}},
        ('sagemaker', 'describe-endpoint'): {'EndpointStatus': 'InService', 'EndpointConfigName': 'async'},
        ('sagemaker', 'describe-endpoint-config'): {'AsyncInferenceConfig': {'OutputConfig': {
            'S3OutputPath': f's3://{flow.bucket}/async-output/', 'S3FailurePath': f's3://{flow.bucket}/async-failure/'}}},
        ('s3api', 'head-object'): {'Metadata': {'sha256': sha}},
        ('cloudformation', 'describe-stacks'): {'Stacks': [stack]},
        ('cloudformation', 'get-template'): {'TemplateBody': template.read_text()},
    }
    flow.aws = FakeAws(responses)
    return responses


def test_deploy_existing_api_validates_configuration_and_outputs_flutter_defines(flow, monkeypatch):
    monkeypatch.setattr('api_deployment.check_api_health', lambda url: {'configured': True, 'mode': 'async'})
    setup_deploy(flow)
    deploy_api(flow)
    defines = json.loads((flow.artifacts / 'flutter-cloud.json').read_text())
    assert defines['API_BASE_URL'].startswith('https://')
    assert defines['MAX_CAPTURE_AGE_SECONDS'] == 15
    assert not any(op in {'create-stack', 'update-stack', 'create-role', 'put-role-policy'} for _, op, _ in flow.aws.calls)


def test_deploy_new_stack_uses_existing_role_without_iam_changes_or_automatic_rollback(flow, monkeypatch):
    monkeypatch.setattr('api_deployment.check_api_health', lambda url: {'configured': True, 'mode': 'async'})
    responses = setup_deploy(flow)
    complete = responses[('cloudformation', 'describe-stacks')]
    sequence = iter([None, complete])
    responses[('cloudformation', 'describe-stacks')] = lambda payload: next(sequence)
    deploy_api(flow)
    creates = [p for _, op, p in flow.aws.calls if op == 'create-stack']
    assert len(creates) == 1
    assert creates[0]['OnFailure'] == 'DO_NOTHING'
    assert 'Capabilities' not in creates[0]
    assert not any(op in {'create-role', 'put-role-policy', 'delete-stack'} for _, op, _ in flow.aws.calls)


def test_deploy_does_not_mark_flutter_configuration_ready_when_api_is_unreachable(flow, monkeypatch):
    setup_deploy(flow)
    def unavailable(url):
        raise Stop('API unreachable')
    monkeypatch.setattr('api_deployment.check_api_health', unavailable)
    with pytest.raises(Stop, match='unreachable'):
        deploy_api(flow)
    assert not (flow.artifacts / 'flutter-cloud.json').exists()


def test_deploy_rejects_synchronous_endpoint(flow):
    replies = setup_deploy(flow)
    replies[('sagemaker', 'describe-endpoint-config')] = {}
    with pytest.raises(Stop, match='asynchronous'):
        deploy_api(flow)


def test_deploy_does_not_replace_differing_stack(flow):
    replies = setup_deploy(flow)
    replies[('cloudformation', 'get-template')] = {'TemplateBody': 'another application'}
    with pytest.raises(Stop, match='stack differs'):
        deploy_api(flow)
    assert not any(op in {'create-stack', 'update-stack', 'delete-stack'} for _, op, _ in flow.aws.calls)


def test_deploy_stops_if_api_changed_after_linux_validation(flow):
    setup_deploy(flow)
    (flow.root / 'backend/lambda/api.py').write_text('changed')
    with pytest.raises(Stop, match='sources/package changed'):
        deploy_api(flow)
