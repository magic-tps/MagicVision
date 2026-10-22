"""API packaging and deployment through an existing laboratory role, never IAM creation."""
import ipaddress
import json
import re
import time
from zipfile import ZipFile
from urllib.request import urlopen

from runtime import MISSING_S3, Stop, digest, load_json, run, save_json


def source_hashes(flow):
    source = flow.root / 'backend/lambda'
    return {p.name: digest(p) for p in sorted(source.iterdir())
            if p.is_file() and (p.suffix == '.py' or p.name in {'requirements.txt', 'Dockerfile.package', '.dockerignore'})}


def package_api(flow):
    flow.check_docker()
    target = flow.artifacts / 'api'
    target.mkdir(parents=True, exist_ok=True)
    image = 'magicvision-api-package:latest'
    run(['docker', 'build', '--platform', 'linux/amd64', '--provenance=false',
         '-f', str(flow.root / 'backend/lambda/Dockerfile.package'), '-t', image,
         str(flow.root / 'backend/lambda')], timeout=1200, live=True)
    run(['docker', 'run', '--rm', '--network', 'none', '--mount',
         f'type=bind,source={target},target=/output', image], timeout=120, live=True)
    archive = target / 'api.zip'
    with ZipFile(archive) as package:
        names = package.namelist()
        if not {'api.py', 'async_jobs.py', 'schema.py'} <= set(names):
            raise Stop('Lambda entrypoint missing in ZIP')
        if any(p.endswith(('.pt', '.tflite', '.dll', '.pyd')) or p.startswith('testing/') for p in names):
            raise Stop('Unexpected model/Windows/test files in Linux Lambda ZIP')
        if sum(i.file_size for i in package.infolist()) >= 250 * 1024 * 1024:
            raise Stop('Lambda uncompressed package exceeds limit')
    save_json(target / 'package-tested.json', {'sha256': digest(archive), 'sources': source_hashes(flow),
        'image_id': run(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'])})
    flow.status(f'Linux Lambda dependencies/import validated; ZIP sha256={digest(archive)}')


def api_settings(flow):
    stack = flow.c.get('api_stack', 'magicvision-async-api')
    role = flow.c.get('api_role_arn', '')
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9-]{0,99}', stack):
        raise Stop('Invalid API stack name')
    if not re.fullmatch(f"arn:aws:iam::{flow.c['account']}:role/[A-Za-z0-9+=,.@_/-]+", role):
        raise Stop('Set MV_API_EXECUTION_ROLE_ARN to the existing Lambda execution role supplied by the administrator')
    try:
        network = ipaddress.ip_network(flow.c.get('allowed_client_cidr', ''), strict=True)
    except ValueError:
        raise Stop('Set MV_ALLOWED_CLIENT_CIDR to the authorized laboratory public egress IPv4 CIDR') from None
    if network.version != 4 or network.prefixlen == 0:
        raise Stop('API CIDR must be IPv4 and must not expose the service to /0')
    return stack, role, str(network)


def write_api_policy(flow, stack):
    account, region, bucket = flow.c['account'], flow.c['region'], flow.bucket
    policy = {'Version': '2012-10-17', 'Statement': [
        {'Effect': 'Allow', 'Action': 'sagemaker:InvokeEndpointAsync',
         'Resource': f"arn:aws:sagemaker:{region}:{account}:endpoint/{flow.c['endpoint']}"},
        {'Effect': 'Allow', 'Action': 's3:ListBucket', 'Resource': f'arn:aws:s3:::{bucket}'},
        {'Effect': 'Allow', 'Action': ['s3:GetObject', 's3:DeleteObject'],
         'Resource': [f'arn:aws:s3:::{bucket}/{p}*' for p in (
             'app-jobs/', 'async-input/app/', 'async-output/', 'async-failure/')]},
        {'Effect': 'Allow', 'Action': 's3:PutObject',
         'Resource': [f'arn:aws:s3:::{bucket}/{p}*' for p in ('app-jobs/', 'async-input/app/')]},
        {'Effect': 'Allow', 'Action': ['logs:CreateLogGroup', 'logs:CreateLogStream', 'logs:PutLogEvents'],
         'Resource': f'arn:aws:logs:{region}:{account}:log-group:/aws/lambda/{stack}-Function-*:*'},
    ]}
    save_json(flow.state / 'proposed-api-role-policy.json', policy)
    save_json(flow.state / 'proposed-api-role-trust.json', {'Version': '2012-10-17', 'Statement': [
        {'Effect': 'Allow', 'Principal': {'Service': 'lambda.amazonaws.com'}, 'Action': 'sts:AssumeRole'}]})


def configure_app_retention(flow):
    """One day fallback for abandoned app jobs; preserve other rules and experiment data."""
    payload = {'Bucket': flow.bucket, 'ExpectedBucketOwner': flow.c['account']}
    current = flow.aws.call('s3api', 'get-bucket-lifecycle-configuration', payload,
                           missing=(r'\(NoSuchLifecycleConfiguration\)',)) or {}
    rules = list(current.get('Rules', []))
    for prefix in ('app-jobs/', 'async-input/app/', 'async-output/app/', 'async-failure/app/'):
        expected = {'ID': 'magicvision-temporary-' + prefix.strip('/').replace('/', '-'),
            'Status': 'Enabled', 'Filter': {'Prefix': prefix}, 'Expiration': {'Days': 1},
            'NoncurrentVersionExpiration': {'NoncurrentDays': 1},
            'AbortIncompleteMultipartUpload': {'DaysAfterInitiation': 1}}
        found = next((r for r in rules if r.get('ID') == expected['ID']), None)
        if found is not None and found != expected:
            raise Stop('Existing MagicVision app lifecycle rule differs; review manually')
        if found is None:
            rules.append(expected)
    if rules != current.get('Rules', []):
        payload['LifecycleConfiguration'] = {'Rules': rules}
        flow.aws.call('s3api', 'put-bucket-lifecycle-configuration', payload)
    flow.status('App temporary prefixes expire after one day; model and CLI experiment prefixes are preserved')


def check_api_environment(flow):
    stack, role, cidr = api_settings(flow)
    write_api_policy(flow, stack)
    role_info = flow.aws.call('iam', 'get-role', {'RoleName': role.rsplit('/', 1)[-1]})['Role']
    if role_info['Arn'] != role:
        raise Stop('Lambda execution role ARN differs')
    statements = role_info['AssumeRolePolicyDocument'].get('Statement', [])
    if not any(s.get('Effect') == 'Allow' and
               'lambda.amazonaws.com' in ([s.get('Principal', {}).get('Service')]
                   if isinstance(s.get('Principal', {}).get('Service'), str)
                   else s.get('Principal', {}).get('Service', [])) for s in statements):
        raise Stop('Existing API role does not trust Lambda; give proposed-api-role-trust.json to administrator')
    return stack, role, cidr


def deploy_api(flow):
    stack, role, cidr = check_api_environment(flow)
    endpoint = flow.aws.call('sagemaker', 'describe-endpoint', {'EndpointName': flow.c['endpoint']})
    if endpoint['EndpointStatus'] != 'InService':
        raise Stop('SageMaker endpoint must be InService before deploying the API')
    endpoint_config = flow.aws.call('sagemaker', 'describe-endpoint-config', {
        'EndpointConfigName': endpoint['EndpointConfigName']})
    if 'AsyncInferenceConfig' not in endpoint_config:
        raise Stop('The selected endpoint does not support asynchronous inference')
    outputs_config = endpoint_config['AsyncInferenceConfig'].get('OutputConfig', {})
    if outputs_config.get('S3OutputPath') != f's3://{flow.bucket}/async-output/' or (
            outputs_config.get('S3FailurePath') != f's3://{flow.bucket}/async-failure/'):
        raise Stop('Async endpoint output/failure paths do not match the API bucket and prefixes')
    artifact = flow.artifacts / 'api/api.zip'
    proof = load_json(flow.artifacts / 'api/package-tested.json')
    if proof.get('sha256') != digest(artifact) or proof.get('sources') != source_hashes(flow):
        raise Stop('API sources/package changed; run package-api.ps1 before deployment')
    key = f"api/{proof['sha256']}/api.zip"
    remote = flow.aws.call('s3api', 'head-object', {'Bucket': flow.bucket, 'Key': key}, missing=MISSING_S3)
    if remote is None:
        flow.aws.call('s3api', 'put-object', {'Bucket': flow.bucket, 'Key': key,
            'Metadata': {'sha256': proof['sha256']}, 'ServerSideEncryption': 'AES256'}, extra=('--body', str(artifact)))
    elif remote.get('Metadata', {}).get('sha256') != proof['sha256']:
        raise Stop('Existing API artifact hash differs; no overwrite performed')
    template = (flow.root / 'backend/api-template.yaml').read_text(encoding='utf-8')
    values = {'CodeBucket': flow.bucket, 'CodeKey': key, 'EndpointName': flow.c['endpoint'],
              'ExecutionRoleArn': role, 'AllowedClientCidr': cidr}
    parameters = [{'ParameterKey': k, 'ParameterValue': v} for k, v in values.items()]
    existing = flow.aws.call('cloudformation', 'describe-stacks', {'StackName': stack},
        missing=(r'Stack with id .+ does not exist',))
    if existing is not None:
        found = existing['Stacks'][0]
        actual = {p['ParameterKey']: p['ParameterValue'] for p in found.get('Parameters', [])}
        old_template = flow.aws.call('cloudformation', 'get-template', {'StackName': stack})['TemplateBody']
        if actual != values or old_template != template:
            raise Stop('Existing API stack differs; choose MV_API_STACK_NAME for a new version or review an update manually')
    configure_app_retention(flow)
    if existing is None:
        flow.aws.call('cloudformation', 'create-stack', {'StackName': stack, 'TemplateBody': template,
            'Parameters': parameters, 'OnFailure': 'DO_NOTHING', 'Tags': [{'Key': 'Project', 'Value': 'MagicVision'}]})
    deadline = time.monotonic() + 900
    while True:
        found = flow.aws.call('cloudformation', 'describe-stacks', {'StackName': stack})['Stacks'][0]
        status = found['StackStatus']
        if status in {'CREATE_COMPLETE', 'UPDATE_COMPLETE'}:
            break
        if not status.endswith('_IN_PROGRESS') or time.monotonic() > deadline:
            raise Stop(f'API stack {status}: inspect CloudFormation events for {stack}; no automatic deletion')
        print(f'[WAIT] API stack {status}', flush=True)
        time.sleep(15)
    outputs = {v['OutputKey']: v['OutputValue'] for v in found.get('Outputs', [])}
    if not outputs.get('ApiBaseUrl', '').startswith('https://'):
        raise Stop('API stack did not return its HTTPS URL')
    save_json(flow.state / 'api-outputs.json', outputs)
    health = check_api_health(outputs['ApiBaseUrl'])
    save_json(flow.state / 'api-health.json', health)
    save_json(flow.artifacts / 'flutter-cloud.json', {'API_BASE_URL': outputs['ApiBaseUrl'],
        'CAMERA_SOURCE': 'usb', 'INFERENCE_PROVIDER': 'aws', 'CONTINUOUS_ANALYSIS': False,
        'API_TIMEOUT_SECONDS': 10, 'API_WARMUP_TIMEOUT_SECONDS': 900, 'MAX_CAPTURE_AGE_SECONDS': 15})
    flow.status('Mobile API ready: ' + outputs['ApiBaseUrl'])


def check_api_health(base_url):
    with urlopen(base_url + '/health', timeout=10) as response:
        health = json.loads(response.read(4096))
    if health != {'configured': True, 'mode': 'async'}:
        raise Stop('Deployed API health is not configured for async inference')
    return health
