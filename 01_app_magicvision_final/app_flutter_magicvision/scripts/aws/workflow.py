"""Resumable deployment steps for PowerShell. No training, no automatic deletion."""
import argparse
import json
import os
import re
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from runtime import (Aws, CommandFailure, Stop, MISSING_ECR, MISSING_S3, MISSING_SM,
                     digest, discover_region, load_json, run, save_json, valid_response)

CONFIG = {'imgsz': 640, 'confidence': .30, 'iou': .65, 'max_det': 100}
TAGS = [{'Key': 'Project', 'Value': 'MagicVision'}]


def contains(actual, expected):
    """AWS describe adds defaults/metadata; compare every requested field recursively."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and contains(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(contains(a, e) for a, e in zip(actual, expected))
    return actual == expected


class Workflow:
    def __init__(self, config):
        self.c = config
        self.root = Path(config['root']).resolve()
        self.artifacts = self.root / 'backend/artifacts'
        self.state = self.artifacts / 'aws'
        self.weights = self.root / 'runs/detect/modelo_final_yolo26m_80e_640_b8/weights/best.pt'
        self.labels_path = self.root / 'ml/configs/classes.txt'
        self.archive = self.artifacts / 'model.tar.gz'
        self.source = self.root / 'backend/sagemaker'
        self.local_image = 'magicvision-yolo26m:latest'
        self.aws = Aws(config.get('region', ''))

    @property
    def labels(self):
        return self.labels_path.read_text(encoding='utf-8').splitlines()

    @property
    def gpu(self):
        return self.c['instance'].startswith(('ml.g', 'ml.p'))

    @property
    def bucket(self):
        return self.c['bucket']

    @property
    def resource_id(self):
        return f"endpoint/{self.c['endpoint']}/variant/AllTraffic"

    def status(self, text):
        print(f'[OK] {text}', flush=True)

    def check_aws(self):
        print(run(['aws', '--version']), flush=True)
        self.aws.region = discover_region(self.c.get('region', ''))
        identity = self.aws.call('sts', 'get-caller-identity')
        account = identity.get('Account', '')
        if not re.fullmatch(r'\d{12}', account):
            raise Stop('STS did not return a valid account ID')
        self.c['account'] = account
        self.c['region'] = self.aws.region
        partition = identity['Arn'].split(':')[1]
        if partition != 'aws':
            raise Stop('This ECR/DNS recipe supports the standard aws partition only; adapt explicitly for this laboratory.')
        self.c['bucket'] = self.c.get('bucket') or f"magicvision-{account}-{self.aws.region}"
        self.c['registry'] = f'{account}.dkr.ecr.{self.aws.region}.amazonaws.com'
        self.c['ecr_uri'] = f"{self.c['registry']}/{self.c['repository']}"
        for key in ('model', 'endpoint_config', 'endpoint'):
            if not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?', self.c[key]):
                raise Stop(f'Invalid SageMaker name: {key}')
        if not re.fullmatch(r'[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]', self.bucket):
            raise Stop('Invalid S3 bucket name')
        save_json(self.state / 'resolved-config.json', self.c)
        self.status(f"AWS authenticated; account={account}, region={self.aws.region}")

    def check_docker(self):
        print(run(['docker', '--version']), flush=True)
        platform = run(['docker', 'info', '--format', '{{.OSType}}/{{.Architecture}}'])
        if not platform.startswith('linux/'):
            raise Stop('Docker Linux engine is required')
        self.status('Docker engine ' + platform)

    def check_environment(self):
        # Authentication/region gates precede any deployment.
        self.check_aws()
        self.check_docker()
        self.check_quota()
        self.resolve_role()

    def package_model(self):
        if not self.archive.exists():
            run([sys.executable, str(self.root / 'backend/package_model.py'),
                 '--weights', str(self.weights), '--labels', str(self.labels_path),
                 '--output', str(self.archive)], live=True)
        self.validate_archive()
        self.status(f'model.tar.gz verified; sha256={digest(self.archive)}')

    def validate_archive(self):
        import hashlib
        with tarfile.open(self.archive, 'r:gz') as archive:
            members = archive.getmembers()
            if len(members) != 3 or {m.name for m in members} != {'best.pt', 'labels.txt', 'config.json'}:
                raise Stop('Archive must contain exactly best.pt, labels.txt and config.json at root')
            if not all(m.isfile() for m in members):
                raise Stop('Archive must contain regular files only')
            with archive.extractfile('best.pt') as weights:
                if hashlib.file_digest(weights, 'sha256').hexdigest() != digest(self.weights):
                    raise Stop('Existing archive differs from best.pt; choose a versioned artifact manually')
            if archive.extractfile('labels.txt').read().decode('utf-8').splitlines() != self.labels:
                raise Stop('Archive labels differ from training labels')
            if json.load(archive.extractfile('config.json')) != CONFIG:
                raise Stop('Archive inference parameters differ')

    def build_docker(self):
        self.check_docker()
        flavor = self.c['torch_flavor']
        if flavor != 'cu118':
            raise Stop('This SageMaker image requires TORCH_FLAVOR=cu118; use MODEL_DEVICE=cpu for local CPU tests')
        run(['docker', 'build', '--platform', 'linux/amd64', '--provenance=false', '--build-arg',
             'TORCH_FLAVOR=' + flavor, '-t', self.local_image, str(self.source)],
            timeout=7200, live=True)
        self.status('Docker image built')

    def image_identity(self):
        info = json.loads(run(['docker', 'image', 'inspect', self.local_image]))[0]
        if info['Os'] != 'linux' or info['Architecture'] != 'amd64':
            raise Stop('Image must be Linux amd64')
        return info['Id']

    def fingerprint(self):
        return {'image_id': self.image_identity(), 'model_sha256': digest(self.archive),
                'sources': {p: digest(self.source / p) for p in
                            ('Dockerfile', 'requirements.txt', 'constraints-cu118.txt', 'serve.py', 'verify_container.py', '.dockerignore')}}

    def prepare_test_image(self):
        from PIL import Image, ImageOps
        requested = self.c.get('test_image')
        output = self.state / 'test.jpg'
        output.parent.mkdir(parents=True, exist_ok=True)
        if requested:
            path = Path(requested).resolve()
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
                image.thumbnail((640, 640))
        else:
            # Reproducible contract test; never represents model accuracy.
            image = Image.new('RGB', (640, 480), (96, 96, 96))
            print('[INFO] Synthetic test.jpg; set MV_TEST_IMAGE for a real authorized scene.')
        image.save(output, 'JPEG', quality=80)
        return output, image.size

    def test_docker(self):
        self.check_docker()
        self.package_model()
        proof = self.fingerprint()
        local_device = self.c.get('local_device', 'cpu')
        if local_device not in {'cpu', 'cuda'}:
            raise Stop('MV_LOCAL_DEVICE must be cpu or cuda')
        test_image, size = self.prepare_test_image()
        with tempfile.TemporaryDirectory(prefix='magicvision-model-') as temporary:
            with tarfile.open(self.archive) as archive:
                for member in archive.getmembers():
                    Path(temporary, member.name).write_bytes(archive.extractfile(member).read())
            name = 'magicvision-local-' + uuid.uuid4().hex[:12]
            started = False
            try:
                gpu_args = ['--gpus', 'all'] if local_device == 'cuda' else []
                run(['docker', 'run', '--detach', '--name', name, '--platform', 'linux/amd64', *gpu_args,
                     '-p', '127.0.0.1::8080', '--mount', f'type=bind,source={temporary},target=/opt/ml/model,readonly',
                     '-e', 'MODEL_DEVICE=' + local_device, self.local_image, 'serve'])
                started = True
                port = run(['docker', 'port', name, '8080/tcp']).splitlines()[0].rsplit(':', 1)[1]
                base = 'http://127.0.0.1:' + port
                deadline = time.monotonic() + 300
                while True:
                    try:
                        with urlopen(base + '/ping', timeout=5) as response:
                            health = json.load(response)
                            if response.status == 200 and health.get('status') == 'ok':
                                break
                    except (URLError, TimeoutError, ConnectionError):
                        pass
                    if run(['docker', 'inspect', '--format', '{{.State.Running}}', name]) != 'true':
                        raise Stop('Local container exited before readiness')
                    if time.monotonic() >= deadline:
                        raise Stop('/ping timed out')
                    time.sleep(3)
                self.status('/ping local; device=' + health['device'])
                if health['device'] != local_device:
                    raise Stop('Local device does not match requested device')
                source_hashes = json.loads(run(['docker', 'exec', name, 'python', '-c',
                    'import hashlib,json; from pathlib import Path; '
                    'print(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() '
                    'for p in ["serve.py","requirements.txt"]}))']))
                if any(source_hashes[p] != proof['sources'][p] for p in source_hashes):
                    raise Stop('Container source differs from workspace; rebuild before testing')
                request = Request(base + '/invocations', data=test_image.read_bytes(),
                                  headers={'Content-Type': 'image/jpeg', 'Accept': 'application/json'})
                with urlopen(request, timeout=180) as response:
                    if response.headers.get_content_type() != 'application/json':
                        raise Stop('Local response is not application/json')
                    result = valid_response(json.load(response), self.labels, size)
                save_json(self.state / 'local-result.json', result)
                # The *same* tested CUDA-enabled image can use CPU locally.
                cuda_version = run(['docker', 'exec', name, 'python', '-c',
                                    'import torch; print(torch.version.cuda or "none")'])
                if self.gpu and cuda_version == 'none':
                    raise Stop('Tested image has no CUDA runtime')
                proof.update({'test_image_sha256': digest(test_image), 'size': list(size),
                              'local_device': health['device'], 'torch_cuda': cuda_version})
                save_json(self.state / 'docker-tested.json', proof)
                self.status('/invocations local; valid JSON saved')
                print(json.dumps(result, indent=2))
            finally:
                if started:
                    try:
                        logs = run(['docker', 'logs', name], merge_output=True)
                        (self.state / ('local-' + local_device + '.log')).write_text(logs + '\n', encoding='utf-8')
                        print(logs)
                    finally:
                        # Logging/console errors must not leave the test container running.
                        run(['docker', 'rm', '-f', name])

    def require_docker_test(self):
        self.validate_archive()
        path = self.state / 'docker-tested.json'
        if not path.exists():
            raise Stop('Run test-docker.ps1 successfully before any AWS resource creation/upload')
        proof = load_json(path)
        if any(proof.get(key) != value for key, value in self.fingerprint().items()):
            raise Stop('Image/model/source changed since Docker test; test again')
        if self.gpu and proof.get('torch_cuda', 'none') == 'none':
            raise Stop('GPU deployment needs a tested CUDA-enabled image')
        return proof

    def create_s3(self):
        existing = self.aws.call('s3api', 'head-bucket', {
            'Bucket': self.bucket, 'ExpectedBucketOwner': self.c['account']}, missing=MISSING_S3)
        if existing is None:
            payload = {'Bucket': self.bucket, 'ObjectOwnership': 'BucketOwnerEnforced'}
            if self.c['region'] != 'us-east-1':
                payload['CreateBucketConfiguration'] = {'LocationConstraint': self.c['region']}
            self.aws.call('s3api', 'create-bucket', payload)
        location = self.aws.call('s3api', 'get-bucket-location', {
            'Bucket': self.bucket, 'ExpectedBucketOwner': self.c['account']}).get('LocationConstraint')
        if (location or 'us-east-1') != self.c['region']:
            raise Stop('Existing bucket region does not match configured region')
        self.aws.call('s3api', 'put-public-access-block', {'Bucket': self.bucket,
            'ExpectedBucketOwner': self.c['account'], 'PublicAccessBlockConfiguration': {
                k: True for k in ('BlockPublicAcls', 'IgnorePublicAcls', 'BlockPublicPolicy', 'RestrictPublicBuckets')}})
        if existing is None:
            self.aws.call('s3api', 'put-bucket-encryption', {'Bucket': self.bucket,
                'ServerSideEncryptionConfiguration': {'Rules': [
                    {'ApplyServerSideEncryptionByDefault': {'SSEAlgorithm': 'AES256'}}]}})
        encryption = self.aws.call('s3api', 'get-bucket-encryption', {'Bucket': self.bucket})
        if any(r['ApplyServerSideEncryptionByDefault']['SSEAlgorithm'] != 'AES256'
               for r in encryption['ServerSideEncryptionConfiguration']['Rules']):
            raise Stop('Existing bucket uses KMS; request explicit key-scoped IAM support before deployment')
        for key in ('async-input/', 'async-output/', 'async-failure/'):
            if self.aws.call('s3api', 'head-object', {'Bucket': self.bucket, 'Key': key},
                             missing=MISSING_S3) is None:
                self.aws.call('s3api', 'put-object', {'Bucket': self.bucket, 'Key': key})
        self.status('Private S3 bucket/prefixes ready: ' + self.bucket)

    def upload_model(self):
        key = 'models/yolo26m/model.tar.gz'
        remote = self.aws.call('s3api', 'head-object', {'Bucket': self.bucket, 'Key': key}, missing=MISSING_S3)
        sha = digest(self.archive)
        if remote is not None:
            if remote.get('Metadata', {}).get('sha256') != sha:
                raise Stop('S3 model key already contains another/unverified artifact. No overwrite; choose a new deployment bucket/version.')
        else:
            self.aws.call('s3api', 'put-object', {'Bucket': self.bucket, 'Key': key,
                'Metadata': {'sha256': sha}, 'ServerSideEncryption': 'AES256',
                'ContentType': 'application/gzip'}, extra=('--body', str(self.archive)))
        remote = self.aws.call('s3api', 'head-object', {'Bucket': self.bucket, 'Key': key})
        if remote['ContentLength'] != self.archive.stat().st_size or remote['Metadata'].get('sha256') != sha:
            raise Stop('Uploaded artifact metadata mismatch')
        self.status(f'Model uploaded/verified: s3://{self.bucket}/{key}')

    def create_ecr(self):
        found = self.aws.call('ecr', 'describe-repositories',
            {'repositoryNames': [self.c['repository']]}, missing=MISSING_ECR)
        if found is None:
            self.aws.call('ecr', 'create-repository', {'repositoryName': self.c['repository'],
                'imageTagMutability': 'IMMUTABLE', 'imageScanningConfiguration': {'scanOnPush': True},
                'encryptionConfiguration': {'encryptionType': 'AES256'}, 'tags': TAGS})
        self.status('ECR repository ready: ' + self.c['ecr_uri'])

    def push_ecr(self):
        proof = self.require_docker_test()
        details = self.aws.call('ecr', 'describe-images', {'repositoryName': self.c['repository'],
            'imageIds': [{'imageTag': 'latest'}]}, missing=(r'\(ImageNotFoundException\)',))
        if details is not None:
            # Reuse only if this exact local image has already been pushed to this repository.
            remote_digest = details['imageDetails'][0]['imageDigest']
            inspected = json.loads(run(['docker', 'image', 'inspect', self.local_image]))[0]
            if self.c['ecr_uri'] + '@' + remote_digest not in (inspected.get('RepoDigests') or []):
                raise Stop('ECR latest exists and is not verified as the tested image. No tag overwrite; use a new repository/version.')
        else:
            password = run(['aws', 'ecr', 'get-login-password', '--region', self.c['region'], '--no-cli-pager'])
            # Temporary Docker config prevents persisting an ECR token in the repository or user config.
            with tempfile.TemporaryDirectory(prefix='magicvision-ecr-login-') as docker_config:
                run(['docker', '--config', docker_config, 'login', '--username', 'AWS',
                     '--password-stdin', self.c['registry']], data=password)
                password = None
                run(['docker', 'tag', self.local_image, self.c['ecr_uri'] + ':latest'])
                run(['docker', '--config', docker_config, 'push', self.c['ecr_uri'] + ':latest'],
                    timeout=7200, live=True)
            details = self.aws.call('ecr', 'describe-images', {'repositoryName': self.c['repository'],
                'imageIds': [{'imageTag': 'latest'}]})
            remote_digest = details['imageDetails'][0]['imageDigest']
        save_json(self.state / 'ecr-pushed.json', {'uri': self.c['ecr_uri'] + '@' + remote_digest,
                  'image_id': proof['image_id'], 'account': self.c['account'], 'region': self.c['region']})
        self.status('Docker image in ECR: ' + self.c['ecr_uri'] + '@' + remote_digest)

    def resolve_role(self):
        arn = self.c.get('role_arn')
        if arn:
            if not re.fullmatch(r'arn:aws:iam::' + self.c['account'] + r':role/[\w+=,.@/-]+', arn):
                raise Stop('Execution role must be an IAM role in the authenticated account')
            role = self.aws.call('iam', 'get-role', {'RoleName': arn.rsplit('/', 1)[1]})['Role']
            if role['Arn'] != arn:
                raise Stop('IAM returned a different role ARN')
        else:
            roles = self.aws.call('iam', 'list-roles').get('Roles', [])
            candidates = [r for r in roles if self.trusts_sagemaker(r)]
            if len(candidates) != 1:
                print('SageMaker trust candidates (permissions still require validation):')
                for candidate in candidates:
                    print(candidate['Arn'])
                self.write_iam_proposal()
                raise Stop('Set MV_EXECUTION_ROLE_ARN to the laboratory-approved role. No IAM roles/policies were created.')
            role = candidates[0]
        if not self.trusts_sagemaker(role):
            raise Stop('Role does not explicitly trust sagemaker.amazonaws.com')
        self.c['role_arn'] = role['Arn']
        save_json(self.state / 'resolved-config.json', self.c)
        self.status('Execution role found: ' + role['Arn'])
        return role['Arn']

    @staticmethod
    def trusts_sagemaker(role):
        for statement in role.get('AssumeRolePolicyDocument', {}).get('Statement', []):
            principal = statement.get('Principal', {})
            services = principal.get('Service', []) if isinstance(principal, dict) else []
            services = [services] if isinstance(services, str) else services
            actions = statement.get('Action', [])
            actions = [actions] if isinstance(actions, str) else actions
            if statement.get('Effect') == 'Allow' and 'sagemaker.amazonaws.com' in services and 'sts:AssumeRole' in actions:
                return True
        return False

    def write_iam_proposal(self):
        bucket_arn = 'arn:aws:s3:::' + self.bucket
        endpoint_arn = f"arn:aws:logs:{self.c['region']}:{self.c['account']}:log-group:/aws/sagemaker/Endpoints/{self.c['endpoint']}"
        policy = {'Version': '2012-10-17', 'Statement': [
            {'Effect': 'Allow', 'Action': ['s3:GetObject'],
             'Resource': [bucket_arn + '/models/yolo26m/model.tar.gz', bucket_arn + '/async-input/*']},
            {'Effect': 'Allow', 'Action': ['s3:PutObject'],
             'Resource': [bucket_arn + '/async-output/*', bucket_arn + '/async-failure/*']},
            {'Effect': 'Allow', 'Action': ['s3:ListBucket'], 'Resource': bucket_arn,
             'Condition': {'StringLike': {'s3:prefix': ['models/yolo26m/*', 'async-input/*', 'async-output/*', 'async-failure/*']}}},
            {'Effect': 'Allow', 'Action': ['s3:GetBucketLocation'], 'Resource': bucket_arn},
            {'Effect': 'Allow', 'Action': ['ecr:GetAuthorizationToken'], 'Resource': '*'},
            {'Effect': 'Allow', 'Action': ['ecr:BatchCheckLayerAvailability', 'ecr:GetDownloadUrlForLayer', 'ecr:BatchGetImage'],
             'Resource': f"arn:aws:ecr:{self.c['region']}:{self.c['account']}:repository/{self.c['repository']}"},
            {'Effect': 'Allow', 'Action': ['logs:CreateLogGroup', 'logs:CreateLogStream', 'logs:PutLogEvents', 'logs:DescribeLogStreams'],
             'Resource': [endpoint_arn, endpoint_arn + ':*']},
            {'Effect': 'Allow', 'Action': ['cloudwatch:PutMetricData'], 'Resource': '*',
             'Condition': {'StringEquals': {'cloudwatch:namespace': 'AWS/SageMaker'}}}]}
        trust = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow',
                 'Principal': {'Service': 'sagemaker.amazonaws.com'}, 'Action': 'sts:AssumeRole',
                 'Condition': {'StringEquals': {'aws:SourceAccount': self.c['account']},
                               'ArnLike': {'aws:SourceArn': f"arn:aws:sagemaker:{self.c['region']}:{self.c['account']}:*"}}}]}
        save_json(self.state / 'execution-role-policy.json', policy)
        save_json(self.state / 'execution-role-trust.json', trust)
        proposed = f"arn:aws:iam::{self.c['account']}:role/service-role/MagicVisionSageMakerExecutionRole"
        print('Administrator: provide an approved SageMaker execution Role ARN, for example ' + proposed)
        print('Proposed least-privilege policy/trust (not applied): ' + str(self.state))

    def check_quota(self):
        quotas = self.aws.call('service-quotas', 'list-service-quotas', {'ServiceCode': 'sagemaker'})['Quotas']
        available = [q for q in quotas if 'for endpoint usage' in q['QuotaName'].lower() and q['Value'] >= 1]
        print('Endpoint quotas >= 1 (quota is not proof of IAM permission or physical capacity):')
        for quota in available:
            print(f"  {quota['QuotaName']}: {quota['Value']}")
        matches = [q for q in quotas if q['QuotaName'].lower() == f"{self.c['instance']} for endpoint usage".lower()]
        if len(matches) != 1 or matches[0]['Value'] < 1:
            raise Stop(f"No usable endpoint quota for {self.c['instance']} in {self.c['region']}. Request quota >= 1; no instance fallback.")
        used = 0
        for endpoint in self.aws.call('sagemaker', 'list-endpoints')['Endpoints']:
            described = self.aws.call('sagemaker', 'describe-endpoint', {'EndpointName': endpoint['EndpointName']})
            if described['EndpointStatus'] == 'Failed':
                continue
            config = self.aws.call('sagemaker', 'describe-endpoint-config',
                                   {'EndpointConfigName': described['EndpointConfigName']})
            runtime_variants = {v['VariantName']: v for v in described.get('ProductionVariants', [])}
            for variant in config.get('ProductionVariants', []) + config.get('ShadowProductionVariants', []):
                if variant.get('InstancePools'):
                    raise Stop('Existing heterogeneous endpoint: automatic quota accounting unavailable; ask administrator to verify capacity.')
                if variant.get('InstanceType') == self.c['instance']:
                    runtime = runtime_variants.get(variant['VariantName'], {})
                    used += max(runtime.get('CurrentInstanceCount', variant.get('InitialInstanceCount', 0)),
                                runtime.get('DesiredInstanceCount', 0))
        quota = matches[0]
        save_json(self.state / 'quota.json', {'name': quota['QuotaName'], 'code': quota['QuotaCode'],
                  'limit': quota['Value'], 'observed_usage': used, 'region': self.c['region']})
        existing = self.aws.call('sagemaker', 'describe-endpoint',
                                 {'EndpointName': self.c['endpoint']}, missing=MISSING_SM)
        if existing is None and quota['Value'] - used < 1:
            raise Stop(f'No free observed quota: limit={quota["Value"]}, usage={used}')
        self.status(f'Quota inspected: {quota["Value"]}; observed usage={used}. AWS creation still checks regional support/capacity.')

    def create_model(self):
        role = self.resolve_role()
        pushed = load_json(self.state / 'ecr-pushed.json')
        if pushed['image_id'] != self.require_docker_test()['image_id'] or not pushed['uri'].startswith(self.c['ecr_uri'] + '@'):
            raise Stop('ECR proof does not match the tested image/repository')
        remote_model = self.aws.call('s3api', 'head-object', {
            'Bucket': self.bucket, 'Key': 'models/yolo26m/model.tar.gz',
            'ExpectedBucketOwner': self.c['account']})
        if remote_model.get('Metadata', {}).get('sha256') != digest(self.archive):
            raise Stop('S3 model is not the locally validated artifact')
        # Digest pinning prevents mutable :latest from changing what SageMaker runs.
        desired = {'ModelName': self.c['model'], 'ExecutionRoleArn': role,
            'EnableNetworkIsolation': True, 'PrimaryContainer': {
                'Image': pushed['uri'], 'ModelDataUrl': f's3://{self.bucket}/models/yolo26m/model.tar.gz',
                'Environment': {'MODEL_PATH': '/opt/ml/model/best.pt',
                                'MODEL_DEVICE': 'cuda' if self.gpu else 'cpu'}}}
        existing = self.aws.call('sagemaker', 'describe-model', {'ModelName': self.c['model']}, missing=MISSING_SM)
        if existing is None:
            self.aws.call('sagemaker', 'create-model', {**desired, 'Tags': TAGS})
        elif not contains(existing, desired):
            raise Stop('Existing SageMaker Model differs; no overwrite. Select new versioned names.')
        save_json(self.state / 'model.json', desired)
        self.status('SageMaker Model ready: ' + self.c['model'])

    def endpoint_configuration(self):
        return {'EndpointConfigName': self.c['endpoint_config'], 'ProductionVariants': [{
            'VariantName': 'AllTraffic', 'ModelName': self.c['model'], 'InitialInstanceCount': 1,
            'InitialVariantWeight': 1.0, 'InstanceType': self.c['instance'],
            'ModelDataDownloadTimeoutInSeconds': 600,
            'ContainerStartupHealthCheckTimeoutInSeconds': 600}],
            'AsyncInferenceConfig': {
                'ClientConfig': {'MaxConcurrentInvocationsPerInstance': 1},
                'OutputConfig': {'S3OutputPath': f's3://{self.bucket}/async-output/',
                                 'S3FailurePath': f's3://{self.bucket}/async-failure/'}}}

    def create_endpoint_config(self):
        self.check_quota()
        desired = self.endpoint_configuration()
        existing = self.aws.call('sagemaker', 'describe-endpoint-config',
                                 {'EndpointConfigName': self.c['endpoint_config']}, missing=MISSING_SM)
        if existing is None:
            self.aws.call('sagemaker', 'create-endpoint-config', {**desired, 'Tags': TAGS})
        elif not contains(existing, desired):
            raise Stop('Existing EndpointConfig differs; no replacement. Select a new name.')
        save_json(self.state / 'endpoint-config.json', desired)
        self.status('Async EndpointConfig ready: ' + self.c['endpoint_config'])

    def display_endpoint(self):
        print(f"\nVoy a crear el endpoint {self.c['endpoint']} usando la instancia "
              f"{self.c['instance']} en la region {self.c['region']}.\n"
              f"Account: {self.c['account']}\nInitialInstanceCount=1; autoscaling 0..1 is configured AFTER creation.\n"
              'The instance accrues charges while provisioned, including startup/idle time before scale-in.\n',
              flush=True)

    def create_endpoint(self):
        # Revalidate the real remote configuration before showing/capturing approval.
        configuration = self.aws.call('sagemaker', 'describe-endpoint-config',
                                      {'EndpointConfigName': self.c['endpoint_config']})
        if not contains(configuration, self.endpoint_configuration()):
            raise Stop('Remote EndpointConfig differs from reviewed configuration')
        endpoint = self.aws.call('sagemaker', 'describe-endpoint',
                                 {'EndpointName': self.c['endpoint']}, missing=MISSING_SM)
        if endpoint is None:
            self.check_quota()
            self.display_endpoint()
            confirmation = f"CREAR {self.c['endpoint']} {self.c['instance']} {self.c['region']}"
            try:
                answer = input('Confirm explicitly by typing:\n' + confirmation + '\n> ')
            except EOFError:
                answer = ''
            if answer != confirmation:
                raise Stop('No endpoint created: explicit confirmation was not received')
            self.aws.call('sagemaker', 'create-endpoint', {'EndpointName': self.c['endpoint'],
                          'EndpointConfigName': self.c['endpoint_config'], 'Tags': TAGS})
        elif endpoint['EndpointConfigName'] != self.c['endpoint_config']:
            raise Stop('Existing endpoint uses another configuration; no automatic update/delete')
        self.wait_endpoint()
        self.status('Async Endpoint InService: ' + self.c['endpoint'])

    def wait_endpoint(self):
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            endpoint = self.aws.call('sagemaker', 'describe-endpoint', {'EndpointName': self.c['endpoint']})
            if endpoint['EndpointConfigName'] != self.c['endpoint_config']:
                raise Stop('Endpoint configuration changed; stop before scaling or inference')
            status = endpoint['EndpointStatus']
            print('EndpointStatus=' + status, flush=True)
            if status == 'InService':
                return endpoint
            if status not in {'Creating', 'Updating', 'SystemUpdating'}:
                raise Stop(f'Endpoint status={status}. AWS FailureReason: {endpoint.get("FailureReason", "not provided")}')
            time.sleep(30)
        raise Stop('Endpoint readiness wait timed out. Endpoint was NOT deleted; inspect status and charges.')

    def configure_autoscaling(self):
        self.wait_endpoint()
        target = {'ServiceNamespace': 'sagemaker', 'ResourceId': self.resource_id,
                  'ScalableDimension': 'sagemaker:variant:DesiredInstanceCount'}
        self.aws.call('application-autoscaling', 'register-scalable-target',
                      {**target, 'MinCapacity': 0, 'MaxCapacity': 1})
        self.aws.call('application-autoscaling', 'put-scaling-policy', {**target,
            'PolicyName': self.c['endpoint'] + '-backlog', 'PolicyType': 'TargetTrackingScaling',
            'TargetTrackingScalingPolicyConfiguration': {
                'TargetValue': 1.0, 'ScaleInCooldown': 300, 'ScaleOutCooldown': 60,
                'CustomizedMetricSpecification': {'MetricName': 'ApproximateBacklogSizePerInstance',
                    'Namespace': 'AWS/SageMaker', 'Dimensions': [{'Name': 'EndpointName', 'Value': self.c['endpoint']}],
                    'Statistic': 'Average'}}})
        policy = self.aws.call('application-autoscaling', 'put-scaling-policy', {**target,
            'PolicyName': self.c['endpoint'] + '-wake', 'PolicyType': 'StepScaling',
            'StepScalingPolicyConfiguration': {'AdjustmentType': 'ChangeInCapacity',
                'MetricAggregationType': 'Average', 'Cooldown': 300,
                'StepAdjustments': [{'MetricIntervalLowerBound': 0, 'ScalingAdjustment': 1}]}})
        self.aws.call('cloudwatch', 'put-metric-alarm', {
            'AlarmName': self.c['endpoint'] + '-wake', 'MetricName': 'HasBacklogWithoutCapacity',
            'Namespace': 'AWS/SageMaker', 'Statistic': 'Average', 'Period': 60,
            'EvaluationPeriods': 2, 'DatapointsToAlarm': 2, 'Threshold': 1,
            'ComparisonOperator': 'GreaterThanOrEqualToThreshold', 'TreatMissingData': 'missing',
            'Dimensions': [{'Name': 'EndpointName', 'Value': self.c['endpoint']}],
            'AlarmActions': [policy['PolicyARN']]})
        actual = self.aws.call('application-autoscaling', 'describe-scalable-targets', {
            'ServiceNamespace': 'sagemaker', 'ResourceIds': [self.resource_id],
            'ScalableDimension': target['ScalableDimension']})['ScalableTargets']
        if len(actual) != 1 or actual[0]['MinCapacity'] != 0 or actual[0]['MaxCapacity'] != 1:
            raise Stop('Autoscaling limits not verified')
        self.status('Autoscaling configured: min=0, max=1, wake from zero enabled')
        print('Cold start includes instance/model startup; measure separately from inference_time_ms.')

    def test_async_inference(self):
        self.wait_endpoint()
        test_image, size = self.prepare_test_image()
        inference_id = str(uuid.uuid4())
        key = f'async-input/{inference_id}/test.jpg'
        self.aws.call('s3api', 'put-object', {'Bucket': self.bucket, 'Key': key,
                      'ContentType': 'image/jpeg', 'ServerSideEncryption': 'AES256'},
                      extra=('--body', str(test_image)))
        started = time.monotonic()
        reply = self.aws.call('sagemaker-runtime', 'invoke-endpoint-async', {
            'EndpointName': self.c['endpoint'], 'InferenceId': inference_id,
            'ContentType': 'image/jpeg', 'Accept': 'application/json',
            'CustomAttributes': 'request_id=' + inference_id,
            'InputLocation': f's3://{self.bucket}/{key}', 'RequestTTLSeconds': 3600,
            'InvocationTimeoutSeconds': 300})
        save_json(self.state / (inference_id + '-invocation.json'), reply)
        print(json.dumps(reply, indent=2), flush=True)
        locations = [('output', reply['OutputLocation']), ('failure', reply.get('FailureLocation'))]
        while time.monotonic() - started < 3600:
            for kind, location in locations:
                if not location:
                    continue
                match = re.fullmatch(r's3://([^/]+)/(.+)', location)
                if not match or match[1] != self.bucket:
                    raise Stop('Unexpected asynchronous output location')
                head = self.aws.call('s3api', 'head-object', {'Bucket': self.bucket, 'Key': match[2]},
                                     missing=MISSING_S3)
                if head is None:
                    continue
                if head['ContentLength'] > 1024 * 1024:
                    raise Stop('Asynchronous output is too large')
                destination = self.state / f'{inference_id}-{kind}.json'
                self.aws.call('s3api', 'get-object', {'Bucket': self.bucket, 'Key': match[2]},
                              extra=(str(destination),))
                body = load_json(destination)
                if kind == 'failure':
                    raise Stop('AWS asynchronous failure:\n' + json.dumps(body, indent=2))
                valid_response(body, self.labels, size)
                save_json(self.state / (inference_id + '-measurement.json'), {
                    'inference_id': inference_id, 'elapsed_ms': round((time.monotonic() - started) * 1000, 2),
                    'processing_time_ms': body['processing_time_ms'], 'output_location': location})
                self.status('test.jpg processed; valid normalized JSON downloaded: ' + str(destination))
                print(json.dumps(body, indent=2))
                self.cloudwatch(inference_id)
                return
            print('Waiting for S3 output (includes queue/cold start)...', flush=True)
            time.sleep(15)
        raise Stop('Timed out waiting for output. Invocation metadata saved; request may still complete. No resources deleted.')

    def cloudwatch(self, inference_id=None):
        group = '/aws/sagemaker/Endpoints/' + self.c['endpoint']
        # CloudWatch ingestion is eventually consistent.
        deadline = time.monotonic() + 180
        pattern = '{ $.event = "request" && $.status = "success" }'
        if inference_id:
            # Only UUIDs generated by this workflow can be used in a log filter.
            inference_id = str(uuid.UUID(inference_id))
            pattern = '{ $.event = "request" && $.status = "success" && $.request_id = "' + inference_id + '" }'
        while True:
            response = self.aws.call('logs', 'filter-log-events', {
                'logGroupName': group, 'startTime': int((time.time() - 3600) * 1000),
                'limit': 100, 'filterPattern': pattern},
                missing=(r'\(ResourceNotFoundException\)',))
            requests = (response or {}).get('events', [])
            if requests:
                entry = requests[-1]
                record = json.loads(entry['message'])
                startup = self.aws.call('logs', 'get-log-events', {'logGroupName': group,
                    'logStreamName': entry['logStreamName'], 'startFromHead': True, 'limit': 100})
                records = []
                for item in startup.get('events', []):
                    try:
                        records.append(json.loads(item['message']))
                    except ValueError:
                        continue
                expected_device = 'cuda' if self.gpu else 'cpu'
                observed = {r.get('event') for r in records if r.get('device') == expected_device}
                if {'startup', 'model_loaded'} <= observed and record.get('device') == expected_device:
                    for item in records:
                        if item.get('event') in {'startup', 'model_loaded'}:
                            print(json.dumps(item))
                    print(entry['message'])
                    self.status('CloudWatch startup/model/device/request logs visible: ' + group)
                    return
            if time.monotonic() >= deadline:
                raise Stop('CloudWatch startup/model/device/request evidence incomplete; inspect ' + group)
            time.sleep(15)

    def destroy_endpoint(self):
        endpoint = self.aws.call('sagemaker', 'describe-endpoint',
                                 {'EndpointName': self.c['endpoint']}, missing=MISSING_SM)
        if endpoint is None:
            print('Endpoint does not exist; nothing deleted.')
            return
        print(f"Delete endpoint {self.c['endpoint']} in account {self.c['account']}, region {self.c['region']}.")
        print('Also deregisters its autoscaling target and removes its wake alarm. Model, config, S3 and ECR are retained.')
        try:
            answer = input(f"Type ELIMINAR {self.c['endpoint']} to confirm:\n> ")
        except EOFError:
            answer = ''
        if answer != f"ELIMINAR {self.c['endpoint']}":
            raise Stop('Deletion not confirmed; no resource deleted')
        target = {'ServiceNamespace': 'sagemaker', 'ResourceId': self.resource_id,
                  'ScalableDimension': 'sagemaker:variant:DesiredInstanceCount'}
        self.aws.call('application-autoscaling', 'deregister-scalable-target', target,
                      missing=(r'\(ObjectNotFoundException\)',))
        self.aws.call('cloudwatch', 'delete-alarms', {'AlarmNames': [self.c['endpoint'] + '-wake']})
        self.aws.call('sagemaker', 'delete-endpoint', {'EndpointName': self.c['endpoint']})
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if self.aws.call('sagemaker', 'describe-endpoint',
                             {'EndpointName': self.c['endpoint']}, missing=MISSING_SM) is None:
                self.status('Endpoint deletion verified. S3/ECR artifacts retained.')
                return
            time.sleep(15)
        raise Stop('Deletion still pending; inspect SageMaker status')

    def deploy(self):
        self.check_environment()
        from api_deployment import check_api_environment
        check_api_environment(self)
        self.package_api()
        self.package_model()
        self.build_docker()
        self.test_docker()
        self.require_docker_test()
        self.create_s3()
        self.upload_model()
        self.create_ecr()
        self.push_ecr()
        self.create_model()
        self.create_endpoint_config()
        # Interactive cost gate is mandatory, including direct create-endpoint.ps1.
        self.create_endpoint()
        try:
            self.configure_autoscaling()
        except Stop:
            print('[ACTION REQUIRED] Endpoint exists and may keep billing. Autoscaling did not finish. '
                  'Resolve the reported permission or explicitly run destroy-endpoint.ps1.')
            raise
        self.test_async_inference()
        self.deploy_api()
        self.summary()

    def summary(self):
        for key in ('bucket', 'ecr_uri', 'model', 'endpoint', 'region', 'instance'):
            print(f'{key}: {self.c.get(key, "unresolved")}')

    def package_api(self):
        from api_deployment import package_api
        package_api(self)

    def deploy_api(self):
        from api_deployment import deploy_api
        deploy_api(self)

    def error_report(self, error):
        self.state.mkdir(parents=True, exist_ok=True)
        text = str(error)
        if isinstance(error, CommandFailure) and error.command[0] == 'aws':
            service, operation = error.command[1:3]
            if getattr(error, 'input_payload', None) is not None:
                text += '\nRequest JSON:\n' + json.dumps(error.input_payload, indent=2)
            iam_service = {'s3api': 's3', 'sagemaker-runtime': 'sagemaker',
                           'service-quotas': 'servicequotas'}.get(service, service)
            special = {
                ('s3api', 'head-bucket'): 's3:ListBucket',
                ('s3api', 'head-object'): 's3:GetObject',
                ('s3api', 'put-public-access-block'): 's3:PutBucketPublicAccessBlock',
                ('s3api', 'put-bucket-encryption'): 's3:PutEncryptionConfiguration',
                ('s3api', 'get-bucket-encryption'): 's3:GetEncryptionConfiguration',
                ('ecr', 'get-login-password'): 'ecr:GetAuthorizationToken',
                ('sagemaker', 'create-model'): 'sagemaker:CreateModel, sagemaker:AddTags, iam:PassRole (execution role only)',
                ('sagemaker', 'create-endpoint-config'): 'sagemaker:CreateEndpointConfig, sagemaker:AddTags',
                ('sagemaker', 'create-endpoint'): 'sagemaker:CreateEndpoint, sagemaker:AddTags',
            }
            permission = special.get((service, operation), iam_service + ':' + ''.join(x.title() for x in operation.split('-')))
            text += '\nRequired IAM action(s): ' + permission
            if self.c.get('account') and service == 'iam':
                api_role = self.c.get('api_role_arn', '')
                payload = getattr(error, 'input_payload', {}) or {}
                if api_role and payload.get('RoleName') == api_role.rsplit('/', 1)[-1]:
                    role = api_role
                    text += '\nAPI Lambda role: see proposed-api-role-policy.json and proposed-api-role-trust.json.'
                else:
                    self.write_iam_proposal()
                    role = self.c.get('role_arn') or f"arn:aws:iam::{self.c['account']}:role/service-role/MagicVisionSageMakerExecutionRole"
                text += '\nExecution Role ARN to request/verify with administrator (not asserted to exist): ' + role
            text += (f'\n\nRequired AWS service/operation: {service}:{operation}\n'
                     'Text for the AWS laboratory administrator:\n'
                     f'I am deploying MagicVision full YOLO26m for asynchronous inference only in '
                     f'account {self.c.get("account", "not authenticated")}, region {self.c.get("region") or "not configured"}. '
                     f'The attached command failed with the exact error above. Please verify the scoped permission '
                     f'for {service}:{operation}, the SageMaker endpoint quota and the provided execution role. '
                     'I do not request AdministratorAccess or training permissions. '
                     'If IAM is restricted, please provide an existing SageMaker execution Role ARN with access '
                     'to the MagicVision model/input/output bucket, ECR repository and endpoint CloudWatch logs; '
                     'the deployer also needs iam:PassRole restricted to that role. '
                     'Autoscaling requires Application Auto Scaling, CloudWatch alarms, '
                     'and its SageMaker service-linked role provisioned by the administrator.')
        else:
            text += '\nDeployment stopped. No automatic cleanup or privilege changes were performed.'
        (self.state / 'last-error.txt').write_text(text + '\n', encoding='utf-8')
        print(text, file=sys.stderr)


def main():
    # Windows redirected consoles may default to cp1252; library logs contain Unicode.
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--step', required=True)
    args = parser.parse_args()
    workflow = Workflow(load_json(args.config))
    steps = {name.replace('_', '-'): getattr(workflow, name) for name in (
        'check_environment', 'package_model', 'build_docker', 'test_docker', 'create_s3',
        'upload_model', 'create_ecr', 'push_ecr', 'create_model', 'create_endpoint_config',
        'create_endpoint', 'configure_autoscaling', 'test_async_inference',
        'destroy_endpoint', 'cloudwatch', 'deploy', 'package_api', 'deploy_api')}
    if args.step not in steps:
        parser.error('Unknown step')
    try:
        if args.step not in {'deploy', 'check-environment', 'package-model', 'build-docker', 'test-docker', 'package-api'}:
            workflow.check_aws()
            if args.step in {'create-s3', 'upload-model', 'create-ecr', 'push-ecr', 'create-model',
                             'create-endpoint-config', 'create-endpoint'}:
                workflow.require_docker_test()
        steps[args.step]()
    except (Stop, OSError, ValueError, KeyError, tarfile.TarError) as error:
        workflow.error_report(error)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
