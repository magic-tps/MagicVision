"""Local HTTPS integration with real model; only AWS transport/camera are simulated."""
import datetime
import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import shutil
import socket
import ssl
import subprocess
import sys
import tarfile
import threading
import time
from urllib.request import urlopen, build_opener, ProxyHandler, HTTPSHandler
from uuid import uuid4

import uvicorn
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'backend/lambda'), str(ROOT / 'backend')]
from api import create_app
from async_jobs import AsyncJobs
from testing.transport import MemoryS3, TestAsyncRuntime


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def run(args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, shell=False, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-model', action='store_true', help='Run current serve.py locally without Docker; AWS transport remains simulated')
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'backend/artifacts/local-integration')
    parser.add_argument('--image', type=Path, default=ROOT / 'ml/datasets/dataset_final_magicvision/images/test/coco_filtered_000000000872.jpg')
    args_cli = parser.parse_args()
    artifacts = args_cli.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    model = artifacts / 'model'
    model.mkdir(exist_ok=True)
    if args_cli.native_model:
        weights = ROOT / 'runs/detect/modelo_final_yolo26m_80e_640_b8/weights/best.pt'
        if not (model / 'best.pt').exists(): shutil.copy2(weights, model / 'best.pt')
        shutil.copy2(ROOT / 'assets/models/labels.txt', model / 'labels.txt')
        (model / 'config.json').write_text(json.dumps({'imgsz':640,'confidence':.30,'iou':.65,'max_det':100}))
    else:
        with tarfile.open(ROOT / 'backend/artifacts/model.tar.gz') as archive:
            if {m.name for m in archive.getmembers()} != {'best.pt', 'labels.txt', 'config.json'}:
                raise ValueError('Unexpected archive members')
            archive.extractall(model, filter='data')
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'localhost')])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName('localhost'),
            x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True).sign(key, hashes.SHA256()))
    cert_path, key_path = artifacts / 'localhost.crt', artifacts / 'localhost.key'
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                          serialization.NoEncryption()))
    model_port, api_port = port(), port()
    container = 'magicvision-integration-' + uuid4().hex[:12]
    server = None
    thread = None
    native_server = None
    native_thread = None
    try:
        if args_cli.native_model:
            from werkzeug.serving import make_server
            sys.path.insert(0, str(ROOT / 'backend/sagemaker'))
            import serve as model_server
            os.environ['MODEL_PATH'] = str(model / 'best.pt')
            os.environ['MODEL_DEVICE'] = 'cuda'
            native_server = make_server('127.0.0.1', model_port, model_server.create_app(*model_server.load_model()))
            native_thread = threading.Thread(target=native_server.serve_forever, daemon=True)
            native_thread.start()
        else:
            run(['docker', 'run', '-d', '--name', container, '--gpus', 'all',
                 '-p', f'127.0.0.1:{model_port}:8080', '-e', 'MODEL_DEVICE=cuda',
                 '--mount', f'type=bind,source={model},target=/opt/ml/model,readonly',
                 'magicvision-yolo26m:latest'])
        model_url = f'http://127.0.0.1:{model_port}'
        deadline = time.monotonic() + 180
        while True:
            try:
                with urlopen(model_url + '/ping', timeout=2) as response:
                    health = json.load(response)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError('Model did not become healthy')
                time.sleep(1)
        s3 = MemoryS3()
        runtime = TestAsyncRuntime(s3, model_url)
        jobs = AsyncJobs(s3, runtime, 'local-test-only', 'local-test-only')
        app = create_app(endpoint='local-test-only', jobs=jobs, mode='async')
        server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=api_port,
            ssl_keyfile=str(key_path), ssl_certfile=str(cert_path), log_level='warning'))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        context = ssl.create_default_context(cafile=str(cert_path))
        local_https = build_opener(ProxyHandler({}), HTTPSHandler(context=context))
        deadline = time.monotonic() + 20
        while True:
            try:
                with local_https.open(f'https://127.0.0.1:{api_port}/health', timeout=2):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError('HTTPS API did not become healthy')
                time.sleep(.1)
        report = artifacts / 'flutter-result.json'
        # Clear only our previous report so a failed run cannot look successful.
        report.unlink(missing_ok=True)
        flutter = shutil.which('flutter')
        if not flutter:
            raise RuntimeError('Flutter missing from PATH')
        args = [flutter, 'test', 'test/local_model_integration_test.dart', '--reporter', 'expanded',
            f'--dart-define=LOCAL_INTEGRATION_URL=https://127.0.0.1:{api_port}',
            f'--dart-define=LOCAL_INTEGRATION_CERT={cert_path}',
            f'--dart-define=LOCAL_INTEGRATION_IMAGE={args_cli.image.resolve()}',
            f'--dart-define=LOCAL_INTEGRATION_REPORT={report}']
        # PowerShell invokes the Windows Flutter batch wrapper with literal arguments.
        if os.name == 'nt':
            invoke = '& ' + ' '.join("'" + str(arg).replace("'", "''") + "'" for arg in args)
            args = ['powershell', '-NoProfile', '-Command', invoke + '\nexit $LASTEXITCODE']
        run(args, cwd=ROOT, timeout=240)
        if len(runtime.calls) != 2:
            raise AssertionError('Expected exactly one warmup and one captured scene')
        if any(k.endswith(('.jpg', '.out')) for k in s3.objects):
            raise AssertionError('Temporary image/model outputs were not cleaned')
        summary = {'validated_at': now.isoformat(), 'model_health': health,
            'model_image': ('native-current-serve:' + hashlib.sha256((model / 'best.pt').read_bytes()).hexdigest()) if args_cli.native_model else subprocess.check_output(['docker', 'inspect', container, '--format', '{{.Image}}'], text=True).strip(),
            'invocations': len(runtime.calls), 'temporary_images_remaining': 0,
            'flutter_report': str(report), 'aws_tested': False, 'usb_tested': False, 'audible_tts_tested': False}
        (artifacts / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if server:
            server.should_exit = True
        if thread:
            thread.join(timeout=10)
        if native_server:
            native_server.shutdown()
        if native_thread:
            native_thread.join(timeout=10)
        if not args_cli.native_model:
            with (artifacts / 'model.log').open('w', encoding='utf-8') as log:
                subprocess.run(['docker', 'logs', container], stdout=log, stderr=subprocess.STDOUT)
            subprocess.run(['docker', 'rm', '-f', container], stdout=subprocess.DEVNULL)
        key_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
