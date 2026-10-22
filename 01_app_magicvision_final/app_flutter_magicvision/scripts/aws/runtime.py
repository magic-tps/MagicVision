"""CLI transport. Never shells out through shell=True or captures credentials to disk."""
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path


class Stop(RuntimeError):
    pass


class CommandFailure(Stop):
    def __init__(self, command, stderr, code):
        self.command, self.stderr, self.code = command, stderr, code
        super().__init__(f"Command: {shlex.join(command)}\nExit: {code}\n{stderr}")


def run(args, *, data=None, timeout=120, live=False, merge_output=False):
    try:
        result = subprocess.run([str(a) for a in args], input=data, text=True,
                                encoding='utf-8', errors='replace', timeout=timeout,
                                stdout=None if live else subprocess.PIPE,
                                stderr=subprocess.STDOUT if live or merge_output else subprocess.PIPE, shell=False)
    except FileNotFoundError:
        raise Stop(f"Executable missing: {args[0]}. Install/configure it before continuing.") from None
    except subprocess.TimeoutExpired:
        raise Stop(f"Command timed out: {shlex.join([str(a) for a in args])}") from None
    if result.returncode:
        raise CommandFailure([str(a) for a in args], result.stderr or result.stdout or 'See command output.', result.returncode)
    return (result.stdout or '').strip()


def digest(path):
    with open(path, 'rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class Aws:
    def __init__(self, region=''):
        self.region = region

    def call(self, service, operation, payload=None, *, extra=(), missing=()):
        args = ['aws', service, operation, '--output', 'json', '--no-cli-pager']
        args += ['--cli-connect-timeout', '10', '--cli-read-timeout', '60']
        if self.region:
            args += ['--region', self.region]
        temporary = None
        try:
            if payload is not None:
                handle, temporary = tempfile.mkstemp(suffix='.json', prefix='magicvision-aws-')
                os.close(handle)
                save_json(temporary, payload)
                args += ['--cli-input-json', 'file://' + Path(temporary).as_posix()]
            args += list(extra)
            try:
                output = run(args)
            except CommandFailure as error:
                # AccessDenied/Forbidden/ExpiredToken must never be interpreted as absence.
                if any(re.search(pattern, error.stderr, re.I) for pattern in missing):
                    return None
                error.input_payload = payload
                raise
            return json.loads(output) if output else {}
        finally:
            if temporary:
                Path(temporary).unlink(missing_ok=True)


MISSING_SM = (r'Could not find (?:model|endpoint|endpoint configuration)',
              r'Cannot find (?:model|endpoint|endpoint configuration)',
              r'\(ResourceNotFound(?:Exception)?\)')
MISSING_S3 = (r'\(404\)', r'\(NoSuchKey\)', r'\(NotFound\)')
MISSING_ECR = (r'\(RepositoryNotFoundException\)',)


def valid_response(body, labels, expected_size):
    if not isinstance(body, dict) or set(body) != {
            'processing_time_ms', 'image_width', 'image_height', 'detections'}:
        raise Stop('Model response has incorrect fields')
    def number(value, lower, upper):
        return type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper
    if not number(body['processing_time_ms'], 0, 3600000):
        raise Stop('Invalid inference time')
    if any(type(body[k]) is not int for k in ('image_width', 'image_height')):
        raise Stop('Invalid image dimension type')
    if (body['image_width'], body['image_height']) != tuple(expected_size):
        raise Stop('Response dimensions do not match test image')
    if not isinstance(body['detections'], list) or len(body['detections']) > 300:
        raise Stop('Invalid detections array')
    for detection in body['detections']:
        if not isinstance(detection, dict) or set(detection) != {'class_id', 'label', 'confidence', 'bbox'}:
            raise Stop('Invalid detection fields')
        class_id = detection['class_id']
        if type(class_id) is not int or not 0 <= class_id < len(labels) or detection['label'] != labels[class_id]:
            raise Stop('Incorrect class/label')
        if not number(detection['confidence'], 0, 1):
            raise Stop('Invalid confidence')
        box = detection['bbox']
        if not isinstance(box, dict) or set(box) != {'left', 'top', 'right', 'bottom'}:
            raise Stop('Invalid bbox fields')
        if not all(number(v, 0, 1) for v in box.values()):
            raise Stop('Bounding box must be finite and normalized')
        if box['left'] >= box['right'] or box['top'] >= box['bottom']:
            raise Stop('Bounding box must have positive area')
    return body


def discover_region(configured=''):
    region = configured or os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION')
    if not region:
        try:
            region = run(['aws', 'configure', 'get', 'region'])
        except CommandFailure as error:
            if error.code != 1 or error.stderr.strip():
                raise
    if not region:
        raise Stop('No AWS region configured. Request the assigned region from your laboratory; do not guess.')
    if not re.fullmatch(r'[a-z]{2}(?:-[a-z]+)+-\d+', region):
        raise Stop('Invalid AWS region syntax')
    return region
