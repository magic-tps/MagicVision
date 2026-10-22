"""Private, temporary S3 jobs. Images never travel back to the mobile client."""
import io
import hashlib
import json
import logging
import time
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from botocore.exceptions import ClientError
from fastapi import HTTPException
from PIL import Image, ImageOps

from schema import InferenceResponse
from depth_schema import frame_attributes, validate_response

log = logging.getLogger('magicvision.jobs')
log.setLevel(logging.INFO)


class AsyncJobs:
    def __init__(self, s3, runtime, bucket, endpoint, clock=time.time,
                 scene_ttl=30, warmup_ttl=900):
        self.s3, self.runtime = s3, runtime
        self.bucket, self.endpoint = bucket, endpoint
        self.clock, self.scene_ttl, self.warmup_ttl = clock, scene_ttl, warmup_ttl

    @staticmethod
    def key(job_id):
        try:
            job_id = str(UUID(job_id))
        except (ValueError, TypeError, AttributeError):
            raise HTTPException(404, 'Job not found') from None
        return f'app-jobs/{job_id}.json'

    def read(self, key, limit=1024 * 1024):
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            if error.response['Error']['Code'] in {'NoSuchKey', '404', 'NotFound'}:
                return None
            raise
        stream = response['Body']
        try:
            data = stream.read(limit + 1)
        finally:
            stream.close()
        if len(data) > limit:
            raise HTTPException(502, 'Oversized inference response')
        return json.loads(data)

    def write(self, job):
        self.s3.put_object(Bucket=self.bucket, Key=self.key(job['job_id']),
            Body=json.dumps(job, allow_nan=False).encode(), ContentType='application/json',
            ServerSideEncryption='AES256')

    def output_key(self, location):
        parsed = urlsplit(location)
        key = parsed.path.lstrip('/')
        if (parsed.scheme != 's3' or parsed.netloc != self.bucket or parsed.query or parsed.fragment
                or not key.startswith(('async-output/', 'async-failure/'))):
            raise HTTPException(502, 'Unexpected inference output location')
        return key

    def cleanup(self, job):
        # Only objects of this job; never the model, bucket or another job.
        for key in (job.get('input_key'), job.get('output_key'), job.get('failure_key')):
            if key:
                try:
                    self.s3.delete_object(Bucket=self.bucket, Key=key)
                except ClientError:
                    log.warning(json.dumps({'event': 'cleanup_retry_needed', 'job_id': job['job_id']}))

    def warmup(self, request_id):
        image = io.BytesIO()
        Image.new('RGB', (640, 480), (96, 96, 96)).save(image, 'JPEG', quality=70)
        return self.submit(image.getvalue(), request_id, (640, 480), kind='warmup')

    def submit(self, data, request_id, size, kind='inference', frame_metadata=None):
        job_id = str(uuid4())
        ttl = self.warmup_ttl if kind == 'warmup' else self.scene_ttl
        job = {'job_id': job_id, 'request_id': request_id, 'kind': kind,
            'status': 'processing', 'expires_at': self.clock() + ttl,
            'width': size[0], 'height': size[1], 'input_key': f'async-input/app/{job_id}.jpg'}
        if frame_metadata is not None:
            job['frame_metadata'] = frame_metadata
            with Image.open(io.BytesIO(data)) as image:
                job['pixel_sha256'] = hashlib.sha256(ImageOps.exif_transpose(image).convert('RGB').tobytes()).hexdigest()
        self.s3.put_object(Bucket=self.bucket, Key=job['input_key'], Body=data,
            ContentType='image/jpeg', ServerSideEncryption='AES256')
        try:
            reply = self.runtime.invoke_endpoint_async(EndpointName=self.endpoint,
                InferenceId=job_id, InputLocation=f's3://{self.bucket}/{job["input_key"]}',
                ContentType='image/jpeg', Accept='application/json',
                CustomAttributes=frame_attributes(frame_metadata) if frame_metadata is not None else f'request_id={request_id}',
                S3OutputPathExtension='app',
                RequestTTLSeconds=max(60, ttl), InvocationTimeoutSeconds=300 if kind == 'warmup' else 30)
            if reply.get('InferenceId') != job_id:
                raise HTTPException(502, 'Invalid inference correlation')
            job['output_key'] = self.output_key(reply['OutputLocation'])
            job['failure_key'] = self.output_key(reply['FailureLocation']) if reply.get('FailureLocation') else None
            self.write(job)
        except Exception:
            self.cleanup(job)
            raise
        log.info(json.dumps({'event': 'submitted', 'request_id': request_id, 'kind': kind}))
        return self.pending(job)

    @staticmethod
    def pending(job):
        return {'job_id': job['job_id'], 'request_id': job['request_id'],
                'status': 'processing', 'poll_after_ms': 1000}

    def get(self, job_id):
        key = self.key(job_id)
        job = self.read(key)
        if job is None:
            raise HTTPException(404, 'Job not found')
        if self.read(key + '.cancelled') is not None or self.clock() >= job['expires_at']:
            self.cleanup(job)
            raise HTTPException(410, 'Capture expired or cancelled; capture again')
        if job['status'] == 'failed':
            raise HTTPException(503, 'Asynchronous inference failed')
        if job['status'] == 'completed':
            return job['result'], 200
        failure = self.read(job['failure_key']) if job.get('failure_key') else None
        if failure is not None:
            job['status'] = 'failed'
            self.write(job)
            self.cleanup(job)
            raise HTTPException(503, 'Asynchronous inference failed')
        body = self.read(job['output_key'])
        if body is None:
            return self.pending(job), 202
        try:
            if job.get('frame_metadata'):
                output = validate_response(body, job['frame_metadata'], (job['width'], job['height']))
                if output['pixel_sha256'] != job['pixel_sha256']:
                    raise ValueError('Processed image differs from uploaded frame')
            else:
                result = InferenceResponse.model_validate({**body, 'request_id': job['request_id']})
                if (result.image_width, result.image_height) != (job['width'], job['height']):
                    raise ValueError('Image dimensions changed')
                output = ({'request_id': job['request_id'], 'status': 'ready'} if job['kind'] == 'warmup'
                          else result.model_dump())
        except (ValueError, TypeError, KeyError):
            job['status'] = 'failed'
            self.write(job)
            self.cleanup(job)
            raise HTTPException(502, 'Invalid inference response') from None
        if self.read(key + '.cancelled') is not None or self.clock() >= job['expires_at']:
            self.cleanup(job)
            raise HTTPException(410, 'Capture expired; capture again')
        job.update(status='completed', result=output)
        self.write(job)
        self.cleanup(job)
        return output, 200

    def cancel(self, job_id):
        key = self.key(job_id)
        job = self.read(key)
        if job:
            # Separate marker prevents a concurrent poll from resurrecting cancellation.
            self.s3.put_object(Bucket=self.bucket, Key=key + '.cancelled',
                Body=b'{"cancelled":true}', ContentType='application/json', ServerSideEncryption='AES256')
            self.cleanup(job)
        # SageMaker may finish an already-running invocation; it will never be delivered.
        return {'status': 'cancelled'}
