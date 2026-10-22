"""HTTPS API (API Gateway/Lambda or a backend behind a TLS reverse proxy).

Only this process receives an IAM execution role. Async images use temporary S3 objects.
"""
import io
import json
import logging
import os
import time
from uuid import UUID, uuid4

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, ReadTimeoutError
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from mangum import Mangum
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import ValidationError
from starlette.responses import JSONResponse

from schema import InferenceResponse
from async_jobs import AsyncJobs
from depth_schema import frame_attributes, validate_metadata, validate_response

MAX_JPEG_BYTES = 2 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_JPEG_BYTES + 16384
log = logging.getLogger('magicvision.inference')
log.setLevel(logging.INFO)


def create_app(runtime=None, endpoint=None, jobs=None, mode=None, s3=None, bucket=None):
    app = FastAPI(title='MagicVision inference', docs_url=None, redoc_url=None)
    app.state.runtime = runtime
    app.state.endpoint = endpoint or os.getenv('SAGEMAKER_ENDPOINT_NAME', '')
    app.state.mode = mode or os.getenv('INFERENCE_MODE', 'sync')
    app.state.jobs = jobs

    def get_jobs():
        if app.state.mode != 'async':
            raise HTTPException(404, 'Asynchronous jobs are not enabled')
        if app.state.jobs is None:
            job_bucket = bucket or os.getenv('ASYNC_BUCKET', '')
            if not job_bucket or not app.state.endpoint:
                raise HTTPException(503, 'Asynchronous inference is not configured')
            config = Config(connect_timeout=2, read_timeout=6, retries={'total_max_attempts': 1})
            app.state.jobs = AsyncJobs(s3 or boto3.client('s3', config=config),
                app.state.runtime or boto3.client('sagemaker-runtime', config=config),
                job_bucket, app.state.endpoint)
        return app.state.jobs

    def job_response(action):
        try:
            body, status = action()
            return JSONResponse(body, status_code=status, headers={
                'Cache-Control': 'no-store', **({'Retry-After': '1'} if status == 202 else {})})
        except HTTPException:
            raise
        except (ReadTimeoutError, ClientError, BotoCoreError):
            raise HTTPException(503, 'Asynchronous inference is unavailable') from None
        except (ValueError, TypeError, KeyError):
            raise HTTPException(502, 'Invalid asynchronous inference response') from None

    @app.middleware('http')
    async def limit_request(request, call_next):
        # Bound the entire body before the multipart parser can spool files.
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_REQUEST_BYTES:
                return JSONResponse({'detail': 'Image too large'}, status_code=413)
        request._body = bytes(body)
        return await call_next(request)

    @app.get('/health')
    def health():
        return {'configured': bool(app.state.endpoint), 'mode': app.state.mode}

    @app.post('/v1/inference/warmup')
    def warmup(request_id: str = Form(...)):
        try:
            request_id = str(UUID(request_id))
        except ValueError:
            raise HTTPException(422, 'Invalid request_id') from None
        if app.state.mode == 'sync':
            return {'request_id': request_id, 'status': 'ready'}
        return job_response(lambda: (get_jobs().warmup(request_id), 202))

    @app.get('/v1/inference/{job_id}')
    def poll(job_id: str):
        return job_response(lambda: get_jobs().get(job_id))

    @app.delete('/v1/inference/{job_id}')
    def cancel(job_id: str):
        return job_response(lambda: (get_jobs().cancel(job_id), 200))

    @app.post('/v1/inference', response_model=InferenceResponse)
    def inference(image: UploadFile = File(...), request_id: str | None = Form(None),
                  timestamp: str | None = Form(None, max_length=64),
                  device_id: str | None = Form(None, max_length=128)):
        started = time.perf_counter()
        try:
            request_id = str(UUID(request_id)) if request_id else str(uuid4())
        except ValueError:
            raise HTTPException(422, 'Invalid request_id') from None
        try:
            data = image.file.read(MAX_JPEG_BYTES + 1)
        finally:
            image.file.close()
        if len(data) > MAX_JPEG_BYTES:
            raise HTTPException(413, 'Image too large')
        if not data or image.content_type != 'image/jpeg':
            raise HTTPException(415, 'A JPEG image is required')
        try:
            with Image.open(io.BytesIO(data)) as decoded:
                width, height = decoded.size
                if decoded.format != 'JPEG' or width > 1280 or height > 1280:
                    raise HTTPException(422, 'Resize image before upload (maximum 1280x1280)')
                decoded.verify()
            with Image.open(io.BytesIO(data)) as decoded:
                width, height = ImageOps.exif_transpose(decoded).size
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(422, 'Invalid JPEG') from None
        if not app.state.endpoint:
            raise HTTPException(503, 'Inference service is not configured')
        if app.state.mode == 'async':
            return job_response(lambda: (get_jobs().submit(data, request_id, (width, height)), 202))
        if app.state.runtime is None:
            # Default credential provider chain uses the Lambda/backend IAM role.
            app.state.runtime = boto3.client('sagemaker-runtime', config=Config(
                connect_timeout=2, read_timeout=8, retries={'total_max_attempts': 1}))
        status = 'error'
        try:
            raw = app.state.runtime.invoke_endpoint(EndpointName=app.state.endpoint,
                ContentType='image/jpeg', Accept='application/json', Body=data,
                CustomAttributes=f'request_id={request_id}')
            stream = raw['Body']
            try:
                payload = stream.read(1024 * 1024 + 1)
            finally:
                stream.close()
            if len(payload) > 1024 * 1024:
                raise ValueError('Oversized model response')
            body = json.loads(payload)
            if not isinstance(body, dict):
                raise ValueError('Invalid model response')
            result = InferenceResponse.model_validate({**body, 'request_id': request_id})
            if (result.image_width, result.image_height) != (width, height):
                raise ValueError('Model image dimensions mismatch')
            status = 'success'
            return result
        except ReadTimeoutError:
            raise HTTPException(504, 'Inference timed out') from None
        except (ClientError, BotoCoreError):
            raise HTTPException(503, 'SageMaker is unavailable') from None
        except (ValueError, TypeError, KeyError, ValidationError):
            raise HTTPException(502, 'Invalid inference response') from None
        finally:
            # Do not log image data, credentials, caller identity or request body.
            log.info(json.dumps({'request_id': request_id, 'status': status,
                'backend_total_ms': round((time.perf_counter() - started) * 1000, 2)}))

    @app.post('/v2/inference')
    def inference_depth(image: UploadFile = File(...), request_id: str = Form(...),
                        frame_id: str = Form(...), timestamp: str = Form(...), camera_id: str = Form(...)):
        meta = dict(request_id=request_id, frame_id=frame_id, timestamp=timestamp, camera_id=camera_id)
        try:
            validate_metadata(meta)
        except (ValueError, TypeError, KeyError):
            raise HTTPException(422, 'Invalid frame metadata') from None
        try:
            data = image.file.read(MAX_JPEG_BYTES + 1)
        finally:
            image.file.close()
        if len(data) > MAX_JPEG_BYTES:
            raise HTTPException(413, 'Image too large')
        if image.content_type != 'image/jpeg' or not data:
            raise HTTPException(415, 'A JPEG image is required')
        try:
            with Image.open(io.BytesIO(data)) as decoded:
                if decoded.format != 'JPEG' or max(decoded.size) > 1280:
                    raise ValueError('Invalid image dimensions')
                oriented = ImageOps.exif_transpose(decoded).convert('RGB')
                size = oriented.size
                import hashlib
                pixel_sha = hashlib.sha256(oriented.tobytes()).hexdigest()
        except (ValueError, OSError, Image.DecompressionBombError):
            raise HTTPException(422, 'Invalid JPEG') from None
        if not app.state.endpoint:
            raise HTTPException(503, 'Inference service is not configured')
        # Include the decoded RGB digest in the job expectation, but not in the
        # transport metadata consumed by FrameIdentity.create.
        if app.state.mode == 'async':
            return job_response(lambda: (get_jobs().submit(data, request_id, size, frame_metadata=meta), 202))
        if app.state.runtime is None:
            app.state.runtime = boto3.client('sagemaker-runtime', config=Config(
                connect_timeout=2, read_timeout=8, retries={'total_max_attempts': 1}))
        try:
            raw = app.state.runtime.invoke_endpoint(EndpointName=app.state.endpoint,
                ContentType='image/jpeg', Accept='application/json', Body=data,
                CustomAttributes=frame_attributes(meta))
            stream = raw['Body']
            try:
                payload = stream.read(1024 * 1024 + 1)
            finally:
                stream.close()
            if len(payload) > 1024 * 1024:
                raise ValueError('Oversized response')
            result = validate_response(json.loads(payload), meta, size)
            if result['pixel_sha256'] != pixel_sha:
                raise ValueError('Processed image differs from uploaded frame')
            return JSONResponse(result, headers={'Cache-Control': 'no-store'})
        except ReadTimeoutError:
            raise HTTPException(504, 'Inference timed out') from None
        except (ClientError, BotoCoreError):
            raise HTTPException(503, 'SageMaker is unavailable') from None
        except (ValueError, TypeError, KeyError):
            raise HTTPException(502, 'Invalid Detection + Depth response') from None

    return app


app = create_app()
handler = Mangum(app, lifespan='off')
