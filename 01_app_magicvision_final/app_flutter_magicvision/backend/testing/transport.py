"""Test-only AWS transport; production API never imports this module."""
import io
import json
import threading
from urllib.request import Request, urlopen

from botocore.exceptions import ClientError


class MemoryS3:
    def __init__(self):
        self.objects = {}
        self.lock = threading.RLock()

    def put_object(self, Bucket, Key, Body=b'', **kwargs):
        with self.lock:
            self.objects[Key] = bytes(Body)
        return {}

    def get_object(self, Bucket, Key):
        with self.lock:
            if Key not in self.objects:
                raise ClientError({'Error': {'Code': 'NoSuchKey'}}, 'GetObject')
            return {'Body': io.BytesIO(self.objects[Key])}

    def delete_object(self, Bucket, Key):
        with self.lock:
            self.objects.pop(Key, None)
        return {}


class TestAsyncRuntime:
    __test__ = False

    def __init__(self, s3, model_url=None):
        self.s3 = s3
        self.model_url = model_url
        self.calls = []

    def invoke_endpoint_async(self, **kwargs):
        self.calls.append(kwargs)
        job = kwargs['InferenceId']
        bucket = kwargs['InputLocation'].split('/')[2]
        reply = {'InferenceId': job,
            'OutputLocation': f's3://{bucket}/async-output/app/{job}.out',
            'FailureLocation': f's3://{bucket}/async-failure/app/{job}.out'}
        if self.model_url:
            threading.Thread(target=self._invoke_model, args=(kwargs, reply), daemon=True).start()
        return reply

    def complete(self, reply, body, failure=False):
        location = reply['FailureLocation' if failure else 'OutputLocation']
        _, _, bucket, key = location.split('/', 3)
        self.s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(body).encode())

    def _invoke_model(self, request, reply):
        try:
            _, _, bucket, key = request['InputLocation'].split('/', 3)
            image = self.s3.get_object(Bucket=bucket, Key=key)['Body'].read()
            call = Request(self.model_url + '/invocations', data=image, headers={
                'Content-Type': 'image/jpeg',
                'X-Amzn-SageMaker-Custom-Attributes': request['CustomAttributes']})
            with urlopen(call, timeout=60) as result:
                self.complete(reply, json.load(result))
        except Exception as error:
            self.complete(reply, {'error_type': type(error).__name__}, failure=True)

