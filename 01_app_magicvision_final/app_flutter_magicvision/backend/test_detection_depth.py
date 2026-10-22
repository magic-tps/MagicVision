"""Synthetic fixtures test math/contracts, never counted as campus data."""
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import uuid4
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'sagemaker'))
from detection_depth import (NAMES, POLICY, FrameIdentity, DepthResult, DepthEstimator,
    DetectionDepthFusion, TemporalTracker, RiskEngine, AlertManager, DetectionDepthPipeline, validate_depth)
from depth_schema import validate_response, frame_attributes
from api import create_app
from test_api import Runtime


def frame():
    image=Image.new('RGB',(100,80))
    identity=FrameIdentity.create(image,str(uuid4()),str(uuid4()),'2026-09-22T12:00:00Z','test-camera')
    return image,identity


def obj(cid=14,distance=None):
    return {'class_id':cid,'class_name':NAMES[cid],'label':NAMES[cid],'confidence':.9,
        'bbox':{'left':.2,'top':.1,'right':.8,'bottom':.95},
        'distance_m':distance,'distance_smoothed_m':distance,'approaching':False}


def test_contract_all_consumers_and_active_mapping():
    from campus_contract import LABELS
    root=HERE.parents[2];ml=root/'magicvision_ml_improvement'
    active=json.loads((ml/'configs/active_dataset.json').read_text())
    mapping=json.loads((root/active['class_mapping']).read_text())
    assert NAMES==LABELS==mapping['names']==active['names'] and len(NAMES)==22
    dart=(HERE.parent/'lib/core/config/campus_contract.dart').read_text()
    start=dart.index('[');end=dart.index('];')
    dart_names=json.loads(dart[start:end+1].replace(',\n]', '\n]'))
    assert dart_names==NAMES
    assert json.loads((HERE/'sagemaker/detection_depth_policy.json').read_text())==json.loads((ml/'configs/detection_depth_policy.json').read_text())
    assert not {'cable','wet_floor','elevator'}&set(NAMES)


def test_legacy_checkpoint_cannot_masquerade_as_campus22():
    with pytest.raises(ValueError,match='Campus22'):
        DetectionDepthPipeline(SimpleNamespace(names=dict(enumerate(NAMES[:10]))),None,{},'cpu')


@pytest.mark.parametrize('field,value',[('request_id','other'),('frame_id','other'),('timestamp','2026-09-22T12:00:01Z'),('camera_id','usb')])
def test_frame_identity_mismatch(field,value):
    _,identity=frame();depth=DepthResult(replace(identity,**{field:value}),np.ones((80,100)),'a'*64,1)
    with pytest.raises(ValueError,match='mismatch'):validate_depth(depth,identity)


def test_depth_shape_invalid_values_and_no_fabricated_distance():
    _,identity=frame();depth=DepthResult(identity,np.full((80,100),np.nan),'a'*64,1)
    result,diagnostics=DetectionDepthFusion().fuse([obj()],depth,identity)
    assert result[0]['distance_m'] is None and result[0]['distance_raw_m'] is None
    assert diagnostics['nan_pixels']==8000
    for values in [np.ones((40,50)),np.ones((80,100,2))]:
        with pytest.raises(ValueError,match='resolution'):validate_depth(replace(depth,depth_map=values),identity)
    values=np.ones((80,100));values[0,:5]=[np.nan,np.inf,-1,0,999]
    _,valid,diag=validate_depth(replace(depth,depth_map=values),identity)
    assert diag['invalid_pixels']==5 and not valid[0,:5].any()


def test_stairs_lower_roi_robust_quantile_and_camera_calibration():
    _,identity=frame();values=np.full((80,100),5.0);values[62:76,:]=2;values[64,40]=100
    depth=DepthResult(identity,values,'a'*64,1)
    fusion=DetectionDepthFusion()
    results,_=fusion.fuse([obj(5)],depth,identity)
    assert results[0]['distance_raw_m']==2 and results[0]['distance_m'] is None
    profile={'validated':True,'camera_id':identity.camera_id,'model_sha256':'a'*64,
        'policy_version':POLICY['version'],'image_width':100,'image_height':80,'log_scale':1.,'log_bias':0.}
    fusion=DetectionDepthFusion(calibration=profile)
    calibrated,_=fusion.fuse([obj(5)],depth,identity)
    assert calibrated[0]['distance_m']==2
    profile['camera_id']='another-lens'
    assert fusion.fuse([obj(5)],depth,identity)[0][0]['distance_m'] is None


@pytest.mark.parametrize('bbox',[[-.1,0,.5,.5],[0,0,2,1],[.5,0,.5,1],[0,0,float('nan'),1]])
def test_invalid_boxes_rejected(bbox):
    _,identity=frame();d=obj();d['bbox']=bbox
    with pytest.raises(ValueError):DetectionDepthFusion().fuse([d],DepthResult(identity,np.ones((80,100)),'a'*64,1),identity)


@pytest.mark.parametrize('cid',[5,6,11,12,13,14,18])
def test_critical_classes_distance_and_friendly_speech(cid):
    risk=RiskEngine();near=risk.assess(obj(cid,.92));far=risk.assess(obj(cid,1.83))
    assert near['risk']=='critical' and 'menos de un metro' in near['message']
    assert far['risk']=='high' and 'aproximadamente 2 metros' in far['message']


def test_context_and_direction_and_priority():
    risk=RiskEngine();door=risk.assess(obj(14,1.2));ext=obj(21,.8);ext['bbox']=[0,.1,.2,.9]
    ext=risk.assess(ext)
    assert ext['risk']=='low' and ext['direction']=='left' and door['hazard_score']>ext['hazard_score']
    assert all(risk.assess(obj(c))['class_priority'] for c in range(22))


def test_tracking_stability_approach_smoothing_expiry_and_invalid_distance():
    tracker=TemporalTracker()
    for index,distance in enumerate([3.8,3.2,2.6,2.1]):
        result=tracker.update([obj(5,distance)],index*1000)[0]
        assert result['track_id']==1
    assert result['approaching'] and result['distance_smoothed_m']==pytest.approx(2.9)
    assert tracker.update([obj(5,None)],4000)[0]['distance_smoothed_m'] is None
    assert tracker.update([obj(5,2)],8001)[0]['track_id']==2
    with pytest.raises(ValueError):tracker.update([],8001)
    policy=copy.deepcopy(POLICY);policy['tracking']['smoothing']='ema';ema=TemporalTracker(policy)
    ema.update([obj(5,2)],0)
    assert ema.update([obj(5,4)],1000)[0]['distance_smoothed_m']==pytest.approx(2.7)


def test_cooldown_escalation_and_only_one_critical():
    manager=AlertManager();risk=RiskEngine()
    def sample(cid,distance,track):return risk.assess({**obj(cid,distance),'track_id':track})
    assert len(manager.select([sample(5,3,1)],0))==1
    assert manager.select([sample(5,2.9,1)],1000)==[]
    assert len(manager.select([sample(5,1.8,1)],2000))==1
    selected=manager.select([sample(5,.8,1),sample(14,.7,2),sample(21,.6,3)],3000)
    assert len(selected)==2 and sum(x['risk']=='critical' for x in selected)==1


class FakeDepth(DepthEstimator):
    def estimate(self,image,identity):return DepthResult(identity,np.full((image.height,image.width),2.),'a'*64,1.)


class FakeDetector:
    names=dict(enumerate(NAMES))
    def predict(self,image,**kwargs):
        box=SimpleNamespace(cls=np.array([14]),conf=np.array([.9]),xyxyn=np.array([[.2,.1,.8,.95]]))
        return [SimpleNamespace(boxes=[box])]


def pipeline():return DetectionDepthPipeline(FakeDetector(),FakeDepth(),{'imgsz':640,'confidence':.3},'cpu')


def test_full_flask_to_lambda_v2_and_legacy_route():
    from serve import create_app as server
    image,identity=frame();buffer=io.BytesIO();image.save(buffer,'JPEG');jpeg=buffer.getvalue()
    meta={k:getattr(identity,k) for k in ('request_id','frame_id','timestamp','camera_id')}
    client=server(FakeDetector(),NAMES,{'imgsz':640,'confidence':.3,'iou':.5,'max_det':100},'cpu',pipeline()).test_client()
    response=client.post('/invocations',data=jpeg,content_type='image/jpeg',headers={'X-Amzn-SageMaker-Custom-Attributes':frame_attributes(meta)})
    assert response.status_code==200
    body=response.json;assert body['detections'][0]['distance_m'] is None
    runtime=Runtime(body);api=TestClient(create_app(runtime,'fixture'))
    result=api.post('/v2/inference',data=meta,files={'image':('frame.jpg',jpeg,'image/jpeg')})
    assert result.status_code==200 and result.json()['frame_id']==identity.frame_id
    body['frame_id']=str(uuid4())
    assert api.post('/v2/inference',data=meta,files={'image':('frame.jpg',jpeg,'image/jpeg')}).status_code==502
    assert client.post('/invocations',data=jpeg,content_type='image/jpeg').status_code==200


def test_pipeline_empty_and_wrong_input_pixels():
    image,identity=frame();p=pipeline();p.detector.predict=lambda *a,**k:[SimpleNamespace(boxes=[])]
    result=p.analyze(image,identity)
    assert result['detections']==[] and result['performance']['total_ms']>=0
    with pytest.raises(ValueError):p.analyze(Image.new('RGB',image.size,'red'),identity)


@pytest.mark.parametrize('wrong_frame',[False,True])
def test_async_v2_preserves_frame_and_rejects_mismatched_result(wrong_frame):
    from async_jobs import AsyncJobs
    from testing.transport import MemoryS3, TestAsyncRuntime
    image,identity=frame();buffer=io.BytesIO();image.save(buffer,'JPEG');data=buffer.getvalue()
    meta={k:getattr(identity,k) for k in ('request_id','frame_id','timestamp','camera_id')}
    s3=MemoryS3();runtime=TestAsyncRuntime(s3);jobs=AsyncJobs(s3,runtime,'private-test-bucket','test-endpoint')
    client=TestClient(create_app(endpoint='test-endpoint',jobs=jobs,mode='async'))
    ticket=client.post('/v2/inference',data=meta,files={'image':('frame.jpg',data,'image/jpeg')})
    assert ticket.status_code==202
    job=jobs.read(jobs.key(ticket.json()['job_id']))
    result=pipeline().analyze(image,identity)
    if wrong_frame:result['pixel_sha256']='0'*64
    runtime.complete({'OutputLocation':'s3://private-test-bucket/'+job['output_key']},result)
    response=client.get('/v1/inference/'+ticket.json()['job_id'])
    assert response.status_code==(502 if wrong_frame else 200)
