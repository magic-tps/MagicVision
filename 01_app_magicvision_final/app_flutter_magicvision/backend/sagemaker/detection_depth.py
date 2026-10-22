"""Same-frame detection/depth fusion. Metric speech requires a camera calibration.

No training, downloads, global trackers, or cross-client state on import.
"""
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

HERE=Path(__file__).resolve().parent
NAMES=json.loads((HERE/'campus_contract.json').read_text(encoding='utf-8'))['names']
POLICY=json.loads((HERE/'detection_depth_policy.json').read_text(encoding='utf-8'))


@dataclass(frozen=True)
class FrameIdentity:
    request_id: str
    frame_id: str
    timestamp: str
    camera_id: str
    pixel_sha256: str
    width: int
    height: int

    @classmethod
    def create(cls,image,request_id,frame_id,timestamp,camera_id):
        parsed=datetime.fromisoformat(timestamp.replace('Z','+00:00'))
        if parsed.tzinfo is None:raise ValueError('Frame timestamp requires timezone')
        if not all(isinstance(x,str) and 0<len(x)<=128 for x in (request_id,frame_id,camera_id)):
            raise ValueError('Invalid frame identifiers')
        return cls(request_id,frame_id,timestamp,camera_id,hashlib.sha256(image.convert('RGB').tobytes()).hexdigest(),image.width,image.height)


@dataclass
class DepthResult:
    identity: FrameIdentity
    depth_map: np.ndarray
    model_sha256: str
    inference_ms: float


class DepthEstimator(ABC):
    @abstractmethod
    def estimate(self,image,identity):pass


class UltralyticsDepthEstimator(DepthEstimator):
    def __init__(self,weights,device='cpu',imgsz=640):
        from ultralytics import YOLO
        path=Path(weights)
        if not path.is_file():raise FileNotFoundError('Depth checkpoint must be downloaded explicitly')
        self.model=YOLO(str(path),task='depth')
        if self.model.task!='depth':raise ValueError('Expected depth model, not detector')
        self.model_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        self.device=device;self.imgsz=imgsz

    def estimate(self,image,identity):
        if FrameIdentity.create(image,identity.request_id,identity.frame_id,identity.timestamp,identity.camera_id)!=identity:
            raise ValueError('Depth input does not match frame identity')
        started=time.perf_counter()
        result=self.model.predict(image,imgsz=self.imgsz,device=self.device,verbose=False,save=False)[0]
        depth=result.depth
        if depth is None:raise ValueError('Depth model returned no depth map')
        values=depth.data if hasattr(depth,'data') and not isinstance(depth,np.ndarray) else depth
        if hasattr(values,'detach'):values=values.detach().float().cpu().numpy()
        values=np.asarray(values,dtype=np.float32).squeeze()
        return DepthResult(identity,values,self.model_sha256,(time.perf_counter()-started)*1000)


def validate_depth(result,identity,policy=POLICY):
    if result.identity!=identity:raise ValueError('Detector/depth frame/request/timestamp mismatch')
    values=np.asarray(result.depth_map)
    if values.ndim!=2 or values.shape!=(identity.height,identity.width):
        raise ValueError('Depth resolution mismatch; do not resize an unknown crop/letterbox')
    cfg=policy['depth']
    valid=np.isfinite(values)&(values>=cfg['min_value'])&(values<=cfg['max_value'])
    diagnostics={'invalid_pixels':int((~valid).sum()),'total_pixels':int(values.size),
        'valid_fraction':float(valid.mean()),'nan_pixels':int(np.isnan(values).sum()),
        'infinite_pixels':int(np.isinf(values).sum()),'negative_pixels':int((values<0).sum())}
    return values,valid,diagnostics


def bbox_values(bbox):
    values=[float(bbox[k]) for k in ('left','top','right','bottom')] if isinstance(bbox,dict) else list(map(float,bbox))
    if len(values)!=4 or not all(math.isfinite(v) and 0<=v<=1 for v in values) or values[2]<=values[0] or values[3]<=values[1]:
        raise ValueError('Invalid normalized bbox')
    return values


class DetectionDepthFusion:
    def __init__(self,policy=POLICY,calibration=None):self.policy=policy;self.calibration=calibration

    def _metric(self,raw,identity,model_sha):
        profile=self.calibration
        if raw is None or not profile or profile.get('validated') is not True:return None
        expected={'camera_id':identity.camera_id,'model_sha256':model_sha,'policy_version':self.policy['version'],
                  'image_width':identity.width,'image_height':identity.height}
        if any(profile.get(k)!=v for k,v in expected.items()):return None
        a=float(profile['log_scale']);b=float(profile['log_bias'])
        if not all(math.isfinite(v) for v in (a,b)) or a<=0:raise ValueError('Invalid calibration')
        value=math.exp(a*math.log(raw)+b)
        return value if .02<=value<=self.policy['depth']['max_value'] else None

    def fuse(self,detections,depth,identity):
        values,valid,diagnostics=validate_depth(depth,identity,self.policy)
        objects=[];cfg=self.policy['depth']
        for det in detections:
            cid=det['class_id'];name=det.get('class_name',det.get('label'))
            if type(cid)!=int or not 0<=cid<len(NAMES) or name!=NAMES[cid]:raise ValueError('Detector class contract mismatch')
            confidence=float(det['confidence'])
            if not math.isfinite(confidence) or not 0<=confidence<=1:raise ValueError('Invalid confidence')
            left,top,right,bottom=bbox_values(det['bbox']);w=right-left;h=bottom-top
            roi_kind='stairs' if name=='stairs' else 'door' if name in ('door','glass_door') else 'ground' if name in self.policy['ground_classes'] else 'default'
            x1,y1,x2,y2=self.policy['roi'][roi_kind]
            l=max(0,int((left+x1*w)*identity.width));r=min(identity.width,math.ceil((left+x2*w)*identity.width))
            t=max(0,int((top+y1*h)*identity.height));b=min(identity.height,math.ceil((top+y2*h)*identity.height))
            mask=valid[t:b,l:r];samples=values[t:b,l:r][mask]
            raw=None;quantiles=None
            if mask.size and samples.size>=cfg['min_pixels'] and mask.mean()>=cfg['min_valid_fraction']:
                lo,hi=np.percentile(samples,cfg['trim_percentiles']);trimmed=samples[(samples>=lo)&(samples<=hi)]
                if trimmed.size>=cfg['min_pixels']:
                    median=float(np.median(trimmed));mad=float(np.median(np.abs(trimmed-median)))
                    if mad>0:trimmed=trimmed[np.abs(trimmed-median)<=3*1.4826*mad]
                    if trimmed.size>=cfg['min_pixels']:
                        quantiles={'p25':float(np.percentile(trimmed,25)),'p50':float(np.median(trimmed))}
                        raw=float(np.percentile(trimmed,cfg['quantile']))
            metric=self._metric(raw,identity,depth.model_sha256)
            objects.append({**det,'class_name':name,'label':name,'distance_raw_m':raw,'distance_m':metric,
                'distance_smoothed_m':metric,'distance_approx_m':None,'direction':None,'risk':None,'track_id':None,
                'approaching':False,'depth_status':'calibrated' if metric is not None else 'uncalibrated' if raw is not None else 'invalid',
                'depth_roi':[l,t,r,b],'depth_quantiles_raw':quantiles,'transparent_surface_uncertain':name=='glass_door'})
        return objects,diagnostics


def iou(a,b):
    a=bbox_values(a);b=bbox_values(b)
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


class TemporalTracker:
    """One instance per camera/client session; never shared across AWS callers."""
    def __init__(self,policy=POLICY):self.cfg=policy['tracking'];self.tracks={};self.next_id=1;self.last_time=None
    def update(self,objects,timestamp_ms):
        if self.last_time is not None and timestamp_ms<=self.last_time:raise ValueError('Out-of-order or repeated frame')
        self.last_time=timestamp_ms
        self.tracks={k:v for k,v in self.tracks.items() if timestamp_ms-v['last_seen']<=self.cfg['max_gap_ms']}
        pairs=sorted([(iou(o['bbox'],t['bbox_history'][-1]),i,k) for i,o in enumerate(objects) for k,t in self.tracks.items()
                      if o['class_id']==t['class_id']],reverse=True)
        assigned={};used=set()
        for score,i,k in pairs:
            if score>=self.cfg['iou_threshold'] and i not in assigned and k not in used:assigned[i]=k;used.add(k)
        for i,obj in enumerate(objects):
            key=assigned.get(i)
            if key is None:
                key=self.next_id;self.next_id+=1
                self.tracks[key]={'class_id':obj['class_id'],'first_seen':timestamp_ms,'last_alert_time':None,
                    'bbox_history':deque(maxlen=self.cfg['history_size']),'distance_history':deque(maxlen=self.cfg['history_size']),
                    'confidence_history':deque(maxlen=self.cfg['history_size']),'ema':None}
            track=self.tracks[key];track['last_seen']=timestamp_ms;track['bbox_history'].append(obj['bbox']);track['confidence_history'].append(obj['confidence'])
            distance=obj.get('distance_m');obj['track_id']=key;obj['approaching']=False
            if distance is None:
                track['distance_history'].clear();track['ema']=None;obj['distance_smoothed_m']=None
                continue
            track['distance_history'].append((timestamp_ms,distance));history=list(track['distance_history'])
            alpha=self.cfg['ema_alpha'];track['ema']=distance if track['ema'] is None else alpha*distance+(1-alpha)*track['ema']
            obj['distance_smoothed_m']=float(np.median([v for _,v in history])) if self.cfg['smoothing']=='rolling_median' else track['ema']
            obj['approaching']=len(history)>=self.cfg['approaching_min_samples'] and history[-1][0]-history[0][0]>=self.cfg['approaching_min_span_ms'] and history[0][1]-history[-1][1]>=self.cfg['approaching_drop_m']
        return objects


class RiskEngine:
    def __init__(self,policy=POLICY):self.policy=policy;self.priority={name:p for p,names in policy['priorities'].items() for name in names}
    def assess(self,obj):
        left,top,right,bottom=bbox_values(obj['bbox']);center=(left+right)/2;area=(right-left)*(bottom-top)
        d=self.policy['direction'];direction='left' if center<d['left_max'] else 'right' if center>d['right_min'] else 'center'
        name=obj['class_name'];priority=self.priority[name];distance=obj.get('distance_smoothed_m',obj.get('distance_m'))
        in_path=direction=='center' and bottom>=self.policy['geometry']['bottom_min']
        risk='low'
        if priority!='contextual':
            if distance is not None:
                risk=next((level for level in ('critical','high','medium') if distance<self.policy['distance_thresholds_m'][level]),'low')
                if direction!='center' and risk=='critical':risk='high'
            elif name=='stairs' or (in_path and priority in ('critical','high')):risk='high'
            elif priority=='critical' or area>=self.policy['geometry']['near_area']:risk='medium'
            if name in ('door','glass_door') and direction=='center' and area>=self.policy['geometry']['large_area']:
                if risk in ('low','medium'):risk='high'
        approx=None if distance is None else 0 if distance<1 else int(math.floor(distance+.5))
        score={'low':0,'medium':100,'high':200,'critical':300}[risk]+{'contextual':0,'medium':10,'high':20,'critical':30}[priority]
        score+=20 if direction=='center' else 0;score+=10 if obj.get('approaching') else 0
        score+=obj['confidence']+(0 if distance is None else 1/(1+distance))
        text=self.policy['names_es'][name]
        if approx==0:text+=' a menos de un metro'
        elif approx is not None:text+=f' a aproximadamente {approx} '+('metro' if approx==1 else 'metros')
        text+=' '+{'center':'al frente','left':'a la izquierda','right':'a la derecha'}[direction]+'.'
        if risk in ('high','critical'):text='Cuidado. '+text
        return {**obj,'direction':direction,'in_path':in_path,'trajectory_basis':'image_corridor_proxy',
            'risk':risk,'class_priority':priority,'distance_approx_m':approx,'hazard_score':score,'message':text}


class AlertManager:
    def __init__(self,policy=POLICY):self.cfg=policy['alerts'];self.alerts={}
    def select(self,objects,now_ms):
        self.alerts={k:v for k,v in self.alerts.items() if now_ms-v['time']<max(60000,self.cfg['cooldown_ms']*4)}
        chosen=[];critical=0
        for obj in sorted(objects,key=lambda x:x['hazard_score'],reverse=True):
            key=(obj.get('track_id'),obj['class_id']);last=self.alerts.get(key);distance=obj.get('distance_smoothed_m')
            rank={'low':0,'medium':1,'high':2,'critical':3}[obj['risk']]
            changed=last is None or rank>last['rank'] or (distance is not None and last['distance'] is not None and last['distance']-distance>=self.cfg['distance_change_m'])
            if last and not changed and now_ms-last['time']<self.cfg['cooldown_ms']:continue
            if obj['risk']=='critical' and critical:continue
            chosen.append(obj);critical+=obj['risk']=='critical';self.alerts[key]={'time':now_ms,'rank':rank,'distance':distance}
            if len(chosen)>=self.cfg['max_per_cycle']:break
        return chosen


class DetectionDepthPipeline:
    def __init__(self,detector,depth,config,device,calibration=None):
        self.detector=detector;self.depth=depth;self.config=config;self.device=device
        self.fusion=DetectionDepthFusion(calibration=calibration);self.risk=RiskEngine()
        if [detector.names[i] for i in range(len(detector.names))]!=NAMES:raise ValueError('Campus22 checkpoint required; do not relabel legacy weights')
    def analyze(self,image,identity):
        if FrameIdentity.create(image,identity.request_id,identity.frame_id,identity.timestamp,identity.camera_id)!=identity:raise ValueError('Detector input/frame mismatch')
        start=time.perf_counter()
        result=self.detector.predict(image,imgsz=self.config['imgsz'],conf=self.config['confidence'],
            iou=self.config.get('iou',.7),max_det=self.config.get('max_det',300),device=self.device,verbose=False,save=False)[0]
        detections=[{'class_id':int(b.cls[0]),'class_name':NAMES[int(b.cls[0])],'confidence':float(b.conf[0]),
                     'bbox':dict(zip(('left','top','right','bottom'),map(float,b.xyxyn[0].tolist())))} for b in result.boxes]
        detection_ms=(time.perf_counter()-start)*1000
        depth=self.depth.estimate(image,identity);fusion_start=time.perf_counter()
        objects,diagnostics=self.fusion.fuse(detections,depth,identity)
        objects=[self.risk.assess(o) for o in objects]
        fusion_ms=(time.perf_counter()-fusion_start)*1000
        return {'schema_version':2,'taxonomy':'magicvision-campus22-v1','request_id':identity.request_id,'frame_id':identity.frame_id,
            'timestamp':identity.timestamp,'camera_id':identity.camera_id,'pixel_sha256':identity.pixel_sha256,
            'image_width':image.width,'image_height':image.height,'detections':objects,'depth_diagnostics':diagnostics,
            'performance':{'detection_ms':detection_ms,'depth_ms':depth.inference_ms,'fusion_ms':fusion_ms,'total_ms':(time.perf_counter()-start)*1000},
            'depth_model_sha256':depth.model_sha256,'tracking_location':'client_per_camera_session'}
