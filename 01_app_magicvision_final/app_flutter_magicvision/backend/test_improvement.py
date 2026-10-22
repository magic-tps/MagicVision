import io
from uuid import uuid4
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from pydantic import ValidationError
from api import create_app
from schema import LABELS, Detection
from test_api import Runtime, jpeg


def test_campus22_contract_preserves_ids_and_explicit_v1_alias():
    assert len(LABELS)==22 and LABELS[9]=='obstacle' and LABELS[10]=='trash_can'
    assert Detection(class_id=10,label='trash_bin',confidence=.2,bbox={'left':.1,'top':.2,'right':.6,'bottom':.8}).label=='trash_bin'
    for index,name in enumerate(LABELS):
        d=Detection(class_id=index,label=name,confidence=.2,bbox={'left':.1,'top':.2,'right':.6,'bottom':.8})
        assert d.class_id==index
    with pytest.raises(ValidationError):
        Detection(class_id=9,label='trash_bin',confidence=.2,bbox={'left':.1,'top':.2,'right':.6,'bottom':.8})


@pytest.mark.parametrize('size',[(540,960),(960,540),(960,960),(1280,1280)])
def test_higher_resolution_roundtrip(size):
    runtime=Runtime({'processing_time_ms':1,'image_width':size[0],'image_height':size[1],
        'model_version':'candidate-hash','detections':[]})
    result=TestClient(create_app(runtime,'test')).post('/v1/inference',data={'request_id':str(uuid4())},files={'image':('frame.jpg',jpeg(size),'image/jpeg')})
    assert result.status_code==200 and result.json()['model_version']=='candidate-hash'


def test_exif_orientation_matches_cloud_coordinate_reference():
    image=Image.new('RGB',(960,540));exif=Image.Exif();exif[274]=6
    b=io.BytesIO();image.save(b,'JPEG',exif=exif)
    runtime=Runtime({'processing_time_ms':1,'image_width':540,'image_height':960,'detections':[]})
    result=TestClient(create_app(runtime,'test')).post('/v1/inference',files={'image':('frame.jpg',b.getvalue(),'image/jpeg')})
    assert result.status_code==200


def test_resource_limits_remain_bounded():
    runtime=Runtime();client=TestClient(create_app(runtime,'test'))
    for size in [(1281,10),(10,1281)]:
        assert client.post('/v1/inference',files={'image':('frame.jpg',jpeg(size),'image/jpeg')}).status_code==422
    assert not runtime.calls
