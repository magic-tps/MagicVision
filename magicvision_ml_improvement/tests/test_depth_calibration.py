"""Numerical fixtures only: these do not validate any real camera."""
import csv
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from calibrate_depth import calibrate, metrics


def measurements(path,leak=False):
    keys=['class','predicted_m','real_m','scenario','camera_id','image_width','image_height','session_id','frame_id','split','measurement_evidence']
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for split in ('calibration','validation'):
            for i,d in enumerate((1,2,4)):
                writer.writerow(dict(zip(keys,['stairs',d*2,d,'unit fixture','fixture-camera',640,480,
                    'same' if leak else split,split+str(i),split,'SYNTHETIC UNIT TEST ONLY'])))


def test_fit_reports_heldout_error_but_never_auto_approves_camera(tmp_path):
    path=tmp_path/'measured.csv';measurements(path)
    profile,report=calibrate(path,'fixture-camera','a'*64,640,480)
    assert not profile['validated'] and report['after']['MAE']<1e-10
    assert report['before']['MAE']>2 and report['by_class']['door']['n']==0


def test_calibration_rejects_leakage_empty_and_wrong_camera(tmp_path):
    path=tmp_path/'measured.csv';measurements(path,leak=True)
    with pytest.raises(ValueError,match='leakage'):calibrate(path,'fixture-camera','a'*64,640,480)
    measurements(path)
    with pytest.raises(ValueError,match='camera'):calibrate(path,'usb','a'*64,640,480)
    path.write_text('class,predicted_m,real_m\n')
    with pytest.raises(ValueError,match='No real'):calibrate(path,'usb','a'*64,640,480)


def test_error_metrics_are_not_precision():
    result=metrics([1,3],[2,2])
    assert result['MAE']==1 and result['RMSE']==1 and result['bias']==0
    assert result['MAPE_percent']==50 and metrics([],[])['MAE'] is None
