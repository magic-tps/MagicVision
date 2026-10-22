import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from download_campus_v5 import road_boxes,select_road,sample_hazard_runs


def test_roadwork_semantics_exclude_fence_drum_and_generic_vehicle():
    names={1:'Cone',2:'Barricade',3:'Fence',4:'Drum',5:'Work Vehicle',6:'Worker'}
    annotations=[{'category_id':k,'bbox':[10,20,20,20]} for k in names]
    assert [b[0] for b in road_boxes(annotations,names,100,100)]==[23,20,0]
    assert road_boxes([{'category_id':1,'bbox':[0,0,10,10],'iscrowd':1}],names,100,100)==[]
    with pytest.raises(ValueError):road_boxes([{'category_id':1,'bbox':[90,90,20,20]}],names,100,100)


def test_roadwork_quota_prefers_different_sequences():
    items=[{'key':f'{g}_{i}','group':g,'boxes':[[23,.5,.5,.2,.2]]} for g in ('a','b','c') for i in range(12)]
    selected=select_road(items,{23:3})
    assert len(selected)==3 and len({r['group'] for r in selected})==3
    many=select_road(items,{23:100})
    assert len(many)==24


def test_frame_labels_are_candidates_with_spacing_and_run_limits():
    labels={i:(2 if i<1000 else 18) for i in range(2000)}
    environments={i:1 for i in labels}
    rows=sample_hazard_runs(labels,environments,quota=20,stride=150,max_per_run=4)
    assert len(rows)==8
    assert {r[0] for r in rows}=={'cable','wet_floor'}
    for cls in ('cable','wet_floor'):
        frames=sorted(r[1] for r in rows if r[0]==cls)
        assert all(b-a>=150 for a,b in zip(frames,frames[1:]))
    assert sample_hazard_runs({0:0},{0:1})==[]
