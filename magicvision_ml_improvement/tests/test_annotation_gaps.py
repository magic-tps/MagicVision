import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from find_annotation_gaps import possible_gaps,select_diverse
from data_pipeline import write_labels


def test_existing_boxes_do_not_become_proposals_but_wrong_class_does():
    box=[.1,.1,.8,.8]
    truth=[{'class_id':10,'bbox':box}]
    pred=[{'class_id':10,'bbox':box,'confidence':.7},{'class_id':9,'bbox':box,'confidence':.8}]
    suggestions=possible_gaps(pred,truth)
    assert len(suggestions)==1
    assert suggestions[0]['reason']=='overlapping_different_class'
    assert suggestions[0]['overlapping_classes']==['trash_bin']
    assert suggestions[0]['status']=='unverified_model_proposal'


def test_missing_predictions_never_prove_complete_labels():
    assert possible_gaps([],[])==[]
    proposals=possible_gaps([{'class_id':0,'bbox':[.1,.1,.8,.8],'confidence':.9}],[])
    assert proposals[0]['reason']=='possible_missing_or_misaligned_box'


def test_priority_sample_is_distinct_and_includes_rare_class(tmp_path):
    rows=[]
    for i,c in enumerate([0,0,0,7]):
        label=tmp_path/f'{i}.txt';write_labels(label,[[c,.5,.5,.2,.2]])
        rows.append({'sha256':str(i),'label':str(label)})
    rows.append(rows[0])
    selected=select_diverse(rows,3)
    assert len(selected)==3 and len({r['sha256'] for r in selected})==3
    assert rows[3] in selected
