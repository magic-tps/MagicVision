"""Conversion tests guarding against semantically wrong or invented annotations."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from download_campus_hazards import auto_boxes, fire_boxes, mask_boxes, voc_boxes


def xml(name='Fire_Extinguisher',right=80):
    return f'''<annotation><filename>a.jpg</filename><size><width>100</width><height>200</height></size>
    <object><name>{name}</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>{right}</xmax><ymax>160</ymax></bndbox></object></annotation>'''.encode()


def test_voc_extinguisher_id_and_normalization():
    name,width,height,boxes=fire_boxes(xml())
    assert (name,width,height)==('a.jpg',100,200)
    assert boxes[0]==pytest.approx([24,.45,.45,.7,.7])


def test_fire_sign_is_not_extinguisher():
    assert fire_boxes(xml('Fire_Suppression_Signage'))[3]==[]


def test_cones_named_voc_only():
    assert voc_boxes(xml('trafficcone'),{'trafficcone':23})[3][0][0]==23
    assert voc_boxes(xml('Fire_Extinguisher'),{'trafficcone':23})[3]==[]


def test_out_of_bounds_is_not_silently_clipped():
    with pytest.raises(ValueError,match='outside'):
        fire_boxes(xml(right=102))


def test_xml_external_declarations_rejected():
    with pytest.raises(ValueError,match='XML'):
        fire_boxes(b'<!DOCTYPE foo>'+xml())


def test_permanent_barrier_is_not_construction_barrier():
    data={'bbox':[[10,20,40,50],[20,30,10,20]],
          'categories':['Road_Barriers','Temporary_Construction_Barriers']}
    assert auto_boxes(data,100,100)==[pytest.approx([20,.25,.4,.1,.2])]


def test_autoexpert_bad_parallel_arrays_rejected():
    with pytest.raises(ValueError,match='count'):
        auto_boxes({'bbox':[[0,0,10,10]],'categories':[]},100,100)


def test_mask_extent_uses_pixel_edges():
    array=np.zeros((20,40),dtype=np.uint8);array[3:8,5:15]=255
    boxes=mask_boxes(Image.fromarray(array))
    assert boxes[0]==pytest.approx([18,10/40,5.5/20,10/40,5/20])


def test_empty_mask_does_not_create_negative_or_fake_box():
    assert mask_boxes(Image.new('L',(40,20),0))==[]
