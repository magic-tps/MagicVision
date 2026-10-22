import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from download_campus_glass import mask_boxes


def test_glass_door_exact_color_not_glass_wall_or_window():
    mask=np.zeros((30,40,3),dtype=np.uint8)
    mask[2:12,4:14]=[120,120,120]
    mask[15:25,20:30]=[224,5,255]
    mask[0,0]=[120,120,120]
    boxes,ignored=mask_boxes(Image.fromarray(mask))
    assert len(boxes)==1 and boxes[0]==pytest.approx([14,9/40,7/30,10/40,10/30])
    assert ignored==1
