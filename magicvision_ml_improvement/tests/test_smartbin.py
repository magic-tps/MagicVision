import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from download_smartbin import map_labels


def test_remaps_only_real_bin_annotations(tmp_path):
    label=tmp_path/'bin.txt';label.write_text('0 0.5 0.4 0.2 0.3\n')
    assert map_labels(label)==[[10,.5,.4,.2,.3]]
    assert label.read_text().startswith('0 ')


@pytest.mark.parametrize('text',['','1 .5 .5 .2 .2\n','0 .1 .1 .8 .8\n'])
def test_background_other_classes_or_invalid_geometry_refused(tmp_path,text):
    label=tmp_path/'bin.txt';label.write_text(text)
    with pytest.raises(ValueError):map_labels(label)
