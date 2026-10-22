import io
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from download_campus_access import decode_member, parse_annotations, select_images


def test_cvat_preserves_exact_class_and_source_attributes():
    xml = b'''<annotations><image name="images/a.jpg" width="100" height="50" task_id="7" id="1">
      <box label="step" xtl="0" ytl="10" xbr="100" ybr="20"><attribute name="type">curb</attribute></box>
      <box label="grab_bar" xtl="10" ytl="0" xbr="90" ybr="10"/>
      <box label="ramp" xtl="30" ytl="10" xbr="80" ybr="40"/>
      <box label="stair" xtl="0" ytl="0" xbr="20" ybr="50"/>
    </image></annotations>'''
    rows, errors = parse_annotations(xml)
    assert not errors
    assert [b[0] for b in rows[0]['boxes']] == [12, 21, 11, 5]
    assert rows[0]['boxes'][0] == pytest.approx([12, .5, .3, 1., .2])
    assert rows[0]['source_annotations'][0]['attributes'] == {'type': 'curb'}
    assert rows[0]['task_id'] == '7'


def test_invalid_annotation_quarantines_image_without_clipping():
    rows, errors = parse_annotations(b'''<annotations><image name="images/a.jpg" width="100" height="50">
      <box label="ramp" xtl="-1" ytl="0" xbr="20" ybr="20"/>
    </image></annotations>''')
    assert not rows and len(errors) == 1
    with pytest.raises(ValueError):
        parse_annotations(b'<!DOCTYPE unsafe><annotations/>')


def test_selection_counts_multiclass_image_once_and_is_reproducible():
    rows = [{'name':str(i), 'boxes':[[11, .5, .5, .2, .2],[21, .5, .5, .2, .2]]} for i in range(5)]
    selected = select_images(rows, {11:3, 21:3})
    assert len(selected) == 3
    assert selected == select_images(list(reversed(rows)), {11:3, 21:3})


@pytest.mark.parametrize('compression', [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_selective_zip_import_verifies_crc_and_exact_size(compression):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=compression) as archive:
        archive.writestr('images/a.jpg', b'example image' * 100)
    with zipfile.ZipFile(buffer) as archive:
        info = archive.getinfo('images/a.jpg')
        member = {key:getattr(info,key) for key in ('filename','header_offset','file_size','compress_size','compress_type','CRC')}
    assert decode_member(buffer.getvalue(), member) == b'example image' * 100
    with pytest.raises(ValueError):
        decode_member(buffer.getvalue(), dict(member, CRC=1))
    with pytest.raises(ValueError):
        decode_member(buffer.getvalue(), dict(member, file_size=1))
