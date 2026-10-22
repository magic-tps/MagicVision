import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from http_range_zip import RangeReader


def fake_response(content,status=206,range_header='bytes 0-3/10'):
    class Response:
        status_code=status
        headers={'Content-Range':range_header}
        def close(self):pass
    result=Response();result.content=content
    return result


def test_seek_across_blocks_uses_cache_and_stops_at_eof(monkeypatch):
    reader=RangeReader('https://example.org/archive.zip',10,block_size=4)
    payload=b'0123456789';calls=[]
    def get(url,headers,**kwargs):
        start,end=map(int,headers['Range'][6:].split('-'));calls.append((start,end))
        return fake_response(payload[start:end+1],range_header=f'bytes {start}-{end}/10')
    monkeypatch.setattr(reader.session,'get',get)
    reader.seek(3);assert reader.read(4)==b'3456'
    reader.seek(-2,2);assert reader.read(8)==b'89'
    assert reader.read(1)==b''
    reader.seek(0);assert reader.read()==payload
    assert calls==[(0,3),(4,7),(8,9)]
    assert reader.bytes_downloaded==10
    reader.close()


@pytest.mark.parametrize('response',[fake_response(b'0123',200),fake_response(b'0123',range_header='bytes 1-4/10'),fake_response(b'01')])
def test_refuses_ignored_or_incomplete_ranges(monkeypatch,response):
    reader=RangeReader('https://example.org/archive.zip',10,block_size=4)
    monkeypatch.setattr(reader.session,'get',lambda *args,**kwargs:response)
    with pytest.raises(ValueError):reader.read(3)
    reader.close()
