from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from download_campus_elevator import license_details, valid_photo
import download_campus_elevator as module
import io
import requests
from PIL import Image


def metadata(license='CC BY-SA 4.0',url='https://creativecommons.org/licenses/by-sa/4.0/'):
    return {'extmetadata':{'LicenseShortName':{'value':license},'LicenseUrl':{'value':url},
        'Artist':{'value':'<a href="/author">Author &amp; Photographer</a>'}}}


def test_preserves_attribution_and_cc_sharealike_license():
    license=license_details(metadata())
    assert license['author']=='Author & Photographer'
    assert license['license']=='CC BY-SA 4.0'


@pytest.mark.parametrize('name,url',[
    ('CC BY-NC 4.0','https://creativecommons.org/licenses/by-nc/4.0/'),
    ('CC BY-SA 4.0','https://example.org/licenses/by-sa/4.0/'),
    ('CC BY 4.0','https://creativecommons.org/licenses/by-sa/4.0/'),
    ('Public domain',''),
])
def test_license_rejects_restrictions_ambiguous_and_mismatched_claims(name,url):
    with pytest.raises(ValueError):
        license_details(metadata(name,url))


def test_photo_filter_rejects_audio_svg_and_untrusted_hosts():
    photo={'mime':'image/jpeg','size':10000,'width':800,'height':600,'url':'https://upload.wikimedia.org/test.jpg'}
    assert valid_photo(photo)
    assert not valid_photo(dict(photo,mime='image/svg+xml'))
    assert not valid_photo(dict(photo,mime='audio/ogg'))
    assert not valid_photo(dict(photo,url='https://untrusted.test/file.jpg'))


def test_thumbnail_is_verified_as_derivative_and_cache_is_hash_bound(tmp_path,monkeypatch):
    buffer=io.BytesIO();Image.new('RGB',(960,480),'blue').save(buffer,format='JPEG')
    calls=[]
    class Response:
        status_code=200
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_content(self,*args):yield buffer.getvalue()
    class Session:
        def get(self,url,**kwargs):calls.append(url);return Response()
    monkeypatch.setattr(module,'session',lambda:Session())
    info={'sha1':'0'*40,'url':'https://upload.wikimedia.org/original.jpg',
          'thumburl':'https://upload.wikimedia.org/thumb/960px-image.jpg',
          'width':4000,'height':2000,'thumbwidth':960,'thumbheight':480}
    target=tmp_path/'thumb.jpg'
    result=module.download_photo(info,target)
    assert result['download_variant']=='standard_thumbnail'
    assert result['download_url']==info['thumburl']
    assert module.download_photo(info,target)==result and len(calls)==1
    target.write_bytes(b'bad')
    module.download_photo(info,target)
    assert len(calls)==2


def test_rate_limit_is_not_retried_in_photo_worker(tmp_path,monkeypatch):
    calls=[]
    class Response:
        status_code=429
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):raise requests.HTTPError('429',response=self)
    class Session:
        def get(self,*args,**kwargs):calls.append(1);return Response()
    monkeypatch.setattr(module,'session',lambda:Session())
    with pytest.raises(requests.HTTPError):
        module.download_photo({'sha1':'0'*40,'url':'https://upload.wikimedia.org/image.jpg',
            'width':500,'height':500},tmp_path/'image.jpg')
    assert len(calls)==1
