"""Fetch official weights to the workspace; never replace the current detector."""
import hashlib
import json
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    folder=ROOT/'models';folder.mkdir(exist_ok=True)
    report=[]
    for name in ('yolo26m.pt','yolo26m-depth.pt'):
        url='https://github.com/ultralytics/assets/releases/download/v8.4.0/'+name
        path=folder/name
        if not path.exists():
            with requests.get(url,stream=True,timeout=(20,90)) as response:
                response.raise_for_status()
                part=path.with_suffix('.pt.part');total=0
                with part.open('wb') as stream:
                    for chunk in response.iter_content(1024*1024):
                        total+=len(chunk)
                        if total>512*1024*1024:raise ValueError('Unexpectedly large checkpoint')
                        stream.write(chunk)
                expected=response.headers.get('Content-Length')
                if expected and total!=int(expected):raise ValueError('Incomplete download')
                part.replace(path)
        report.append({'name':name,'source':url,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        print(json.dumps(report[-1]),flush=True)
    (ROOT/'reports/yolo26_official_models.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
