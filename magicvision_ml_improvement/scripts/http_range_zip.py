"""Read a public ZIP by verified HTTP byte ranges, avoiding a 13 GB download."""
from collections import OrderedDict
import io
import requests


class RangeReader(io.RawIOBase):
    def __init__(self,url,size,block_size=4*1024*1024):
        self.url=url;self.size=size;self.pos=0;self.block_size=block_size
        self.session=requests.Session();self.cache=OrderedDict();self.bytes_downloaded=0

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        if whence not in (0,1,2):raise ValueError('Invalid ZIP seek origin')
        target=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if target<0:raise ValueError('Negative ZIP seek')
        self.pos=target;return target

    def read(self,size=-1):
        end=self.size if size<0 else min(self.size,self.pos+size)
        if end-self.pos>256*1024*1024:raise ValueError('Read ZIP members as a bounded stream')
        output=[]
        while self.pos<end:
            index=self.pos//self.block_size
            if index not in self.cache:
                start=index*self.block_size;stop=min(start+self.block_size,self.size)-1
                response=self.session.get(self.url,headers={'Range':f'bytes={start}-{stop}'},stream=True,timeout=(20,90))
                try:
                    if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{stop}/{self.size}':
                        raise ValueError(f'Publisher does not serve requested range: {response.status_code} {response.headers.get("Content-Range")}')
                    chunk=response.content
                    if len(chunk)!=stop-start+1:raise ValueError('Incomplete ZIP range')
                    self.bytes_downloaded+=len(chunk);self.cache[index]=chunk
                    if len(self.cache)>8:self.cache.popitem(last=False)
                finally:response.close()
            self.cache.move_to_end(index)
            chunk=self.cache[index];offset=self.pos%self.block_size
            count=min(end-self.pos,len(chunk)-offset)
            output.append(chunk[offset:offset+count]);self.pos+=count
        return b''.join(output)

    def close(self):
        self.session.close();super().close()
