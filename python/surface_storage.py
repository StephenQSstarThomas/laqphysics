"""Bounded NPY shards for temporal arrays. The cap includes the file header."""
import json,os
from pathlib import Path
import numpy as np

DEFAULT_FILE_BYTES=4_000_000_000
HARD_FILE_BYTES=5_000_000_000

def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)

def file_limit(value=DEFAULT_FILE_BYTES):
    value=int(value)
    if not 4096<value<=HARD_FILE_BYTES:raise ValueError('max_file_bytes must be >4096 and <=5,000,000,000 (decimal bytes)')
    return value

def capped_npz(path,max_file_bytes=DEFAULT_FILE_BYTES,**arrays):
    """Preflight uncompressed size; atomic promotion and actual-size verification."""
    path=Path(path);limit=file_limit(max_file_bytes)
    estimate=sum(np.asarray(v).nbytes+1024 for v in arrays.values())+4096
    if estimate>limit:raise ValueError(f'{path.name}: checkpoint needs about {estimate} bytes, above file cap {limit}')
    temporary=path.with_name(path.stem+'.tmp.npz');np.savez(temporary,**arrays)
    if temporary.stat().st_size>limit:
        temporary.unlink();raise ValueError('NPZ exceeds configured file cap')
    os.replace(temporary,path)

class ShardedArray:
    def __init__(self,folder,shape=None,dtype=np.complex128,mode='r',max_file_bytes=DEFAULT_FILE_BYTES):
        self.folder=Path(folder);self.mode=mode;self._index=None;self._array=None
        manifest=self.folder/'array.json'
        if mode=='w+':
            if manifest.exists():raise FileExistsError(manifest)
            self.folder.mkdir(parents=True,exist_ok=True);self.shape=tuple(map(int,shape));self.dtype=np.dtype(dtype)
            limit=file_limit(max_file_bytes);frame=int(np.prod(self.shape[1:]))*self.dtype.itemsize
            self.frames_per_file=(limit-4096)//frame
            if self.frames_per_file<1:raise ValueError('one array frame exceeds file cap; refine storage layout')
            self.limit=limit
            atomic_json(manifest,{'shape':self.shape,'dtype':self.dtype.str,'frames_per_file':self.frames_per_file,'max_file_bytes':limit})
        else:
            m=json.loads(manifest.read_text());self.shape=tuple(m['shape']);self.dtype=np.dtype(m['dtype'])
            self.frames_per_file=m['frames_per_file'];self.limit=m['max_file_bytes']

    def __len__(self):return self.shape[0]

    def _open(self,index):
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        part=index//self.frames_per_file
        if part!=self._index:
            self.flush();self._array=None
            path=self.folder/f'part_{part:06d}.npy'
            if not path.exists():
                if self.mode=='r':raise FileNotFoundError(path)
                n=min(self.frames_per_file,len(self)-part*self.frames_per_file)
                self._array=np.lib.format.open_memmap(path,mode='w+',dtype=self.dtype,shape=(n,*self.shape[1:]))
                if path.stat().st_size>self.limit:raise RuntimeError('shard exceeded file cap')
            else:self._array=np.load(path,mmap_mode='r' if self.mode=='r' else 'r+')
            self._index=part
        return index%self.frames_per_file

    def __getitem__(self,index):
        if isinstance(index,slice):return np.array([self[i] for i in range(*index.indices(len(self)))])
        if not np.isscalar(index):return np.array([self[int(i)] for i in index])
        row=self._open(int(index));return self._array[row]

    def __setitem__(self,index,value):
        if self.mode=='r':raise ValueError('read-only array')
        row=self._open(int(index));self._array[row]=value

    def flush(self):
        if self._array is not None and self.mode!='r':self._array.flush()

    def close(self):self.flush();self._array=None;self._index=None

class TimeSelection:
    def __init__(self,array,indices):
        self.array=array;self.indices=np.asarray(indices);self.shape=(len(indices),*array.shape[1:])
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):return self.array[int(self.indices[index])]

def load_flux(out):
    out=Path(out)
    if (out/'flux.npy').exists():return np.load(out/'flux.npy',mmap_mode='r')
    if (out/'flux_shards/array.json').exists():return ShardedArray(out/'flux_shards')
    raise FileNotFoundError('No raw surface history; an online spectrum run intentionally does not write flux.npy')
