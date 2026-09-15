#!/usr/bin/env python3
"""Export small finished artifacts; retain ignored local links to large scratch data.

Run only after writers using the project output path have exited. The source
scratch directory remains intact. Replacing a real destination is refused.
"""
import argparse,hashlib,json,os,shutil,tempfile
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--destination',required=True);a=p.parse_args()
    source=Path(a.source).resolve();dest=Path(a.destination).absolute()
    if dest.exists() and not dest.is_symlink():raise FileExistsError('destination is a real directory; refusing replacement')
    if dest.is_symlink() and dest.resolve()!=source:raise ValueError('destination points to a different source')
    stage=Path(tempfile.mkdtemp(prefix='.convergence-export-',dir=dest.parent));copied={};large=[]
    excluded={'code','ground_cache','quarantine','__pycache__'}
    for directory,dirs,files in os.walk(source,followlinks=False):
        dirs[:]=[x for x in dirs if x not in excluded]
        for name in files:
            original=Path(directory)/name;relative=original.relative_to(source)
            if name.startswith('.') or '.tmp.' in name:continue
            target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True)
            is_large=(name.endswith('.npy') or name.startswith('checkpoint') or name in ['initial.npz','final.npz'])
            if is_large:
                if not original.exists():continue
                actual=original.resolve();target.symlink_to(actual);stat=actual.stat()
                entry={'path':str(relative),'server_path':str(actual),'logical_bytes':stat.st_size,'allocated_bytes':stat.st_blocks*512}
                if name.startswith('checkpoint') or name in ['initial.npz','final.npz']:entry['sha256']=digest(actual)
                if name=='flux.npy':
                    record=original.parent/'channel_projection.json'
                    if record.exists():
                        metadata=json.loads(record.read_text());entry['reverse_C_frame_sha256']=metadata.get('raw_frames_reverse_sha256')
                        entry['frame_hash_description']=metadata.get('frame_hash_order');entry['nframes']=metadata.get('nframes')
                large.append(entry);continue
            if original.suffix not in {'.json','.npz','.png','.pdf','.md','.log','.txt'}:continue
            shutil.copy2(original,target);sha=digest(original)
            if digest(target)!=sha:raise RuntimeError(f'copy checksum mismatch: {relative}')
            copied[str(relative)]={'sha256':sha,'bytes':target.stat().st_size}
    snapshots={p.parent.name:json.loads(p.read_text()) for p in (source/'code').glob('*/numerical_snapshot.json')}
    (stage/'numerical_snapshots.json').write_text(json.dumps(snapshots,indent=2)+'\n')
    manifest={'source_scratch':str(source),'small_files':copied,'large_local_files':large,
              'large_data_in_git':False,'note':'Large files remain intact in server scratch. A fresh clone can reproduce from configuration but cannot resume or reproject without these files.'}
    (stage/'export_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if dest.is_symlink():dest.unlink()
    os.replace(stage,dest)
    print('Exported',len(copied),'small artifacts;',len(large),'ignored data links retained. Source scratch preserved.')

if __name__=='__main__':main()
