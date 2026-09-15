#!/usr/bin/env python3
"""Freeze the numerical code used by long workers while reports evolve."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--root',default='/playpen1/shiqiu/laqphysics-data/convergence_work/code');a=p.parse_args()
files=[*root.glob('python/*.py'),*root.glob('scripts/*.py'),*root.glob('src/*.f90'),*root.glob('build/*.so'),root/'Makefile']
hashes={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)}
key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest();dest=Path(a.root)/key[:16]
if not dest.exists():
    for f in files:
        target=dest/f.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target);target.chmod(0o444)
    meta={'id':key,'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'sha256':hashes}
    (dest/'numerical_snapshot.json').write_text(json.dumps(meta,indent=2)+'\n')
for relative,digest in hashes.items():assert hashlib.sha256((dest/relative).read_bytes()).hexdigest()==digest
print(dest)
