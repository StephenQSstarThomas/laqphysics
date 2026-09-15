#!/usr/bin/env python3
"""Run CPU postprocessing after a complete, validated propagation checkpoint."""
import argparse,json,time,subprocess,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--ionic-device',default='cpu');p.add_argument('--validate-only',action='store_true');p.add_argument('--reuse-spectrum',action='store_true');a=p.parse_args()
out=Path(a.out).resolve();snapshot=Path(a.snapshot).resolve()
while True:
    try:meta=json.loads((out/'run.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):meta={}
    if meta.get('complete'):break
    time.sleep(10)
history=json.loads((out/'history.json').read_text());tol=meta['config'].get('linear_tolerance',1e-10)
for row in history:
    if not 0<=row['real_region_norm']<=1+1e-6:raise ValueError('invalid physical inner norm')
    if row['exchange_error']>1e-7:raise ValueError('exchange symmetry failed')
    if row.get('linear_residual',0)>tol*1.01:raise ValueError('true linear residual exceeded tolerance')
if meta['ground_residual']>1e-8:raise ValueError('ground-state residual failed')
segments=[meta]+[json.loads(p.read_text()) for p in out.glob('run_segment*.json')]
if any(s['signature']!=meta['signature'] for s in segments):raise ValueError('checkpoint segment signature mismatch')
if max(s.get('maximum_linear_residual_this_invocation',0.) for s in segments)>tol*1.01:
    raise ValueError('maximum true linear residual exceeded tolerance')
print('PROPAGATION CHECKS PASSED',out.name,flush=True)
if not a.validate_only:
    if a.reuse_spectrum and (out/'spectrum.npz').exists():
        with np.load(out/'spectrum.npz') as spectrum:
            if str(spectrum['source_signature'])!=meta['signature']:raise ValueError('stored spectrum configuration mismatch')
            if not np.isfinite(spectrum['amplitudes']).all():raise ValueError('stored spectrum is nonfinite')
        print('REUSED completed spectrum; no raw history needed',out.name,flush=True)
    else:
        subprocess.run([sys.executable,str(snapshot/'scripts/extract_projected.py'),'--out',str(out),'--ionic-device',a.ionic_device],check=True)
