#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from open_system import propagate,diagnostics
out=Path(__file__).resolve().parents[1]/'results/decoherence';out.mkdir(parents=True,exist_ok=True)
e=np.linspace(.35,.85,41);w=np.full(len(e),e[1]-e[0]);w[[0,-1]]*=.5;rows=[]
for gamma in (0.,.002,.01,.05):
    rho=propagate(e,w,gamma=gamma);d=diagnostics(rho,len(e));d['gamma']=gamma;rows.append(d)
    np.savez(out/f'gamma{gamma:g}.npz',energy=e,weights=w,rho=rho)
    assert d['trace_error']<1e-9 and d['minimum_eigenvalue']>-1e-8
(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
