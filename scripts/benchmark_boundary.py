#!/usr/bin/env python3
"""Measured irECS/CAP/large-box comparison for a free outgoing radial packet.

This is an absorber benchmark; NOT a benchmark of reduced R-matrix scattering.
"""
import json,sys,time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh,eig,solve
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from fedvr import make_grid

def main():
    root=Path(__file__).resolve().parents[1];rows=[]
    edges=[0,.5,1]+list(range(2,21));refedges=edges+list(range(21,201))
    ref=make_grid(refedges,6);r=ref.r.real;t0=time.perf_counter()
    initial=lambda r:np.exp(-(r-8)**2/(2*1.5**2)+1.1j*r)/(np.pi*1.5**2)**.25
    e,u=eigh(ref.kinetic.toarray());c=u.conj().T@(initial(r)*np.sqrt(ref.weights))
    times=np.array([0.,8.,16.,24.,40.,60.]);reference=np.array([u@(np.exp(-1j*e*t)*c) for t in times])
    reference_time=time.perf_counter()-t0
    for tail,angle in [(12,.3),(20,.3),(28,.3),(20,.5),(28,.5),(36,.5)]:
        g=make_grid(edges,6,ecs_angle=angle,tail=tail,alpha=.8);t0=time.perf_counter()
        e,u=eig(g.kinetic.toarray());c=solve(u,initial(g.r)*np.sqrt(g.weights))
        wave=np.array([u@(np.exp(-1j*e*t)*c) for t in times]);elapsed=time.perf_counter()-t0
        inside=g.r.real<19.9;ri=ref.r.real<19.9
        diff=wave[:,inside]/np.sqrt(g.weights[inside])-reference[:,ri]/np.sqrt(ref.weights[ri])
        err=np.sqrt(np.sum(abs(diff)**2*ref.weights[ri].real,axis=1))
        rows.append({'method':'irECS','tail':tail,'angle':angle,'n':len(g.r),'max_inner_L2_error':float(max(err)),
                     'inner_L2_errors':err.tolist(),'max_eigenvalue_imag':float(e.imag.max()),'seconds':elapsed})
    for width in (10,20):
        g=make_grid(edges+list(range(21,21+width)),6);t0=time.perf_counter();r=g.r.real
        cap=.5*np.clip((r-20)/width,0,1)**4
        e,u=eig(g.kinetic.toarray()-1j*np.diag(cap));c=solve(u,initial(r)*np.sqrt(g.weights))
        wave=np.array([u@(np.exp(-1j*e*t)*c) for t in times]);elapsed=time.perf_counter()-t0
        inside=r<19.9;ri=ref.r.real<19.9
        diff=wave[:,inside]/np.sqrt(g.weights[inside])-reference[:,ri]/np.sqrt(ref.weights[ri])
        err=np.sqrt(np.sum(abs(diff)**2*ref.weights[ri].real,axis=1))
        rows.append({'method':'CAP','width':width,'n':len(g.r),'max_inner_L2_error':float(max(err)),
                     'inner_L2_errors':err.tolist(),'seconds':elapsed})
    result={'reference_n':len(ref.r),'reference_seconds':reference_time,'times':times.tolist(),'runs':rows,
            'limitation':'free-packet inner-region accuracy; Coulomb cutoff and driven spectra require separate convergence'}
    (root/'results/boundary_benchmark.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    assert min(r['max_inner_L2_error'] for r in rows if r['method']=='irECS')<1e-4
if __name__=='__main__':main()
