#!/usr/bin/env python3
"""Check the neutral-He near-resonance contaminating a 0.75 a.u. ionic control."""
from pathlib import Path
import sys,json,time,numpy as np
from scipy.sparse.linalg import LinearOperator,eigsh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from run3d import setup
from angular import radial_hydrogen
root=Path(__file__).resolve().parents[1]
c=json.loads((root/'configs/followup/pump_probe_half_frequency.json').read_text());c['M']=0;c['total_Lmax']=1;c['pulses']=[]
h,_=setup(c);Eg,g,rg=h.ground(cache_dir=root/'results/followup/ground_cache')
odd=np.array([L==1 for l,j,L in h.basis])
def project(x):
    u=np.asarray(x).reshape(h.shape,order='F').copy();u[:,:,~odd]=0
    u=(u+u.transpose(1,0,2)[:,:,h.exchange]*h.exchange_sign[None,None,:])/2
    return u.ravel(order='F')
op=LinearOperator((h.size,h.size),matvec=lambda x:project(h.apply(project(x))).real,dtype=float)
r=h.r.real;w=h.grid.weights.real
u1=r*radial_hydrogen(1,0,r,Z=2)*np.sqrt(w);u2=r*radial_hydrogen(2,1,r,Z=1)*np.sqrt(w)
guess=np.zeros(h.shape);guess[:,:,h.basis.index((0,1,1))]=np.outer(u1,u2);guess[:,:,h.basis.index((1,0,1))]=np.outer(u2,u1)
start=time.perf_counter();E,v=eigsh(op,k=1,which='SA',v0=project(guess.ravel(order='F')).real,tol=1e-9,ncv=40,maxiter=20000)
state=v[:,0];Dg=h.apply(g,(0,0,1),False)-h.apply(g)
d=float(abs(np.vdot(state,Dg)));gap=float(E[0]-Eg)
result={'ground_energy':Eg,'singlet_P_energy':float(E[0]),'excitation_energy':gap,'dipole_z':d,'oscillator_strength':2*gap*d*d,
        'detuning_from_075':gap-.75,'rabi_scale_at_F01068':.1068*d,
        'eigen_residual':float(np.linalg.norm(h.apply(state)-E[0]*state)),'exchange_error':h.exchange_error(state),
        'seconds':time.perf_counter()-start,'scope':'same l=3, p=4, truncated-Coulomb grid as half-frequency TDSE; not an infinite-basis atomic reference'}
(root/'results/followup/neutral_resonance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
