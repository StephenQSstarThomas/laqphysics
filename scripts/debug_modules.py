#!/usr/bin/env python3
"""Save every step after EVERY module on bounded grids, per 程序纠错原则.md."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.fft import fft2,ifft2,fftfreq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from native import split_stage
from fedvr import make_grid
from helium3d import Helium
from propagate import exponential_action
from scipy.linalg import expm

def main():
    root=Path(__file__).resolve().parents[1];out=root/'results/debug_modules';out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(4823);n=16;dt=.02;p=2*np.pi*fftfreq(n,.2)
    psi=np.asfortranarray(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)));psi/=np.linalg.norm(psi)
    v=np.asfortranarray(rng.normal(size=(n,n))+0j);rows=[]
    for step in range(12):
        A=.03*np.sin(step*dt);snap={'input':psi.copy(),'potential':v,'momentum':p}
        ref=psi.copy()
        for stage in (1,2,3):
            split_stage(psi,v,p,dt,A,stage)
            if stage in (1,3):ref*=np.exp(-1j*v*dt/2)
            else:ref=ifft2(fft2(ref)*np.exp(-.5j*dt*((p[:,None]+A)**2+(p[None,:]+A)**2)))
            error=float(np.max(abs(psi-ref)));rows.append({'solver':'1d','step':step,'module':stage,'error':error})
            snap[f'native_stage{stage}']=psi.copy();snap[f'reference_stage{stage}']=ref.copy()
            assert error<2e-13
        np.savez_compressed(out/f'1d_step{step:04d}.npz',time=step*dt,dt=dt,vector_potential=A,**snap)
    h=Helium(make_grid([0,1,2],2),1,M=None)
    psi=rng.normal(size=h.size)+1j*rng.normal(size=h.size);psi/=np.linalg.norm(psi)
    for i in range(6):
        A=np.array([.04*np.cos(i*dt),.03*np.sin(i*dt),.02]);H=h.reference_sparse(A,True).toarray()
        snap={'input':psi.copy(),'reference_hamiltonian':H};errors=[]
        def apply(x):
            y=h.apply(x,A,True);err=np.max(abs(y-H@x));errors.append(float(err));return y
        def trace(name,value):snap[name]=value
        y,info=exponential_action(apply,psi,dt,tol=1e-12,trace=trace)
        reference=expm(-1j*dt*H)@psi;err=float(np.max(abs(y-reference)))
        snap['reference_output']=reference
        rows.append({'solver':'3d','step':i,'matvec_errors':errors,'propagator_error':err,**info})
        np.savez_compressed(out/f'3d_step{i:04d}.npz',time=i*dt,dt=dt,**snap)
        assert max(errors)<2e-12 and err<2e-11
        psi=y
    (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('Saved every stage of 12 split-operator steps and 6 Arnoldi steps:',out)
if __name__=='__main__':main()
