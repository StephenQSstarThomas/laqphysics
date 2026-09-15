"""Continuum-inclusive He+ TDSE for prepared oriented states; no neutral He source.

This is the appropriate separated-ion control problem after the first electron
has left. It does not claim to change that electron's unconditional spectrum.
"""
from pathlib import Path
from types import SimpleNamespace
import json,time,os,signal,hashlib
import numpy as np
import torch
from scipy.sparse import eye
from scipy.linalg import eig,inv
from fedvr import make_grid
from surface3d import Ionic
from implicit import cf4_step
from run3d import vector_function
from output_lock import exclusive_output

class IonGPU:
    a2_coefficient=.5
    def __init__(self,ion,device='cuda:0'):
        self.ion=ion;self.device=torch.device(device);self.n=ion.n;self.na=ion.na;self.size=ion.n*ion.na
        self.tensor=lambda a:torch.as_tensor(np.array(a,copy=True),dtype=torch.complex128,device=self.device)
        matrices=[ion.h0,*ion.P,ion.identity]
        pattern=sum(abs(a) for a in matrices).tocsr();pattern.sort_indices()
        row=np.repeat(np.arange(self.size),np.diff(pattern.indptr));col=pattern.indices
        self.ptr=torch.as_tensor(pattern.indptr.astype(np.int64),device=self.device);self.col=torch.as_tensor(col.astype(np.int64),device=self.device)
        self.data=self.tensor(np.array([np.asarray(a[row,col]).ravel() for a in matrices]));self.diagonal=self.tensor(ion.h0.diagonal())
        self.field=None
        eigen=[];S=[];Si=[];condition=[]
        for l in sorted(set(l for l,m in ion.states)):
            idx=ion.index[(l,0)];block=ion.h0[idx*self.n:(idx+1)*self.n,idx*self.n:(idx+1)*self.n].toarray()
            e,v=eig(block);eigen.append(e);S.append(v);Si.append(inv(v));condition.append(float(np.linalg.cond(v)))
        if max(condition)>1e12:raise RuntimeError('ill-conditioned ionic eigenbasis')
        self.eigen=self.tensor(np.array([eigen[l] for l,m in ion.states]));self.S=self.tensor(np.array([S[l] for l,m in ion.states]));self.Si=self.tensor(np.array([Si[l] for l,m in ion.states]))
        self.condition_numbers=condition;self.separable=SimpleNamespace(apply=self.precondition)

    def host(self,x):return x.detach().cpu().numpy()

    def precondition(self,x,alpha,shift):
        u=x.reshape(self.na,self.n,1)
        z=(self.Si@u).squeeze(-1)/(1+1j*alpha*(self.eigen+shift))
        return (self.S@z[:,:,None]).reshape(-1)

    def apply(self,x,field,velocity=True):
        field=tuple(float(a) for a in field)
        if self.field!=field:
            coeff=self.tensor([1.,*field,.5*np.dot(field,field)])
            values=coeff@self.data
            self.H=torch.sparse_csr_tensor(self.ptr,self.col,values,size=(self.size,self.size),device=self.device,check_invariants=False)
            self.field=field
        return torch.sparse.mm(self.H,x[:,None]).ravel()

@exclusive_output('.propagation.lock')
def run(config,out,device='cuda:0',resume=False,max_steps=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    checkpoint=out/'checkpoint.npz'
    if not resume and ((out/'run.json').exists() or checkpoint.exists()):raise FileExistsError(out)
    saved=np.load(checkpoint) if resume else None
    if saved is not None and str(saved['signature'])!=signature:raise ValueError('config changed; refusing unsafe resume')
    grid=make_grid(**config['radial']);lmax=config['lmax']
    dummy=SimpleNamespace(grid=grid,n=len(grid.r),r=grid.r,cut=np.ones(len(grid.r)),ch=[((l,m),(0,0)) for l in range(lmax+1) for m in range(-l,l+1)])
    ion=Ionic(dummy);engine=IonGPU(ion,device);initial=tuple(config.get('initial',[2,1,1]))
    psi=saved['psi'] if resume else ion.final_states([initial])[:,0];state=engine.tensor(psi)
    labels=[(n,l,m) for n in range(1,7) for l in range(min(n,lmax+1)) for m in range(-l,l+1)]
    bound=ion.final_states(labels);avec,pulses=vector_function(config);T=max(p.start+p.duration for p,pol in pulses)
    if np.max(abs(avec(T)))>1e-10:
        raise ValueError('Final bound populations require zero final vector potential or a final-state gauge transformation')
    steps=int(np.ceil(T/config.get('dt',.25)));dt=T/steps;begin=int(saved['step']) if resume else 0
    history=json.loads((out/'history.json').read_text()) if resume else []
    history=[row for row in history if row['time']<=begin*dt+1e-9]
    maxres=float(saved['maxres']) if resume else 0.;start=time.perf_counter()
    stop=steps if max_steps is None else min(steps,begin+max_steps)
    if stop<=begin:raise ValueError('no remaining steps')
    interrupted=[False];handlers={}
    for sig in (signal.SIGUSR1,signal.SIGTERM):
        handlers[sig]=signal.signal(sig,lambda signum,frame:interrupted.__setitem__(0,True))
    try:
        for i in range(begin,stop):
            state,info=cf4_step(engine,state,avec,i*dt,dt,reference_energy=-2/initial[0]**2,tol=config.get('linear_tolerance',1e-11))
            maxres=max(maxres,info['linear_residual'])
            if interrupted[0]:stop=i+1
            if (i+1)%config.get('checkpoint_every',1000)==0 or i+1==stop:
                psi=engine.host(state);prob=abs(bound.T@psi)**2
                real_weight=np.tile(grid.interior_weights/abs(grid.weights),ion.na);norm=float(sum(abs(psi)**2*real_weight))
                p3=float(sum(p for label,p in zip(labels,prob) if label[:2]==(3,1)));d3=float(sum(p for label,p in zip(labels,prob) if label[:2]==(3,2)))
                history.append({'time':(i+1)*dt,'P_3p':p3,'P_3d':d3,'inner_norm':norm})
                temp=out/'history.tmp.json';temp.write_text(json.dumps(history,indent=2)+'\n');os.replace(temp,out/'history.json')
                np.savez(out/'checkpoint.tmp.npz',psi=psi,step=i+1,signature=signature,maxres=maxres)
                os.replace(out/'checkpoint.tmp.npz',checkpoint)
                print(f'ION {i+1}/{steps}, P3p={p3:.7g}, P3d={d3:.3g}, wall={time.perf_counter()-start:.1f}s',flush=True)
            if i+1==stop:break
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
    np.savez(out/'final.npz',psi=psi);np.savez(out/'populations.npz',labels=labels,probabilities=prob)
    meta={'config':config,'signature':signature,'dt':dt,'steps':steps,'completed_steps':stop,'complete':stop==steps,'maximum_linear_residual':maxres,'seconds_this_invocation':time.perf_counter()-start,
          'P_3p':p3,'P_3d':d3,'inner_norm':norm,'bound_n_le_6_probability':float(sum(prob)),
          'scope':'prepared He+ one-electron TDSE; continuum present, only n<=6 bound populations reported',
          'eigenbasis_condition_numbers':engine.condition_numbers}
    (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n');(out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    return meta
