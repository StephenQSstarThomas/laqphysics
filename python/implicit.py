"""Stiffly stable fourth-order propagation: CF4 Magnus + [2/2] Pade.

GPU linear systems use right-preconditioned GMRES and verify the TRUE residual.
The reference energy is removed analytically, preserving the field-free ground
phase exactly. This method still requires timestep and spectrum convergence.
"""
import numpy as np
import torch
from scipy.sparse import eye
from scipy.sparse.linalg import splu
from scipy.linalg import eig,inv

class SeparablePreconditioner:
    """Exact resolvent of the sum of two uncoupled ionic radial Hamiltonians.
    Electron repulsion and A.p stay in the true GMRES operator, not in this
    preconditioner. No approximation is made to the converged linear solve.
    """
    def __init__(self,engine):
        h=engine.h;self.engine=engine
        labels=[(l,j) for l,j,L in h.basis] if hasattr(h,'basis') else [(a[0],b[0]) for a,b in h.ch]
        maxl=max(max(p) for p in labels);T=h.grid.kinetic.toarray();r=h.r
        values=[];vectors=[];inverses=[];conditions=[]
        for l in range(maxl+1):
            H=T+np.diag(-2*h.cut/r+l*(l+1)/(2*r*r));e,S=eig(H)
            conditions.append(float(np.linalg.cond(S)));values.append(e);vectors.append(S);inverses.append(inv(S))
        if max(conditions)>1e12:raise RuntimeError('ionic eigenbasis too ill-conditioned for separable preconditioner')
        self.condition_numbers=conditions
        self.S1=engine.tensor(np.array([vectors[a] for a,b in labels]));self.S2=engine.tensor(np.array([vectors[b] for a,b in labels]))
        self.I1=engine.tensor(np.array([inverses[a] for a,b in labels]));self.I2=engine.tensor(np.array([inverses[b] for a,b in labels]))
        self.energy=engine.tensor(np.array([values[b][:,None]+values[a][None,:] for a,b in labels]))

    def apply(self,x,alpha,shift):
        h=self.engine;u=x.reshape(h.nc,h.n,h.n)
        z=self.I2@u@self.I1.transpose(-1,-2)
        z=z/(1+1j*alpha*(self.energy+shift))
        return (self.S2@z@self.S1.transpose(-1,-2)).reshape(-1)

SQRT3=np.sqrt(3.)
C1=.5-SQRT3/6;C2=.5+SQRT3/6
A1=(3-2*SQRT3)/12;A2=(3+2*SQRT3)/12

@torch.no_grad()
def gmres(apply,b,preconditioner,x0=None,tol=1e-11,restart=32,max_restarts=10):
    M=preconditioner if callable(preconditioner) else lambda y:preconditioner*y
    x=b.clone() if x0 is None else x0.clone();bnorm=torch.linalg.vector_norm(b).item()
    if bnorm==0:return torch.zeros_like(b),{'iterations':0,'relative_residual':0.}
    iterations=0
    for cycle in range(max_restarts):
        r=b-apply(x);beta=torch.linalg.vector_norm(r).item()
        if beta<=tol*bnorm:return x,{'iterations':iterations,'relative_residual':beta/bnorm}
        Q=torch.empty((restart+1,b.numel()),dtype=b.dtype,device=b.device).T
        H=torch.zeros((restart+1,restart),dtype=b.dtype,device=b.device);Q[:,0]=r/beta
        for j in range(restart):
            v=apply(M(Q[:,j].contiguous()));iterations+=1
            for rep in range(2):
                z=Q[:,:j+1].mH@v;H[:j+1,j]+=z;v-=Q[:,:j+1]@z
            norm=torch.linalg.vector_norm(v).item();H[j+1,j]=norm
            if norm<1e-14 or (j+1)%4==0 or j+1==restart:
                m=j+1;small=H[:m+1,:m].cpu().numpy();rhs=np.zeros(m+1,complex);rhs[0]=beta
                y=np.linalg.lstsq(small,rhs,rcond=None)[0];estimate=np.linalg.norm(rhs-small@y)
                if estimate<tol*bnorm or norm<1e-14 or m==restart:
                    candidate=x+M(Q[:,:m]@torch.as_tensor(y,dtype=b.dtype,device=b.device))
                    true=torch.linalg.vector_norm(b-apply(candidate)).item()/bnorm
                    if true<=tol:return candidate,{'iterations':iterations,'relative_residual':true}
                    if norm<1e-14 or m==restart:x=candidate;break
            Q[:,j+1]=v/norm
    raise RuntimeError(f'GMRES did not converge: residual={true:.3e}, iterations={iterations}')

@torch.no_grad()
def pade_step(engine,x,field,dt,reference_energy=0.,tol=1e-11):
    def K(y):return engine.apply(y,field,True)-reference_energy*y
    a2=getattr(engine,'a2_coefficient',1.)
    diagonal=engine.diagonal+a2*sum(float(a)**2 for a in field)-reference_energy
    iterations=0;residual=0.
    for root in (3+1j*SQRT3,3-1j*SQRT3):
        alpha=dt/root
        def A(y):return y+1j*alpha*K(y)
        preconditioner=(lambda y:engine.separable.apply(y,alpha,a2*sum(float(a)**2 for a in field)-reference_energy)) if hasattr(engine,'separable') else 1/(1+1j*alpha*diagonal)
        s,info=gmres(A,x,preconditioner,tol=tol)
        x=2*s-x;iterations+=info['iterations'];residual=max(residual,info['relative_residual'])
    return x,{'linear_iterations':iterations,'linear_residual':residual}

@torch.no_grad()
def cf4_step(engine,x,avec,t,dt,reference_energy=0.,tol=1e-11,depth=0):
    fields=np.array([avec(t+C1*dt),avec(t+C2*dt)],float)
    iterations=0;residual=0.;original=x
    try:
        for w in ((A2,A1),(A1,A2)):
            effective=2*np.einsum('a,aj->j',w,fields)
            correction=getattr(engine,'a2_coefficient',1.)*(np.dot(w,np.sum(fields**2,axis=1))-.5*np.dot(effective,effective))
            x,info=pade_step(engine,x,effective,dt/2,reference_energy,tol)
            x*=np.exp(-1j*dt*(correction+reference_energy/2))
            iterations+=info['linear_iterations'];residual=max(residual,info['linear_residual'])
        return x,{'linear_iterations':iterations,'linear_residual':residual,'dimension':0,'residual_estimate':None}
    except RuntimeError as error:
        if 'GMRES did not converge' not in str(error) or depth>=6:raise
        x,a=cf4_step(engine,original,avec,t,dt/2,reference_energy,tol/2,depth+1)
        x,b=cf4_step(engine,x,avec,t+dt/2,dt/2,reference_energy,tol/2,depth+1)
        return x,{'linear_iterations':a['linear_iterations']+b['linear_iterations'],
                  'linear_residual':max(a['linear_residual'],b['linear_residual']),
                  'dimension':0,'residual_estimate':None,'subdivided':True}

class SparseCF4:
    """Small ionic systems: sparse LU instead of iterative solves; supports negative dt."""
    def __init__(self,matrix_at,key_at=None):self.matrix_at=matrix_at;self.key_at=key_at;self.key=None;self.factors=None
    def step(self,x,t,dt):
        key=(dt,self.key_at(t+C1*dt),self.key_at(t+C2*dt)) if self.key_at else None
        if key is None or key!=self.key or self.factors is None:
            h1=self.matrix_at(t+C1*dt);h2=self.matrix_at(t+C2*dt);I=eye(h1.shape[0],format='csc');self.factors=[]
            for H in (A2*h1+A1*h2,A1*h1+A2*h2):
                for root in (3+1j*SQRT3,3-1j*SQRT3):
                    alpha=dt/root;self.factors.append(splu((I+1j*alpha*H).tocsc()))
            self.key=key
        for lu in self.factors:x=2*lu.solve(x)-x
        return x
