"""Multi-RHS ionic adjoint propagation with the same CF4-Pade formula as sparse LU.

The physical operator, all RHS columns and residuals remain complex128. Quadratic
field coefficients are passed independently, so the CF4 combinations exactly
match SparseCF4, including its zero reference-energy convention.
"""
from types import SimpleNamespace
import numpy as np
import torch
from scipy.linalg import eig,inv

class IonicAdjoint:
    a2_coefficient=0.
    def __init__(self,ion,nrhs,device='cuda:0',preconditioner_precision='complex64'):
        self.ion=ion;self.device=torch.device(device);self.n=ion.n;self.na=ion.na;self.nrhs=nrhs;self.ndof=self.n*self.na
        self.tensor=lambda x:torch.as_tensor(np.array(x,copy=True),dtype=torch.complex128,device=self.device)
        self.mixed=hasattr(ion,'Pq')
        matrices=[ion.h0,*ion.Pq,*ion.Df,*ion.Q] if self.mixed else [ion.h0,*ion.P,ion.identity]
        matrices=[x.conj().T.tocsr() for x in matrices]
        pattern=sum(abs(x) for x in matrices).tocsr();pattern.sort_indices()
        rows=np.repeat(np.arange(self.ndof),np.diff(pattern.indptr));cols=pattern.indices
        self.ptr=torch.tensor(pattern.indptr.astype(np.int64),device=self.device);self.col=torch.tensor(cols.astype(np.int64),device=self.device)
        self.data=self.tensor([np.asarray(x[rows,cols]).ravel() for x in matrices]);self.diagonal=self.tensor(np.repeat(matrices[0].diagonal(),nrhs))
        dtype=torch.complex64 if preconditioner_precision=='complex64' else torch.complex128;self.preconditioner_dtype=dtype
        values={};vectors={};inverses={};conditions={}
        for l in sorted({l for l,m in ion.states}):
            i=next(i for i,(ll,m) in enumerate(ion.states) if ll==l)
            h=matrices[0][i*self.n:(i+1)*self.n,i*self.n:(i+1)*self.n].toarray();e,S=eig(h)
            values[l]=e;vectors[l]=S;inverses[l]=inv(S);conditions[l]=float(np.linalg.cond(S))
        if max(conditions.values())>1e12:raise ValueError('ill-conditioned ionic adjoint preconditioner')
        convert=lambda x:torch.as_tensor(np.array(x),dtype=dtype,device=self.device)
        self.S=convert([vectors[l] for l,m in ion.states]);self.Si=convert([inverses[l] for l,m in ion.states]);self.energy=convert([values[l] for l,m in ion.states])
        self.separable=SimpleNamespace(apply=self.precondition,flexible=dtype==torch.complex64)
        self.condition_numbers=conditions;self.key=None

    def coefficients(self,field):
        return np.asarray(field) if self.mixed else np.r_[field,.5*np.dot(field,field)]

    def precondition(self,x,alpha,shift):
        u=x.to(self.preconditioner_dtype).reshape(self.na,self.n,self.nrhs)
        y=(self.Si@u)/(1+1j*alpha*(self.energy+shift))[:,:,None]
        return (self.S@y).reshape(-1).to(x.dtype)

    @torch.no_grad()
    def apply(self,x,field,velocity=True):
        key=tuple(float(x) for x in field)
        if key!=self.key:
            values=self.tensor([1.,*field])@self.data
            self.H=torch.sparse_csr_tensor(self.ptr,self.col,values,size=(self.ndof,self.ndof),device=self.device,check_invariants=False);self.key=key
        return torch.sparse.mm(self.H,x.reshape(self.ndof,self.nrhs)).reshape(-1)

    def host(self,x):return x.detach().cpu().numpy().reshape(self.ndof,self.nrhs)
