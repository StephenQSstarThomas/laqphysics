"""Exact total-L=0 angular compression for field-free singlet helium.

The coupled angular function is sum_m (-1)^(l-m) Y_lm Y_l,-m/sqrt(2l+1).
This reduces the ground solve to lmax+1 angular channels, without changing lmax
or the radial discretization used by the full TDSE.
"""
from pathlib import Path
import hashlib,json,os
import numpy as np
from scipy.sparse.linalg import LinearOperator,eigsh,lobpcg
from scipy.linalg import eigh
from sympy.physics.wigner import wigner_3j
from native import tensor_apply

class GroundS:
    @classmethod
    def from_grid(cls,grid,lmax,cutoff_radii=None):
        """Build a ground-only S representation without constructing unused L,M blocks."""
        from types import SimpleNamespace
        from tdse1d import cutoff
        r=grid.r;n=len(r);T=grid.kinetic.tocsr();cut=np.ones(n) if cutoff_radii is None else cutoff(r.real,*cutoff_radii)
        diag=np.stack([(-2*cut/r+l*(l+1)/(2*r*r))[:,None]+(-2*cut/r+l*(l+1)/(2*r*r))[None,:] for l in range(lmax+1)],axis=2)
        i,j=np.indices((n,n));lo=r[np.minimum(i,j)];hi=r[np.maximum(i,j)]
        rad=np.stack([lo**k/hi**(k+1)*cut[:,None]*cut[None,:] for k in range(2*lmax+1)],axis=2)
        parent=SimpleNamespace(n=n,nc=lmax+1,shape=(n,n,lmax+1),grid=grid,r=np.asfortranarray(r),cut=cut,
            ptr=T.indptr.astype(np.int32),col=T.indices.astype(np.int32),tv=T.data.astype(complex),rad=np.asfortranarray(rad),
            ch=[((l,0),(l,0)) for l in range(lmax+1)],diag=np.asfortranarray(diag))
        result=cls(parent);result.compact_only=True;return result

    def __init__(self,h):
        self.parent=h;self.n=h.n;self.lmax=max(a[0] for a,b in h.ch);self.nc=self.lmax+1
        self.shape=(self.n,self.n,self.nc);self.size=np.prod(self.shape)
        for name in ['r','ptr','col','tv','rad']:setattr(self,name,getattr(h,name))
        self.diag=np.asfortranarray(np.stack([h.diag[:,:,h.ch.index(((l,0),(l,0)))] for l in range(self.nc)],axis=2))
        ptr=[0];cols=[];orders=[];values=[]
        for l in range(self.nc):
            for j in range(self.nc):
                for k in range(abs(l-j),l+j+1,2):
                    c=(-1)**(l+j)*np.sqrt((2*l+1)*(2*j+1))*float(wigner_3j(l,j,k,0,0,0))**2
                    if abs(c)>1e-14:cols.append(j);orders.append(k);values.append(c)
            ptr.append(len(cols))
        self.vptr=np.array(ptr,np.int32);self.vc=np.array(cols,np.int32);self.vl=np.array(orders,np.int32);self.vcoef=np.array(values,complex)
        self.dptr=np.zeros(self.nc+1,np.int32);self.dc=np.zeros(0,np.int32);self.de=np.zeros(0,np.int32)
        self.dcoef=np.empty((3,0),complex,order='F');self.ldiff=np.zeros(0)

    def apply(self,x):return tensor_apply(self,x)

    def expand(self,x):
        if getattr(self,'compact_only',False):return np.asarray(x).copy()
        u=np.asarray(x).reshape(self.shape,order='F');out=np.zeros(self.parent.shape,complex,order='F')
        for c,((l,m),(j,n)) in enumerate(self.parent.ch):
            if l==j and m==-n:out[:,:,c]=(-1)**(l-m)/np.sqrt(2*l+1)*u[:,:,l]
        return out.ravel(order='F')

    def solve(self,tol=1e-10,cache_dir=None,expand=True,method='lobpcg'):
        if self.parent.grid.ecs_angle:raise ValueError('ground solve requires real grid')
        digest=hashlib.sha256(b'helium-coupled-S-v1')
        for a in [self.r,self.parent.grid.weights,self.tv,self.col,self.ptr,self.parent.cut]:digest.update(np.ascontiguousarray(a).tobytes())
        digest.update(str((self.lmax,tol)).encode());key=digest.hexdigest()
        path=Path(cache_dir)/('ground_'+key+'.npz') if cache_dir else None
        if path is not None and path.exists():
            data=np.load(path);small=data['state'];E=float(data['energy']);res=float(data['residual'])
            return E,self.expand(small) if expand else small,res
        r=self.r.real;u=r*np.exp(-1.6875*r)*np.sqrt(self.parent.grid.weights.real)
        x=np.zeros(self.shape);x[:,:,0]=u[:,None]*u[None,:]
        op=LinearOperator((self.size,self.size),matvec=lambda v:self.apply(v).real,dtype=float)
        if method=='lobpcg':
            T=self.parent.grid.kinetic.toarray().real;r=self.r.real;cut=self.parent.cut
            spectra=[];vectors=[]
            for l in range(self.nc):
                e,U=eigh(T+np.diag(-2*cut/r+l*(l+1)/(2*r*r)));spectra.append(e);vectors.append(U)
            shift=2*min(spectra[0])-.5
            def precondition(y):
                y=np.asarray(y).reshape(self.shape,order='F');out=np.empty(self.shape)
                for l in range(self.nc):
                    U=vectors[l];e=spectra[l];z=U.T@y[:,:,l]@U
                    out[:,:,l]=U@(z/(e[:,None]+e[None,:]-shift))@U.T
                return out.ravel(order='F')
            M=LinearOperator((self.size,self.size),matvec=precondition,dtype=float)
            E,v,history=lobpcg(op,x.reshape(-1,1,order='F'),M=M,largest=False,tol=tol,maxiter=250,retResidualNormsHistory=True)
            self.last_iterations=len(history)
        elif method=='eigsh':
            E,v=eigsh(op,k=1,which='SA',v0=x.ravel(order='F'),tol=tol,ncv=min(40,self.size-1),maxiter=20000)
        else:raise ValueError('unknown ground eigensolver')
        small=v[:,0].astype(complex);res=float(np.linalg.norm(self.apply(small)-E[0]*small))
        if path is not None:
            path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.stem+f'.{os.getpid()}.tmp.npz')
            np.savez(tmp,state=small,energy=E[0],residual=res);os.replace(tmp,path)
        return float(E[0]),self.expand(small) if expand else small,res
