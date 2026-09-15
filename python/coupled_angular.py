"""Bipolar spherical harmonics for linearly polarized M=0 helium calculations.

Keep individual l<=lmax and total L<=total_Lmax with natural parity. All angular
matrices are obtained by an explicit Clebsch--Gordan change of basis from the
independently verified product-harmonic implementation. Total-L truncation is a
separate, convergible approximation; this is not a fitted effective potential.
"""
import numpy as np
from sympy.physics.wigner import wigner_3j
from helium3d import Helium
from ground_s import GroundS

class CoupledHelium:
    def __init__(self,uncoupled,total_Lmax):
        h=uncoupled;self.uncoupled=h
        if any(a[1]+b[1]!=0 for a,b in h.ch):raise ValueError('coupled backend currently requires M=0')
        self.lmax=max(a[0] for a,b in h.ch);self.total_Lmax=total_Lmax
        self.basis=[(l,j,L) for l in range(self.lmax+1) for j in range(self.lmax+1)
                    for L in range(abs(l-j),min(l+j,total_Lmax)+1) if (l+j-L)%2==0]
        self.n=h.n;self.nc=len(self.basis);self.shape=(self.n,self.n,self.nc);self.size=np.prod(self.shape)
        self.grid=h.grid;self.r=h.r;self.cut=h.cut
        for name in ['ptr','col','tv','rad']:setattr(self,name,getattr(h,name))
        U=np.zeros((h.nc,self.nc))
        for c,(l,j,L) in enumerate(self.basis):
            for a,((ll,m),(jj,n)) in enumerate(h.ch):
                if l==ll and j==jj:
                    U[a,c]=(-1)**(l-j)*np.sqrt(2*L+1)*float(wigner_3j(l,j,L,m,n,0))
        self.transform=U
        if np.linalg.norm(U.T@U-np.eye(self.nc))>1e-11:raise RuntimeError('CG transformation is not isometric')
        self.diag=np.asfortranarray(np.stack([h.diag[:,:,h.ch.index(((l,0),(j,0)))] for l,j,L in self.basis],axis=2))
        V=[]
        for lam in range(h.rad.shape[2]):
            A=np.zeros((h.nc,h.nc),complex)
            for a in range(h.nc):
                for k in range(h.vptr[a],h.vptr[a+1]):
                    if h.vl[k]==lam:A[a,h.vc[k]]+=h.vcoef[k]
            V.append(U.T@A@U)
        C=np.zeros((2,h.nc,h.nc),complex)
        for a in range(h.nc):
            for k in range(h.dptr[a],h.dptr[a+1]):C[h.de[k]-1,a,h.dc[k]]+=h.dcoef[2,k]
        C=np.array([U.T@A@U for A in C])
        vp=[0];vc=[];vl=[];vv=[];dp=[0];dc=[];de=[];dv=[];ld=[]
        for a,(l,j,L) in enumerate(self.basis):
            for b,(ll,jj,LL) in enumerate(self.basis):
                for lam,A in enumerate(V):
                    if abs(A[a,b])>1e-13:vc.append(b);vl.append(lam);vv.append(A[a,b])
                for e in range(2):
                    if abs(C[e,a,b])>1e-13:
                        dc.append(b);de.append(e+1);dv.append([0,0,C[e,a,b]])
                        bra=(l,j)[e];ket=(ll,jj)[e];ld.append(bra*(bra+1)-ket*(ket+1))
            vp.append(len(vc));dp.append(len(dc))
        for name,values in [('vptr',vp),('vc',vc),('vl',vl),('dptr',dp),('dc',dc),('de',de)]:setattr(self,name,np.array(values,np.int32))
        self.vcoef=np.array(vv,complex);self.dcoef=np.asfortranarray(np.array(dv,complex).reshape(-1,3).T);self.ldiff=np.array(ld,float)
        self.exchange=np.array([self.basis.index((j,l,L)) for l,j,L in self.basis])
        self.exchange_sign=np.array([(-1)**(l+j-L) for l,j,L in self.basis])

    prepare_surface=Helium.prepare_surface
    flux=Helium.flux
    reference_sparse=Helium.reference_sparse

    def apply(self,x,field=(0.,0.,0.),velocity=False):
        if abs(field[0])+abs(field[1])>1e-15:raise ValueError('M=0 coupled basis only supports z polarization')
        return Helium.apply(self,x,field,velocity)

    def expand(self,x):
        u=np.asarray(x).reshape(self.n*self.n,self.nc,order='F')
        return (u@self.transform.T).ravel(order='F')

    def project(self,x):
        u=np.asarray(x).reshape(self.n*self.n,self.uncoupled.nc,order='F')
        return (u@self.transform).ravel(order='F')

    def ground(self,tol=1e-10,cache_dir=None):
        import os
        s=GroundS(self.uncoupled);E,g,res=s.solve(tol,cache_dir or os.environ.get('HELIUM_GROUND_CACHE'),expand=False)
        small=g.reshape(s.shape,order='F');out=np.zeros(self.shape,complex,order='F')
        for l in range(self.lmax+1):out[:,:,self.basis.index((l,l,0))]=small[:,:,l]
        return E,out.ravel(order='F'),res

    def exchange_error(self,x):
        u=np.asarray(x).reshape(self.shape,order='F')
        return float(np.linalg.norm(u-u.transpose(1,0,2)[:,:,self.exchange]*self.exchange_sign[None,None,:]))
