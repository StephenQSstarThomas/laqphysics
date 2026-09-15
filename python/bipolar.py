"""Direct bipolar angular algebra, retaining both parities and all requested M.

Unlike the axial natural-parity shortcut, this basis also contains the unnatural
parity states reached by successive pulses of different polarizations.
"""
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
from sympy.physics.wigner import wigner_3j,wigner_6j
from angular import channels
from helium3d import Helium
from ground_s import GroundS

@lru_cache(None)
def coupled_coulomb(l,j,ll,jj,L,k):
    if not(abs(l-ll)<=k<=l+ll and abs(j-jj)<=k<=j+jj) or (l+ll+k)%2 or (j+jj+k)%2:return 0.
    # Both reduced C matrix elements contribute their (-1)^l phases.
    # In particular the k=0 matrix must be +identity in EVERY parity sector.
    return float((-1)**(l+ll+L)*np.sqrt((2*l+1)*(2*j+1)*(2*ll+1)*(2*jj+1))*
                 wigner_3j(l,k,ll,0,0,0)*wigner_3j(j,k,jj,0,0,0)*wigner_6j(l,j,L,jj,ll,k))

@lru_cache(None)
def coupled_tensor(bra,ket,k,q,electron):
    l,j,L,M=bra;ll,jj,J,N=ket
    if M!=N+q or abs(L-J)>k or L+J<k:return 0j
    if electron==2:
        sign=(-1)**(l+j-L+ll+jj-J)
        return sign*coupled_tensor((j,l,L,M),(jj,ll,J,N),k,q,1)
    if j!=jj or abs(l-ll)>k or l+ll<k or (l+ll+k)%2:return 0j
    return complex(float((-1)**(L-M+j+J+k)*np.sqrt((2*L+1)*(2*J+1)*(2*l+1)*(2*ll+1))*
        wigner_3j(l,k,ll,0,0,0)*wigner_3j(L,k,J,-M,q,N)*wigner_6j(l,L,j,J,ll,k)))

class BipolarHelium:
    def __init__(self,grid,lmax=3,total_Lmax=3,M=None,cutoff_radii=None,natural_parity=False):
        from tdse1d import cutoff
        self.grid=grid;self.r=np.asfortranarray(grid.r);self.n=len(self.r);self.lmax=lmax;self.total_Lmax=total_Lmax
        self.basis=[(l,j,L,m) for l in range(lmax+1) for j in range(lmax+1)
                    for L in range(abs(l-j),min(l+j,total_Lmax)+1) if not natural_parity or (l+j-L)%2==0
                    for m in range(-L,L+1) if M is None or m==M]
        self.index={x:i for i,x in enumerate(self.basis)};self.nc=len(self.basis);self.shape=(self.n,self.n,self.nc);self.size=np.prod(self.shape)
        self.cut=np.ones(self.n) if cutoff_radii is None else cutoff(self.r.real,*cutoff_radii)
        T=grid.kinetic.tocsr();self.ptr=T.indptr.astype(np.int32);self.col=T.indices.astype(np.int32);self.tv=T.data.astype(complex)
        r=self.r;self.diag=np.asfortranarray(np.stack([(-2*self.cut/r+l*(l+1)/(2*r*r))[:,None]+(-2*self.cut/r+j*(j+1)/(2*r*r))[None,:] for l,j,L,m in self.basis],axis=2))
        i,j=np.indices((self.n,self.n));lo=r[np.minimum(i,j)];hi=r[np.maximum(i,j)]
        self.rad=np.asfortranarray(np.stack([lo**k/hi**(k+1)*self.cut[:,None]*self.cut[None,:] for k in range(2*lmax+1)],axis=2))
        vp=[0];vc=[];vl=[];vv=[];dp=[0];dc=[];de=[];dv=[];ld=[]
        by_LM={}
        for b,(_,_,L,m) in enumerate(self.basis):by_LM.setdefault((L,m),[]).append(b)
        for a,(l,j,L,m) in enumerate(self.basis):
            for b in by_LM[(L,m)]:
                ll,jj,_,_=self.basis[b]
                for k in range(max(abs(l-ll),abs(j-jj)),min(l+ll,j+jj)+1):
                    value=coupled_coulomb(l,j,ll,jj,L,k)
                    if abs(value)>1e-13:vc.append(b);vl.append(k);vv.append(value)
            for e in [1,2]:
                for b in self._tensor_candidates(a,1,e):
                    minus=coupled_tensor(self.basis[a],self.basis[b],1,-1,e);plus=coupled_tensor(self.basis[a],self.basis[b],1,1,e)
                    value=np.array([(minus-plus)/np.sqrt(2),1j*(minus+plus)/np.sqrt(2),coupled_tensor(self.basis[a],self.basis[b],1,0,e)])
                    if np.max(abs(value))>1e-13:
                        dc.append(b);de.append(e);dv.append(value);left=self.basis[a][e-1];right=self.basis[b][e-1];ld.append(left*(left+1)-right*(right+1))
            vp.append(len(vc));dp.append(len(dc))
        for name,v in [('vptr',vp),('vc',vc),('vl',vl),('dptr',dp),('dc',dc),('de',de)]:setattr(self,name,np.array(v,np.int32))
        self.vcoef=np.array(vv,complex);self.dcoef=np.asfortranarray(np.array(dv,complex).reshape(-1,3).T);self.ldiff=np.array(ld,float)
        self.exchange=np.array([self.index[(j,l,L,m)] for l,j,L,m in self.basis]);self.exchange_sign=np.array([(-1)**(l+j-L) for l,j,L,m in self.basis])
        product=channels(lmax,M);lookup={x:i for i,x in enumerate(product)};rows=[];cols=[];values=[]
        for c,(l,j,L,m) in enumerate(self.basis):
            for m1 in range(-l,l+1):
                m2=m-m1
                if abs(m2)>j:continue
                value=(-1)**(l-j+m)*np.sqrt(2*L+1)*float(wigner_3j(l,j,L,m1,m2,-m))
                if abs(value)>1e-14:rows.append(lookup[((l,m1),(j,m2))]);cols.append(c);values.append(value)
        self.transform=csr_matrix((values,(rows,cols)),shape=(len(product),self.nc))
        difference=self.transform.T@self.transform-csr_matrix(np.eye(self.nc))
        if np.max(abs(difference.data),initial=0)>1e-12:raise RuntimeError('bipolar transformation is not isometric')
        self.uncoupled=SimpleNamespace(grid=grid,r=self.r,n=self.n,nc=len(product),ch=product,cut=self.cut)
        self.uncoupled.prepare_surface=lambda radius:Helium.prepare_surface(self.uncoupled,radius)

    def _tensor_candidates(self,a,k,electron):
        l,j,L,m=self.basis[a];result=set()
        for other in range(max(0,self.basis[a][electron-1]-k),min(self.lmax,self.basis[a][electron-1]+k)+1):
            ll,jj=(other,j) if electron==1 else (l,other)
            for J in range(max(abs(ll-jj),abs(L-k)),min(ll+jj,L+k,self.total_Lmax)+1):
                for N in range(max(-J,m-k),min(J,m+k)+1):
                    b=self.index.get((ll,jj,J,N))
                    if b is not None:result.add(b)
        return sorted(result)

    def tensor_matrix(self,electron,k,q):
        rows=[];cols=[];values=[]
        for a in range(self.nc):
            for b in self._tensor_candidates(a,k,electron):
                v=coupled_tensor(self.basis[a],self.basis[b],k,q,electron)
                if abs(v)>1e-13:rows.append(a);cols.append(b);values.append(v)
        return csr_matrix((values,(rows,cols)),shape=(self.nc,self.nc))

    prepare_surface=Helium.prepare_surface
    flux=Helium.flux
    apply=Helium.apply
    operator=Helium.operator

    def expand(self,x):
        a=np.asarray(x).reshape(self.n*self.n,self.nc,order='F')
        return (self.transform@a.T).T.ravel(order='F')

    def project(self,x):
        a=np.asarray(x).reshape(self.n*self.n,self.uncoupled.nc,order='F')
        return (self.transform.T@a.T).T.ravel(order='F')

    def ground(self,tol=1e-10,cache_dir=None):
        # Use this Hamiltonian's actual cutoff and radial multipoles.
        parent=SimpleNamespace(n=self.n,nc=self.lmax+1,shape=(self.n,self.n,self.lmax+1),grid=self.grid,r=self.r,cut=self.cut,
            ptr=self.ptr,col=self.col,tv=self.tv,rad=self.rad,ch=[((l,0),(l,0)) for l in range(self.lmax+1)],
            diag=np.asfortranarray(np.stack([self.diag[:,:,self.index[(l,l,0,0)]] for l in range(self.lmax+1)],axis=2)))
        s=GroundS(parent);s.compact_only=True;E,g,res=s.solve(tol=tol,cache_dir=cache_dir,expand=False)
        g=g.reshape(s.shape,order='F');out=np.zeros(self.shape,complex,order='F')
        for l in range(self.lmax+1):out[:,:,self.index[(l,l,0,0)]]=g[:,:,l]
        return E,out.ravel(order='F'),res

    def exchange_error(self,x):
        u=np.asarray(x).reshape(self.shape,order='F')
        return float(np.linalg.norm(u-u.transpose(1,0,2)[:,:,self.exchange]*self.exchange_sign[None,None,:]))
