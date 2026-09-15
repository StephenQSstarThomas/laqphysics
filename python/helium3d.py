"""Genuine two-electron 3D Coulomb Hamiltonian in a truncated product Y_lm basis.

Both radial coordinates evolve. No fitted He binding energy or single-active-electron
potential is used. Truncation is set by lmax and the FE-DVR grid.
"""
import numpy as np
from scipy.sparse import csr_matrix, diags, kron, eye
from scipy.sparse.linalg import LinearOperator,eigsh
from angular import channels,coulomb,cartesian
from native import tensor_apply

class Helium:
    def __init__(self,grid,lmax=2,M=0,cutoff_radii=None):
        self.grid=grid;self.r=np.asfortranarray(grid.r, dtype=complex)
        self.ch=channels(lmax,M);self.n=len(self.r);self.nc=len(self.ch)
        self.shape=(self.n,self.n,self.nc);self.size=np.prod(self.shape)
        t=grid.kinetic.tocsr()
        self.ptr=t.indptr.astype(np.int32);self.col=t.indices.astype(np.int32)
        self.tv=t.data.astype(complex)
        r=self.r;self.diag=np.empty(self.shape,complex,order='F')
        self.cut=np.ones(self.n)
        if cutoff_radii is not None:
            from tdse1d import cutoff
            self.cut=cutoff(r.real,*cutoff_radii)
        for c,((l,m),(ll,mm)) in enumerate(self.ch):
            u=-2*self.cut/r+l*(l+1)/(2*r*r);v=-2*self.cut/r+ll*(ll+1)/(2*r*r)
            self.diag[:,:,c]=u[:,None]+v[None,:]
        # min/max follow ordered real contour parameter, also on the ECS ray.
        i,j=np.indices((self.n,self.n));lo=r[np.minimum(i,j)];hi=r[np.maximum(i,j)]
        self.rad=np.asfortranarray(np.stack([lo**lam/hi**(lam+1)*self.cut[:,None]*self.cut[None,:] for lam in range(2*lmax+1)],axis=2))
        vp=[0];vc=[];vl=[];vv=[];dp=[0];dc=[];de=[];dv=[];dl=[]
        for a,bra in enumerate(self.ch):
            for b,ket in enumerate(self.ch):
                for lam in range(2*lmax+1):
                    value=coulomb(bra,ket,lam)
                    if abs(value)>1e-13:vc.append(b);vl.append(lam);vv.append(value)
                for electron in (0,1):
                    other=1-electron
                    if bra[other]!=ket[other]:continue
                    value=cartesian(bra[electron],ket[electron])
                    if np.max(np.abs(value))<1e-13:continue
                    dc.append(b);de.append(electron+1);dv.append(value)
                    l=bra[electron][0];ll=ket[electron][0];dl.append(l*(l+1)-ll*(ll+1))
            vp.append(len(vc));dp.append(len(dc))
        for name,values in [('vptr',vp),('vc',vc),('vl',vl),('dptr',dp),('dc',dc),('de',de)]:
            setattr(self,name,np.array(values,dtype=np.int32))
        self.vcoef=np.array(vv,complex)
        self.dcoef=np.asfortranarray(np.array(dv,complex).reshape(-1,3).T)
        self.ldiff=np.array(dl,dtype=float)
        self.exchange=np.array([self.ch.index((b,a)) for a,b in self.ch])

    def prepare_surface(self,radius):
        """Exact DISCRETE commutator [H,Theta(r2-R)], including velocity gauge.
        A hard projector gives compact support on FE elements touching the surface.
        """
        theta=(self.r.real>radius).astype(float)
        T=self.grid.kinetic.toarray()
        comm=T*(theta[None,:]-theta[:,None])
        self.surface_indices=np.flatnonzero(np.max(abs(comm),axis=1)>1e-14)
        if not len(self.surface_indices):raise ValueError('empty surface commutator')
        if np.any(abs(self.r[self.surface_indices].imag)>1e-12):raise ValueError('surface must be inside real grid')
        if np.any(self.cut[self.surface_indices]>1e-12):raise ValueError('Volkov surface stencil overlaps nonzero potential; increase R')
        self.surface_T=comm[self.surface_indices,:]
        self.surface_P=(1j*comm*(self.r[None,:]-self.r[:,None]))[self.surface_indices,:]
        return self.surface_indices

    def flux(self,psi,field):
        u=np.asarray(psi).reshape(self.shape,order='F')
        out=np.einsum('ij,caj->cia',self.surface_T,u.transpose(2,0,1)).transpose(2,1,0)
        for a in range(self.nc):
            for k in range(self.dptr[a],self.dptr[a+1]):
                if self.de[k]!=2:continue
                coef=np.dot(field,self.dcoef[:,k])
                if abs(coef)>1e-15:out[:,:,a]+=coef*(u[:,:,self.dc[k]]@self.surface_P.T)
        return out

    def apply(self,x,field=(0.,0.,0.),velocity=False):
        return tensor_apply(self,x,field,velocity)

    def operator(self,field=(0.,0.,0.),velocity=False):
        return LinearOperator((self.size,self.size),matvec=lambda x:self.apply(x,field,velocity),dtype=complex)

    def ground(self,tol=1e-10):
        if self.grid.ecs_angle:raise ValueError('ground eigsh requires a real grid')
        r=self.r.real;w=self.grid.weights.real
        u=r*np.exp(-1.6875*r)*np.sqrt(w)
        x=np.zeros(self.shape,order='F');x[:,:,self.ch.index(((0,0),(0,0)))]=u[:,None]*u[None,:]
        # Real symmetric interface avoids complex-ARPACK overhead for a real Hamiltonian.
        op=LinearOperator((self.size,self.size),matvec=lambda y:self.apply(y).real,dtype=float)
        e,v=eigsh(op,k=1,which='SA',v0=x.ravel(order='F'),tol=tol,ncv=35,maxiter=10000)
        psi=v[:,0].astype(complex);res=np.linalg.norm(self.apply(psi)-e[0]*psi)
        return float(e[0]),psi,float(res)

    def exchange_error(self,x):
        u=np.asarray(x).reshape(self.shape,order='F')
        return float(np.linalg.norm(u-u.transpose(1,0,2)[:,:,self.exchange]))

    def reference_sparse(self,field=(0.,0.,0.),velocity=False):
        """Independent Kronecker assembly for SMALL-grid module verification only."""
        n=self.n;T=self.grid.kinetic;I=eye(n,format='csr')
        radial=kron(I,T)+kron(T,I)
        h=kron(eye(self.nc),radial)+diags(self.diag.ravel(order='F'))
        for a in range(self.nc):
            for k in range(self.vptr[a],self.vptr[a+1]):
                A=csr_matrix(([self.vcoef[k]],([a],[self.vc[k]])),shape=(self.nc,self.nc))
                h+=kron(A,diags(self.rad[:,:,self.vl[k]].ravel(order='F')))
            for k in range(self.dptr[a],self.dptr[a+1]):
                coef=np.dot(field,self.dcoef[:,k])
                if abs(coef)<1e-15:continue
                R=diags(self.r)
                if velocity:R=1j*(T@R-R@T+diags(self.ldiff[k]/(2*self.r)))
                op=kron(I,R) if self.de[k]==1 else kron(R,I)
                A=csr_matrix(([coef],([a],[self.dc[k]])),shape=(self.nc,self.nc))
                h+=kron(A,op)
        if velocity:h+=eye(self.size)*np.dot(field,field)
        return h.tocsr()
