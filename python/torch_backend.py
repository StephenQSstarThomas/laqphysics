"""Optional complex128 tensor backend, independently checked against Fortran.

CUDA uses sparse radial and angular matrices (with a dense angular comparison option); no full
six-dimensional Hamiltonian is assembled. State ordering matches the Fortran ABI.
"""
import numpy as np
import torch
from scipy.linalg import expm
from scipy.sparse import diags,csr_matrix

class TorchHamiltonian:
    def __init__(self,h,device='cuda:0',angular_sparse=True,coulomb_backend='sparse'):
        self.h=h;self.device=torch.device(device);self.n=h.n;self.nc=h.nc;self.size=h.size
        self.dtype=torch.complex128
        def tensor(x):return torch.as_tensor(np.array(x,copy=True),dtype=self.dtype,device=self.device)
        self.tensor=tensor
        self.angular_sparse=angular_sparse
        T=h.grid.kinetic.tocsr();B=T@diags(h.r)-diags(h.r)@T
        def sparse(A):
            A=A.tocsr()
            return torch.sparse_csr_tensor(torch.as_tensor(A.indptr.astype(np.int64),device=self.device),
                torch.as_tensor(A.indices.astype(np.int64),device=self.device),tensor(A.data),size=A.shape,device=self.device,check_invariants=True)
        self.T=sparse(T);self.B=sparse(B);self.r=tensor(h.r)
        self.diag=tensor(h.diag.ravel(order='F').reshape(h.nc,h.n,h.n))
        diagonal=h.diag.copy()
        td=h.grid.kinetic.diagonal()
        diagonal+=td[:,None,None]+td[None,:,None]
        for a in range(h.nc):
            for j in range(h.vptr[a],h.vptr[a+1]):
                if h.vc[j]==a:diagonal[:,:,a]+=h.vcoef[j]*h.rad[:,:,h.vl[j]]
        self.diagonal=tensor(diagonal.ravel(order='F'))
        self.V=[];self.rad=[];potential_matrices=[]
        for lam in range(h.rad.shape[2]):
            A=np.zeros((h.nc,h.nc),complex)
            for a in range(h.nc):
                for j in range(h.vptr[a],h.vptr[a+1]):
                    if h.vl[j]==lam:A[a,h.vc[j]]+=h.vcoef[j]
            self.V.append(sparse(csr_matrix(A)) if angular_sparse else tensor(A));self.rad.append(tensor(h.rad[:,:,lam].T));potential_matrices.append(A)
        if coulomb_backend not in ('sparse','blocks'):raise ValueError('unknown Coulomb contraction')
        self.coulomb_backend=coulomb_backend;self.coulomb_blocks=[];self.coulomb_block_bytes=0
        if coulomb_backend=='blocks':
            # Coulomb is diagonal in total L (coupled basis), or conserves M and
            # parity (product basis). Sum multipoles ONCE per radial node pair.
            keys=[L for l,j,L in h.basis] if hasattr(h,'basis') else [(m+mm,(l+ll)%2) for (l,m),(ll,mm) in h.ch]
            groups=[np.array([i for i,k in enumerate(keys) if k==key]) for key in sorted(set(keys))]
            matrices=np.array(potential_matrices);radial=torch.stack(self.rad).reshape(len(self.rad),-1).T.contiguous()
            for group in groups:
                outside=np.array([i for i in range(h.nc) if i not in group])
                if len(outside) and np.max(abs(matrices[:,group[:,None],outside]))>1e-12:
                    raise ValueError('Coulomb violates requested symmetry blocks')
                block=matrices[:,group[:,None],group]
                W=(radial@tensor(block.reshape(len(self.rad),-1))).reshape(h.n*h.n,len(group),len(group))
                self.coulomb_blocks.append((torch.as_tensor(group,device=self.device),W))
                self.coulomb_block_bytes+=W.numel()*W.element_size()
        C=np.zeros((2,3,h.nc,h.nc),complex);L=np.zeros_like(C)
        for a in range(h.nc):
            for j in range(h.dptr[a],h.dptr[a+1]):
                b=h.dc[j];e=h.de[j]-1
                C[e,:,a,b]+=h.dcoef[:,j];L[e,:,a,b]+=h.dcoef[:,j]*h.ldiff[j]
        self.C=tensor(C);self.L=tensor(L)
        self.edge_data=[]
        for e in range(2):
            mask=np.max(abs(C[e]),axis=0)>1e-14
            pattern=csr_matrix(mask);rows=np.repeat(np.arange(h.nc),np.diff(pattern.indptr));cols=pattern.indices
            self.edge_data.append((torch.as_tensor(pattern.indptr.astype(np.int64),device=self.device),
                torch.as_tensor(cols.astype(np.int64),device=self.device),tensor(C[e,:,rows,cols].T),tensor(L[e,:,rows,cols].T)))
        self._field=None
        if hasattr(h,'surface_indices'):
            self.J=tensor(h.surface_T);self.JP=tensor(h.surface_P)

    def state(self,x):return self.tensor(np.asarray(x).ravel())
    def host(self,x):return x.detach().cpu().numpy()
    def synchronize(self):
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)

    def fields(self,field):
        key=tuple(float(x) for x in field)
        if self._field!=key:
            f=self.tensor(key)
            if self.angular_sparse:
                self.Cf=[];self.Lf=[]
                for ptr,col,C,L in self.edge_data:
                    for target,values in [(self.Cf,f@C),(self.Lf,f@L)]:
                        target.append(torch.sparse_csr_tensor(ptr,col,values,size=(self.nc,self.nc),device=self.device,check_invariants=False))
            else:
                self.Cf=torch.einsum('a,eaij->eij',f,self.C)
                self.Lf=torch.einsum('a,eaij->eij',f,self.L)
            self._field=key
        return key

    def radial(self,A,u,electron):
        # u[c,j,i] represents the Fortran u[i,j,c].
        if electron==1:
            x=u.permute(2,0,1).reshape(self.n,-1).contiguous()
            return torch.sparse.mm(A,x).reshape(self.n,self.nc,self.n).permute(1,2,0)
        x=u.permute(1,0,2).reshape(self.n,-1).contiguous()
        return torch.sparse.mm(A,x).reshape(self.n,self.nc,self.n).permute(1,0,2)

    def mix(self,A,u):
        flat=u.reshape(self.nc,-1).contiguous()
        return (torch.sparse.mm(A,flat) if A.layout==torch.sparse_csr else A@flat).reshape_as(u)

    @torch.no_grad()
    def apply(self,x,field=(0.,0.,0.),velocity=True):
        u=x.reshape(self.nc,self.n,self.n)
        out=self.diag*u+self.radial(self.T,u,1)+self.radial(self.T,u,2)
        if self.coulomb_backend=='blocks':
            for indices,W in self.coulomb_blocks:
                v=u[indices].permute(1,2,0).reshape(self.n*self.n,len(indices),1).contiguous()
                y=(W@v).reshape(self.n,self.n,len(indices)).permute(2,0,1)
                out[indices]=out[indices]+y
        else:
            for A,R in zip(self.V,self.rad):out+=self.mix(A,u)*R[None,:,:]
        f=self.fields(field)
        if any(abs(a)>1e-30 for a in f):
            if velocity:
                out+=1j*self.mix(self.Cf[0],self.radial(self.B,u,1))
                out+=1j*self.mix(self.Cf[1],self.radial(self.B,u,2))
                out+=.5j*self.mix(self.Lf[0],u)/self.r[None,None,:]
                out+=.5j*self.mix(self.Lf[1],u)/self.r[None,:,None]
                out+=sum(a*a for a in f)*u
            else:
                out+=self.mix(self.Cf[0],u)*self.r[None,None,:]
                out+=self.mix(self.Cf[1],u)*self.r[None,:,None]
        return out.reshape(-1)

    @torch.no_grad()
    def flux(self,x,field):
        self.fields(field);u=x.reshape(self.nc,self.n,self.n)
        out=self.J@u
        if any(abs(a)>1e-30 for a in field):out+=self.mix(self.Cf[1],self.JP@u)
        return self.host(out.permute(2,1,0))

@torch.no_grad()
def exponential_action(apply,x,dt,tol=1e-10,maxdim=48):
    beta=torch.linalg.vector_norm(x).item()
    if beta==0:return x.clone(),{'dimension':0,'residual_estimate':0.}
    # Columns are contiguous: transpose of a (maxdim+1, size) allocation.
    Q=torch.empty((maxdim+1,x.numel()),dtype=x.dtype,device=x.device).T
    H=torch.zeros((maxdim+1,maxdim),dtype=x.dtype,device=x.device);Q[:,0]=x/beta
    for j in range(maxdim):
        v=apply(Q[:,j].contiguous())
        for repeat in range(2):
            z=Q[:,:j+1].mH@v;H[:j+1,j]+=z;v-=Q[:,:j+1]@z
        norm=torch.linalg.vector_norm(v);H[j+1,j]=norm;norm_value=norm.item();happy=norm_value<1e-13
        if happy or j>=3 and ((j+1)%4==0 or j+1==maxdim):
            m=j+1;small=H[:m,:m].cpu().numpy();c=expm(-1j*dt*small)[:,0]*beta
            err=abs(dt*norm_value*c[-1])
            if happy or err<tol:
                return Q[:,:m]@torch.as_tensor(c,dtype=x.dtype,device=x.device),{'dimension':m,'residual_estimate':float(err)}
        if j+1<maxdim:Q[:,j+1]=v/norm
    raise RuntimeError(f'GPU Arnoldi did not converge: dim={maxdim}, dt={dt}, estimate={err:.3e}')

def step(apply_at,x,t,dt,tol=1e-10,maxdim=48,depth=0):
    try:return exponential_action(lambda y:apply_at(t+dt/2,y),x,dt,tol,maxdim)
    except RuntimeError as error:
        if 'Arnoldi did not converge' not in str(error) or depth>=8:raise
        y,a=step(apply_at,x,t,dt/2,tol/2,maxdim,depth+1)
        y,b=step(apply_at,y,t+dt/2,dt/2,tol/2,maxdim,depth+1)
        return y,{'dimension':max(a['dimension'],b['dimension']),
                  'residual_estimate':a['residual_estimate']+b['residual_estimate'],'subdivided':True}
