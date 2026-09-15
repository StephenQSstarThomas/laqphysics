"""Physical mixed length/velocity gauge with a compact radial gauge function.

psi_g=exp(i A.r f(r)) psi_V. With q=r(1-f), the one-electron terms are
H0 + A.i[H0,q rhat] + E.r f rhat
 + 1/2 [A^2(q/r)^2 + (q'^2-(q/r)^2)(A.rhat)^2].
The electron-electron potential is unchanged. These are independently projected
continuum operators, not a unitary relabeling of the discretized velocity matrix.
Outside the switch f=0, so the existing velocity-gauge surface flux is valid.
"""
import numpy as np
import torch
from scipy.sparse import csr_matrix,diags,bmat,kron,eye
from angular import cartesian
from surface3d import Ionic
from torch_backend import TorchHamiltonian

PAIRS=((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))

def profile(r,inner,outer):
    r=np.asarray(r);x=r.real
    if inner==outer==0:f=np.zeros(len(r));fp=f.copy()
    else:
        if not 0<=inner<outer:raise ValueError('mixed gauge needs 0 <= inner < outer')
        s=np.clip((x-inner)/(outer-inner),0,1)
        f=1-10*s**3+15*s**4-6*s**5
        fp=(-30*s*s+60*s**3-30*s**4)/(outer-inner)
    q=r*(1-f);qp=1-f-r*fp
    return q,r*f,(1-f)**2,qp**2-(1-f)**2

def parameters(avec,efunc,t):
    A=np.asarray(avec(t));E=np.asarray(efunc(t))
    quadratic=[A[a]*A[b]*(1 if a==b else 2) for a,b in PAIRS]
    return np.r_[A,E,quadratic]

def angular_squares(states):
    """Exact projected rhat_a rhat_b; includes the lmax+1 intermediate shell."""
    top=max(l for l,m in states)+1
    extended=[(l,m) for l in range(top+1) for m in range(-l,l+1)]
    N=np.array([[cartesian(a,b) for b in extended] for a in states]).transpose(2,0,1)
    squares=np.array([.5*(N[a]@N[b].conj().T+N[b]@N[a].conj().T) for a,b in PAIRS])
    if np.max(abs(sum(squares[:3])-np.eye(len(states))))>2e-12:raise RuntimeError('incomplete angular quadratic operator')
    return squares

def two_electron_squares(h):
    if hasattr(h,'tensor_matrix'):
        identity=eye(h.nc,format='csr');result=[]
        for electron in [1,2]:
            C={q:h.tensor_matrix(electron,2,q) for q in range(-2,3)}
            symmetric=(C[2]+C[-2])/np.sqrt(6)
            result.append([identity/3-C[0]/3+symmetric,identity/3-C[0]/3-symmetric,
                identity/3+2*C[0]/3,-1j*(C[2]-C[-2])/np.sqrt(6),
                (C[-1]-C[1])/np.sqrt(6),1j*(C[-1]+C[1])/np.sqrt(6)])
        return result
    base=getattr(h,'uncoupled',h);states=sorted({a for a,b in base.ch}|{b for a,b in base.ch})
    lookup={a:i for i,a in enumerate(states)};one=angular_squares(states);result=[]
    for electron in [0,1]:
        rows=[];cols=[];values=[]
        for a,bra in enumerate(base.ch):
            for b,ket in enumerate(base.ch):
                if bra[1-electron]!=ket[1-electron]:continue
                value=one[:,lookup[bra[electron]],lookup[ket[electron]]]
                if np.max(abs(value))>1e-14:rows.append(a);cols.append(b);values.append(value)
        values=np.array(values)
        matrices=[]
        for k in range(6):
            Q=csr_matrix((values[:,k],(rows,cols)),shape=(base.nc,base.nc));Q.eliminate_zeros()
            if hasattr(h,'transform'):Q=csr_matrix(h.transform.T@(Q@h.transform))
            Q.data[abs(Q.data)<1e-13]=0.;Q.eliminate_zeros()
            matrices.append(Q)
        result.append(matrices)
    return result

class MixedHamiltonian(TorchHamiltonian):
    a2_coefficient=0.  # A_a A_b are independent affine coefficients in CF4.
    def __init__(self,h,device='cuda:0',inner=2.,outer=4.,**kwargs):
        super().__init__(h,device,**kwargs)
        q,F,c0,c2=profile(h.r,inner,outer)
        self.has_momentum=bool(np.any(q!=0));self.has_dipole=bool(np.any(F!=0));self.has_quadratic=bool(np.any(c0!=0) or np.any(c2!=0))
        T=h.grid.kinetic.tocsr();B=T@diags(q)-diags(q)@T
        self.Bq=(torch.sparse_csr_tensor(torch.tensor(B.indptr.astype(np.int64),device=self.device),
            torch.tensor(B.indices.astype(np.int64),device=self.device),self.tensor(B.data),size=B.shape,device=self.device,check_invariants=True) if B.nnz else None)
        self.qr2=self.tensor(q/(h.r*h.r));self.F=self.tensor(F);self.c0=self.tensor(c0);self.c2=self.tensor(c2)
        self.quadratic=[]
        for matrices in two_electron_squares(h):
            pattern=sum(abs(Q) for Q in matrices).tocsr();pattern.eliminate_zeros();pattern.sort_indices()
            rows=np.repeat(np.arange(h.nc),np.diff(pattern.indptr));cols=pattern.indices
            data=np.array([np.asarray(Q[rows,cols]).ravel() for Q in matrices])
            self.quadratic.append((torch.tensor(pattern.indptr.astype(np.int64),device=self.device),
                torch.tensor(cols.astype(np.int64),device=self.device),self.tensor(data)))
        self.mixed_key=None

    def combine(self,coeff,pattern):
        ptr,col,values=pattern
        return torch.sparse_csr_tensor(ptr,col,coeff@values,size=(self.nc,self.nc),device=self.device,check_invariants=False)

    @torch.no_grad()
    def apply(self,x,field,velocity=True):
        if len(field)!=12:raise ValueError('mixed gauge requires A, E and six quadratic coefficients')
        key=tuple(float(v) for v in field)
        if self.mixed_key!=key:
            A=self.tensor(field[:3]);E=self.tensor(field[3:6]);Q=self.tensor(field[6:])
            self.CA=[];self.LA=[];self.CE=[];self.QA=[]
            for e,(ptr,col,C,L) in enumerate(self.edge_data):
                self.CA.append(self.combine(A,(ptr,col,C)));self.LA.append(self.combine(A,(ptr,col,L)))
                self.CE.append(self.combine(E,(ptr,col,C)));self.QA.append(self.combine(Q,self.quadratic[e]))
            self.mixed_key=key
        u=x.reshape(self.nc,self.n,self.n)
        out=super().apply(x,(0.,0.,0.),True).reshape_as(u)
        for electron,axis in [(1,(None,None,slice(None))),(2,(None,slice(None),None))]:
            e=electron-1
            if self.Bq is not None:out+=1j*self.mix(self.CA[e],self.radial(self.Bq,u,electron))
            if self.has_momentum:out+=.5j*self.mix(self.LA[e],u)*self.qr2[axis]
            if self.has_dipole:out+=self.mix(self.CE[e],u)*self.F[axis]
            if self.has_quadratic:
                out+=.5*sum(field[6:9])*self.c0[axis]*u
                out+=.5*self.mix(self.QA[e],u)*self.c2[axis]
        return out.reshape(-1)

class MixedIonic(Ionic):
    def __init__(self,h,magnetic_numbers=None,inner=2.,outer=4.):
        super().__init__(h,magnetic_numbers)
        q,F,c0,c2=profile(h.r,inner,outer);T=h.grid.kinetic.tocsr();B=T@diags(q)-diags(q)@T
        blocksP=[[[csr_matrix((self.n,self.n),dtype=complex) if a==b else None for b in range(self.na)] for a in range(self.na)] for _ in range(3)]
        blocksD=[[[csr_matrix((self.n,self.n),dtype=complex) if a==b else None for b in range(self.na)] for a in range(self.na)] for _ in range(3)]
        for a,(l,m) in enumerate(self.states):
            for b,(ll,mm) in enumerate(self.states):
                coeff=cartesian((l,m),(ll,mm))
                P=1j*(B+diags((l*(l+1)-ll*(ll+1))*q/(2*h.r*h.r)))
                for k in range(3):
                    if abs(coeff[k])>1e-14:blocksP[k][a][b]=coeff[k]*P;blocksD[k][a][b]=coeff[k]*diags(F)
        self.Pq=[bmat(x,format='csr') for x in blocksP];self.Df=[bmat(x,format='csr') for x in blocksD]
        square=angular_squares(self.states)
        self.Q=[.5*(kron(csr_matrix(S),diags(c2),format='csr')+(kron(eye(self.na),diags(c0),format='csr') if k<3 else csr_matrix(self.h0.shape,dtype=complex))) for k,S in enumerate(square)]

    def matrix(self,field):
        if len(field)!=12:raise ValueError('mixed ionic gauge requires 12 coefficients')
        H=self.h0.copy()
        for coefficient,operator in zip(field,[*self.Pq,*self.Df,*self.Q]):
            if coefficient:H=H+coefficient*operator
        return H.tocsr()
