"""Arnoldi exponential action, also valid for non-Hermitian ECS Hamiltonians."""
import numpy as np
from scipy.linalg import expm
from scipy.linalg.blas import zgemv

def exponential_action(apply,x,dt,tol=1e-11,maxdim=60,trace=None,cache_conjugates=True,blas_orthogonalization=True):
    beta=np.linalg.norm(x)
    if beta==0:return x.copy(),{'dimension':0,'residual_estimate':0.}
    Q=np.empty((len(x),maxdim+1),complex,order='F');H=np.zeros((maxdim+1,maxdim),complex)
    Qc=np.empty_like(Q,order='F') if cache_conjugates and not blas_orthogonalization else None
    Q[:,0]=x/beta
    if Qc is not None:Qc[:,0]=Q[:,0].conj()
    for j in range(maxdim):
        v=apply(Q[:,j])
        if trace is not None:trace(f'Hq{j:03d}',v.copy())
        # Twice modified Gram--Schmidt, avoids ghost eigenvalues/loss of orthogonality.
        for repeat in range(2):
            if blas_orthogonalization:
                # BLAS conjugate-transpose avoids copying the entire growing basis.
                projection=zgemv(1.,Q[:,:j+1],v,trans=2)
                H[:j+1,j]+=projection
                v=zgemv(-1.,Q[:,:j+1],projection,beta=1.,y=v,overwrite_y=1)
            else:
                projection=(Qc[:,:j+1].T if Qc is not None else Q[:,:j+1].conj().T)@v
                H[:j+1,j]+=projection;v-=Q[:,:j+1]@projection
        H[j+1,j]=np.linalg.norm(v)
        happy=abs(H[j+1,j])<1e-13
        if happy or j>=3 and ((j+1)%4==0 or j+1==maxdim):
            m=j+1;c=expm(-1j*dt*H[:m,:m])[:,0]*beta
            err=abs(dt*H[m,m-1]*c[-1])
            if happy or err<tol:
                out=Q[:,:m]@c
                if trace is not None:
                    trace('arnoldi_Q',Q[:,:m].copy());trace('arnoldi_H',H[:m,:m].copy());trace('output',out.copy())
                return out,{'dimension':m,'residual_estimate':float(err)}
        if j+1<maxdim:
            Q[:,j+1]=v/H[j+1,j]
            if Qc is not None:Qc[:,j+1]=Q[:,j+1].conj()
    raise RuntimeError(f'Arnoldi did not converge: dim={maxdim}, dt={dt}, estimate={err:.3e}; reduce dt')

def step(apply_at,x,t,dt,tol=1e-11,maxdim=60,depth=0,trace=None):
    """Midpoint Hamiltonian (O(dt^2) for driven problems); halve on Krylov failure."""
    try:
        return exponential_action(lambda y:apply_at(t+dt/2,y),x,dt,tol,maxdim,trace)
    except RuntimeError:
        if depth>=8:raise
        y,a=step(apply_at,x,t,dt/2,tol/2,maxdim,depth+1,trace)
        y,b=step(apply_at,y,t+dt/2,dt/2,tol/2,maxdim,depth+1,trace)
        return y,{'dimension':max(a['dimension'],b['dimension']),
                  'residual_estimate':a['residual_estimate']+b['residual_estimate'],'subdivided':True}
