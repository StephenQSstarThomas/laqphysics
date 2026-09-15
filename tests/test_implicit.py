import numpy as np,pytest
torch=pytest.importorskip('torch')
from scipy.sparse import csr_matrix
from scipy.integrate import solve_ivp
from implicit import SparseCF4,SeparablePreconditioner,cf4_step
from fedvr import make_grid
from helium3d import Helium
from torch_backend import TorchHamiltonian

def test_cf4_sparse_fourth_order_and_backward_adjoint():
    rng=np.random.default_rng(103);A=rng.normal(size=(6,6));B=rng.normal(size=(6,6))
    H0=(A+A.T)/2-.03j*np.diag(np.arange(6));D=(B+B.T)/2
    h=lambda t:csr_matrix(H0+.3*np.cos(1.7*t)*D)
    exact=solve_ivp(lambda t,x:(-1j*h(t)@x.reshape(6,6)).ravel(),(0,.4),np.eye(6,dtype=complex).ravel(),method='DOP853',rtol=1e-12,atol=1e-14).y[:,-1].reshape(6,6)
    errors=[]
    for dt in [.1,.05]:
        prop=SparseCF4(h);U=np.eye(6,dtype=complex)
        for i in range(round(.4/dt)):U=prop.step(U,i*dt,dt)
        errors.append(np.linalg.norm(U-exact))
        back=SparseCF4(lambda t:h(t).conj().T);V=np.eye(6,dtype=complex)
        for i in range(round(.4/dt)):V=back.step(V,.4-i*dt,-dt)
        np.testing.assert_allclose(V,U.conj().T,atol=2e-13)
    assert errors[1]<errors[0]/12

def test_separable_resolvent_inverts_exact_ionic_sum():
    h=Helium(make_grid([0,.5,1,2,4],3,ecs_angle=.4,tail=6),1,M=None)
    op=TorchHamiltonian(h,'cpu');pre=SeparablePreconditioner(op)
    rng=np.random.default_rng(816);x=rng.normal(size=h.size)+1j*rng.normal(size=h.size)
    alpha=.02/(3+1j*np.sqrt(3));shift=2.9;y=op.host(pre.apply(op.state(x),alpha,shift)).reshape(h.shape,order='F')
    out=np.empty_like(y)
    for c,((l,m),(ll,mm)) in enumerate(h.ch):
        t=h.grid.kinetic.toarray();r=h.r
        h1=t+np.diag(-2*h.cut/r+l*(l+1)/(2*r*r));h2=t+np.diag(-2*h.cut/r+ll*(ll+1)/(2*r*r))
        out[:,:,c]=y[:,:,c]+1j*alpha*(h1@y[:,:,c]+y[:,:,c]@h2.T+shift*y[:,:,c])
    np.testing.assert_allclose(out.ravel(order='F'),x,atol=3e-10,rtol=3e-10)

@pytest.mark.parametrize('a2_coefficient',[.5,1.])
def test_torch_cf4_scalar_phase_and_time_ordering(a2_coefficient):
    rng=np.random.default_rng(677);z=rng.normal(size=(3,5,5));D=(z+z.transpose(0,2,1))/2
    H=np.diag(np.linspace(-2.9,1.1,5)).astype(complex)-.02j*np.diag(np.arange(5))
    class DenseEngine:
        diagonal=torch.tensor(np.diag(H),dtype=torch.complex128)
        def apply(self,x,field,velocity):
            matrix=H+np.einsum('a,aij->ij',field,D)+a2_coefficient*np.dot(field,field)*np.eye(5)
            return torch.tensor(matrix,dtype=torch.complex128)@x
    engine=DenseEngine();engine.a2_coefficient=a2_coefficient
    field=lambda t:np.array([.4*np.cos(1.7*t),.3*np.sin(2.1*t),.2*t])
    initial=rng.normal(size=5)+1j*rng.normal(size=5);initial/=np.linalg.norm(initial)
    exact=solve_ivp(lambda t,x:-1j*engine.apply(torch.tensor(x),field(t),True).numpy(),[0,.4],initial,method='DOP853',rtol=1e-12,atol=1e-14).y[:,-1]
    errors=[]
    for dt in [.1,.05]:
        x=torch.tensor(initial)
        for i in range(round(.4/dt)):x,info=cf4_step(engine,x,field,i*dt,dt,-2.9,tol=1e-13)
        errors.append(np.linalg.norm(x.numpy()-exact))
    assert errors[1]<errors[0]/12
