import numpy as np
import pytest
torch=pytest.importorskip('torch')
from types import SimpleNamespace
from fedvr import make_grid
from surface3d import Ionic
from mixed_gauge import MixedIonic,parameters
from implicit import SparseCF4,cf4_step
from ionic_adjoint import IonicAdjoint

@pytest.mark.parametrize('mixed',[False,True])
def test_multi_rhs_adjoint_matches_independent_sparse_factorization(mixed):
    g=make_grid([0,.5,1,2,4,8],3,tail=8,ecs_angle=.4)
    h=SimpleNamespace(grid=g,r=g.r,n=len(g.r),cut=np.ones(len(g.r)),ch=[((l,m),(0,0)) for l in range(3) for m in range(-l,l+1)])
    ion=MixedIonic(h,inner=1,outer=4) if mixed else Ionic(h)
    A=lambda t:np.array([.03*np.cos(t),.02*np.sin(1.3*t),.01]);E=lambda t:np.array([.03*np.sin(t),-.026*np.cos(1.3*t),0])
    field=(lambda t:parameters(A,E,t)) if mixed else A
    chi=ion.dual_final_states([(1,0,0),(2,1,1)])
    reference=SparseCF4(lambda t:ion.matrix(field(t)).conj().T)
    engine=IonicAdjoint(ion,2,'cpu');state=engine.tensor(chi).reshape(-1)
    for i in range(8):
        t=.16-i*.02;chi=reference.step(chi,t,-.02)
        state,info=cf4_step(engine,state,lambda t:engine.coefficients(field(t)),t,-.02,0.,tol=1e-12)
        assert info['linear_residual']<1e-12 and state.dtype==torch.complex128
    np.testing.assert_allclose(engine.host(state),chi,atol=2e-11,rtol=2e-10)
