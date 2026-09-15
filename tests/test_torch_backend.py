import os,numpy as np,pytest
torch=pytest.importorskip('torch')
from fedvr import make_grid
from helium3d import Helium
from torch_backend import TorchHamiltonian,exponential_action
from propagate import exponential_action as reference_exp

@pytest.mark.parametrize('angle',[0.,.4])
@pytest.mark.parametrize('coulomb_backend',['sparse','blocks'])
def test_torch_against_fortran_all_axes_flux_and_step(angle,coulomb_backend):
    device=os.environ.get('HELIUM_TEST_DEVICE','cpu')
    h=Helium(make_grid([0,.5,1,2,4,6,8],3,ecs_angle=angle,tail=8),1,M=None,cutoff_radii=[1,2]);h.prepare_surface(5)
    op=TorchHamiltonian(h,device,coulomb_backend=coulomb_backend);rng=np.random.default_rng(386)
    x=rng.normal(size=h.size)+1j*rng.normal(size=h.size);x/=np.linalg.norm(x);f=(.13,.23,.34)
    for velocity in (False,True):
        np.testing.assert_allclose(op.host(op.apply(op.state(x),f,velocity)),h.apply(x,f,velocity),atol=2e-12,rtol=2e-13)
    np.testing.assert_allclose(op.flux(op.state(x),f),h.flux(x,f),atol=2e-12,rtol=2e-12)
    y,_=exponential_action(lambda y:op.apply(y,f),op.state(x),.02,tol=1e-11)
    z,_=reference_exp(lambda y:h.apply(y,f,True),x,.02,tol=1e-11)
    np.testing.assert_allclose(op.host(y),z,atol=2e-11,rtol=2e-11)
