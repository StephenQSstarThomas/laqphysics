import numpy as np,pytest
from fedvr import make_grid
from helium3d import Helium
from coupled_angular import CoupledHelium

@pytest.mark.parametrize('angle',[0.,.4])
def test_bipolar_projected_hamiltonian_and_flux(angle):
    u=Helium(make_grid([0,.5,1,2,4,6,8],3,ecs_angle=angle,tail=6),2,M=0,cutoff_radii=[1,2])
    c=CoupledHelium(u,2);u.prepare_surface(5);c.prepare_surface(5)
    rng=np.random.default_rng(619);x=rng.normal(size=c.size)+1j*rng.normal(size=c.size);field=(0,0,.15)
    for velocity in [False,True]:
        np.testing.assert_allclose(c.apply(x,field,velocity),c.project(u.apply(c.expand(x),field,velocity)),rtol=3e-13,atol=3e-11)
    full=u.flux(c.expand(x),field)
    np.testing.assert_allclose(c.flux(x,field),np.tensordot(full,c.transform,axes=([-1],[0])),rtol=5e-13,atol=3e-11)
    if not angle:
        E,g,res=c.ground();EE,gg,rr=u.ground()
        assert abs(E-EE)<1e-9 and abs(abs(np.vdot(c.expand(g),gg))-1)<1e-9
