import numpy as np
import pytest
from fedvr import make_grid
from helium3d import Helium
from bipolar import BipolarHelium

@pytest.mark.parametrize('M,L,natural',[(0,2,True),(None,2,False),(None,4,False)])
def test_direct_bipolar_algebra_against_product_projection(M,L,natural):
    grid=make_grid([0,1,2,4,6,8],3,tail=6,ecs_angle=.3)
    full=Helium(grid,2,M=M,cutoff_radii=[1,2]);coupled=BipolarHelium(grid,2,L,M,cutoff_radii=[1,2],natural_parity=natural)
    rng=np.random.default_rng(484);x=rng.normal(size=coupled.size)+1j*rng.normal(size=coupled.size)
    U=coupled.transform;np.testing.assert_allclose((U.T@U).toarray(),np.eye(coupled.nc),atol=1e-14)
    fields=[[0,0,0],[0,0,.07]] if M==0 else [[0,0,0],[.03,-.04,.07]]
    for field in fields:
        for velocity in [False,True]:
            expected=coupled.project(full.apply(coupled.expand(x),field,velocity))
            np.testing.assert_allclose(coupled.apply(x,field,velocity),expected,atol=3e-11,rtol=3e-12)
    full.prepare_surface(5);coupled.prepare_surface(5)
    f=full.flux(coupled.expand(x),fields[-1]);expected=(U.T@f.reshape(-1,full.nc).T).T.reshape(coupled.n,len(coupled.surface_indices),coupled.nc)
    np.testing.assert_allclose(coupled.flux(x,fields[-1]),expected,atol=3e-11,rtol=3e-12)

def test_all_M_basis_retains_unnatural_parity_and_singlet_ground():
    grid=make_grid([0,.5,1,2,4,8],4)
    h=BipolarHelium(grid,2,2,M=None);full=Helium(grid,2,M=None)
    assert (1,1,1,1) in h.index
    E,x,r=h.ground();F,y,s=full.ground()
    assert abs(E-F)<1e-9 and abs(abs(np.vdot(h.expand(x),y))-1)<1e-9
    assert h.exchange_error(x)<1e-10

def test_bipolar_device_and_quadratic_operators_match_product_basis():
    from torch_backend import TorchHamiltonian
    from mixed_gauge import MixedHamiltonian,parameters
    grid=make_grid([0,1,2,4,6,8],3,tail=6,ecs_angle=.3)
    full=Helium(grid,2,M=None,cutoff_radii=[1,2]);h=BipolarHelium(grid,2,2,M=None,cutoff_radii=[1,2])
    rng=np.random.default_rng(908);x=rng.normal(size=h.size)+1j*rng.normal(size=h.size)
    op=TorchHamiltonian(h,'cpu',coulomb_backend='blocks')
    np.testing.assert_allclose(op.host(op.apply(op.state(x),[.03,.02,.01])),h.apply(x,[.03,.02,.01],True),atol=3e-11,rtol=3e-12)
    mixed=MixedHamiltonian(h,'cpu',inner=1,outer=3);large=MixedHamiltonian(full,'cpu',inner=1,outer=3)
    coeff=parameters(lambda t:np.array([.03,.02,.01]),lambda t:np.array([-.02,.01,.04]),0)
    expected=h.project(large.host(large.apply(large.state(h.expand(x)),coeff)))
    np.testing.assert_allclose(mixed.host(mixed.apply(mixed.state(x),coeff)),expected,atol=3e-11,rtol=3e-12)

def test_sparse_bipolar_surface_projection_matches_explicit_expansion():
    from surface3d import extract
    from surface_projection import project,integrate
    full=Helium(make_grid([0,1,2,4,6,8],3,tail=6,ecs_angle=.3),2,M=None,cutoff_radii=[1,2]);full.prepare_surface(5)
    h=BipolarHelium(full.grid,2,2,M=None,cutoff_radii=[1,2]);h.prepare_surface(5)
    t=np.linspace(0,.2,5);A=lambda s:np.array([.03*np.sin(np.pi*s/.2)**2,.01*np.sin(2*np.pi*s/.2),0])
    rng=np.random.default_rng(227);f=rng.normal(size=(len(t),h.n,len(h.surface_indices),h.nc))+1j*rng.normal(size=(len(t),h.n,len(h.surface_indices),h.nc))
    labels=[(1,0,0),(2,1,-1),(2,1,0),(2,1,1)];E=np.linspace(.2,.9,5);angles=(np.linspace(.1,3,9),np.linspace(0,6,9))
    expected,_=extract(full,f,t,A,labels,E,angles,t[1],angular_transform=h.transform.toarray(),propagator='cf4')
    q,lm,_=project(full,f,t,A,labels,t[1],angular_transform=h.transform,propagator='cf4')
    actual,_=integrate(q,lm,full.r[full.surface_indices].real,full.grid.weights[full.surface_indices].real,t,A,E,angles,t[1])
    np.testing.assert_allclose(actual,expected,atol=3e-12,rtol=3e-12)
