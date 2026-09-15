import numpy as np
from fedvr import make_grid
from helium3d import Helium
from ground_s import GroundS

def test_coupled_S_is_exact_projection_and_isometry():
    h=Helium(make_grid([0,.5,1,2,4,8],4),2,M=None)
    s=GroundS(h);rng=np.random.default_rng(813);x=rng.normal(size=s.size)+1j*rng.normal(size=s.size)
    np.testing.assert_allclose(np.linalg.norm(s.expand(x)),np.linalg.norm(x),rtol=1e-14)
    np.testing.assert_allclose(s.expand(s.apply(x)),h.apply(s.expand(x)),rtol=3e-13,atol=3e-11)

def test_compressed_ground_matches_original_and_cache(tmp_path):
    h=Helium(make_grid([0,.5,1,2,4,8],4),2,M=0)
    E,x,r=h.ground(cache_dir=tmp_path);Eold,y,rold=h.ground(method='uncoupled')
    assert abs(E-Eold)<1e-9 and abs(abs(np.vdot(x,y))-1)<1e-9
    E2,z,r2=h.ground(cache_dir=tmp_path)
    np.testing.assert_array_equal(z,x);assert E==E2

def test_preconditioned_ground_matches_arpack_on_identical_grid():
    h=Helium(make_grid([0,.5,1,2,4,8],5),3,M=0)
    s=GroundS(h);E,x,r=s.solve(method='lobpcg');F,y,rr=s.solve(method='eigsh')
    assert abs(E-F)<1e-9 and abs(abs(np.vdot(x,y))-1)<1e-9
    assert r<1e-8 and rr<1e-8
