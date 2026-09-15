import numpy as np
from scipy.linalg import eigvalsh
from fedvr import make_grid
from ground_s import GroundS
from helium3d import Helium

def test_uniform_order_list_is_identical_and_variable_bridge_is_symmetric():
    a=make_grid([0,.5,1,2,4],4,tail=8,ecs_angle=.4);b=make_grid([0,.5,1,2,4],[4,4,4,4],tail=8,ecs_angle=.4)
    np.testing.assert_array_equal(a.r,b.r);np.testing.assert_array_equal(a.kinetic.toarray(),b.kinetic.toarray())
    c=make_grid([0,.5,1,2,4,8,16,32],[8,8,8,6,6,6,6])
    H=c.kinetic.toarray()+np.diag(-2/c.r)
    np.testing.assert_allclose(H,H.T,atol=1e-12)
    assert abs(eigvalsh(H)[0]+2)<1e-8

def test_ground_only_constructor_matches_full_parent():
    g=make_grid([0,.5,1,2,4,8],4)
    h=Helium(g,2);s=GroundS.from_grid(g,2)
    E,x,r=s.solve();F,y,rr=h.ground()
    assert abs(E-F)<1e-9 and abs(np.linalg.norm(x)-1)<1e-12
