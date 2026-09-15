import numpy as np
from types import SimpleNamespace
from fedvr import make_grid
from surface3d import Ionic

def test_extended_bound_state_uses_analytic_radial_dual():
    g=make_grid([0,.5,1,2,4,8],8,ecs_angle=.5,tail=40,alpha=.8)
    h=SimpleNamespace(grid=g,r=g.r,n=len(g.r),cut=np.ones(len(g.r)),ch=[((l,m),(0,0)) for l in range(2) for m in range(-l,l+1)])
    ion=Ionic(h);ket=ion.final_states([(5,1,1)])[:,0];dual=ion.dual_final_states([(5,1,1)])[:,0]
    correct=np.vdot(dual,ket);incorrect=np.vdot(ket,ket)
    np.testing.assert_allclose(correct,1,atol=2e-7)
    assert abs(incorrect-1)>.01
