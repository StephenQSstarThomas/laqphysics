import numpy as np
from fedvr import make_grid
from helium3d import Helium
from surface3d import Ionic
from angular import cartesian

def test_sparse_ionic_momentum_equals_full_commutator_and_m_selection():
    h=Helium(make_grid([0,1,2,4],3,ecs_angle=.4,tail=6),1,M=None)
    full=Ionic(h);H0=full.h0.toarray();A=np.array([.03,.05,.07]);reference=H0.copy()+np.eye(len(H0))*np.dot(A,A)/2
    for axis in range(3):
        D=np.zeros_like(H0)
        for a,bra in enumerate(full.states):
            for b,ket in enumerate(full.states):
                D[a*h.n:(a+1)*h.n,b*h.n:(b+1)*h.n]=cartesian(bra,ket)[axis]*np.diag(h.r)
        reference+=A[axis]*1j*(H0@D-D@H0)
    np.testing.assert_allclose(full.matrix(A).toarray(),reference,atol=1e-12,rtol=2e-13)
    restricted=Ionic(h,{0})
    idx=np.concatenate([np.arange(full.index[lm]*h.n,(full.index[lm]+1)*h.n) for lm in restricted.states])
    np.testing.assert_allclose(restricted.matrix([0,0,.07]).toarray(),full.matrix([0,0,.07]).toarray()[np.ix_(idx,idx)],atol=1e-13)
