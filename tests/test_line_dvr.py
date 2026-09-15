import numpy as np
import pytest
from scipy.sparse import kron,eye,diags
from scipy.linalg import expm
from line_dvr import line_grid,SoftDVR

@pytest.mark.parametrize('angle',[0.,.5])
def test_line_tensor_has_regular_origin_and_matches_sparse(angle):
    h=SoftDVR(line_grid([0,1,2],order=2,tail=4,angle=angle),radii=(.5,1.5))
    rng=np.random.default_rng(793);x=rng.normal(size=h.size)+1j*rng.normal(size=h.size);A=.12
    native=h.apply(x,A);h.backend='sparse';sparse=h.apply(x,A)
    assert np.any(h.r==0) and np.all(np.isfinite(native))
    np.testing.assert_allclose(native,sparse,rtol=2e-13,atol=2e-12)

def test_separable_ionic_split_against_exact_two_electron_matrix():
    h=SoftDVR(line_grid([0,1,2],order=2,tail=3,angle=.4),radii=(.5,1.5))
    rng=np.random.default_rng(791);x=rng.normal(size=h.size)+1j*rng.normal(size=h.size);x/=np.linalg.norm(x)
    A=.1;one=h.ionic0+A*h.P+.5*A*A*eye(h.n)
    H=(kron(eye(h.n),one)+kron(one,eye(h.n))+diags(h.vee.ravel(order='F'))).toarray()
    errors=[]
    for dt in [.004,.002]:
        ref=expm(-1j*dt*H)@x;actual=h.split(x,A,dt);errors.append(np.linalg.norm(actual-ref))
    assert errors[1]<errors[0]/6 and errors[1]<1e-6
