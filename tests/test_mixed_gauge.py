import numpy as np
import pytest
torch=pytest.importorskip('torch')
from types import SimpleNamespace
from scipy.special import eval_legendre
from fedvr import make_grid
from helium3d import Helium
from surface3d import Ionic
from mixed_gauge import MixedHamiltonian,MixedIonic,parameters,profile,angular_squares

@pytest.mark.parametrize('inner,outer,velocity',[(0.,0.,True),(100.,101.,False)])
def test_uniform_gauge_limits_match_independent_native_operators(inner,outer,velocity):
    h=Helium(make_grid([0,.5,1,2,4,8],4),1,M=None)
    op=MixedHamiltonian(h,'cpu',inner,outer);rng=np.random.default_rng(511)
    x=rng.normal(size=h.size)+1j*rng.normal(size=h.size);A=np.array([.03,-.04,.05]);E=np.array([.02,.03,-.01])
    coeff=parameters(lambda t:A,lambda t:E,0)
    actual=op.host(op.apply(op.state(x),coeff));expected=h.apply(x,A if velocity else E,velocity)
    np.testing.assert_allclose(actual,expected,atol=3e-11,rtol=3e-12)

def test_quadratic_angular_operator_includes_missing_intermediate_shell():
    states=[(l,m) for l in range(3) for m in range(-l,l+1)]
    squares=angular_squares(states)
    np.testing.assert_allclose(sum(squares[:3]),np.eye(len(states)),atol=1e-14)
    # Analytic <p0|cos^2 theta|p0> = 3/5. Truncating the intermediate d shell
    # would instead give only the s-shell contribution 1/3.
    assert abs(squares[2,states.index((1,0)),states.index((1,0))]-.6)<1e-14

def test_one_electron_continuum_gauge_identity_converges_radially():
    errors=[];z,w=np.polynomial.legendre.leggauss(48);A=.03;E=.02;lmax=6
    harmonics=np.array([np.sqrt((2*l+1)/2)*eval_legendre(l,z) for l in range(lmax+1)])
    for order in [4,6,8]:
        g=make_grid([0,.5,1,2,3,4,6,8,12],order)
        dummy=SimpleNamespace(grid=g,n=len(g.r),r=g.r,cut=np.ones(len(g.r)),ch=[((l,0),(0,0)) for l in range(lmax+1)])
        ion=Ionic(dummy);mixed=MixedIonic(dummy,inner=2,outer=4);psi=ion.final_states([(1,0,0)])[:,0]
        q,F,c0,c2=profile(g.r,2,4)
        U=np.einsum('lz,jz,rz,z->rlj',harmonics,harmonics,np.exp(1j*A*F[:,None]*z[None,:]),w)
        transform=lambda x:np.einsum('rlj,jr->lr',U,x.reshape(lmax+1,-1)).reshape(-1)
        v=transform(psi);coeff=parameters(lambda t:np.array([0,0,A]),lambda t:np.array([0,0,E]),0)
        defect=mixed.matrix(coeff)@v-transform(ion.matrix([0,0,A])@psi)-E*(mixed.Df[2]@v)
        errors.append(np.linalg.norm(defect))
        H=mixed.matrix(coeff);assert np.max(abs((H-H.conj().T).data),initial=0)<1e-11
    assert errors[1]<errors[0]/3 and errors[2]<errors[1]/3,errors
    assert errors[-1]<2e-7,errors

def test_two_electron_mixed_terms_equal_independent_ionic_kronecker_sum():
    h=Helium(make_grid([0,1,2,4,6],3),1,M=None)
    ion=MixedIonic(h,inner=1,outer=4);op=MixedHamiltonian(h,'cpu',inner=1,outer=4)
    coeff=parameters(lambda t:np.array([.02,.03,-.01]),lambda t:np.array([.01,-.02,.04]),0)
    rng=np.random.default_rng(215);x=rng.normal(size=h.size)+1j*rng.normal(size=h.size)
    na=len(ion.states);X=x.reshape(na,na,h.n,h.n).transpose(0,3,1,2).reshape(na*h.n,na*h.n)
    delta=(ion.matrix(coeff)-ion.h0).toarray();reference=delta@X+X@delta.T
    reference=reference.reshape(na,h.n,na,h.n).transpose(0,2,3,1).reshape(-1)+h.apply(x)
    np.testing.assert_allclose(op.host(op.apply(op.state(x),coeff)),reference,atol=2e-11,rtol=2e-12)

def test_coupled_mixed_gauge_is_projection_of_product_basis():
    from coupled_angular import CoupledHelium
    h=Helium(make_grid([0,1,2,4,6],3),2,M=0);c=CoupledHelium(h,2)
    full=MixedHamiltonian(h,'cpu',inner=1,outer=4);small=MixedHamiltonian(c,'cpu',inner=1,outer=4)
    rng=np.random.default_rng(882);x=rng.normal(size=c.size)+1j*rng.normal(size=c.size)
    coeff=parameters(lambda t:np.array([0,0,.03]),lambda t:np.array([0,0,.02]),0)
    expected=c.project(full.host(full.apply(full.state(c.expand(x)),coeff)))
    np.testing.assert_allclose(small.host(small.apply(small.state(x),coeff)),expected,atol=3e-11,rtol=2e-12)
