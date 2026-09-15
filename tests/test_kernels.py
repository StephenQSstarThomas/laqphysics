import numpy as np
import pytest
from scipy.fft import fft2,ifft2,fftfreq
from scipy.linalg import expm
from native import split_stage
from fedvr import make_grid,derivative
from helium3d import Helium
from propagate import exponential_action

def test_fft_split_every_module():
    rng=np.random.default_rng(842);n=16;dt=.03;A=.23
    psi=np.asfortranarray(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))
    ref=psi.copy();v=rng.normal(size=(n,n))-.01j;p=2*np.pi*fftfreq(n,.3)
    for stage in (1,2,3):
        split_stage(psi,v,p,dt,A,stage)
        if stage in (1,3):ref*=np.exp(-1j*dt*v/2)
        else:ref=ifft2(fft2(ref)*np.exp(-1j*dt*((p[:,None]+A)**2+(p[None,:]+A)**2)/2))
        np.testing.assert_allclose(psi,ref,atol=2e-13,rtol=2e-13)

def test_fft_rejects_non_power_of_two():
    with pytest.raises(ValueError):split_stage(np.zeros((12,12),complex,order='F'),np.zeros((12,12)),np.zeros(12),.1,0,2)

def test_cached_operator_matches_uncached_stages():
    from tdse1d import Model
    from pulses import Pulse
    rng=np.random.default_rng(988);m=Model(32,10);p=Pulse(cycles=2)
    a=np.asfortranarray(rng.normal(size=(32,32))+1j*rng.normal(size=(32,32)));b=a.copy(order='F')
    m.step(a,.5,.01,p)
    m.step(b,.5,.01,p,trace=lambda stage,psi:None)
    np.testing.assert_allclose(a,b,atol=2e-13,rtol=2e-13)

@pytest.mark.parametrize('angle',[0.,.3])
@pytest.mark.parametrize('velocity',[False,True])
def test_tensor_against_independent_sparse(angle,velocity):
    g=make_grid([0,.5,1,2,4],3,ecs_angle=angle,tail=5 if angle else 0)
    h=Helium(g,1,M=None);f=np.array([.13,.23,.34])
    rng=np.random.default_rng(49);x=rng.normal(size=h.size)+1j*rng.normal(size=h.size)
    ref=h.reference_sparse(f,velocity)
    np.testing.assert_allclose(h.apply(x,f,velocity),ref@x,rtol=2e-13,atol=2e-12)
    if not angle:np.testing.assert_allclose(ref.toarray(),ref.toarray().conj().T,atol=2e-12)

def test_arnoldi_against_dense_exponential():
    rng=np.random.default_rng(762);A=rng.normal(size=(20,20))+1j*rng.normal(size=(20,20))
    H=(A+A.conj().T)/2-.1j*np.diag(np.arange(20))
    x=rng.normal(size=20)+1j*rng.normal(size=20)
    y,info=exponential_action(lambda z:H@z,x,.3,tol=1e-12)
    np.testing.assert_allclose(y,expm(-.3j*H)@x,atol=2e-11,rtol=2e-11)

def test_dvr_hydrogen_analytic_energies():
    from scipy.linalg import eigvalsh
    g=make_grid([0,.5,1,2,4,8,16,32],8)
    for l in (0,1,2):
        h=g.kinetic.toarray()+np.diag(-2/g.r+l*(l+1)/(2*g.r*g.r))
        e=eigvalsh(h)
        np.testing.assert_allclose(e[0],-2/(l+1)**2,atol=2e-7)

def test_polynomial_derivative():
    x=np.array([-1.,-.4,.2,.8,1.]);D=derivative(x)
    np.testing.assert_allclose(D@(x**4),4*x**3,atol=1e-13)

def test_high_order_irecs_has_no_exponentially_growing_free_modes():
    from scipy.linalg import eigvals
    g=make_grid([0,.5,1,2,4,8,16,20],6,ecs_angle=.5,tail=40,alpha=.8)
    e=eigvals(g.kinetic.toarray())
    assert e.imag.max()<1e-8

def test_irecs_bridge_uses_real_element_mass_for_interior_probability():
    g=make_grid([0,1,2,4,8],6,ecs_angle=.5,tail=20)
    inside=g.r.real<=8;physical=np.zeros(len(g.r),complex)
    physical[inside]=np.sin(np.pi*g.r[inside].real/16)
    coefficients=physical*np.sqrt(g.weights)
    probability=np.sum(abs(coefficients)**2*g.interior_weights/abs(g.weights))
    np.testing.assert_allclose(probability,4,atol=1e-8)
    assert abs(np.sum(abs(coefficients[inside])**2)-probability)>1e-3
