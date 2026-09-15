import numpy as np
import pytest
from scipy.linalg import expm
from angular import C,bound_dipole,two_photon
from pulses import Pulse
from esss import analytic,driven,finite_band,reduced_ion

def test_field_vector_potential_consistency():
    p=Pulse(cycles=8);t=np.linspace(.001,p.duration-.001,1000);h=1e-5
    np.testing.assert_allclose(-(p.vector(t+h)-p.vector(t-h))/(2*h),p.electric(t),atol=2e-10)
    assert abs(p.vector(p.duration+100))<1e-14

def test_ionic_dipole_and_selection_rules():
    # Analytic hydrogenic <2p0|z|1s> = 256/(243 sqrt(2)) / Z.
    np.testing.assert_allclose(bound_dipole((2,1,0),(1,0,0),0),256/(243*np.sqrt(2))/2,rtol=2e-12)
    assert C((1,1),1,1,(0,0))!=0
    assert C((1,0),1,1,(0,0))==0
    assert two_photon((2,1,1),(1,0,0),1,.75,nmax=5)==0
    for m in range(-2,3):
        assert two_photon((3,2,m),(2,1,1),-1,5/36,nmax=5)==0
    assert bound_dipole((3,2,2),(2,1,1),1)!=0
    assert bound_dipole((3,2,0),(2,1,1),-1)!=0

def test_esss_integral_matches_independent_ode():
    p=Pulse(cycles=12);e=np.linspace(.45,.75,41)
    a,_=analytic(e,p,nt=3001);b=driven(e,p,dt=.08)
    np.testing.assert_allclose(a,b,atol=5e-9,rtol=2e-7)

def test_finite_band_norm_conservation():
    e=np.linspace(.1,1.1,101);w=np.full(101,.01);w[[0,-1]]*=.5
    y,_=finite_band(e,w,Pulse(cycles=8),dg=.5213*(1+.1*(e-.6)))
    assert abs(np.vdot(y,y)-1)<2e-9

def test_no_signalling_unitary_and_local_dephasing():
    # An entangled joint state; amplitude columns label outgoing energy modes.
    rng=np.random.default_rng(653);a=rng.normal(size=(2,25))+1j*rng.normal(size=(2,25));a/=np.linalg.norm(a)
    U=expm(-.67j*np.array([[.4,1],[1,-.4]]));b=U@a
    np.testing.assert_allclose(a.T@a.conj(),b.T@b.conj(),atol=5e-16)
    rho=np.outer(a.ravel(),a.ravel().conj()).reshape(2,25,2,25)
    original=np.einsum('aiaj->ij',rho)
    for eta in (0,.2,1):
        # Exact CPTP dephasing, not ad hoc damping of wavefunction amplitudes.
        d=rho.copy();d[0,:,1,:]*=eta;d[1,:,0,:]*=eta
        np.testing.assert_allclose(np.einsum('aiaj->ij',d),original,atol=2e-16)
        assert np.linalg.eigvalsh(d.reshape(50,50)).min()>-1e-14

def test_entanglement_sector_normalization():
    a=np.diag([1,1]).astype(complex)/2
    rho,info=reduced_ion(a,np.ones(2))
    assert info['sector_probability']==.5
    np.testing.assert_allclose(info['entropy_bits'],1,atol=1e-14)
    np.testing.assert_allclose(info['negativity_pure'],.5,atol=1e-14)

def test_lab_to_rotating_frame_equations_6_to_8():
    # Derives the ODE convention directly from the explicit lab-frame RWA Hamiltonian.
    eg=-2.9;e1=-2.;e2=-.5;e=.63;omega=1.5;t=.37;f=.021;dg=.5213;d12=.4823
    lab=np.diag([eg,e1+e,e2+e]).astype(complex)
    for a,b,d in [(1,0,dg),(2,1,d12)]:
        lab[a,b]=f*d*np.exp(-1j*omega*t);lab[b,a]=np.conj(lab[a,b])
    alpha=np.array([eg,eg+omega,eg+2*omega]);R=np.diag(np.exp(-1j*alpha*t))
    rotated=R.conj().T@lab@R-np.diag(alpha)
    expected=np.array([[0,f*dg,0],[f*dg,e+e1-eg-omega,f*d12],[0,f*d12,e+e2-eg-2*omega]])
    np.testing.assert_allclose(rotated,expected,atol=1e-15)

def test_delayed_pulse_cep_and_complex_dipoles():
    e=np.linspace(.5,.7,17);p=Pulse(cycles=8,start=7.,cep=.31);q=Pulse(cycles=8)
    dg=.5*np.exp(.2j);d12=.48*np.exp(-.3j)
    a,_=analytic(e,p,dg=dg,d12=d12,nt=4001);b=driven(e,p,dg=dg,d12=d12,dt=.05)
    np.testing.assert_allclose(a,b,atol=2e-9,rtol=2e-7)
    base,_=analytic(e,q,dg=dg,d12=d12,nt=4001);phase=np.exp(1j*(p.omega*p.start-p.cep))
    np.testing.assert_allclose(a,base*np.array([phase,phase**2])[:,None],atol=2e-9,rtol=2e-7)

def test_lindblad_zero_dephasing_matches_wavefunction_with_detuning_and_phase():
    from open_system import propagate
    e=np.linspace(.35,.85,9);w=np.full(9,e[1]-e[0]);w[[0,-1]]*=.5
    pulse=Pulse(omega=1.49,cycles=3,start=2,cep=.4);dg=.5*np.exp(.2j);d12=.48*np.exp(-.3j)
    y,_=finite_band(e,w,pulse,dg=dg,d12=d12)
    rho=propagate(e,w,pulse,gamma=0,dg=dg,d12=d12,dt=.025)
    np.testing.assert_allclose(rho,np.outer(y,y.conj()),atol=3e-9,rtol=3e-7)
