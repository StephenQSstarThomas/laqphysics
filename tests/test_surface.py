import numpy as np
from scipy.integrate import simpson
from fedvr import make_grid
from helium3d import Helium

def test_free_packet_surface_amplitude_both_directions():
    # Analytic free TDSE wave packet, independently fixes normal, phase and k Jacobian.
    sigma=3.;R=25.;t=np.linspace(0,180,18001);k=np.linspace(.6,1.6,61)
    for sign in (1,-1):
        k0=sign*1.1;kk=sign*k;b=np.zeros(len(k),complex)
        for x,normal in [(-R,-1),(R,1)]:
            z=1+1j*t/sigma**2
            psi=(np.pi*sigma**2)**(-.25)/np.sqrt(z)*np.exp(-(x-k0*t)**2/(2*sigma**2*z)+1j*k0*x-.5j*k0*k0*t)
            dp=(-(x-k0*t)/(sigma**2*z)+1j*k0)*psi
            chi=np.exp(-1j*kk[:,None]*x+.5j*kk[:,None]**2*t)/np.sqrt(2*np.pi)
            b+=normal*simpson(chi*(kk[:,None]*psi/2-.5j*dp),x=t,axis=1)
        expected=np.sqrt(sigma)/np.pi**.25*np.exp(-sigma**2*(kk-k0)**2/2)
        # At finite T a slow part remains inside. The exact identity includes this
        # endpoint term; assuming the finite-time flux is the entire spectrum is wrong.
        from numpy.polynomial.legendre import leggauss
        x,w=leggauss(256);x=x*R;w=w*R;T=t[-1];z=1+1j*T/sigma**2
        psi=(np.pi*sigma**2)**(-.25)/np.sqrt(z)*np.exp(-(x-k0*T)**2/(2*sigma**2*z)+1j*k0*x-.5j*k0*k0*T)
        interior=np.exp(.5j*kk**2*T)*np.sum(np.exp(-1j*kk[:,None]*x)*psi*w,axis=1)/np.sqrt(2*np.pi)
        np.testing.assert_allclose(b+interior,expected,rtol=2e-7,atol=2e-7)

def test_3d_surface_commutator():
    h=Helium(make_grid([0,1,2,4,6,8],3),1,M=None,cutoff_radii=[1,2])
    idx=h.prepare_surface(5.)
    rng=np.random.default_rng(557);p=rng.normal(size=h.size)+1j*rng.normal(size=h.size)
    A=np.array([.12,.23,.34]);theta=(h.r.real>5.).astype(float)[None,:,None]
    u=p.reshape(h.shape,order='F')
    actual=h.flux(p,A)
    reference=(h.apply((theta*u).ravel(order='F'),A,True).reshape(h.shape,order='F')-
               theta*h.apply(p,A,True).reshape(h.shape,order='F'))[:,idx,:]
    np.testing.assert_allclose(actual,reference,atol=2e-12,rtol=2e-12)

def test_residual_pulse_impulse_is_not_silently_projected_as_field_free():
    import pytest
    from pulses import Pulse
    from surface3d import extract
    p=Pulse(cycles=2.5,field=.01,cep=np.pi/2)
    t=np.linspace(0,p.duration,2001)
    np.testing.assert_allclose(p.vector(p.duration),-simpson(p.electric(t),x=t),atol=2e-13)
    assert abs(p.vector(p.duration))>1e-4
    h=Helium(make_grid([0,1,2,4,6,8],3),1,M=0,cutoff_radii=[1,2]);idx=h.prepare_surface(5.)
    with pytest.raises(ValueError,match='zero final vector potential'):
        extract(h,np.zeros((2,h.n,len(idx),h.nc),complex),np.array([0,p.duration]),
                lambda s:np.array([0.,0.,p.vector(s)]),[(1,0,0)],np.array([.6]),(np.array([1.]),np.array([0.])),p.duration)

def test_surface_excludes_real_coordinate_with_complex_bridge_mass():
    import pytest
    h=Helium(make_grid([0,1,2,4,6,8],3,tail=8,ecs_angle=.5),1,M=0,cutoff_radii=[1,2])
    with pytest.raises(ValueError,match='ECS bridge'):h.prepare_surface(6.)
    h.prepare_surface(5.)
