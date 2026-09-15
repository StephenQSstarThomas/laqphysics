"""Energy-normalized essential-state model, Yu & Madsen PRA 98, 033404.

The published Dg=.5213 is a fitted 1D-model value; it is NOT a 3D He matrix element.
"""
import numpy as np
from scipy.integrate import simpson, solve_ivp
from pulses import Pulse

def analytic(energy, pulse=Pulse(), dg=.5213, d12=.4823, eg=-2.90, e1=-2., nt=6001):
    """Eqs (19)--(21), exact resonance and flat continuum extended to all real E."""
    energy=np.asarray(energy);end=pulse.start+pulse.duration;t=np.linspace(0,end,nt)
    phase0=np.exp(1j*(pulse.omega*pulse.start-pulse.cep))
    cg=np.exp(-np.pi*abs(dg)**2*pulse.field**2/4*pulse.area_squared(t))
    source=pulse.field*dg/2*pulse.envelope(t)*cg
    theta=pulse.field*abs(d12)/2*(pulse.area_envelope(end)-pulse.area_envelope(t))
    a=np.empty((2,len(energy)),complex)
    # Batch to bound memory for the resolution sweeps.
    for begin in range(0,len(energy),128):
        e=energy[begin:begin+128]
        phase=np.exp(-1j*(e[:,None]-(pulse.omega+eg-e1))*(end-t))
        a[0,begin:begin+128]=-1j*phase0*simpson(phase*(source*np.cos(theta))[None,:],x=t,axis=1)
        relative=np.conj(d12)/abs(d12) if abs(d12)>0 else 1.
        a[1,begin:begin+128]=-relative*phase0**2*simpson(phase*(source*np.sin(theta))[None,:],x=t,axis=1)
    return a,float(cg[-1]**2)

def driven(energy, pulse=Pulse(), dg=.5213, d12=.4823, eg=-2.90, e1=-2., e2=-.5,
           dt=.1, trace=None):
    """Independent RK4 integration of Eqs (7),(8), with analytic Markov cg (19).

    Supports detuning, complex energy-dependent Dg and optional every-stage dumps.
    This is NOT the finite-band depletion/backcoupling model below.
    """
    energy=np.asarray(energy); g=np.broadcast_to(np.asarray(dg),energy.shape)
    if np.ptp(np.abs(g))>1e-12:
        raise ValueError('Markov source requires flat |Dg|; use finite_band for general Dg')
    d=np.array([energy+e1-eg-pulse.omega,energy+e2-eg-2*pulse.omega])
    a=np.zeros_like(d,dtype=complex)
    phase0=np.exp(1j*(pulse.omega*pulse.start-pulse.cep))
    def rhs(t,y):
        f=pulse.field*pulse.envelope(t)/2
        z=d*y; z=z.astype(complex)
        z[0]+=f*d12*phase0.conjugate()*y[1]+f*g*phase0*np.exp(-np.pi*abs(g[0])**2*pulse.field**2/4*pulse.area_squared(t))
        z[1]+=f*np.conj(d12)*phase0*y[0]
        return -1j*z
    end=pulse.start+pulse.duration;n=int(np.ceil(end/dt));dt=end/n
    for i in range(n):
        t=i*dt;k1=rhs(t,a); y2=a+dt*k1/2;k2=rhs(t+dt/2,y2)
        y3=a+dt*k2/2;k3=rhs(t+dt/2,y3);y4=a+dt*k3;k4=rhs(t+dt,y4)
        a+=dt*(k1+2*k2+2*k3+k4)/6
        if trace is not None:
            trace(i,t+dt,{'k1':k1,'stage2':y2,'k2':k2,'stage3':y3,'k3':k3,'stage4':y4,'k4':k4,'state':a.copy()})
    return a

def finite_band(energy,weights,pulse=Pulse(),dg=.5213,d12=.4823,eg=-2.9,e1=-2.,e2=-.5):
    """Hermitian quadrature discretization of Eqs (6)--(8); includes backcoupling.

    b_j=sqrt(w)*c_j ensures Euclidean norm conservation. No artificial renormalization.
    """
    e=np.asarray(energy);w=np.asarray(weights)
    if e.ndim!=1 or w.shape!=e.shape or np.any(w<=0):
        raise ValueError('positive quadrature weights required')
    n=len(e);g=np.broadcast_to(dg,e.shape)*np.sqrt(w)
    phase0=np.exp(1j*(pulse.omega*pulse.start-pulse.cep))
    det=np.r_[e+e1-eg-pulse.omega,e+e2-eg-2*pulse.omega]
    def fun(t,y):
        f=pulse.field*pulse.envelope(t)/2;b=y[1:].reshape(2,n)
        out=np.empty_like(y);out[0]=f*phase0.conjugate()*np.vdot(g,b[0])
        out[1:]=det*y[1:]
        out[1:1+n]+=f*(g*phase0*y[0]+d12*phase0.conjugate()*b[1]);out[1+n:]+=f*np.conj(d12)*phase0*b[0]
        return -1j*out
    y=np.zeros(1+2*n,complex);y[0]=1
    sol=solve_ivp(fun,(0,pulse.start+pulse.duration),y,method='DOP853',rtol=2e-10,atol=2e-12)
    if not sol.success:raise RuntimeError(sol.message)
    return sol.y[:,-1],sol

def reduced_ion(amplitudes,weights):
    a=np.asarray(amplitudes);w=np.asarray(weights)
    rho=(a*w)@a.conj().T
    p=np.trace(rho).real
    if p<=0:raise ValueError('empty single-ionization sector')
    rho/=p;lam=np.linalg.eigvalsh(rho).clip(0,1)
    nz=lam[lam>1e-15]
    return rho,{'sector_probability':float(p),'purity':float(np.sum(lam**2)),
                'entropy_bits':float(-np.sum(nz*np.log2(nz))),
                'negativity_pure':float(((np.sum(np.sqrt(lam)))**2-1)/2)}
