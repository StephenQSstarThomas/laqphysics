"""Finite-band ESSS density matrix with a trace-preserving Lindblad dephaser.

This controlled small-grid model includes decoherence DURING ionization. It is
phenomenological; gamma is not inferred from an actual helium environment.
"""
import numpy as np
from scipy.sparse import diags,csr_matrix
from pulses import Pulse

def propagate(energy,weights,pulse=Pulse(cycles=20),gamma=0.,dg=.5213,d12=.4823,dt=.05,eg=-2.9,e1=-2.,e2=-.5):
    e=np.asarray(energy);w=np.asarray(weights);n=len(e);dim=1+2*n
    if gamma<0 or np.any(w<=0):raise ValueError('gamma>=0 and weights>0')
    H0=diags(np.r_[0.,e+e1-eg-pulse.omega,e+e2-eg-2*pulse.omega],format='csr',dtype=complex)
    phase=np.exp(1j*(pulse.omega*pulse.start-pulse.cep))
    row=[];col=[];values=[]
    for i in range(n):
        for a,b,v in [(1+i,0,dg*np.sqrt(w[i])*phase),(1+n+i,1+i,np.conj(d12)*phase)]:
            row.extend([a,b]);col.extend([b,a]);values.extend([v,np.conj(v)])
    D=csr_matrix((values,(row,col)),shape=(dim,dim),dtype=complex)
    # Eigenvalues of L=sqrt(gamma/2)*(P1-P2); rho_12 decays at gamma.
    ell=np.sqrt(gamma/2)*np.r_[0.,np.ones(n),-np.ones(n)]
    damping=-.5*(ell[:,None]-ell[None,:])**2
    rho=np.zeros((dim,dim),complex);rho[0,0]=1
    end=pulse.start+pulse.duration;steps=int(np.ceil(end/dt));dt=end/steps
    def rhs(t,r):
        H=H0+pulse.field*pulse.envelope(t)/2*D
        return -1j*(H@r-(H.T@r.T).T)+damping*r
    for i in range(steps):
        t=i*dt;k1=rhs(t,rho);k2=rhs(t+dt/2,rho+dt*k1/2)
        k3=rhs(t+dt/2,rho+dt*k2/2);k4=rhs(t+dt,rho+dt*k3)
        rho+=dt*(k1+2*k2+2*k3+k4)/6
    return rho

def diagnostics(rho,n):
    block=rho[1:,1:];prob=np.trace(block).real
    joint=block/prob
    pt=joint.reshape(2,n,2,n).transpose(2,1,0,3).reshape(2*n,2*n)
    return {'trace_error':float(abs(np.trace(rho)-1)),
            'hermiticity_error':float(np.linalg.norm(rho-rho.conj().T)),
            'minimum_eigenvalue':float(np.linalg.eigvalsh(rho).min()),
            'single_ionization_probability':float(prob),
            'conditional_negativity':float((np.sum(abs(np.linalg.eigvalsh(pt)))-1)/2)}
