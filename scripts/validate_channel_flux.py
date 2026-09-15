#!/usr/bin/env python3
"""Independent separable ion+Volkov wave packet validates the COMPLETE 1D extractor.

Synthetic surface data are explicitly not a helium calculation. The core follows
an independently integrated full ionic TDSE, including core excitation.
"""
from pathlib import Path
import sys,json,numpy as np
from scipy.integrate import solve_ivp,cumulative_trapezoid,simpson
from scipy.fft import fft,ifft
from numpy.polynomial.legendre import leggauss
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from tdse1d import Model,extract
from pulses import Pulse
root=Path(__file__).resolve().parents[1];out=root/'results/channel_flux_control';out.mkdir(exist_ok=True)
n=128;model=Model(n,40);ei,ui,d=model.ionic();p=Pulse(omega=float(ei[1]-ei[0]),cycles=12,field=.0534)
dt=.02;times=np.arange(9001)*dt;A=p.vector(times)
intA=cumulative_trapezoid(A,times,initial=0);intA2=cumulative_trapezoid(A*A,times,initial=0)
def rhs(t,y):return -1j*(ifft((model.p+p.vector(t))**2/2*fft(y))+model.ven*y)
sol=solve_ivp(rhs,(0,times[-1]),ui[:,0].astype(complex),t_eval=times,method='DOP853',rtol=2e-10,atol=2e-12)
assert sol.success
data=np.lib.format.open_memmap(out/'surface.npy',mode='w+',dtype=complex,shape=(len(times),2,n,2))
sigma=3.;k0=1.1;R=25.
for j,x in enumerate((-R,R)):
    z=1+1j*times/sigma**2;xx=x-intA
    psi=(np.pi*sigma**2)**(-.25)/np.sqrt(z)*np.exp(-(xx-k0*times)**2/(2*sigma**2*z)+1j*k0*xx-.5j*k0*k0*times-.5j*intA2)
    dp=(-(xx-k0*times)/(sigma*sigma*z)+1j*k0)*psi
    data[:,0,:,j]=sol.y.T*psi[:,None]/np.sqrt(2)
    data[:,1,:,j]=sol.y.T*dp[:,None]/np.sqrt(2)
data.flush();np.savez(out/'initial.npz',ionic_states=ui)
config={'grid':{'n':n,'halfbox':40.},'pulse':p.__dict__,'surface_stride':2}
(out/'run.json').write_text(json.dumps({'synthetic':True,'config':config,'surface_dt':dt,'positions':[-R,R]},indent=2)+'\n')
energy,P=extract(out,ionic_substeps=4);k=np.r_[-np.sqrt(2*energy),np.sqrt(2*energy)]
B=np.sqrt(sigma)/np.pi**.25*np.exp(-sigma*sigma*(k-k0)**2/2)
x,w=leggauss(256);x=x*R;w=w*R;t=times[-1];z=1+1j*t/sigma**2;xx=x-intA[-1]
packet=(np.pi*sigma**2)**(-.25)/np.sqrt(z)*np.exp(-(xx-k0*t)**2/(2*sigma*sigma*z)+1j*k0*xx-.5j*k0*k0*t-.5j*intA2[-1])
interior=np.exp(1j*(k*k*t/2+k*intA[-1]+intA2[-1]/2))*np.sum(np.exp(-1j*k[:,None]*x)*packet*w,axis=1)/np.sqrt(2*np.pi)
pop=abs(ui[:,:2].conj().T@sol.y[:,-1]*model.dx)**2
ne=len(energy);expected=pop[:,None]*(abs(B[:ne]-interior[:ne])**2+abs(B[ne:]-interior[ne:])**2)/np.sqrt(2*energy)
err=simpson(abs(P-expected),x=energy,axis=1)/simpson(expected,x=energy,axis=1)
result={'purpose':'synthetic separable core+free electron control, not a helium spectrum','core_final_populations':pop.tolist(),
        'spectrum_L1_relative_error':err.tolist(),'dt_surface':dt,'dt_ionic':dt/4}
np.savez(out/'expected.npz',energy=energy,expected=expected,actual=P)
(out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
assert max(err)<2e-3
