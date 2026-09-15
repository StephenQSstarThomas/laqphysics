#!/usr/bin/env python3
"""Bound-state He+ control calculations. Continuum loss is intentionally excluded
and results must NOT be interpreted as full ionization-inclusive transfer efficiencies.
"""
from pathlib import Path
import sys,json,time,numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from angular import cartesian,radial_dipole,bound_dipole
from pulses import Pulse
root=Path(__file__).resolve().parents[1];out=root/'results/ionic_controls';out.mkdir(exist_ok=True)
states=[(n,l,m) for n in range(1,5) for l in range(n) for m in range(-l,l+1)]
energy=np.array([-2/n**2 for n,l,m in states]);D=np.zeros((3,len(states),len(states)),complex)
for a,(n,l,m) in enumerate(states):
    for b,(nn,ll,mm) in enumerate(states):
        angular=cartesian((l,m),(ll,mm))
        if np.max(abs(angular))<1e-14:continue
        D[:,a,b]=angular*radial_dipole(n,l,nn,ll)
assert np.max(abs(D-D.conj().transpose(0,2,1)))<1e-12

def run(name,initial,pulse,polarization):
    index=states.index(initial);E=energy-energy[index]
    def rhs(t,y):
        phase=pulse.omega*t;f=pulse.field*pulse.envelope(t)
        if polarization=='z':field=np.array([0,0,f*np.cos(phase)])
        else:field=f/np.sqrt(2)*np.array([np.cos(phase),polarization*np.sin(phase),0])
        return -1j*(E*y+np.einsum('a,aij,j->i',field,D,y,optimize=True))
    y=np.zeros(len(states),complex);y[index]=1;t=np.linspace(0,pulse.duration,401);start=time.perf_counter()
    sol=solve_ivp(rhs,(0,pulse.duration),y,method='DOP853',t_eval=t,rtol=2e-9,atol=2e-11)
    if not sol.success:raise RuntimeError(sol.message)
    pop=abs(sol.y[:,-1])**2
    result={'name':name,'initial':initial,'polarization':polarization,'pulse':pulse.__dict__,
            'basis_nmax':4,'continuum_included':False,'norm_error':float(abs(sum(pop)-1)),
            'seconds':time.perf_counter()-start,'populations':{str(s):float(p) for s,p in zip(states,pop) if p>1e-12},
            'P_2p':float(sum(p for s,p in zip(states,pop) if s[:2]==(2,1))),
            'P_3p':float(sum(p for s,p in zip(states,pop) if s[:2]==(3,1))),
            'P_3d':float(sum(p for s,p in zip(states,pop) if s[:2]==(3,2)))}
    np.savez(out/(name+'.npz'),time=t,states=states,populations=abs(sol.y)**2)
    assert result['norm_error']<2e-6
    print(json.dumps(result),flush=True)
    return result

results=[]
for pol in ['z',1,-1]:
    results.append(run('resonant_'+str(pol),(1,0,0),Pulse(omega=1.5,cycles=201,field=.02),pol))
results.append(run('half_frequency_circular',(1,0,0),Pulse(omega=.75,cycles=100,field=.04),1))
for pol in [1,-1]:
    results.append(run('two_photon_'+str(pol),(2,1,1),Pulse(omega=5/36,cycles=224,field=.003),pol))
(out/'summary.json').write_text(json.dumps({'scope':'qualitative bound-state control model, not continuum-converged rates','runs':results},indent=2)+'\n')
