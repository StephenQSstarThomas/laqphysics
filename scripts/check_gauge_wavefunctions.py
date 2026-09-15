#!/usr/bin/env python3
"""Pure length/velocity gauge check on an identical finite real domain.

Dirichlet boundaries are gauge invariant. This compares full wavefunctions at
A(T)=0, without phase fitting; it is not an open-boundary photoelectron spectrum.
"""
from pathlib import Path
import json,time,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from fedvr import make_grid
from helium3d import Helium
from torch_backend import TorchHamiltonian
from mixed_gauge import MixedHamiltonian,parameters
from implicit import SeparablePreconditioner,cf4_step
from pulses import Pulse
out=Path('results/convergence_complete');rows=[]
pulse=Pulse(omega=1.5,cycles=2,field=.03);A=lambda t:np.array([0,0,pulse.vector(t)]);E=lambda t:np.array([0,0,pulse.electric(t)])
for order,lmax in [(4,2),(6,3),(8,4)]:
    h=Helium(make_grid([0,.5,1,2,4,6,8,12],order),lmax,M=0)
    Eg,gs,res=h.ground(tol=1e-10);steps=int(np.ceil(pulse.duration/.02));dt=pulse.duration/steps;states=[];statistics=[]
    for gauge in ['velocity','length']:
        engine=TorchHamiltonian(h,'cpu') if gauge=='velocity' else MixedHamiltonian(h,'cpu',inner=100,outer=101)
        engine.separable=SeparablePreconditioner(engine,'complex64');state=engine.state(gs)
        field=A if gauge=='velocity' else lambda t:parameters(A,E,t)
        start=time.perf_counter();maximum=0.
        for i in range(steps):
            state,info=cf4_step(engine,state,field,i*dt,dt,Eg,tol=1e-12);maximum=max(maximum,info['linear_residual'])
        x=engine.host(state);states.append(x);statistics.append({'gauge':gauge,'norm':float(np.vdot(x,x).real),'maximum_true_linear_residual':maximum,'seconds':time.perf_counter()-start})
    error=float(np.linalg.norm(states[0]-states[1]));row={'radial_order':order,'lmax':lmax,'real_box':12,'dt':dt,'wavefunction_L2_difference_without_phase_fit':error,'runs':statistics}
    rows.append(row);print(json.dumps(row),flush=True)
    (out/'pure_gauge_wavefunctions.json').write_text(json.dumps({'scope':'identical finite real box, full two-electron wavefunction comparison after a two-cycle pulse','comparisons':rows},indent=2)+'\n')
