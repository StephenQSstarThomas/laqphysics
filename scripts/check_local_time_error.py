#!/usr/bin/env python3
"""Step doubling on an actual correlated TDSE checkpoint, including its phase.

This is a local diagnostic; only a complete finer-dt spectrum establishes global
time convergence. The source checkpoint and its live surface history are read only.
"""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from run3d import setup,vector_function
from torch_backend import TorchHamiltonian
from implicit import SeparablePreconditioner,cf4_step
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--device',default='cpu');p.add_argument('--levels',type=int,default=3)
p.add_argument('--checkpoint',default='checkpoint.npz');a=p.parse_args()
out=Path(a.out);config=json.loads((out/'config.json').read_text())
with np.load(out/a.checkpoint) as saved:
    x=saved['psi'];step=int(saved['step']);Eg=float(saved['ground_energy']);signature=str(saved['signature'])
avec,pulses=vector_function(config);T=config.get('end_time',max(p.start+p.duration for p,pol in pulses)+config.get('post_time',60.))
nsteps=(np.load(out/'flux.npy',mmap_mode='r').shape[0]-1)*config.get('surface_stride',1);dt=T/nsteps;t=step*dt
_,h=setup(config);h.prepare_surface(config['surface']);engine=TorchHamiltonian(h,a.device);engine.separable=SeparablePreconditioner(engine);state=engine.state(x)
gs=np.load(out/'checkpoint.npz')['ground']
native=h.apply(x,avec(t),True);tensor=engine.host(engine.apply(state,avec(t),True))
operator_error=float(np.linalg.norm(native-tensor)/np.linalg.norm(native))
np.savez(out/'checkpoint_local_time.npz',psi=x,step=step,ground_energy=Eg,signature=signature)
weight=h.grid.interior_weights/abs(h.grid.weights);weight=weight[:,None,None]*weight[None,:,None]
mask=h.r.real<=8;core=weight*mask[:,None,None]*mask[None,:,None];rows=[]
for level in range(a.levels):
    d=dt/2**level;start=time.perf_counter();full,info=cf4_step(engine,state,avec,t,d,Eg,tol=1e-12)
    half,i1=cf4_step(engine,state,avec,t,d/2,Eg,tol=5e-13)
    half,i2=cf4_step(engine,half,avec,t+d/2,d/2,Eg,tol=5e-13)
    one=engine.host(full);two=engine.host(half);difference=float(np.linalg.norm(one-two)/np.linalg.norm(two))
    error2=abs((one-two).reshape(h.shape,order='F'))**2;state2=abs(two.reshape(h.shape,order='F'))**2
    flux1=h.flux(one,avec(t+d));flux2=h.flux(two,avec(t+d))
    row={'dt_au':d,'full_vs_two_half_relative_L2':difference,'richardson_fine_local_error_estimate':difference/15,
         'real_interior_relative_L2':float(np.sqrt(np.sum(error2*weight)/np.sum(state2*weight))),
         'core_relative_L2_r_le_8':float(np.sqrt(np.sum(error2*core)/np.sum(state2*core))),
         'ground_amplitude_absolute_difference':float(abs(np.vdot(gs,one-two))),
         'ground_probability_absolute_difference':float(abs(abs(np.vdot(gs,one))**2-abs(np.vdot(gs,two))**2)),
         'surface_flux_relative_L2_difference':float(np.linalg.norm(flux1-flux2)/np.linalg.norm(flux2)),
         'maximum_true_linear_residual':max(q['linear_residual'] for q in [info,i1,i2]),'seconds':time.perf_counter()-start}
    rows.append(row);print(json.dumps(row),flush=True)
result={'source_signature':signature,'checkpoint_step':step,'time_au':t,'vector_potential':avec(t).tolist(),
        'actual_checkpoint_Torch_vs_Fortran_H_relative_L2':operator_error,
        'step_doubling':rows,'scope':'Same actual correlated state at all step sizes, no phase fitting; Richardson estimates assume the asymptotic order, not a global spectrum-convergence certificate.'}
(out/'local_step_doubling.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
