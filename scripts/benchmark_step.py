#!/usr/bin/env python3
"""Seconds per two-electron step, peak GPU memory and ionic-preparation cost of one input.

Builds exactly the Torch Hamiltonian, complex64 separable preconditioner and CF4
step that simulate_spectrum.py uses, starts from the field-free ground state and
times a few steps at a chosen time (default: middle of the last pulse, where the
linear solves are hardest). Also times backward adjoint steps for the recorded
ionic channels (the serial preparation phase before propagation). Nothing is
written into a run directory; this is a sizing tool, not a result.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'python'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--steps',type=int,default=3);p.add_argument('--time',type=float,help='time of the timed steps (a.u.)')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--ground-cache');p.add_argument('--ionic-steps',type=int,default=5)
    a=p.parse_args();config=json.loads(Path(a.config).read_text(encoding='utf-8'))
    import torch;torch.set_num_threads(a.threads)
    from run3d import setup,vector_function
    from campaign_resources import estimate
    from torch_backend import TorchHamiltonian
    from implicit import SeparablePreconditioner,cf4_step
    from pulses import Pulse
    resources=estimate(config);avec,pulses=vector_function(config);dt=resources['dt_actual']
    last=max(pulses,key=lambda x:x[0].start)[0];t0=a.time if a.time is not None else last.start+last.duration/2
    started=time.perf_counter();hreal,h=setup(config);h.prepare_surface(config['surface'])
    E,gs,_=hreal.ground(tol=config.get('ground_tolerance',1e-9),cache_dir=a.ground_cache)
    ratio=np.sqrt(h.grid.weights/hreal.grid.weights);psi=(gs.reshape(h.shape,order='F')*ratio[:,None,None]*ratio[None,:,None]).ravel(order='F')
    engine=TorchHamiltonian(h,a.device,coulomb_backend=config.get('coulomb_contraction','sparse'))
    engine.separable=SeparablePreconditioner(engine,precision='complex64');state=engine.state(psi);setup_seconds=time.perf_counter()-started
    if engine.device.type=='cuda':torch.cuda.reset_peak_memory_stats(engine.device)
    state,info=cf4_step(engine,state,avec,t0,dt,E,tol=config.get('linear_tolerance',1e-10))
    if engine.device.type=='cuda':torch.cuda.synchronize(engine.device)
    rows=[];start=time.perf_counter()
    for i in range(a.steps):
        state,info=cf4_step(engine,state,avec,t0+(i+1)*dt,dt,E,tol=config.get('linear_tolerance',1e-10));frame=engine.flux(state,avec(t0+(i+2)*dt))
        if engine.device.type=='cuda':torch.cuda.synchronize(engine.device)
        rows.append({'linear_iterations':info['linear_iterations'],'linear_residual':info['linear_residual']})
    per_step=(time.perf_counter()-start)/a.steps
    from ionic_probe import ion_model
    from streaming_surface import AdjointStepper
    labels=[tuple(x) for x in config.get('ionic_channels',[[1,0,0],[2,1,0]])];ion=ion_model(config)
    stepper=AdjointStepper(ion,avec,'cf4','cpu');stepper.reset(ion.dual_final_states(list(dict.fromkeys(labels))))
    stepper.step(t0,-dt);start=time.perf_counter()
    for i in range(a.ionic_steps):stepper.step(t0-(i+1)*dt,-dt)
    ionic=(time.perf_counter()-start)/a.ionic_steps;workers=int(config.get('storage',{}).get('replay_workers',0))
    report={'config':a.config,'device':a.device,'torch_threads':a.threads,'steps_in_run':resources['steps'],'dt':dt,'timed_at':t0,
            'setup_seconds_including_ground_state':setup_seconds,'seconds_per_2e_step':per_step,'linear_steps':rows,
            'peak_cuda_allocated_GiB':torch.cuda.max_memory_allocated(engine.device)/2**30 if engine.device.type=='cuda' else None,
            'FGMRES_estimate_GiB':resources['FGMRES_Q_Z_bytes']/2**30,
            'projected_propagation_hours':per_step*resources['steps']/3600,
            'ionic_channels':len(labels),'seconds_per_ionic_adjoint_step_cpu':ionic,
            'projected_serial_ionic_preparation_hours':ionic*resources['steps']/3600,
            'replay_workers':workers,
            'note':'Shared-machine timings; one replay worker has one thread, so replay needs about (ionic step time x threads here / workers) per frame.'}
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
