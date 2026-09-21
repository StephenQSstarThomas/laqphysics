"""Independent backward blocks for CPU lookahead; no MPI or shared output writes."""
from types import SimpleNamespace
import signal,os
import numpy as np

_context=None

def initialize(config,times,ends,endpoint_folder):
    global _context
    # Only the parent checkpoints the coupled propagation. Pre-timeout signals
    # must not kill a child midway through a block and break the parent's future.
    for sig in (signal.SIGUSR1,signal.SIGTERM):signal.signal(sig,signal.SIG_IGN)
    import torch
    torch.set_num_threads(1)
    from fedvr import make_grid
    from angular import channels
    from tdse1d import cutoff
    from surface3d import Ionic
    from surface_storage import ShardedArray
    from streaming_surface import AdjointStepper
    from run3d import vector_function,electric_function
    g=make_grid(**config['radial']);h=SimpleNamespace(grid=g,r=g.r,n=len(g.r),cut=cutoff(g.r.real,*config['cutoff_radii']),ch=channels(config['lmax'],config.get('M',0)))
    labels=[tuple(x) for x in config.get('ionic_channels',[[1,0,0],[2,1,0]])]
    folded=all(p.get('polarization','z')=='z' for p in config['pulses']);mag={abs(m) for n,l,m in labels} if folded else None
    A,_=vector_function(config);field=A;gauge=config.get('gauge',{'type':'velocity'})
    if gauge['type']=='mixed':
        from mixed_gauge import MixedIonic,parameters
        ion=MixedIonic(h,mag,gauge['inner'],gauge['outer']);E=electric_function(config);field=lambda t:parameters(A,E,t)
    else:ion=Ionic(h,mag)
    stepper=AdjointStepper(ion,field,config.get('ionic_propagator','cf4'),config['storage'].get('ionic_device','cpu'))
    _context=(config,np.asarray(times),np.asarray(ends),ShardedArray(endpoint_folder),stepper)

def block(index):
    config,times,ends,states,stepper=_context;start=int(ends[index]);end=int(ends[index+1])
    chi=np.array(states[index+1]);stepper.reset(chi)
    result=np.empty((end-start+1,*chi.shape),complex);result[-1]=chi;result[0]=states[index]
    substeps=config.get('surface_stride',1);d=float(times[1]-times[0])/substeps
    for i in range(end,start+1,-1):
        for sub in range(substeps):stepper.step(times[i]-sub*d,-d)
        result[i-1-start]=stepper.chi
    return result,stepper.maximum_residual
