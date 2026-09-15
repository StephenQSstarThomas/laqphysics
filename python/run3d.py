"""Checkpointed production driver. See configs and scripts/*.slurm for invocations."""
import argparse,hashlib,json,time,os,signal
from pathlib import Path
import numpy as np
from fedvr import make_grid
from helium3d import Helium
from propagate import step
from pulses import Pulse
from output_lock import exclusive_output

def setup(config):
    gconfig=dict(config['radial']);angle=gconfig.pop('ecs_angle',0.)
    real=make_grid(**gconfig);complex_grid=make_grid(**gconfig,ecs_angle=angle)
    args={'lmax':config['lmax'],'M':config.get('M',0),'cutoff_radii':config['cutoff_radii']}
    pair=(Helium(real,**args),Helium(complex_grid,**args))
    if 'total_Lmax' in config:
        if config.get('M',0)!=0 or any(p.get('polarization','z')!='z' for p in config.get('pulses',[])):
            raise ValueError('total_Lmax compression currently supports M=0 and z polarization only')
        from coupled_angular import CoupledHelium
        pair=tuple(CoupledHelium(h,config['total_Lmax']) for h in pair)
    return pair

def vector_function(config):
    pulses=[(Pulse(**p['pulse']),p.get('polarization','z')) for p in config['pulses']]
    def avec(t):
        v=np.zeros(3)
        for p,pol in pulses:
            if pol=='z':v[2]+=p.vector(t)
            elif pol in ('sigma+','sigma-'):
                # E=F/sqrt(2)*(x cos(wt)+/- y sin(wt)); same cycle-average
                # intensity as the linear pulse F cos(wt). Positive absorption q=+1
                # for E_x cos + E_y sin with the current spherical convention.
                from dataclasses import replace
                v[0]+=p.vector(t)/np.sqrt(2)
                v[1]+=replace(p,cep=p.cep-np.pi/2).vector(t)/np.sqrt(2)*(1 if pol=='sigma+' else -1)
            else:raise ValueError('polarization must be z, sigma+, or sigma-')
        return v
    return avec,pulses

@exclusive_output('.propagation.lock')
def run(config,out,resume=False,max_steps=None,backend='fortran',device='cuda:0',ground_cache=None,coulomb_contraction=None):
    if backend not in ('fortran','torch'):raise ValueError('backend must be fortran or torch')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    surface=out/'flux.npy'
    if os.environ.get('HELIUM_SURFACE_ROOT') and not surface.exists() and not surface.is_symlink():
        # Different campaigns often reuse case names. Their raw histories must
        # remain distinct even when they share a scratch root.
        namespace=hashlib.sha256(str(out.resolve()).encode()).hexdigest()[:16]
        storage=Path(os.environ['HELIUM_SURFACE_ROOT']).expanduser().resolve()/(out.name+'-'+namespace)
        storage.mkdir(parents=True,exist_ok=True);surface.symlink_to(storage/'flux.npy')
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    interrupted=[False]
    previous={}
    for sig in (signal.SIGUSR1,signal.SIGTERM):
        previous[sig]=signal.signal(sig,lambda signum,frame:interrupted.__setitem__(0,True))
    hreal,h=setup(config);idx=h.prepare_surface(config['surface']);avec,pulses=vector_function(config)
    if config.get('M',0) is not None and any(pol!='z' for p,pol in pulses):
        raise ValueError('circular polarization requires M=null (all magnetic channels)')
    T=max(p.start+p.duration for p,pol in pulses)+config.get('post_time',60.)
    if 'end_time' in config:
        if config['end_time']<max(p.start+p.duration for p,pol in pulses):raise ValueError('end_time precedes end of pulse')
        T=float(config['end_time'])
    if np.max(abs(avec(T)))>1e-10:
        raise ValueError('This production driver requires zero final vector potential for its bound-state spectra')
    stride=config.get('surface_stride',1);nsteps=int(np.ceil(T/config.get('dt',.05)))
    nsteps+=(-nsteps)%stride;dt=T/nsteps;ns=nsteps//stride+1
    shape=(ns,h.n,len(idx),h.nc);disk=np.prod(shape)*16
    if disk>config.get('max_surface_gib',12)*2**30:raise ValueError(f'flux needs {disk/2**30:.2f} GiB; adjust resolution/stride/storage budget explicitly')
    checkpoint=out/'checkpoint.npz';t0=time.perf_counter()
    if resume:
        saved=np.load(checkpoint)
        if str(saved['signature'])!=signature:raise ValueError('config changed; refusing unsafe resume')
        psi=saved['psi'];begin=int(saved['step']);E=float(saved['ground_energy']);gs=saved['ground'];gres=float(saved['ground_residual'])
        flux=np.lib.format.open_memmap(out/'flux.npy',mode='r+')
        history=json.loads((out/'history.json').read_text())
        history=[r for r in history if r['step']<=begin]
    else:
        if checkpoint.exists() or (out/'flux.npy').exists():raise FileExistsError('output exists; use --resume or another directory')
        free=__import__('shutil').disk_usage(surface.resolve().parent).free
        if disk>free*.8:raise ValueError('insufficient disk space for surface history')
        E,gs,gres=hreal.ground(tol=config.get('ground_tolerance',1e-9),cache_dir=ground_cache)
        ratio=np.sqrt(h.grid.weights/hreal.grid.weights)
        psi=(gs.reshape(h.shape,order='F')*ratio[:,None,None]*ratio[None,:,None]).ravel(order='F')
        begin=0;history=[]
        flux=np.lib.format.open_memmap(out/'flux.npy',mode='w+',dtype=complex,shape=shape)
        flux[0]=h.flux(psi,avec(0))
        (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        print(f'ground E={E:.10f}, residual={gres:.3e}; {h.shape}, flux={disk/2**30:.3f}GiB',flush=True)
    stop=nsteps if max_steps is None else min(nsteps,begin+max_steps)
    # Checkpoints are only made on an available surface record.
    stop-=stop%stride
    engine=None
    if backend=='torch':
        from torch_backend import TorchHamiltonian,step as device_step
        engine=TorchHamiltonian(h,device,coulomb_backend=coulomb_contraction or config.get('coulomb_contraction','sparse'));state=engine.state(psi)
        if config.get('time_integrator','arnoldi')=='cf4-pade':
            from implicit import SeparablePreconditioner,cf4_step
            engine.separable=SeparablePreconditioner(engine)
    if config.get('time_integrator','arnoldi')=='cf4-pade' and engine is None:
        raise ValueError('cf4-pade currently requires the torch backend (CPU or CUDA)')
    max_linear_residual=0.;linear_iterations=0
    def save(i):
        flux.flush()
        hist_temp=out/'history.tmp.json'
        hist_temp.write_text(json.dumps(history,indent=2)+'\n');os.replace(hist_temp,out/'history.json')
        temp=out/'checkpoint.tmp.npz'
        np.savez(temp,psi=psi,step=i,signature=signature,ground_energy=E,ground=gs,ground_residual=gres)
        os.replace(temp,checkpoint)
    for i in range(begin,stop):
        if engine is None:
            psi,info=step(lambda t,y:h.apply(y,avec(t),velocity=True),psi,i*dt,dt,
                          tol=config.get('krylov_tolerance',1e-10),maxdim=config.get('krylov_dimension',48))
        else:
            if config.get('time_integrator','arnoldi')=='cf4-pade':
                state,info=cf4_step(engine,state,avec,i*dt,dt,E,tol=config.get('linear_tolerance',1e-10))
                max_linear_residual=max(max_linear_residual,info['linear_residual']);linear_iterations+=info['linear_iterations']
            else:
                state,info=device_step(lambda t,y:engine.apply(y,avec(t),velocity=True),state,i*dt,dt,
                              tol=config.get('krylov_tolerance',1e-10),maxdim=config.get('krylov_dimension',48))
        if (i+1)%stride==0:
            flux[(i+1)//stride]=h.flux(psi,avec((i+1)*dt)) if engine is None else engine.flux(state,avec((i+1)*dt))
        if interrupted[0] and (i+1)%stride==0:stop=i+1
        every=config.get('checkpoint_every',100)
        if ((i+1)%every==0 and (i+1)%stride==0) or i+1==stop:
            if engine is not None:psi=engine.host(state)
            u=psi.reshape(h.shape,order='F')
            # The bridge has complex total mass; use ONLY its finite real element
            # contribution, not |complex mass|, in the physical inner-region norm.
            f=h.grid.interior_weights/abs(h.grid.weights)
            norm=float(np.sum(abs(u)**2*f[:,None,None]*f[None,:,None]))
            pg=float(abs(np.vdot(gs,psi))**2)
            row={'step':i+1,'time':(i+1)*dt,'real_region_norm':norm,'ground_population':pg,
                 'exchange_error':h.exchange_error(psi),**info}
            history.append(row);save(i+1)
            print(f'3D {i+1}/{nsteps}, inner norm={norm:.8f}, Pg={pg:.8f}, wall={time.perf_counter()-t0:.1f}s',flush=True)
        if i+1==stop:break
    meta={'signature':signature,'config':config,'ground_energy':E,'ground_residual':gres,'nrad':h.n,'channels':h.nc,
          'dt':dt,'nsteps':nsteps,'completed_steps':stop,'complete':stop==nsteps,'surface_bytes':int(disk),
          'surface_indices':idx.tolist(),'seconds_this_invocation':time.perf_counter()-t0,
          'backend':backend,'device':device if engine is not None else 'cpu','precision':'complex128',
          'ground_angular_basis':'coupled total L=0, expanded without angular approximation',
          'time_integrator':config.get('time_integrator','arnoldi'),
          'coulomb_contraction':engine.coulomb_backend if engine is not None else 'Fortran multipoles',
          'precomputed_coulomb_bytes':engine.coulomb_block_bytes if engine is not None else 0,
          'maximum_linear_residual_this_invocation':max_linear_residual,'linear_iterations_this_invocation':linear_iterations}
    (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
    for sig,handler in previous.items():signal.signal(sig,handler)
    return meta

@exclusive_output('.extraction.lock')
def extract_run(out,output='spectrum.npz',channels=None,stop_time=None):
    from surface3d import extract
    out=Path(out)
    if (out/'run.json').exists():meta=json.loads((out/'run.json').read_text())
    elif stop_time is not None:
        config=json.loads((out/'config.json').read_text());saved=np.load(out/'checkpoint.npz')
        signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        if str(saved['signature'])!=signature:raise ValueError('checkpoint configuration mismatch')
        _,pulses=vector_function(config)
        T=config.get('end_time',max(p.start+p.duration for p,pol in pulses)+config.get('post_time',60.))
        records=np.load(out/'flux.npy',mmap_mode='r').shape[0]
        meta={'config':config,'dt':T/((records-1)*config.get('surface_stride',1)),'signature':signature,'complete':False}
    else:raise ValueError('propagation incomplete')
    if not meta['complete'] and stop_time is None:raise ValueError('propagation incomplete')
    if stop_time is not None and output=='spectrum.npz':raise ValueError('prefix extraction requires a distinct spectrum name')
    if stop_time is not None and not meta['complete']:
        saved=np.load(out/'checkpoint.npz')
        if str(saved['signature'])!=meta['signature']:raise ValueError('checkpoint configuration mismatch')
        if stop_time>int(saved['step'])*meta['dt']+1e-9:raise ValueError('prefix exceeds flushed checkpoint history')
    config=meta['config'];hr,h=setup(config);h.prepare_surface(config['surface']);avec,pulses=vector_function(config)
    f=np.load(out/'flux.npy',mmap_mode='r');dt=meta['dt']*config.get('surface_stride',1);t=np.arange(len(f))*dt
    if stop_time is not None:
        last=int(round(stop_time/dt))
        if last<1 or last>=len(f) or abs(last*dt-stop_time)>1e-8:
            raise ValueError('stop_time must equal an available surface time')
        f=f[:last+1];t=t[:last+1]
    energy=np.linspace(*config.get('spectrum_energy',[.3,.9,121]));theta=np.linspace(0,np.pi,61);phi=np.zeros(61)
    labels=[(1,0,0),(2,1,0)] if config.get('M',0)==0 else [(1,0,0),(2,1,-1),(2,1,0),(2,1,1)]
    labels=[tuple(a) for a in config.get('ionic_channels',labels)]
    if channels is not None:labels=[tuple(a) for a in channels]
    if any(n<=l or l<0 or l>config['lmax'] or abs(m)>l for n,l,m in labels):
        raise ValueError('ionic_channels must contain valid (n,l,m) within lmax')
    if config.get('M',0) is None:
        # Full solid-angle quadrature: Gauss-Legendre theta and uniform azimuth.
        from numpy.polynomial.legendre import leggauss
        z,w=leggauss(20);ph=np.arange(32)*2*np.pi/32
        theta,phi=np.meshgrid(np.arccos(z),ph,indexing='ij');theta=theta.ravel();phi=phi.ravel()
    reference=h.uncoupled if hasattr(h,'uncoupled') else h
    reference.prepare_surface(config['surface'])
    transform=h.transform if hasattr(h,'transform') else None
    diagnostics={'requested_times':[s for s in config.get('ionic_transfer_times',[]) if s<=t[-1]]}
    b,pes=extract(reference,f,t,avec,labels,energy,(theta,phi),dt,substeps=config.get('surface_stride',1),angular_transform=transform,
                  propagator=config.get('ionic_propagator','expm'),diagnostics=diagnostics)
    if config.get('M',0)==0:
        from scipy.integrate import simpson
        total=2*np.pi*simpson(pes*np.sin(theta),x=theta,axis=2)
    else:total=np.sum(pes.reshape(len(labels),len(energy),20,32)*w[None,None,:,None],axis=(2,3))*2*np.pi/32
    destination=out/output;temporary=destination.with_name(destination.stem+'.tmp.npz')
    np.savez_compressed(temporary,energy=energy,theta=theta,phi=phi,amplitudes=b,pes=pes,angle_integrated=total,labels=labels,
                        ionic_projection='analytic radial c-dual',extraction_end_time=t[-1],source_signature=meta['signature'])
    transfer_name='ionic_transfer.json' if output=='spectrum.npz' else Path(output).stem+'_ionic_transfer.json'
    if diagnostics['requested_times']:
        temp=out/(transfer_name+'.tmp.json');temp.write_text(json.dumps(diagnostics,indent=2)+'\n');os.replace(temp,out/transfer_name)
    os.replace(temporary,destination)
    return total

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--out',required=True)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--max-steps',type=int)
    parser.add_argument('--extract',action='store_true')
    parser.add_argument('--backend',choices=['fortran','torch'],default='fortran')
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--ground-cache')
    parser.add_argument('--coulomb-contraction',choices=['sparse','blocks'],help='equivalent contraction implementation; recorded separately from physical configuration')
    parser.add_argument('--spectrum-name',default='spectrum.npz');parser.add_argument('--all-n2',action='store_true')
    parser.add_argument('--stop-time',type=float,help='extract only this prefix; requires --spectrum-name')
    a=parser.parse_args()
    if a.extract:extract_run(a.out,a.spectrum_name,[[1,0,0],[2,0,0],[2,1,-1],[2,1,0],[2,1,1]] if a.all_n2 else None,a.stop_time)
    else:run(json.loads(Path(a.config).read_text()),a.out,a.resume,a.max_steps,a.backend,a.device,a.ground_cache,a.coulomb_contraction)
if __name__=='__main__':main()
