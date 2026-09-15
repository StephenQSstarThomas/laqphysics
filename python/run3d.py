"""Checkpointed production driver. See configs and scripts/*.slurm for invocations."""
import argparse,hashlib,json,time,os,signal
from pathlib import Path
import numpy as np
from fedvr import make_grid
from helium3d import Helium
from propagate import step
from pulses import Pulse

def setup(config):
    gconfig=dict(config['radial']);angle=gconfig.pop('ecs_angle',0.)
    real=make_grid(**gconfig);complex_grid=make_grid(**gconfig,ecs_angle=angle)
    args={'lmax':config['lmax'],'M':config.get('M',0),'cutoff_radii':config['cutoff_radii']}
    return Helium(real,**args),Helium(complex_grid,**args)

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

def run(config,out,resume=False,max_steps=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    interrupted=[False]
    previous={}
    for sig in (signal.SIGUSR1,signal.SIGTERM):
        previous[sig]=signal.signal(sig,lambda signum,frame:interrupted.__setitem__(0,True))
    hreal,h=setup(config);idx=h.prepare_surface(config['surface']);avec,pulses=vector_function(config)
    if config.get('M',0) is not None and any(pol!='z' for p,pol in pulses):
        raise ValueError('circular polarization requires M=null (all magnetic channels)')
    T=max(p.start+p.duration for p,pol in pulses)+config.get('post_time',60.)
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
        free=__import__('shutil').disk_usage(out).free
        if disk>free*.8:raise ValueError('insufficient disk space for surface history')
        E,gs,gres=hreal.ground(tol=config.get('ground_tolerance',1e-9))
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
    def save(i):
        flux.flush()
        hist_temp=out/'history.tmp.json'
        hist_temp.write_text(json.dumps(history,indent=2)+'\n');os.replace(hist_temp,out/'history.json')
        temp=out/'checkpoint.tmp.npz'
        np.savez(temp,psi=psi,step=i,signature=signature,ground_energy=E,ground=gs,ground_residual=gres)
        os.replace(temp,checkpoint)
    for i in range(begin,stop):
        psi,info=step(lambda t,y:h.apply(y,avec(t),velocity=True),psi,i*dt,dt,
                      tol=config.get('krylov_tolerance',1e-10),maxdim=config.get('krylov_dimension',48))
        if (i+1)%stride==0:flux[(i+1)//stride]=h.flux(psi,avec((i+1)*dt))
        if interrupted[0] and (i+1)%stride==0:stop=i+1
        every=config.get('checkpoint_every',100)
        if ((i+1)%every==0 and (i+1)%stride==0) or i+1==stop:
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
          'surface_indices':idx.tolist(),'seconds_this_invocation':time.perf_counter()-t0}
    (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
    for sig,handler in previous.items():signal.signal(sig,handler)
    return meta

def extract_run(out):
    from surface3d import extract
    out=Path(out);meta=json.loads((out/'run.json').read_text())
    if not meta['complete']:raise ValueError('propagation incomplete')
    config=meta['config'];hr,h=setup(config);h.prepare_surface(config['surface']);avec,pulses=vector_function(config)
    f=np.load(out/'flux.npy',mmap_mode='r');dt=meta['dt']*config.get('surface_stride',1);t=np.arange(len(f))*dt
    energy=np.linspace(*config.get('spectrum_energy',[.3,.9,121]));theta=np.linspace(0,np.pi,61);phi=np.zeros(61)
    labels=[(1,0,0),(2,1,0)] if config.get('M',0)==0 else [(1,0,0),(2,1,-1),(2,1,0),(2,1,1)]
    labels=[tuple(a) for a in config.get('ionic_channels',labels)]
    if any(n<=l or l<0 or l>config['lmax'] or abs(m)>l for n,l,m in labels):
        raise ValueError('ionic_channels must contain valid (n,l,m) within lmax')
    if config.get('M',0) is None:
        # Full solid-angle quadrature: Gauss-Legendre theta and uniform azimuth.
        from numpy.polynomial.legendre import leggauss
        z,w=leggauss(20);ph=np.arange(32)*2*np.pi/32
        theta,phi=np.meshgrid(np.arccos(z),ph,indexing='ij');theta=theta.ravel();phi=phi.ravel()
    b,pes=extract(h,f,t,avec,labels,energy,(theta,phi),dt,substeps=config.get('surface_stride',1))
    if config.get('M',0)==0:
        from scipy.integrate import simpson
        total=2*np.pi*simpson(pes*np.sin(theta),x=theta,axis=2)
    else:total=np.sum(pes.reshape(len(labels),len(energy),20,32)*w[None,None,:,None],axis=(2,3))*2*np.pi/32
    np.savez(out/'spectrum.npz',energy=energy,theta=theta,phi=phi,amplitudes=b,pes=pes,angle_integrated=total,labels=labels)
    return total

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--out',required=True)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--max-steps',type=int)
    parser.add_argument('--extract',action='store_true');a=parser.parse_args()
    if a.extract:extract_run(a.out)
    else:run(json.loads(Path(a.config).read_text()),a.out,a.resume,a.max_steps)
if __name__=='__main__':main()
