#!/usr/bin/env python3
"""Cache the ionic projection, then independently refine spectral quadrature."""
import argparse,json,hashlib,os,sys,time,uuid
from pathlib import Path
import numpy as np
from scipy.integrate import simpson
repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'python'))
from run3d import setup,vector_function,electric_function
from surface_projection import project,integrate
from output_lock import exclusive_output
from surface_storage import ShardedArray,TimeSelection,load_flux,capped_npz,file_limit,atomic_json
from streaming_surface import energy_grid,directions,numerical_signature

def source_signature():
    digest=hashlib.sha256()
    for name in ['surface_projection.py','surface3d.py','mixed_gauge.py','implicit.py','angular.py','bipolar.py','ionic_adjoint.py',
                 'run3d.py','pulses.py','fedvr.py','helium3d.py','coupled_angular.py','tdse1d.py','surface_storage.py']:
        digest.update((repo/'python'/name).read_bytes())
    return digest.hexdigest()

@exclusive_output('.projection.lock')
def execute(out,args):
    out=Path(out);meta=json.loads((out/'run.json').read_text())
    if not meta['complete']:raise ValueError('propagation must finish before projection')
    config=meta['config'];A,pulses=vector_function(config)
    labels=config.get('ionic_channels',[[1,0,0],[2,1,0]])
    dt=meta['dt']*config.get('surface_stride',1);gauge=config.get('gauge',{'type':'velocity'})
    mode=config.get('storage',{}).get('mode','legacy_raw');limit=file_limit(config.get('storage',{}).get('max_file_bytes',4_000_000_000))
    if mode=='spectrum':
        if args.energy or args.theta or args.phi or args.angular_rule or args.post_time is not None or args.time_stride!=1 or args.reproject:
            raise ValueError('spectrum-only run retained the final amplitudes, not a time history; use storage.mode=projected for future quadrature refinements')
        if args.name!='spectrum.npz':__import__('shutil').copy2(out/'spectrum.npz',out/args.name)
        print('Reused online final spectrum',out/args.name,flush=True);return
    if mode=='projected':
        layout=json.loads((out/'surface_online/layout.json').read_text());completion=json.loads((out/'surface_online/complete.json').read_text())
        if not completion['complete'] or layout['recipe']['configuration']!=meta['signature']:raise ValueError('incomplete or mismatched online projection')
        if layout['recipe']['kernel']!=numerical_signature():raise ValueError('online numerical code changed; use the recorded snapshot for re-integration')
        old={'nframes':layout['times']['count'],'outer_states':layout['outer_states'],'r':layout['r'],'weights':layout['weights'],
             'projection_final_time':(layout['times']['count']-1)*dt}
        labels=layout['labels'];q=ShardedArray(out/'surface_online/projected')
    else:
        old,q=raw_projection(out,meta,config,args,A,gauge,dt,labels,limit)
    times=np.arange(old['nframes'])*dt
    indices=np.arange(0,len(times),args.time_stride)
    if indices[-1]!=len(times)-1:indices=np.r_[indices,len(times)-1]
    if args.post_time is not None:
        time_limit=max(p.start+p.duration for p,pol in pulses)+args.post_time
        indices=indices[times[indices]<=time_limit+1e-10]
    if len(indices)<2:raise ValueError('insufficient integration times')
    times=times[indices];q=q if np.array_equal(indices,np.arange(len(q))) else TimeSelection(q,indices)
    energy=np.linspace(args.energy[0],args.energy[1],int(args.energy[2])) if args.energy else energy_grid(config)
    if mode=='projected' and not any([args.theta,args.phi,args.angular_rule]):
        theta,phi,weights=directions(config);rule='gauss'
    else:
        axial=config.get('M',0)==0;nt=args.theta or (61 if axial else 20);nf=args.phi or (1 if axial else 32)
        rule=args.angular_rule or ('simpson' if axial else 'gauss')
        if rule=='simpson':
            theta1=np.linspace(0,np.pi,nt);wt=simpson(np.eye(nt),x=theta1,axis=1)*np.sin(theta1)
        else:
            z,wt=np.polynomial.legendre.leggauss(nt);theta1=np.arccos(z[::-1]);wt=wt[::-1]
        phi1=np.arange(nf)*2*np.pi/nf;theta,phi=np.meshgrid(theta1,phi1,indexing='ij');theta=theta.ravel();phi=phi.ravel()
        weights=np.repeat(wt,nf)*2*np.pi/nf
    if len(labels)*len(energy)*len(theta)*32+65536>limit:raise ValueError('requested spectrum exceeds file cap')
    b,pes=integrate(q,[tuple(x) for x in old['outer_states']],np.array(old['r']),np.array(old['weights']),times,A,energy,(theta,phi),dt)
    total=np.sum(pes*weights[None,None,:],axis=2)
    output=out/args.name
    capped_npz(output,limit,energy=energy,theta=theta,phi=phi,solid_angle_weights=weights,amplitudes=b,pes=pes,angle_integrated=total,labels=labels,
               projection_final_time=old['projection_final_time'],flux_integral_end_time=times[-1],source_signature=meta['signature'],
               angular_rule=rule,surface_time_stride=args.time_stride,gauge=gauge['type'])
    print('Wrote',output,flush=True)

def raw_projection(out,meta,config,args,A,gauge,dt,labels,limit):
    cache=out/'projected_channels.npy';record=out/'channel_projection.json'
    recipe={'configuration':meta['signature'],'labels':labels,'gauge':gauge,'dt':dt,'kernel_sha256':source_signature(),'ionic_device':args.ionic_device}
    old=json.loads(record.read_text()) if record.exists() else None
    if old is not None and old['recipe']!=recipe and not args.reproject:
        raise ValueError('projection source/recipe changed; use --reproject after reviewing the change')
    available=old is not None and ((out/old['cache_path']/'array.json').exists() if 'cache_path' in old else cache.exists())
    if not available or args.reproject:
        _,h=setup(config);h.prepare_surface(config['surface']);base=getattr(h,'uncoupled',h);base.prepare_surface(config['surface'])
        flux=load_flux(out);times=np.arange(len(flux))*dt;diagnostics={'requested_times':config.get('ionic_transfer_times',[])}
        factory=None;ionic_field=None
        if gauge['type']=='mixed':
            from mixed_gauge import MixedIonic,parameters
            E=electric_function(config);ionic_field=lambda t:parameters(A,E,t)
            factory=lambda h,m:MixedIonic(h,m,gauge['inner'],gauge['outer'])
        temp=out/'projected_cache'/uuid.uuid4().hex;start=time.perf_counter()
        q,lm,info=project(base,flux,times,A,labels,dt,substeps=config.get('surface_stride',1),
                         angular_transform=getattr(h,'transform',None),propagator=config.get('ionic_propagator','expm'),
                         diagnostics=diagnostics,output=temp,ion_factory=factory,ionic_field=ionic_field,ionic_device=args.ionic_device,max_file_bytes=limit)
        q.close();del q
        old={'recipe':recipe,'nframes':len(times),'outer_states':lm,'r':base.r[base.surface_indices].real.tolist(),
             'weights':base.grid.weights[base.surface_indices].real.tolist(),'projection_final_time':float(times[-1]),
             'projection_seconds':time.perf_counter()-start,'cache_path':str(temp.relative_to(out)),**info}
        # Serialize before promoting the expensive data. A serialization error
        # must never leave a new cache paired with an old completion record.
        serialized=json.dumps(old,indent=2)+'\n'
        diagnostic_text=json.dumps(diagnostics,indent=2)+'\n' if diagnostics['requested_times'] else None
        temporary=out/'channel_projection.tmp.json';temporary.write_text(serialized)
        os.replace(temporary,record)
        if diagnostic_text is not None:(out/'ionic_transfer.json').write_text(diagnostic_text)
        print('Projected',out.name,'in',old['projection_seconds'],'seconds',flush=True)
    return old,ShardedArray(out/old['cache_path']) if 'cache_path' in old else np.load(cache,mmap_mode='r')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--name',default='spectrum.npz')
    p.add_argument('--energy',nargs=3,type=float);p.add_argument('--theta',type=int);p.add_argument('--phi',type=int)
    p.add_argument('--angular-rule',choices=['simpson','gauss']);p.add_argument('--post-time',type=float)
    p.add_argument('--time-stride',type=int,default=1);p.add_argument('--reproject',action='store_true')
    p.add_argument('--ionic-device',help='optional Torch CPU/CUDA adjoint solver; default is sparse LU')
    args=p.parse_args()
    if args.time_stride<1:raise ValueError('time stride must be positive')
    execute(args.out,args)
