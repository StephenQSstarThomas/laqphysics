"""Online channel-resolved tSURFF with small backward-ionic checkpoints.

Known pulses fix chi_c(t)=U_ion(T,t)^dagger phi_c. A backward pass saves only
block endpoints; a bounded RAM block is replayed during forward two-electron
propagation. Each surface source is contracted immediately. No raw flux history
is needed. Volkov time integrals and complex amplitudes retain all interference.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.integrate import simpson
from scipy.special import spherical_jn,sph_harm_y
from surface3d import Ionic
from surface_contract import SurfaceContractor
from surface_storage import ShardedArray,atomic_json,capped_npz,file_limit

def numerical_signature():
    root=Path(__file__).parent;h=hashlib.sha256()
    for name in ['streaming_surface.py','surface_contract.py','surface_storage.py','surface3d.py','ionic_adjoint.py',
                 'implicit.py','mixed_gauge.py','angular.py','fedvr.py','tdse1d.py','pulses.py',
                 'run3d.py','helium3d.py','coupled_angular.py','bipolar.py','torch_backend.py','native.py','ground_s.py','ionic_replay.py']:
        h.update(name.encode()+b'\0');h.update((root/name).read_bytes())
    h.update((root.parent/'src/kernels.f90').read_bytes())
    return h.hexdigest()

def directions(config):
    s=config.get('spectrum',{});axial=config.get('M',0)==0
    nt=int(s.get('theta_points',24 if not axial else 32));nf=int(s.get('phi_points',1 if axial else 32))
    if nt<2 or nf<1 or (not axial and nf<2):raise ValueError('invalid angular quadrature')
    z,w=np.polynomial.legendre.leggauss(nt);t,p=np.meshgrid(np.arccos(z),np.arange(nf)*2*np.pi/nf,indexing='ij')
    return t.ravel(),p.ravel(),np.repeat(w,nf)*2*np.pi/nf

def energy_grid(config):
    segments=config.get('spectrum',{}).get('energy_segments')
    result=np.unique(np.concatenate([np.linspace(float(a),float(b),int(n)) for a,b,n in segments])) if segments else np.linspace(*config.get('spectrum_energy',[.01,1.,397]))
    if len(result)<3 or np.any(result<=0) or not np.isfinite(result).all():raise ValueError('invalid positive energy grid')
    return result

class VolkovAccumulator:
    def __init__(self,contractor,config,avec,max_file_bytes):
        h=contractor.h;self.labels=contractor.labels;self.avec=avec
        self.energy=energy_grid(config);self.k=np.sqrt(2*self.energy)
        if np.any(self.energy<=0):raise ValueError('spectrum energies must be positive')
        self.theta,self.phi,self.solid_angle_weights=directions(config)
        self.unit=np.array([np.sin(self.theta)*np.cos(self.phi),np.sin(self.theta)*np.sin(self.phi),np.cos(self.theta)]).T
        count=len(self.labels)*len(self.energy)*len(self.theta)
        if 2*count*16+32768>max_file_bytes:raise ValueError('spectral accumulator exceeds file cap; reduce or tile energy/angle grid')
        r=h.r[h.surface_indices].real;w=h.grid.weights[h.surface_indices].real
        self.tests=np.array([np.sqrt(2/np.pi)*(-1j)**l*np.sqrt(w)[None,:]*r[None,:]*spherical_jn(l,self.k[:,None]*r[None,:]) for l,m in contractor.outerstates])
        self.harmonics=np.array([sph_harm_y(l,m,self.theta,self.phi) for l,m in contractor.outerstates])
        self.amplitude=np.zeros((len(self.labels),len(self.energy),len(self.theta)),complex)
        self.previous=np.zeros_like(self.amplitude);self.intA=np.zeros(3);self.intA2=0.;self.time=0.;self.index=-1

    def add(self,index,time,q):
        if index!=self.index+1:raise ValueError('nonsequential spectrum samples')
        if self.index>=0:
            d=time-self.time;before=self.avec(self.time);after=self.avec(time)
            self.intA+=d*(before+after)/2;self.intA2+=d*(np.dot(before,before)+np.dot(after,after))/2
        else:d=0.
        radial=np.einsum('jls,les->jel',q,self.tests)
        phase=np.exp(1j*(self.energy[:,None]*time+self.k[:,None]*(self.unit@self.intA)[None,:]+self.intA2/2))
        current=1j*np.sqrt(2)*(radial@self.harmonics)*phase[None,:,:]
        if self.index>=0:self.amplitude+=d*(self.previous+current)/2
        self.previous=current;self.time=float(time);self.index=int(index)

    def payload(self):
        return {'amplitude':self.amplitude,'previous_integrand':self.previous,'intA':self.intA,
                'intA2':self.intA2,'time':self.time,'sample_index':self.index}

    def restore(self,data):
        self.amplitude=data['amplitude'].copy();self.previous=data['previous_integrand'].copy()
        self.intA=data['intA'].copy();self.intA2=float(data['intA2']);self.time=float(data['time']);self.index=int(data['sample_index'])

    def write(self,path,signature,max_file_bytes):
        pes=abs(self.amplitude)**2*self.k[None,:,None]
        total=np.sum(pes*self.solid_angle_weights[None,None,:],axis=2)
        capped_npz(path,max_file_bytes,energy=self.energy,theta=self.theta,phi=self.phi,
                   solid_angle_weights=self.solid_angle_weights,amplitudes=self.amplitude,pes=pes,
                   angle_integrated=total,labels=self.labels,source_signature=signature,
                   extraction_end_time=self.time,projection_final_time=self.time,
                   spectrum_scope='single ionization into the explicitly recorded bound ionic channels',
                   angular_rule='gauss',storage_method='online adjoint block replay')

class TorchVolkovAccumulator(VolkovAccumulator):
    """The same coherent integral on the allocated GPU, entirely complex128."""
    def __init__(self,contractor,config,avec,max_file_bytes,device):
        super().__init__(contractor,config,avec,max_file_bytes)
        import torch
        self.torch=torch;self.device=torch.device(device)
        self.tensor=lambda x:torch.as_tensor(np.array(x,copy=True),dtype=torch.complex128,device=self.device)
        self.amplitude=self.tensor(self.amplitude);self.previous=self.tensor(self.previous)
        self.tests_gpu=self.tensor(self.tests);self.harmonics_gpu=self.tensor(self.harmonics)
        self.energy_gpu=torch.as_tensor(self.energy,dtype=torch.float64,device=self.device)
        self.k_gpu=torch.as_tensor(self.k,dtype=torch.float64,device=self.device)
        self.unit_gpu=torch.as_tensor(self.unit,dtype=torch.float64,device=self.device)

    def add(self,index,time,q):
        torch=self.torch
        if index!=self.index+1:raise ValueError('nonsequential spectrum samples')
        if self.index>=0:
            d=time-self.time;before=self.avec(self.time);after=self.avec(time)
            self.intA+=d*(before+after)/2;self.intA2+=d*(np.dot(before,before)+np.dot(after,after))/2
        else:d=0.
        radial=torch.einsum('jls,les->jel',self.tensor(q),self.tests_gpu)
        displacement=self.unit_gpu@torch.as_tensor(self.intA,dtype=torch.float64,device=self.device)
        phase=torch.exp(1j*(self.energy_gpu[:,None]*time+self.k_gpu[:,None]*displacement[None,:]+self.intA2/2))
        current=1j*np.sqrt(2)*(radial@self.harmonics_gpu)*phase[None,:,:]
        if self.index>=0:self.amplitude+=d*(self.previous+current)/2
        self.previous=current;self.time=float(time);self.index=int(index)

    def payload(self):
        return {'amplitude':self.amplitude.cpu().numpy(),'previous_integrand':self.previous.cpu().numpy(),
                'intA':self.intA,'intA2':self.intA2,'time':self.time,'sample_index':self.index}

    def restore(self,data):
        self.amplitude=self.tensor(data['amplitude']);self.previous=self.tensor(data['previous_integrand'])
        self.intA=data['intA'].copy();self.intA2=float(data['intA2']);self.time=float(data['time']);self.index=int(data['sample_index'])

    def write(self,path,signature,max_file_bytes):
        device_array=self.amplitude
        try:
            self.amplitude=device_array.cpu().numpy();super().write(path,signature,max_file_bytes)
        finally:self.amplitude=device_array

class AdjointStepper:
    def __init__(self,ion,field,method='cf4',device='lu'):
        self.ion=ion;self.field=field;self.method=method;self.device=device;self.maximum_residual=0.;self.engine=None
        if method=='cf4' and device=='lu':
            from implicit import SparseCF4
            self.solver=SparseCF4(lambda t:ion.matrix(field(t)).conj().T,key_at=lambda t:tuple(field(t)))
        elif method!='cf4' or device not in ('cpu','cuda:0'):raise ValueError('online projection uses cf4 and ionic_device lu/cpu/cuda:0')

    def reset(self,chi):
        if self.device!='lu':
            from ionic_adjoint import IonicAdjoint
            if self.engine is None:self.engine=IonicAdjoint(self.ion,chi.shape[1],self.device)
            self.state=self.engine.tensor(chi).reshape(-1)
        self.chi=chi.copy()

    def step(self,t,dt):
        if self.device=='lu':self.chi=self.solver.step(self.chi,t,dt)
        else:
            from implicit import cf4_step
            self.state,info=cf4_step(self.engine,self.state,lambda s:self.engine.coefficients(self.field(s)),t,dt,0.,tol=1e-12)
            self.maximum_residual=max(self.maximum_residual,info['linear_residual']);self.chi=self.engine.host(self.state)
        return self.chi

class OnlineSurface:
    def __init__(self,h,config,out,times,avec,ionic_field=None,ion_factory=None,resume=False,accumulator_device='cpu'):
        self.out=Path(out);self.config=config;self.times=np.asarray(times);self.avec=avec
        self.sample_dt=float(times[1]-times[0])
        if not np.allclose(np.diff(times),self.sample_dt,rtol=1e-10,atol=1e-12):raise ValueError('online propagation requires uniform surface sampling')
        settings=config['storage'];self.mode=settings.get('mode','spectrum');self.limit=file_limit(settings.get('max_file_bytes',4_000_000_000))
        if self.mode not in ('spectrum','projected'):raise ValueError('unknown online storage mode')
        self.block=int(settings.get('ionic_block_frames',128))
        if self.block<1:raise ValueError('ionic block must be positive')
        self.workers=int(settings.get('replay_workers',0));self.pool=None;self.futures={}
        if not 0<=self.workers<=64:raise ValueError('invalid CPU replay worker count')
        if self.workers and settings.get('ionic_device','cpu').startswith('cuda'):
            raise ValueError('parallel replay workers must use a CPU ionic solver')
        self.signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();self.kernel=numerical_signature()
        self.folder=self.out/'surface_online';self.folder.mkdir(parents=True,exist_ok=True)
        labels=[tuple(x) for x in config.get('ionic_channels',[[1,0,0],[2,1,0]])]
        if any(n<=l or l<0 or l>config['lmax'] or abs(m)>l for n,l,m in labels):raise ValueError('invalid ionic projection channel')
        folded=all(p.get('polarization','z')=='z' for p in config['pulses'])
        unique=list(dict.fromkeys((n,l,abs(m) if folded else m) for n,l,m in labels));columns=[unique.index((n,l,abs(m) if folded else m)) for n,l,m in labels]
        base=getattr(h,'uncoupled',h);base.prepare_surface(config['surface']);magnetic={m for n,l,m in unique} if folded else None
        self.ion=(ion_factory or Ionic)(base,magnetic)
        self.contract=SurfaceContractor(base,self.ion,labels,unique,columns,folded,getattr(h,'transform',None))
        self.stepper=AdjointStepper(self.ion,ionic_field or avec,config.get('ionic_propagator','cf4'),settings.get('ionic_device','cpu'))
        self.ends=np.unique(np.r_[np.arange(0,len(times),self.block),len(times)-1]);self.end_lookup={int(v):i for i,v in enumerate(self.ends)}
        self.record=self.folder/'preparation.json';endpoint_dir=self.folder/'ionic_endpoints'
        recipe={'configuration':self.signature,'kernel':self.kernel,'frames':len(times),'block':self.block,'ionic_device':self.stepper.device}
        if self.record.exists():
            old=json.loads(self.record.read_text())
            if old['recipe']!=recipe:raise ValueError('online projection recipe changed; use a fresh output directory')
            self.lowest=old['lowest_ready_frame'];self.stepper.maximum_residual=old.get('maximum_ionic_residual',0.)
            self.endpoints=ShardedArray(endpoint_dir,mode='r+')
        else:
            if resume:raise ValueError('missing online surface preparation for resume')
            self.endpoints=ShardedArray(endpoint_dir,(len(self.ends),self.ion.na*base.n,len(unique)),mode='w+',max_file_bytes=self.limit)
            self.lowest=len(times)-1;self.endpoints[-1]=self.ion.dual_final_states(unique);self.endpoints.flush()
            atomic_json(self.record,{'recipe':recipe,'lowest_ready_frame':self.lowest,'complete':False,'maximum_ionic_residual':0.})
        self.recipe=recipe;self.cached_block=None;self.cached_chi=None;self.generation=-1
        self.accumulator=(TorchVolkovAccumulator(self.contract,config,avec,self.limit,accumulator_device)
                          if str(accumulator_device).startswith('cuda') else VolkovAccumulator(self.contract,config,avec,self.limit))
        self.projected=None
        if self.mode=='projected':
            folder=self.folder/'projected';mode='r+' if (folder/'array.json').exists() else 'w+'
            self.projected=ShardedArray(folder,(len(times),*self.contract.shape),mode=mode,max_file_bytes=self.limit)
        atomic_json(self.folder/'layout.json',{'recipe':recipe,'mode':self.mode,'labels':labels,'outer_states':self.contract.outerstates,
                     'r':base.r[base.surface_indices].real.tolist(),'weights':base.grid.weights[base.surface_indices].real.tolist(),
                     'times':{'dt':float(times[1]-times[0]),'count':len(times)},'max_file_bytes':self.limit,
                     'raw_flux_written':False,'accumulator_device':str(accumulator_device),
                     'note':'Ionic endpoints plus bounded RAM replay; complex Volkov amplitudes accumulated online.'})
        self.transfers=[]

    def _backstep(self,i):
        d=self.sample_dt/self.config.get('surface_stride',1)
        for sub in range(self.config.get('surface_stride',1)):self.stepper.step(self.times[i]-sub*d,-d)
        return self.stepper.chi

    def prepare(self,stop_requested=lambda:False):
        if self.lowest==0:return True
        self.stepper.reset(np.array(self.endpoints[self.end_lookup[self.lowest]]))
        for i in range(self.lowest,0,-1):
            chi=self._backstep(i)
            if i-1 in self.end_lookup:
                self.endpoints[self.end_lookup[i-1]]=chi;self.endpoints.flush();self.lowest=i-1
                atomic_json(self.record,{'recipe':self.recipe,'lowest_ready_frame':self.lowest,'complete':self.lowest==0,
                            'maximum_ionic_residual':self.stepper.maximum_residual})
                if self.lowest%(self.block*20)==0:print('IONIC PREP',self.lowest,'frames remaining',flush=True)
                if stop_requested():return False
        return True

    def chi_at(self,index):
        if self.lowest!=0:raise ValueError('ionic preparation is incomplete')
        # B=1 is the explicit speed/storage tradeoff: keep every ionic state,
        # in bounded shards, and do no replay. B>1 retains only endpoints.
        if index in self.end_lookup:return np.asarray(self.endpoints[self.end_lookup[index]])
        block=index//self.block;start=int(self.ends[block]);end=int(self.ends[block+1])
        if self.cached_block!=block:
            if self.workers:
                self._prefetch(block)
                self.cached_chi,residual=self.futures.pop(block).result()
                self.stepper.maximum_residual=max(self.stepper.maximum_residual,residual)
                self._prefetch(block+1)
            else:
                chi=np.array(self.endpoints[block+1]);self.stepper.reset(chi)
                self.cached_chi=np.empty((end-start+1,*chi.shape),complex);self.cached_chi[-1]=chi
                for i in range(end,start+1,-1):self.cached_chi[i-1-start]=self._backstep(i)
            self.cached_block=block
        return self.cached_chi[index-start]

    def _prefetch(self,index):
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        from ionic_replay import initialize,block
        if self.pool is None:
            self.pool=ProcessPoolExecutor(max_workers=self.workers,mp_context=get_context('spawn'),initializer=initialize,
                initargs=(self.config,self.times,self.ends,str(self.folder/'ionic_endpoints')))
        for i in range(index,min(len(self.ends)-1,index+2*self.workers)):
            if i not in self.futures:self.futures[i]=self.pool.submit(block,i)

    def close(self):
        if self.pool is not None:
            for future in self.futures.values():future.cancel()
            self.pool.shutdown(wait=True,cancel_futures=True);self.pool=None;self.futures.clear()
        self.endpoints.close()
        if self.projected is not None:self.projected.close()

    def record_frame(self,index,frame):
        chi=self.chi_at(index);q=self.contract(chi,frame)
        self.accumulator.add(index,self.times[index],q)
        if self.projected is not None:self.projected[index]=q
        for requested in self.config.get('ionic_transfer_times',[]):
            closest=int(np.argmin(abs(self.times-requested)))
            if closest==index:
                initial=self.ion.final_states([(1,0,0)])[:,0];amplitude=(chi.conj().T@initial)[self.contract.columns]
                if self.contract.folded:amplitude=np.where(np.array(self.contract.labels)[:,2]==0,amplitude,0.)
                self.transfers.append({'requested_time':requested,'actual_time':float(self.times[index]),'labels':self.contract.labels,
                    'amplitude_real':amplitude.real.tolist(),'amplitude_imag':amplitude.imag.tolist(),'probabilities':(abs(amplitude)**2).tolist(),
                    'scope':'transition from prepared ionic 1s into recorded final bound channels'})

    def stage_checkpoint(self):
        bank=1 if self.generation==0 else 0
        if self.projected is not None:self.projected.flush()
        capped_npz(self.folder/f'accumulator_{bank}.npz',self.limit,**self.accumulator.payload(),signature=self.signature,kernel=self.kernel,
                   transfers=json.dumps(self.transfers),maximum_ionic_residual=self.stepper.maximum_residual)
        return bank

    def restore(self,bank,step):
        with np.load(self.folder/f'accumulator_{bank}.npz') as d:
            if str(d['signature'])!=self.signature or str(d['kernel'])!=self.kernel:raise ValueError('spectrum checkpoint signature mismatch')
            if int(d['sample_index'])!=step and not(step==0 and int(d['sample_index'])==-1):raise ValueError('wavefunction/spectrum checkpoint mismatch')
            self.accumulator.restore(d);self.transfers=json.loads(str(d['transfers']))
            self.stepper.maximum_residual=max(self.stepper.maximum_residual,float(d['maximum_ionic_residual']))
        self.generation=bank

    def finish(self):
        if self.accumulator.index!=len(self.times)-1:raise ValueError('spectrum integral incomplete')
        self.accumulator.write(self.out/'spectrum.npz',self.signature,self.limit)
        atomic_json(self.out/'ionic_transfer.json',{'requested_times':self.config.get('ionic_transfer_times',[]),'transfers_from_1s':self.transfers})
        atomic_json(self.folder/'complete.json',{'recipe':self.recipe,'complete':True,'sample_index':self.accumulator.index,
                    'maximum_ionic_residual':self.stepper.maximum_residual,'raw_flux_written':False})
