"""Two-electron soft-Coulomb TDSE, parameters exactly from Yu & Madsen (2018).

Fourier split operator in velocity gauge; Fortran/OpenMP real-time propagation.
Smooth interaction truncation makes the Volkov exterior Hamiltonian exact beyond
cutoff_outer for THIS truncated model. Cutoff convergence remains a physical check.
"""
from pathlib import Path
import json,time
import numpy as np
from scipy.fft import fft,ifft,fft2,ifft2,fftfreq
from scipy.sparse.linalg import LinearOperator,eigsh
from scipy.linalg import eigh
from native import split_stage,split_cached
from pulses import Pulse

ZEN=1.1225;ZEE=.6317;SOFT=.09169

def cutoff(x,inner,outer):
    if not 0<inner<outer:raise ValueError('0<cutoff_inner<cutoff_outer')
    u=np.clip((abs(x)-inner)/(outer-inner),0,1)
    return 1-10*u**3+15*u**4-6*u**5

class Model:
    def __init__(self,n=512,halfbox=40.,cutoff_inner=None,cutoff_outer=None,cap_start=None,cap_strength=.5):
        if n<16 or n&(n-1):raise ValueError('n must be power of 2 >=16')
        self.n=n;self.dx=2*halfbox/n
        self.x=(np.arange(n)-n//2)*self.dx
        self.p=2*np.pi*fftfreq(n,self.dx)
        g=np.ones(n) if cutoff_inner is None else cutoff(self.x,cutoff_inner,cutoff_outer)
        self.ven=-ZEN/np.sqrt(self.x**2+SOFT)*g
        self.vee=ZEE/np.sqrt((self.x[:,None]-self.x[None,:])**2+SOFT)*g[:,None]*g[None,:]
        v=self.ven[:,None]+self.ven[None,:]+self.vee
        self.v=np.asfortranarray(v,dtype=complex)
        self.cap=np.zeros(n)
        if cap_start is not None:
            if not 0<cap_start<halfbox:raise ValueError('CAP must lie within the box')
            self.cap=cap_strength*np.clip((abs(self.x)-cap_start)/(halfbox-cap_start),0,1)**4
            self.v-=1j*(self.cap[:,None]+self.cap[None,:])
        self.kin=(self.p[:,None]**2+self.p[None,:]**2)/2

    def apply(self,y):
        y=np.asarray(y).reshape(self.n,self.n)
        return (ifft2(self.kin*fft2(y))+self.v.real*y).ravel()

    def ground(self,tol=1e-10):
        n=self.n;v0=np.exp(-self.x[:,None]**2-self.x[None,:]**2).ravel()
        op=LinearOperator((n*n,n*n),matvec=lambda y:self.apply(y).real,dtype=float)
        e,v=eigsh(op,k=1,which='SA',v0=v0,ncv=25,tol=tol,maxiter=10000)
        psi=np.asfortranarray(v[:,0].reshape(n,n)/self.dx,dtype=complex)
        return float(e[0]),psi,float(np.linalg.norm(self.apply(v[:,0])-e[0]*v[:,0]))

    def ionic(self):
        T=ifft((self.p**2/2)[:,None]*fft(np.eye(self.n),axis=0),axis=0).real
        e,u=eigh(T+np.diag(self.ven))
        u/=np.sqrt(self.dx)
        d12=abs(np.dot(u[:,0]*self.x,u[:,1])*self.dx)
        return e,u,float(d12)

    def step(self,psi,t,dt,pulse,trace=None):
        if trace is None:
            if not hasattr(self,'_cached_dt') or self._cached_dt!=dt:
                self._vhalf=np.asfortranarray(np.exp(-.5j*dt*self.v));self._cached_dt=dt
            pp=np.exp(-.5j*dt*(self.p+float(pulse.vector(t+dt/2)))**2)
            split_cached(psi,self._vhalf,pp)
            return
        for stage in (1,2,3):
            split_stage(psi,self.v,self.p,dt,float(pulse.vector(t+dt/2)),stage)
            if trace is not None:trace(stage,psi.copy())

    def surface(self,psi,radius):
        idx=np.array([int(np.argmin(abs(self.x+radius))),int(np.argmin(abs(self.x-radius)))])
        if idx[0]<4 or idx[1]+4>=self.n:raise ValueError('surface derivative stencil outside box')
        # 8th-order central derivative, independently tested with free packets.
        deriv=np.zeros((self.n,2),complex)
        coeff=np.array([4/5,-1/5,4/105,-1/280])
        for k,c in enumerate(coeff,1):deriv+=c*(psi[:,idx+k]-psi[:,idx-k])/self.dx
        return np.stack([psi[:,idx],deriv]),self.x[idx]

def run(config,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'surface.npy').exists() or (out/'run.json').exists():raise FileExistsError('use a fresh output directory')
    pulse=Pulse(**config['pulse']);m=Model(**config['grid'])
    radius=config['surface'];dt=config.get('dt',.04);tail=config.get('post_time',60.)
    if not config['grid']['cutoff_outer']<radius<config['grid']['cap_start']:
        raise ValueError('requires cutoff_outer < surface < cap_start')
    steps=int(np.ceil((pulse.duration+tail)/dt));dt=(pulse.duration+tail)/steps
    stride=config.get('surface_stride',1)
    if steps%stride:steps+=stride-steps%stride;dt=(pulse.duration+tail)/steps
    t0=time.perf_counter();E,gs,res=m.ground();ei,ui,d12=m.ionic();psi=gs.copy(order='F')
    np.savez(out/'initial.npz',ground=gs,ionic_energies=ei,ionic_states=ui,x=m.x)
    hist=np.lib.format.open_memmap(out/'surface.npy',mode='w+',dtype='complex128',shape=(steps//stride+1,2,m.n,2))
    hist[0],positions=m.surface(psi,radius)
    records=[]
    for i in range(steps):
        m.step(psi,i*dt,dt,pulse)
        if (i+1)%stride==0:hist[(i+1)//stride]=m.surface(psi,radius)[0]
        if (i+1)%max(1,steps//100)==0 or i+1==steps:
            norm=float(np.sum(abs(psi)**2)*m.dx**2);pg=float(abs(np.vdot(gs,psi)*m.dx**2)**2)
            records.append([dt*(i+1),norm,pg])
        if (i+1)%max(1,steps//10)==0:
            hist.flush();print(f'1D {i+1}/{steps}, norm={norm:.8f}, elapsed={time.perf_counter()-t0:.1f}s',flush=True)
    hist.flush();np.savez(out/'final.npz',psi=psi)
    np.savetxt(out/'populations.csv',records,delimiter=',',header='time,norm,ground_population')
    meta={'config':config,'energy':E,'ground_residual':res,'ionic_energies':ei[:4].tolist(),'d12':d12,
          'dt':dt,'surface_dt':dt*stride,'steps':steps,'positions':positions.tolist(),
          'seconds':time.perf_counter()-t0,'exchange_error':float(np.linalg.norm(psi-psi.T)*m.dx),
          'final_norm':norm,'final_ground_population':pg}
    (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta

def extract(out,energy=None,nchannels=2,ionic_substeps=None):
    """Two-electron tSURFF with backwards propagated final ionic channel functions.

    The bound electron evolves under the FULL 1D ionic Hamiltonian, not a frozen 1s
    projection during the pulse. Factor sqrt(2) counts either identical electron.
    """
    out=Path(out);meta=json.loads((out/'run.json').read_text());config=meta['config']
    pulse=Pulse(**config['pulse']);m=Model(**config['grid']);dt=meta['surface_dt']
    substeps=config.get('surface_stride',1) if ionic_substeps is None else ionic_substeps
    a=np.load(out/'initial.npz');chi=a['ionic_states'][:,:nchannels].astype(complex)
    data=np.load(out/'surface.npy',mmap_mode='r');nt=len(data);t=np.arange(nt)*dt
    energy=np.linspace(.45,.75,301) if energy is None else np.asarray(energy)
    if np.any(energy<=0):raise ValueError('positive energies required for dP/dE Jacobian')
    k=np.r_[-np.sqrt(2*energy),np.sqrt(2*energy)]
    A=pulse.vector(t)
    from scipy.integrate import cumulative_trapezoid
    intA=cumulative_trapezoid(A,t,initial=0);intA2=cumulative_trapezoid(A*A,t,initial=0)
    b=np.zeros((nchannels,len(k)),complex);positions=np.asarray(meta['positions'])
    # Chi(t)=U(T,t)^dagger phi. Backwards propagation uses H_ion^dagger.
    v=m.ven+1j*m.cap
    for i in range(nt-1,-1,-1):
        projected=np.einsum('xn,axs->ans',chi.conj(),data[i])*m.dx
        phase=np.exp(1j*(k*k*t[i]/2+k*intA[i]+intA2[i]/2))
        integrand=np.zeros_like(b)
        for side,sign in enumerate((-1,1)):
            test=phase*np.exp(-1j*k*positions[side])/np.sqrt(2*np.pi)
            integrand+=sign*test[None,:]*((k[None,:]/2+A[i])*projected[0,:,side,None]-.5j*projected[1,:,side,None])
        b+=np.sqrt(2)*dt*(.5 if i in (0,nt-1) else 1)*integrand
        if i:
            d=-dt/substeps
            for j in range(substeps):
                amid=float(pulse.vector(t[i]+(j+.5)*d))
                half=np.exp(-1j*d*v/2)[:,None]
                chi=half*ifft(np.exp(-1j*d*(m.p+amid)**2/2)[:,None]*fft(half*chi,axis=0),axis=0)
    ne=len(energy);pes=(abs(b[:,:ne])**2+abs(b[:,ne:])**2)/np.sqrt(2*energy)[None,:]
    np.savez(out/'spectrum.npz',energy=energy,amplitudes_k=b,pes=pes)
    return energy,pes
