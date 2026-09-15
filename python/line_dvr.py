"""Independent 1D two-electron FE-DVR + irECS check, sharing the Fortran tensor kernel.

This isolates FFT/CAP boundary errors from the already independently tested flux algebra.
"""
import json,time
from pathlib import Path
import numpy as np
from scipy.special import roots_jacobi,eval_legendre,roots_genlaguerre,eval_laguerre
from scipy.sparse import csr_matrix,diags,eye
from scipy.sparse.linalg import eigsh,LinearOperator,expm_multiply
from scipy.linalg import eigh,expm
from fedvr import Grid,derivative
from tdse1d import cutoff,ZEN,ZEE,SOFT
from native import tensor_apply
from pulses import Pulse
from propagate import step

def line_grid(edges,order=6,tail=28,angle=.5,alpha=.8):
    edges=np.asarray(edges,float)
    if edges[0]!=0 or np.any(np.diff(edges)<=0) or tail<1 or not 0<=angle<np.pi/2:raise ValueError('invalid line grid')
    both=np.r_[-edges[:0:-1],edges];x=np.r_[-1.,roots_jacobi(order-1,1,1)[0],1.]
    w=2/(order*(order+1)*eval_legendre(order,x)**2);D=derivative(x)
    nf=(len(both)-1)*order+1;n=nf+2*tail
    r=np.zeros(n,complex);weight=np.zeros(n,complex);K=np.zeros((n,n),complex)
    for i,(a,b) in enumerate(zip(both[:-1],both[1:])):
        idx=np.arange(tail+i*order,tail+(i+1)*order+1);h=b-a
        r[idx]=a+h*(x+1)/2;weight[idx]+=h*w/2;K[np.ix_(idx,idx)]+=D.T@(w[:,None]*D)/h
    interior_weights=weight.real.copy()
    y=np.r_[0.,roots_genlaguerre(tail,1)[0]];wy=1/((tail+1)*eval_laguerre(tail,y)**2);wp=wy*np.exp(y)
    dy=derivative(y);np.fill_diagonal(dy,.5);dy[0,0]=-tail/2
    d=dy*np.exp((y[None,:]-y[:,None])/2)-np.eye(tail+1)/2;h=np.exp(1j*angle)/alpha
    for sign,idx in [(-1,np.arange(tail,-1,-1)),(1,np.arange(tail+nf-1,n))]:
        r[idx]=sign*(edges[-1]+h*y);weight[idx]+=h*wp;K[np.ix_(idx,idx)]+=d.T@(wp[:,None]*d)/(2*h)
    K/=np.sqrt(weight[:,None]*weight[None,:]);K[abs(K)<1e-13]=0
    return Grid(r,weight,csr_matrix(K),edges[-1],order,angle,interior_weights)

class SoftDVR:
    def __init__(self,g,radii=(24,30),backend='fortran'):
        self.grid=g;self.r=np.asfortranarray(g.r);self.n=len(g.r);self.nc=1;self.shape=(self.n,self.n,1);self.size=self.n**2
        c=cutoff(self.r.real,*radii);self.ven=-ZEN/np.sqrt(self.r**2+SOFT)*c
        self.vee=ZEE/np.sqrt((self.r[:,None]-self.r[None,:])**2+SOFT)*c[:,None]*c[None,:]
        V=self.ven[:,None]+self.ven[None,:]+self.vee
        self.diag=np.asfortranarray(V[:,:,None]);T=g.kinetic.tocsr()
        self.ptr=T.indptr.astype(np.int32);self.col=T.indices.astype(np.int32);self.tv=T.data.astype(complex)
        self.vptr=np.zeros(2,np.int32);self.vc=np.zeros(0,np.int32);self.vl=np.zeros(0,np.int32);self.vcoef=np.zeros(0,complex)
        self.rad=np.zeros((self.n,self.n,1),complex,order='F')
        self.dptr=np.array([0,2],np.int32);self.dc=np.zeros(2,np.int32);self.de=np.array([1,2],np.int32)
        self.dcoef=np.asfortranarray([[0,0],[0,0],[1,1]],dtype=complex);self.ldiff=np.zeros(2)
        self.ionic0=T+diags(self.ven);self.P=1j*(T@diags(self.r)-diags(self.r)@T)
        self.backend=backend;self._A=None

    def apply(self,x,A=0):
        if self.backend=='fortran':return tensor_apply(self,x,(0,0,A),velocity=True)
        if self._A!=A:
            self._ion=self.ionic0+A*self.P+diags(np.full(self.n,A*A/2));self._A=A
        u=x.reshape(self.n,self.n,order='F')
        return (self._ion@u+(self._ion@u.T).T+self.vee*u).ravel(order='F')

    def ground(self):
        if self.grid.ecs_angle:raise ValueError('ground requires real grid')
        e,u=eigh(self.ionic0.toarray());x=np.outer(u[:,0],u[:,0]).ravel(order='F')
        op=LinearOperator((self.size,)*2,matvec=lambda x:self.apply(x).real,dtype=float)
        ev,vec=eigsh(op,k=1,which='SA',v0=x.real,tol=1e-10,ncv=30)
        return float(ev[0]),vec[:,0].astype(complex),e,u,float(abs(np.dot(u[:,0]*self.r,u[:,1])))

    def surface(self,R):
        theta=(abs(self.r.real)>R).astype(float);T=self.grid.kinetic.toarray()
        J=T*(theta[None,:]-theta[:,None]);self.idx=np.flatnonzero(np.max(abs(J),axis=1)>1e-14)
        if np.any(abs(self.r[self.idx].imag)>1e-12):raise ValueError('complex surface')
        self.J=J[self.idx];self.JP=(1j*J*(self.r[None,:]-self.r[:,None]))[self.idx]

    def flux(self,psi,A):return psi.reshape((self.n,self.n),order='F')@(self.J+A*self.JP).T

    def split(self,psi,A,dt):
        if getattr(self,'_split_dt',None)!=dt:
            self._half=np.exp(-.5j*dt*self.vee);self._split_dt=dt;self._split_A=None
        if self._split_A!=A:
            H=self.ionic0.toarray()+A*self.P.toarray()+A*A/2*np.eye(self.n)
            self._U=expm(-1j*dt*H);self._split_A=A
        u=self._half*psi.reshape(self.n,self.n,order='F')
        return (self._half*(self._U@u@self._U.T)).ravel(order='F')

def run(out,cycles=60,dt=.01,integrator='split'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'flux.npy').exists():raise FileExistsError(out)
    edges=[0,.5,1,2,4,8,12,16,20,24,28,32,36,40]
    real=SoftDVR(line_grid(edges,angle=0),backend='sparse');h=SoftDVR(line_grid(edges),backend='sparse');p=Pulse(cycles=cycles)
    h.surface(34);E,gs,ei,ui,d12=real.ground();ratio=np.sqrt(h.grid.weights/real.grid.weights)
    np.savez(out/'initial.npz',ground=gs,ionic_states=ui,ionic_energies=ei,r=h.r,weights=h.grid.weights)
    psi=(gs.reshape(h.n,h.n,order='F')*ratio[:,None]*ratio[None,:]).ravel(order='F')
    T=p.duration+70;steps=int(np.ceil(T/dt));steps+=(-steps)%2;dt=T/steps;stride=2
    flux=np.lib.format.open_memmap(out/'flux.npy',mode='w+',dtype=complex,shape=(steps//2+1,h.n,len(h.idx)))
    flux[0]=h.flux(psi,0);history=[];start=time.perf_counter()
    print('DVR ground',E,'ion',ei[:2], 'D12',d12,'grid',h.n,flush=True)
    for i in range(steps):
        if integrator=='arnoldi':
            psi,info=step(lambda t,y:h.apply(y,float(p.vector(t))),psi,i*dt,dt,tol=1e-10,maxdim=48)
        else:
            psi=h.split(psi,float(p.vector((i+.5)*dt)),dt);info={'dimension':0}
        if (i+1)%2==0:flux[(i+1)//2]=h.flux(psi,float(p.vector((i+1)*dt)))
        if (i+1)%1000==0 or i+1==steps:
            pg=float(abs(np.vdot(gs,psi))**2);history.append([(i+1)*dt,pg]);flux.flush()
            print(f'DVR {i+1}/{steps}, Pg={pg:.8f}, dim={info["dimension"]}, wall={time.perf_counter()-start:.1f}',flush=True)
    np.savez(out/'initial.npz',ground=gs,ionic_states=ui,ionic_energies=ei,r=h.r,weights=h.grid.weights)
    np.savez(out/'final.npz',psi=psi);np.savetxt(out/'populations.csv',history,delimiter=',',header='time,ground_population')
    meta={'method':'1d_fedvr_irecs','integrator':integrator,'config':{'pulse':p.__dict__,'radial':{'edges':edges,'order':6,'tail':28,'angle':.5,'alpha':.8},'cutoff_radii':[24,30],'surface':34},
          'energy':E,'ionic_energies':ei[:4].tolist(),'d12':d12,'dt':dt,'surface_dt':2*dt,'steps':steps,
          'final_ground_population':pg,'seconds':time.perf_counter()-start,'exchange_error':float(np.linalg.norm(psi.reshape(h.n,h.n)-psi.reshape(h.n,h.n).T))}
    (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
    extract(out)

def extract(out):
    from scipy.integrate import cumulative_trapezoid
    out=Path(out);m=json.loads((out/'run.json').read_text());cfg=m['config'];h=SoftDVR(line_grid(**cfg['radial']),cfg['cutoff_radii']);h.surface(cfg['surface']);p=Pulse(**cfg['pulse'])
    initial=np.load(out/'initial.npz');chi=initial['ionic_states'][:,:2].astype(complex)
    # Real basis coefficients differ only in exponentially small tails of final bound states.
    realgrid=line_grid(**{**cfg['radial'],'angle':0});chi*=np.sqrt(h.grid.weights/realgrid.weights)[:,None]
    data=np.load(out/'flux.npy',mmap_mode='r');dt=m['surface_dt'];t=np.arange(len(data))*dt;A=p.vector(t)
    intA=cumulative_trapezoid(A,t,initial=0);intA2=cumulative_trapezoid(A*A,t,initial=0)
    energy=np.linspace(.45,.75,301);k=np.r_[-np.sqrt(2*energy),np.sqrt(2*energy)]
    test=np.exp(-1j*k[:,None]*h.r[h.idx].real)*np.sqrt(h.grid.weights[h.idx].real)/(2*np.pi)**.5
    b=np.zeros((2,len(k)),complex);Id=eye(h.n,format='csr')
    for i in range(len(t)-1,-1,-1):
        integrand=(chi.conj().T@data[i])@test.T
        phase=np.exp(1j*(k*k*t[i]/2+k*intA[i]+intA2[i]/2))
        b+=1j*np.sqrt(2)*dt*(.5 if i in (0,len(t)-1) else 1)*integrand*phase
        if i:
            for j in range(2):
                a=float(p.vector(t[i]-(j+.5)*dt/2));H=h.ionic0+a*h.P+a*a/2*Id;op=(1j*dt/2*H.conj().T).tocsr()
                chi=expm_multiply(op,chi,traceA=op.diagonal().sum())
    ne=len(energy);P=(abs(b[:,:ne])**2+abs(b[:,ne:])**2)/np.sqrt(2*energy)
    np.savez(out/'spectrum.npz',energy=energy,pes=P,amplitudes_k=b)

if __name__=='__main__':
    import argparse
    q=argparse.ArgumentParser();q.add_argument('--out',required=True);q.add_argument('--dt',type=float,default=.01);q.add_argument('--cycles',type=float,default=60);q.add_argument('--extract',action='store_true');q.add_argument('--integrator',choices=['split','arnoldi'],default='split');a=q.parse_args()
    if a.extract:extract(a.out)
    else:run(a.out,a.cycles,a.dt,a.integrator)
