"""Multichannel single-ionization tSURFF on the 3D two-electron DVR grid."""
import numpy as np
from scipy.linalg import eigh,expm
from scipy.special import spherical_jn,sph_harm_y
from scipy.sparse import csr_matrix,diags,block_diag,bmat,eye
from angular import cartesian

class Ionic:
    def __init__(self,h,magnetic_numbers=None):
        self.h=h;self.states=sorted(set(a for a,b in h.ch if magnetic_numbers is None or a[1] in magnetic_numbers));self.n=h.n;self.na=len(self.states)
        self.index={a:i for i,a in enumerate(self.states)}
        T=h.grid.kinetic.tocsr();r=h.r
        self.h0=block_diag([T+diags(-2*h.cut/r+l*(l+1)/(2*r*r)) for l,m in self.states],format='csr')
        blocks=[[[csr_matrix((self.n,self.n),dtype=complex) if a==b else None for b in range(self.na)] for a in range(self.na)] for axis in range(3)]
        base=T@diags(r)-diags(r)@T
        for a,(l,m) in enumerate(self.states):
            for b,(ll,mm) in enumerate(self.states):
                coef=cartesian((l,m),(ll,mm))
                if np.max(abs(coef))<1e-14:continue
                pm=1j*(base+diags((l*(l+1)-ll*(ll+1))/(2*r)))
                for axis in range(3):
                    if abs(coef[axis])>1e-14:blocks[axis][a][b]=coef[axis]*pm
        self.P=[bmat(axis,format='csr') for axis in blocks]
        self.T=T
        self.identity=eye(self.n*self.na,format='csr')

    def final_states(self,labels):
        from angular import radial_hydrogen
        result=[]
        for n,l,m in labels:
            z=np.zeros((self.na,self.n),complex)
            # Analytic Coulomb bound state, evaluated on the analytic ECS contour.
            z[self.index[(l,m)]]=self.h.r*radial_hydrogen(n,l,self.h.r)*np.sqrt(self.h.grid.weights)
            result.append(z.ravel())
        return np.array(result).T

    def dual_final_states(self,labels):
        # Radial bound functions are real on the physical contour. Their analytic
        # continuation is paired without conjugating the complex radial argument:
        # b = phi_theta.T @ psi_theta. Adjoint backward propagation therefore
        # starts from phi_theta.conj(), not from the ket phi_theta.
        return self.final_states(labels).conj()

    def matrix(self,A):
        H=self.h0+self.identity*np.dot(A,A)/2
        for a,P in zip(A,self.P):
            if abs(a)>0:H=H+a*P
        return H.tocsr()

def extract(h,flux,times,avec,labels,energy,directions,dt,substeps=1,angular_transform=None,propagator='expm',diagnostics=None):
    """Backward adjoint ionic propagation; exact discrete commutator integration.

    directions=(theta,phi); dP/dE/dOmega = k |b(k,Omega)|^2. The sqrt(2)
    exchange factor applies to the spin-singlet symmetric spatial wavefunction.
    """
    from scipy.integrate import cumulative_trapezoid
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import expm_multiply
    energy=np.asarray(energy);k=np.sqrt(2*energy);theta,phi=directions
    unit=np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]).T
    av=np.array([avec(t) for t in times]);intA=cumulative_trapezoid(av,times,axis=0,initial=0)
    if np.max(abs(av[-1]))>1e-10:
        raise ValueError('Bound-state tSURFF extraction requires zero final vector potential; a residual impulse needs a final-state gauge transformation')
    # A z-only pulse preserves the ionic magnetic quantum number exactly. Keep the
    # target m sectors only; for elliptical/circular fields retain the full basis.
    magnetic={m for n,l,m in labels} if np.max(abs(av[:,:2]))<1e-14 else None
    ion=Ionic(h,magnetic);chi=ion.dual_final_states(labels)
    ionic_step=None
    if propagator=='cf4':
        from implicit import SparseCF4
        ionic_step=SparseCF4(lambda t:ion.matrix(avec(t)).conj().T,key_at=lambda t:tuple(avec(t)))
    elif propagator!='expm':raise ValueError('ionic propagator must be expm or cf4')
    intA2=cumulative_trapezoid(np.sum(av*av,axis=1),times,initial=0)
    b=np.zeros((len(labels),len(k),len(theta)),complex)
    idx=h.surface_indices;r=h.r[idx].real;w=h.grid.weights[idx].real
    active=[c for c,(inner,outer) in enumerate(h.ch) if inner in ion.index]
    active_inner=[ion.index[h.ch[c][0]] for c in active]
    outstates=sorted(set(h.ch[c][1] for c in active))
    tests={lm:np.sqrt(2/np.pi)*(-1j)**lm[0]*np.sqrt(w)[None,:]*r[None,:]*spherical_jn(lm[0],k[:,None]*r[None,:]) for lm in outstates}
    harmonics=np.array([sph_harm_y(lm[0],lm[1],theta,phi) for lm in outstates])
    requests={int(np.argmin(abs(times-t))):float(t) for t in diagnostics.get('requested_times',[])} if diagnostics is not None else {}
    initial_1s=ion.final_states([(1,0,0)])[:,0] if requests and (0,0) in ion.index else None
    if diagnostics is not None:diagnostics['transfers_from_1s']=[]
    for i in range(len(times)-1,-1,-1):
        if i in requests:
            amplitude=chi.conj().T@initial_1s if initial_1s is not None else np.zeros(len(labels),complex)
            diagnostics['transfers_from_1s'].append({'requested_time':requests[i],'actual_time':float(times[i]),
                'labels':labels,'amplitude_real':amplitude.real.tolist(),'amplitude_imag':amplitude.imag.tolist(),
                'probabilities':(abs(amplitude)**2).tolist(),
                'scope':'transition to recorded final bound states; omitted bound states and ionization are not traced here'})
        v=chi.reshape(ion.na,h.n,len(labels))
        current=flux[i][:,:,active] if angular_transform is None else np.tensordot(flux[i],angular_transform[active].T,axes=([-1],[0]))
        projected=np.einsum('ria,rsa->isa',v[active_inner].conj().transpose(1,2,0),current)
        grouped={lm:np.zeros((len(labels),len(idx)),complex) for lm in outstates}
        for a,c in enumerate(active):grouped[h.ch[c][1]]+=projected[:,:,a]
        # Sum core channels before forming the large (E,theta,phi) array.
        radial=np.stack([grouped[outer]@tests[outer].T for outer in outstates],axis=-1)
        # One BLAS contraction avoids a large (ion,E,angle) temporary for every m.
        integral=radial@harmonics
        phase=np.exp(1j*(energy[:,None]*times[i]+k[:,None]*(unit@intA[i])[None,:]+intA2[i]/2))
        b+=1j*np.sqrt(2)*dt*(.5 if i in (0,len(times)-1) else 1)*integral*phase[None,:,:]
        if i:
            d=dt/substeps
            for j in range(substeps):
                if ionic_step is not None:chi=ionic_step.step(chi,times[i]-j*d,-d)
                else:
                    A=avec(times[i]-(j+.5)*d)
                    H=ion.matrix(A).conj().T
                    op=csr_matrix(1j*d*H)
                    chi=expm_multiply(op,chi,traceA=op.diagonal().sum())
    return b,abs(b)**2*k[None,:,None]
