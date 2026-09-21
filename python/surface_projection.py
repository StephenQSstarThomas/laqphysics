"""Energy-independent ionic projection of a tSURFF surface history.

The costly adjoint propagation is performed once. The retained history is enough
to repeat energy/angle quadrature and the final-time integral without rerunning
the two-electron TDSE. No ionic or angular channels are truncated here.
"""
import hashlib
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import spherical_jn,sph_harm_y
from surface3d import Ionic

def project(h,flux,times,avec,labels,dt,substeps=1,angular_transform=None,
            propagator='expm',diagnostics=None,output=None,fold_m=True,ion_factory=None,ionic_field=None,ionic_device=None,max_file_bytes=None):
    from scipy.sparse import csr_matrix
    from scipy.sparse import issparse
    from scipy.sparse.linalg import expm_multiply
    from implicit import SparseCF4
    labels=[tuple(x) for x in labels];av=np.array([avec(t) for t in times])
    if np.max(abs(av[-1]))>1e-10:raise ValueError('projection requires zero final vector potential')
    linear=bool(np.max(abs(av[:,:2]))<1e-14)
    folded=bool(linear and fold_m)
    represented=[(n,l,abs(m) if folded else m) for n,l,m in labels]
    unique=list(dict.fromkeys(represented));columns=[unique.index(x) for x in represented]
    magnetic={m for n,l,m in unique} if linear else None
    ion=(ion_factory or Ionic)(h,magnetic);chi=ion.dual_final_states(unique)
    field=ionic_field or avec
    accelerated=None;state=None;maximum_linear_residual=0.
    if ionic_device is not None:
        if propagator!='cf4':raise ValueError('accelerated ionic propagation requires cf4')
        from ionic_adjoint import IonicAdjoint
        from implicit import cf4_step
        accelerated=IonicAdjoint(ion,len(unique),ionic_device);state=accelerated.tensor(chi).reshape(-1)
    if propagator=='cf4':
        solver=SparseCF4(lambda t:ion.matrix(field(t)).conj().T,key_at=lambda t:tuple(field(t)))
    elif propagator=='expm':solver=None
    else:raise ValueError('unknown ionic propagator')
    active=[c for c,(inner,outer) in enumerate(h.ch)
            if (inner[0],abs(inner[1]) if folded else inner[1]) in ion.index]
    outerstates=sorted({h.ch[c][1] for c in active});outerindex={x:i for i,x in enumerate(outerstates)}
    shape=(len(times),len(labels),len(outerstates),len(h.surface_indices))
    if output is not None and max_file_bytes is not None:
        from surface_storage import ShardedArray
        history=ShardedArray(output,shape,mode='w+',max_file_bytes=max_file_bytes)
    elif output is not None:
        if np.prod(shape)*16+4096>5_000_000_000:raise ValueError('projected history needs bounded shards; supply max_file_bytes')
        history=np.lib.format.open_memmap(output,mode='w+',dtype=complex,shape=shape)
    else:history=np.empty(shape,complex)
    # In an M=0 coupled basis, each column has fixed (l1,l2). With z polarization,
    # a target m picks exactly one product harmonic per coupled column. Contract
    # its CG coefficient with chi before expanding the surface tensor.
    sparse_transform=angular_transform is not None and issparse(angular_transform)
    fast=bool(angular_transform is not None and linear and not sparse_transform)
    if fast:
        U=angular_transform;lookup={pair:i for i,pair in enumerate(h.ch)}
        pairs=[]
        for c in range(U.shape[1]):
            support=np.flatnonzero(abs(U[:,c])>1e-14)
            pair={(h.ch[i][0][0],h.ch[i][1][0]) for i in support}
            if len(pair)!=1:raise ValueError('coupled column has no unique radial angular pair')
            pairs.append(next(iter(pair)))
        coeff=np.zeros((len(labels),U.shape[1]));inner=np.zeros_like(coeff,dtype=int);groups=[]
        for j,(n,l,m) in enumerate(labels):
            group=np.full(U.shape[1],-1,int)
            for c,(l1,l2) in enumerate(pairs):
                pair=((l1,m),(l2,-m))
                if pair in lookup:
                    coeff[j,c]=U[lookup[pair],c]
                    inner[j,c]=ion.index[(l1,abs(m) if folded else m)]
                    group[c]=outerindex[(l2,-m)]
            groups.append(group)
    else:
        coefficients=None;coupled_columns=None
        if sparse_transform:
            coo=angular_transform.tocoo()
            valid=np.array([(h.ch[row][0][0],abs(h.ch[row][0][1]) if folded else h.ch[row][0][1]) in ion.index for row in coo.row])
            active=coo.row[valid];coupled_columns=coo.col[valid];coefficients=coo.data[valid]
        inner=np.array([ion.index[(h.ch[c][0][0],abs(h.ch[c][0][1]) if folded else h.ch[c][0][1])] for c in active])
        out=np.array([outerindex[h.ch[c][1]] for c in active]);order=np.argsort(out,kind='stable')
        starts=np.r_[0,np.flatnonzero(np.diff(out[order]))+1]
        mask=np.array([[h.ch[c][0][1]==label[2] for c in active] for label in labels]) if folded else None
    requests={int(np.argmin(abs(times-t))):float(t) for t in (diagnostics or {}).get('requested_times',[])}
    initial=ion.final_states([(1,0,0)])[:,0] if requests else None
    if diagnostics is not None:diagnostics['transfers_from_1s']=[]
    digest=hashlib.sha256()
    for i in range(len(times)-1,-1,-1):
        if i in requests:
            amplitude=(chi.conj().T@initial)[columns]
            if folded:amplitude=np.where(np.array(labels)[:,2]==0,amplitude,0.)
            diagnostics['transfers_from_1s'].append({'requested_time':requests[i],'actual_time':float(times[i]),'labels':labels,
                'amplitude_real':amplitude.real.tolist(),'amplitude_imag':amplitude.imag.tolist(),'probabilities':(abs(amplitude)**2).tolist(),
                'scope':'recorded final bound channels only'})
        frame=np.asarray(flux[i]);digest.update(np.ascontiguousarray(frame).tobytes())
        v=chi.reshape(ion.na,h.n,len(unique));result=np.zeros(shape[1:],complex)
        if fast:
            weights=np.array([v[inner[j],:,columns[j]].T.conj()*coeff[j][None,:] for j in range(len(labels))])
            projected=np.einsum('jrc,rsc->jsc',weights,frame)
            for j,group in enumerate(groups):
                for outer in np.unique(group[group>=0]):result[j,outer]=projected[j][:,group==outer].sum(axis=1)
        else:
            current=frame[:,:,coupled_columns] if sparse_transform else (frame[:,:,active] if angular_transform is None else np.tensordot(frame,angular_transform[active].T,axes=([-1],[0])))
            weights=v[inner][:,:,columns].conj().transpose(1,2,0)
            if mask is not None:weights*=mask[None,:,:]
            projected=np.einsum('rja,rsa->jsa',weights,current)
            if coefficients is not None:projected*=coefficients[None,None,:]
            reduced=np.add.reduceat(projected[:,:,order],starts,axis=2).transpose(0,2,1)
            result[:,out[order][starts],:]=reduced
        history[i]=result
        if i:
            d=dt/substeps
            for sub in range(substeps):
                t=times[i]-sub*d
                if accelerated is not None:
                    state,info=cf4_step(accelerated,state,lambda s:accelerated.coefficients(field(s)),t,-d,0.,tol=1e-12)
                    maximum_linear_residual=max(maximum_linear_residual,info['linear_residual'])
                    chi=accelerated.host(state)
                elif solver is not None:chi=solver.step(chi,t,-d)
                else:
                    operator=csr_matrix(1j*d*ion.matrix(field(t-d/2)).conj().T)
                    chi=expm_multiply(operator,chi,traceA=operator.diagonal().sum())
    if hasattr(history,'flush'):history.flush()
    return history,outerstates,{'folded_m_sectors':folded,'coupled_direct_contraction':fast,
        'ionic_device':ionic_device or 'scipy_sparse_lu','ionic_maximum_true_linear_residual':maximum_linear_residual,
        'raw_frames_reverse_sha256':digest.hexdigest(),'frame_hash_order':'reverse time, each frame in C byte order'}

def integrate(projected,outerstates,r,weights,times,avec,energy,directions,dt,stop_index=None):
    energy=np.asarray(energy);k=np.sqrt(2*energy);theta,phi=directions
    unit=np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]).T
    av=np.array([avec(t) for t in times]);intA=cumulative_trapezoid(av,times,axis=0,initial=0)
    intA2=cumulative_trapezoid(np.sum(av*av,axis=1),times,initial=0)
    tests=np.array([np.sqrt(2/np.pi)*(-1j)**l*np.sqrt(weights)[None,:]*r[None,:]*spherical_jn(l,k[:,None]*r[None,:]) for l,m in outerstates])
    harmonics=np.array([sph_harm_y(l,m,theta,phi) for l,m in outerstates])
    end=len(times)-1 if stop_index is None else int(stop_index)
    if not 1<=end<len(times):raise ValueError('invalid integration endpoint')
    selected=times[:end+1];quadrature=np.empty(len(selected))
    quadrature[0]=(selected[1]-selected[0])/2;quadrature[-1]=(selected[-1]-selected[-2])/2
    quadrature[1:-1]=(selected[2:]-selected[:-2])/2
    b=np.zeros((projected.shape[1],len(energy),len(theta)),complex)
    for i in range(end,-1,-1):
        radial=np.einsum('jls,les->jel',projected[i],tests)
        integral=radial@harmonics
        phase=np.exp(1j*(energy[:,None]*times[i]+k[:,None]*(unit@intA[i])[None,:]+intA2[i]/2))
        b+=1j*np.sqrt(2)*quadrature[i]*integral*phase[None,:,:]
    return b,abs(b)**2*k[None,:,None]
