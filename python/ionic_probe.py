"""Prepared-ion probe: forward He+ evolution on the two-electron model's own ion.

The ion uses exactly the tSURFF ionic Hamiltonian of a 3D input: the same ECS
radial grid, Coulomb cutoff, velocity-gauge vector potential and CF4 sparse-LU
propagator as the online adjoint replay. Two uses:

1. Design a two-photon probe of the prepared ion (which final bound states are
   reached, how much is ionized) before spending two-electron propagation time.
2. Map coherent channel amplitudes through a probe that acts only after the
   emitted electrons are beyond the surface (tSURFF channel identity):
       b_c(k,T3) = sum_c' <phi_c|U_ion(T3,T2)|phi_c'> b_c'(k,T2).
   The Volkov factor cancels exactly, so relative channel phases are retained.
   It neglects electrons emitted or crossing the surface after T2 and bound ionic
   components outside the recorded input channels; the full three-pulse TDSE is
   the reference and the residual is reported, never assumed zero.
"""
from types import SimpleNamespace
import numpy as np
from fedvr import make_grid
from surface3d import Ionic
from tdse1d import cutoff
from pulses import Pulse

def ion_model(config,lmax=None):
    """He+ with the given input's radial grid, cutoff and single-electron l range."""
    grid=make_grid(**config['radial']);top=config['lmax'] if lmax is None else int(lmax)
    cut=cutoff(grid.r.real,*config['cutoff_radii']) if config.get('cutoff_radii') else np.ones(len(grid.r))
    h=SimpleNamespace(grid=grid,n=len(grid.r),r=grid.r,cut=cut,ch=[((l,m),(0,0)) for l in range(top+1) for m in range(-l,l+1)])
    return Ionic(h)

def bound_labels(ion,nmax):
    return [(n,l,m) for n in range(1,nmax+1) for l in range(min(n,max(l for l,m in ion.states)+1)) for m in range(-l,l+1)]

def model_levels(ion,labels):
    """Field-free eigenvalues of this discretized ion nearest to -2/n^2 (radial problem only)."""
    result={}
    for l in sorted({l for n,l,m in labels}):
        i=ion.index[(l,0)] if (l,0) in ion.index else next(i for i,(ll,m) in enumerate(ion.states) if ll==l)
        block=ion.h0[i*ion.n:(i+1)*ion.n,i*ion.n:(i+1)*ion.n].toarray();values=np.linalg.eigvals(block)
        for n in sorted({n for n,ll,m in labels if ll==l}):
            target=-2/n**2;result[(n,l)]=complex(values[np.argmin(abs(values-target))])
    return result

def pulse_field(pulses):
    """Velocity-gauge A(t) for [{'pulse':{...},'polarization':...}] entries, as in run3d."""
    from run3d import vector_function
    return vector_function({'pulses':pulses})[0]

def transfer_matrix(config,pulses,t0,t1,labels_in,labels_out,dt=None,lmax=None,device='cpu'):
    """U[c,c'] = phi_c^T U_ion(t1,t0) phi_c' with the analytic ECS c-dual projection.

    Computed as tSURFF does: the output duals are propagated backward with the
    adjoint ionic CF4 stepper (device 'lu', 'cpu' or 'cuda:0'), one column per
    output channel, then contracted with the input kets. Both endpoints must have
    A=0 so that field-free bound states are the physical ionic states in the
    velocity gauge used by the two-electron solver.
    """
    from streaming_surface import AdjointStepper
    ion=ion_model(config,lmax);avec=pulse_field(pulses)
    for t in (t0,t1):
        if np.max(abs(avec(t)))>1e-10:raise ValueError('probe transfer endpoints require zero vector potential')
    labels_in=[tuple(map(int,x)) for x in labels_in];labels_out=[tuple(map(int,x)) for x in labels_out]
    missing=[x for x in labels_in+labels_out if (x[1],x[2]) not in ion.index]
    if missing:raise ValueError(f'labels outside the ionic angular basis: {missing}')
    steps=max(1,int(np.ceil((t1-t0)/(dt or config.get('dt',.06)))));h=(t1-t0)/steps
    stepper=AdjointStepper(ion,avec,'cf4',device);stepper.reset(ion.dual_final_states(labels_out))
    for i in range(steps):stepper.step(t1-i*h,-h)
    U=stepper.chi.conj().T@ion.final_states(labels_in)
    return U,{'t0':float(t0),'t1':float(t1),'steps':steps,'dt_actual':h,'device':device,
              'maximum_linear_residual':None if device=='lu' else stepper.maximum_residual,
              'recorded_output_probability':(abs(U)**2).sum(axis=0).tolist()}

def probe_window(config):
    """Start/end of the probe pulses (entries whose role is 'probe')."""
    probes=[Pulse(**p['pulse']) for p in config['pulses'] if p.get('role')=='probe']
    if not probes:raise ValueError('configuration has no pulse with role=probe')
    return min(p.start for p in probes),max(p.start+p.duration for p in probes)

def factorized_amplitudes(amplitudes,labels_in,U,labels_out):
    """Apply the ionic channel map to b[c',E,Omega]; returns b[c,E,Omega]."""
    b=np.asarray(amplitudes);labels_in=[tuple(map(int,x)) for x in labels_in]
    if b.shape[0]!=len(labels_in) or U.shape!=(len(labels_out),len(labels_in)):raise ValueError('channel map shape mismatch')
    return np.tensordot(U,b,axes=([1],[0]))
