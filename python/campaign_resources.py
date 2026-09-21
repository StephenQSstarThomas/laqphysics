"""Allocation estimates from the actual FE-DVR grid, before expensive angular setup."""
import math
import numpy as np
from fedvr import make_grid
from pulses import Pulse
from surface_storage import file_limit
from streaming_surface import energy_grid,directions

def estimate(config):
    c=config;g=make_grid(**c['radial']);n=len(g.r);top=c['lmax'];M=c.get('M',0)
    if 'total_Lmax' in c:
        nc=sum((2*L+1 if M is None else int(abs(M)<=L)) for l in range(top+1) for j in range(top+1)
               for L in range(abs(l-j),min(l+j,c['total_Lmax'])+1)
               if (c.get('angular_representation')=='bipolar' and not c.get('natural_parity',False)) or (l+j+L)%2==0)
    else:
        from angular import channels
        nc=len(channels(top,M))
    T=g.kinetic.toarray();theta=(g.r.real>c['surface']).astype(float)
    idx=np.flatnonzero(np.max(abs(T*(theta[None,:]-theta[:,None])),axis=1)>1e-14)
    if len(idx)==0 or np.any(abs(g.weights[idx].imag)>1e-12):raise ValueError('invalid surface or ECS bridge intersection')
    end=c.get('end_time',max(Pulse(**p['pulse']).start+Pulse(**p['pulse']).duration for p in c['pulses'])+c.get('post_time',60))
    stride=c.get('surface_stride',1);steps=math.ceil(end/c['dt']);steps+=(-steps)%stride;frames=steps//stride+1
    s=c.get('storage',{});limit=file_limit(s.get('max_file_bytes',4_000_000_000));B=int(s.get('ionic_block_frames',128))
    labels=[tuple(x) for x in c.get('ionic_channels',[[1,0,0],[2,1,0]])]
    linear=all(p.get('polarization','z')=='z' for p in c['pulses'])
    unique=set((n,l,abs(m) if linear else m) for n,l,m in labels);mag={m for n,l,m in unique}
    ion_na=sum(sum(abs(m)<=l for m in mag) for l in range(top+1)) if linear else (top+1)**2
    outer_na=(top+1)**2 if M is None else sum(sum(abs(m)<=l for m in {mm for l in range(top+1) for mm in range(-l,l+1) if abs(mm) in mag}) for l in range(top+1))
    nenergy=len(energy_grid(c));nangles=len(directions(c)[0]);wave=n*n*nc*16;amplitude=len(labels)*nenergy*nangles*16
    ends=math.ceil((frames-1)/B)+1;endpoint_frame=n*ion_na*len(unique)*16
    qframe=len(labels)*outer_na*len(idx)*16;endpoint=ends*endpoint_frame
    raw=frames*n*len(idx)*nc*16;mode=s.get('mode','spectrum')
    q=frames*qframe if mode=='projected' else 0
    final_spectrum=int(amplitude*1.5)+len(labels)*nenergy*8+65536
    accumulator=2*amplitude+65536;checkpoint=2*wave+65536
    largest=max(checkpoint,accumulator,final_spectrum,min(endpoint,limit-4096)+4096,(min(q,limit-4096)+4096) if q else 0)
    if checkpoint>limit or accumulator>limit or final_spectrum>limit:raise ValueError('wavefunction/spectrum checkpoint exceeds file cap')
    if endpoint_frame+4096>limit or qframe+4096>limit:raise ValueError('one time frame exceeds file cap')
    persistent=endpoint+q+2*accumulator+checkpoint+final_spectrum
    if mode in ('raw_shards','legacy_raw'):persistent+=raw+frames*qframe
    return {'radial_points':n,'angular_channels':nc,'steps':steps,'dt_actual':end/steps,'duration_au':end,'surface_frames':frames,
            'energy_points':nenergy,'angular_points':nangles,'ionic_final_channels':len(labels),'ionic_unique_states':len(unique),
            'raw_flux_equivalent_bytes':raw,'raw_flux_written':mode in ('raw_shards','legacy_raw'),
            'ionic_endpoints_bytes':endpoint,'optional_projected_history_bytes':q,'one_wavefunction_bytes':wave,
            'wave_checkpoint_upper_bytes':checkpoint,'one_spectrum_checkpoint_upper_bytes':accumulator,'final_spectrum_upper_bytes':final_spectrum,
            'spectral_accumulator_pair_bytes':2*amplitude,
            'persistent_output_estimate_bytes':persistent,'temporary_write_peak_estimate_bytes':persistent+checkpoint+accumulator,
            'largest_output_file_upper_bytes':largest,'max_file_bytes':limit,
            'ionic_replay_RAM_bytes':(B+1)*endpoint_frame,
            'replay_workers':int(s.get('replay_workers',0)),
            'replay_array_RAM_estimate_including_lookahead_bytes':(1+4*int(s.get('replay_workers',0)))*(B+1)*endpoint_frame,
            'FGMRES_Q_Z_bytes':33*wave,
            'scope':'Array sizes plus conservative headers; GPU operators, workspaces, ground cache and logs add overhead. Not a runtime or peak GPU guarantee.'}
