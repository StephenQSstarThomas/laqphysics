#!/usr/bin/env python3
"""Compare the optimized full-history extraction to previously recorded spectra."""
from pathlib import Path
import sys,json,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from run3d import setup,vector_function
from surface3d import extract
root=Path(__file__).resolve().parents[1]/'results/followup';rows=[]
for name in ['pump_probe_plus','res_l6_L2_p6']:
    out=root/name;meta=json.loads((out/'run.json').read_text());config=meta['config'];data=np.load(out/'spectrum.npz')
    _,h=setup(config);h.prepare_surface(config['surface']);reference=h.uncoupled if hasattr(h,'uncoupled') else h
    reference.prepare_surface(config['surface']);transform=getattr(h,'transform',None)
    flux=np.load(out/'flux.npy',mmap_mode='r');dt=meta['dt']*config.get('surface_stride',1);times=np.arange(len(flux))*dt
    select=np.arange(0,len(data['energy']),max(1,(len(data['energy'])-1)//10));energy=data['energy'][select]
    avec,_=vector_function(config);start=time.perf_counter()
    b,p=extract(reference,flux,times,avec,[tuple(x) for x in data['labels']],energy,(data['theta'],data['phi']),dt,
                angular_transform=transform,propagator=config.get('ionic_propagator','expm'))
    old=data['amplitudes'][:,select];oldp=data['pes'][:,select]
    row={'case':name,'number_of_times':len(times),'energy_indices':select.tolist(),'seconds':time.perf_counter()-start,
         'amplitude_max_relative_difference':float(np.max(abs(b-old))/np.max(abs(old))),
         'intensity_max_relative_difference':float(np.max(abs(p-oldp))/np.max(oldp))}
    assert row['amplitude_max_relative_difference']<1e-10
    rows.append(row);print(json.dumps(row),flush=True)
(root/'surface_contraction_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
