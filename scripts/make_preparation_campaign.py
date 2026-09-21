#!/usr/bin/env python3
"""Fixed-area circular pi pulses and stronger first-pulse candidates, declared before running."""
import json,sys
from pathlib import Path
from copy import deepcopy
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'python'))
from pulses import Pulse
from campaign_resources import estimate
dest=root/'configs/preparation_20260921';dest.mkdir(exist_ok=True)
base=json.loads((root/'configs/followup/pump_probe_plus.json').read_text())
from angular import radial_dipole,cartesian
dipole=float(abs(radial_dipole(1,0,2,1)*cartesian((1,0),(0,0))[2]))
cases={};rows=[]
for name,F,N,start,control_N in [
    ('short_pump_F012_N24_pi201',.12,24,240.,201),
    ('longer_pump_F008_N48_pi201',.08,48,360.,201),
    ('short_pump_F012_N24_pi402',.12,24,240.,402)]:
    c=deepcopy(base);field=1.5/(dipole*control_N)
    c['pulses']=[{'pulse':{'omega':1.2,'cycles':N,'field':F},'polarization':'z'},
                 {'pulse':{'omega':1.5,'cycles':control_N,'field':field,'start':start},'polarization':'sigma+'}]
    c.pop('end_time',None);c['post_time']=100.;c['ionic_transfer_times']=[start]
    c['spectrum_energy']=[.01,2.4,957]
    c['spectrum']={'energy_segments':[[.01,.18,35],[.18,.75,571],[.75,2.4,166]],'theta_points':12,'phi_points':16}
    c['storage']={'mode':'spectrum','max_file_bytes':4_000_000_000,'ionic_block_frames':128,'ionic_device':'lu'}
    c['design']={'scope':'calibration on the matched previous small model, not converged production','target_old_band':[.18,.42],
                 'unwanted_new_band':[.52,.68],'pi_area_RWA':float(dipole*field*Pulse(**c['pulses'][1]['pulse']).duration/2),
                 'hydrogenic_1s_2p_dipole_au':dipole,'note':'Weakening a fixed-area pi pulse lowers fluence but also narrows the new band; peak height must be measured separately.'}
    cases[name]=c
    for kind in ['pump_only','control_only']:
        d=deepcopy(c);d['pulses']=[d['pulses'][0 if kind=='pump_only' else 1]]
        if kind=='pump_only':d['end_time']=start;d['ionic_transfer_times']=[]
        cases[name+'_'+kind]=d
for name,c in cases.items():
    (dest/(name+'.json')).write_text(json.dumps(c,indent=2)+'\n');rows.append({'case':name,'resources':estimate(c)})
plan={'cases':list(cases),'primary_cases':[n for n in cases if not n.endswith(('_pump_only','_control_only'))],
      'targets':{'peak_height_ratio_03_over_06_min':3.,'yield_ratio_03_over_06_min':5.,'old_band_2p_plus1_fraction_min':.95},
      'scope':'bounded calibration; choose a candidate from actual spectra, then refine radial/angular/time/channel cutoffs',
      'resources':rows}
(dest/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print('Prepared',len(cases),'configurations. Dipole:',dipole)
