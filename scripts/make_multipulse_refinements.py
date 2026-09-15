#!/usr/bin/env python3
"""Bounded next-stage circular/multipulse refinements, explicitly awaiting TDSE runs."""
from pathlib import Path
from copy import deepcopy
import json
root=Path(__file__).resolve().parents[1];source=root/'configs/followup';dest=root/'configs/followup_multipulse';dest.mkdir(exist_ok=True)
names=['pump_only','pump_probe_plus','pump_probe_minus','pump_probe_linear','probe_only_plus','pump_probe_half_frequency','pump_probe_plus_delay192']
cases={}
for name in names:
    c=json.loads((source/(name+'.json')).read_text());c['lmax']=4 if name=='pump_probe_half_frequency' else 3
    c['radial']['order']=[8 if b<=2 else 6 for b in c['radial']['edges'][1:]]
    c.update(dt=.06,max_surface_gib=128,coulomb_contraction='blocks',linear_tolerance=1e-11)
    if name=='pump_probe_half_frequency':c['spectrum_energy']=[.1,2.4,767];c['pump_reference_time']=144.
    cases[name]=c
pairs=[]
for name,axis in [('pump_probe_plus_l4','individual lmax 3 -> 4'),('pump_probe_plus_radial8','inner/outer radial order 8/6 -> 10/8'),('pump_probe_plus_dt003','dt 0.06 -> 0.03')]:
    c=deepcopy(cases['pump_probe_plus'])
    if name.endswith('l4'):c['lmax']=4
    if name.endswith('radial8'):c['radial']['order']=[10 if b<=2 else 8 for b in c['radial']['edges'][1:]]
    if name.endswith('dt003'):c['dt']=.03
    cases[name]=c;pairs.append({'reference':'pump_probe_plus','refinement':name,'axis':axis})
for name,c in cases.items():(dest/(name+'.json')).write_text(json.dumps(c,indent=2)+'\n')
plan={'scope':'next-stage multipulse refinement; not certified production spectra',
      'cases':list(cases),'pairs':pairs,'acceptance':{'peak_shift_au':.001,'normalized_shape_L1':.02,'relative_yield':.02,
      'gate_channels':[[1,0,0],[2,1,1]],'energy_windows':[[.18,.42],[.52,.68]],
      'extra_channels':'all n<=2 recorded; old and new energy bands are tested separately'},
      'completion_requires':'These are the first refinement pairs. Continue failed axes, then check cutoff, absorber and the changed frequency regime independently. Prepared-ion and open-system tasks have separate scope.'}
(dest/'production_plan.json').write_text(json.dumps(plan,indent=2)+'\n');print('Prepared',len(cases),'candidate configurations; none marked complete.')
