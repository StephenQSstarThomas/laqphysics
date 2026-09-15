#!/usr/bin/env python3
"""Predeclare multipulse production candidates and independent convergence axes.

Only configurations are generated here. Their existence is never evidence of
completed propagation or convergence. Retain both bipolar parities and all M.
"""
import json
from copy import deepcopy
from pathlib import Path
root=Path(__file__).resolve().parents[1];source=root/'configs/followup';dest=root/'configs/convergence_complete/multipulse';dest.mkdir(exist_ok=True)
names=['pump_only','pump_probe_plus','pump_probe_minus','pump_probe_linear','probe_only_plus','pump_probe_half_frequency','pump_probe_plus_delay192']
edges=[0,.25,.5,.75,1,1.25,1.5,2,2.5,3,4,5,6,8,10,12,14,16,18,20,22,24,26,28]
cases={};pairs=[]
for name in names:
    c=json.loads((source/(name+'.json')).read_text())
    c.update(lmax=4,total_Lmax=4 if name=='pump_probe_half_frequency' else 3,angular_representation='bipolar',
             natural_parity=False,M=None,dt=.06,linear_tolerance=1e-10,ground_tolerance=1e-10,
             max_surface_gib=160,coulomb_contraction='blocks',surface_stride=1)
    c['radial'].update(edges=edges,order=[12 if x<=2 else 6 for x in edges[1:]])
    c['end_time']=144. if name=='pump_only' else (600. if name.endswith('delay192') else 552.)
    c['spectrum_energy']=[.1,2.4,921] if name=='pump_probe_half_frequency' else [.1,1.,451]
    cases[name]=c

def variant(base,suffix,axis,modify,windows=None,channels=None):
    c=deepcopy(cases[base]);modify(c);name=base+'_'+suffix;cases[name]=c
    pair={'reference':base,'refinement':name,'axis':axis}
    if windows:pair['energy_windows']=windows
    if channels:pair['acceptance']={'peak_shift_au':.001,'normalized_shape_L1':.02,'relative_yield':.02,'gate_channels':channels}
    pairs.append(pair)

variant('pump_probe_plus','inner14','near-nuclear radial order 12 -> 14',lambda c:c['radial'].update(order=[14 if x<=2 else 6 for x in edges[1:]]))
variant('pump_probe_plus','outer8','outer radial order 6 -> 8',lambda c:c['radial'].update(order=[12 if x<=2 else 8 for x in edges[1:]]))
variant('pump_probe_plus','l6','individual l 4 -> 6',lambda c:c.update(lmax=6))
variant('pump_probe_plus','L4','total L 3 -> 4',lambda c:c.update(total_Lmax=4))
variant('pump_probe_plus','dt003','time step 0.06 -> 0.03',lambda c:c.update(dt=.03))
variant('pump_probe_plus','tail32','absorber order 24 -> 32',lambda c:c['radial'].update(tail=32))
variant('pump_probe_plus','mixed','independent mixed/velocity gauge',lambda c:c.update(gauge={'type':'mixed','inner':4.,'outer':8.}))
def extent(c):c['radial']['edges']+=[30,32];c['radial']['order']+=[6,6]
variant('pump_probe_plus','extent32','real-domain extent 28 -> 32',extent)
variant('pump_probe_plus_extent32','surface24','surface 20 -> 24 only',lambda c:c.update(surface=24))
variant('pump_probe_plus_extent32_surface24','cutoff20','Coulomb cutoff [12,16] -> [16,20] only',lambda c:c.update(cutoff_radii=[16,20]))
for suffix,axis,modify in [
 ('l6','half-frequency individual l 4 -> 6',lambda c:c.update(lmax=6)),
 ('L5','half-frequency total L 4 -> 5',lambda c:c.update(total_Lmax=5)),
 ('outer8','half-frequency outer order 6 -> 8',lambda c:c['radial'].update(order=[12 if x<=2 else 8 for x in edges[1:]])),
 ('dt003','half-frequency time step 0.06 -> 0.03',lambda c:c.update(dt=.03))]:
    variant('pump_probe_half_frequency',suffix,axis,modify,[[.18,.42],[.52,.68],[1.22,1.42],[1.95,2.2]],[[1,0,0]])
plan={'scope':'all-M bipolar multipulse convergence, full correlated two-electron TDSE; candidates awaiting actual gates',
      'cases':list(cases),'pairs':pairs,'acceptance':{'peak_shift_au':.001,'normalized_shape_L1':.02,'relative_yield':.02,
      'gate_channels':[[1,0,0],[2,1,1]],'energy_windows':[[.18,.42],[.52,.68]]},
      'completion_requires':'All listed propagation comparisons must pass. Also check extraction quadrature, opposite helicity and the delayed old-electron ionic map. No peak alignment. A failed axis requires further refinement.'}
for name,c in cases.items():(dest/(name+'.json')).write_text(json.dumps(c,indent=2)+'\n')
(dest/'production_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print('Prepared',len(cases),'candidates; none declared converged.')
