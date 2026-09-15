#!/usr/bin/env python3
"""Create a bounded, one-axis-at-a-time production convergence campaign.

Writing a configuration does not mark its calculation complete. The reference
ground-state grid was checked locally; spectra require actual TDSE execution.
"""
from pathlib import Path
from copy import deepcopy
import json
root=Path(__file__).resolve().parents[1];dest=root/'configs/followup'
base=json.loads((dest/'res_l8_L2_p6.json').read_text())
edges=[0,.25,.5,.75,1,1.25,1.5,2,2.5,3,4,5,6,8,10,12,16,20,24,28]
base['radial']['edges']=edges;base['radial']['order']=[12 if b<=2 else 6 for b in edges[1:]]
base.update(lmax=10,total_Lmax=3,dt=.06,max_surface_gib=64,coulomb_contraction='blocks',
            ionic_channels=[[1,0,0],[2,0,0],[2,1,-1],[2,1,0],[2,1,1]])
cases={'prod_reference':base};pairs=[]
for name,axis in [('prod_radial14','near-nucleus radial order 12 -> 14'),
                  ('prod_outer8','outer radial order 6 -> 8, inner order remains 12'),
                  ('prod_l12','individual electron lmax 10 -> 12'),
                  ('prod_L4','total Lmax 3 -> 4'),('prod_dt003','time step 0.06 -> 0.03'),
                  ('prod_radius','potential cutoff [12,16] -> [16,20], surface 20 -> 24'),
                  ('prod_absorber','irECS tail 24 -> 32, angle 0.5 -> 0.6')]:
    c=deepcopy(base)
    if name=='prod_radial14':c['radial']['order']=[14 if b<=2 else 6 for b in edges[1:]]
    if name=='prod_outer8':c['radial']['order']=[12 if b<=2 else 8 for b in edges[1:]]
    if name=='prod_l12':c['lmax']=12
    if name=='prod_L4':c['total_Lmax']=4
    if name=='prod_dt003':c['dt']=.03
    if name=='prod_radius':
        c['cutoff_radii']=[16,20];c['surface']=24
        c['radial']['edges']=edges+[32];c['radial']['order'].append(6)
    if name=='prod_absorber':c['radial'].update(tail=32,ecs_angle=.6)
    cases[name]=c;pairs.append({'reference':'prod_reference','refinement':name,'axis':axis})
for name,c in cases.items():(dest/(name+'.json')).write_text(json.dumps(c,indent=2)+'\n')
plan={'scope':'180-cycle resonant helium spectrum, independent refinement axes',
      'cases':list(cases),'pairs':pairs,'acceptance':{'peak_shift_au':.001,'normalized_shape_L1':.02,'relative_yield':.02,
       'gate_channels':[[1,0,0],[2,1,0]],'extra_channels':'all n<=2 stored and reported; original dominant channels define these gates'},
      'completion_requires':'All pairs have actual complete TDSE spectra and pass. Then repeat relevant axes for every changed physical pulse regime.'}
(dest/'production_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print('\n'.join(cases))
