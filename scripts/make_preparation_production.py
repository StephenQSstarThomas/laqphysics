#!/usr/bin/env python3
"""Production/refinement inputs for the selected preparation, not completed results."""
import json,sys
from copy import deepcopy
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'python'))
from campaign_resources import estimate
base=root/'configs/preparation_20260921';out=base/'production';out.mkdir(exist_ok=True)
c=json.loads((base/'selected_F008_N48_pi201_refined_n3.json').read_text())
edges=[0,.25,.5,.75,1,1.25,1.5,2,2.5,3,4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32]
c['radial'].update(edges=edges,order=[12 if x<=2 else 6 for x in edges[1:]])
c.update(lmax=4,total_Lmax=4,cutoff_radii=[16,20],surface=24.,dt=.06,post_time=120.,linear_tolerance=1e-10)
c['spectrum'].update(theta_points=24,phi_points=32)
c['storage'].update(mode='projected',ionic_device='cpu',replay_workers=4,accumulator_device='auto')
c['design']['scope']='production candidate: requires the independent comparisons listed in this plan'
cases={'reference':c};pairs=[]
for kind in ['pump_only','control_only']:
    d=deepcopy(c);d['pulses']=[d['pulses'][0 if kind=='pump_only' else 1]]
    if kind=='pump_only':d['end_time']=360.;d['ionic_transfer_times']=[]
    cases[kind]=d
def variant(name,axis,change,reference='reference'):
    d=deepcopy(cases[reference]);change(d);cases[name]=d;pairs.append({'reference':reference,'refinement':name,'axis':axis})
variant('l6','individual angular cutoff 4 -> 6',lambda d:d.update(lmax=6))
variant('L5','total angular cutoff 4 -> 5',lambda d:d.update(total_Lmax=5))
variant('inner14','inner radial order 12 -> 14',lambda d:d['radial'].update(order=[14 if x<=2 else 6 for x in edges[1:]]))
variant('outer8','outer radial order 6 -> 8',lambda d:d['radial'].update(order=[12 if x<=2 else 8 for x in edges[1:]]))
variant('dt003','time step 0.06 -> 0.03',lambda d:d.update(dt=.03))
variant('surface28','surface radius 24 -> 28',lambda d:d.update(surface=28.))
variant('cutoff24','Coulomb cutoff [16,20] -> [20,24]',lambda d:d.update(cutoff_radii=[20,24]),reference='surface28')
variant('tail32','absorber order 24 -> 32',lambda d:d['radial'].update(tail=32))
variant('angle06','ECS angle 0.5 -> 0.6',lambda d:d['radial'].update(ecs_angle=.6))
variant('alpha10','Laguerre scale 0.8 -> 1',lambda d:d['radial'].update(alpha=1.))
variant('mixed_gauge','velocity vs independent mixed gauge',lambda d:d.update(gauge={'type':'mixed','inner':4.,'outer':8.}))
variant('post240','post-pulse propagation 120 -> 240 a.u.',lambda d:d.update(post_time=240.))
variant('ionic_n4','recorded final ion n <= 3 -> 4',lambda d:d.update(ionic_channels=[[n,l,m] for n in range(1,5) for l in range(n) for m in range(-l,l+1)]))
pairs[-1]['allow_ionic_extension']=True
resources=[]
for name,d in cases.items():
    (out/(name+'.json')).write_text(json.dumps(d,indent=2)+'\n');resources.append({'case':name,**estimate(d)})
plan={'cases':list(cases),'pairs':pairs,'scope':'selected two-pulse preparation, production candidates; none certified by configuration alone',
      'acceptance':{'peak_shift_au':.001,'normalized_shape_L1':.02,'relative_yield':.02,
                    'gate_total':True,'gate_channels':[[1,0,0],[2,1,1]],'energy_windows':[[.18,.42],[.52,.68]]},
      'extraction_pairs':[
          {'reference':'reference','refinement':'reference','refinement_file':'spectrum_angles.npz','axis':'angular quadrature 24x32 -> 32x48'},
          {'reference':'reference','refinement':'reference','refinement_file':'spectrum_energy.npz','axis':'energy spacing 0.0005 -> 0.00025 in target bands'},
          {'reference':'reference','refinement':'reference','refinement_file':'spectrum_stride2.npz','axis':'surface quadrature every step vs every 2 steps'}],
      'completion_requires':'Pass all listed comparisons, refine spectral/time quadrature from the retained projected history, test final ionic-channel completeness and confirm the final combined setting.'}
(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
(out/'resource_estimates.json').write_text(json.dumps(resources,indent=2)+'\n')
print('Prepared',len(cases),'production candidates. Largest estimated file:',max(r['largest_output_file_upper_bytes'] for r in resources),'bytes')
