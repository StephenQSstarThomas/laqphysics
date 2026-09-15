#!/usr/bin/env python3
"""Final prepared-ion populations and an independently computed perturbative check."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]/'results/followup'
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');rows=[]
for case in ['ion_2p_co','ion_2p_counter','ion_2p_counter_dt05','ion_2p_counter_weak']:
    if not (root/case/'run.json').exists():continue
    meta=json.loads((root/case/'run.json').read_text())
    if not meta['complete']:continue
    data=np.load(root/case/'populations.npz');history=json.loads((root/case/'history.json').read_text())
    rows.append({'case':case,'final_P3p':meta['P_3p'],'final_P3d':meta['P_3d'],
                 'final_inner_norm':meta['inner_norm'],'recorded_n_le_6_bound_probability':meta['bound_n_le_6_probability'],
                 'final_3p_m_populations':{str(m):float(p) for (n,l,m),p in zip(data['labels'],data['probabilities']) if (n,l)==(3,1)}})
    for ax,key in zip(axes,['P_3p','P_3d']):
        if case not in ['ion_2p_co','ion_2p_counter']:continue
        ax.semilogy([r['time'] for r in history],np.maximum([r[key] for r in history],1e-22),label=case)
        ax.set(xlabel='Time (a.u.)',ylabel=key);ax.legend(fontsize=8)
axes[0].set_title('Allowed counter-helicity final 3p transfer')
axes[1].set_title('Transient field-free 3d projection vanishes after pulse')
fig.savefig(root/'figures/ionic_circulation.png',dpi=180);plt.close(fig)
green=json.loads((root/'ionic_virtual/summary.json').read_text())['radial_convergence'][-1]
config=json.loads((root/'ion_2p_counter/run.json').read_text())['config'];p=config['pulses'][0]['pulse']
T=p['cycles']*2*np.pi/p['omega'];M=green['two_photon_2p_to_3p'];delta=green['final_stark_coefficient']-green['initial_stark_coefficient']
perturbative=[]
for field in [.003,.0015]:
    predictions={'field':field}
    for stark in [False,True]:
        def rhs(t,y):
            f2=field**2*np.sin(np.pi*t/T)**4
            return -1j*np.array([[0,M*f2/4],[M*f2/4,delta*f2 if stark else 0]])@y
        sol=solve_ivp(rhs,[0,T],np.array([1.,0.],complex),method='DOP853',rtol=1e-11,atol=1e-13)
        predictions['with_stark' if stark else 'without_stark']=float(abs(sol.y[1,-1])**2)
    perturbative.append(predictions)
result={'runs':rows,'two_state_second_order_P3p':perturbative,
        'scope':'Prepared He+ TDSE, continuum included. Field-free projections during the pulse contain virtual dressing and are gauge dependent.',
        'perturbation_limit':'Second-order estimates explain suppression but differ from TDSE by about 9%; they are not substituted for the full propagation.'}
byname={r['case']:r for r in rows}
if 'ion_2p_counter_dt05' in byname:
    a=byname['ion_2p_counter_dt05'];b=byname['ion_2p_counter']
    result['time_refinement']={'nominal_dt_coarse':.5,'nominal_dt_fine':.25,'P3p_relative_change':b['final_P3p']/a['final_P3p']-1,
                               'inner_norm_absolute_change':b['final_inner_norm']-a['final_inner_norm']}
if 'ion_2p_counter_weak' in byname:
    a=byname['ion_2p_counter_weak'];b=byname['ion_2p_counter']
    result['weak_field_check']={'relative_to_field_fourth_power_scaling':16*a['final_P3p']/b['final_P3p'],
        'TDSE_relative_difference_from_second_order_with_Stark':a['final_P3p']/perturbative[1]['with_stark']-1}
(root/'ionic_tdse_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
