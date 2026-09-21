#!/usr/bin/env python3
"""Test the separated-pulse interpretation against the actual one-pulse controls."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.integrate import simpson
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from surface_storage import atomic_json

def load(folder):
    obs=json.loads((folder/'observables.json').read_text())
    with np.load(folder/obs['artifacts']['spectrum']) as f:d={k:f[k] for k in ['energy','angle_integrated','labels']}
    return d,obs

def main():
    p=argparse.ArgumentParser();p.add_argument('--out-root',required=True);a=p.parse_args();root=Path(a.out_root)
    names=['longer_pump_F008_N48_pi201__ccc6f27abe','longer_pump_F008_N48_pi201_pump_only__a8372ebfe4','longer_pump_F008_N48_pi201_control_only__f2dbbeaa05']
    (both,ob),(pump,op),(control,oc)=[load(root/name) for name in names]
    e=both['energy'];assert np.array_equal(e,pump['energy']) and np.array_equal(e,control['energy'])
    first=pump['angle_integrated'][list(map(tuple,pump['labels'])).index((1,0,0))]
    after=both['angle_integrated'][list(map(tuple,both['labels'])).index((2,1,1))]
    population=json.loads((root/names[1]/'history.json').read_text())[-1]['ground_population']
    transfer=ob['prepared_ion_P_2p_plus1'];pred_old=transfer*first
    pred_new=population*control['angle_integrated'].sum(axis=0);actual=both['angle_integrated'].sum(axis=0)
    def metrics(a,b,lo,hi):
        mask=(e>=lo)&(e<=hi);x=e[mask];y=a[mask];z=b[mask];ya=float(simpson(y,x=x));yb=float(simpson(z,x=x))
        return {'energy_window':[lo,hi],'predicted_yield':ya,'measured_yield':yb,'relative_yield_difference':yb/ya-1,
                'absolute_density_relative_L1':float(simpson(abs(y-z),x=x)/ya),
                'normalized_shape_L1':float(simpson(abs(y/ya-z/yb),x=x))}
    result={'cases':names,'prepared_ion_transfer':transfer,'neutral_ground_survival_after_pulse1':population,
            'old_band_2p_plus1':metrics(pred_old,after,.18,.42),'new_band_total':metrics(pred_new,actual,.52,.68),
            'scope':'Small-model separated-pulse diagnostic using independently measured transfer/survival, with no fitted scales or energy shifts. Not a convergence gate.'}
    atomic_json(root/'control_factorization_diagnostic.json',result)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,predicted,measured,window,title in [(axes[0],pred_old,after,(.24,.36),'Old band: measured ion-transfer prediction'),(axes[1],pred_new,actual,(.55,.65),'New band: surviving-neutral prediction')]:
        ax.plot(e,predicted,'k--',lw=1.8,label='Independent single-pulse prediction')
        ax.plot(e,measured,color='C3',lw=1.1,label='Actual two-pulse spectrum')
        ax.set(xlim=window,ylim=(0,None),xlabel='Electron energy (a.u.)',ylabel='dP/dE (a.u.)',title=title);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Control-group check: absolute scales fixed by independent dynamics',fontsize=11)
    figures=root/'figures';figures.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(figures/('separated_pulse_controls.'+ext),dpi=240)
    plt.close(fig);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
