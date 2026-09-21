#!/usr/bin/env python3
"""Compare pulse designs on their actual absolute-energy and probability scales."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'python'))
from surface_storage import atomic_json
from scipy.integrate import simpson

def load(folder):
    source=folder/'spectrum.npz'
    if not source.exists():source=folder/json.loads((folder/'observables.json').read_text())['artifacts']['spectrum']
    with np.load(source) as d:data={k:d[k] for k in ['energy','angle_integrated','labels','source_signature'] if k in d}
    data['file_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    return data

def main():
    p=argparse.ArgumentParser();p.add_argument('--out-root',required=True);a=p.parse_args();base=Path(a.out_root)
    designs=[(repo/'results/followup/pump_probe_plus','Previous: F1=.02, N1=12; F2=.0534, N2=76','0.55',':'),
        (base/'longer_pump_F008_N48_pi201__ccc6f27abe','Selected: F1=.08, N1=48; pi pulse N2=201 (small model)','C0','--')]
    fine=base/'selected_F008_N48_pi201_refined_n3_accelerated__b9f521f837'
    if (fine/'observables.json').exists():designs.append((fine,'Selected: finer radial/angle basis, final ion n<=3','C3','-'))
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained');records=[]
    for folder,label,color,style in designs:
        d=load(folder);meta=json.loads((folder/'run.json').read_text());e=d['energy'];y=d['angle_integrated'].sum(axis=0)
        assert meta['complete']
        if 'source_signature' in d:assert str(d['source_signature'])==meta['signature']
        for ax in axes:ax.plot(e,y,color=color,ls=style,lw=1.5,label=label)
        measures=[]
        for lo,hi in [(.18,.42),(.52,.68)]:
            mask=(e>=lo)&(e<=hi);measures.append({'window':[lo,hi],'peak':float(y[mask].max()),'yield':float(simpson(y[mask],x=e[mask]))})
        records.append({'directory':str(folder),'label':label,'signature':meta['signature'],'pulses':meta['config']['pulses'],
                        'spectrum_file_sha256':d['file_sha256'],'embedded_signature_present':'source_signature' in d,
                        'ground_energy':meta.get('ground_energy'),'bands':measures,
                        'peak_ratio':measures[0]['peak']/measures[1]['peak'],'yield_ratio':measures[0]['yield']/measures[1]['yield']})
    for ax,window,title in zip(axes,[(.24,.36),(.55,.65)],['Original-electron band near 0.3 a.u.','Later-ionization band near 0.6 a.u.']):
        ax.set(xlim=window,ylim=(0,None),xlabel='Electron energy (a.u.)',ylabel='Total recorded SI: dP/dE (a.u.)',title=title);ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',fontsize=8);fig.suptitle('Pulse preparation: absolute spectra, no fitted energy shifts or peak normalization',fontsize=11)
    out=base/'figures';out.mkdir(exist_ok=True)
    for extension in ['png','pdf']:fig.savefig(out/('preparation_before_after.'+extension),dpi=250)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained')
    for name,label in [('short_pump_F012_N24_pi201__de8c245203','N2=201, F2=0.02004'),('short_pump_F012_N24_pi402__43b121bb2b','N2=402, F2=0.01002')]:
        d=load(base/name);ax.plot(d['energy'],d['angle_integrated'].sum(axis=0),label=label)
    ax.set(xlim=(.585,.615),ylim=(0,None),xlabel='Electron energy (a.u.)',ylabel='Total recorded SI: dP/dE (a.u.)',
           title='Same pi area: a longer weaker control narrows the peak\nIts integrated yield decreases; peak height changes little')
    ax.legend();ax.grid(alpha=.2)
    for extension in ['png','pdf']:fig.savefig(out/('fixed_pi_area_width_vs_height.'+extension),dpi=250)
    plt.close(fig)
    atomic_json(base/'pulse_design_comparison.json',{'cases':records,'scope':'Different pulse designs and stated discretizations; these comparisons do not constitute a convergence test.'})

if __name__=='__main__':main()
