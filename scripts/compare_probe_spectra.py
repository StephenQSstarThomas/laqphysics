#!/usr/bin/env python3
"""Compare finished SI cases band-by-band against a reference case.

Two uses: (1) factorized probe (apply_ionic_probe.py) against the full three-pulse
TDSE of the same model; (2) probe sigma-/sigma+ against the two-pulse source or
the no-probe control. For every common ionic channel and the recorded total it
reports the band yield ratio, normalized-shape L1 difference, peak shift and,
when both files hold amplitudes on the same grid, the complex relative L2
difference. Nothing is aligned or rescaled.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.integrate import simpson
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'python'))
from surface_storage import atomic_json

COLORS=['#2a78d6','#eb6834','#1baf7a'];STYLES=['-','--',':']
INK='#0b0b0b';MUTED='#52514e'

def load(folder):
    folder=Path(folder);meta=json.loads((folder/'run.json').read_text(encoding='utf-8'))
    with np.load(folder/'spectrum.npz') as d:
        data={k:d[k] for k in d.files if k in ('energy','angle_integrated','labels','amplitudes','theta','phi','source_signature','extraction_end_time')}
    if str(data['source_signature'])!=meta['signature']:raise ValueError(f'{folder}: spectrum/configuration mismatch')
    data['labels']=[tuple(map(int,x)) for x in data['labels']];data['meta']=meta;data['folder']=str(folder.resolve());return data

def band(e,y,low,high):
    m=(e>=low)&(e<=high)
    if m.sum()<3:raise ValueError('window not covered by the energy grid')
    return e[m],y[m]

def metrics(ref,case,label,low,high):
    def curve(d):
        y=d['angle_integrated'].sum(axis=0) if label=='total' else d['angle_integrated'][d['labels'].index(label)]
        return band(d['energy'],y,low,high)
    ea,ya=curve(ref);eb,yb=curve(case);yb=np.interp(ea,eb,yb) if len(ea)!=len(eb) or np.any(ea!=eb) else yb
    Ya=simpson(ya,x=ea);Yb=simpson(yb,x=ea)
    row={'yield_reference':float(Ya),'yield_case':float(Yb),'yield_ratio':float(Yb/Ya) if Ya>0 else None,
         'peak_reference':float(ea[ya.argmax()]),'peak_case':float(ea[yb.argmax()])}
    row['peak_shift']=row['peak_case']-row['peak_reference']
    row['normalized_shape_L1']=float(simpson(abs(ya/Ya-yb/Yb),x=ea)) if Ya>0 and Yb>0 else None
    # Complex amplitudes carry the field-free ionic phase exp(-i E_c T): compare them only
    # for the same final time (e.g. factorized vs full three-pulse run of one input).
    same_end='extraction_end_time' in ref and 'extraction_end_time' in case and abs(float(ref['extraction_end_time'])-float(case['extraction_end_time']))<1e-9
    if label!='total' and same_end and 'amplitudes' in ref and 'amplitudes' in case and ref['amplitudes'].shape[1:]==case['amplitudes'].shape[1:] \
       and np.array_equal(ref['energy'],case['energy']):
        m=(ref['energy']>=low)&(ref['energy']<=high)
        a=ref['amplitudes'][ref['labels'].index(label)][m];b=case['amplitudes'][case['labels'].index(label)][m]
        row['complex_relative_L2']=float(np.linalg.norm(b-a)/np.linalg.norm(a)) if np.linalg.norm(a)>0 else None
    return row

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',required=True,help='LABEL=DIR of the reference case')
    p.add_argument('--case',action='append',required=True,help='LABEL=DIR; up to three')
    p.add_argument('--window',nargs=2,type=float,default=[.18,.42]);p.add_argument('--channels',nargs='*',default=['2,1,1'])
    p.add_argument('--out',required=True,help='output stem (writes .json and .png)')
    a=p.parse_args()
    if len(a.case)>3:raise ValueError('at most three compared cases per figure (facet further comparisons)')
    name,folder=a.reference.split('=',1);ref=load(folder);cases=[(n,load(f)) for n,f in (c.split('=',1) for c in a.case)]
    provenance=lambda d:{'run':d['folder'],'method':d['meta'].get('method','full TDSE'),'source_run':d['meta'].get('source_run'),
                         'configuration_signature':d['meta']['signature']}
    channels=['total']+[tuple(map(int,x.split(','))) for x in a.channels]
    low,high=a.window;result={'reference':{'label':name,**provenance(ref)},'window':[low,high],'comparisons':[]}
    for label,case in cases:
        rows={}
        for ch in channels:
            if ch!='total' and (ch not in ref['labels'] or ch not in case['labels']):continue
            rows[str(list(ch)) if ch!='total' else 'total']=metrics(ref,case,ch,low,high)
        result['comparisons'].append({'label':label,**provenance(case),'channels':rows})
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);atomic_json(out.with_suffix('.json'),result)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    shown=[c for c in channels if c=='total' or c in ref['labels']]
    fig,axes=plt.subplots(1,len(shown),figsize=(4.2*len(shown),3.8),layout='constrained',squeeze=False)
    for ax,ch in zip(axes[0],shown):
        y=lambda d:d['angle_integrated'].sum(axis=0) if ch=='total' else d['angle_integrated'][d['labels'].index(ch)]
        e,v=band(ref['energy'],y(ref),low,high);ax.plot(e,v,color=INK,lw=3.4,alpha=.85,label=name,zorder=1)
        for j,(label,case) in enumerate(cases):
            if ch!='total' and ch not in case['labels']:continue
            e,v=band(case['energy'],y(case),low,high);ax.plot(e,v,color=COLORS[j],lw=1.5,ls=STYLES[j],label=label,zorder=2+j)
        ax.set_title('recorded SI total' if ch=='total' else f'ion (n,l,m)={ch}',fontsize=10,color=INK)
        ax.set_xlabel('Electron energy (a.u.)',color=MUTED);ax.grid(alpha=.15);ax.tick_params(colors=MUTED)
        for s in ('top','right'):ax.spines[s].set_visible(False)
    axes[0][0].set_ylabel('dP/dE (a.u.), absolute',color=MUTED)
    handles,names=axes[0][0].get_legend_handles_labels()
    fig.legend(handles,names,loc='outside lower center',ncol=len(names),frameon=False,fontsize=9)
    fig.suptitle(f'Old band [{low:g}, {high:g}] a.u.: absolute spectra, no rescaling (numbers in {out.with_suffix(".json").name})',fontsize=9,color=INK)
    fig.savefig(out.with_suffix('.png'),dpi=180);plt.close(fig)
    for c in result['comparisons']:
        print(c['label'],{k:{m:(round(v,6) if isinstance(v,float) else v) for m,v in r.items() if m in ('yield_ratio','normalized_shape_L1','peak_shift','complex_relative_L2')} for k,r in c['channels'].items()})

if __name__=='__main__':main()
