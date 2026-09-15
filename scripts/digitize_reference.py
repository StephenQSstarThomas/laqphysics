#!/usr/bin/env python3
"""Extract Fig.3(a) vector paths from the supplied PDF; no manual spectral shifts/fits.

Axis calibration uses the visible .52/.54 energy ticks and 0/1 probability ticks.
The small finite precision of the published vector paths remains part of this reference.
"""
from pathlib import Path
import re,sys,json,subprocess,xml.etree.ElementTree as ET
import numpy as np
from scipy.integrate import simpson
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from esss import driven
from pulses import Pulse
root=Path(__file__).resolve().parents[1];svg=root/'docs/literature/yu2018_page5.svg'
if not svg.exists():
    subprocess.run(['pdftocairo','-f','5','-l','5','-svg',str(root/'参考文献/Channel resolved e-e correlation.pdf'),str(svg)],check=True)
tree=ET.parse(svg).getroot();number=r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?'
out=root/'results/literature_comparison';out.mkdir(exist_ok=True);curves={}
for name,color,baseline in [('esss1','rgb(100%, 0%, 0%)',177.551712),('esss2','rgb(0%, 0%, 100%)',152.008220),
                            ('tdse1','rgb(0%, 74.900818%, 0%)',177.551712),('tdse2','rgb(100%, 65.097046%, 0%)',152.008220)]:
    points=[]
    for path in tree.iter():
        if path.get('stroke')!=color:continue
        d=path.get('d','')
        if name.startswith('esss') and len(d)<150:continue
        if name.startswith('tdse') and path.get('stroke-linecap')!='round':continue
        commands=re.findall('[A-Za-z]',re.sub(r'[eE][-+]?\d+','',d))
        if set(commands)-set('ML'):continue
        xy=np.array([float(x) for x in re.findall(number,d)]).reshape(-1,2)
        a,b,c,dd,e,f=map(float,re.findall(number,path.get('transform')))
        xx=a*xy[:,0]+c*xy[:,1]+e;yy=b*xy[:,0]+dd*xy[:,1]+f
        for x,y in zip(xx,yy):
            E=.52+(x-92.764152)*.02/(114.360658-92.764152)
            if .525<E<.66 and 75<y<179:
                P=(baseline-y)/(177.551712-126.408771)
                points.append((E,P))
    data=np.unique(np.array(points),axis=0);data=data[np.argsort(data[:,0])]
    curves[name]=data
    np.savetxt(out/(name+'.csv'),data,delimiter=',',header='electron_energy,probability_density')
    assert len(data)>60
e=np.linspace(.53,.65,481)
ground=json.loads((root/'results/ground_convergence.json').read_text())
g=next(r for r in ground if r['model']=='1d' and r['n']==512 and r['halfbox']==16.)
model=abs(driven(e,Pulse(cycles=60),eg=g['energy'],e1=g['ionic'][0],e2=g['ionic'][1],d12=g['d12'],dt=.05))**2
comparisons={};fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for j,ax in enumerate(axes):
    ref=curves[f'esss{j+1}'];paper=curves[f'tdse{j+1}'];rp=np.interp(e,ref[:,0],ref[:,1])
    comparisons[f'channel{j+1}']={'paper_esss_peak':float(ref[ref[:,1].argmax(),0]),
        'paper_tdse_peak':float(paper[paper[:,1].argmax(),0]),
        'our_esss_vs_paper_relative_L1':float(simpson(abs(model[j]-rp),x=e)/simpson(rp,x=e))}
    ax.plot(ref[:,0],ref[:,1],'k-',label='paper ESSS vector curve')
    ax.plot(paper[:,0],paper[:,1],'o',ms=2,label='paper TDSE markers')
    ax.plot(e,model[j],'--',label='our ESSS, converged model thresholds')
    for name in ['1d_dt0005','1d_radius_refined','1d_dvr_split']:
        file=root/'results'/name/'spectrum.npz'
        if not file.exists():continue
        a=np.load(file);pp=np.interp(e,a['energy'],a['pes'][j])
        comparisons[f'channel{j+1}'][name+'_vs_paper_esss_relative_L1']=float(simpson(abs(pp-rp),x=e)/simpson(rp,x=e))
        ax.plot(a['energy'],a['pes'][j],label=name)
    ax.set(xlim=(.53,.65),xlabel='Energy (a.u.)',ylabel='dP/dE',title=f'Fig.3(a), channel {j+1}');ax.legend(fontsize=7)
fig.savefig(out/'reference_comparison.png',dpi=180);plt.close(fig)
result={'source':'supplied Yu & Madsen PRA 98 033404 PDF, page 5 Fig.3(a)',
        'method':'vector paths with visible tick calibration, no fitted energy offset',
        'axis_calibration':{'E0':.52,'x0':92.764152,'E1':.54,'x1':114.360658,'P1_y0':177.551712,'P1_y1':126.408771},
        'computed_1d_thresholds':g,'comparisons':comparisons}
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(comparisons,indent=2))
assert max(c['our_esss_vs_paper_relative_L1'] for c in comparisons.values())<.04
