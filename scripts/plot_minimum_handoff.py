#!/usr/bin/env python3
"""Plot the actual long-pulse radial comparison in absolute energy."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--out-root',default='results/convergence_complete');a=p.parse_args();root=Path(a.out_root)
audit=json.loads((root/'handoff_acceptance.json').read_text())
row=next(x for x in audit['checks'] if x['check']=='near-nuclear radial 12 -> 14')
if not isinstance(row['details'],dict):raise ValueError('full radial comparison is not yet available')
metrics=row['details'];first=np.load(root/'reference/spectrum.npz');second=np.load(root/'inner14/spectrum.npz')
fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True,layout='constrained');e=second['energy']
for col,(label,title) in enumerate([((1,0,0),'Residual ion 1s'),((2,1,0),'Residual ion 2p, m=0')]):
    ia=list(map(tuple,first['labels'])).index(label);ib=list(map(tuple,second['labels'])).index(label)
    pa=np.interp(e,first['energy'],first['angle_integrated'][ia]);pb=second['angle_integrated'][ib]
    axes[0,col].plot(e,pa,label='inner order 12',lw=1.5)
    axes[0,col].plot(e,pb,'--',label='inner order 14',lw=1.3)
    axes[0,col].set(title=title,ylabel='dP/dE (a.u.)');axes[0,col].legend(fontsize=8)
    axes[1,col].plot(e,100*(pb-pa)/pa.max(),color='C2');axes[1,col].axhline(0,color='0.6',lw=.7)
    axes[1,col].set(xlabel='Electron energy (a.u.)',ylabel='Change / reference peak (%)',
                    title=f"Shape L1: {100*metrics['normalized_shape_L1'][col]:.3f}%; yield: {100*metrics['relative_yield_change'][col]:+.4f}%")
    for ax in axes[:,col]:ax.grid(alpha=.2);ax.set_xlim(e[0],e[-1])
fig.suptitle('180-cycle TDSE: independent inner radial refinement, no peak alignment')
folder=root/'figures';folder.mkdir(exist_ok=True);fig.savefig(folder/'radial_minimum_handoff.png',dpi=180)
plt.close(fig);print(folder/'radial_minimum_handoff.png')
