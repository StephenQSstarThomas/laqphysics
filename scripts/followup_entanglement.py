#!/usr/bin/env python3
"""Conditional mode entanglement and exact post-pulse local dephasing controls.

Use the electron Schmidt subspace (rank <= number of ionic channels), so no
huge energy-angle density matrix is formed. This is not in-pulse decoherence.
"""
from pathlib import Path
import json,argparse,numpy as np
from scipy.integrate import simpson
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];parser=argparse.ArgumentParser();parser.add_argument('--out-root',default=str(root/'results/followup'));args=parser.parse_args()
base=Path(args.out_root);(base/'figures').mkdir(parents=True,exist_ok=True);rows=[]

def state_matrix(meta,data,window):
    E=data['energy'];select=(E>=window[0])&(E<=window[1]);e=E[select];we=simpson(np.eye(len(e)),x=e,axis=1)
    a=data['amplitudes'][:,select];labels=data['labels'];ni=len(labels)
    if meta['config'].get('M',0)==0:
        theta=data['theta'];wt=simpson(np.eye(len(theta)),x=theta,axis=1)*np.sin(theta)
        nf=2*int(max(abs(labels[:,2])))+1;phi=np.arange(nf)*2*np.pi/nf
        a=a[:,:,:,None]*np.exp(-1j*labels[:,2,None]*phi[None,:])[:,None,None,:]
        weight=(we*np.sqrt(2*e))[:,None,None]*wt[None,:,None]*np.full((1,1,nf),2*np.pi/nf)
    else:
        z,wt=np.polynomial.legendre.leggauss(20);a=a.reshape(ni,len(e),20,32)
        weight=(we*np.sqrt(2*e))[:,None,None]*wt[None,:,None]*np.full((1,1,32),2*np.pi/32)
    A=(a*np.sqrt(weight)[None,...]).reshape(ni,-1);prob=float(np.sum(abs(A)**2))
    # Small Gram eigenproblem also avoids allocating the large right singular vectors.
    rho=A@A.conj().T/prob;lam,U=np.linalg.eigh(rho);lam=lam.clip(0,1)
    keep=lam>1e-14;B=U[:,keep]*np.sqrt(lam[keep])[None,:]
    return B,prob,lam

for name in ['pump_only','pump_probe_plus','pump_probe_minus','pump_probe_linear','pump_probe_half_frequency','pump_probe_plus_delay192','res_l6_L2_p4','res_l6_L2_p6','res_l8_L2_p6']:
    out=base/name
    if not (out/'spectrum.npz').exists():continue
    meta=json.loads((out/'run.json').read_text());data=np.load(out/'spectrum.npz')
    windows=[(.18,.42),(.52,.68),(float(data['energy'][0]),float(data['energy'][-1]))] if name.startswith('pump') else [(.54,.65)]
    for window in windows:
        e=data['energy'];lo=max(window[0],e[0]);hi=min(window[1],e[-1])
        if hi<=lo:continue
        B,prob,lam=state_matrix(meta,data,(lo,hi));ni,ne=B.shape
        if prob<1e-12:continue
        rho=np.outer(B.ravel(),B.ravel().conj()).reshape(ni,ne,ni,ne);original=np.einsum('aiaj->ij',rho);deph=[]
        for gammaT in [0,.2,.5,1,2,5,10]:
            r=rho.copy();factor=np.eye(ni)+(1-np.eye(ni))*np.exp(-gammaT);r*=factor[:,None,:,None]
            pt=r.transpose(2,1,0,3).reshape(ni*ne,ni*ne);negative=float((np.sum(abs(np.linalg.eigvalsh(pt)))-1)/2)
            difference=float(np.max(abs(np.einsum('aiaj->ij',r)-original)))
            deph.append({'gammaT':gammaT,'negativity':negative,'electron_reduced_state_change':difference})
            assert difference<1e-12
        nz=lam[lam>1e-14]
        rows.append({'run':name,'energy_window':[lo,hi],'recorded_sector_probability':prob,
                     'entropy_bits':float(-sum(nz*np.log2(nz))),'purity':float(sum(lam**2)),
                     'post_pulse_local_dephasing':deph,
                     'scope':'conditional on recorded ionic channels/window; post-pulse CPTP control, not dynamical bath simulation'})
(base/'entanglement.json').write_text(json.dumps(rows,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
for r in rows:
    if r['energy_window'][0]>=.5:
        d=r['post_pulse_local_dephasing'];ax.plot([x['gammaT'] for x in d],[x['negativity'] for x in d],label=r['run'])
ax.set(xlabel='gamma x post-pulse dephasing time',ylabel='Conditional negativity',title='Measured TDSE amplitudes: post-pulse local dephasing control');ax.legend(fontsize=7)
fig.savefig(base/'figures/post_pulse_dephasing.png',dpi=180);plt.close(fig)
print('Analyzed conditional sectors:',len(rows))
