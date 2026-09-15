#!/usr/bin/env python3
"""Reproduce published ESSS panels; independently verify dynamics and physics controls."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.integrate import simpson
from scipy.signal import find_peaks
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from pulses import Pulse
from esss import analytic,driven,reduced_ion
from angular import bound_dipole,two_photon

def negativity(rho,ni,ne):
    pt=rho.reshape(ni,ne,ni,ne).transpose(2,1,0,3).reshape(ni*ne,ni*ne)
    return float((np.linalg.norm(np.linalg.eigvalsh(pt),ord=1)-1)/2)

def main():
    root=Path(__file__).resolve().parents[1];out=root/'results/esss';out.mkdir(parents=True,exist_ok=True)
    gs=next(r for r in json.loads((root/'results/ground_convergence.json').read_text()) if r['model']=='1d' and r['n']==512 and r['halfbox']==16.)
    eg=gs['energy'];e1=gs['ionic'][0]
    e=np.linspace(.50,.70,1201);rows=[];fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for cycles,ax in zip((60,120,180,360),axs.flat):
        p=Pulse(cycles=cycles);a,pg=analytic(e,p,eg=eg,e1=e1,nt=8001);P=abs(a)**2
        peaks=[e[find_peaks(y,prominence=y.max()*.05)[0]].tolist() for y in P]
        probability=float(simpson(P.sum(axis=0),x=e));check=driven(e[::20],p,eg=eg,e1=e1,e2=e1+p.omega,dt=.1)
        _,ent=reduced_ion(a,np.full(len(e),e[1]-e[0]))
        row={'cycles':cycles,'ground_population':pg,'continuum_window_probability':probability,'balance_error':1-pg-probability,
             'peak_energies':peaks,'ode_max_amplitude_error':float(np.max(abs(check-a[:,::20]))),'entanglement':ent}
        rows.append(row);np.savez(out/f'nc{cycles}.npz',energy=e,amplitudes=a,pes=P,pg=pg)
        for j in range(2):ax.plot(e,P[j],label=f'ionic channel {j+1}')
        ax.set(title=f'{cycles} cycles',xlabel='Electron energy (a.u.)',ylabel='dP/dE (a.u.)',xlim=(.55,.65));ax.legend()
    fig.suptitle('Yu & Madsen (2018), Eqs. 19–21: computed 1D thresholds');fig.savefig(out/'published_esss_panels.png',dpi=180);plt.close(fig)
    # Angular extension for a 1S initial state and one-photon linearly polarized source:
    # amplitude factor Y10; ion-only dressing preserves this minimal-model angular factor.
    d3=bound_dipole((2,1,0),(1,0,0),0);p=Pulse(cycles=240)
    aa,_=analytic(e,p,d12=d3,eg=-2.903724377,nt=10001)
    P=abs(aa)**2;P/=simpson(P.sum(axis=0),x=e)
    theta=np.linspace(0,np.pi,181);ang=3/(4*np.pi)*np.cos(theta)**2
    density=P.sum(axis=0)[:,None]*ang
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');im=ax.pcolormesh(theta*180/np.pi,e,density,shading='auto')
    ax.set(xlabel='Emission angle (degrees)',ylabel='Energy (a.u.)',ylim=(.55,.65),title='3D angular ESSS: normalized shape; source strength is not a 3D ab initio result')
    fig.colorbar(im,ax=ax,label='Normalized dP/dE/dOmega');fig.savefig(out/'angular_esss.png',dpi=180);plt.close(fig)
    # Exact local CPTP maps on the ion; keep energy marginal and mixed-state negativity.
    coarse=aa[:,::30];coarse/=np.linalg.norm(coarse);ne=coarse.shape[1]
    rho=np.outer(coarse.ravel(),coarse.ravel().conj());before=np.sum(abs(coarse)**2,axis=0)
    deco=[]
    for gammaT in (0,.2,.5,1,2,5,20):
        eta=np.exp(-gammaT);r=rho.reshape(2,ne,2,ne).copy();r[0,:,1,:]*=eta;r[1,:,0,:]*=eta
        marginal=np.einsum('aiaj->ij',r).diagonal().real
        deco.append({'gamma_times_duration':gammaT,'negativity':negativity(r.reshape(2*ne,2*ne),2,ne),
                     'electron_marginal_max_change':float(max(abs(marginal-before)))})
    U=expm(-.83j*np.array([[0,1],[1,0]]));rot=U@coarse
    controls={'local_unitary_marginal_error':float(max(abs(np.sum(abs(rot)**2,axis=0)-before))),
              'local_unitary_conditional_change':float(max(abs(abs(rot[0])**2-abs(coarse[0])**2))),
              'd12_3d':d3,'d12_1d':.4823,'rabi_frequency_3d':.0534*d3,
              'expected_one_photon_energy':1.5-(-2-(-2.903724377)),
              'two_photon_2p_to_3d_sigma_minus':two_photon((3,2,-1),(2,1,1),-1,5/36,nmax=10),
              'two_photon_2p_to_3p_sigma_minus_bound_intermediates':two_photon((3,1,-1),(2,1,1),-1,5/36,nmax=10),
              'note':'Nonzero two-photon amplitude uses bound intermediates only; not a quantitative transition rate.',
              'dephasing':deco}
    # Short animation of pulse-duration dependence, explicitly model predictions.
    energies=np.linspace(.55,.65,301);fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    lines=[ax.plot([],[],label=f'channel {j+1}')[0] for j in range(2)]
    ax.set(xlim=(.55,.65),ylim=(0,35),xlabel='Energy (a.u.)',ylabel='dP/dE');ax.legend()
    frames=[]
    for N in np.linspace(40,360,33):frames.append(abs(analytic(energies,Pulse(cycles=float(N)),eg=eg,e1=e1,nt=4001)[0])**2)
    def update(i):
        for j in range(2):lines[j].set_data(energies,frames[i][j])
        ax.set_title(f'1D ESSS pulse-duration scan: {40+10*i} cycles');return lines
    FuncAnimation(fig,update,frames=len(frames),interval=120).save(out/'duration_scan.gif',writer=PillowWriter(fps=8));plt.close(fig)
    (out/'verification.json').write_text(json.dumps({'thresholds':{'eg':eg,'e1':e1,'e2_for_exact_resonance':e1+1.5},'published_parameters':rows,'physics_controls':controls},indent=2)+'\n')
    print(json.dumps({'panels':rows,'physics_controls':controls},indent=2))
    assert max(r['ode_max_amplitude_error'] for r in rows)<2e-7
    assert controls['local_unitary_marginal_error']<1e-14
if __name__=='__main__':main()
