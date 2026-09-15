#!/usr/bin/env python3
"""Radial Green-function checks of the proposed 2p->3p two-photon control.

The finite radial resolvent includes bound AND discretized continuum intermediate
states. AC Stark shifts are perturbative estimates, not full strong-field rates.
"""
from pathlib import Path
import sys,json,numpy as np
from scipy.linalg import solve
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from fedvr import make_grid
from angular import C,radial_hydrogen,two_photon

root=Path(__file__).resolve().parents[1];out=root/'results/followup/ionic_virtual';out.mkdir(exist_ok=True)
edges=[0,.25,.5,1,2,3,4,6,8,12,16,24,32,48,64,80]

def make_solver(order):
    grid=make_grid(edges,order);r=grid.r.real;w=grid.weights.real;T=grid.kinetic.toarray().real
    def u(n,l):return r*radial_hydrogen(n,l,r)*np.sqrt(w)
    def H(l):return T+np.diag(-2/r+l*(l+1)/(2*r*r))
    def amplitude(initial,final,q,omega):
        ni,li,mi=initial;nf,lf,mf=final;Ei=-2/ni**2;answer=0.
        if abs(mf)>lf:return 0.
        for lm in [li-1,li+1]:
            mm=mi+q
            if lm<0 or abs(mm)>lm:continue
            angular=C((lf,mf),1,q,(lm,mm))*C((lm,mm),1,q,(li,mi))
            if not angular:continue
            z=solve((Ei+omega)*np.eye(len(r))-H(lm),r*u(ni,li),assume_a='sym')
            answer+=angular*np.dot(r*u(nf,lf),z)
        return float(answer)
    def shift_coefficient(state,q,omega):
        n,l,m=state;E=-2/n**2;answer=0.
        for frequency,helicity in [(omega,q),(-omega,-q)]:
            for lm in [l-1,l+1]:
                mm=m+helicity
                if lm<0 or abs(mm)>lm:continue
                angular=C((lm,mm),1,helicity,(l,m))
                source=angular*r*u(n,l)
                z=solve((E+frequency)*np.eye(len(r))-H(lm),source,assume_a='sym')
                answer+=np.dot(source,z)
        return float(answer/4)  # delta E = F0^2 * coefficient
    return amplitude,shift_coefficient

rows=[]
for order in [6,8,10,12]:
    amplitude,shift=make_solver(order);omega=5/36
    M=amplitude((2,1,1),(3,1,-1),-1,omega)
    forbidden=amplitude((2,1,1),(3,2,-1),-1,omega)
    si=shift((2,1,1),-1,omega);sf=shift((3,1,-1),-1,omega)
    row={'order':order,'two_photon_2p_to_3p':M,'two_photon_2p_to_3d':forbidden,
         'initial_stark_coefficient':si,'final_stark_coefficient':sf,
         'peak_differential_shift_at_F003':(sf-si)*.003**2,
         'peak_two_photon_rabi_at_F003':abs(M)*.003**2/2}
    rows.append(row);print(row,flush=True)
bound=[]
for nmax in [4,8,12,20]:
    bound.append({'nmax':nmax,'bound_only_amplitude':two_photon((3,1,-1),(2,1,1),-1,5/36,nmax=nmax)})
result={'model':'He+ Z=2; full radial resolvent, electric dipole, field-free two-photon resonance',
        'radial_convergence':rows,'bound_only_comparison':bound,
        'limitation':'Stark estimates assume perturbative virtual dressing. Near intermediate resonances, use full time evolution; no experimental transfer efficiency is claimed.'}
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(bound,indent=2))
