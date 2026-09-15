"""Condon--Shortley spherical harmonics and exact Wigner/Gaunt angular algebra."""
from functools import lru_cache
import numpy as np
from sympy.physics.wigner import wigner_3j
from scipy.integrate import quad
from scipy.special import eval_genlaguerre, factorial

@lru_cache(None)
def C(bra,k,q,ket):
    l,m=bra;j,n=ket
    if m!=n+q:return 0.
    return float((-1)**m*np.sqrt((2*l+1)*(2*j+1))*wigner_3j(l,k,j,0,0,0)*wigner_3j(l,k,j,-m,q,n))

def cartesian(bra,ket):
    minus=C(bra,1,-1,ket);plus=C(bra,1,1,ket)
    return np.array([(minus-plus)/np.sqrt(2),1j*(minus+plus)/np.sqrt(2),C(bra,1,0,ket)],complex)

def coulomb(bra,ket,lam):
    a,b=bra;c,d=ket
    return sum((-1.)**q*C(a,lam,q,c)*C(b,lam,-q,d) for q in range(-lam,lam+1))

def channels(lmax,M=0):
    if lmax<0:raise ValueError('lmax>=0')
    one=[(l,m) for l in range(lmax+1) for m in range(-l,l+1)]
    return [(a,b) for a in one for b in one if M is None or a[1]+b[1]==M]

def radial_hydrogen(n,l,r,Z=2.):
    if n<1 or l<0 or l>=n:raise ValueError('requires 0<=l<n')
    x=2*Z*np.asarray(r)/n
    norm=(2*Z/n)**1.5*np.sqrt(factorial(n-l-1)/(2*n*factorial(n+l)))
    return norm*np.exp(-x/2)*x**l*eval_genlaguerre(n-l-1,2*l+1,x)

@lru_cache(None)
def radial_dipole(n,l,nn,ll,Z=2.):
    return quad(lambda r:r**3*radial_hydrogen(n,l,r,Z)*radial_hydrogen(nn,ll,r,Z),0,np.inf,
                epsabs=2e-12,epsrel=2e-12)[0]

def bound_dipole(bra,ket,q,Z=2.):
    n,l,m=bra;nn,ll,mm=ket
    return C((l,m),1,q,(ll,mm))*radial_dipole(n,l,nn,ll,Z)

def two_photon(bra,ket,q,omega,nmax=12,Z=2.):
    """Bound intermediate states only. Structural zeros are exact; nonzero absolute
    amplitudes are not converged two-photon rates (continuum also contributes).
    """
    ei=-Z**2/(2*ket[0]**2);amp=0.
    for n in range(1,nmax+1):
        for l in range(n):
            m=ket[2]+q
            if abs(m)>l:continue
            mid=(n,l,m);a=bound_dipole(bra,mid,q,Z)*bound_dipole(mid,ket,q,Z)
            if abs(a)<1e-15:continue
            den=ei+omega+Z**2/(2*n*n)
            if abs(den)<1e-12:raise ValueError('resonant intermediate state requires linewidth/time evolution')
            amp+=a/den
    return amp
