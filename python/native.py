"""Thin checked ctypes interface; Fortran column-major memory is part of the contract."""
import ctypes as ct
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB = ct.CDLL(str(ROOT / 'build' / os.environ.get('HELIUM_LIBRARY', 'libhelium.so')))
def ptr(a):
    return a.ctypes.data_as(ct.c_void_p)

def split_stage(psi, potential, momentum, dt, avec, stage):
    n = len(momentum)
    if n < 2 or n & (n - 1):
        raise ValueError('FFT size must be a power of two')
    if psi.shape != (n,n) or psi.dtype != np.complex128 or not psi.flags.f_contiguous:
        raise ValueError('psi must be complex128 and Fortran contiguous')
    if stage not in (1,2,3):
        raise ValueError('stage must be 1, 2 or 3')
    v = np.asfortranarray(potential, dtype=np.complex128)
    p = np.ascontiguousarray(momentum, dtype=np.float64)
    LIB.split2_stage(ct.c_int(n),ptr(psi),ptr(v),ptr(p),ct.c_double(dt),ct.c_double(avec),ct.c_int(stage))

def tensor_apply(h, psi, field=(0.,0.,0.), velocity=False):
    z = np.asfortranarray(np.asarray(psi).reshape(h.shape, order='F'),dtype=np.complex128)
    out = np.empty_like(z,order='F')
    f = np.ascontiguousarray(field,dtype=np.float64)
    if f.shape != (3,):
        raise ValueError('field must have three Cartesian components')
    args = [ct.c_int(h.n),ct.c_int(h.nc),ct.c_int(len(h.tv))]
    args += [ptr(getattr(h,a)) for a in ('ptr','col','tv','diag')]
    args += [ct.c_int(len(h.vc))]
    args += [ptr(getattr(h,a)) for a in ('vptr','vc','vl','vcoef','rad')]
    args += [ct.c_int(len(h.dc))]
    args += [ptr(getattr(h,a)) for a in ('dptr','dc','de','dcoef','ldiff','r')]
    args += [ptr(f),ct.c_int(int(velocity)),ptr(z),ptr(out)]
    LIB.tensor_apply(*args)
    return out.ravel(order='F')

def split_cached(psi,vhalf,pphase):
    n=len(pphase)
    if n<2 or n&(n-1) or psi.shape!=(n,n) or not psi.flags.f_contiguous or psi.dtype!=np.complex128:
        raise ValueError('power-of-two, complex128, Fortran-contiguous wavefunction required')
    LIB.split2_cached(ct.c_int(n),ptr(psi),ptr(vhalf),ptr(pphase))
