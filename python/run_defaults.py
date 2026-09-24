"""Defaults that simulate_spectrum.py fills in before hashing an input.

The hash of the returned dictionary is the run signature, so every tool that must
agree with a finished run (e.g. apply_ionic_probe.py) uses this one function.
"""
import json

def effective_config(config,storage_mode=None):
    c=json.loads(json.dumps(config))
    c.setdefault('storage',{'mode':'spectrum','max_file_bytes':4_000_000_000,'ionic_block_frames':128,'ionic_device':'cpu'})
    c['storage'].setdefault('accumulator_device','auto')
    if storage_mode:c['storage']['mode']=storage_mode
    c.setdefault('time_integrator','cf4-pade');c.setdefault('ionic_propagator','cf4')
    c.setdefault('spectrum',{'theta_points':24 if c.get('M',0) is None else 32,'phi_points':32 if c.get('M',0) is None else 1})
    return c
