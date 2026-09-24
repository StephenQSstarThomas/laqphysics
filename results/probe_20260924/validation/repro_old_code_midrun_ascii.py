#!/usr/bin/env python3
"""Reproduce the cluster failure: HELIUM_BUNDLE=<unpacked code root> python this.py <simulate_spectrum args>.

Start Python with a UTF-8 locale and PYTHONUTF8=0; LC_CTYPE is switched to C just
before publication, as happened on the compute node after propagation. With the
20260921 bundle this raises the reported UnicodeEncodeError (spectrum_report.py
lines 141/177); with the fixed code the same run publishes normally.
"""
import locale,os,sys,runpy
root=os.environ['HELIUM_BUNDLE'];sys.path.insert(0,os.path.join(root,'python'))
import spectrum_report
original=spectrum_report.publish
def publish(*args,**kwargs):
    locale.setlocale(locale.LC_CTYPE,'C')
    return original(*args,**kwargs)
spectrum_report.publish=publish
sys.argv=[os.path.join(root,'scripts/simulate_spectrum.py')]+sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
