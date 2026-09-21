import json,subprocess,sys
from pathlib import Path
import numpy as np

def case(tmp_path,complete=True):
    out=tmp_path/'case';out.mkdir()
    np.save(out/'flux.npy',np.ones((3,2,1,1),complex))
    np.savez(out/'spectrum.npz',source_signature='test',angle_integrated=np.array([[1.,2.,1.]]))
    (out/'run.json').write_text(json.dumps({'complete':complete,'signature':'test','config':{},'nsteps':2,'nrad':2,'channels':1,'surface_indices':[0]}))
    return out

def invoke(out,*args):
    script=Path(__file__).resolve().parents[1]/'scripts/clean_surface_history.py'
    return subprocess.run([sys.executable,str(script),'--out',str(out),*args],capture_output=True,text=True)

def test_raw_cleanup_is_readonly_by_default_and_requires_retention_choice(tmp_path):
    out=case(tmp_path);assert invoke(out).returncode==0 and (out/'flux.npy').exists()
    assert invoke(out,'--apply').returncode!=0 and (out/'flux.npy').exists()
    assert invoke(out,'--apply','--final-spectrum-only').returncode==0
    assert not (out/'flux.npy').exists() and (out/'spectrum.npz').exists()

def test_incomplete_raw_history_is_never_removed(tmp_path):
    out=case(tmp_path,False);assert invoke(out,'--apply','--final-spectrum-only').returncode!=0
    assert (out/'flux.npy').exists()
