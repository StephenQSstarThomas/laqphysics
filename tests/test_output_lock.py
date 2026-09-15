import subprocess,sys
import pytest
from output_lock import exclusive_output

def test_another_process_cannot_write_until_owner_releases(tmp_path):
    @exclusive_output('.propagation.lock')
    def write(config,out):
        (out/'result.txt').write_text('one writer')
    code='import fcntl,sys; f=open(sys.argv[1],"a"); fcntl.flock(f,fcntl.LOCK_EX); print("locked",flush=True); sys.stdin.readline()'
    owner=subprocess.Popen([sys.executable,'-c',code,str(tmp_path/'.propagation.lock')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        assert owner.stdout.readline().strip()=='locked'
        with pytest.raises(RuntimeError,match='refusing concurrent output writes'):write({},tmp_path)
        assert not (tmp_path/'result.txt').exists()
    finally:
        owner.communicate('\n',timeout=10)
    write({},tmp_path)
    assert (tmp_path/'result.txt').read_text()=='one writer'
