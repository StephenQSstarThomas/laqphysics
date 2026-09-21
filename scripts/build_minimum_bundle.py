#!/usr/bin/env python3
"""Build a small standalone source/result handoff from explicitly selected files."""
import argparse,gzip,hashlib,io,json,os,subprocess,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
PREFIXES=('python/','src/','tests/','configs/preparation_20260921/','docs/iteration20260921/',
          'docs/cluster/user_slurm_20260921/','results/preparation_20260921/')
SCRIPTS=('simulate_spectrum.py','extract_projected.py','refine_online_spectrum.py','refine_preparation_spectra.sh',
         'render_spectra.py','summarize_spectra.py','audit_spectra_campaign.py','clean_surface_history.py',
         'snapshot_numerics.py','spectrum_job.sh','spectra_cpu.slurm','spectra_gpu.slurm','submit_spectra.sh',
         'make_preparation_campaign.py','make_preparation_production.py','analyze_preparation_controls.py')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=root/'deliveries/laqphysics-minimum-20260921.tar.gz')
    p.add_argument('--allow-dirty',action='store_true',help='diagnostic builds only; record every selected working-tree override');a=p.parse_args()
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
    explicit={'Makefile','requirements.txt','requirements-gpu.txt'}|{'scripts/'+s for s in SCRIPTS}
    members={name:root/name for name in tracked if name and (name in explicit or name.startswith(PREFIXES))}
    for target,source in [('README.md','README.md'),('handoff.sh','handoff.sh'),('verify_bundle.py','verify_bundle.py'),('configs/handoff_smoke.json','handoff_smoke.json')]:
        members[target]=root/'packaging/minimum'/source
    missing=explicit-set(members)
    if missing:raise ValueError('missing required tracked package files: '+str(sorted(missing)))
    changed=set(subprocess.check_output(['git','diff','HEAD','--name-only','-z'],cwd=root).decode().split('\0'))
    overrides=sorted(changed&{str(path.relative_to(root)) for path in members.values()})
    if overrides and not a.allow_dirty:raise ValueError('commit selected source changes before a release build, or use --allow-dirty for a diagnostic build')
    files={};payload={};modes={}
    for name,path in sorted(members.items()):
        if path.is_symlink():raise ValueError('package members must be ordinary files: '+name)
        data=path.read_bytes()
        if len(data)>50_000_000:raise ValueError('large simulation data do not belong in the minimal package: '+name)
        payload[name]=data;modes[name]=path.stat().st_mode&0o777
        files[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    manifest={'format':1,'package':'laqphysics-minimum-20260921','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
              'working_tree_override_files':overrides,'files':files,'full_production_convergence_certified':False,
              'scope':'Source, local tests, six completed SI energy spectra, provenance and production inputs; build/runtime dependencies and historical restart arrays are supplied separately.'}
    payload['bundle_manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode();modes['bundle_manifest.json']=0o644
    dest=a.output.resolve();dest.parent.mkdir(parents=True,exist_ok=True);temporary=dest.with_name(dest.name+'.tmp')
    with temporary.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed,tarfile.open(mode='w',fileobj=compressed) as archive:
        for name,data in sorted(payload.items()):
            info=tarfile.TarInfo('laqphysics-minimum-20260921/'+name);info.size=len(data);info.mode=modes[name];info.mtime=0
            archive.addfile(info,io.BytesIO(data))
    os.replace(temporary,dest);digest=hashlib.sha256(dest.read_bytes()).hexdigest()
    dest.with_name(dest.name+'.sha256').write_text(digest+'  '+dest.name+'\n')
    print(json.dumps({'archive':str(dest),'bytes':dest.stat().st_size,'sha256':digest,'members':len(payload),
                      'source_commit':manifest['source_commit'],'working_tree_override_files':overrides},indent=2))

if __name__=='__main__':main()
