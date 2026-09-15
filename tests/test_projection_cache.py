"""A changed physical cutoff implementation must invalidate projected histories."""
import importlib.util
import shutil
from pathlib import Path

def test_projection_cache_tracks_cutoff_implementation(tmp_path):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('extract_projected_cache_test',root/'scripts/extract_projected.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (tmp_path/'python').mkdir()
    for source in (root/'python').glob('*.py'):shutil.copy2(source,tmp_path/'python'/source.name)
    module.repo=tmp_path;before=module.source_signature()
    # Substituting this dependency used to leave the projection signature fixed.
    (tmp_path/'python/tdse1d.py').write_text('def cutoff(r, inner, outer):\n    return 0*r\n')
    assert module.source_signature()!=before
