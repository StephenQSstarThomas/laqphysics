#!/usr/bin/env bash
set -euo pipefail
package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$package_root"
python_bin="${HELIUM_PYTHON:-python3}"
output_root="${HELIUM_OUTPUT_ROOT:-$package_root/local_runs}"
export PYTHONPATH="$package_root/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ $# == 0 && -t 0 ]]; then
  echo '1 校验包  2 编译与测试  3 CPU 小算例  4 重画已交付谱  5 生产资源估算'
  read -r -p '选择编号（其他输入退出）：' choice
  case "$choice" in 1) set -- verify;; 2) set -- test;; 3) set -- smoke;; 4) set -- render;; 5) set -- estimate;; *) exit 0;; esac
fi
action="${1:-help}"; if [[ $# -gt 0 ]]; then shift; fi
case "$action" in
  verify) exec "$python_bin" verify_bundle.py;;
  test)
    make all debug
    mkdir -p "$output_root/checks" "$output_root/test_tmp"
    export TMPDIR="$output_root/test_tmp"
    "$python_bin" -m pytest -q tests | tee "$output_root/checks/tests_release.log"
    HELIUM_LIBRARY=libhelium_debug.so "$python_bin" -m pytest -q tests | tee "$output_root/checks/tests_debug.log"
    ;;
  smoke)
    make all
    exec "$python_bin" scripts/simulate_spectrum.py --config configs/handoff_smoke.json --out-root "$output_root/smoke" --device "${1:-cpu}" --cpu-threads "${HELIUM_TORCH_THREADS:-2}"
    ;;
  render)
    "$python_bin" - "$output_root/replotted" <<'PY'
import shutil,sys
from pathlib import Path
source=Path('results/preparation_20260921').resolve();target=Path(sys.argv[1]).resolve()
if source==target or source in target.parents:raise ValueError('Choose a separate output root for re-rendering')
shutil.copytree(source,target,dirs_exist_ok=True)
PY
    exec "$python_bin" scripts/render_spectra.py --out-root "$output_root/replotted"
    ;;
  estimate)
    make all
    exec "$python_bin" scripts/simulate_spectrum.py --config "${1:-configs/preparation_20260921/production/reference.json}" --out-root "$output_root/production" --estimate-only
    ;;
  snapshot)
    make all
    exec "$python_bin" scripts/snapshot_numerics.py --root "$output_root/code"
    ;;
  submit)
    : "${HELIUM_OUTPUT_ROOT:?Set HELIUM_OUTPUT_ROOT to persistent cluster scratch before submitting}"
    export HELIUM_REPO="$package_root"
    if [[ $# == 0 ]]; then set -- gpu; fi
    exec scripts/submit_spectra.sh "$@"
    ;;
  *)
    echo './handoff.sh verify | test | smoke [cpu|cuda:0] | render | estimate [CONFIG] | snapshot | submit [gpu|cpu] [sbatch options]'
    echo 'HELIUM_PYTHON selects Python; HELIUM_OUTPUT_ROOT selects all new outputs (local default: ./local_runs).'
    ;;
esac
