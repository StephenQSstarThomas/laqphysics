#!/usr/bin/env bash
# Run after production/reference finishes, using its recorded numerical snapshot.
set -euo pipefail
# Reports and indexes contain UTF-8 (Chinese, sigma, <=). Never depend on the node locale.
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
case_dir="${1:?Usage: refine_preparation_spectra.sh CASE_DIR NUMERICAL_SNAPSHOT}"
snapshot_dir="${2:?Pass the SNAPSHOT path printed by the original job}"
python_bin="${HELIUM_PYTHON:-python3}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
common=(--out "$case_dir" --snapshot "$snapshot_dir" --device "${HELIUM_DEVICE:-cpu}")
"$python_bin" "$repo_dir/scripts/refine_online_spectrum.py" "${common[@]}" --theta 32 --phi 48 --name spectrum_angles.npz
"$python_bin" "$repo_dir/scripts/refine_online_spectrum.py" "${common[@]}" --energy .18 .75 2281 --name spectrum_energy.npz
"$python_bin" "$repo_dir/scripts/refine_online_spectrum.py" "${common[@]}" --time-stride 2 --name spectrum_stride2.npz
