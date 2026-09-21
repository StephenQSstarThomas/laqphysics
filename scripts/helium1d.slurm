#!/usr/bin/env bash
#SBATCH --job-name=helium1d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/helium1d-%j.log
set -euo pipefail
repo_dir="${HELIUM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$repo_dir"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p build
flock build/compile.lock make all
python_bin="${HELIUM_PYTHON:-python3}"
config_file="${HELIUM_CONFIG:-configs/1d_production.json}"
result_dir="${HELIUM_OUT:-results/slurm-1d-${SLURM_JOB_ID:-local}}"
if [[ -e "$result_dir" ]]; then
    echo "Use a fresh HELIUM_OUT. The 1D driver is not restartable." >&2
    exit 2
fi
srun --cpu-bind=cores "$python_bin" scripts/run1d.py --config "$config_file" --out "$result_dir"
