#!/usr/bin/env bash
#SBATCH --job-name=helium3d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=1-00:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=logs/helium3d-%j.log
set -euo pipefail
# Pass site-specific --partition and --account to sbatch; none are guessed here.
repo_dir="${HELIUM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$repo_dir"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo_dir/python${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${HELIUM_PYTHON:-python3}"
config_file="${HELIUM_CONFIG:-configs/3d_resonant.json}"
result_dir="${HELIUM_OUT:-results/slurm-3d-${SLURM_JOB_ID:-local}}"
mkdir -p build
flock build/compile.lock make all
args=(--config "$config_file" --out "$result_dir")
if [[ -f "$result_dir/checkpoint.npz" ]]; then args+=(--resume); fi
srun --cpu-bind=cores "$python_bin" -m run3d "${args[@]}"
# A pre-timeout signal leaves a valid incomplete checkpoint; only extract completed jobs.
if "$python_bin" -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1]))["complete"])' "$result_dir/run.json"; then
    srun --cpu-bind=cores "$python_bin" -m run3d --out "$result_dir" --extract
fi
