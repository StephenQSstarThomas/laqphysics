#!/usr/bin/env bash
#SBATCH --job-name=helium-followup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=helium-followup-%j.log
set -euo pipefail
repo_dir="${HELIUM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$repo_dir"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p build
flock build/compile.lock make all
python_bin="${HELIUM_PYTHON:-python3}"
group_name="${HELIUM_GROUP:-convergence}"
output_root="${HELIUM_OUTPUT_ROOT:-results/followup}"
args=(--group "$group_name" --config-dir "${HELIUM_CONFIG_DIR:-configs/followup}" --out-root "$output_root" --device cuda:0 --memory-fraction "${HELIUM_GPU_MEMORY_FRACTION:-0.8}")
if [[ -n "${HELIUM_CASE:-}" ]]; then args+=(--case "$HELIUM_CASE"); fi
srun "$python_bin" scripts/followup_campaign.py "${args[@]}"
