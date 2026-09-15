#!/usr/bin/env bash
#SBATCH --job-name=helium-convergence
#SBATCH --array=0-7%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=helium-production-%A-%a.log
set -euo pipefail
repo_dir="${HELIUM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$repo_dir"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python_bin="${HELIUM_PYTHON:-python3}"
mkdir -p build
flock build/compile.lock make all
plan_file="${HELIUM_PLAN:-configs/followup/production_plan.json}"
config_dir="${HELIUM_CONFIG_DIR:-$(dirname "$plan_file")}"
case_name="$($python_bin -c 'import json,os,sys; p=json.load(open(sys.argv[1])); print(p["cases"][int(os.environ["SLURM_ARRAY_TASK_ID"])])' "$plan_file")"
srun "$python_bin" scripts/followup_campaign.py --case "$case_name" \
  --config-dir "$config_dir" \
  --out-root "${HELIUM_OUTPUT_ROOT:-results/followup}" --device cuda:0 \
  --memory-fraction "${HELIUM_GPU_MEMORY_FRACTION:-0.8}"
