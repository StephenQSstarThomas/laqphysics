#!/usr/bin/env bash
#SBATCH --job-name=helium-full-convergence
#SBATCH --array=0-12%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=helium-convergence-%A-%a.log
set -euo pipefail
repo_dir="${HELIUM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$repo_dir"
python_bin="${HELIUM_PYTHON:-python3}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
plan_file="${HELIUM_PLAN:-configs/convergence_complete/production_plan.json}"
config_dir="${HELIUM_CONFIG_DIR:-$(dirname "$plan_file")}"
output_root="${HELIUM_OUTPUT_ROOT:?Set HELIUM_OUTPUT_ROOT to persistent scratch before submitting}"
export HELIUM_SURFACE_ROOT="${HELIUM_SURFACE_ROOT:-$output_root/raw}"
mkdir -p "$output_root/code" build
flock build/compile.lock make all
snapshot_dir="$(flock "$output_root/code/.snapshot.lock" "$python_bin" scripts/snapshot_numerics.py --root "$output_root/code")"
case_name="$("$python_bin" -c 'import json,os,sys; print(json.load(open(sys.argv[1]))["cases"][int(os.environ["SLURM_ARRAY_TASK_ID"])])' "$plan_file")"
srun "$python_bin" "$snapshot_dir/scripts/followup_campaign.py" --case "$case_name" \
  --config-dir "$config_dir" --out-root "$output_root" --device cuda:0 \
  --memory-fraction "${HELIUM_GPU_MEMORY_FRACTION:-0.8}" --preconditioner-precision complex64 --propagate-only
# A time-limit signal saves a valid checkpoint but does not complete a spectrum.
"$python_bin" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["complete"] else 95)' "$output_root/$case_name/run.json"
"$python_bin" "$snapshot_dir/scripts/wait_extract.py" --out "$output_root/$case_name" --snapshot "$snapshot_dir" --ionic-device cpu --reuse-spectrum
