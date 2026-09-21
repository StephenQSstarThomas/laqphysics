#!/usr/bin/env bash
#SBATCH --job-name=he-SI-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=logs/he-SI-CPU-%A-%a.log
set -euo pipefail
repo_dir="${HELIUM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
export HELIUM_DEVICE=cpu
export HELIUM_CONFIG="${HELIUM_CONFIG:-configs/3d_resonant.json}"
exec bash "$repo_dir/scripts/spectrum_job.sh"
