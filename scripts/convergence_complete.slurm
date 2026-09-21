#!/usr/bin/env bash
#SBATCH --job-name=he-SI-convergence
#SBATCH --partition=GPU40G
#SBATCH --array=0-12%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=logs/he-SI-%A-%a.log
set -euo pipefail
repo_dir="${HELIUM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
export HELIUM_DEVICE=cuda:0
export HELIUM_PLAN="${HELIUM_PLAN:-configs/convergence_complete/production_plan.json}"
export HELIUM_STORAGE_MODE="${HELIUM_STORAGE_MODE:-projected}"
exec bash "$repo_dir/scripts/spectrum_job.sh"
