#!/usr/bin/env bash
#SBATCH --job-name=he-SI-preparation
#SBATCH --partition=GPU40G
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
export HELIUM_CONFIG="${HELIUM_CONFIG:-configs/preparation_20260921/production/reference.json}"
exec bash "$repo_dir/scripts/spectrum_job.sh"
