#!/usr/bin/env bash
#SBATCH --job-name=helium-oriented-ion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=helium-ion-%j.log
set -euo pipefail
cd "${HELIUM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python_bin="${HELIUM_PYTHON:-python3}"
mkdir -p build
flock build/compile.lock make all
srun "$python_bin" scripts/ionic_tdse_campaign.py --auto-resume --device cpu \
  --config "${HELIUM_CONFIG:-configs/followup/ion_2p_counter.json}" \
  --out "${HELIUM_OUT:-results/followup/ion_2p_counter}"
