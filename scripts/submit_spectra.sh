#!/usr/bin/env bash
# Usage: HELIUM_OUTPUT_ROOT=/scratch/... scripts/submit_spectra.sh gpu [sbatch options]
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
target="${1:-gpu}"; if [[ $# -gt 0 ]]; then shift; fi
case "$target" in gpu|cpu) ;; *) echo 'First argument must be gpu or cpu.' >&2; exit 2;; esac
: "${HELIUM_OUTPUT_ROOT:?Set HELIUM_OUTPUT_ROOT to persistent scratch}"
mkdir -p logs
# logs must exist BEFORE sbatch opens its output, not inside the job body.
exec sbatch "$@" "scripts/spectra_${target}.slurm"
