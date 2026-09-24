#!/usr/bin/env bash
# Common body for the supplied site's CPU/GPU sbatch entry points.
set -euo pipefail
# Reports and indexes contain UTF-8 (Chinese, sigma, <=). Never depend on the node locale.
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
repo_dir="${HELIUM_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$repo_dir"
if [[ "${SLURM_NTASKS:-1}" != 1 ]]; then
  echo 'This solver uses one process per case; use job arrays for independent cases, not multiple MPI ranks.' >&2
  exit 2
fi
output_root="${HELIUM_OUTPUT_ROOT:?Set HELIUM_OUTPUT_ROOT to persistent scratch}"
python_bin="${HELIUM_PYTHON:-python3}"
export OMP_NUM_THREADS="${HELIUM_OMP_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_PROC_BIND=spread OMP_PLACES=cores
torch_threads="${HELIUM_TORCH_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
plan_file="${HELIUM_PLAN:-configs/preparation_20260921/plan.json}"
config_dir="${HELIUM_CONFIG_DIR:-$(dirname "$plan_file")}"
if [[ -n "${HELIUM_CONFIG:-}" ]]; then
  config_file="$HELIUM_CONFIG"
else
  case_name="$("$python_bin" -c 'import json,os,sys;p=json.load(open(sys.argv[1]));names=p.get("primary_cases",p["cases"]);print(names[int(os.environ.get("SLURM_ARRAY_TASK_ID","0"))])' "$plan_file")"
  config_file="$config_dir/$case_name.json"
fi
mkdir -p build "$output_root/code"
flock build/compile.lock make all
if [[ -n "${HELIUM_SNAPSHOT:-}" ]]; then
  snapshot_dir="$HELIUM_SNAPSHOT"
else
  snapshot_dir="$(flock "$output_root/code/.snapshot.lock" "$python_bin" scripts/snapshot_numerics.py --root "$output_root/code")"
fi
args=(--config "$config_file" --out-root "$output_root" --device "${HELIUM_DEVICE:-cuda:0}"
      --memory-fraction "${HELIUM_GPU_MEMORY_FRACTION:-0.8}" --cpu-threads "$torch_threads" --tag "${HELIUM_RUN_TAG:-production}")
if [[ -n "${HELIUM_STORAGE_MODE:-}" ]]; then args+=(--storage "$HELIUM_STORAGE_MODE"); fi
echo "INPUT=$config_file OUTPUT_ROOT=$output_root SNAPSHOT=$snapshot_dir THREADS=$torch_threads"
# The Python entry does propagation, final spectrum and named PNG/PDF/CSV together.
# Exit 95 means a valid checkpoint, not a completed single-ionization spectrum.
srun --cpu-bind=cores "$python_bin" "$snapshot_dir/scripts/simulate_spectrum.py" "${args[@]}"
