#!/usr/bin/env bash
# Slurm template: use only after confirming the target PACE system uses sbatch.
# All scheduler options (allocation/partition/resources) come from the caller.
set -euo pipefail

if [[ "${1:-}" == "--help" || "$#" -eq 0 ]]; then
  cat <<'EOF'
Usage: bash scripts/pace/submit_checks.sh <your sbatch resource/options...>

Examples of options to supply after confirming local PACE policy:
  --account=<your allocation> --partition=<your partition>
  --time=<your limit> --mem=<your memory request> --cpus-per-task=<your CPUs>

Set WTUMI_PYTHON to the prepared Linux Python executable.
Optional: WTUMI_WORKDIR, WTUMI_CONFIG, WTUMI_OUTPUT_ROOT, WTUMI_DEMO_CONFIG.
For a rendering job, set MUJOCO_GL and WTUMI_TEST_MUJOCO_RENDER as appropriate.
This template does not choose an account, queue, GPU or software module.
EOF
  exit 0
fi
if ! command -v sbatch >/dev/null 2>&1; then
  printf 'sbatch is unavailable. This file is a Slurm template, not a verified scheduler setup.\n' >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_dir="${WTUMI_WORKDIR:-$(cd -- "$script_dir/../.." && pwd -P)}"
project_dir="$(cd -- "$project_dir" && pwd -P)"
export WTUMI_WORKDIR="$project_dir"
: "${WTUMI_PYTHON:=$project_dir/.venv/bin/python}"
export WTUMI_PYTHON
if [[ ! -x "$WTUMI_PYTHON" ]]; then
  printf 'Prepared Python executable not found: %s\n' "$WTUMI_PYTHON" >&2
  exit 2
fi
# Submit an explicit script; arguments are forwarded as sbatch flags, not shell code.
exec sbatch --chdir="$project_dir" "$@" "$script_dir/run_checks.sh"
