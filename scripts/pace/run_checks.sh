#!/usr/bin/env bash
# Local or allocated compute-node checks. This script never submits a job.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_dir="${WTUMI_WORKDIR:-$(cd -- "$script_dir/../.." && pwd -P)}"
cd -- "$project_dir"
project_dir="$(pwd -P)"
python_bin="${WTUMI_PYTHON:-$project_dir/.venv/bin/python}"
config_path="${WTUMI_CONFIG:-configs/scenes/static_door.yaml}"
if [[ ! -x "$python_bin" ]]; then
  printf 'Python executable not found: %s\nSet WTUMI_PYTHON to the Linux environment Python.\n' "$python_bin" >&2
  exit 2
fi
if [[ ! -f "$config_path" ]]; then
  printf 'Reconstruction config not found: %s\n' "$config_path" >&2
  exit 2
fi

# mktemp avoids collisions and never mixes this check with an old run.
output_root="${WTUMI_OUTPUT_ROOT:-$project_dir/outputs}"
mkdir -p -- "$output_root"
check_dir="$(mktemp -d "$output_root/pace_check_XXXXXXXX")"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$check_dir/matplotlib"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$check_dir/cache}"
export PYTHONDONTWRITEBYTECODE=1
printf 'Check output: %s\n' "$check_dir"
"$python_bin" - <<'PY' > "$check_dir/environment.json"
import importlib.metadata
import json
import platform
packages={}
for name in ('wtumi-recon','numpy','open3d','Pillow','PyYAML','matplotlib','plotly','mujoco'):
    try: packages[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: packages[name]=None
print(json.dumps({'python':platform.python_version(),'platform':platform.platform(),
                  'packages':packages},indent=2))
PY
"$python_bin" -m pytest -q -p no:cacheprovider 2>&1 | tee "$check_dir/tests.log"
"$python_bin" -m wtumi_recon validate --config "$config_path" 2>&1 | tee "$check_dir/validate.log"
"$python_bin" -m wtumi_recon run --config "$config_path" --output "$check_dir/reconstruction" 2>&1 | tee "$check_dir/reconstruction.log"
if [[ -n "${WTUMI_DEMO_CONFIG:-}" ]]; then
  "$python_bin" -m wtumi_recon demo --config "$WTUMI_DEMO_CONFIG" --output "$check_dir/sim_demo" 2>&1 | tee "$check_dir/sim_demo.log"
fi
printf 'Checks completed: %s\n' "$check_dir"
