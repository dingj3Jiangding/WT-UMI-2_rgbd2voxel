#!/usr/bin/env bash
# Run inside a PACE compute-node shell; --dry-run is safe on the Mac.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/pace/run_scene_suite.sh [--output PATH] [--dry-run]
       bash scripts/pace/run_scene_suite.sh --output PATH --replace-scenes SCENE [SCENE ...]

Run room, doorway and G1 stairs with the pace30 preset: fresh 640x480 RGB-D
observations and synchronized 30-fps camera/voxel MP4 videos.

  --output PATH          Result directory, relative to the project or absolute.
                         Default: outputs/g1_vision_30fps_pace_<timestamp>_<process ID>
  --replace-scenes ...   Rerun selected scenes inside an existing suite result.
                         Choices: room doorway g1_stairs. Other scenes are preserved.
                         Requires --output or WTUMI_SUITE_OUTPUT and suite.json.
  --dry-run              Print the command without running Python or creating files.
  --help                 Show this help.

Run after request_node.sh has opened a compute-node shell, or through srun
inside your allocation. A real run requires SLURM_JOB_ID and SLURM_STEP_ID.
Suggested allocation: 1 GPU, 8 CPUs, 32G memory and a 2-hour time limit.
These are resource requests, not measured runtime or memory guarantees.

WTUMI_WORKDIR overrides the project root; WTUMI_PYTHON selects an executable
Python path (default: .venv/bin/python). WTUMI_SUITE_OUTPUT can supply --output.
Install the project's sim and video extras before running this helper.
This script does not request nodes, install packages, sync files or use SSH.
Logs and caches are kept in outputs/logs, outside the experiment directory.
EOF
}

output="${WTUMI_SUITE_OUTPUT:-}"
dry_run=false
replace_scenes=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        printf 'Missing value for --output.\n' >&2
        exit 2
      fi
      output="$2"
      shift 2
      ;;
    --replace-scenes)
      shift
      previous_count=${#replace_scenes[@]}
      while [[ $# -gt 0 && "$1" != --* ]]; do
        case "$1" in
          room|doorway|g1_stairs) ;;
          *) printf 'Unknown replacement scene: %s. Choose room, doorway or g1_stairs.\n' "$1" >&2; exit 2 ;;
        esac
        if [[ ${#replace_scenes[@]} -gt 0 ]]; then
          for selected_scene in "${replace_scenes[@]}"; do
            if [[ "$selected_scene" == "$1" ]]; then
              printf 'Duplicate replacement scene: %s.\n' "$1" >&2
              exit 2
            fi
          done
        fi
        replace_scenes+=("$1")
        shift
      done
      if [[ ${#replace_scenes[@]} -eq "$previous_count" ]]; then
        printf 'Supply at least one scene after --replace-scenes.\n' >&2
        exit 2
      fi
      ;;
    --dry-run) dry_run=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ${#replace_scenes[@]} -gt 0 && -z "$output" ]]; then
  printf 'Scene replacement requires --output or WTUMI_SUITE_OUTPUT naming an existing suite.\n' >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_dir="${WTUMI_WORKDIR:-$(cd -- "$script_dir/../.." && pwd -P)}"
cd -- "$project_dir"
project_dir="$(pwd -P)"
python_bin="${WTUMI_PYTHON:-$project_dir/.venv/bin/python}"
if [[ ! -f "$project_dir/pyproject.toml" || ! -f "$project_dir/scripts/run_scene_suite.py" ]]; then
  printf 'Project files are missing under %s. Sync the current project first.\n' "$project_dir" >&2
  exit 2
fi
if [[ -z "$output" ]]; then
  output="outputs/g1_vision_30fps_pace_$(date +%Y%m%d_%H%M%S)_$$"
fi
if [[ "$output" != /* ]]; then
  output="$project_dir/$output"
fi
while [[ "$output" != / && "$output" == */ ]]; do output="${output%/}"; done
if [[ ${#replace_scenes[@]} -gt 0 ]]; then
  if [[ -L "$output" || ! -d "$output" || ! -f "$output/suite.json" || -L "$output/suite.json" ]]; then
    printf 'Scene replacement requires an existing suite directory with suite.json, without symlinks: %s\n' "$output" >&2
    exit 2
  fi
elif [[ -e "$output" || -L "$output" ]]; then
  printf 'Result path already exists: %s\nChoose a new directory; no files were changed.\n' "$output" >&2
  exit 2
fi
cpus="${SLURM_CPUS_PER_TASK:-8}"
if [[ ! "$cpus" =~ ^[1-9][0-9]*$ ]]; then
  printf 'SLURM_CPUS_PER_TASK must be a positive integer.\n' >&2
  exit 2
fi
args=("$project_dir/scripts/run_scene_suite.py" --preset pace30 --output "$output")
if [[ ${#replace_scenes[@]} -gt 0 ]]; then
  args+=(--replace-scenes "${replace_scenes[@]}")
fi
if "$dry_run"; then
  printf 'Preview only; no Python process, experiment or log directory will be created.\n'
  printf 'Result directory: %s\n' "$output"
  printf '%q ' env MUJOCO_GL=egl MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 "OMP_NUM_THREADS=$cpus" OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "$python_bin" "${args[@]}"
  printf '\n'
  exit 0
fi
if [[ -z "${SLURM_JOB_ID:-${SLURM_JOBID:-}}" || -z "${SLURM_STEP_ID:-}" ]]; then
  printf 'Run inside an allocated Slurm compute-node shell, not a login-node shell.\n' >&2
  printf 'Use request_node.sh, or invoke this helper through srun within your allocation.\n' >&2
  exit 2
fi
if [[ ! -x "$python_bin" ]]; then
  printf 'Python executable not found: %s\nSet WTUMI_PYTHON to your prepared Linux Python executable.\n' "$python_bin" >&2
  exit 2
fi

logs_root="$project_dir/outputs/logs"
mkdir -p -- "$logs_root"
log_dir="$(mktemp -d "$logs_root/scene_suite_XXXXXXXX")"
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$cpus" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$log_dir/matplotlib"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$log_dir/cache}"
printf 'Result directory: %s\n' "$output"
printf 'Result folder name: %s\n' "${output##*/}"
printf 'Log file: %s/run.log\n' "$log_dir"
"$python_bin" "${args[@]}" 2>&1 | tee "$log_dir/run.log"
printf 'Completed result: %s\n' "$output"
printf 'Gallery: %s/index.html\n' "$output"
