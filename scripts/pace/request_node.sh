#!/usr/bin/env bash
# Run on a PACE login node. Preview with --dry-run on any machine.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/pace/request_node.sh [options]

Request an interactive Slurm allocation and open a shell on a compute node.
Default: one node, one task, 4 CPU cores, 8G memory, 00:30:00, no GPU.

  --cpus N          CPU cores for the single task (default: 4).
  --mem SIZE        Total memory, e.g. 8G or 16000M (default: 8G).
  --time HH:MM:SS   Time limit (default: 00:30:00).
  --gpu             Also request one GPU using --gres=gpu:1.
  --account NAME    Slurm account; otherwise use the account default.
  --partition NAME Slurm partition; otherwise use the cluster default.
  --qos NAME        Slurm quality of service; otherwise use the account default.
  --dry-run         Print the command without requesting any resources.
  --help            Show this help.

WTUMI_PACE_ACCOUNT, WTUMI_PACE_PARTITION and WTUMI_PACE_QOS can supply defaults.
WTUMI_WORKDIR can override the project directory. No site-specific account,
partition or QoS is assumed. GPU availability and EGL support depend on the node.

The compute shell enters the project and activates .venv when it exists.
Load required PACE software modules before launching this helper.
Type exit to end the shell and release this allocation. Use squeue -u "$USER"
to inspect your jobs. This script does not install dependencies or run experiments.
EOF
}

cpus=4
memory=8G
time_limit=00:30:00
gpu=false
dry_run=false
account="${WTUMI_PACE_ACCOUNT:-}"
partition="${WTUMI_PACE_PARTITION:-}"
qos="${WTUMI_PACE_QOS:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    --help|-h) usage; exit 0 ;;
    --cpus|--mem|--time|--account|--partition|--qos)
      option="$1"
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        printf 'Missing value for %s\n' "$option" >&2
        exit 2
      fi
      case "$option" in
        --cpus) cpus="$2" ;;
        --mem) memory="$2" ;;
        --time) time_limit="$2" ;;
        --account) account="$2" ;;
        --partition) partition="$2" ;;
        --qos) qos="$2" ;;
      esac
      shift 2 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! "$cpus" =~ ^[1-9][0-9]*$ ]]; then
  printf 'CPU count must be a positive integer.\n' >&2
  exit 2
fi
if [[ ! "$memory" =~ ^[1-9][0-9]*[KkMmGgTt]?$ ]]; then
  printf 'Memory must be positive, e.g. 8G or 16000M. Bare numbers are MiB.\n' >&2
  exit 2
fi
if [[ ! "$time_limit" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ || ! "$time_limit" =~ [1-9] ]]; then
  printf 'Time limit must be positive and use HH:MM:SS, e.g. 00:30:00.\n' >&2
  exit 2
fi
for value in "$account" "$partition" "$qos"; do
  if [[ -n "$value" && ! "$value" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ ]]; then
    printf 'Account, partition and QoS must each be a single name without spaces or shell characters.\n' >&2
    exit 2
  fi
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_dir="${WTUMI_WORKDIR:-$(cd -- "$script_dir/../.." && pwd -P)}"
project_dir="$(cd -- "$project_dir" && pwd -P)"
shell_rc="$project_dir/scripts/pace/compute_shell.sh"
if [[ ! -r "$shell_rc" || ! -f "$project_dir/pyproject.toml" ]]; then
  printf 'Project files are missing under %s. Sync the complete project first.\n' "$project_dir" >&2
  exit 2
fi

args=(--job-name=wtumi-interactive --nodes=1 --ntasks=1
      "--cpus-per-task=$cpus" "--mem=$memory" "--time=$time_limit"
      "--chdir=$project_dir")
if [[ -n "$account" ]]; then args+=("--account=$account"); fi
if [[ -n "$partition" ]]; then args+=("--partition=$partition"); fi
if [[ -n "$qos" ]]; then args+=("--qos=$qos"); fi
if "$gpu"; then args+=(--gres=gpu:1); fi
# srun starts the shell on allocated compute resources, not on the login node.
args+=(--pty bash --rcfile "$shell_rc" -i)
if "$dry_run"; then
  printf 'Preview only; no Slurm request will be submitted:\n'
  printf '%q ' srun "${args[@]}"
  printf '\n'
  exit 0
fi
if [[ -n "${SLURM_JOB_ID:-${SLURM_JOBID:-}}" ]]; then
  printf 'Already inside Slurm job %s. Use the current allocation or exit it before requesting a new one.\n' "${SLURM_JOB_ID:-${SLURM_JOBID:-}}" >&2
  exit 2
fi
if ! command -v srun >/dev/null 2>&1; then
  printf 'Slurm srun is unavailable. Run this helper on PACE, or use --dry-run for a local preview.\n' >&2
  exit 2
fi
printf 'Requesting 1 node, 1 task, %s CPU cores, %s memory, %s time limit' "$cpus" "$memory" "$time_limit"
if "$gpu"; then printf ', 1 GPU'; fi
printf '.\nWaiting for Slurm to allocate resources. A compute-node shell will open when ready.\n'
exec srun "${args[@]}"
