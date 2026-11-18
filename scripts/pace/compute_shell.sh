# Bash rcfile for request_node.sh; sourced inside the compute-node shell.
# Keep this interactive: do not enable errexit, nounset or pipefail here.
if [[ -r "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

wtumi_session_dir="$(builtin cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
if ! builtin cd -- "$wtumi_session_dir"; then
  printf 'Could not enter project directory: %s\n' "$wtumi_session_dir" >&2
  return 1
fi
printf '\nCompute host: %s | Slurm job: %s\n' "$(hostname)" "${SLURM_JOB_ID:-unknown}"
printf 'Project: %s\n' "$PWD"
if [[ -r .venv/bin/activate ]]; then
  if source .venv/bin/activate; then
    printf 'Activated project .venv. Use python to run the project.\n'
  else
    printf 'Project .venv activation failed. Check your Linux environment before running Python.\n' >&2
  fi
else
  printf 'No project .venv found. Prepare the Linux environment using docs/PACE_runbook.md.\n'
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
printf 'Run bash scripts/pace/run_checks.sh when ready. Type exit to release this allocation.\n\n'
unset wtumi_session_dir
