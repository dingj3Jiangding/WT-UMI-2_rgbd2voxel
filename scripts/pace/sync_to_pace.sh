#!/usr/bin/env bash
# Run on Mac to upload source updates. This script does not submit compute jobs.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/pace/sync_to_pace.sh [--dry-run] [--with-replay]
       [--host SSH_HOST] [--remote-dir PATH]

Default host: login-ice.pace.gatech.edu (uses your SSH configuration)
Default destination: projects/WT-UMI-2_rgbd2voxel (relative to remote home)

--dry-run       Show proposed changes without writing remote files/directories.
--with-replay   Also upload the small saved M5 RGB-D fixture and Mac baseline.
--host         SSH alias or user@host; authentication is handled by SSH.
--remote-dir   Relative-to-home or absolute path without spaces/shell characters.
--help         Show this help.

Environment defaults: WTUMI_PACE_HOST, WTUMI_PACE_DIR, WTUMI_WORKDIR.
Source folders and selected root files are uploaded. Virtual environments,
Git metadata, caches and other outputs are not included. Same-name remote
source files are updated; remote-only files are never deleted. After deleting
or renaming local source files, remove obsolete remote copies separately.
EOF
}

host="${WTUMI_PACE_HOST:-login-ice.pace.gatech.edu}"
remote_dir="${WTUMI_PACE_DIR:-projects/WT-UMI-2_rgbd2voxel}"
dry_run=false
with_replay=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=true; shift ;;
    --with-replay) with_replay=true; shift ;;
    --host|--remote-dir)
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        printf 'Missing value for %s\n' "$1" >&2
        exit 2
      fi
      if [[ "$1" == --host ]]; then host="$2"; else remote_dir="$2"; fi
      shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

# Limit values that will be interpreted by the remote login shell.
if [[ ! "$host" =~ ^[A-Za-z0-9_][A-Za-z0-9_.@-]*$ ]]; then
  printf 'Use an SSH alias or user@host without spaces or shell characters.\n' >&2
  exit 2
fi
if [[ ! "$remote_dir" =~ ^[A-Za-z0-9_./-]+$ || "$remote_dir" == -* ]]; then
  printf 'Remote directory must be a plain relative or absolute path; use projects/name, not ~/projects/name.\n' >&2
  exit 2
fi
if [[ "$remote_dir" =~ (^|/)\.\.?(/|$) || -z "${remote_dir//\//}" ]]; then
  printf 'Choose a project directory, without dot or parent-directory components.\n' >&2
  exit 2
fi
remote_dir="${remote_dir%/}"
command -v rsync >/dev/null 2>&1 || { printf 'rsync is required on this machine.\n' >&2; exit 2; }
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_dir="${WTUMI_WORKDIR:-$(cd -- "$script_dir/../.." && pwd -P)}"
cd -- "$project_dir"

# Relative arguments avoid the incompatible /./ cut-point in macOS openrsync.
sources=(README.md pyproject.toml requirements.txt requirements-mujoco.txt .gitignore
         src/wtumi_recon scripts tests configs assets docs scenes)
if "$with_replay"; then
  fixture=outputs/m5_mujoco_demo
  sources+=("$fixture/reconstruction.yaml" "$fixture/experiment.yaml"
            "$fixture/capture/manifest.json" "$fixture/capture/frames"
            "$fixture/demo.json" "$fixture/experiment/summary.json")
fi
for source in "${sources[@]}"; do
  if [[ ! -e "$source" ]]; then
    printf 'Missing source: %s/%s\n' "$PWD" "$source" >&2
    exit 2
  fi
done

options=(-azcR --itemize-changes
         --exclude=.DS_Store --exclude=__pycache__/ --exclude='*.py[cod]'
         --exclude='.venv/' --exclude='.git/' --exclude='.pytest_cache/'
         --exclude='*.egg-info/' --exclude=MUJOCO_LOG.TXT)
if "$dry_run"; then
  options+=(--dry-run)
  remote_command=rsync
  printf 'Preview only: %s -> %s:%s/\n' "$PWD" "$host" "$remote_dir"
else
  # Create the destination using the same SSH session used by rsync.
  remote_command="mkdir -p '$remote_dir' && exec rsync"
  printf 'Uploading: %s -> %s:%s/\n' "$PWD" "$host" "$remote_dir"
fi
exec rsync "${options[@]}" --rsync-path="$remote_command" -- "${sources[@]}" "$host:$remote_dir/"
