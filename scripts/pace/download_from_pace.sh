#!/usr/bin/env bash
# Run on the Mac to download a complete PACE experiment result.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/pace/download_from_pace.sh RUN_NAME [options]
       bash scripts/pace/download_from_pace.sh --list [options]

Download remote PROJECT/outputs/RUN_NAME/ into local OUTPUT_ROOT/RUN_NAME/.
Example:
  bash scripts/pace/download_from_pace.sh g1_vision_30fps_pace

  --host HOST        SSH alias or user@host.
                     Default: djiang94@login-ice.pace.gatech.edu
  --remote-dir PATH  Remote project directory, relative to home or absolute.
                     Default: projects/WT-UMI-2_rgbd2voxel
  --local-root PATH  Local result parent directory; RUN_NAME is appended.
                     Default: this project's outputs directory.
                     Relative paths are resolved from the project root.
  --list             List remote output entries instead of downloading.
  --no-progress      Disable download progress output.
  --dry-run          Print the resolved command; no connection or file changes.
  --help             Show this help.

Environment defaults: WTUMI_PACE_HOST, WTUMI_PACE_DIR, WTUMI_DOWNLOAD_ROOT,
WTUMI_WORKDIR. SSH handles authentication; no password is saved in this script.
Run names are single folder names, without outputs/ or a trailing slash.
The destination is created automatically. Repeat the command to update changed
files or retry an interrupted transfer. Same-name files use the PACE version;
local-only files are preserved. No remote files are modified or deleted.
No Python environment, Slurm allocation or compressed archive is required.
Progress is enabled by default. A terminal shows a bar, percentage and speed.
Newer rsync versions show overall progress; older Mac versions show progress
for the current file, so 100% can appear before the full download finishes.
Redirected output keeps native rsync progress without a graphical bar.
EOF
}

host="${WTUMI_PACE_HOST:-djiang94@login-ice.pace.gatech.edu}"
remote_dir="${WTUMI_PACE_DIR:-projects/WT-UMI-2_rgbd2voxel}"
local_root="${WTUMI_DOWNLOAD_ROOT:-}"
run_name=""
list=false
dry_run=false
progress=true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host|--remote-dir|--local-root)
      option="$1"
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        printf 'Missing value for %s.\n' "$option" >&2
        exit 2
      fi
      case "$option" in
        --host) host="$2" ;;
        --remote-dir) remote_dir="$2" ;;
        --local-root) local_root="$2" ;;
      esac
      shift 2
      ;;
    --list) list=true; shift ;;
    --no-progress) progress=false; shift ;;
    --dry-run) dry_run=true; shift ;;
    --help|-h) usage; exit 0 ;;
    -*) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    *)
      if [[ -n "$run_name" ]]; then
        printf 'Supply one run folder name per invocation.\n' >&2
        exit 2
      fi
      run_name="$1"
      shift
      ;;
  esac
done

if "$list"; then
  if [[ -n "$run_name" ]]; then
    printf 'Use either a run folder name or --list.\n' >&2
    exit 2
  fi
elif [[ ! "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  printf 'Supply the exact run folder name, e.g. g1_vision_30fps_pace.\n' >&2
  printf 'Use --list to inspect remote outputs.\n' >&2
  exit 2
fi
if [[ ! "$host" =~ ^[A-Za-z0-9_][A-Za-z0-9_.@-]*$ ]]; then
  printf 'Use an SSH alias or user@host without spaces or shell characters.\n' >&2
  exit 2
fi
if [[ ! "$remote_dir" =~ ^[A-Za-z0-9_./-]+$ || "$remote_dir" == -* ||
      "$remote_dir" =~ (^|/)\.\.?(/|$) || -z "${remote_dir//\//}" ]]; then
  printf 'Remote project path must contain plain path components, without dot/parent components or shell characters.\n' >&2
  exit 2
fi
while [[ "$remote_dir" == */ ]]; do remote_dir="${remote_dir%/}"; done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_dir="${WTUMI_WORKDIR:-$(cd -- "$script_dir/../.." && pwd -P)}"
project_dir="$(cd -- "$project_dir" && pwd -P)"
if [[ ! -f "$project_dir/pyproject.toml" ]]; then
  printf 'Project root not found: %s\nSet WTUMI_WORKDIR to the local project directory.\n' "$project_dir" >&2
  exit 2
fi

remote_outputs="$host:$remote_dir/outputs/"
if "$list"; then
  args=(--list-only "$remote_outputs")
else
  local_root="${local_root:-$project_dir/outputs}"
  if [[ "$local_root" != /* ]]; then local_root="$project_dir/$local_root"; fi
  local_dir="${local_root%/}/$run_name"
  if [[ -L "$local_dir" || ( -e "$local_dir" && ! -d "$local_dir" ) ]]; then
    printf 'Local result path must be a directory, not a file or symlink: %s\n' "$local_dir" >&2
    exit 2
  fi
  progress_mode=none
  progress_args=(-v)
  if "$progress"; then
    progress_mode=file
    progress_args=(-v --progress)
    # Probe local option support only; this does not connect to PACE.
    if command -v rsync >/dev/null 2>&1 &&
       rsync --info=progress2 --no-inc-recursive --version >/dev/null 2>&1; then
      progress_mode=overall
      progress_args=(--info=progress2 --no-inc-recursive)
    fi
  fi
  args=(-az "${progress_args[@]}" --partial-dir=.rsync-partial --safe-links
        "$remote_outputs$run_name/" "$local_dir/")
fi

if "$dry_run"; then
  printf 'Preview only; no SSH connection or local directories will be created.\n'
  printf '%q ' rsync "${args[@]}"
  printf '\n'
  exit 0
fi
command -v rsync >/dev/null 2>&1 || { printf 'rsync is required on the local machine.\n' >&2; exit 2; }
if "$list"; then
  printf 'Remote output entries: %s\n' "$remote_outputs"
  exec rsync "${args[@]}"
fi

mkdir -p -- "$local_dir"
printf 'Downloading: %s%s/\nLocal result: %s/\n' "$remote_outputs" "$run_name" "$local_dir"
case "$progress_mode" in
  overall)
    progress_label='Overall'
    printf 'Progress: overall file data. The initial file-list scan may take a moment.\n'
    ;;
  file)
    progress_label='Current file'
    printf 'Progress: current file. A file reaching 100%% does not finish the full download.\n'
    ;;
esac

download() {
  if "$progress" && [[ -t 1 ]] && command -v awk >/dev/null 2>&1; then
    # Keep SSH prompts/errors on stderr and preserve rsync's status through the pipe.
    if rsync "${args[@]}" | awk -v label="$progress_label" -f "$script_dir/rsync_progress.awk"; then
      return 0
    else
      local statuses=("${PIPESTATUS[@]}")
      if [[ "${statuses[0]}" -ne 0 ]]; then return "${statuses[0]}"; fi
      return "${statuses[1]}"
    fi
  else
    rsync "${args[@]}"
  fi
}

if download; then
  printf 'Download completed: %s\n' "$local_dir"
  if [[ -f "$local_dir/index.html" ]]; then
    printf 'Open the result gallery on the Mac:\n'
    printf '%q ' open "$local_dir/index.html"
    printf '\n'
  fi
else
  status=$?
  printf 'Download did not complete (exit %s). Check the run name with --list and retry the same command.\n' "$status" >&2
  exit "$status"
fi
