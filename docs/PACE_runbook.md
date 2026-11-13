# Develop on Mac, validate on PACE

The project provides portable check scripts and a Slurm submission template. PACE has not been connected or tested; the account, partition, scheduler, software modules, GPU, and driver setup have not been confirmed. Actual logs from the target system determine whether a run succeeds.

## 1. Transfer source and data, then create a Linux environment

Copy the project source, configurations, and RGB-D data to your PACE working directory. Do not copy the Mac `.venv`. Avoid Mac absolute paths in remote configurations; image paths in sequence manifests should be relative to the dataset directory.

### Repeated uploads with rsync

Use `scripts/pace/sync_to_pace.sh` on Mac for the initial upload and later source updates. It uses the SSH configuration for `login-ice.pace.gatech.edu` and defaults to `~/projects/WT-UMI-2_rgbd2voxel` on PACE. Password or Duo prompts, if required, are handled by SSH in your terminal. Both machines need rsync.

From the Mac project root, preview and perform the first upload:

```bash
bash scripts/pace/sync_to_pace.sh --dry-run --with-replay
bash scripts/pace/sync_to_pace.sh --with-replay
```

The destination is created during an actual upload; dry-run does not create remote directories. Source sync includes the package, scripts, tests, configurations, assets, documentation and selected root files. `--with-replay` additionally includes the small M5 capture, its portable entry configs and the Mac baseline metrics. It excludes full historical meshes/viewers. Mac virtual environments, Git metadata, caches and other outputs are excluded.

After editing code on Mac, run:

```bash
bash scripts/pace/sync_to_pace.sh
```

Checksums determine which file contents changed, so an unchanged project does not need to be uploaded again. This is one-way sync: same-name remote source files are updated. Keep Mac as the source copy; edits made directly to those files on PACE can be overwritten. Remote-only files are not deleted. If you delete or rename local source files, clean up obsolete remote copies separately.

To choose a different account or directory:

```bash
bash scripts/pace/sync_to_pace.sh \
  --host your_username@login-ice.pace.gatech.edu \
  --remote-dir projects/WT-UMI-2_rgbd2voxel
```

A relative remote directory is relative to the remote home directory. An absolute path can select your assigned storage location. Paths must not contain spaces, shell characters or dot/parent-directory components; use `projects/name` rather than `~/projects/name`. `WTUMI_PACE_HOST` and `WTUMI_PACE_DIR` can set persistent defaults in your local shell configuration. `--help` lists all options.

The Linux environment is created once. With the editable install below, ordinary Python source updates take effect the next time you run the program; rerun `pip install -e '.[test,sim]'` if dependency declarations change. Source sync does not start or restart a PACE job.

### Create the Linux environment

Prepare Python 3.12 using your team's PACE setup, then run:

```bash
cd "/your/remote/project"
python3.12 -m venv .venv
./.venv/bin/python -m pip install -e '.[test,sim]'
./.venv/bin/python -m pip check
```

`pyproject.toml` declares cross-platform dependency ranges. The Mac `requirements.txt` records local versions and does not establish that every version has been validated on Linux. MuJoCo is installed through the `sim` extra. The first example has been validated on Mac; the Linux graphics backend, GPU drivers, and cluster environment still need testing. Run the core pipeline before enabling actual simulator capture.

## 2. Run checks on an allocated compute node

The following works locally or on a compute node with allocated resources:

```bash
export WTUMI_WORKDIR="/your/remote/project"
export WTUMI_PYTHON="$WTUMI_WORKDIR/.venv/bin/python"
bash scripts/pace/run_checks.sh
```

The script records environment versions, runs tests, validates the configuration, and reconstructs the existing static door scene in that order. Matplotlib uses Agg without opening a window. Outputs go to a new `outputs/pace_check_<random_suffix>/` directory, preserving earlier results.

Optional environment variables:

| Variable | Purpose |
|---|---|
| `WTUMI_WORKDIR` | Project path; defaults to a path derived from the script location |
| `WTUMI_PYTHON` | Prepared Python executable; an absolute path is recommended |
| `WTUMI_CONFIG` | Configuration to validate and reconstruct; defaults to `configs/scenes/static_door.yaml`. Relative paths resolve from the project root |
| `WTUMI_OUTPUT_ROOT` | Parent directory for check outputs; defaults to the project's `outputs/` |
| `WTUMI_DEMO_CONFIG` | Optional complete simulator demo configuration, such as `configs/demos/mujoco_room.yaml` |

To test an existing RGB-D capture, prepare a configuration referencing the sequence and point `WTUMI_CONFIG` to it. This script does not install dependencies, access remote services, or submit jobs. If `WTUMI_DEMO_CONFIG` is set, it runs MuJoCo capture, the resolution experiment, viewer export, and error plotting after the checks.

Run reconstruction and simulation on allocated compute resources, following the instructions for your PACE system.

### Enable MuJoCo offscreen rendering checks

On an allocated Linux GPU node with confirmed EGL support, you can use:

```bash
export MUJOCO_GL=egl
export WTUMI_TEST_MUJOCO_RENDER=1
export WTUMI_DEMO_CONFIG=configs/demos/mujoco_room.yaml
bash scripts/pace/run_checks.sh
```

`WTUMI_TEST_MUJOCO_RENDER=1` enables actual rendering tests for Z-depth, pixel centers, and camera coordinates. These tests are explicitly skipped when the variable is not set. EGL availability depends on node drivers and resource allocation; it has not been validated on PACE for this project. CPU-only/headless nodes may require a separate OSMesa setup. Mac uses the default CGL backend; do not set the Linux EGL environment variable there.

You can also run the complete demo on its own:

```bash
./.venv/bin/python scripts/run_sim_demo.py
```

The new output directory's `README.md` links to comparison images, error curves, and the 3D viewer. To reconstruct data already captured on Mac, copy the entire demo output directory and use its `reconstruction.yaml` or `experiment.yaml`; both use relative paths. Internal `config_resolved.yaml` and `source_manifest.json` files record provenance and may contain paths from the original machine, so they are not entry points for cross-host replay.

## 3. If the system uses Slurm

`submit_checks.sh` is an `sbatch` template. Confirm that your system uses Slurm before using it. The script does not choose an allocation, partition, time limit, memory request, or CPU/GPU request.

```bash
bash scripts/pace/submit_checks.sh --help
```

After setting `WTUMI_WORKDIR` and `WTUMI_PYTHON` as above, pass your confirmed resource options to the script. Replace every placeholder in this example:

```bash
bash scripts/pace/submit_checks.sh \
  --account="your_allocation" \
  --partition="your_partition" \
  --time="approved_time_limit" \
  --mem="appropriate_memory_request" \
  --cpus-per-task="allocated_cpu_count"
```

Arguments are forwarded unchanged to `sbatch`; the script adds only the project working directory and the check script to execute. Initialize software modules or the environment before submission, and confirm that the job can use them under the cluster's environment export rules. The template has not submitted an actual PACE job.

## 4. Save and compare results

Each check directory contains:

- `environment.json`: Python, operating system, and main dependency versions.
- `tests.log` / `validate.log` / `reconstruction.log`: logs for each step; any failure stops execution.
- `reconstruction/metrics.json`: novel-view metrics and the actual fusion frame list.
- `reconstruction/renders/<view>/comparison.png`: reference and reconstructed RGB-D comparisons.
- Optional `sim_demo/` and `sim_demo.log`: one capture, multiple reconstruction resolutions, novel-view comparisons, and error curves.

For the first migration, confirm that tests pass, held-out images are excluded from fusion, and image/depth coordinates are correct before running simulator capture. Copy images and HTML results back to Mac for inspection. Floating-point results may vary slightly across operating systems or dependency versions; assess images, coverage, and errors together rather than requiring byte-for-byte file equality as a general rule.

Generate error curves from an existing experiment summary:

```bash
./.venv/bin/python -m wtumi_recon plot --summary outputs/your_experiment/summary.json
```

The command produces PNG and SVG plots with the same base name. The horizontal axis is physical voxel edge length. MAE/RMSE use only valid depth overlap; coverage uses all valid reference depth pixels as its denominator. Lines show descriptive means of available repeats, and faint points show individual runs. Missing metrics remain gaps and do not represent zero error.
