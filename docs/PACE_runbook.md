# Develop on Mac, validate on PACE

The M5 room capture/reconstruction demo has completed on a PACE Linux compute node using Python 3.12.5, MuJoCo 3.17.0, Open3D 0.19.0 and EGL. The project provides portable check scripts and a Slurm submission template. Account, partition and QoS selections remain account-specific; actual logs from the target system determine whether a new run succeeds.

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

The confirmed Python module is `python/3.12.5`. From the project directory, load it and verify the interpreter before creating a new Linux environment:

```bash
cd "/your/remote/project"
module load python/3.12.5
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,sim]'
python -m pip check
```

`python3 --version` should show 3.12.5 before creating the environment. Loading a module does not upgrade an existing Python 3.9 virtual environment: deactivate and back up an old environment before recreating it. On later logins, load the same module and activate `.venv` without recreating it.

`pyproject.toml` declares cross-platform dependency ranges. The Mac `requirements.txt` records local versions and does not establish that every version has been validated on Linux. MuJoCo is installed through the `sim` extra. The M5 baseline has run on PACE, but the graphics backend and allocation still need checking in each new compute environment.

## 2. Run checks on an allocated compute node

### Request an interactive shell

After syncing from Mac, run this on a PACE login node from the project root:

```bash
bash scripts/pace/request_node.sh
```

The helper requests one node, one task, four CPU cores, 8G of total memory, and 30 minutes. These are starter settings for the small baseline, not measured minimum requirements. It uses `srun --pty` to start the shell on allocated compute resources. Requests may wait in the scheduler queue.

The shell prints its hostname and Slurm job ID, enters the project, and activates the project's `.venv` if present. It reads your `.bashrc` before activating the project environment and sets `OMP_NUM_THREADS` to the allocated CPU count. Load any required PACE software modules before requesting the shell. If `.venv` is absent, the shell still opens and prints setup guidance.

Within that shell, you can run:

```bash
python --version
bash scripts/pace/run_checks.sh
```

Use `exit` to end the interactive shell and release the allocation. `squeue -u "$USER"` lists your jobs. This helper refuses a new request when `SLURM_JOB_ID` is already set; use the current allocation or exit it first.

For fresh MuJoCo capture, request one GPU on a suitable partition:

```bash
bash scripts/pace/request_node.sh --gpu
```

This adds `--gres=gpu:1`; it does not select a GPU model or validate EGL. Once the allocated node's EGL support is confirmed, use the rendering commands below. Saved RGB-D replay and the default CPU tests do not require a GPU.

Resource and scheduler overrides:

```bash
bash scripts/pace/request_node.sh --cpus 8 --mem 16G --time 01:00:00
bash scripts/pace/request_node.sh \
  --account your_account --partition your_partition --qos your_qos
```

The second command contains placeholders: use only values authorized for your account. When omitted, Slurm's configured defaults apply. You may set `WTUMI_PACE_ACCOUNT`, `WTUMI_PACE_PARTITION`, and `WTUMI_PACE_QOS` in your PACE shell configuration instead of repeating these options. The helper does not guess cluster-specific values.

If Slurm reports an account, partition or QoS error, inspect your available configuration:

```bash
sinfo -o "%P %a %l %G"
sacctmgr show assoc where user="$USER" format=Account,Partition,QOS,DefaultQOS
```

A local preview works without Slurm and requests no resources:

```bash
bash scripts/pace/request_node.sh --dry-run
bash scripts/pace/request_node.sh --help
```

The preview uses the current machine's project path; the path is resolved again when the script runs on PACE. Argument handling and shell initialization were verified locally with a simulated `srun`. MuJoCo checks and the M5 demo have also run on an actual PACE compute node.

### Run the checks

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

`WTUMI_TEST_MUJOCO_RENDER=1` enables actual rendering tests for Z-depth, pixel centers, and camera coordinates. These tests are explicitly skipped when the variable is not set. EGL worked for the recorded PACE room demo, but availability still depends on node drivers and resource allocation. CPU-only/headless nodes may require a separate OSMesa setup. Mac uses the default CGL backend; do not set the Linux EGL environment variable there.

The analytic tilted-plane test disables multisampling to match production capture. Older source copies left the test at MuJoCo's default four samples and could report a millimetre-scale depth bias on EGL. Sync the corrected source rather than relaxing the geometric tolerance.

### Run the new staircase occlusion benchmark

After synchronizing source from Mac, request a GPU compute shell, load Python and activate the project environment. No additional dependencies beyond `.[test,sim]` are required. Check the new attached-camera renderer, then run the benchmark:

```bash
MUJOCO_GL=egl WTUMI_TEST_MUJOCO_RENDER=1 python -m pytest -q -rs tests/test_rig_capture.py
MUJOCO_GL=egl MPLBACKEND=Agg python scripts/run_occlusion_demo.py
```

Require the rendering tests to pass, not skip. The M6 staircase/box scene completed separately on PACE on 2026-11-16; the recorded directory is `outputs/occlusion_demo_20261116_184221_889717/`. Each new scene or graphics environment still needs its own checks. The output directory contains a README, three-condition comparisons, raw/masked capture playback, occupancy progression and an offline viewer. See [M6](m6_stairs_occlusion.md) for definitions and replay. Use `exit` after the run to release an interactive allocation.

### Run observation-memory playback (M7)

After source synchronization, an allocated compute node can reuse the saved paired M6 capture without invoking the simulator:

```bash
MPLBACKEND=Agg python scripts/run_memory_demo.py \
  --capture outputs/occlusion_demo_20261116_184221_889717/capture
```

For fresh capture, omit `--capture` and add `MUJOCO_GL=egl` on a GPU node. M7 has its own validation scope; successful M6 execution does not validate M7. Download the entire output directory and open `index.html` for both final viewers, the temporal 3D viewer and all animations. See [M7](m7_visual_memory.md).

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
