# Three-scene G1 suite at 30 fps on PACE

The `pace30` preset runs `room`, `doorway` and `g1_stairs` with 640 x 480 RGB-D capture. It samples the existing four-second trajectory every 1/30 second, including both endpoints: 121 integration observations per scene, plus the same two held-out evaluation views. MuJoCo renders fresh RGB, depth and segmentation at each sampled pose, and each observation is integrated once. Only camera/rig waypoints are interpolated; images and maps are not interpolated.

Each `camera_voxel.mp4` contains all 121 synchronized frames at 30 fps, lasting about 4.033 seconds including the final frame. The left panel is recorded head-camera RGB. The right panel is the corresponding cumulative voxel map rendered in MuJoCo from a fixed overview. The canvas is 1280 x 622 pixels including labels. The GIF samples at most 60 frames for a compact preview; use MP4 to assess 30 fps playback.

Robot geometry, door angle, camera mount and voxel sizes are retained. Room and doorway still use 40.625 mm voxels; stairs uses 21.875 mm voxels. Occupancy sampling stride increases from 4 to 8 for the doubled camera width. This preset changes the observation budget relative to the 13-frame baseline, so differences in mapping error are not caused by playback rate alone. Prescribed G1 motion remains a fixed-pose replay, without locomotion or physical manipulation.

The current room and doorway templates use a carried box and G1 carry pose, as does the stairs template. The box is visible in recorded camera RGB but excluded from static environment fusion. Changing a template does not change old captures or maps; use the targeted rerun below to update an existing suite.

## 1. Sync from the Mac

Run from the local project directory:

```bash
bash scripts/pace/sync_to_pace.sh \
  --host djiang94@login-ice.pace.gatech.edu
```

The existing sync includes source, scene configurations and G1 assets, while excluding virtual environments and experiment results.

## 2. Request a PACE compute-node shell

On PACE, using the working software modules and Python environment from earlier experiments:

```bash
cd ~/projects/WT-UMI-2_rgbd2voxel
bash scripts/pace/request_node.sh \
  --gpu --cpus 8 --mem 32G --time 02:00:00
```

Wait for the compute-node shell. The helper enters the project and activates its existing `.venv`. If your allocation requires account, partition or QoS options, supply your assigned values. The two-hour value is the requested time limit, not a runtime estimate. MuJoCo EGL rendering uses the GPU; the current Open3D integration also needs CPU and memory resources.

## 3. Prepare the environment and inspect the plan

Inside the allocated shell:

```bash
source .venv/bin/activate
python -m pip install -e '.[test,sim,video]'
python -m pip check
python scripts/run_scene_suite.py --preset pace30 --dry-run
```

The plan should show `integration_frames: 121`, `playback_fps: 30.0`, `camera_width: 640`, `camera_height: 480`, `box_enabled: true` and `robot_pose: carry` for all three current templates. The plan does not render, encode or create an output directory. The actual preset checks the H.264 encoder before starting capture; MP4 is required for this preset. The `video` extra supplies FFmpeg through `imageio-ffmpeg`.

For an optional short EGL check:

```bash
MUJOCO_GL=egl WTUMI_TEST_MUJOCO_RENDER=1 \
  python -m pytest -q tests/test_scene_preset_render.py
```

This test uses the first four resampled poses per scene at 640 x 480, a reduced reconstruction grid, and decodes the resulting MP4 to verify 30 fps. It is not the full experiment.

## 4. Run all three scenes

In the same compute-node shell:

```bash
bash scripts/pace/run_scene_suite.sh \
  --output outputs/g1_vision_30fps_pace
```

Use a new output name if that directory already exists, or explicitly select scenes with `--replace-scenes` as described below. The runner refuses to overwrite results by default. Without `--output`, it generates a unique name and prints the exact result directory and folder name; copy that printed value when downloading.

The runner selects EGL, noninteractive plotting and thread counts from the allocation. It records a log under `outputs/logs/scene_suite_*/run.log`, outside the result directory. It requires a Slurm compute step, as opened by `request_node.sh`; an allocation ID alone does not establish that a login-node shell is a compute shell. If using an existing `salloc` allocation, launch the command with `srun`.

Equivalent direct command on an allocated compute node:

```bash
MUJOCO_GL=egl MPLBACKEND=Agg \
  python scripts/run_scene_suite.py --preset pace30 \
  --output outputs/g1_vision_30fps_pace
```

Run only one of these alternatives for a given output directory. Each scene saves all 121 prefix maps, so the complete result is larger than the earlier 13-frame experiment. Capture, reconstruction, evaluation and encoding are offline work; 30 fps describes the output video, not measured processing throughput. Native interactive playback can be slower depending on graphics and disk performance.

## 5. Download on the Mac

From the local project directory, using the same output name as the successful PACE run:

```bash
bash scripts/pace/download_from_pace.sh g1_vision_30fps_pace
open outputs/g1_vision_30fps_pace/index.html
```

The downloader creates the destination automatically. Run the same command again to transfer changed files or retry an interrupted download. Local-only files are preserved; same-name files use the PACE version. No Python environment is needed.

Progress is enabled automatically. In a terminal, the downloader shows a bar, percentage, transfer speed and rsync's time field. With newer rsync versions, `Overall` tracks the full file list after an initial scan. With older Mac versions, `Current file` tracks each file separately; reaching 100% only finishes that file. Wait for `Download completed` to confirm the full transfer. Redirected output retains native rsync progress. Add `--no-progress` to disable it.

Use `bash scripts/pace/download_from_pace.sh --list` to check the exact remote result names. Add `--dry-run` to preview a download command without connecting or creating directories. `--local-root /path/to/results` changes the local parent directory; the run name is appended. Use `--host` and `--remote-dir` if your account or remote project location differs from the defaults; see `--help` for details.

The suite contains `room/`, `doorway/` and `g1_stairs/`. Each has `camera_voxel.mp4`, GIF and PNG previews, recorded RGB-D, cumulative maps, evaluation results and native-viewer inputs. `preset.json` records the requested schedule and resolution; each scene's `camera_voxel.json` records actual encoded fps, frame count and source-prefix alignment. Copy the complete directory to preserve portable replay. Exit the compute-node shell when finished to release the interactive allocation.

## 6. Replace room and doorway in the original suite

Sync the updated code and templates from the Mac using step 1. On PACE, open the compute-node shell as in step 2, then run:

```bash
bash scripts/pace/run_scene_suite.sh \
  --output outputs/g1_vision_30fps_pace \
  --replace-scenes room doorway
```

This keeps 640 x 480 capture, 121 actual observations and 30 fps MP4 output. Both selected scenes are captured and reconstructed again with the carried box; re-encoding old no-box maps would not introduce visual occlusion. The `g1_stairs` directory is retained. The suite gallery and evaluation summary are refreshed to reference the completed results.

Replacement is explicit and requires a recognized existing suite. The selected scenes are built in temporary directories before replacing their old generated results. PACE therefore needs additional space during the rerun. Wait for successful completion before downloading. If a transaction is interrupted, follow the recovery message instead of manually deleting directories.

Download back into the same local folder with:

```bash
bash scripts/pace/download_from_pace.sh g1_vision_30fps_pace
open outputs/g1_vision_30fps_pace/index.html
```

The downloader updates same-name files and restores deleted local videos. Removing only MP4/GIF files frees little space compared with stored captures and cumulative maps. Keep those files if native reconstruction replay is still needed.
