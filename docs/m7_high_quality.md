# M7: higher-resolution capture and smoother playback on PACE

The high-quality preset uses a 640 by 480 RGB-D camera and 61 actual camera poses, sampled every 1/15 second from simulation timestamp 0 through 4 seconds. MuJoCo renders new RGB, depth and segmentation images at each pose. The positions and yaw angles interpolate the original five camera waypoints; the images and reconstructed maps are not interpolated.

The observe-first condition has 30 clear frames, followed by 31 box-occluded frames. Its box phase starts at timestamp 2 seconds. The always-occluded condition uses the same 61 poses with the box present throughout. This remains a scheduled comparison of paired captures, without physical box pickup, walking or stair climbing.

The preset exports MP4 animations at 15 frames per second. Playback lasts about 4.07 seconds, including the final frame. Encoding at 15 frames per second does not establish that capture or reconstruction runs at 15 frames per second. The interactive 3D viewer targets the selected playback rate, but its actual speed depends on browser and graphics performance. Use the MP4 files for predictable meeting playback.

The map still uses 160 cells along each side of the 3.5 metre reconstruction volume: each TSDF cell is 21.875 millimetres wide. Raising camera resolution improves the input images but does not independently refine the voxel grid. The occupancy grid stays at resolution 64. Its pixel stride changes from 2 to 8 to preserve approximately the same angular ray spacing at four times the image width.

Each observation is integrated once per condition. Intermediate snapshots save the mesh, occupancy and held-out evaluation outputs; the final snapshot exports the complete reconstruction result. There is still evaluation, image encoding and file export for every prefix, so this is an offline benchmark rather than an online mapping performance claim.

The dense experiment has more visual information than the original five-frame M7 baseline. Compare its two conditions with each other. A numerical comparison against the five-frame result also changes the observation budget and clear-prefix schedule.

## 1. Upload the source from your Mac

Run in a terminal on the Mac:

```bash
cd "/Users/fudijie/Learning_Material/GT MSR/26Fall/WT-UMI-2_rgbd2voxel"
bash scripts/pace/sync_to_pace.sh \
  --host djiang94@login-ice.pace.gatech.edu \
  --remote-dir projects/WT-UMI-2_rgbd2voxel
```

The sync sends source, configuration, tests and documentation. It excludes virtual environments and experimental output, and does not delete remote-only files. This preset requires a fresh capture; the earlier 160 by 120 capture does not match its calibration or pose schedule.

## 2. Request a compute node on PACE

In your PACE terminal:

```bash
cd ~/projects/WT-UMI-2_rgbd2voxel
module load python/3.12.5
bash scripts/pace/request_node.sh \
  --gpu --cpus 8 --mem 32G --time 01:00:00
```

Wait for the allocated compute-node shell. The helper enters the project and activates its existing `.venv`. The one-hour value is an allocation limit, not a predicted runtime. If your account requires a partition, account or quality-of-service flag, use the values assigned by PACE; this project does not assume site-specific values. If you already have an appropriate allocation, use it directly.

The GPU is used for MuJoCo EGL rendering. The current Open3D TSDF integration uses the CPU, so a larger GPU alone will not speed up every part of this workflow.

## 3. Update the environment and check rendering

Inside the allocated compute node:

```bash
source .venv/bin/activate
python --version
python -m pip install -e '.[test,sim,video]'
python -m pip check

MUJOCO_GL=egl WTUMI_TEST_MUJOCO_RENDER=1 \
  python -m pytest -q -rs tests/test_rig_capture.py
```

Python should report the version used to create this PACE virtual environment, normally 3.12.5. The `video` extra installs `imageio-ffmpeg` and its platform-specific FFmpeg executable for MP4 encoding; no separately installed `ffmpeg` shell command is required.

## 4. Run the high-quality experiment

Run on that same allocated compute node:

```bash
WTUMI_RUN="m7_hq_pace_$(date +%Y%m%d_%H%M%S)"
MUJOCO_GL=egl MPLBACKEND=Agg \
  python scripts/run_memory_demo.py \
  --config configs/demos/stairs_memory_hq.yaml \
  --output "outputs/$WTUMI_RUN"
printf 'Result folder: %s\n' "$WTUMI_RUN"
```

Keep the displayed result folder name for downloading. An output directory must be new or empty; the timestamp avoids replacing an earlier run. When the command finishes, `demo.json` records the actual settings, paths and measured results. A failed command can leave a partial result directory; choose a new name when rerunning.

The 3D temporal viewer contains 61 map snapshots and can be substantially larger than the five-frame example. Download the full result if you want replay configurations, raw observations, diagnostics and all viewers. Downloading only MP4 files is sufficient for showing those videos, but does not preserve the complete experiment.

## 5. Download and open the results on the Mac

Use a Mac terminal, not the PACE terminal. Replace the example value of `WTUMI_RUN` with the name printed by the successful run:

```bash
cd "/Users/fudijie/Learning_Material/GT MSR/26Fall/WT-UMI-2_rgbd2voxel"
WTUMI_RUN="m7_hq_pace_YYYYMMDD_HHMMSS"
mkdir -p "outputs/$WTUMI_RUN"
rsync -avz \
  "djiang94@login-ice.pace.gatech.edu:projects/WT-UMI-2_rgbd2voxel/outputs/$WTUMI_RUN/" \
  "outputs/$WTUMI_RUN/"
open "outputs/$WTUMI_RUN/index.html"
```

The result gallery provides the following animations. Their canvas dimensions include labels and side-by-side panels; they do not change the 640 by 480 source camera resolution.

| File | Canvas dimensions | Content |
|---|---|---|
| `capture_history.mp4` | 1440 by 1240 | Raw and masked RGB for both observation histories |
| `reconstruction_history.mp4` | 2560 by 1000 | Cumulative maps rendered from one shared held-out camera |
| `occupancy_history.mp4` | 1600 by 956 | A fixed spatial slice through both occupancy maps |

`animation_metadata.json` records encoding, dimensions, playback rate, frame count, duration and capture timestamps. `history_viewer.html` provides temporal 3D playback; the gallery also links final interactive maps and measured progress curves. Held-out views and analytical stair samples remain evaluation-only.

After the run, `exit` releases the interactive PACE allocation. Reopening the downloaded HTML and MP4 files on the Mac does not require a PACE allocation.
