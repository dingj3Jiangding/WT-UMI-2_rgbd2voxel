# M8: independent head orientation under payload occlusion

M8 compares a fixed camera orientation with a prescribed head scan while the robot proxy continuously carries the box. The two conditions have identical body and payload trajectories, camera centers, timestamps, frame budgets, camera calibration and map settings. Only head-camera orientation changes. It is an orientation baseline, not an adaptive next-view planner or a locomotion controller.

The camera can turn without rotating the torso or box. Per-frame `head_yaw_degrees` and `head_pitch_degrees` are optional trajectory fields, defaulting to zero. Positive yaw rotates rig-forward +Y toward -X; positive pitch raises the optical axis relative to the calibrated mounting orientation. The local camera rotation is `Rz(head_yaw) @ calibrated_rotation @ Rx(head_pitch)`. The mounting position stays fixed. Saved captures contain the actual simulator camera pose and per-frame rig-to-camera transform. Configured limits are numerical benchmark bounds, not measured neck joint limits.

Rotation changes the field of view. At a fixed optical center, it cannot reveal a surface point whose ray is physically blocked by the box. The preset uses a scan chosen before evaluating results: yaw keyframes of 0, +25, 0, -25, 0 degrees and pitch offsets of 0, +10, 0, +10, 0 degrees over four seconds. This experiment does not assume that scanning improves coverage.

## Run

Activate the project environment and install optional simulator/video dependencies:

```bash
source .venv/bin/activate
python -m pip install -e '.[test,sim,video]'
python scripts/run_view_demo.py
```

The default `configs/demos/stairs_views.yaml` uses 13 actual poses, a 320 by 240 RGB-D camera and MP4 playback at 3 frames per second. It is a short local validation preset. Results go to a new `outputs/view_demo_<timestamp>/` directory. `--output` can select a new or empty destination. Existing results are never overwritten.

For 61 actual poses, a 640 by 480 camera and 15-frame-per-second MP4 playback, use `configs/demos/stairs_views_hq.yaml`. On PACE, first sync source from the Mac using `bash scripts/pace/sync_to_pace.sh`. Use the [PACE allocation and environment workflow](m7_high_quality.md), then run on the allocated compute node:

```bash
WTUMI_RUN="m8_head_scan_pace_$(date +%Y%m%d_%H%M%S)"
MUJOCO_GL=egl MPLBACKEND=Agg \
  python scripts/run_view_demo.py \
  --config configs/demos/stairs_views_hq.yaml \
  --output "outputs/$WTUMI_RUN"
printf 'Result folder: %s\n' "$WTUMI_RUN"
```

The package entry point is also available:

```bash
python -m wtumi_recon view-demo --config configs/demos/stairs_views.yaml
```

M8 needs new captures because camera orientations differ. It does not reuse the old M7 capture as if those orientations had been recorded. The high-quality PACE experiment remains to be run by the user; the local standard preset has completed with real MuJoCo rendering.

## Inspect

Open the result's `index.html` after downloading the entire result directory. The gallery contains:

- Raw and masked input, cumulative reconstruction and occupancy MP4 animations.
- Temporal 3D playback with the actual current camera optical-axis directions, making rotation visible even though both conditions share camera centers.
- Final mesh, voxel, point-cloud and occupancy viewers for each condition.
- Per-frame depth coverage, sampled surface completeness and unknown-region curves.
- `summary.csv`, `observation_history.json`, `visibility_evaluation.json` and `demo.json`.

Replay a final result from its relative configuration after moving the entire output folder:

```bash
python -m wtumi_recon run --config outputs/YOUR_RESULT/configs/head_scan.yaml
```

Playback frame rate is presentation timing. Browser 3D playback is best effort, and neither establishes real-time mapping throughput.

## What is measured

The map receives only the `occluded` condition's masked RGB-D observations. Each frame is integrated once into TSDF and occupancy. Paired no-box captures are used for validation, and are never selected for fusion. Two shared environment-only evaluation cameras and analytic stair samples are also excluded from map updates.

The visibility diagnostic projects the same analytic tread/riser samples into actual masked integration depth. A sample is observed when nearest-pixel camera-Z depth agrees within the configured tolerance. It records currently observed, newly observed, cumulatively observed and never-observed sample counts. The final comparison lists samples gained, lost, observed by both and never observed by either condition. Sample coordinates and tolerance are checked before comparing sets.

Never observed is a sampled-surface label: it can mean occlusion, missing depth, or being outside the field of view or depth range. It does not identify the reason and is not a count of unknown finite voxels. Reconstruction proximity is reported separately, since a nearby triangle is not proof that the corresponding surface was measured. Depth errors exclude missing predictions; surface completeness does not establish mesh precision or collision safety.

Both conditions are checked for matching body/box transforms, camera centers, frame schedules, calibration, held-out RGB-D and stair masks. These are exact simulated poses and masks. No physical neck dynamics, walking, pickup, localization error, dynamic clearing or tactile fusion is implemented.

## Local measured result

The completed 13-frame, 320 by 240 run is `outputs/m8_head_scan/`. It used a 21.875 millimetre TSDF cell width, a 54.6875 millimetre occupancy cell width and a 30 millimetre evaluation tolerance.

| Measure | Fixed head | Independent head scan |
|---|---:|---:|
| Stair depth coverage | 37.04% | 37.12% |
| Surface samples within 30 millimetres of the mesh | 51.79% | 51.79% |
| Cumulatively observed staircase samples | 1015 of 1960 | 1015 of 1960 |
| Never-observed staircase samples | 945 | 945 |
| Unknown cells in the specified region | 73.49% | 73.72% |
| Mean absolute depth error on overlap | 2.83 millimetres | 2.68 millimetres |

The scan gained zero sampled staircase surfaces and lost zero compared with the fixed orientation. Small changes in rasterized depth coverage and occupancy do not demonstrate recovery of the physically hidden staircase. The result supports testing lateral camera-center displacement as a separate next intervention. It does not show that head rotation is useless in other scenes or fields of view.

The two conditions within this run are controlled. The standard and high-quality presets have different frame budgets and image resolutions, so their numerical results are not a single-variable comparison.
