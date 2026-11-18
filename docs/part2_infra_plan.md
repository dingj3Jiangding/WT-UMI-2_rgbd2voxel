# Part 2: reusable RGB-D reconstruction infrastructure

## Goal

Build a scene-agnostic RGB-D reconstruction pipeline that can run the same code on a cube, a door, a tabletop, or a real RGB-D sequence. A new scene should require a new data directory and configuration file, not changes to the reconstruction code.

The first closed loop is:

```text
RGB-D frames + calibration + poses
    -> camera-frame points
    -> world-frame geometry
    -> occupancy or TSDF map
    -> rendered RGB-D
    -> quantitative evaluation
```

## Stable contracts

- Lengths and depth values are in metres.
- Images use one RGB/depth resolution and pixel alignment.
- `K` is a 3x3 pinhole intrinsic matrix.
- The public internal pose convention is `T_wc`, mapping camera-frame points to world-frame points.
- Dataset adapters may use another convention, but convert to `T_wc` at the boundary.
- Depth is camera-Z depth, with integer pixel centres; the internal camera axes are right/down/forward (OpenCV).
- The optional occupancy grid explicitly distinguishes `unknown`, `free`, and `occupied`. Near-surface TSDF displays alone do not establish free space.
- Integration observations and held-out evaluation views are separate. A reference is declared as simulator ground truth or observed RGB-D; observed data is not silently treated as clean truth.

## Configuration-driven scene layout

Saved RGB-D sequences now use this implemented layout:

```text
data/scenes/<scene_name>/
  manifest.json
  frames/<view_id>/
    rgb.png
    depth_m.npy
```

The manifest contains fixed calibration, camera-to-world poses, timestamps, relative image paths, reference provenance and the integration/held-out split. Adapters convert depth to floating-point metres before writing it; the core does not guess a depth scale.

Schema 3 reconstruction YAML selects `source.type: rgbd_sequence`, a manifest, and map/evaluation parameters. The existing schema 2 YAML still generates simple synthetic geometry. Both feed the same reconstruction pipeline. Simulator assets, capture configuration and reconstruction configuration are separate.

## Milestones

### M0: geometry sanity

Test pixel back-projection, pose transforms, voxel indexing, and depth units with analytic points and a synthetic cube.

### M1: single-frame TSDF loop

Generate one RGB-D frame, integrate a uniform TSDF, extract a mesh, and render the same camera view. This catches calibration and coordinate errors before multi-view fusion. The separate occupancy surface-point helper is not yet a free/unknown ray-carving implementation.

The macOS baseline uses Open3D `UniformTSDFVolume`. In probes in this local Open3D 0.20 environment, `ScalableTSDFVolume` returned empty meshes. The root cause is not established; this is not evidence of a general Apple Silicon limitation. The scalable backend remains experimental.

### M2: multi-frame TSDF baseline

Use known poses to integrate multiple synthetic frames. Scene objects, camera intrinsics and poses, TSDF bounds and resolution are configured in YAML. Examples cover a colored cube plus sphere and a static door plus frame/handle/floor. Extract a point cloud and mesh, then render original and held-out views. Held-out images are generated only after integration and never sent to TSDF. RGB comes from reconstructed vertex colors, not scene colors. Keep direct voxel raycasting as a later alternative to the current mesh intermediate.

Save source/resolved configurations, input frames, K/T_wc, integration IDs, reconstructed RGB-D, missing/extra masks, a uniform TSDF snapshot and metrics. See README for executable commands and the exact supported scope.

### M3: evaluation and diagnostics

M2 includes depth MAE/RMSE, coverage, silhouette IoU, RGB MAE/PSNR, depth-inlier coverage and wall times, plus comparison images and standard JSON metrics. An offline HTML viewer now displays the reconstructed mesh, observed near-zero TSDF cells, surface point cloud and integration/held-out camera frusta from saved outputs. The voxel display is an inspection layer, not a complete occupancy map.

Controlled resolution/noise experiments are implemented with a separate YAML and the `experiment` command. Synthetic Gaussian Z-depth noise and independent pixel dropout affect integration inputs only; clean references remain separate. Paired inputs and held-out references are verified by hashes. Saved reports include coverage, overlap errors, inlier coverage, missing/extra pixels, grid counts, timings and descriptive per-seed statistics. The nine-case door example is a single-seed pilot.

Memory benchmarking, 3D voxel IoU and Chamfer distance remain future additions. Silhouette IoU is a 2D mask metric and must not be reported as 3D voxel IoU.

### M4: partial observability

Implemented an opt-in bounded evidence grid alongside Uniform TSDF. Exact grid traversal records free cells before each measured endpoint and occupied endpoint cells; invalid depths add no evidence and occluded regions stay unknown. Only actual integration observations update the map, with per-frame unique hit/free counts. Occupied evidence wins conflicts; this is a conservative static evidence policy, not calibrated occupancy probability, solid filling or whole-cell collision clearance.

Save full states/counts, statistics and configured world-point probes. Inspect three static orthogonal slices or use the HTML viewer's axis/index slice controls. Front-only, added-side-view and 50% missing-depth fixtures verify occlusion and evidence transitions. The occupancy grid shares the TSDF volume bounds but can use a separate resolution.

Temporal clearing, probabilistic sensor fusion, object-interior inference and tactile updates remain future work. This evidence representation provides a starting point for later tactile probing experiments.

### M5: simulator adapter and portable RGB-D sequences — implemented on Mac and PACE

Use MuJoCo as the team's first, replaceable simulator example. Capture a static room from four camera poses and reserve a fifth pose for evaluation. MuJoCo camera axes are converted to the OpenCV convention; the actual simulator calibration and camera pose are exported. The world is Z-up, carried through metadata to the result viewer.

The simulator-independent `RGBDSequence` reader and `SequenceWriter` preserve RGB, metric camera-Z depth, poses and reference declarations. Arrays load lazily; held-out pixels are read for evaluation after integration. Other simulators and future camera adapters can write the same format without changing TSDF fusion. Real-camera capture, RGB/depth alignment and pose estimation are not implemented yet.

The `demo` command and `scripts/run_sim_demo.py` capture once, reuse identical inputs at three physical voxel sizes, render the held-out view, and save RGB-D comparisons, PNG/SVG error curves and a three-dimensional viewer. Portable entry configurations use relative paths and can move with the full dataset. Resolved configs and copied source manifests are provenance records, not cross-host entry points.

This is a static capture at camera waypoints, not a continuous simulated trajectory. The first example contains no robot, door motion or tactile input. RGB reconstruction uses fused vertex colours and does not reproduce the simulator's full material or lighting model.

Original M5 Mac verification: 139 tests passed with actual MuJoCo rendering enabled; the full room demo completed. A later PACE EGL room run also completed; the downloaded measurement record is `outputs/pace_meeting_results/`. See [M5 results](m5_results.md) for the historical Mac milestone and [PACE runbook](PACE_runbook.md) for current instructions.

### M6: head-camera staircase occlusion — implemented on Mac and PACE

A kinematic rig moves a head-height camera, torso proxy and carried box together near a four-step staircase. Three conditions compare five no-box frames, the first box frame and all five box frames. Box/self pixels are excluded before both map updates. Two shared evaluation cameras and analytic stair samples quantify missing geometry. M6 ran separately on Mac and PACE; see [M6](m6_stairs_occlusion.md).

### M7: visual history before later occlusion — implemented on Mac and PACE

Two histories use identical poses and frame budgets: always carrying a box, or first observing without it and then using box observations. Actual paired simulator frames are scheduled into portable datasets; no physical pickup is simulated. Every observation prefix is reconstructed and evaluated. Both final viewers, temporal 3D playback and capture/occupancy/render animations are exported automatically. A common cohort of preobserved stair samples currently hidden behind the box measures geometric representation after occlusion. See [M7](m7_visual_memory.md).

Saved M6 captures can be replayed without graphics initialization. Fresh capture remains available. Held-out references and analytic evaluation geometry never enter fusion. Prefix replay is a benchmark implementation, not a real-time mapping claim.

The high-quality fresh EGL capture has completed on PACE: 61 actual frames per condition, 640 by 480 input and verified 15-frame-per-second MP4 exports. See [the measured PACE result](m7_pace_results.md).

### M8: independent head orientation — implemented and locally validated

Optional per-frame head yaw/pitch changes camera orientation independently of the torso and box. Fixed-head and prescribed-scan conditions share body/payload poses, optical centers, calibration, timestamps, frame budgets and held-out references. Incremental reconstruction, camera-direction playback and an evaluation-only observed/never-observed stair sample diagnostic are implemented. The local 13-frame run measured no additional visible stair samples from this scan; rotation alone cannot unblock a physically occluded ray at the same optical center. See [M8](m8_head_scan.md). The 61-frame high-quality preset is ready for a separate PACE run.

### Next: lateral observations and realistic errors

1. Run the M8 high-quality preset on PACE and inspect the recorded camera poses, visibility sets and animations.
2. Test lateral camera-center displacement or a separate lateral body path while carrying the box, preserving equal frame budgets and documenting any changed pose constraints. Assess physical feasibility separately; do not claim kinematic waypoints are executable robot motion.
3. Introduce controlled pose error, depth noise and imperfect self/payload masking before interpreting ideal-simulator results as real-robot performance.
4. Define a later tactile-update interface that records observation source and preserves unknown space. Tactile fusion and dynamic clearing remain separate implementations.

The current project direction is a head-mounted RGB-D vision-to-voxel module with later tactile observations for missing local information. Robot control and behavior-model adaptation remain outside this infrastructure.

## Intended module boundaries

```text
src/wtumi_recon/
  config.py       YAML and validation
  io.py           scene manifests, images, poses, calibration
  sequence.py     portable RGB-D sequence reader/writer
  mujoco_capture.py  MuJoCo camera capture and convention conversion
  geometry.py     back-projection and frame transforms
  occupancy.py    bounded static ray-evidence map and legacy point helper
  occupancy_diagnostics.py  orthogonal state slices
  tsdf.py         Open3D TSDF integration and extraction
  render.py       voxel/mesh RGB-D rendering
  metrics.py      reconstruction metrics
  viewer.py       offline 3D inspection of saved results
  sensor.py       reproducible synthetic depth perturbations
  experiments.py  controlled parameter sweeps and comparison reports
  experiment_plots.py  physical voxel size versus error/coverage plots
  demo.py         one capture, shared-input sweep and result entry points
```

The reconstruction command accepts a config path and output directory. The viewer accepts an existing result directory and optional display settings. This keeps scene changes outside the core implementation.

## First acceptance test

```text
synthetic cube config
  -> run baseline command
  -> output reconstructed RGB-D and geometry
  -> pass geometry unit tests
  -> write metrics.json
```

Do not add tactile inputs or behavior-model fine-tuning to this pipeline. Those belong to the later tactile exploration and behavior adaptation experiments.
