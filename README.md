# WT-UMI Part 2: RGB-D reconstruction infrastructure

Configuration-driven static scene reconstruction: MuJoCo / simple synthetic scenes / saved RGB-D sequences → TSDF → mesh → RGB-D rendered from original and held-out viewpoints.
RGB is interpolated from the fused vertex colors of the reconstructed mesh. The simulator captures observations and generates evaluation references; reconstruction recovers surfaces from images, calibration and camera poses.

Use English for all project-authored interface text, documentation, code comments, docstrings and generated reports.

## Start with the MuJoCo room (M5)

MuJoCo is installed and verified in the local Mac environment. From the project root, run:

```bash
./.venv/bin/python scripts/run_sim_demo.py
```

The program creates `outputs/sim_demo_<timestamp>/` and performs the following steps:

1. Capture RGB-D from four positions of one virtual camera in a static room containing walls, a floor, a table and colored boxes.
2. Capture a fifth, held-out viewpoint for evaluation only. This frame is excluded from map fusion.
3. Fuse the four observations into TSDF grids with voxel sizes of 31.250, 21.875 and 15.625 mm.
4. Render the fifth viewpoint from the reconstructed surface and generate RGB/depth comparisons, error curves and an offline 3D viewer.

The observations are captured once and reused at every voxel size. Here, moving the camera means placing it at configured positions to observe different surfaces. Continuous motion, a robot-mounted camera and collision-free trajectories are not simulated yet.

Existing measured results are in `outputs/m5_mujoco_demo/`. Open them directly:

```bash
open outputs/m5_mujoco_demo/experiment/runs/clean_n160_s7/renders/novel_front_right/comparison.png
open outputs/m5_mujoco_demo/experiment/error_vs_voxel_size.png
open outputs/m5_mujoco_demo/experiment/runs/clean_n160_s7/viewer.html
```

The viewer's Mesh, Voxel and Point cloud modes display the reconstructed surface, near-surface voxels and surface points, respectively. Camera markers distinguish fusion viewpoints from evaluation-only viewpoints.
See [M5 measured results](docs/m5_results.md) for the results, error interpretation and validation scope. See the [PACE runbook](docs/PACE_runbook.md) for Linux/PACE setup and execution. PACE has not been tested yet.

### Sync Mac changes to PACE

Run these commands from the Mac project root. The first upload can include the small saved RGB-D replay fixture:

```bash
bash scripts/pace/sync_to_pace.sh --dry-run --with-replay
bash scripts/pace/sync_to_pace.sh --with-replay
```

For later source updates, run `bash scripts/pace/sync_to_pace.sh`. The default destination is `login-ice.pace.gatech.edu:projects/WT-UMI-2_rgbd2voxel`, using your SSH account configuration. The script transfers changed source files and excludes environments and general output directories. It does not delete remote-only files; remove obsolete source files separately after local deletions or renames. See the [PACE runbook](docs/PACE_runbook.md) for authentication, destination overrides and Linux setup.

### Change the simulated scene

| What to change | File to edit |
|---|---|
| Room, table, object geometry and colors | `assets/simulators/mujoco/room.xml` (MuJoCo MJCF) |
| Capture positions, evaluation viewpoints and image size | `configs/capture/mujoco_room.yaml` |
| Reconstruction bounds, voxel resolution and experiment settings | `configs/demos/mujoco_room.yaml` |

Copy the configuration and scene files, then select the new example with `--config`. Relative paths are resolved from the directory containing each configuration file. After changing the scene, ensure that the map bounds cover the target region; cameras may be outside the map. MuJoCo uses a Z-up world frame, and the exported data preserves this information for 3D viewing. Internal camera coordinates still follow the OpenCV convention.

### Use captured data independently

`capture/manifest.json` records calibration, each frame's `T_wc`, timestamps, image paths and the integration/held_out split. `capture/frames/<id>/` contains `rgb.png` and `depth_m.npy`.
RGB is an aligned uint8 image. Depth is floating-point camera-Z depth in metres, not Euclidean distance to the camera center; 0 denotes invalid depth. Frame paths are relative to the manifest. The current sequence format uses fixed intrinsics and image dimensions.

Capture only:

```bash
./.venv/bin/python -m wtumi_recon capture \
  --config configs/capture/mujoco_room.yaml --output outputs/my_room_capture
```

Reconstruct the data from an existing complete demo without starting MuJoCo again:

```bash
./.venv/bin/python -m wtumi_recon run \
  --config outputs/m5_mujoco_demo/reconstruction.yaml
```

To reconstruct independently captured data, use the following schema 3 configuration. Save this example as `outputs/my_room_capture/reconstruction.yaml`:

```yaml
schema_version: 3
source:
  type: rgbd_sequence
  manifest: manifest.json
reconstruction:
  backend: uniform
  volume_length_m: 3.5
  origin_m: [-0.25, -0.25, -0.25]
  resolution: 160
  sdf_trunc_m: 0.08
evaluation:
  depth_tolerance_m: 0.02
```

Future simulator or camera exporters can write the same format through `SequenceWriter` in `sequence.py` and use the same reconstruction entry point. Real devices still need capture adapters, RGB-D alignment and a source of camera poses.
Reference types explicitly distinguish `simulator_ground_truth` from `observed_rgbd`; the latter is another observation and must not be treated as error-free ground truth.
When moving data between machines, copy the entire demo directory and use its top-level `reconstruction.yaml` or `experiment.yaml`. Internal `config_resolved.yaml` and `source_manifest.json` files record provenance and may contain absolute paths from the original machine.

## Run M2

Run from the repository root using the existing environment's Python executable. The activation script may still point to the project's previous location:

```bash
./.venv/bin/python -m wtumi_recon validate --config configs/scenes/multiview_cube.yaml
./.venv/bin/python -m wtumi_recon run --config configs/scenes/multiview_cube.yaml --output outputs/m2_cube
./.venv/bin/python -m wtumi_recon run --config configs/scenes/static_door.yaml --output outputs/m2_door
```

The output directory must be new or empty to avoid mixing experiments. On subsequent runs, change `--output` or omit it to create a directory with a timestamp automatically.
The two examples differ only in their YAML configuration; no core code changes are needed. `scripts/run_m2.py run --config ...` is a convenience wrapper around the same entry point.

## Inspect results

The output directory contains:

- `renders/novel_front_right/comparison.png`: RGB, depth and error comparisons for a viewpoint excluded from fusion.
- `metrics.json`: per-view metrics, actual fusion frame IDs, summaries weighted by pixel counts and the runtime environment.
- `mesh.ply` / `pointcloud.ply`: reconstructed 3D surfaces.
- `inputs/<view>/rgb.png`, `depth_m.npy`: fusion frames contain the observations actually sent to TSDF, including any noise or missing depth; held-out frames contain evaluation references only. Depth is in metres.
- `references/<view>/`: reference RGB-D used for metrics and error maps. Synthetic examples have clean references; sequence data declares its reference type at the source. Older outputs may lack this directory.
- `occupancy.npz` / `occupancy.json` / `occupancy_slices.png`: spatial states, evidence statistics and central slices saved when M4 is enabled.
- `renders/<view>/`: reconstructed RGB-D, depth error arrays and missing/extra pixel masks.
- `manifest.json`: per-frame intrinsics, camera-to-world poses and the integration/held_out split.
- `config_source.yaml` / `config_resolved.yaml`: original and resolved configurations.
- `tsdf_volume.npz`: normalized TSDF values, weights, colors and grid parameters for the Uniform backend. A weight of 0 means unobserved and must not be treated as free space.

On Mac, open an image with `open outputs/m2_door/renders/novel_front_right/comparison.png`.

### Rotate and inspect 3D results in a browser

Existing results do not need to be reconstructed again. Run:

```bash
./.venv/bin/python -m wtumi_recon view --result outputs/m2_door
open outputs/m2_door/viewer.html
```

Use `--result outputs/m2_cube` to inspect the cube and sphere instead. The exported `viewer.html` embeds the plotting library and geometry, so it can be opened offline or copied on its own. The adjacent `viewer.json` records display settings and counts. The door scene's approximately 13 MB HTML file may take time to load initially.

1. **Mesh**: inspect the triangular surface extracted from the fused TSDF. Drag to rotate, scroll to zoom and use the toolbar to switch to panning.
2. **Voxel**: inspect observed cubic cells near the surface, colored with fused TSDF colors. The default filter is `weight >= 1` and `abs(tsdf) * sdf_trunc_m <= voxel_size_m`.
3. **Point cloud**: inspect the reconstructed surface point cloud.
4. Enable **Show camera poses**: teal cameras participate in fusion, and yellow cameras are used only for evaluation. Each marker represents a frame's camera pose, so one moving camera appears at multiple positions.

Switch to Voxel and zoom into the door handle and frame to inspect the 15.625 mm voxel size. Rotate to the back to check which regions lack observation support. Mesh and Voxel outlines differ slightly: the mesh interpolates the zero-level surface, while the voxel mode displays a layer of cells near zero.

These near-surface voxels do not form a complete occupied/free/unknown map. A weight of 0 indicates unobserved space, while other cells may be hidden by distance or weight filters; hidden cells must not all be treated as free. The default display limit is 100,000 cells. Exceeding it produces an error rather than silently sampling cells or reducing resolution.

Optional parameters include `--band-voxels 0.5` to narrow the displayed band, `--min-weight 2` to raise the fusion weight threshold and `--max-voxels 200000` to increase the cell limit. These affect viewing only and do not change reconstruction files. `--output outputs/door_view.html` sets the output HTML path and the matching JSON path. Re-exporting can update an existing viewer; a JSON file that is not viewer metadata will not be overwritten. Older results without `tsdf_volume.npz` can still display a mesh and point cloud, but the Voxel button is disabled.

## Change synthetic scenes

Copy an M2 YAML file and edit `scene.objects` and `views`:

- box: `size_m`, `translation_m`, `color_rgb`.
- sphere: `radius_m`, `translation_m` (sphere center), `color_rgb`.
- mesh: `path` (relative to the YAML file) and optional transforms/colors. Model coordinates are interpreted in metres. Only geometry and vertex colors are supported; remove texture materials first.
- `rotation_deg_xyz`: optional Euler angles in degrees. Rotation is applied about the object's local origin before translation.
- `color_mode: coordinate`: generate a vertex color gradient from object-local coordinates to check color reconstruction.
- Cameras: provide `eye`, `target` and `up`, or supply `T_wc` directly. `split: integration` participates in fusion; `held_out` is used only for evaluation.

Internal units are metres. Camera coordinates follow OpenCV: x right, y down and z forward. Depth is camera-Z depth, and pixel centers use integer coordinates `(u,v)`.
The synthetic examples also use a downward world y-axis, so their default camera up direction is `[0,-1,0]`.
The Uniform map spans `[origin_m, origin_m + volume_length_m]`. The scene must fit within this range, while cameras may lie outside it.
Uniform voxel size is determined solely by `volume_length_m / resolution`; M2 configurations reject a conflicting additional `voxel_size_m` value.

## Metric definitions and limitations

Coverage is the number of overlapping valid-depth pixels divided by the number of valid reference-depth pixels. MAE, RMSE and RGB MAE/PSNR are computed only where both depth maps are valid.
Missing pixels, extra pixels and silhouette IoU are reported separately. `accurate_reference_fraction` requires both a reconstruction and a depth error within the tolerance.
When there is no overlap, errors are JSON `null`. If colors match exactly, `rgb_exact_match=true` and PSNR is `null` because its mathematical value is infinite.
Comparison images use magenta for missing pixels, cyan for extra pixels and red for depth error. Black background pixels are excluded from color metrics.

The pipeline currently supports static scenes with known poses and optional controlled depth noise and random dropout. Simple synthetic scenes have no lighting or shadows. The MuJoCo example uses simulator shading but disables shadows, reflections and multisampling. Reconstruction uses fused vertex colors and does not solve for materials or lighting. Unobserved back surfaces are not completed automatically.
MuJoCo and generic saved sequences are supported, along with an optional static ray-evidence map. Real cameras, pose estimation, dynamic robots or door motion, and tactile input are not connected yet. Whether RGB-D maps serve as initial robot maps or only as training/evaluation data remains an experimental decision; the current capture and reconstruction pipeline does not depend on that choice.
In simple probes with the local Open3D 0.20 installation, the Scalable backend returned an empty mesh. The cause is unresolved. Examples explicitly select Uniform and do not switch backends silently.
The room example checks the data pipeline and the effect of resolution. Its results alone do not establish sufficient accuracy for planning around door handles or contact regions.

## Reconstruction quality experiments: voxel size and depth noise

The three entry points serve different purposes: `run` performs one reconstruction, `view` inspects an existing 3D result, and `experiment` runs configured comparisons in batches.

```bash
./.venv/bin/python -m wtumi_recon experiment --config configs/experiments/door_quality.yaml
```

By default, this creates `outputs/quality_<timestamp>/`. Use `--output` to specify another new or empty directory.

The example contains nine door-scene experiments:

| Variable | Settings |
|---|---|
| Grid resolution | 80³, 160³, 240³ |
| Corresponding voxel size | 31.250, 15.625, 10.417 mm, with a fixed 2.5 m volume side length |
| Input condition | Clean; 5 mm Gaussian depth noise; the same noise plus 10% random depth dropout |

The 5 mm value is the standard deviation of camera-Z depth noise. The 10% value is the independent dropout probability for each originally valid pixel. These are synthetic perturbations, not a calibrated model of a specific camera. RGB receives no color noise, but colors are masked where depth becomes invalid. The scene, cameras, poses and TSDF truncation distance remain fixed.

Fusion inputs are saved in `inputs/`, and clean evaluation references in `references/`. The left column of each comparison is labeled Reference and differs from noisy input. Omitting `sensor` defaults to zero noise and zero random dropout, so existing scene commands remain valid.

See [M3 measured results](docs/m3_results.md) for the local results and interpretation.

After an experiment, generate curves of voxel size versus MAE, RMSE and coverage. The MuJoCo demo performs this step automatically:

```bash
./.venv/bin/python -m wtumi_recon plot --summary outputs/your_experiment/summary.json
```

In the new output directory, inspect `overview.png` first, then read `report.md`:

- `overview.png`: reconstructed RGB and depth errors for the first held-out view. In the default configuration, each row is an input condition and each column is a resolution.
- `summary.csv` / `summary.json`: summaries across all held-out views, per-run times, grid cell counts and descriptive statistics across seeds.
- `configs/`: scene YAML files for rerunning individual cases. `runs/`: complete reconstruction outputs for each case.
- `completed_runs.json`: records completed experiments so earlier results remain available if a later case fails.

For a given seed and input condition, depth inputs are identical across grid resolutions, verified by hashes. Clean held-out references are also checked. The example uses only seed=7 and is an initial comparison. Set `seeds` to `[7, 19, 31]` in the experiment YAML to add repeats. Clean input runs once per resolution; other input conditions repeat across seeds.

Cases run sequentially. Uniform cell counts grow with the cube of resolution; `grid_cells` is not measured memory usage. `total_seconds` includes synthesis, fusion, rendering and most file exports, so it is not a pure reconstruction frame rate. MAE/RMSE consider only valid-depth overlap and must be interpreted alongside coverage, missing/extra pixels and coverage within the 20 mm tolerance. Scene-wide means alone do not establish handle accuracy.

For a single reconstruction, the scene YAML can also include this top-level configuration:

```yaml
sensor:
  depth_noise_std_m: 0.005
  dropout_probability: 0.10
  seed: 7
```

## M4: distinguish unknown, free and occupied space

M4 adds a bounded ray-evidence map. It reads the actual depth images, intrinsics and `T_wc` poses used for fusion. Held-out images do not update the map.

| State | Evidence |
|---|---|
| unknown (gray, 0) | No valid ray has provided evidence for the cell |
| free (teal, 1) | A valid measurement ray traversed the cell before its endpoint cell, and no hit evidence exists |
| occupied (orange, 2) | At least one valid measurement endpoint falls inside the cell |

Rays traverse cells at grid boundaries and stop at their depth endpoints. Measurements that are 0, NaN, infinite or outside the depth range do not update the map. Occluded regions retain their previous state. Each frame adds at most one free count and one hit count per cell, preventing pixel density from directly acting as confidence.

Free space records evidence of ray traversal and does not guarantee that every part of a cell is clear. Occupied space records surface hits without filling object interiors. A cell may receive both hit and free evidence; its state is then occupied, and both counts are retained. Different rays crossing different parts of one coarse cell can also produce this combination, so `conflict_cells` is not a count of sensor errors. This policy is intended for static scenes and does not yet clear old evidence from moving objects.

### Inspect the map

Open the existing door-scene results:

```bash
open outputs/m4_door/occupancy_slices.png
open outputs/m4_door/viewer.html
```

The viewer includes an **Occupancy** mode for spatial-state slices. Select the X, Y or Z axis, then drag and release the index slider to inspect a layer at a different position. The page shows the layer's world-coordinate bounds. Only one layer is displayed so the full unknown volume does not obscure the interior. Enable camera poses to help interpret observation directions.

Generate a new result, using a new directory with a timestamp by default:

```bash
./.venv/bin/python -m wtumi_recon run --config configs/scenes/m4_static_door.yaml
# Replace this path with the output directory printed above.
./.venv/bin/python -m wtumi_recon view --result outputs/your_run
```

Enable occupancy by adding the following top-level configuration to any Uniform scene YAML. It is disabled by default for existing scenes:

```yaml
occupancy:
  enabled: true
  resolution: 80
  pixel_stride: 1
  probes:
    - id: check_point
      point_m: [1.0, 1.0, 1.0]
```

The occupancy grid shares the Uniform TSDF's world origin and volume bounds, but its resolution can be set independently. The door example uses an 80³ occupancy grid (31.25 mm), while TSDF remains 160³ (15.625 mm). `pixel_stride: 1` uses all valid pixels; a larger stride samples fewer rays and may leave more cells unknown. `probes` queries states at world coordinates. Positions outside the map return `outside` and are not treated as free.

See [M4 measured results](docs/m4_results.md) for the full results and probe interpretation. The comparison image is `outputs/m4_occupancy_comparison.png`.

### Verify occlusion behavior

Three configurations share the same geometry: an occluding panel, a box behind it and a distant background wall.

- `m4_occlusion_front.yaml`: fuses only the front observation. The side camera is used for evaluation and does not update occupancy.
- `m4_occlusion_reveal.yaml`: adds a right-side observation, revealing some previously occluded space and surfaces.
- `m4_occlusion_missing.yaml`: randomly removes 50% of front-view depth to verify that missing observations are not treated as free space.

Results are in `outputs/m4_occlusion_front/`, `outputs/m4_occlusion_reveal/` and `outputs/m4_occlusion_missing/`. All use 80³ grids with 25 mm cells. Inspect `probes` in `occupancy.json` to verify individual state transitions. The NPZ file stores full `(x,y,z)` states and evidence counts. An unknown state inside an object does not imply that the interior is empty.

## Environment and tests

The local environment is already configured. Recreate the environment when moving to another machine; do not copy `.venv`:

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install -e '.[test,sim]'
./.venv/bin/python -m pytest -q -p no:cacheprovider
```

Local reconstruction tests use Open3D 0.20, Python 3.12 and CPU execution. This Open3D wheel requires Homebrew `libusb` on the local machine.
Tests cover analytic geometry, off-center pixels, color interpolation, poses, held-out isolation and multi-view reconstruction end to end. Viewer checks cover voxel coordinates/colors, shared-face removal, camera frusta, offline export and source-file protection.

MuJoCo is an optional `sim` dependency. The locally verified version is 3.17.0. `requirements-mujoco.txt` records the local package combination. See the PACE runbook for Linux setup and rendering backend selection.
The default test run skips two tests that require actual OpenGL rendering. On Mac, run the following in a normal terminal to include them. The M5 verification passed all 139 tests:

```bash
WTUMI_TEST_MUJOCO_RENDER=1 ./.venv/bin/python -m pytest -q -p no:cacheprovider
```

Mac uses the default CGL backend; do not set the Linux-specific `MUJOCO_GL=egl` on Mac. Tests also cover data relocation, independent sequence reconstruction and reuse of one capture at multiple resolutions.

M1 remains available as a minimal demo: `./.venv/bin/python scripts/run_m1.py --output outputs/m1_updated`.
The updated code standardizes pixel centers and corrects the RGB source. Existing M1 outputs have not been overwritten, and rerun values may differ slightly from older results.
See the [infrastructure plan](docs/part2_infra_plan.md) for the full roadmap.
