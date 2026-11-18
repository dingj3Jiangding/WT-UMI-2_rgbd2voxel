# M6: Head-camera mapping with carried-box occlusion

This benchmark extends the static reconstruction backend to a moving kinematic sensor rig. A camera attached at head height and a carried box move through a configured approach trajectory near a staircase. The environment is static. Simulator camera poses and geometry segmentation are known inputs; this is not a walking controller, SLAM implementation, or tactile fusion system.

## Run

From the project root with the project environment active:

```bash
python scripts/run_occlusion_demo.py
```

The equivalent package command is:

```bash
python -m wtumi_recon occlusion-demo --config configs/demos/stairs_occlusion.yaml
```

On an allocated PACE GPU node, use EGL:

```bash
MUJOCO_GL=egl MPLBACKEND=Agg python scripts/run_occlusion_demo.py
```

The default output is a new `outputs/occlusion_demo_<timestamp>/` directory. An explicit `--output` must be new or empty. Source synchronization uses the existing `scripts/pace/sync_to_pace.sh`; no new Python dependencies are required beyond `.[test,sim]`.

## Experimental conditions

| Condition | Fusion input | Purpose |
|---|---|---|
| `unoccluded_multi` | All no-box observations, with robot pixels excluded | Paired visual baseline |
| `occluded_single` | First masked observation from the box sequence | Initial information available under occlusion |
| `occluded_multi` | All masked observations from that same box sequence | Information accumulated by moving the camera and box |

The two capture conditions use matching timestamps and realized camera poses. The single-frame case references the original box sequence files rather than recapturing them. All conditions use identical, environment-only held-out RGB-D references. Held-out images, stair masks and analytic stair surfaces are evaluation data; none are fused.

The box is absent only in the no-box baseline and evaluation references. In the box condition, actual box/robot pixels are identified by simulator geometry segmentation and set to zero depth before either TSDF or occupancy integration. No depth is inserted behind the box. Previously observed static surfaces persist; areas never observed remain unknown. Payload and rig geometry/poses are saved separately in the capture record.

## Change the setup

- `configs/capture/stairs_rig.yaml`: camera calibration and mounting, staircase dimensions, box/torso geometry, trajectory, held-out cameras and fixed region of interest.
- `configs/demos/stairs_occlusion.yaml`: TSDF volume, occupancy resolution/stride, geometric error threshold and analytic surface sampling spacing.

Coordinates are in metres in a Z-up world. The rig faces local +Y; yaw rotates around world Z. The staircase origin is its center X, front Y and base Z. Each configured step has a full tread depth and a rise height. Camera poses use the existing OpenCV convention and `T_wc` transforms. The camera is a child of the rig, not a world camera teleported independently of the payload.

The generated MJCF and capture configuration are saved with the run. Motion is prescribed kinematic playback with `mj_forward`, not a physical locomotion rollout. Masking currently uses simulator IDs; real-device integration needs calibrated mounting, a pose source and a suitable robot/payload mask provider.

Each integration frame must retain some valid environment depth after masking. A completely blocked frame currently stops the benchmark with a validation error; skipping such frames during continuous operation is not implemented.

## Inspect outputs

Open the run's `README.md` for links to:

- `capture/capture_preview.gif`: raw and masked observations along the trajectory.
- `condition_comparison.png`: the three conditions at the same first held-out camera. Tables pool all held-out stair masks.
- `observation_progress.png`: unknown occupancy fraction in the fixed staircase region and the fraction of analytic stair samples in observed cells, as more frames arrive.
- `occupancy_progress.gif`: the same center-X Y-Z occupancy slice over time for both capture conditions.
- `runs/occluded_multi/viewer.html`: offline mesh, surface voxels, point cloud, camera poses and occupancy slices.
- `summary.csv` and `runs/<condition>/stair_metrics.json`: numeric results, counts, thresholds and metric definitions.
- `demo.json`: run configuration, paired-input verification, results and output paths.

Each sequence keeps `frames/<id>/raw_rgb.png`, `raw_depth_m.npy` and `excluded_mask.png` alongside the masked `rgb.png` and `depth_m.npy` consumed by reconstruction. The capture record retains robot/box geometry separately. The first-frame manifest lives beside its source manifest so that shared frame paths stay portable.

## Metric definitions

Stair-only depth coverage is the fraction of valid reference depth pixels inside the held-out stair mask that have valid predicted depth. Mean absolute depth error and root mean square depth error use only the overlap. The within-tolerance fraction uses all valid reference pixels in the mask, so missing depth is not counted as accurate.

Surface completeness is measured from deterministic samples on analytic treads and front risers to the nearest triangle in the reconstructed mesh. Surface coverage counts distances within the configured threshold. This direction tests how much reference geometry has been recovered; it does not measure the precision of every reconstructed triangle. Solid interiors are not reference surfaces.

Unknown/free/occupied fractions are counted at occupancy voxel centers inside a fixed region. A sampled ray provides free-space evidence before its measured endpoint; occupied cells contain observed surface returns. Finite cells may contain both ray traversal and surface evidence. The fraction of ground-truth surface samples falling in cells labelled free is a diagnostic, not a collision-safety probability. Out-of-volume samples are reported separately.

The region includes solid stair interiors and hidden volumes, which are not filled by surface observations. Its unknown fraction is not a surface-completeness score and is not expected to reach zero. Read it alongside exposed-surface coverage and stair-only depth metrics.

TSDF and occupancy grids can have different cell sizes. Both sizes and the geometric tolerance are recorded in the report. Simulator poses and masks remove two important real-world error sources, so these results do not establish performance under pose drift, sensor noise or imperfect segmentation.

## Replay and tests

Copy the entire output directory to preserve the paired sequences and references. For example:

```bash
python -m wtumi_recon run --config outputs/your_run/configs/occluded_multi.yaml
```

The replay configs and sequence frame paths are relative. Internal resolved configs retain original provenance paths and are not portable entry points. Replay does not launch MuJoCo.

Run the tests after installing the test/simulator extras. Actual MuJoCo rendering must be explicitly enabled on a machine with a working graphics backend:

```bash
WTUMI_TEST_MUJOCO_RENDER=1 python -m pytest -q tests/test_rig_capture.py tests/test_stair_metrics.py tests/test_occlusion_demo.py tests/test_occlusion_integration.py
```

Use `MUJOCO_GL=egl` as well on PACE. Without the rendering opt-in, real rendering tests are skipped and cannot validate the graphics setup.

## Measured Mac baseline

The default configuration completed on Mac on 2026-11-16 with real MuJoCo rendering. It uses five integration poses, two held-out cameras, 21.875 mm TSDF cells, 54.6875 mm occupancy cells and a 30 mm geometric tolerance. A separate PACE EGL run also completed that day; its downloaded record is `outputs/occlusion_demo_20261116_184221_889717/`. Its stair depth coverage is 95.38%, 34.01% and 37.96% for no-box five-frame, box single-frame and box five-frame conditions, respectively. The table below remains the Mac measurement.

| Condition | Stair depth coverage | Mean absolute depth error | Root mean square depth error | Exposed-surface coverage within 30 mm | Unknown fraction in the fixed region |
|---|---:|---:|---:|---:|---:|
| No box / five frames | 95.39% | 5.44 mm | 33.19 mm | 99.69% | 53.85% |
| Box / first frame | 33.93% | 8.93 mm | 63.48 mm | 50.00% | 78.07% |
| Box / five frames | 37.98% | 4.39 mm | 8.56 mm | 51.48% | 74.17% |

The box covers about 31.5% of image pixels and hides much of the staircase. The prescribed motion recovers a small amount of additional geometry, while the lower steps remain largely unobserved. The smaller overlap-only depth error in the box case does not indicate a better complete map: most reference stair pixels have no prediction. Unknown volumes are retained rather than filled with ground-truth geometry.
