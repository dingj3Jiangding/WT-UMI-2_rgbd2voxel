# M5: MuJoCo capture and RGB-D reconstruction results

On 2026-11-11, actual MuJoCo rendering, independent sequence export, TSDF reconstruction, and held-out-view evaluation were completed locally on Mac. The full example is saved in `outputs/m5_mujoco_demo/`. PACE scripts are ready but have not run on the remote cluster.

## What this run validates

The room contains a floor, walls, a table, and colored boxes. The same virtual camera is placed at front, left, right, and elevated positions to capture four 240 × 180 RGB-D frames for fusion. A fifth view, `novel_front_right`, provides only an evaluation reference.

```text
MuJoCo room + camera capture configuration
  → Save RGB-D, calibration K, poses T_wc, and view splits
  → Fuse four frames into a TSDF voxel map
  → Extract a mesh with fused colors
  → Render RGB-D from the fifth view
  → Compare against MuJoCo's fifth frame and measure error
```

Moving the camera currently means switching between fixed capture positions. Different positions observe different surfaces and reduce occlusion. Continuous camera motion, robot actions, and dynamic objects are not simulated. Timestamps describe the capture schedule; physics is not stepped.

The resulting map validates the reconstruction pipeline. Whether this map later becomes a robot prior or is used only for training and evaluation remains a separate project decision.

## Run and inspect the example

Run from the project root. A new timestamped output directory is created by default:

```bash
./.venv/bin/python scripts/run_sim_demo.py
```

The program captures once, runs experiments at three resolutions, and saves comparison images, error curves, and a 3D viewer. The output directory's `README.md` links to the results.

Open the saved results from this run directly:

```bash
open outputs/m5_mujoco_demo/experiment/runs/clean_n160_s7/renders/novel_front_right/comparison.png
open outputs/m5_mujoco_demo/experiment/error_vs_voxel_size.png
open outputs/m5_mujoco_demo/experiment/runs/clean_n160_s7/viewer.html
```

Key files:

- [Fifth-view RGB-D comparison](../outputs/m5_mujoco_demo/experiment/runs/clean_n160_s7/renders/novel_front_right/comparison.png): the reference comes from MuJoCo, and the prediction comes from the reconstructed surface.
- [Voxel size versus error](../outputs/m5_mujoco_demo/experiment/error_vs_voxel_size.png): mean absolute error, root mean square error, and coverage; an SVG is also saved.
- [Complete experiment table](../outputs/m5_mujoco_demo/experiment/report.md): results at all three resolutions.
- [Run record](../outputs/m5_mujoco_demo/demo.json): configuration, environment, input reuse checks, and output paths.
- `capture/manifest.json` and `capture/frames/`: independently saved raw capture data.
- `reconstruction.yaml` and `experiment.yaml`: replay entry points using relative paths.

Reconstruct the saved data without capturing again:

```bash
./.venv/bin/python -m wtumi_recon run \
  --config outputs/m5_mujoco_demo/reconstruction.yaml
```

To migrate, copy the entire demo directory. The replay configurations above move with the data. Internal `config_resolved.yaml` and `source_manifest.json` files record provenance and may contain paths from the original machine.

## Results and interpretation

Only grid resolution changes: the map edge length is fixed at 3.5 m, capture images and camera poses are identical, and the TSDF truncation distance is 0.08 m. No additional random depth noise or dropout is applied. The table reports the single held-out view.

| Grid resolution | Voxel edge mm | Coverage | MAE mm | RMSE mm | Coverage within 20 mm |
|---|---:|---:|---:|---:|---:|
| 112³ | 31.250 | 98.49% | 16.954 | 130.584 | 95.41% |
| 160³ | 21.875 | 98.73% | 11.919 | 105.653 | 96.64% |
| 224³ | 15.625 | 98.87% | 11.470 | 103.865 | 96.83% |

Coverage is the fraction of valid reference pixels for which valid depth is also rendered. It does not establish that those depths are accurate. MAE and RMSE are computed only where both images have depth. Coverage within 20 mm requires a prediction whose absolute depth error is no greater than 20 mm, with all valid reference pixels as the denominator.

The 160³ result has 42,650 overlapping pixels, 550 missing pixels, and 0 extra pixels. In the overlap region, median error is 1.154 mm and the 95th percentile is 8.889 mm, but 450 pixels have errors above 100 mm, with a maximum of about 1.77 m. These large errors at a small number of locations make RMSE much higher than MAE.

The comparison image shows problems near object silhouettes, occlusion boundaries, and some gaps. A small shift in a predicted silhouette can place the same pixel on a foreground box in one image and a distant wall in the other, producing a large depth error. Fused colors also show blurring and shading differences; the reconstruction does not precisely reproduce all materials and lighting.

Finer voxels improve the metrics in this run, but the gain from 21.875 mm to 15.625 mm is small. One scene and one held-out view cannot establish the same trend for all scenes. Suitability for contact tasks requires local evaluation around doorframes, handles, narrow gaps, and similar regions. Unobserved map regions are not automatically completed.

## Validation scope

Local environment: Python 3.12.2, MuJoCo 3.17.0, Open3D 0.20.0, and macOS arm64. With actual MuJoCo rendering tests enabled, 139 tests passed:

```bash
WTUMI_TEST_MUJOCO_RENDER=1 ./.venv/bin/python -m pytest -q -p no:cacheprovider
```

Validation covers metric camera-Z depth, analytic plane depth, off-center pixels, MuJoCo/OpenCV coordinate conversion, data format checks, held-out isolation, independent sequence reconstruction, replay after directory relocation, and reuse of one capture at different voxel resolutions. Two tests are explicitly skipped when rendering is not enabled. The actual RGB-D comparison and error curve were inspected; browser interaction with the offline HTML was not manually validated in this run.

Finite precision can cause differences between static geometry and rasterization at object silhouettes. Passing analytic plane tests does not imply exact agreement at every silhouette pixel.

The PACE check script passed local stub validation for execution order, the optional demo branch, logs, separate output directories, and stopping on failure. This does not validate the Linux rendering backend or remote jobs. See the [PACE runbook](PACE_runbook.md).

## Next validation steps

Prepare a separate Linux environment on PACE, replay this capture, then perform a fresh MuJoCo capture. This separates reconstruction portability checks from simulator rendering setup. Next, add a static doorframe and more evaluation views through configuration and MJCF, then assess occlusion and errors in contact-relevant regions.

The current implementation covers static scenes with known camera poses. Robot motion, dynamic door opening, tactile fusion, pose estimation for real devices, and map access by the execution policy require later implementation or decisions.
