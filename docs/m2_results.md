# M2: Multi-view static scene validation

Historical record: generated M2 output directories were removed on 2026-11-17. Configurations and tests are retained; rerun the reconstruction commands before opening the outputs linked below.

## Completed work

- The same pipeline switches between a color-gradient cube with a sphere and a door with a frame, handle, and floor using two YAML files.
- Multiple RGB-D frames with known camera poses are fused into a Uniform TSDF, a mesh is extracted, and RGB-D is rendered from the reconstruction.
- RGB uses barycentric interpolation of the fused mesh's vertex colors. Known colors from the original scene are not copied into the prediction.
- Each configuration includes one view excluded from fusion. Outputs record the actual `integrated_frame_ids`; tests also check the actual integration calls and poses.
- Outputs include all observations, K/T_wc, configurations, TSDF/weight/color arrays, the mesh, per-view renders, and metrics.

## Measured novel-view results

| Scene | Fusion frames | Novel-view coverage | Depth MAE | Depth RMSE | RGB PSNR |
|---|---:|---:|---:|---:|---:|
| Cube + sphere | 4 | 97.09% | 5.22 mm | 18.67 mm | 34.07 dB |
| Static door | 3 | 98.79% | 2.63 mm | 19.53 mm | 28.99 dB |

Both use an effective voxel edge length of 15.625 mm. MAE, RMSE, and RGB PSNR are computed only at pixels with valid depth in both the reference and reconstruction.
The door scene misses 325 reference pixels and produces 651 extra pixels on the reference background; its 2D silhouette IoU is 96.46%.
Pixels with a reconstruction and a depth error no greater than 20 mm account for 97.37% of all valid reference pixels in the door scene.
MAE is much smaller than RMSE, indicating that some local errors remain large. Reporting only the mean would hide errors at silhouettes and occlusion boundaries.

These results use synthetic data with no noise, no complex lighting, and known accurate poses. They do not establish real-camera accuracy.
The scenes remain static, without robot motion, door rotation, pose estimation, or tactile input.

## Run the examples

Run from the repository root. Omitting the output directory creates a new one without overwriting earlier experiments:

```bash
./.venv/bin/python -m wtumi_recon run --config configs/scenes/static_door.yaml
./.venv/bin/python -m wtumi_recon run --config configs/scenes/multiview_cube.yaml
./.venv/bin/python -m pytest -q -p no:cacheprovider
```

Open `renders/novel_front_right/comparison.png` in the output directory printed by the program.
Each image shows the reference on the left, reconstruction in the middle, and error on the right; RGB is on the top row and depth on the bottom row.
Magenta marks missing pixels, cyan marks extra pixels, and red shows depth error, with full red corresponding to at least 20 mm.
The saved `depth_m.npy` contains metric depth; grayscale images are for inspection only.

## Validation scope

Tests cover independent analytic plane depths, integer pixel centers, rotated cameras, barycentric color interpolation, scene color changes, actual multi-frame fusion, held-out isolation, configuration round trips, missing/extra pixels, and metrics with no overlap.
M1 was also rerun. After standardizing pixel centers to integer coordinates and removing millimeter depth quantization, M1 coverage was 90.25%, which differs from the old helper's half-pixel sampling result. Earlier outputs were preserved.

Further work can examine edge/handle errors and the effects of voxel resolution and noise, then add a real RGB-D data adapter. Unobserved TSDF cells alone do not form a complete free/occupied/unknown probability map.

## Offline 3D viewer

```bash
./.venv/bin/python -m wtumi_recon view --result outputs/m2_door
open outputs/m2_door/viewer.html
```

The viewer switches between the reconstructed mesh, near-surface voxels, and surface point cloud, and can display fusion and evaluation camera poses. It reads saved outputs without regenerating images or repeating fusion.

The default selection is `weight >= 1` and `abs(tsdf) * sdf_trunc_m <= voxel_size_m`. It selects 6,391 cells for the cube and sphere and 24,168 cells for the door, both with a voxel edge length of 15.625 mm. Each cell is drawn at its saved TSDF world coordinates; internal faces between adjacent cells are omitted. The near-surface cell layer differs slightly from the interpolated mesh outline and should not be interpreted as a solid-object occupancy map.

Automated tests verified voxel indices/colors/coordinates, camera frusta, mesh faces, and offline file export. The browser tool's local-file access policy blocked automated preview, so this milestone did not include a manual check of actual page rendering or mouse interaction. Use the `open` command above to inspect it in a local browser.
