# M4: Spatial states and occlusion validation

An optional bounded ray-evidence map is implemented with three states: unknown (no observation evidence), free (ray traversal evidence), and occupied (a surface hit). It reads only the actual depth observations used for fusion; evaluation images do not update the map.

## Observed state changes

Three examples use the same occluder, box behind it, and background wall. The occupancy grid is 80³, totaling 512,000 cells with an edge length of 25 mm.

| Observation condition | Unknown cells | Free cells | Occupied cells |
|---|---:|---:|---:|
| Front only | 427,595 | 81,276 | 3,129 |
| Front + right | 393,865 | 112,055 | 6,080 |
| Front, 50% random depth dropout | 431,830 | 77,544 | 2,626 |

Adding the side observation reduces the number of unknown cells by 33,730. Missing depth can only reduce valid evidence; it cannot automatically clear unmeasured space to free. The change in occupied cells also reflects additional surfaces observed from the new view.

| Probe in the comparison image | Front only | Added right view |
|---|---|---|
| 1: In front of the occluder | free | free |
| 2: Occluder surface | occupied | occupied |
| 3: A location in free space behind the occluder | unknown | free |
| 4: Right surface of the box behind the occluder | unknown | occupied |
| 5: Inside the box behind the occluder | unknown | unknown |

These states are queried at the same world coordinates; positions and states are saved in each output directory's `occupancy.json`. Probes outside the map return `outside`. An unknown state inside the box means there is no observation evidence; it does not mean that the interior is empty.

## Inspect the results

```bash
open outputs/m4_occupancy_comparison.png
open outputs/m4_occlusion_reveal/viewer.html
open outputs/m4_door/viewer.html
```

From left to right, the comparison shows the front view, added right view, and front view with 50% random dropout. Gray is unknown, teal is free, and orange is occupied. The slice shows one cell layer at Y=1.0125 m; numbered markers correspond to the table above.

In the viewer, select **Occupancy**, choose the X/Y/Z axis, then drag and release the index slider. The coordinate range of each layer appears in the sidebar. Enable cameras to display fusion and evaluation poses. Mesh, near-surface TSDF voxels, and point cloud remain available as separate modes.

The door scene uses an 80³ occupancy grid with 31.25 mm cells and a 160³ TSDF with 15.625 mm cells. They share world bounds but store data independently. Keep the two voxel sizes distinct when inspecting the results.

## Rerun or use other scenes

```bash
./.venv/bin/python -m wtumi_recon run --config configs/scenes/m4_static_door.yaml
./.venv/bin/python -m wtumi_recon run --config configs/scenes/m4_occlusion_front.yaml
./.venv/bin/python -m wtumi_recon run --config configs/scenes/m4_occlusion_reveal.yaml
./.venv/bin/python -m wtumi_recon run --config configs/scenes/m4_occlusion_missing.yaml
```

Each run creates a timestamped directory by default and saves `occupancy.npz`, `occupancy.json`, and `occupancy_slices.png`. Then use `view --result <printed_output_directory>` to generate the viewer. Any Uniform scene can enable `occupancy.enabled: true` in YAML without changing the reconstruction core.

## Validation and limitations

- 82 tests passed, covering ray traversal, boundaries, rotated cameras, invalid depth, conflicting evidence, file I/O, occlusion, and novel-view isolation.
- An independent per-cell ray/AABB check matched the implementation on 2,508 random and boundary rays.
- Saved occupancy statistics match viewer statistics for all four examples; the door scene retains its original M2 depth metrics.
- Static slices of the door and occlusion examples were inspected. Viewer geometry and control logic passed mocked Node tests. Actual browser rendering and mouse interaction were not validated in this milestone; an earlier automatic preview was blocked because the tool did not permit the file:// protocol.

Free means that at least one sampled ray passed through the cell; it does not guarantee that every position in the cell is empty. Occupied records surface hits without inferring solid interiors. Each frame contributes at most one free and one hit count per cell. When both kinds of evidence coexist, occupied takes priority and the raw counts are retained. Coexistence can arise from rays passing through different parts of a coarse cell, so the count of conflicting cells is not a count of sensor errors.

This is static, deterministic evidence accumulation. It does not yet include probabilistic fusion, clearing old evidence from dynamic objects, or tactile updates. At the M4 milestone, the external RGB-D adapter planned for M5 was still pending.

The front-only and dropout examples have two held-out views, while the added-side-view example has one. To compare their RGB-D reconstruction quality, use the shared `novel` view. This report compares only occupancy states and probes within the same spatial bounds.
