# Quick scene changes and native MuJoCo playback

The single-scene workflow keeps environment geometry, the camera mount, capture waypoints, held-out views and map settings in one scene bundle. It supports room, stairs, doorway and `g1_stairs` templates. The room, doorway and G1 stairs examples use an actual Unitree robot visual model; see [the G1 guide](unitree_g1.md). A run captures only the configured condition and uses the same reconstruction pipeline for each template.

## Start locally

From the project directory, activate the existing environment:

```bash
source .venv/bin/activate
python scripts/scene.py list
python scripts/scene.py view --scene stairs
```

The last command opens a native MuJoCo window and replays the configured camera/rig waypoints. On macOS the script automatically uses the environment's `mjpython` launcher. Close the window before continuing in the same terminal.

Create an editable copy, inspect it, then reconstruct it:

```bash
python scripts/scene.py create --template stairs --output scenes/my_stairs
# Edit scenes/my_stairs/scene.yaml.
python scripts/scene.py view --scene scenes/my_stairs
python scripts/scene.py run --scene scenes/my_stairs --output outputs/my_stairs_run
python scripts/scene.py view --result outputs/my_stairs_run
```

Choose a new output directory for each run; existing results are never overwritten. To switch scenes, replace the `--scene` value with `room`, `doorway`, or another bundle directory. A bundle is a directory containing `scene.yaml` and any relative mesh assets. Arbitrary existing MuJoCo XML models are not accepted as scene bundles; the lower-level simulator capture workflow remains available for them.

## Native viewer controls

| Key | Display/action |
|---|---|
| `1` | Original simulator scene |
| `2` | Reconstructed mesh after the selected number of observations |
| `3` | Observed near-surface TSDF voxels after the selected number of observations |
| `4` | Original scene with the reconstructed mesh overlaid |
| `P` | Play/pause the saved sequence |
| `[` / `]` | Previous/next observation |
| `C` | Head-mounted camera/free orbit view |
| `I` | Show/hide the current captured RGB and held-out RGB-D comparison insets |
| `O` / `L` | Open/close the selected door by 10 degrees in source-scene preview |
| `N` | Select the next door in source-scene preview |

Modes 2–4 and the saved-image insets require a completed `--result` run. Standard MuJoCo mouse controls orbit, zoom and pan the view. Source and reconstructed geometry are in the same world coordinates. The native mesh is cyan for geometry inspection; the voxel display, saved RGB-D comparisons and final HTML mesh retain captured colors. The map is reconstructed from captured RGB-D; it is not copied from simulator geometry.

The optional image panels use fixed pixel sizes along the bottom of the window. Enlarge the window if a panel is clipped, or press `I` to hide the panels and use the whole view for 3D geometry.

The voxel display selects actual observed cells from the saved TSDF with a near-zero signed distance. It represents surface evidence, not a filled solid or every cell in the volume. A bounded subset is displayed when there are too many cells; the saved data remains complete. The final HTML viewer also exposes occupancy slices for unknown/free/occupied evidence.

This is playback of saved reconstruction steps. Capture and reconstruction run before opening the result. The rig follows prescribed kinematic poses; opening a scene does not implement walking, manipulation or physical pickup. Exact simulator poses and segmentation masks are used. Playback speed is independent of reconstruction throughput.

## What to change

All distances are metres, angles are degrees, and the world is Z-up. The rig faces +Y at zero yaw.

| Field in `scene.yaml` | Purpose |
|---|---|
| `environment.objects` | Static boxes, spheres, cylinders and relative mesh assets |
| `environment.doors` | Hinged door dimensions, mounting transform, opening limits and fixed capture angle |
| `camera` | Image dimensions, vertical field of view, range and head-camera mount |
| `torso`, `box` | Optional rig geometry and carried occluder |
| `trajectory` | Timestamped rig poses, with optional independent head yaw/pitch |
| `held_out` | Separate reference viewpoints used only for evaluation |
| `reconstruction` | Uniform TSDF bounds, resolution and truncation distance |
| `occupancy` | Ray-evidence grid resolution and sampling stride |
| `evaluation` | Depth agreement tolerance |
| `playback.fps` | Display speed of saved observations |

Box `size_m` specifies full dimensions, not MuJoCo half extents. Keep target geometry within the reconstruction cube (`origin_m` through `origin_m + volume_length_m`). Cameras may be outside that volume. Changing geometry can require adjusting camera poses and map bounds. Doors have actual MuJoCo hinge joints. Their angle is fixed during a reconstruction run so the static TSDF does not fuse multiple door positions.

Generate a standalone MuJoCo model without capturing images:

```bash
python scripts/scene.py build --scene scenes/my_stairs --output outputs/my_stairs_model
python .venv/bin/mjpython -m mujoco.viewer --mjcf=outputs/my_stairs_model/model.xml
```

The generated `model.xml` and copied assets can be loaded by a standard MuJoCo viewer. That viewer shows the original scene. Use `scene.py view --result ...` for cumulative reconstructed geometry and TSDF voxel controls.

## Enclosed room and operable door

Both room templates have a 6.4 x 10.0 m interior, four enclosing walls 2.4 m high, and no roof so the layout can be inspected from above. The floor and outer wall footprint is 6.6 x 10.2 m.

- `room`: a centered longitudinal partition is 5.0 m long, half the interior room length. The camera path travels around both ends and observes both sides.
- `doorway`: a transverse partition divides the room at its midpoint. Its central opening is 1.0 m wide and 2.05 m high. A 0.98 x 2.03 m leaf rotates about a vertical hinge; the camera stays on the near side, so its path remains valid when the door is closed. The default opening angle is 75 degrees, with limits of 0 to 110 degrees.

Both templates use the original-scale G1 in a fixed carry pose, with the torso proxy disabled and a carried box enabled. The box moves with the rig and occludes the head-camera image; robot and box pixels are masked before TSDF and occupancy integration. The provisional head-mounted camera is at `[0, 0.05, 1.36]` m in the rig frame. The camera path and held-out viewpoints have expanded with the room; wall heights, thicknesses, door dimensions and robot scale retain their physical sizes. The camera range is 14 m. Older saved results retain their original no-box configuration until recaptured.

Inspect either template:

```bash
python scripts/scene.py view --scene room
python scripts/scene.py view --scene doorway
```

Close one window before running the next command. Orbit upward with the standard MuJoCo mouse controls to inspect the four walls and center partition. `C` switches to the interior camera view.

Create an editable doorway configuration:

```bash
python scripts/scene.py create --template doorway --output scenes/my_doorway
```

The door entry uses this convention:

```yaml
environment:
  # Retain the generated objects list here.
  doors:
  - name: passage
    hinge_position_m: [-0.49, 3.675, 0.01]
    width_m: 0.98
    height_m: 2.03
    thickness_m: 0.045
    closed_yaw_degrees: 0.0
    angle_degrees: 75.0
    range_degrees: [0.0, 110.0]
    color_rgba: [0.58, 0.33, 0.15, 1.0]
```

The hinge position is the bottom of the hinge axis. At zero opening, the leaf extends along local +X, rotated by `closed_yaw_degrees` about world +Z. Positive opening rotates +X toward +Y. Change `angle_degrees` to 0 for a closed-door experiment or 90 for a right-angle opening. `O` and `L` change only the current source-window preview; closing the window discards those preview edits. Saved-result viewers lock the door at its captured angle, including when replaying different observation counts.

```bash
# After editing angle_degrees in scenes/my_doorway/scene.yaml:
python scripts/scene.py run --scene scenes/my_doorway --output outputs/my_doorway_run
python scripts/scene.py view --result outputs/my_doorway_run
```

The door is included in environment RGB-D and reconstruction. Only robot and carried-box pixels are masked. Capture holds every door at its configured angle for both integration and held-out views. Door manipulation, contact forces, moving-object fusion and automatic clearing of old door geometry are outside this workflow. The generated standalone MJCF includes the hinge and starts at the configured angle; the custom preview changes its joint position without stepping physics.

The room volume is 10.4 m wide with 256 cells per axis (40.625 mm per voxel), covering the complete shell. This covers the enlarged room within the current 256-cell dense-grid limit; its voxel size is coarser than the earlier 28.125 mm examples. The scene contains 13 integration views and 2 held-out views. Existing custom bundles and earlier result directories retain their saved layouts; use the template again to create a new enclosed-room bundle.

## Output and evaluation

| Output | Content |
|---|---|
| `scene/` | Portable source model, configuration and copied assets |
| `capture/` | Raw/masked RGB-D, exclusion masks, poses and portable sequence manifest |
| `history/` | Intermediate meshes, TSDF volumes, occupancy and held-out render comparisons |
| `reconstruction/` | Final full map, point cloud, volumes, metrics and HTML viewer |
| `history.json` | Ordered prefix paths and per-step held-out metrics |
| `scene_run.json` | Run summary and portable result paths |
| `index.html` | Input/reconstruction preview animations and final-viewer link |

Robot and box pixels are excluded from static-map integration. Unknown geometry is not filled from the simulator. The fixed held-out images never enter fusion. Coverage reports reference pixels with a predicted depth; it does not establish correctness. Mean absolute depth error and root mean square depth error are calculated on overlapping valid depth pixels. The depth agreement fraction additionally counts missing predictions against all valid reference pixels.

Copy the entire result directory when moving between machines. For PACE, sync the source (including `scenes/`), request a GPU compute node using the existing PACE workflow, then run:

```bash
MUJOCO_GL=egl MPLBACKEND=Agg python scripts/scene.py run --scene stairs --output outputs/stairs_pace
```

Download the whole result and use `python scripts/scene.py view --result outputs/stairs_pace` on the Mac. A headless compute node can render and reconstruct; native interactive viewing needs a graphical desktop.

## Camera versus cumulative voxel experiment

```bash
python scripts/run_scene_suite.py --output outputs/my_g1_vision_suite
```

The `--preset pace30` option generates 121 fresh 640 x 480 observations per scene and encodes the complete sequence at 30 fps. See [the PACE 30 fps workflow](g1_suite_30fps_pace.md). The original default remains 13 observations at 3 fps.

The three scene directories are `room`, `doorway` and `g1_stairs`. The suite `index.html` links to synchronized camera/voxel videos, each full scene gallery, and `summary.csv`. Each left panel uses the saved raw head-camera RGB image. Each right panel uses only observed near-surface cells from that same prefix's saved TSDF, rendered as cubes in MuJoCo from a fixed overview; it does not render the original environment. A camera-position/direction marker is reference geometry, not a reconstructed object. Labels report displayed and selected cell counts when voxel display sampling is necessary. Full TSDF snapshots are unchanged.

MP4 export uses the existing optional `.[video]` extra. A GIF is always exported. Playback follows the saved observation rate, not an estimate of real-time reconstruction performance. The comparison uses different viewpoints: it shows the relationship between camera observations and a growing 3D map. For depth accuracy, use the existing held-out reference/render comparisons within each scene's gallery.

The completed local suite is `outputs/g1_vision_suite`. Existing earlier example directories retain their original saved geometry. The two enlarged-room experiments use 40.625 mm voxels; G1 stairs retains its 21.875 mm setting and carried-box occlusion. These different scene conditions do not constitute a controlled comparison of mapping methods.

## Earlier small-room example results

These earlier 3.2 x 5.0 m rooms with proxy robots were captured on macOS before the enlargement and G1 replacement. They remain as historical results and do not reflect the current templates. Each contains 13 observations, 2 held-out views and 28.125 mm TSDF voxels:

```bash
python scripts/scene.py view --result outputs/enclosed_room_demo
python scripts/scene.py view --result outputs/enclosed_doorway_demo
```

Close one window before opening the other. The doorway result uses the configured 75-degree opening throughout capture. The source viewer remains available for interactive opening and closing.

| Saved result | Held-out depth coverage | Mean absolute depth error | Root mean square depth error |
|---|---:|---:|---:|
| `outputs/enclosed_room_demo` | 91.41% | 7.20 mm | 34.90 mm |
| `outputs/enclosed_doorway_demo` | 89.20% | 9.27 mm | 132.62 mm |

These metrics describe the two held-out views, not the fraction of the entire room reconstructed. Doorway boundary and occlusion errors can produce a large root mean square error even when the mean absolute error is small. Coverage alone is not a geometric accuracy guarantee.

## Historical local template results

The earlier room, stairs and doorway layouts were captured and reconstructed on macOS on 2026-11-17. The room and doorway entries below predate the enclosed-room and hinged-door update; they are not measurements of the current templates. Each uses 13 integration views, 2 held-out views, 320 x 240 images and a 21.875 mm TSDF voxel size. The whole workflow took approximately 7–11 seconds per scene on this Mac, including evaluation/export; this is not a real-time benchmark.

| Saved result | Held-out depth coverage | Mean absolute depth error | Root mean square depth error |
|---|---:|---:|---:|
| `outputs/quick_room` | 90.95% | 7.84 mm | 81.41 mm |
| `outputs/quick_stairs` | 29.59% | 1.83 mm | 25.60 mm |
| `outputs/quick_doorway` | 92.91% | 8.44 mm | 83.24 mm |

These scenes have different geometry and occlusion, so their numbers are not a ranking of mapping methods. The stairs template carries a large box that hides much of the lower scene. Its small error on the visible overlap does not compensate for the missing geometry. Mac OpenGL emits an `ARB_clip_control` depth-precision warning; these results are a local workflow validation, with PACE EGL validation still available through the command above.
